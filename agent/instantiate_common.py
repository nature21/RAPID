"""Shared implementation for primitive and strategy instantiation helpers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

from instantiation_core.runner import cartesian_goals_path_for_metrics

from .codex_exec import (
    AGENT_WORK_DIR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_MODEL,
    REASONING_EFFORTS,
    CodexExecConfig,
    WORKSPACE,
    run_codex_exec,
    stage_workspace_codex_guidance,
)
from .descriptions import load_language_description
from .env_modules import normalize_env_modules, validate_env_modules_importable
from .instantiation_context import (
    build_instantiation_prompt,
    instantiation_dir,
    prepare_scene_evidence,
    program_dir,
    stage_canonical_instantiation,
    validate_package_name,
)
from .instantiation_validation import (
    learned_source_digests,
    record_learned_source_digests,
    report_program_contract_validation,
    report_instantiation_validation,
    sandboxed_instantiation_findings,
)


def build_parser(kind: str) -> argparse.ArgumentParser:
    flag = f"--{kind}"
    parser = argparse.ArgumentParser(
        description=f"Bind an existing RAPID {kind} to one MuJoCo task scene."
    )
    parser.add_argument(flag, required=True, dest="program")
    parser.add_argument("--env-module", required=True)
    parser.add_argument("--name", required=True)
    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal", type=str, default=None)
    goal_group.add_argument("--goal-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--thinking-effort",
        choices=list(REASONING_EFFORTS),
        default="high",
    )
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument(
        "--no-stage-guidance",
        action="store_true",
        help="Do not refresh workspace AGENTS.md and staged skills.",
    )
    return parser


def preserve_instantiation_results(
    run_dir: Path,
    *,
    kind: str,
    program: str,
    instance: str,
) -> Path | None:
    """Retain the best completed execution outside the ephemeral agent run."""

    smoke_dir = run_dir / "smoke"
    candidates = list(smoke_dir.glob("metrics*.json"))
    plural = "primitives" if kind == "primitive" else "strategies"
    result_dir = (
        WORKSPACE
        / "results_instantiation"
        / plural
        / program
        / instance
    )
    existing = result_dir / "metrics.json"
    if existing.is_file() and existing.stat().st_mtime >= run_dir.stat().st_mtime:
        candidates.append(existing)
    scored: list[tuple[bool, float, Path, dict[str, object]]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        binding = payload.get("binding")
        metrics = payload.get("metrics")
        success = bool(
            payload.get("program_success")
            or isinstance(binding, dict) and binding.get("program_success")
            or isinstance(metrics, dict) and metrics.get("success")
        )
        scored.append((success, path.stat().st_mtime, path, payload))
    if not scored:
        return None

    success, _, selected, _ = max(scored, key=lambda item: (item[0], item[1]))
    result_dir.mkdir(parents=True, exist_ok=True)
    destination_metrics = result_dir / "metrics.json"
    selected_cartesian_goals = cartesian_goals_path_for_metrics(selected)
    if selected != destination_metrics:
        shutil.copy2(selected, destination_metrics)
        suffix = selected.stem.removeprefix("metrics")
        video = selected.with_name(f"trajectory{suffix}.mp4")
        if video.is_file():
            shutil.copy2(video, result_dir / "trajectory.mp4")
        if selected_cartesian_goals.is_file():
            shutil.copy2(
                selected_cartesian_goals,
                result_dir / "cartesian_goals.json",
            )
    selection = {
        "program_success": success,
        "source_metrics": str(selected),
        "source_cartesian_goals": (
            str(selected_cartesian_goals)
            if selected_cartesian_goals.is_file()
            else None
        ),
        "source_run": str(run_dir),
    }
    (result_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_dir


def run(kind: str, argv: list[str] | None = None) -> None:
    args = build_parser(kind).parse_args(argv)
    try:
        program = validate_package_name(args.program, label=kind)
        instance = validate_package_name(args.name, label="instance name")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    source = program_dir(kind, program)
    if not source.is_dir():
        print(f"error: selected {kind} does not exist: {source}", file=sys.stderr)
        raise SystemExit(2)
    report_program_contract_validation(kind, program)
    destination = instantiation_dir(kind, program, instance)

    env_modules = normalize_env_modules([args.env_module])
    findings = validate_env_modules_importable(env_modules)
    if findings:
        print("error: env module not importable:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        raise SystemExit(2)
    scene_module = env_modules[0]

    try:
        goal = load_language_description(args.goal, args.goal_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not goal:
        print("error: goal text cannot be empty", file=sys.stderr)
        raise SystemExit(2)

    resume_existing = destination.exists()
    if resume_existing:
        findings = sandboxed_instantiation_findings(
            kind,
            program,
            instance,
            scene_module=scene_module,
            goal=goal,
        )
        if findings:
            print(
                f"error: cannot safely resume instantiation: {destination}",
                file=sys.stderr,
            )
            for finding in findings:
                print(f"  - {finding}", file=sys.stderr)
            raise SystemExit(2)
        print(f"[instantiate_{kind}_exec] resuming instantiation: {destination}")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = AGENT_WORK_DIR / f"instantiate_{kind}_exec_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[instantiate_{kind}_exec] run dir: {run_dir}")

    if not args.no_stage_guidance:
        staged = stage_workspace_codex_guidance()
        print(f"[instantiate_{kind}_exec] staged Codex guidance files: {len(staged)}")

    inventory, images = prepare_scene_evidence(scene_module, run_dir)
    print(f"[instantiate_{kind}_exec] scene inventory: {inventory}")
    print(
        f"[instantiate_{kind}_exec] rendered {len(images)} initial scene view(s)"
    )
    prompt = build_instantiation_prompt(
        kind=kind,
        program=program,
        instance=instance,
        scene_module=scene_module,
        goal=goal,
        inventory=inventory,
        images=images,
        run_dir=run_dir,
    )
    learned_before = learned_source_digests()
    if not resume_existing:
        stage_canonical_instantiation(
            destination,
            kind=kind,
            program=program,
            instance=instance,
            scene_module=scene_module,
            goal=goal,
        )
    try:
        run_codex_exec(
            prompt,
            CodexExecConfig(
                model=args.model,
                reasoning_effort=args.thinking_effort,
                max_turns=args.max_turns,
                local_images=images,
                output_last_message=run_dir / "codex_final.md",
                max_retries=max(0, args.max_retries),
                retry_delay=max(0.0, args.retry_delay),
            ),
        )
        report_instantiation_validation(
            kind,
            program,
            instance,
            learned_before=learned_before,
            run_dir=run_dir,
            scene_module=scene_module,
            goal=goal,
            fatal=False,
        )
    except BaseException:
        changed = record_learned_source_digests(
            run_dir / "learned_program_digests.json",
            learned_before,
        )
        if changed:
            print(
                f"[instantiate_{kind}_exec] learned source changed before "
                f"the failed binding run ended: {', '.join(changed)}",
                file=sys.stderr,
            )
        raise
    finally:
        try:
            retained = preserve_instantiation_results(
                run_dir,
                kind=kind,
                program=program,
                instance=instance,
            )
        except (OSError, ValueError) as exc:
            print(
                f"[instantiate_{kind}_exec] warning: could not preserve "
                f"instantiation results: {exc}",
                file=sys.stderr,
            )
        else:
            if retained is not None:
                print(
                    f"[instantiate_{kind}_exec] retained results: {retained}"
                )


__all__ = [
    "build_parser",
    "preserve_instantiation_results",
    "run",
]
