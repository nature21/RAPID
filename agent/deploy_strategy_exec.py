"""Fast semantic binding and one-shot execution of a frozen RAPID strategy."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.codex_exec import (
    AGENT_WORK_DIR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    REASONING_EFFORTS,
    REPO_ROOT,
    STRATEGIES_DIR,
    WORKSPACE,
    CodexExecConfig,
    run_codex_exec,
    stage_workspace_codex_guidance,
)
from agent.deployment_context import (
    DEPLOY_CONTEXT_RESULT_PREFIX,
    DeploymentBindingDecision,
    build_deployment_correction_prompt,
    build_deployment_prompt,
    parse_deployment_binding_decision,
)
from agent.descriptions import load_language_description
from agent.env_modules import normalize_env_modules, validate_env_modules_importable
from agent.instantiation_context import (
    CANONICAL_INIT,
    CANONICAL_MAIN,
    CANONICAL_PARENT_INIT,
    initial_binding_manifest,
    instantiation_dir,
    instantiation_module,
    render_scene,
    validate_package_name,
)
from agent.instantiation_validation import (
    sandboxed_instantiation_findings,
    validate_instantiation_main,
)
from agent.sandbox_validation import PERMISSION_PROFILE


DEFAULT_DEPLOY_MODEL = os.environ.get("RAPID_COMPOSITION_MODEL", "gpt-6-luna")
_AUDIT_TRUE_KEYS = {"audit_advisory_triggered", "transit_validation_advisory"}
_AUDIT_FALSE_KEYS = {
    "controller_state_handoff_valid",
    "retained_timing_valid",
    "transit_convergence_valid",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind a frozen RAPID strategy and execute it once."
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--env-module", required=True)
    parser.add_argument("--name", required=True)
    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal", type=str, default=None)
    goal_group.add_argument("--goal-file", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_DEPLOY_MODEL)
    parser.add_argument(
        "--thinking-effort",
        choices=list(REASONING_EFFORTS),
        default="low",
    )
    parser.add_argument(
        "--bind-only",
        action="store_true",
        help="Validate and write the binding without executing the strategy.",
    )
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument(
        "--no-stage-guidance",
        action="store_true",
        help="Do not refresh workspace AGENTS.md and staged skills.",
    )
    return parser


def _sandbox_command(module: str, *args: str) -> list[str]:
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("`codex` CLI was not found on PATH")
    if os.environ.get("RAPID_CODEX_PERMISSION_PROFILE") != "1":
        raise RuntimeError(
            "strategy deployment must run through scripts/rapid-codex-run"
        )
    return [
        codex,
        "sandbox",
        "-C",
        str(WORKSPACE),
        "-P",
        PERMISSION_PROFILE,
        "--",
        str(REPO_ROOT / "scripts" / "rapid-python"),
        "-P",
        "-m",
        module,
        *args,
    ]


def _run_context_worker(request: Mapping[str, object]) -> dict[str, object]:
    """Run dynamic strategy/scene inspection in the restricted sandbox."""

    completed = subprocess.run(
        _sandbox_command(
            "agent.deployment_context_worker",
            json.dumps(dict(request), sort_keys=True),
        ),
        cwd=WORKSPACE,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=300.0,
    )
    payload_text: str | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(DEPLOY_CONTEXT_RESULT_PREFIX):
            payload_text = line.removeprefix(DEPLOY_CONTEXT_RESULT_PREFIX)
    if completed.returncode != 0 or payload_text is None:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(
            f"deployment context worker failed with exit {completed.returncode}: "
            f"{detail or 'no result'}"
        )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"deployment context worker returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise RuntimeError(f"deployment context worker failed: {detail}")
    return payload


def _prepare_initial_view(scene_module: str, run_dir: Path) -> Path:
    """Render one deployment view instead of attaching three full-size images."""

    evidence_dir = run_dir / "scene_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "initial_view.png"
    return render_scene(scene_module, output, azimuth=None)


def _write_deployment_package(
    destination: Path,
    *,
    strategy: str,
    instance: str,
    scene_module: str,
    goal: str,
    roles: Mapping[str, str | list[str]],
) -> Path:
    """Write a canonical package only after the proposed roles validate."""

    if destination.exists():
        allowed = {"__init__.py", "binding.json", "main.py", "__pycache__"}
        unexpected = sorted(
            path.name for path in destination.iterdir() if path.name not in allowed
        )
        findings: list[str] = []
        parent_init = destination.parent / "__init__.py"
        if (
            not parent_init.is_file()
            or parent_init.read_text(encoding="utf-8") != CANONICAL_PARENT_INIT
        ):
            findings.append(f"noncanonical parent package: {parent_init}")
        init_path = destination / "__init__.py"
        if (
            not init_path.is_file()
            or init_path.read_text(encoding="utf-8") != CANONICAL_INIT
        ):
            findings.append(f"noncanonical deployment package: {init_path}")
        main_path = destination / "main.py"
        if not main_path.is_file():
            findings.append(f"missing deployment entrypoint: {main_path}")
        else:
            findings.extend(validate_instantiation_main(main_path))
        if unexpected:
            findings.append(
                "unexpected deployment package entries: " + ", ".join(unexpected)
            )
        if findings:
            raise ValueError(
                "cannot replace existing deployment package: "
                + "; ".join(findings)
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent_init = destination.parent / "__init__.py"
        if parent_init.exists():
            if parent_init.read_text(encoding="utf-8") != CANONICAL_PARENT_INIT:
                raise ValueError(f"noncanonical parent package: {parent_init}")
        else:
            parent_init.write_text(CANONICAL_PARENT_INIT, encoding="utf-8")
        destination.mkdir()

    (destination / "__init__.py").write_text(CANONICAL_INIT, encoding="utf-8")
    (destination / "main.py").write_text(
        CANONICAL_MAIN,
        encoding="utf-8",
    )
    payload = json.loads(
        initial_binding_manifest(
            kind="strategy",
            program=strategy,
            instance=instance,
            scene_module=scene_module,
            goal=goal,
        )
    )
    payload["roles"] = dict(roles)
    manifest = destination / "binding.json"
    temporary = destination / ".binding.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return manifest


def _decision_from_file(path: Path) -> DeploymentBindingDecision:
    if not path.is_file():
        raise ValueError("deployment agent did not write a final response")
    return parse_deployment_binding_decision(path.read_text(encoding="utf-8"))


def _validated_decision(
    decision: DeploymentBindingDecision,
    validation: Mapping[str, object],
    run_dir: Path,
) -> DeploymentBindingDecision:
    """Retain the structurally resolved mapping and its semantic source."""
    if validation.get("findings") or "roles" not in validation:
        return decision
    resolved = parse_deployment_binding_decision(json.dumps({
        "schema_version": "rapid.deploy-binding-response/v1",
        "compatible": True,
        "roles": validation["roles"],
        "diagnostics": [],
    }))
    (run_dir / "validated_binding_roles.json").write_text(
        json.dumps(
            {"proposed": decision.roles, "resolved": resolved.roles},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return replace(decision, roles=resolved.roles)


def _audit_findings(value: object, *, path: str = "metrics") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        advisories = value.get("audit_advisories")
        if isinstance(advisories, Sequence) and not isinstance(
            advisories, (str, bytes)
        ):
            findings.extend(f"{path}:{item}" for item in advisories if item)
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _AUDIT_TRUE_KEYS and item is True:
                findings.append(item_path)
            if (
                key.endswith("_audit_valid") or key in _AUDIT_FALSE_KEYS
            ) and item is False:
                findings.append(item_path)
            if isinstance(item, (Mapping, list, tuple)):
                findings.extend(_audit_findings(item, path=item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(_audit_findings(item, path=f"{path}[{index}]"))
    return sorted(set(findings))


def _report_execution_result(strategy: str, instance: str) -> bool:
    result_dir = (
        WORKSPACE / "results_instantiation" / "strategies" / strategy / instance
    )
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        print(
            f"error: deployment metrics were not written: {metrics_path}",
            file=sys.stderr,
        )
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: deployment metrics are invalid: {exc}", file=sys.stderr)
        return False
    binding = payload.get("binding")
    binding_valid = payload.get("binding_valid")
    program_success = payload.get("program_success")
    if isinstance(binding, Mapping):
        binding_valid = binding.get("binding_valid", binding_valid)
        program_success = binding.get("program_success", program_success)
    metrics = payload.get("metrics")
    if program_success is None and isinstance(metrics, Mapping):
        program_success = metrics.get("success")
    audit = _audit_findings(metrics if isinstance(metrics, Mapping) else {})
    audit_status = "advisory" if audit else "clean"
    print(
        "[deploy_strategy_exec] "
        f"binding_valid={str(binding_valid).lower()} "
        f"program_success={str(program_success).lower()} "
        f"audit={audit_status}"
    )
    print(f"[deploy_strategy_exec] results: {result_dir}")
    return binding_valid is True and program_success is True


def _run_frozen_strategy(module: str) -> int:
    """Execute once in the same restricted profile, with no budget overrides."""

    completed = subprocess.run(
        _sandbox_command(module),
        cwd=WORKSPACE,
        env=os.environ.copy(),
        check=False,
    )
    return int(completed.returncode)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        strategy = validate_package_name(args.strategy, label="strategy")
        instance = validate_package_name(args.name, label="instance name")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not (STRATEGIES_DIR / strategy).is_dir():
        print(
            f"error: selected strategy does not exist: {STRATEGIES_DIR / strategy}",
            file=sys.stderr,
        )
        return 2
    env_modules = normalize_env_modules([args.env_module])
    findings = validate_env_modules_importable(env_modules)
    if findings:
        print(
            "error: env module not importable: " + "; ".join(findings),
            file=sys.stderr,
        )
        return 2
    scene_module = env_modules[0]
    try:
        goal = load_language_description(args.goal, args.goal_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not goal:
        print("error: goal text cannot be empty", file=sys.stderr)
        return 2

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = AGENT_WORK_DIR / f"deploy_strategy_exec_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deploy_strategy_exec] run dir: {run_dir}")
    if not args.no_stage_guidance:
        staged = stage_workspace_codex_guidance()
        print(f"[deploy_strategy_exec] staged guidance files: {len(staged)}")

    try:
        context_payload = _run_context_worker(
            {"action": "context", "strategy": strategy, "scene_module": scene_module}
        )
        context = context_payload.get("context")
        if not isinstance(context, dict):
            raise RuntimeError("deployment context worker omitted context")
        context_path = run_dir / "deployment_context.json"
        context_path.write_text(
            json.dumps(context, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        view = _prepare_initial_view(scene_module, run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: could not prepare deployment evidence: {exc}", file=sys.stderr)
        return 2

    response_path = run_dir / "binding_response.json"
    prompt = build_deployment_prompt(
        strategy=strategy,
        scene_module=scene_module,
        instance=instance,
        goal=goal,
        context=context,
        has_view=True,
    )
    config = CodexExecConfig(
        model=args.model,
        reasoning_effort=args.thinking_effort,
        max_turns=max(1, args.max_turns),
        local_images=(view,),
        output_last_message=response_path,
        max_retries=max(0, args.max_retries),
        retry_delay=max(0.0, args.retry_delay),
    )
    thread_id = run_codex_exec(prompt, config)
    correction_findings: list[str]
    try:
        decision = _decision_from_file(response_path)
    except ValueError as exc:
        decision = None
        correction_findings = [str(exc)]
    else:
        if not decision.compatible:
            print(
                "[deploy_strategy_exec] incompatible: "
                + "; ".join(decision.diagnostics),
                file=sys.stderr,
            )
            return 1
        validation = _run_context_worker(
            {
                "action": "validate_roles",
                "strategy": strategy,
                "scene_module": scene_module,
                "instance": instance,
                "goal": goal,
                "roles": decision.roles,
            }
        )
        raw_findings = validation.get("findings")
        if not isinstance(raw_findings, list) or not all(
            isinstance(item, str) for item in raw_findings
        ):
            print("error: role validator returned invalid findings", file=sys.stderr)
            return 2
        correction_findings = list(raw_findings)
        decision = _validated_decision(decision, validation, run_dir)

    if correction_findings:
        correction_path = run_dir / "binding_response_correction.json"
        correction_prompt = build_deployment_correction_prompt(
            correction_findings
        )
        correction_images: tuple[Path, ...] = ()
        if thread_id is None:
            correction_prompt = prompt + "\n\n" + correction_prompt
            correction_images = (view,)
        correction_config = CodexExecConfig(
            model=args.model,
            reasoning_effort=args.thinking_effort,
            max_turns=max(1, args.max_turns),
            local_images=correction_images,
            output_last_message=correction_path,
            max_retries=max(0, args.max_retries),
            retry_delay=max(0.0, args.retry_delay),
        )
        run_codex_exec(
            correction_prompt,
            correction_config,
            resume_thread_id=thread_id,
        )
        try:
            decision = _decision_from_file(correction_path)
        except ValueError as exc:
            print(
                f"error: corrected deployment response is invalid: {exc}",
                file=sys.stderr,
            )
            return 1
        if not decision.compatible:
            print(
                "[deploy_strategy_exec] incompatible after validation: "
                + "; ".join(decision.diagnostics),
                file=sys.stderr,
            )
            return 1
        validation = _run_context_worker(
            {
                "action": "validate_roles",
                "strategy": strategy,
                "scene_module": scene_module,
                "instance": instance,
                "goal": goal,
                "roles": decision.roles,
            }
        )
        final_findings = validation.get("findings")
        if not isinstance(final_findings, list) or final_findings:
            print(
                "error: corrected deployment binding remains invalid: "
                + "; ".join(str(item) for item in (final_findings or [])),
                file=sys.stderr,
            )
            return 1

        decision = _validated_decision(decision, validation, run_dir)

    assert decision is not None
    destination = instantiation_dir("strategy", strategy, instance)
    try:
        manifest = _write_deployment_package(
            destination,
            strategy=strategy,
            instance=instance,
            scene_module=scene_module,
            goal=goal,
            roles=decision.roles,
        )
    except (OSError, ValueError) as exc:
        print(f"error: could not write deployment package: {exc}", file=sys.stderr)
        return 1
    layout_findings = sandboxed_instantiation_findings(
        "strategy",
        strategy,
        instance,
        scene_module=scene_module,
        goal=goal,
    )
    if layout_findings:
        print("error: deployment package validation failed:", file=sys.stderr)
        for finding in layout_findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"[deploy_strategy_exec] binding valid: {manifest}")
    if args.bind_only:
        return 0

    module = instantiation_module("strategy", strategy, instance) + ".main"
    returncode = _run_frozen_strategy(module)
    if returncode != 0:
        print(
            f"error: frozen strategy execution failed with exit {returncode}",
            file=sys.stderr,
        )
        return returncode
    return 0 if _report_execution_result(strategy, instance) else 1


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_DEPLOY_MODEL", "build_parser", "main", "run"]
