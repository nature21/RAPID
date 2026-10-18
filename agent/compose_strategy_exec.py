"""Strategy-invention agent launched through `codex exec`."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primitive_core.common import GLOBAL_MAX_OPTIMIZATION_BUDGET

from agent.codex_exec import (
    AGENT_WORK_DIR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_MODEL,
    REASONING_EFFORTS,
    REPO_ROOT,
    STRATEGIES_DIR,
    CodexExecConfig,
    copy_demo_sources,
    normalize_demos,
    stage_workspace_codex_guidance,
)
from agent.descriptions import load_language_description
from agent.env_modules import normalize_env_modules, validate_env_modules_importable
from agent.invention_session import InventionSession
from agent.invention_status import (
    finalize_invention_status,
    initialize_invention_status,
)
from agent.strategy_context import (
    build_strategy_run_prompt,
    list_existing_primitives,
    list_existing_strategies,
    next_free_strategy_name,
)
from agent.strategy_validation import report_strategy_layout_validation
from agent.instantiation_validation import (
    report_generated_generalization_layout,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invent a new strategy from demos using `codex exec`."
    )
    parser.add_argument("demo_paths", nargs="+", type=Path)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num-variants", type=_at_least_two_int, default=2)
    parser.add_argument(
        "--variant-seed",
        type=int,
        default=7,
        help="Seed for deterministic geometry and pose variants.",
    )
    parser.add_argument(
        "--max-iters",
        type=_positive_int,
        default=GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
        help="Per-attempt optimizer iteration ceiling.",
    )
    parser.add_argument(
        "--env-module",
        "--env-modules",
        dest="env_modules",
        nargs="+",
        action="append",
        default=[],
    )
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--description-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--thinking-effort",
        choices=list(REASONING_EFFORTS),
        default="high",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Keyframes to extract per demo video.",
    )
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries after transient model-capacity failures (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Initial retry wait in seconds; doubles up to max(120, initial wait) (default: %(default)s).",
    )
    parser.add_argument(
        "--library-mode",
        choices=["open", "closed"],
        default="open",
    )
    parser.add_argument(
        "--no-stage-guidance",
        action="store_true",
        help="Do not refresh workspace AGENTS.md and .agents/skills templates.",
    )
    return parser.parse_args()


def _at_least_two_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    args = _parse_args()
    env_modules = normalize_env_modules(args.env_modules)
    env_module_findings = validate_env_modules_importable(env_modules)
    if env_module_findings:
        print("error: env module not importable:", file=sys.stderr)
        for finding in env_module_findings:
            print(f"  - {finding}", file=sys.stderr)
        sys.exit(2)

    try:
        language_description = load_language_description(
            args.description,
            args.description_file,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    resolved: list[Path] = []
    for path in args.demo_paths:
        real_path = path.expanduser().resolve()
        if not real_path.exists():
            print(f"error: demo path not found: {path}", file=sys.stderr)
            sys.exit(2)
        resolved.append(real_path)

    if args.name:
        strategy_name = args.name
        if (STRATEGIES_DIR / strategy_name).exists():
            print(
                f"error: src/strategies/{strategy_name} already exists. "
                "Pick another --name or remove the folder.",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        strategy_name = next_free_strategy_name()
    print(f"[compose_strategy_exec] strategy name: {strategy_name}")

    available_primitives = list_existing_primitives()
    existing_strategies = list_existing_strategies()
    print(
        f"[compose_strategy_exec] {len(available_primitives)} primitive(s) available: "
        f"{', '.join(available_primitives) or '(none)'}"
    )
    print(
        f"[compose_strategy_exec] {len(existing_strategies)} existing strategy(ies): "
        f"{', '.join(existing_strategies) or '(none)'}"
    )

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = AGENT_WORK_DIR / f"compose_strategy_exec_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compose_strategy_exec] run dir: {run_dir.relative_to(REPO_ROOT)}")
    initialize_invention_status(
        kind="strategy",
        num_variants=args.num_variants,
        run_dir=run_dir,
    )

    if not args.no_stage_guidance:
        staged = stage_workspace_codex_guidance()
        print(f"[compose_strategy_exec] staged Codex guidance files: {len(staged)}")

    demo_sources = copy_demo_sources(resolved, run_dir)
    frames = normalize_demos(demo_sources, run_dir, args.num_frames)
    if not frames:
        print("error: no usable demo frames produced.", file=sys.stderr)
        sys.exit(2)
    print(
        f"[compose_strategy_exec] {len(frames)} demo frame(s) prepared under "
        f"{(run_dir / 'demo_frames').relative_to(REPO_ROOT)}"
    )

    prompt = build_strategy_run_prompt(
        demo_frames=frames,
        demo_sources=demo_sources,
        strategy_name=strategy_name,
        available_primitives=available_primitives,
        existing_strategies=existing_strategies,
        max_iters=args.max_iters,
        num_variants=args.num_variants,
        variant_seed=args.variant_seed,
        library_mode=args.library_mode,
        env_modules=env_modules,
        language_description=language_description,
    )

    print(
        f"[compose_strategy_exec] model={args.model} "
        f"reasoning_effort={args.thinking_effort}"
    )
    print(
        f"[compose_strategy_exec] max_iters={args.max_iters} "
        f"num_variants={args.num_variants} variant_seed={args.variant_seed} "
        f"max_turns={args.max_turns} library_mode={args.library_mode} "
        f"max_retries={args.max_retries} retry_delay={args.retry_delay:g}s"
    )
    if env_modules:
        print("[compose_strategy_exec] env_modules=" + ", ".join(env_modules))
    if language_description:
        print(f"[compose_strategy_exec] description_chars={len(language_description)}")
    print("[compose_strategy_exec] launching codex exec...\n")

    session = InventionSession(
        kind="strategy",
        run_dir=run_dir,
        iteration_ceiling=args.max_iters,
        config=CodexExecConfig(
            model=args.model,
            reasoning_effort=args.thinking_effort,
            max_turns=args.max_turns,
            max_retries=max(0, args.max_retries),
            retry_delay=max(0.0, args.retry_delay),
        ),
    )
    provenance_findings = session.run(prompt, frames)
    layout_findings = report_strategy_layout_validation(
        strategy_name,
        fatal=False,
    )
    generalization_findings = report_generated_generalization_layout(
        "strategy",
        strategy_name,
        args.num_variants,
        fatal=False,
    )
    finalize_invention_status(
        kind="strategy",
        programs=(strategy_name,),
        num_variants=args.num_variants,
        run_dir=run_dir,
        validation_findings=(
            *provenance_findings,
            *layout_findings,
            *generalization_findings,
        ),
    )


if __name__ == "__main__":
    main()
