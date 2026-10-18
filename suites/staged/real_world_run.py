"""Bind and execute one real-world scene from a frozen evaluation suite."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from suites.staged.experiment_paths import (
    add_experiment_root_argument,
    configure_experiment_root,
)

from suites.staged.codex_launcher import _sync_instantiation_runtime
from suites.staged.gpu_render_broker import GPU_RENDER_MODES
from suites.staged.strategy_test import (
    build_parser as strategy_test_parser,
    normalize_test_case,
    resolve_strategy,
    resolve_suite_workspace,
    run_strategy_test,
)
from suites.staged.workspaces import repo_root_from_env


EXECUTION_POLICY_ENV = "RAPID_EXECUTION_POLICY"
GPU_RENDER_MODE_ENV = "RAPID_GPU_RENDER"
EXECUTION_POLICY = "attempt"
SCHEMA_VERSION = "rapid.real_world_run/v1"

RECORDING_ENV = {
    "record_width": "RAPID_RECORD_WIDTH",
    "record_height": "RAPID_RECORD_HEIGHT",
    "record_fps": "RAPID_RECORD_FPS",
    "record_playback_speed": "RAPID_RECORD_PLAYBACK_SPEED",
}


@dataclass(frozen=True)
class RealWorldTarget:
    """Resolved source checkpoint, task scene, and destination run."""

    eval_run: Path
    source_workspace: Path
    run_dir: Path
    scene_id: str
    scene_package: Path
    scene_module: str
    strategy: str
    goal: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-real-world-run",
        description=(
            "Bind and execute a tasks/invention_real2sim scene from an existing eval "
            "suite on the local machine. The execution policy is always attempt."
        ),
    )
    parser.add_argument(
        "scene_id",
        help="Real2Sim scene package name, for example scene_3.",
    )
    parser.add_argument(
        "eval_suite",
        help=(
            "Existing eval-suite ID or path, for example "
            "fr3_osc_demo_1_001_eval."
        ),
    )
    parser.add_argument(
        "--goal",
        default=None,
        help=(
            "Binding goal override. By default, reuse goal_text from the "
            "strategy's frozen example_0 binding."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--thinking-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--optimizer", choices=("cem", "mppi"), default="cem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu-render",
        choices=GPU_RENDER_MODES,
        default="required",
        help="GPU rendering policy (default: required, matching eval runs).",
    )
    parser.add_argument("--record-width", type=int, default=480)
    parser.add_argument("--record-height", type=int, default=270)
    parser.add_argument("--record-fps", type=int, default=10)
    parser.add_argument("--record-playback-speed", type=float, default=2.0)
    parser.add_argument(
        "--max-parallel-rollouts",
        type=int,
        default=None,
        help=(
            "Optional local CPU rollout worker cap. By default the runtime "
            "uses its affinity-aware automatic limit."
        ),
    )
    add_experiment_root_argument(parser)
    return parser


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _frozen_goal(workspace: Path, strategy: str) -> str:
    binding_path = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / "example_0"
        / "binding.json"
    )
    binding = _read_json_object(
        binding_path,
        description="frozen example_0 binding",
    )
    goal = binding.get("goal_text")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError(
            f"frozen example_0 binding has no goal_text: {binding_path}; "
            "pass --goal explicitly"
        )
    return goal.strip()


def resolve_target(
    root: Path,
    *,
    scene_id: str,
    eval_suite: str,
    goal: str | None,
) -> RealWorldTarget:
    root = root.resolve()
    if not scene_id.isidentifier() or scene_id.startswith("_"):
        raise ValueError(f"invalid Real2Sim scene ID: {scene_id!r}")
    scene_package = (root / "tasks" / "invention_real2sim" / scene_id).resolve()
    expected_scene_root = (root / "tasks" / "invention_real2sim").resolve()
    try:
        scene_package.relative_to(expected_scene_root)
    except ValueError as exc:  # defensive; isidentifier() already rejects separators
        raise ValueError(f"invalid Real2Sim scene ID: {scene_id!r}") from exc
    scene_module = normalize_test_case(root, str(scene_package))

    eval_run, source_workspace = resolve_suite_workspace(root, eval_suite)
    eval_manifest_path = eval_run / "benchmark_eval.json"
    eval_manifest = _read_json_object(
        eval_manifest_path,
        description="eval-suite manifest",
    )
    declared_strategy = eval_manifest.get("strategy")
    requested_strategy = (
        declared_strategy if isinstance(declared_strategy, str) else None
    )
    strategy = resolve_strategy(source_workspace / "src", requested_strategy)

    selected_goal = goal.strip() if goal is not None else _frozen_goal(
        source_workspace,
        strategy,
    )
    if not selected_goal:
        raise ValueError("binding goal cannot be empty")

    return RealWorldTarget(
        eval_run=eval_run,
        source_workspace=source_workspace,
        run_dir=(eval_run / "real_world_runs" / scene_id).resolve(),
        scene_id=scene_id,
        scene_package=scene_package,
        scene_module=scene_module,
        strategy=strategy,
        goal=selected_goal,
    )


def _positive_runtime_values(args: argparse.Namespace) -> dict[str, str]:
    values = {
        environment_name: getattr(args, option_name)
        for option_name, environment_name in RECORDING_ENV.items()
    }
    if args.max_parallel_rollouts is not None:
        values["RAPID_MAX_PARALLEL_ROLLOUTS"] = args.max_parallel_rollouts
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return {name: str(value) for name, value in values.items()}


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _strategy_args(
    target: RealWorldTarget,
    *,
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = [
        str(target.eval_run),
        str(target.scene_package),
        target.goal,
        "--strategy",
        target.strategy,
        "--name",
        target.scene_id,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--optimization-budget-mode",
        "maximum",
        "--sandbox-execution",
    ]
    if args.model:
        values.extend(("--model", args.model))
    if args.thinking_effort:
        values.extend(("--thinking-effort", args.thinking_effort))
    if args.max_turns is not None:
        values.extend(("--max-turns", str(args.max_turns)))
    return strategy_test_parser().parse_args(values)


def _archive_outputs(
    target: RealWorldTarget,
    *,
    workspace: Path,
) -> dict[str, str]:
    result_root = (
        workspace
        / "results_instantiation"
        / "strategies"
        / target.strategy
        / target.scene_id
    )
    binding_root = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / target.strategy
        / target.scene_id
    )
    archived: dict[str, str] = {}
    if (binding_root / "binding.json").is_file():
        destination = target.run_dir / "instantiation"
        shutil.copytree(binding_root, destination)
        archived["binding"] = str(destination / "binding.json")

    for filename, key in (
        ("metrics.json", "metrics"),
        ("metrics.arrays.npz", "metric_arrays"),
        ("cartesian_goals.json", "cartesian_goals"),
    ):
        source = result_root / filename
        if source.is_file():
            destination = target.run_dir / filename
            shutil.copy2(source, destination)
            archived[key] = str(destination)

    trajectory = result_root / "trajectory.mp4"
    if trajectory.is_file():
        destination = target.run_dir / "trajectory.mp4"
        shutil.move(str(trajectory), destination)
        archived["trajectory"] = str(destination)
    return archived


def _workspace_case_paths(target: RealWorldTarget) -> tuple[Path, Path, Path]:
    workspace = target.source_workspace
    binding_root = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / target.strategy
        / target.scene_id
    )
    result_root = (
        workspace
        / "results_instantiation"
        / "strategies"
        / target.strategy
        / target.scene_id
    )
    task_root = workspace / "task_inputs" / Path(
        *target.scene_module.removesuffix(".scene").split(".")
    )
    return binding_root, result_root, task_root


def _require_clean_workspace_case(target: RealWorldTarget) -> None:
    occupied = [
        path
        for path in _workspace_case_paths(target)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        detail = ", ".join(str(path) for path in occupied)
        raise FileExistsError(
            "eval workspace already contains state for this scene; refusing "
            f"to overwrite it: {detail}"
        )


def _clean_workspace_case(target: RealWorldTarget) -> None:
    """Remove only state created for this scene from the shared eval workspace."""

    for path in _workspace_case_paths(target):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _metric_status(metrics_path: str | None) -> tuple[object, object]:
    if metrics_path is None:
        return None, None
    path = Path(metrics_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    binding_valid = payload.get("binding_valid")
    program_success = payload.get("program_success")
    nested = payload.get("binding")
    if isinstance(nested, dict):
        if binding_valid is None:
            binding_valid = nested.get("binding_valid")
        if program_success is None:
            program_success = nested.get("program_success")
    return binding_valid, program_success


def _write_run_record(
    target: RealWorldTarget,
    *,
    workspace: Path,
    args: argparse.Namespace,
    execution_returncode: int,
    artifacts: Mapping[str, str],
) -> tuple[Path, bool]:
    binding_valid, program_success = _metric_status(artifacts.get("metrics"))
    required = ("binding", "metrics", "cartesian_goals", "trajectory")
    missing = [name for name in required if name not in artifacts]
    artifact_ready = not missing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "eval_suite": str(target.eval_run),
        "source_workspace": str(target.source_workspace),
        "workspace": str(workspace),
        "scene_id": target.scene_id,
        "scene_module": target.scene_module,
        "strategy": target.strategy,
        "goal_text": target.goal,
        "execution_policy": EXECUTION_POLICY,
        "optimization_budget_mode": "maximum",
        "optimizer": args.optimizer,
        "seed": args.seed,
        "execution_returncode": execution_returncode,
        "binding_valid": binding_valid,
        "program_success": program_success,
        "artifact_ready": artifact_ready,
        "missing_artifacts": missing,
        "artifacts": dict(artifacts),
    }
    output = target.run_dir / "real_world_run.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, artifact_ready


def run_real_world(args: argparse.Namespace, *, root: Path) -> int:
    target = resolve_target(
        root,
        scene_id=args.scene_id,
        eval_suite=args.eval_suite,
        goal=args.goal,
    )
    if target.run_dir.exists():
        raise FileExistsError(
            f"real-world run already exists: {target.run_dir}; "
            "use a different eval suite or remove the run explicitly"
        )
    _require_clean_workspace_case(target)

    print(f"[rapid-real-world] eval suite: {target.eval_run}", flush=True)
    print(f"[rapid-real-world] scene: {target.scene_module}", flush=True)
    print(f"[rapid-real-world] strategy: {target.strategy}", flush=True)
    print(f"[rapid-real-world] output: {target.run_dir}", flush=True)

    environment = {
        EXECUTION_POLICY_ENV: EXECUTION_POLICY,
        GPU_RENDER_MODE_ENV: args.gpu_render,
        **_positive_runtime_values(args),
    }
    target.run_dir.mkdir(parents=True)
    _sync_instantiation_runtime(
        repo_root=root,
        src=target.source_workspace / "src",
    )
    with _temporary_environment(environment):
        execution_returncode = run_strategy_test(
            _strategy_args(target, args=args),
            repo_root=root,
        )

    artifacts = _archive_outputs(target, workspace=target.source_workspace)
    record_path, artifact_ready = _write_run_record(
        target,
        workspace=target.source_workspace,
        args=args,
        execution_returncode=execution_returncode,
        artifacts=artifacts,
    )
    _clean_workspace_case(target)
    print(f"[rapid-real-world] run record: {record_path}", flush=True)
    for name in ("trajectory", "cartesian_goals", "metrics", "binding"):
        if path := artifacts.get(name):
            print(f"[rapid-real-world] {name}: {path}", flush=True)
    if not artifact_ready:
        missing = ", ".join(
            json.loads(record_path.read_text(encoding="utf-8"))["missing_artifacts"]
        )
        print(
            f"[rapid-real-world] missing required artifacts: {missing}",
            file=sys.stderr,
        )
        return execution_returncode or 1
    return execution_returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_experiment_root(args, repo_root_from_env().resolve())
    try:
        return run_real_world(args, root=repo_root_from_env().resolve())
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"[rapid-real-world] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
