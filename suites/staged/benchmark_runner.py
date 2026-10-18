"""Evaluate one frozen strategy on sealed benchmark cases in fresh workspaces."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from agent.instantiation_context import (
    CANONICAL_INIT,
    CANONICAL_MAIN,
    CANONICAL_PARENT_INIT,
    initial_binding_manifest,
)
from benchmark_private.cli import private_seal_root
from benchmark_private.generator import tree_sha256
from benchmark_private.specs import (
    BENCHMARK_VERSION,
    CASES_PER_STRATEGY,
    STRATEGY_CONTRACTS,
)
from benchmark_private.success import (
    evaluate_benchmark_success,
    rescore_retained_benchmark_evaluation,
)
from suites.staged.codex_launcher import _copy_task_bind, _sync_instantiation_runtime
from suites.staged.codex_permissions import PROFILE_NAME, run_codex_home_path
from suites.staged.gpu_render_broker import (
    GPU_RENDER_MODE_ENV,
    GPU_RENDER_QUEUE_ENV,
    start_gpu_render_broker,
)
from suites.staged.strategy_test import (
    _run as run_strategy_command,
    build_parser as strategy_test_parser,
    resolve_suite_workspace,
    run_strategy_test,
)
from suites.staged.workspaces import create_isolated_workspace, repo_root_from_env


EXECUTION_POLICY_ENV = "RAPID_EXECUTION_POLICY"
EXECUTION_POLICIES = ("strict", "attempt")
DEFAULT_EXECUTION_POLICY = "attempt"


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _frozen_checkpoint_digest(src: Path, strategy: str) -> str:
    roots = [
        src / "strategies" / strategy,
        src / "primitives",
        src
        / "instantiations"
        / "strategies"
        / strategy
        / "example_0",
    ]
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        digest.update(root.relative_to(src).as_posix().encode())
        digest.update(_tree_digest(root).encode())
    return digest.hexdigest()


def _case_indices(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return tuple(range(1, CASES_PER_STRATEGY + 1))
    result: set[int] = set()
    for raw in values:
        for part in raw.split(","):
            text = part.strip()
            if not text:
                continue
            if "-" in text:
                start, end = (int(item) for item in text.split("-", 1))
                if start > end:
                    raise ValueError(f"case range must be ascending: {text}")
                result.update(range(start, end + 1))
            else:
                result.add(int(text))
    invalid = sorted(item for item in result if not 1 <= item <= CASES_PER_STRATEGY)
    if invalid:
        raise ValueError(f"case indices must be in 1..50: {invalid}")
    if not result:
        raise ValueError("at least one benchmark case must be selected")
    return tuple(sorted(result))


def _resolve_seal_root(root: Path, value: Path | None) -> Path:
    if value is None:
        return private_seal_root(root)
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _load_seal(
    root: Path,
    strategy_id: int,
    *,
    seal_root: Path | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    seal_root = _resolve_seal_root(root, seal_root)
    seal_path = seal_root / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(f"benchmark seal is missing or incompatible: {seal_path}")
    strategy = seal.get("strategies", {}).get(str(strategy_id))
    if not isinstance(strategy, dict) or strategy.get("case_count") != CASES_PER_STRATEGY:
        raise ValueError(f"strategy {strategy_id} is not fully sealed")
    records = {int(item["case_index"]): dict(item) for item in strategy["cases"]}
    if len(records) != CASES_PER_STRATEGY or any(
        item.get("certified") is not True for item in records.values()
    ):
        raise ValueError(f"strategy {strategy_id} seal contains uncertified cases")
    if not seal.get("sealed"):
        print(
            "[rapid-benchmark] full corpus is not sealed; evaluating the "
            f"complete strategy {strategy_id} seal only",
            file=sys.stderr,
        )
    return seal, records


def _load_private_manifest(
    root: Path,
    strategy_id: int,
    case_index: int,
    *,
    expected_sha256: str,
    seal_root: Path | None = None,
) -> dict[str, Any]:
    path = (
        _resolve_seal_root(root, seal_root)
        / "manifests"
        / f"strategy_{strategy_id}"
        / f"case_{case_index:03d}.json"
    )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"sealed private manifest checksum mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _all_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(_all_strings(item))
        return result
    if isinstance(value, list | tuple):
        result = set()
        for item in value:
            result.update(_all_strings(item))
        return result
    return set()


def _bound_scene_body(name: str, bound_names: set[str]) -> bool:
    """Accept either a body role or that body's canonical collision geom."""

    return name in bound_names or f"{name}_collision" in bound_names


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _strategy_test_namespace(
    *,
    run_dir: Path,
    case_dir: Path,
    goal: str,
    strategy: str,
    instance: str,
    args: argparse.Namespace,
) -> argparse.Namespace:
    parser = strategy_test_parser()
    values = [str(run_dir), str(case_dir), goal, "--strategy", strategy, "--name", instance]
    values.extend(["--optimizer", args.optimizer, "--seed", str(args.seed)])
    values.extend(["--optimization-budget-mode", "maximum"])
    values.extend([
        "--execution-log",
        str(Path("results_instantiation") / "strategies" / strategy / instance / "execution.log"),
    ])
    values.extend(
        [
            "--state-trajectory-npz",
            str(
                Path("results_instantiation")
                / "strategies"
                / strategy
                / instance
                / "state_trajectory.npz"
            ),
        ]
    )
    values.append("--sandbox-execution")
    values.append("--no-record") if args.no_record else None
    values.append("--allow-program-failure") if args.allow_program_failure else None
    if args.model:
        values.extend(["--model", args.model])
    if args.thinking_effort:
        values.extend(["--thinking-effort", args.thinking_effort])
    if args.max_turns is not None:
        values.extend(["--max-turns", str(args.max_turns)])
    return parser.parse_args(values)


def _remap_authoritative_roles(
    value: Any,
    *,
    source_target: str,
    case_target: str,
    entity_renames: dict[str, str] | None = None,
) -> Any:
    """Replace all materialized scene entity names in a frozen binding."""

    if isinstance(value, str):
        renames = {source_target: case_target, **(entity_renames or {})}
        for source in sorted(renames, key=len, reverse=True):
            if value == source or value.startswith(f"{source}_"):
                return renames[source] + value[len(source) :]
        return value
    if isinstance(value, list):
        return [
            _remap_authoritative_roles(
                item,
                source_target=source_target,
                case_target=case_target,
                entity_renames=entity_renames,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _remap_authoritative_roles(
                item,
                source_target=source_target,
                case_target=case_target,
                entity_renames=entity_renames,
            )
            for key, item in value.items()
        }
    return value


def _write_authoritative_binding(
    workspace: Path,
    source_workspace: Path,
    *,
    strategy_id: int,
    strategy: str,
    instance: str,
    scene_module: str,
    goal: str,
    private_manifest: dict[str, Any],
) -> Path:
    """Write the hidden entity-remapped version of the learned demo binding."""

    source_path = (
        source_workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / "example_0"
        / "binding.json"
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            "authoritative evaluation requires the frozen example_0 binding: "
            f"{source_path}"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    roles = source.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"frozen example binding has no roles: {source_path}")
    contract = STRATEGY_CONTRACTS[strategy_id]
    case_target = str(private_manifest["authoritative_roles"]["target"])
    remapped_roles = _remap_authoritative_roles(
        roles,
        source_target=contract.target_body,
        case_target=case_target,
        entity_renames={
            str(source): str(target)
            for source, target in private_manifest.get("entity_renames", {}).items()
        },
    )

    destination = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / instance
    )
    if destination.exists():
        raise FileExistsError(f"authoritative binding already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_init = destination.parent / "__init__.py"
    if parent_init.is_file():
        if parent_init.read_text(encoding="utf-8") != CANONICAL_PARENT_INIT:
            raise ValueError(f"noncanonical parent package: {parent_init}")
    else:
        parent_init.write_text(CANONICAL_PARENT_INIT, encoding="utf-8")
    destination.mkdir()
    (destination / "__init__.py").write_text(CANONICAL_INIT, encoding="utf-8")
    (destination / "main.py").write_text(CANONICAL_MAIN, encoding="utf-8")
    payload = json.loads(
        initial_binding_manifest(
            kind="strategy",
            program=strategy,
            instance=instance,
            scene_module=scene_module,
            goal=goal,
        )
    )
    payload["roles"] = remapped_roles
    payload["inputs"] = source.get("inputs", {})
    manifest = destination / "binding.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _run_authoritative_case(
    *,
    root: Path,
    source_workspace: Path,
    workspace: Path,
    case_dir: Path,
    private_manifest: dict[str, Any],
    strategy_id: int,
    strategy: str,
    instance: str,
    args: argparse.Namespace,
) -> int:
    """Execute one oracle-bound case without exposing hidden data to code."""

    _sync_instantiation_runtime(repo_root=root, src=workspace / "src")
    _copy_task_bind(case_dir, repo_root=root, workspace=workspace)
    scene_module = ".".join(case_dir.relative_to(root).parts) + ".scene"
    _write_authoritative_binding(
        workspace,
        source_workspace,
        strategy_id=strategy_id,
        strategy=strategy,
        instance=instance,
        scene_module=scene_module,
        goal=str(private_manifest["goal"]),
        private_manifest=private_manifest,
    )
    result_root = (
        workspace / "results_instantiation" / "strategies" / strategy / instance
    )
    metrics_path = result_root / "metrics.json"
    command = [
        shutil.which("codex") or "codex",
        "sandbox",
        "-C",
        str(workspace),
        "-P",
        PROFILE_NAME,
        "--",
        str(root / "scripts" / "rapid-python"),
        "-m",
        f"instantiations.strategies.{strategy}.{instance}.main",
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--metrics-json",
        str(metrics_path),
        "--state-trajectory-npz",
        str(result_root / "state_trajectory.npz"),
    ]
    if not args.no_record:
        command.extend(["--record", str(result_root / "trajectory.mp4")])
    else:
        command.append("--no-record")
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(workspace / "agent_work" / "home"),
            "CODEX_HOME": str(run_codex_home_path(workspace)),
            "RAPID_REPO_ROOT": str(root),
            "RAPID_WORKSPACE": str(workspace),
            "RAPID_ACTIVE_SRC_DIR": str(workspace / "src"),
            "RAPID_SRC_DIR": str(workspace / "src"),
            "TMPDIR": str(workspace / "agent_work" / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment["RAPID_OPTIMIZATION_BUDGET_MODE"] = "maximum"
    Path(environment["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    print("[rapid-benchmark] executing authoritative binding in sealed sandbox")
    broker = None
    try:
        broker = start_gpu_render_broker(
            workspace=workspace,
            env=environment,
            mode=environment.get(GPU_RENDER_MODE_ENV, "auto"),
        )
        if broker is not None:
            environment[GPU_RENDER_QUEUE_ENV] = str(broker.queue_path)
        else:
            environment.pop(GPU_RENDER_QUEUE_ENV, None)
        return run_strategy_command(
            command,
            cwd=workspace,
            env=environment,
            log_path=result_root / "execution.log",
        )
    finally:
        if broker is not None:
            broker.stop()


def _case_result(
    workspace: Path,
    *,
    strategy_id: int,
    strategy: str,
    instance: str,
    private_manifest: dict[str, Any],
    returncode: int,
) -> dict[str, Any]:
    result_root = workspace / "results_instantiation" / "strategies" / strategy / instance
    metrics_path = result_root / "metrics.json"
    state_trajectory_path = result_root / "state_trajectory.npz"
    binding_path = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / instance
        / "binding.json"
    )
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    binding = (
        json.loads(binding_path.read_text(encoding="utf-8"))
        if binding_path.is_file()
        else {}
    )
    bound_names = _all_strings(binding.get("roles", {}))
    authoritative = private_manifest["authoritative_roles"]
    expected = [
        str(authoritative["target"]),
        *[str(item) for item in authoritative.get("fixtures", [])],
        *[str(item) for item in authoritative.get("secondary_dynamic", [])],
    ]
    covered = [name for name in expected if _bound_scene_body(name, bound_names)]
    program_success = metrics.get("program_success")
    binding_valid = metrics.get("binding_valid")
    if program_success is None and isinstance(metrics.get("binding"), dict):
        program_success = metrics["binding"].get("program_success")
    if binding_valid is None and isinstance(metrics.get("binding"), dict):
        binding_valid = metrics["binding"].get("binding_valid")
    if binding_valid is None and binding_path.is_file():
        # The binding package is published only after host-side contract
        # validation, so it remains authoritative when strategy preflight
        # fails before a metrics file can be written.
        binding_valid = True
    record = {
        "returncode": returncode,
        "binding_written": binding_path.is_file(),
        "binding_valid": binding_valid is True,
        "target_bound": _bound_scene_body(
            str(authoritative["target"]),
            bound_names,
        ),
        "authoritative_scene_role_coverage": len(covered) / len(expected) if expected else 1.0,
        "covered_scene_roles": covered,
        "expected_scene_roles": expected,
        "program_success": program_success is True,
        "metrics_path": str(metrics_path),
        "execution_log_path": str(result_root / "execution.log"),
        "state_trajectory_path": str(state_trajectory_path),
        "binding_path": str(binding_path),
        "bucket": private_manifest["parameters"]["bucket"],
        "diagnostic_dimension": private_manifest["parameters"].get("diagnostic_dimension"),
        "stress_axis": private_manifest["parameters"].get("stress_axis"),
    }
    scene_module = binding.get("scene_module")
    return _attach_benchmark_evaluation(
        record,
        strategy_id,
        metrics,
        state_trajectory_path=state_trajectory_path,
        private_manifest=private_manifest,
        scene_module=scene_module if isinstance(scene_module, str) else None,
    )


def _attach_benchmark_evaluation(
    record: dict[str, Any],
    strategy_id: int,
    metrics: dict[str, Any],
    *,
    state_trajectory_path: str | Path | None = None,
    private_manifest: Mapping[str, Any] | None = None,
    scene_module: str | None = None,
) -> dict[str, Any]:
    """Attach a benchmark-owned decision to one retained case record."""

    result = _attach_execution_diagnostics(record, metrics)
    existing = result.get("benchmark_evaluation")
    if (
        state_trajectory_path is None
        and private_manifest is None
        and isinstance(existing, Mapping)
        and existing.get("source") == "state_trajectory"
    ):
        benchmark_evaluation = rescore_retained_benchmark_evaluation(
            strategy_id,
            existing,
        )
        if benchmark_evaluation is None:
            return result
    else:
        benchmark_evaluation = evaluate_benchmark_success(
            strategy_id,
            metrics,
            state_trajectory_path=state_trajectory_path,
            private_manifest=private_manifest,
            scene_module=scene_module,
        )
    if benchmark_evaluation is None and isinstance(existing, Mapping):
        return result
    result["benchmark_evaluation"] = benchmark_evaluation
    result["benchmark_success"] = (
        bool(benchmark_evaluation["success"])
        if benchmark_evaluation is not None
        else None
    )
    if (
        result["benchmark_success"] is True
        and result.get("execution_completed") is True
    ):
        result["failure_stage"] = None
        result["failure_segment"] = None
        result["failure_type"] = None
        result["failure_message"] = None
    return result


def _logged_execution_failure(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover the last traceback exception when no trajectory was retained."""
    path = record.get("execution_log_path")
    if not path:
        return None
    try:
        text = Path(str(path)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "Traceback (most recent call last):" not in text:
        return None
    traceback = text.rsplit("Traceback (most recent call last):", 1)[1]
    matches = re.findall(
        r"^(?:[\w]+\.)*([\w]*(?:Error|Exception|Violation|Failure|SearchExhausted)):\s*(.*)$",
        traceback, re.MULTILINE,
    )
    if not matches:
        return None
    failure_type, message = matches[-1]
    stage = {
        "FR3TransitViolation": "transit",
        "FR3TransitPlanningFailure": "transit",
        "PhysicalPreconditionError": "primitive_geometry",
        "CandidateSearchExhausted": "primitive_geometry",
    }.get(failure_type, "execution")
    return {"failure_stage": stage, "failure_segment": None,
            "failure_type": failure_type, "failure_message": message}


def _attach_execution_diagnostics(
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify where execution stopped without changing benchmark scoring."""

    result = dict(record)
    binding_written = bool(result.get("binding_written"))
    if not binding_written:
        binding_path = Path(str(result.get("binding_path", "")))
        binding_written = binding_path.is_file()
    if result.get("binding_valid") is not True and binding_written:
        result["binding_valid"] = True
    binding_path = Path(str(result.get("binding_path", "")))
    expected_roles = result.get("expected_scene_roles")
    if binding_path.is_file() and isinstance(expected_roles, list):
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            binding = {}
        bound_names = _all_strings(binding.get("roles", {}))
        expected = [str(value) for value in expected_roles]
        covered = [
            name for name in expected if _bound_scene_body(name, bound_names)
        ]
        result["covered_scene_roles"] = covered
        result["authoritative_scene_role_coverage"] = (
            len(covered) / len(expected) if expected else 1.0
        )
        if expected:
            result["target_bound"] = _bound_scene_body(expected[0], bound_names)

    nested = metrics.get("metrics")
    core = nested if isinstance(nested, Mapping) else metrics
    metrics_available = bool(metrics)
    partial = core.get("partial_strategy") is True
    phase_metrics = core.get("phase_metrics")
    if not isinstance(phase_metrics, Mapping):
        phase_metrics = {}
    completed_segments = core.get("completed_segments")
    ordered_segments = (
        [str(value) for value in completed_segments]
        if isinstance(completed_segments, list)
        else [str(value) for value in phase_metrics]
    )
    failed_primitives: list[str] = []
    for segment in ordered_segments:
        phase = phase_metrics.get(segment)
        if (
            "__to__" in segment
            or segment.endswith("__rejected")
            or not isinstance(phase, Mapping)
        ):
            continue
        if phase.get("effect_success") is False or phase.get("success") is False:
            failed_primitives.append(segment)

    result["execution_started"] = bool(metrics_available or binding_written)
    result["execution_completed"] = bool(metrics_available and not partial)
    result["metrics_available"] = metrics_available
    result["failed_primitive_segments"] = failed_primitives
    attempt_continuations = core.get("attempt_continuations")
    if not isinstance(attempt_continuations, list):
        attempt_continuations = []
    result["attempt_continuations"] = attempt_continuations
    result["strict_would_abort"] = bool(
        core.get("strict_would_abort") is True or attempt_continuations
    )
    result.pop("downstream_failure_segment", None)
    result.pop("downstream_failure_type", None)
    result.pop("downstream_failure_message", None)

    if result.get("binding_valid") is not True:
        result.update(
            {
                "failure_stage": "binding",
                "failure_segment": None,
                "failure_type": "BindingFailure",
                "failure_message": "no validated binding package was produced",
            }
        )
        return result
    if not metrics_available:
        failure = _logged_execution_failure(result)
        if failure is not None:
            result.update(failure)
            return result
        result.update(
            {
                "failure_stage": "strategy_preflight",
                "failure_segment": None,
                "failure_type": "StrategyPreflightFailure",
                "failure_message": (
                    "strategy execution stopped before runtime metrics were written"
                ),
            }
        )
        return result

    failed_segment = core.get("failed_segment")
    failure_type = core.get("failure_type")
    failure_message = core.get("failure_message")
    if partial:
        if failed_primitives and (
            failure_type == "ValueError"
            and "gripper aperture" in str(failure_message).lower()
        ):
            root_segment = failed_primitives[-1]
            result.update(
                {
                    "failure_stage": "primitive_effect",
                    "failure_segment": root_segment,
                    "failure_type": "PrimitiveEffectFailure",
                    "failure_message": (
                        f"{root_segment} failed its effect predicate before "
                        "the downstream grasp geometry was resolved"
                    ),
                    "downstream_failure_segment": failed_segment,
                    "downstream_failure_type": failure_type,
                    "downstream_failure_message": failure_message,
                }
            )
        elif failure_type == "FR3TransitViolation" or "__to__" in str(
            failed_segment
        ):
            result.update(
                {
                    "failure_stage": "transit",
                    "failure_segment": failed_segment,
                    "failure_type": failure_type or "TransitFailure",
                    "failure_message": failure_message
                    or "the inter-primitive transit could not be certified",
                }
            )
        else:
            result.update(
                {
                    "failure_stage": "primitive_geometry",
                    "failure_segment": failed_segment,
                    "failure_type": failure_type or "ExecutionFailure",
                    "failure_message": failure_message
                    or "strategy execution stopped during primitive resolution",
                }
            )
        return result

    if result.get("program_success") is True:
        result.update(
            {
                "failure_stage": None,
                "failure_segment": None,
                "failure_type": None,
                "failure_message": None,
            }
        )
    else:
        root_segment = failed_primitives[-1] if failed_primitives else None
        result.update(
            {
                "failure_stage": "completed_unsuccessful",
                "failure_segment": root_segment,
                "failure_type": "PrimitiveEffectFailure"
                if root_segment
                else "PredicateFailure",
                "failure_message": (
                    f"{root_segment} failed its effect predicate"
                    if root_segment
                    else "the completed strategy did not satisfy its predicate"
                ),
            }
        )
    return result


def _archive_shared_case(
    workspace: Path,
    run_dir: Path,
    case_dir: Path,
    *,
    root: Path,
    strategy: str,
    instance: str,
) -> dict[str, str]:
    """Retain one shared-workspace result, then remove case-local state."""

    artifact_root = run_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    instance_root = (
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / instance
    )
    result_root = (
        workspace / "results_instantiation" / "strategies" / strategy / instance
    )
    archived: dict[str, str] = {}
    if instance_root.is_dir():
        binding_archive = artifact_root / "instantiation"
        shutil.copytree(instance_root, binding_archive)
        archived["binding_path"] = str(binding_archive / "binding.json")
    if result_root.is_dir():
        for filename in (
            "execution.log",
            "metrics.json",
            "metrics.arrays.npz",
            "cartesian_goals.json",
            "state_trajectory.npz",
        ):
            source = result_root / filename
            if source.is_file():
                destination = artifact_root / filename
                shutil.copy2(source, destination)
                if filename == "metrics.json":
                    archived["metrics_path"] = str(destination)
                elif filename == "execution.log":
                    archived["execution_log_path"] = str(destination)
                elif filename == "state_trajectory.npz":
                    archived["state_trajectory_path"] = str(destination)
        trajectory = result_root / "trajectory.mp4"
        if trajectory.is_file():
            destination = artifact_root / "trajectory.mp4"
            shutil.move(str(trajectory), destination)
            archived["trajectory_path"] = str(destination)

    # The next binding agent must not see previous benchmark bindings, task
    # scenes, or result artifacts. Only frozen checkpoint state is reused.
    if instance_root.exists():
        shutil.rmtree(instance_root)
    if result_root.exists():
        shutil.rmtree(result_root)
    task_copy = workspace / "task_inputs" / case_dir.relative_to(root)
    if task_copy.exists():
        shutil.rmtree(task_copy)
    return archived


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def scored_success(item: dict[str, Any]) -> bool:
        benchmark_success = item.get("benchmark_success")
        if isinstance(benchmark_success, bool):
            return benchmark_success
        return item.get("program_success") is True

    successes = sum(scored_success(item) for item in records)
    by_bucket: dict[str, dict[str, int]] = {}
    by_dimension: dict[str, dict[str, int]] = {}
    by_stress_axis: dict[str, dict[str, int]] = {}
    for item in records:
        for mapping, key in (
            (by_bucket, str(item["bucket"])),
            (by_dimension, str(item["diagnostic_dimension"] or "none")),
            (by_stress_axis, str(item["stress_axis"] or "none")),
        ):
            entry = mapping.setdefault(key, {"successes": 0, "total": 0})
            entry["total"] += 1
            entry["successes"] += int(scored_success(item))
    benchmark_scored = sum(
        isinstance(item.get("benchmark_success"), bool) for item in records
    )
    disagreements = sum(
        item.get("benchmark_success") != item.get("program_success")
        for item in records
        if isinstance(item.get("benchmark_success"), bool)
        and isinstance(item.get("program_success"), bool)
    )
    forced_continuation_cases = sum(
        bool(item.get("strict_would_abort")) for item in records
    )
    successes_requiring_forced_continuation = sum(
        bool(item.get("strict_would_abort")) and scored_success(item)
        for item in records
    )
    return {
        "successes": successes,
        "total": len(records),
        "success_rate": successes / len(records) if records else None,
        "benchmark_scored_cases": benchmark_scored,
        "strategy_predicate_successes": sum(
            item.get("program_success") is True for item in records
        ),
        "benchmark_strategy_disagreements": disagreements,
        "forced_continuation_cases": forced_continuation_cases,
        "successes_requiring_forced_continuation": (
            successes_requiring_forced_continuation
        ),
        "wilson_95": _wilson(successes, len(records)),
        "target_binding_accuracy": (
            sum(bool(item["target_bound"]) for item in records) / len(records)
            if records
            else None
        ),
        "binding_valid_rate": (
            sum(bool(item["binding_valid"]) for item in records) / len(records)
            if records
            else None
        ),
        "mean_authoritative_scene_role_coverage": (
            sum(float(item["authoritative_scene_role_coverage"]) for item in records)
            / len(records)
            if records
            else None
        ),
        "by_bucket": by_bucket,
        "by_diagnostic_dimension": by_dimension,
        "by_stress_axis": by_stress_axis,
    }


def _validated_resume_record(
    path: Path,
    *,
    strategy_id: int,
    case_index: int,
    case_sha256: str,
    checkpoint_sha256: str,
    binding_mode: str,
    execution_policy: str = "strict",
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "strategy_id": strategy_id,
        "case_index": case_index,
        "case_sha256": case_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "binding_mode": binding_mode,
        "optimization_budget_mode": "maximum",
        "execution_policy": execution_policy,
    }
    mismatches = []
    for key, value in expected.items():
        actual = record.get(key)
        if key == "execution_policy" and actual is None:
            actual = "strict"
        if actual != value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"resume record does not match the sealed run ({', '.join(mismatches)}): {path}"
        )
    metrics_path = Path(str(record.get("metrics_path", "")))
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics = {}
    refreshed = _attach_benchmark_evaluation(record, strategy_id, metrics)
    if refreshed != record:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(refreshed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return refreshed


def run_benchmark(args: argparse.Namespace, *, root: Path) -> int:
    os.environ[EXECUTION_POLICY_ENV] = args.execution_policy
    _, source_workspace = resolve_suite_workspace(root, args.source_suite)
    checkpoint_digest = _frozen_checkpoint_digest(source_workspace / "src", args.strategy)
    seal_root = _resolve_seal_root(root, args.seal_root)
    _, sealed_cases = _load_seal(
        root,
        args.strategy_id,
        seal_root=seal_root,
    )
    indices = _case_indices(args.case)
    output_root = (
        args.output.expanduser().resolve()
        if args.output
        else root
        / "benchmark_runs"
        / source_workspace.parent.name
        / f"strategy_{args.strategy_id}"
        / args.binding_mode
    )
    output_root.mkdir(parents=True, exist_ok=True)
    scene_root = (
        args.scene_root.expanduser().resolve()
        if args.scene_root is not None
        else root / "tasks" / "sim_benchmark" / f"strategy_{args.strategy_id}"
    )
    try:
        scene_root.relative_to(root / "tasks")
    except ValueError as exc:
        raise ValueError(f"benchmark scene root must live under {root / 'tasks'}") from exc
    if not scene_root.is_dir():
        raise FileNotFoundError(f"benchmark scene root does not exist: {scene_root}")

    shared_workspace_info = None
    if args.workspace_mode == "shared":
        shared_run_dir = output_root / "_shared_workspace"
        if (shared_run_dir / "workspace" / "src").is_dir():
            if not args.resume:
                raise FileExistsError(
                    f"shared benchmark workspace already exists: {shared_run_dir}; "
                    "use --resume or a new --output"
                )
            shared_workspace = shared_run_dir / "workspace"
            shared_workspace_info = {
                "run_dir": shared_run_dir,
                "workspace": shared_workspace,
                "src": shared_workspace / "src",
            }
        else:
            created = create_isolated_workspace(
                shared_run_dir,
                repo_root=root,
                from_workspace=source_workspace,
            )
            shared_workspace_info = {
                "run_dir": created.run_dir,
                "workspace": created.workspace,
                "src": created.src,
            }
        if (
            _frozen_checkpoint_digest(shared_workspace_info["src"], args.strategy)
            != checkpoint_digest
        ):
            raise RuntimeError("shared workspace changed the frozen strategy checkpoint")
    elif args.workspace_mode == "direct":
        shared_workspace_info = {
            "run_dir": source_workspace.parent,
            "workspace": source_workspace,
            "src": source_workspace / "src",
        }
    records: list[dict[str, Any]] = []
    for case_index in indices:
        case_dir = scene_root / f"case_{case_index:03d}_scene"
        sealed = sealed_cases[case_index]
        if tree_sha256(case_dir) != sealed["case_sha256"]:
            raise ValueError(f"sealed case checksum mismatch: {case_dir}")
        run_dir = output_root / f"case_{case_index:03d}"
        result_path = run_dir / "benchmark_result.json"
        if result_path.is_file() and args.resume:
            records.append(
                _validated_resume_record(
                    result_path,
                    strategy_id=args.strategy_id,
                    case_index=case_index,
                    case_sha256=str(sealed["case_sha256"]),
                    checkpoint_sha256=checkpoint_digest,
                    binding_mode=args.binding_mode,
                    execution_policy=args.execution_policy,
                )
            )
            continue
        if run_dir.exists():
            raise FileExistsError(
                f"benchmark case workspace already exists: {run_dir}; "
                "use --resume or a new --output"
            )
        if args.workspace_mode in {"direct", "shared"}:
            assert shared_workspace_info is not None
            run_dir.mkdir(parents=True)
            execution_run_dir = shared_workspace_info["run_dir"]
            workspace = shared_workspace_info["workspace"]
            workspace_src = shared_workspace_info["src"]
        else:
            workspace_info = create_isolated_workspace(
                run_dir,
                repo_root=root,
                from_workspace=source_workspace,
            )
            execution_run_dir = run_dir
            workspace = workspace_info.workspace
            workspace_src = workspace_info.src
        if _frozen_checkpoint_digest(workspace_src, args.strategy) != checkpoint_digest:
            raise RuntimeError("fresh workspace changed the frozen strategy checkpoint")
        private_manifest = _load_private_manifest(
            root,
            args.strategy_id,
            case_index,
            expected_sha256=str(sealed["manifest_sha256"]),
            seal_root=seal_root,
        )
        instance = f"benchmark_{case_index:03d}"
        if args.binding_mode == "agent":
            _sync_instantiation_runtime(repo_root=root, src=workspace_src)
            test_args = _strategy_test_namespace(
                run_dir=execution_run_dir,
                case_dir=case_dir,
                goal=private_manifest["goal"],
                strategy=args.strategy,
                instance=instance,
                args=args,
            )
            returncode = run_strategy_test(test_args, repo_root=root)
        else:
            returncode = _run_authoritative_case(
                root=root,
                source_workspace=source_workspace,
                workspace=workspace,
                case_dir=case_dir,
                private_manifest=private_manifest,
                strategy_id=args.strategy_id,
                strategy=args.strategy,
                instance=instance,
                args=args,
            )
        record = _case_result(
            workspace,
            strategy_id=args.strategy_id,
            strategy=args.strategy,
            instance=instance,
            private_manifest=private_manifest,
            returncode=returncode,
        )
        if args.workspace_mode in {"direct", "shared"}:
            record.update(
                _archive_shared_case(
                    workspace,
                    run_dir,
                    case_dir,
                    root=root,
                    strategy=args.strategy,
                    instance=instance,
                )
            )
            if _frozen_checkpoint_digest(workspace_src, args.strategy) != checkpoint_digest:
                raise RuntimeError(
                    f"case {case_index:03d} changed the frozen strategy checkpoint"
                )
        record.update(
            {
                "strategy_id": args.strategy_id,
                "case_index": case_index,
                "case_sha256": sealed["case_sha256"],
                "checkpoint_sha256": checkpoint_digest,
                "binding_mode": args.binding_mode,
                "optimization_budget_mode": "maximum",
                "execution_policy": args.execution_policy,
            }
        )
        result_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(record)
        if not args.no_summary:
            summary_path = output_root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "strategy_id": args.strategy_id,
                        "strategy": args.strategy,
                        "scene_root": str(scene_root),
                        "seal_root": str(seal_root),
                        "binding_mode": args.binding_mode,
                        "checkpoint_sha256": checkpoint_digest,
                        "execution_policy": args.execution_policy,
                        "optimization_budget": {
                            "mode": "maximum",
                            "early_stopping": "primitive_policy",
                        },
                        "cases": records,
                        "summary": _summary(records),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    if args.allow_program_failure:
        return 0 if all(item["binding_valid"] for item in records) else 1
    return 0 if all(
        item.get("benchmark_success")
        if isinstance(item.get("benchmark_success"), bool)
        else item.get("program_success")
        for item in records
    ) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_suite", help="Frozen learned suite run or workspace")
    parser.add_argument("--strategy-id", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help=(
            "Directory containing case_NNN_scene packages. Defaults to "
            "tasks/sim_benchmark/strategy_ID."
        ),
    )
    parser.add_argument(
        "--seal-root",
        type=Path,
        default=None,
        help=(
            "Private seal/manifests root. Defaults to the production sealed "
            "benchmark; candidate scene sets should pass their isolated seal root."
        ),
    )
    parser.add_argument(
        "--binding-mode",
        choices=("authoritative", "agent"),
        default="authoritative",
        help=(
            "Use the hidden remapped demo binding (default), or separately "
            "measure image/language semantic binding with the deployment agent."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Case number, comma list, or range; defaults to 1-50",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workspace-mode",
        choices=("isolated", "shared", "direct"),
        default="isolated",
        help=(
            "Use one fresh workspace per case (default), or reuse one frozen "
            "workspace while archiving and removing all case-local state. "
            "Direct mode uses the supplied suite workspace itself and is meant "
            "for a disposable eval-suite copy."
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
        "--execution-policy",
        choices=EXECUTION_POLICIES,
        default=DEFAULT_EXECUTION_POLICY,
        help=(
            "Propagate finite rejected transit trajectories in benchmark attempt "
            "mode (default), or strictly stop on execution rejection."
        ),
    )
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Write case results only; a parallel parent will aggregate summary.json.",
    )
    parser.add_argument("--allow-program-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args, root=repo_root_from_env().resolve())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[rapid-benchmark] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
