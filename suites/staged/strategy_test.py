"""Bind and test one learned strategy inside an existing suite workspace."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from suites.staged.experiment_paths import (
    add_experiment_root_argument,
    configure_experiment_root,
    experiment_root,
)

from .codex_permissions import PROFILE_NAME, run_codex_home_path
from .gpu_render_broker import (
    GPU_RENDER_MODE_ENV,
    GPU_RENDER_QUEUE_ENV,
    GpuRenderBrokerError,
    start_gpu_render_broker,
)
from .workspaces import repo_root_from_env


_SAFE_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class StrategyTestTarget:
    """Resolved paths and names for one strategy generalization test."""

    run_dir: Path
    workspace: Path
    src: Path
    strategy: str
    scene_module: str
    instance: str

    @property
    def module(self) -> str:
        return (
            f"instantiations.strategies.{self.strategy}."
            f"{self.instance}.main"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-strategy-test",
        description=(
            "Bind an existing strategy in a suite-run workspace to a new "
            "task scene, then execute and evaluate the generated binding."
        ),
    )
    parser.add_argument(
        "suite_run",
        help=(
            "Suite-run ID, run directory, or workspace directory. A short ID "
            "expands to EXPERIMENT_ROOT/ID."
        ),
    )
    parser.add_argument(
        "test_case",
        help=(
            "Task scene module or task-package path, for example "
            "tasks.invention_real2sim.new_scene."
        ),
    )
    parser.add_argument(
        "description",
        help="Natural-language goal/description passed to the binding agent.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy package name. Inferred when exactly one exists.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Instantiation name. Defaults to the task package name.",
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
        "--optimization-budget-mode",
        choices=("reference", "maximum"),
        default="reference",
        help="Select primitive reference budgets or the global maximum budget.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Final test metrics path, relative to the workspace by default.",
    )
    parser.add_argument(
        "--execution-log",
        type=Path,
        help="Retain execution stdout/stderr at this workspace-relative path.",
    )
    parser.add_argument(
        "--state-trajectory-npz",
        type=Path,
        default=None,
        help=(
            "Retain achieved nq+nv states for benchmark-owned evaluation; "
            "relative paths are resolved under the workspace."
        ),
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Final recording path, relative to the workspace by default.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not record the final strategy test.",
    )
    parser.add_argument(
        "--allow-program-failure",
        action="store_true",
        help=(
            "Return success when binding_valid is true even if the final "
            "program_success predicate is false."
        ),
    )
    parser.add_argument(
        "--sandbox-execution",
        action="store_true",
        help=(
            "Execute frozen strategy code under the workspace "
            "read-isolation profile."
        ),
    )
    add_experiment_root_argument(parser)
    return parser


def resolve_suite_workspace(
    repo_root: Path,
    value: str,
) -> tuple[Path, Path]:
    """Resolve a short run ID, run path, or direct workspace path."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        if len(candidate.parts) == 1:
            candidate = experiment_root(repo_root) / candidate
        else:
            candidate = repo_root / candidate
    candidate = candidate.resolve()
    if (candidate / "workspace" / "src").is_dir():
        return candidate, candidate / "workspace"
    if (candidate / "src").is_dir():
        return candidate.parent, candidate
    raise ValueError(
        f"could not resolve suite-run workspace from {value!r}; expected "
        f"{candidate / 'workspace' / 'src'} or {candidate / 'src'}"
    )


def normalize_test_case(repo_root: Path, value: str) -> str:
    """Resolve a task module or repository task-package path to ``.scene``."""

    text = value.strip().removesuffix("/")
    if not text:
        raise ValueError("test case cannot be empty")

    if text.startswith("tasks."):
        module = text.removesuffix(".scene")
        package = repo_root / module.replace(".", "/")
    else:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if candidate.name == "scene.py":
            candidate = candidate.parent
        try:
            relative = candidate.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"test case must live under {repo_root / 'tasks'}"
            ) from exc
        if not relative.parts or relative.parts[0] != "tasks":
            raise ValueError(
                f"test case must live under {repo_root / 'tasks'}"
            )
        module = ".".join(relative.parts)
        package = candidate

    if not all(part.isidentifier() for part in module.split(".")):
        raise ValueError(f"test case is not an importable module: {module!r}")
    if not package.is_dir() or not (package / "scene.py").is_file():
        raise ValueError(
            f"test case has no scene.py: {package / 'scene.py'}"
        )
    return f"{module}.scene"


def resolve_strategy(src: Path, requested: str | None) -> str:
    """Resolve one strategy package and reject legacy contract-less packages."""

    root = src / "strategies"
    candidates = (
        sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir()
            and not child.name.startswith((".", "_"))
            and (child / "__init__.py").is_file()
        )
        if root.is_dir()
        else []
    )
    if requested is None:
        if len(candidates) != 1:
            detail = ", ".join(candidates) or "(none)"
            raise ValueError(
                "could not infer one strategy; pass --strategy. "
                f"Candidates: {detail}"
            )
        strategy = candidates[0]
    else:
        strategy = requested
    if not _SAFE_PACKAGE.fullmatch(strategy):
        raise ValueError(f"invalid strategy package name: {strategy!r}")
    package = root / strategy
    if not package.is_dir():
        raise ValueError(f"strategy package does not exist: {package}")
    missing = [
        filename
        for filename in (
            "__init__.py",
            "binding_contract.py",
            "program.py",
            "predicate.py",
            "main.py",
        )
        if not (package / filename).is_file()
    ]
    if missing:
        raise ValueError(
            f"strategy {strategy!r} is not declarative-binding compatible; "
            f"missing {', '.join(missing)}"
        )
    return strategy


def resolve_target(
    repo_root: Path,
    *,
    suite_run: str,
    test_case: str,
    strategy: str | None,
    instance: str | None,
) -> StrategyTestTarget:
    run_dir, workspace = resolve_suite_workspace(repo_root, suite_run)
    src = workspace / "src"
    strategy_name = resolve_strategy(src, strategy)
    scene_module = normalize_test_case(repo_root, test_case)
    instance_name = instance or scene_module.removesuffix(".scene").split(".")[-1]
    if not _SAFE_PACKAGE.fullmatch(instance_name):
        raise ValueError(f"invalid instantiation name: {instance_name!r}")
    return StrategyTestTarget(
        run_dir=run_dir,
        workspace=workspace,
        src=src,
        strategy=strategy_name,
        scene_module=scene_module,
        instance=instance_name,
    )


def _workspace_output(workspace: Path, value: Path | None, default: str) -> Path:
    output = value or Path(default)
    if not output.is_absolute():
        output = workspace / output
    resolved = output.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            f"strategy-test output must stay under {workspace}: {resolved}"
        ) from exc
    return resolved


def _status(payload: dict[str, Any]) -> tuple[object, object]:
    binding_valid = payload.get("binding_valid")
    program_success = payload.get("program_success")
    binding = payload.get("binding")
    if isinstance(binding, dict):
        if binding_valid is None:
            binding_valid = binding.get("binding_valid")
        if program_success is None:
            program_success = binding.get("program_success")
    return binding_valid, program_success


def _run(
    command: Sequence[str], *, cwd: Path, env: dict[str, str],
    log_path: Path | None = None,
) -> int:
    print(f"[rapid-strategy-test] $ {shlex.join(command)}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[rapid-strategy-test] execution log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(command), cwd=cwd, env=env, check=False,
                stdout=log, stderr=subprocess.STDOUT,
            )
        return int(completed.returncode)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def run_strategy_test(args: argparse.Namespace, *, repo_root: Path) -> int:
    target = resolve_target(
        repo_root,
        suite_run=args.suite_run,
        test_case=args.test_case,
        strategy=args.strategy,
        instance=args.name,
    )
    description = args.description.strip()
    if not description:
        raise ValueError("language description cannot be empty")
    if args.no_record and args.record is not None:
        raise ValueError("--record and --no-record cannot be used together")
    print(f"[rapid-strategy-test] workspace: {target.workspace}")
    print(f"[rapid-strategy-test] strategy: {target.strategy}")
    print(f"[rapid-strategy-test] scene: {target.scene_module}")
    print(f"[rapid-strategy-test] instance: {target.instance}")

    bind_command = [
        str(repo_root / "scripts" / "rapid-codex-run"),
        "--workspace",
        str(target.workspace),
        "--",
        "agent/deploy_strategy_exec.py",
        "--strategy",
        target.strategy,
        "--env-module",
        target.scene_module,
        "--name",
        target.instance,
        "--goal",
        description,
        "--bind-only",
    ]
    if args.model:
        bind_command.extend(["--model", args.model])
    if args.thinking_effort:
        bind_command.extend(["--thinking-effort", args.thinking_effort])
    if args.max_turns is not None:
        bind_command.extend(["--max-turns", str(args.max_turns)])

    base_env = os.environ.copy()
    base_env["RAPID_REPO_ROOT"] = str(repo_root)
    if _run(bind_command, cwd=repo_root, env=base_env) != 0:
        print("[rapid-strategy-test] binding agent failed", file=sys.stderr)
        return 1

    result_root = (
        Path("results_instantiation")
        / "strategies"
        / target.strategy
        / target.instance
    )
    metrics_path = _workspace_output(
        target.workspace,
        args.metrics_json,
        str(result_root / "metrics.json"),
    )
    state_trajectory_path = (
        _workspace_output(
            target.workspace,
            args.state_trajectory_npz,
            str(result_root / "state_trajectory.npz"),
        )
        if args.state_trajectory_npz is not None
        else None
    )
    record_path = (
        None
        if args.no_record
        else _workspace_output(
            target.workspace,
            args.record,
            str(result_root / "trajectory.mp4"),
        )
    )
    test_command = [
        str(repo_root / "scripts" / "rapid-python"),
        "-m",
        target.module,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--metrics-json",
        str(metrics_path),
    ]
    if state_trajectory_path is not None:
        test_command.extend(
            ["--state-trajectory-npz", str(state_trajectory_path)]
        )
    if record_path is not None:
        test_command.extend(["--record", str(record_path)])
    else:
        test_command.append("--no-record")

    test_env = base_env.copy()
    test_env.update(
        {
            "RAPID_WORKSPACE": str(target.workspace),
            "RAPID_ACTIVE_SRC_DIR": str(target.src),
            "RAPID_SRC_DIR": str(target.src),
        }
    )
    test_env["RAPID_OPTIMIZATION_BUDGET_MODE"] = args.optimization_budget_mode
    if args.sandbox_execution:
        codex = shutil.which("codex")
        if codex is None:
            print("[rapid-strategy-test] codex CLI was not found", file=sys.stderr)
            return 1
        test_command = [
            codex,
            "sandbox",
            "-C",
            str(target.workspace),
            "-P",
            PROFILE_NAME,
            "--",
            *test_command,
        ]
        test_env["HOME"] = str(target.workspace / "agent_work" / "home")
        test_env["CODEX_HOME"] = str(run_codex_home_path(target.workspace))
    broker = None
    try:
        try:
            broker = start_gpu_render_broker(
                workspace=target.workspace,
                env=test_env,
                mode=test_env.get(GPU_RENDER_MODE_ENV, "auto"),
            )
        except GpuRenderBrokerError as exc:
            print(f"[rapid-strategy-test] GPU rendering failed: {exc}", file=sys.stderr)
            return 1
        if broker is not None:
            test_env[GPU_RENDER_QUEUE_ENV] = str(broker.queue_path)
        else:
            test_env.pop(GPU_RENDER_QUEUE_ENV, None)
        log_options = {}
        if args.execution_log is not None:
            log_options["log_path"] = _workspace_output(
                target.workspace, args.execution_log, str(result_root / "execution.log"),
            )
        returncode = _run(test_command, cwd=target.workspace, env=test_env, **log_options)
    finally:
        if broker is not None:
            broker.stop()
    if returncode != 0:
        print(
            f"[rapid-strategy-test] strategy execution failed with exit "
            f"{returncode}",
            file=sys.stderr,
        )
        return returncode
    if not metrics_path.is_file():
        print(
            f"[rapid-strategy-test] metrics file was not written: {metrics_path}",
            file=sys.stderr,
        )
        return 1
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[rapid-strategy-test] could not read metrics: {exc}",
            file=sys.stderr,
        )
        return 1

    binding_valid, program_success = _status(payload)
    print(f"[rapid-strategy-test] binding_valid: {binding_valid}")
    print(f"[rapid-strategy-test] program_success: {program_success}")
    print(f"[rapid-strategy-test] metrics: {metrics_path}")
    if state_trajectory_path is not None:
        print(
            "[rapid-strategy-test] state trajectory: "
            f"{state_trajectory_path}"
        )
    if record_path is not None:
        print(f"[rapid-strategy-test] recording: {record_path}")
    if binding_valid is not True:
        return 1
    if not isinstance(program_success, bool):
        print(
            "[rapid-strategy-test] program_success is missing or not boolean",
            file=sys.stderr,
        )
        return 1
    if program_success is not True and not args.allow_program_failure:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_experiment_root(args, repo_root_from_env().resolve())
    try:
        return run_strategy_test(args, repo_root=repo_root_from_env().resolve())
    except ValueError as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
