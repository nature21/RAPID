"""Permission-profile based Codex launcher for RAPID suite workspaces."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .codex_permissions import (
    PROFILE_NAME,
    codex_arg0_helper_read_grants,
    host_tool_read_grants,
    inspect_conda_root,
    path_exposes_blocked_checkpoint,
    run_codex_home_path,
    safe_pythonpath_read_grants,
    write_codex_home_bootstrap,
    write_workspace_codex_config,
)
from .gpu_render_broker import (
    GPU_RENDER_MODE_ENV,
    GPU_RENDER_MODES,
    GPU_RENDER_QUEUE_ENV,
    GpuRenderBrokerError,
    parse_gpu_render_mode,
    start_gpu_render_broker,
)
from .workspaces import repo_root_from_env


AGENT_ENTRYPOINT_MODULES = {
    "deploy_strategy_exec.py": "agent.deploy_strategy_exec",
    "construct_primitive_exec.py": "agent.construct_primitive_exec",
    "compose_strategy_exec.py": "agent.compose_strategy_exec",
    "instantiate_primitive_exec.py": "agent.instantiate_primitive_exec",
    "instantiate_strategy_exec.py": "agent.instantiate_strategy_exec",
}


def _die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _quote_cmd(args: Sequence[object]) -> str:
    return " ".join(shlex_quote(str(arg)) for arg in args)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _abs_existing_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.exists():
        _die(f"path does not exist: {candidate}")
    return candidate.resolve()


def _path_is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return child == parent
    return True


def _workspace_tag(workspace: Path) -> str:
    raw = workspace.parent.name
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in raw)
    return safe or "workspace"


def _ensure_task_input_parent(workspace: Path, rel: Path) -> Path:
    task_inputs = workspace / "task_inputs"
    if task_inputs.is_symlink():
        _die(f"workspace task_inputs must not be a symlink: {task_inputs}")
    current = task_inputs
    current.mkdir(parents=True, exist_ok=True)
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink():
            _die(f"workspace task_inputs parent must not be a symlink: {current}")
        current.mkdir(exist_ok=True)

    task_inputs_real = task_inputs.resolve()
    parent_real = current.resolve()
    if not _path_is_under(parent_real, task_inputs_real):
        _die(f"task input destination escapes workspace/task_inputs: {current}")
    return current


def _copy_task_bind(path: str | Path, *, repo_root: Path, workspace: Path) -> Path:
    resolved = _abs_existing_path(path, repo_root=repo_root)
    tasks_root = (repo_root / "tasks").resolve()
    if not _path_is_under(resolved, tasks_root):
        _die(f"--task-bind must point under {tasks_root}: {resolved}")
    if not resolved.is_dir():
        _die(f"--task-bind must point to a task directory: {resolved}")

    rel = resolved.relative_to(repo_root)
    dest = workspace / "task_inputs" / rel
    dest_parent = _ensure_task_input_parent(workspace, rel)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.copy.", dir=dest_parent))
    try:
        shutil.copytree(resolved, tmp, dirs_exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _copy_task_package_helpers(resolved, repo_root=repo_root, workspace=workspace)
    return dest


def _copy_task_package_helpers(task_root: Path, *, repo_root: Path, workspace: Path) -> None:
    """Copy direct ancestor package .py files needed by a selected task package."""
    tasks_root = (repo_root / "tasks").resolve()
    helper_dirs: list[Path] = []
    current = task_root.parent
    while _path_is_under(current, tasks_root) and current != tasks_root.parent:
        helper_dirs.append(current)
        if current == tasks_root:
            break
        current = current.parent

    for helper_dir in reversed(helper_dirs):
        for source in sorted(helper_dir.glob("*.py")):
            rel = source.resolve().relative_to(repo_root)
            destination_parent = _ensure_task_input_parent(workspace, rel)
            destination = destination_parent / source.name
            if destination.is_symlink():
                _die(f"task input helper destination must not be a symlink: {destination}")
            shutil.copy2(source, destination)


def _sync_workspace_runtime_file(
    relative_path: Path,
    *,
    repo_root: Path,
    src: Path,
) -> bool:
    """Refresh one canonical shared-runtime file in a reused workspace."""

    source = repo_root / "workspace/src" / relative_path
    if not source.is_file():
        _die(f"canonical workspace runtime file is missing: {source}")
    destination = src / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _die(f"workspace runtime file must not be a symlink: {destination}")
    if not _path_is_under(destination.parent.resolve(), src):
        _die(f"workspace runtime file escapes the source tree: {destination}")
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        return False
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=".record.py.refresh.",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_raw)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _sync_gpu_render_client(*, repo_root: Path, src: Path) -> bool:
    """Refresh shared recording infrastructure in a reused suite workspace."""

    return _sync_workspace_runtime_file(
        Path("scene_io/record.py"),
        repo_root=repo_root,
        src=src,
    )


def _sync_instantiation_runtime(*, repo_root: Path, src: Path) -> bool:
    """Refresh shared execution and retained-artifact support."""

    changed = False
    for relative_path in (
        Path("instantiation_core/entrypoint.py"),
        Path("instantiation_core/runner.py"),
        Path("scene_io/record.py"),
        Path("strategies/runtime.py"),
    ):
        refreshed = _sync_workspace_runtime_file(
            relative_path,
            repo_root=repo_root,
            src=src,
        )
        changed = refreshed or changed
    return changed


def _env_module_to_task_root(module: str, *, repo_root: Path) -> str:
    if not module.startswith("tasks."):
        _die(f"--env-module must start with tasks.: {module}")
    root = module.removesuffix(".scene").replace(".", "/")
    resolved = repo_root / root
    if not resolved.is_dir():
        _die(
            f"--env-module does not map to an existing task package: {module} "
            f"(expected {root}/scene.py)"
        )
    if not (resolved / "scene.py").is_file():
        _die(f"--env-module task package has no scene.py: {module} (expected {root}/scene.py)")
    return root


def _add_task_bind_candidate(
    task_binds: list[str],
    path: str,
    *,
    repo_root: Path,
) -> None:
    candidate = repo_root / path
    if not candidate.is_dir():
        return
    candidate_abs = candidate.resolve()
    for existing in task_binds:
        existing_path = Path(existing)
        if not existing_path.is_absolute():
            existing_path = repo_root / existing_path
        if existing_path.is_dir() and _path_is_under(candidate_abs, existing_path.resolve()):
            return
    task_binds.append(path)


def _add_env_module_task_bind(
    task_binds: list[str],
    module: str,
    *,
    repo_root: Path,
) -> None:
    root = _env_module_to_task_root(module, repo_root=repo_root)
    _add_task_bind_candidate(task_binds, root, repo_root=repo_root)


def _add_env_module_task_binds(
    task_binds: list[str],
    agent_args: Sequence[str],
    *,
    repo_root: Path,
) -> None:
    expecting = False
    for arg in agent_args:
        if expecting:
            if arg.startswith("--"):
                expecting = False
            else:
                _add_env_module_task_bind(task_binds, arg, repo_root=repo_root)
                continue

        if arg in {"--env-module", "--env-modules"}:
            expecting = True
        elif arg.startswith("--env-module=") or arg.startswith("--env-modules="):
            _add_env_module_task_bind(
                task_binds,
                arg.split("=", 1)[1],
                repo_root=repo_root,
            )


def _candidate_demo_args(agent_args: Sequence[str]) -> list[str]:
    """Return positional demo arguments from an RAPID invent launcher command.

    The launcher command shape is normally:

        agent/construct_primitive_exec.py DEMO... --option value ...

    Only the positional arguments before the first option are demos. Values for
    options such as ``--description`` must not be treated as path candidates.
    """
    args = list(agent_args)
    if not args:
        return []

    start = 0
    if args[0] == "-m" and len(args) >= 2:
        start = 2
    elif args[0].endswith(".py") or "/" in args[0]:
        start = 1

    demos: list[str] = []
    for raw in args[start:]:
        if raw.startswith("-"):
            break
        demos.append(raw)
    return demos


def _selected_demo_read_grants(agent_args: Sequence[str], *, repo_root: Path) -> list[Path]:
    grants: list[Path] = []
    demos_root = repo_root / "demos"
    for raw in _candidate_demo_args(agent_args):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if not exists:
            continue
        resolved = path.resolve()
        if resolved.is_file() and _path_is_under(resolved, demos_root.resolve()):
            grants.append(resolved)
    return grants


def _option_value(agent_args: Sequence[str], option: str) -> str | None:
    for index, argument in enumerate(agent_args):
        if argument == option and index + 1 < len(agent_args):
            return agent_args[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def _instantiation_workspace_grants(
    agent_args: Sequence[str],
    *,
    src: Path,
) -> tuple[list[Path] | None, list[Path]]:
    """Return directory write grants and canonical read-only overrides.

    Codex's Linux sandbox applies workspace safeguards beneath every writable
    subtree. Granting the manifest file itself therefore makes the sandbox
    probe ``binding.json/.git`` and ``binding.json/.codex`` as directories.
    Grant the containing package instead and protect its canonical Python
    package initializer with a more-specific read rule. ``main.py`` retains a
    validated execution-only defaults mapping. All learned and shared source
    remains covered by the workspace's read-only root grant.
    """

    invocation = " ".join(agent_args[:2])
    deploy_only = False
    if "instantiate_primitive_exec" in invocation:
        kind = "primitives"
        program = _option_value(agent_args, "--primitive")
    elif "instantiate_strategy_exec" in invocation:
        kind = "strategies"
        program = _option_value(agent_args, "--strategy")
    elif "deploy_strategy_exec" in invocation:
        kind = "strategies"
        program = _option_value(agent_args, "--strategy")
        deploy_only = True
    else:
        return None, []
    instance = _option_value(agent_args, "--name")
    safe_name = re.compile(r"^[a-z][a-z0-9_]*$")
    if program is None or not safe_name.fullmatch(program):
        _die(
            f"invalid or missing instantiation program package name: {program!r}"
        )
    if instance is None or not safe_name.fullmatch(instance):
        _die(
            f"invalid or missing instantiation instance package name: {instance!r}"
        )
    if deploy_only:
        # The deployment model returns JSON through its final message. The host
        # launcher, not Codex, writes the canonical binding package afterward.
        return [], []
    target = (
        src
        / "instantiations"
        / kind
        / program
        / instance
    )
    return [target], [target / "__init__.py"]


def _safe_python_agent_args(
    agent_args: Sequence[str],
    *,
    repo_root: Path,
) -> list[str]:
    """Use safe-path module execution for RAPID's supported agent entry points."""

    args = list(agent_args)
    if not args:
        return ["-P"]
    candidate = Path(args[0]).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        relative = candidate.resolve().relative_to(repo_root)
    except (OSError, ValueError):
        relative = None
    module = (
        AGENT_ENTRYPOINT_MODULES.get(relative.name)
        if relative is not None and relative.parent == Path("agent")
        else None
    )
    if module is not None:
        return ["-P", "-m", module, *args[1:]]
    if args[0] == "-m":
        return ["-P", *args]
    return ["-P", *args]


@dataclass(frozen=True)
class PreparedLaunch:
    workspace: Path
    src: Path
    codex_home: Path
    agent_home: Path
    workspace_config: Path
    codex_home_config: Path
    command: list[str]
    env: dict[str, str]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rapid-codex-run",
        description="Launch an RAPID Codex agent using workspace-local permission profiles.",
    )
    parser.add_argument("--workspace", required=True, help="Candidate workspace to expose writable.")
    parser.add_argument("--src", default=None, help="Candidate source tree. Defaults to WORKSPACE/src.")
    parser.add_argument("--task-bind", action="append", default=[], help="Task root under tasks/ to copy into task_inputs/.")
    parser.add_argument("--ro-bind", action="append", default=[], help="Additional path to grant read-only access.")
    parser.add_argument("--no-codex-auth", action="store_true", help="Do not copy host CODEX_HOME/auth.json.")
    parser.add_argument(
        "--codex-home-root",
        default=os.environ.get("RAPID_CODEX_HOME_ROOT"),
        help=(
            "Optional external directory for workspace-specific CODEX_HOME "
            "directories. Defaults to RUN_DIR/codex_home."
        ),
    )
    parser.add_argument("--probe-permissions", action="store_true", help="Run a Codex permission probe.")
    parser.add_argument("--dry-run", action="store_true", help="Print setup paths and command without executing.")
    parser.add_argument(
        "--gpu-render",
        choices=GPU_RENDER_MODES,
        default=os.environ.get(GPU_RENDER_MODE_ENV, "auto"),
        help=(
            "Host NVIDIA/EGL render broker policy: auto uses it when available, "
            "required fails before agent launch if unavailable, and off disables it "
            f"(default: ${GPU_RENDER_MODE_ENV} or auto)."
        ),
    )
    parser.add_argument("agent_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.agent_args and args.agent_args[0] == "--":
        args.agent_args = args.agent_args[1:]
    if not args.agent_args and not args.probe_permissions:
        parser.error("pass agent arguments after --, or use --probe-permissions")
    try:
        args.gpu_render = parse_gpu_render_mode(args.gpu_render)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _fresh_codex_home(
    workspace: Path,
    *,
    repo_root: Path,
    codex_home_root: str | None,
    shared_codex_tmp: Path | None = None,
) -> tuple[Path, Path]:
    tag = _workspace_tag(workspace)
    agent_home = workspace / "agent_work" / "home"
    agent_home.mkdir(parents=True, exist_ok=True)

    if codex_home_root:
        root = Path(codex_home_root).expanduser()
        if not root.is_absolute():
            root = repo_root / root
        root.mkdir(parents=True, exist_ok=True)
        codex_home = root.resolve() / f"{tag}_codex"
    else:
        codex_home = run_codex_home_path(workspace)

    codex_home.mkdir(parents=True, exist_ok=True)
    _link_shared_codex_tmp(codex_home, shared_codex_tmp)
    (codex_home / "tmp" / "arg0").mkdir(parents=True, exist_ok=True)
    return agent_home, codex_home


def _link_shared_codex_tmp(codex_home: Path, shared_codex_tmp: Path | None) -> bool:
    """Link a fresh run home to the host's curated-plugin cache directory.

    Codex keeps session transcripts, authentication, and SQLite state directly
    under ``CODEX_HOME``; those remain run-local. The ``.tmp`` directory holds
    the curated plugin checkout plus its sync lock and checksum, so linking the
    whole directory keeps Codex's cache synchronization atomic across runs.

    Existing directories are deliberately left alone. This avoids deleting a
    historical run's data when a run is resumed after upgrading RAPID.
    """
    if shared_codex_tmp is None:
        return False

    target = shared_codex_tmp.expanduser().resolve()
    link = codex_home / ".tmp"
    # Avoid trying to symlink a cache to itself when the caller already points
    # CODEX_HOME at this run (a common setup for ``rapid-resume-workspace``).
    if link.resolve() == target:
        return False
    if link.is_symlink():
        return link.resolve() == target
    if link.exists():
        return False

    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return True


def prepare_launch(args: argparse.Namespace) -> PreparedLaunch:
    repo_root = repo_root_from_env().resolve()
    workspace = _abs_existing_path(args.workspace, repo_root=repo_root)
    if not workspace.is_dir():
        _die(f"workspace is not a directory: {workspace}")
    src = _abs_existing_path(args.src or (workspace / "src"), repo_root=repo_root)
    if not src.is_dir():
        _die(f"source tree is not a directory: {src}")
    if not _path_is_under(src, workspace):
        _die("--src must live inside --workspace")
    if workspace == (repo_root / "workspace").resolve():
        _die("refusing to launch against the canonical workspace; pass an isolated suite workspace")

    _sync_instantiation_runtime(repo_root=repo_root, src=src)

    for path in (
        workspace / "agent_work",
        workspace / "agent_work" / "tmp",
        workspace / "results",
        workspace / "results_instantiation",
    ):
        path.mkdir(parents=True, exist_ok=True)

    host_codex_home = None
    if os.environ.get("CODEX_HOME"):
        host_codex_home = Path(os.environ["CODEX_HOME"]).expanduser()
    elif os.environ.get("HOME"):
        host_codex_home = Path(os.environ["HOME"]).expanduser() / ".codex"
    if host_codex_home is not None:
        if not host_codex_home.is_absolute():
            host_codex_home = repo_root / host_codex_home
        host_codex_home = host_codex_home.resolve()

    agent_home, codex_home = _fresh_codex_home(
        workspace,
        repo_root=repo_root,
        codex_home_root=args.codex_home_root,
        shared_codex_tmp=(host_codex_home / ".tmp")
        if host_codex_home is not None
        else None,
    )

    task_binds = list(args.task_bind)
    _add_env_module_task_binds(task_binds, args.agent_args, repo_root=repo_root)
    for task_bind in task_binds:
        _copy_task_bind(task_bind, repo_root=repo_root, workspace=workspace)

    conda_root = inspect_conda_root()
    selected_demo_grants = _selected_demo_read_grants(args.agent_args, repo_root=repo_root)
    extra_read_grants: list[Path] = []
    for raw in args.ro_bind:
        path = _abs_existing_path(raw, repo_root=repo_root)
        if path_exposes_blocked_checkpoint(path, repo_root=repo_root, workspace=workspace):
            _die(f"--ro-bind would expose non-candidate RAPID checkpoints: {path}")
        extra_read_grants.append(path)
    extra_read_grants.extend(
        safe_pythonpath_read_grants(
            os.environ.get("PYTHONPATH", ""),
            repo_root=repo_root,
            workspace=workspace,
        )
    )
    extra_read_grants.extend(host_tool_read_grants(("ffmpeg", "ffprobe"), conda_root=conda_root))
    if codex_home != run_codex_home_path(workspace):
        extra_read_grants.extend(codex_arg0_helper_read_grants(codex_home))
    (
        workspace_write_grants,
        workspace_read_overrides,
    ) = _instantiation_workspace_grants(args.agent_args, src=src)
    workspace_config = write_workspace_codex_config(
        workspace,
        repo_root=repo_root,
        read_grants=extra_read_grants,
        allowed_blocked_read_grants=selected_demo_grants,
        workspace_write_grants=workspace_write_grants,
        workspace_read_overrides=workspace_read_overrides,
        conda_root=conda_root,
    )

    codex_home_config = write_codex_home_bootstrap(
        codex_home,
        workspace=workspace,
        host_config=(host_codex_home / "config.toml") if host_codex_home else None,
    )

    if not args.no_codex_auth and host_codex_home and (host_codex_home / "auth.json").is_file():
        source_auth = host_codex_home / "auth.json"
        dest_auth = codex_home / "auth.json"
        if source_auth.resolve() != dest_auth.resolve():
            shutil.copy2(source_auth, dest_auth)

    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        if not env.get(variable):
            env[variable] = "1"
    env.update(
        {
            "HOME": str(agent_home),
            "CODEX_HOME": str(codex_home),
            "CODEX_SQLITE_HOME": str(codex_home),
            "TMPDIR": str(workspace / "agent_work" / "tmp"),
            "XDG_CACHE_HOME": str(workspace / "agent_work" / "tmp/rapid_cache"),
            "MPLCONFIGDIR": str(
                workspace / "agent_work" / "tmp/rapid_cache/matplotlib"
            ),
            "RAPID_REPO_ROOT": str(repo_root),
            "RAPID_WORKSPACE": str(workspace),
            "RAPID_ACTIVE_SRC_DIR": str(src),
            "RAPID_SRC_DIR": str(src),
            "RAPID_CODEX_PERMISSION_PROFILE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)

    command = [
        str(repo_root / "scripts" / "rapid-python"),
        *_safe_python_agent_args(args.agent_args, repo_root=repo_root),
    ]
    return PreparedLaunch(
        workspace=workspace,
        src=src,
        codex_home=codex_home,
        agent_home=agent_home,
        workspace_config=workspace_config,
        codex_home_config=codex_home_config,
        command=command,
        env=env,
    )


def _probe_command(prepared: PreparedLaunch) -> list[str]:
    script = "\n".join(
        [
            "set -eu",
            "test -w .",
            "test -r task_inputs 2>/dev/null || true",
            "! test -w task_inputs 2>/dev/null",
            "test -x \"$RAPID_REPO_ROOT/scripts/rapid-python\"",
            "test -r \"$RAPID_REPO_ROOT/runtime_assets/fr3/fr3_task_v1/robot.xml\"",
            "! test -w \"$RAPID_REPO_ROOT/runtime_assets/fr3/fr3_task_v1/robot.xml\"",
            "! test -e \"$RAPID_REPO_ROOT/workspace/src/primitives\"",
            "! test -e \"$RAPID_REPO_ROOT/tasks\"",
            "! test -e \"$RAPID_REPO_ROOT/suite_runs\"",
            "! test -e \"$HOME/.codex\"",
            "test -r \"$CODEX_HOME/tmp/arg0\"",
            "! test -w \"$CODEX_HOME/tmp/arg0\"",
            "! test -e \"$CODEX_HOME/config.toml\"",
            "! test -w /tmp",
            "test -w \"$TMPDIR\"",
            "echo rapid-permission-probe-ok",
        ]
    )
    return [
        "codex",
        "sandbox",
        "-C",
        str(prepared.workspace),
        "-P",
        PROFILE_NAME,
        "--",
        "sh",
        "-c",
        script,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    prepared = prepare_launch(args)

    print(f"[rapid-codex-run] workspace       : {prepared.workspace}")
    print(f"[rapid-codex-run] src             : {prepared.src}")
    print(f"[rapid-codex-run] workspace config: {prepared.workspace_config}")
    print(f"[rapid-codex-run] CODEX_HOME      : {prepared.codex_home}")
    print(f"[rapid-codex-run] bootstrap config: {prepared.codex_home_config}")
    print(f"[rapid-codex-run] GPU rendering   : {args.gpu_render}")

    if args.probe_permissions:
        command = _probe_command(prepared)
        if args.dry_run:
            print("[rapid-codex-run] probe command:")
            print(_quote_cmd(command))
            return 0
        return subprocess.run(command, cwd=prepared.workspace, env=prepared.env, check=False).returncode

    if args.dry_run:
        print("[rapid-codex-run] launch command:")
        print(_quote_cmd(prepared.command))
        return 0

    launch_env = prepared.env.copy()
    launch_env[GPU_RENDER_MODE_ENV] = args.gpu_render
    broker = None
    try:
        try:
            broker = start_gpu_render_broker(
                workspace=prepared.workspace,
                env=launch_env,
                mode=args.gpu_render,
            )
        except GpuRenderBrokerError as exc:
            _die(f"GPU rendering is required but unavailable: {exc}")
        if broker is not None:
            _sync_gpu_render_client(
                repo_root=repo_root_from_env().resolve(),
                src=prepared.src,
            )
            launch_env[GPU_RENDER_QUEUE_ENV] = str(broker.queue_path)
        else:
            launch_env.pop(GPU_RENDER_QUEUE_ENV, None)

        completed = subprocess.run(
            prepared.command,
            cwd=repo_root_from_env().resolve(),
            env=launch_env,
            check=False,
        )
        return int(completed.returncode)
    finally:
        if broker is not None:
            broker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
