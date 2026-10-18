"""Command-line utilities for staged RAPID suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from suites.staged.workspaces import (
    create_isolated_workspace,
    migrate_workspace_to_current_format,
    merge_workspaces,
    run_module_in_workspace,
)
from suites.staged.codex_permissions import run_codex_home_path
from suites.staged.runtime_integrity import verify_runtime_manifest


def _cmd_prepare_runtime(args: argparse.Namespace) -> int:
    workspace = Path(args.run_dir).expanduser().resolve() / "workspace"
    if workspace.exists():
        verify_runtime_manifest(workspace)
    else:
        create_isolated_workspace(Path(args.run_dir))
    print(workspace)
    return 0


def _cmd_create_workspace(args: argparse.Namespace) -> int:
    ws = create_isolated_workspace(
        Path(args.run_dir),
        from_workspace=Path(args.from_workspace) if args.from_workspace else None,
    )
    print(f"workspace: {ws.workspace}")
    print(f"src      : {ws.src}")
    print(f"codex cfg: {ws.workspace / '.codex' / 'config.toml'}")
    print(f"codex home cfg: {run_codex_home_path(ws.workspace) / 'config.toml'}")
    print()
    print("Use these environment variables for agent launchers:")
    print(f"export RAPID_WORKSPACE={ws.workspace}")
    print(f"export RAPID_ACTIVE_SRC_DIR={ws.src}")
    print(f"export RAPID_SRC_DIR={ws.src}")
    return 0


def _cmd_merge_workspaces(args: argparse.Namespace) -> int:
    ws = merge_workspaces(
        Path(args.run_dir),
        [Path(path) for path in args.source_workspace],
        base_workspace=Path(args.base_workspace) if args.base_workspace else None,
    )
    print(f"workspace  : {ws.workspace}")
    print(f"src        : {ws.src}")
    print(f"merge index: {ws.workspace / 'merge_index.yaml'}")
    print(f"codex cfg  : {ws.workspace / '.codex' / 'config.toml'}")
    print(f"codex home cfg: {run_codex_home_path(ws.workspace) / 'config.toml'}")
    print()
    print("Use these environment variables for agent launchers:")
    print(f"export RAPID_WORKSPACE={ws.workspace}")
    print(f"export RAPID_ACTIVE_SRC_DIR={ws.src}")
    print(f"export RAPID_SRC_DIR={ws.src}")
    return 0


def _cmd_migrate_workspace(args: argparse.Namespace) -> int:
    for workspace in args.workspace:
        try:
            record = migrate_workspace_to_current_format(Path(workspace))
        except FileNotFoundError as exc:
            print(f"workspace: {Path(workspace).expanduser().resolve()}")
            print(f"  skipped            : {exc}")
            continue
        print(f"workspace: {record['workspace']}")
        print(f"  copied_infra       : {', '.join(record['copied_infra']) or '(none)'}")
        print(
            "  migrated_test_scenes: "
            f"{', '.join(record['migrated_test_scenes']) or '(none)'}"
        )
        print(
            "  removed_scene_boiler: "
            f"{', '.join(record['removed_test_scene_boilerplate']) or '(none)'}"
        )
        print(
            "  removed_legacy_src : "
            f"{', '.join(record['removed_legacy_src']) or '(none)'}"
        )
        print(f"  updated_readme     : {record['updated_readme']}")
        print(f"  codex_config       : {record['codex_config']}")
        print(f"  codex_home_config  : {record['codex_home_config']}")
    return 0


def _cmd_run_module(args: argparse.Namespace) -> int:
    module_args = list(args.module_args)
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]
    return run_module_in_workspace(
        Path(args.workspace),
        args.module,
        module_args,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for primitive construction and strategy composition trials.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-runtime", help="Freeze or verify a batch runtime checkpoint.")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_prepare_runtime)

    p = sub.add_parser("create-workspace", help="Create an isolated candidate workspace.")
    p.add_argument("--run-dir", required=True, help="Run directory to create.")
    p.add_argument(
        "--from-workspace",
        default=None,
        help=(
            "Optional existing checkpoint to copy learned state from. "
            "Accepts either RUN or RUN/workspace."
        ),
    )
    p.set_defaults(func=_cmd_create_workspace)

    p = sub.add_parser("merge-workspaces", help="Merge learned workspace checkpoints.")
    p.add_argument("--run-dir", required=True, help="Run directory to create.")
    p.add_argument(
        "--base-workspace",
        default=None,
        help=(
            "Base checkpoint. Accepts either RUN or RUN/workspace. "
            "Defaults to the canonical workspace."
        ),
    )
    p.add_argument(
        "--source-workspace",
        action="append",
        required=True,
        help=(
            "Learned source checkpoint to merge. Accepts either RUN or "
            "RUN/workspace. Repeat for each source."
        ),
    )
    p.set_defaults(func=_cmd_merge_workspaces)

    p = sub.add_parser("migrate-workspace", help="Update existing workspaces to the current layout.")
    p.add_argument(
        "--workspace",
        action="append",
        required=True,
        help=(
            "Existing workspace checkpoint to migrate in place. Accepts either "
            "RUN or RUN/workspace. Repeat for each workspace."
        ),
    )
    p.set_defaults(func=_cmd_migrate_workspace)

    p = sub.add_parser("run-module", help="Run a module using a candidate workspace src tree.")
    p.add_argument("--workspace", required=True, help="Candidate workspace path.")
    p.add_argument(
        "--module",
        required=True,
        help="Module name, e.g. tasks.invention_real2sim.push_scene.scene.",
    )
    p.add_argument("module_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_run_module)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
