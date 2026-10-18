"""Workspace checkpoint helpers for staged suite runs.

The suite protocol uses an isolated workspace for each trial. A workspace
is the checkpoint: construction runs add learned primitives and validation scenes
to it, and composition runs start from either one learned workspace or a merged
workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .runtime_integrity import (
    require_shared_source,
    runtime_manifest_path,
    shared_source_hashes,
    verify_runtime_manifest,
    write_runtime_manifest,
)

from .codex_permissions import (
    run_codex_home_path,
    write_codex_home_bootstrap,
    write_workspace_codex_config,
)


IGNORE_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}

LEGACY_SRC_EXCLUDE_NAMES = {
    "benchmark_tasks",
    "benchmarks",
    "judo",
    "solutions",
    "test_scenes",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LEGACY_IMPORT_REWRITES = {
    "test_scenes._common": "scene_io",
    "benchmark_tasks.": "tasks.",
}

INVENT_WORKSPACE_TEMPLATE_REL = Path("agent/templates/invent_workspace")


def repo_root_from_env() -> Path:
    """Return the canonical project root."""
    env_root = os.environ.get("RAPID_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _host_codex_config(repo_root: Path) -> Path | None:
    if os.environ.get("CODEX_HOME"):
        host_codex_home = Path(os.environ["CODEX_HOME"]).expanduser()
    elif os.environ.get("HOME"):
        host_codex_home = Path(os.environ["HOME"]).expanduser() / ".codex"
    else:
        return None
    if not host_codex_home.is_absolute():
        host_codex_home = repo_root / host_codex_home
    return host_codex_home.resolve() / "config.toml"


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in IGNORE_NAMES}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _ignore_src(_directory: str, names: list[str]) -> set[str]:
    ignored = _ignore(_directory, names)
    ignored.update(name for name in names if name in LEGACY_SRC_EXCLUDE_NAMES)
    return ignored


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def _copy_src_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=_ignore_src, dirs_exist_ok=True)


def _copy_missing_tree(src: Path, dst: Path) -> bool:
    """Copy only absent files from a canonical shared-infrastructure tree."""

    copied = False
    for source in sorted(path for path in src.rglob("*") if path.is_file()):
        if any(part in IGNORE_NAMES for part in source.parts):
            continue
        destination = dst / source.relative_to(src)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied = True
    return copied


def _digest_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part in IGNORE_NAMES for part in file_path.parts):
            continue
        rel = file_path.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _same_tree(left: Path, right: Path) -> bool:
    return left.is_dir() and right.is_dir() and _digest_tree(left) == _digest_tree(right)


def _concrete_dirs(path: Path, *, allowed: set[str] | None = None) -> list[Path]:
    if not path.is_dir():
        return []
    allowed = allowed or set()
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir()
        and child.name not in allowed
        and child.name != "__pycache__"
        and not child.name.startswith(".")
    ]


def _primitive_rename_replacements(
    primitive: str,
    target_primitive: str,
) -> dict[str, str]:
    if target_primitive == primitive:
        return {}
    return {
        f"primitive_registry.{primitive}": (
            f"primitive_registry.{target_primitive}"
        ),
        f"primitives.{primitive}": f"primitives.{target_primitive}",
        (
            f"instantiations.primitives.{primitive}"
        ): f"instantiations.primitives.{target_primitive}",
        f"test_scenes.{primitive}": f"test_scenes.{target_primitive}",
    }


def _rewrite_text_tree(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file_path in files:
        if file_path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not replacements:
            continue
        pattern = re.compile(
            "|".join(
                re.escape(old)
                for old in sorted(replacements, key=len, reverse=True)
            )
        )
        new_text = pattern.sub(
            lambda match: replacements[match.group(0)],
            text,
        )
        if new_text != text:
            file_path.write_text(new_text, encoding="utf-8")


def _rewrite_binding_manifest_program(
    instantiations_root: Path,
    *,
    program_kind: str,
    source_name: str,
    target_name: str,
) -> None:
    """Update strict manifest identity when a merged program is renamed."""

    if source_name == target_name or not instantiations_root.is_dir():
        return
    for manifest_path in instantiations_root.rglob("binding.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("program_kind") != program_kind
            or payload.get("program_name") != source_name
        ):
            continue
        payload["program_name"] = target_name
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    return slug or "source"


def _source_id(workspace: Path, used: set[str]) -> str:
    raw = workspace.parent.name if workspace.name == "workspace" else workspace.name
    base = _safe_slug(raw)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def audit_zero_primitive_baseline(src: Path) -> list[str]:
    """Return leakage risks in a canonical zero-primitive source tree."""
    checks = {
        "primitives": set(),
        "strategies": set(),
    }
    findings: list[str] = []
    for package, allowed in checks.items():
        for path in _concrete_dirs(src / package, allowed=allowed):
            findings.append(path.relative_to(src).as_posix())
    for kind in ("primitives", "strategies"):
        root = src / "instantiations" / kind
        for path in _concrete_dirs(root):
            findings.append(path.relative_to(src).as_posix())
    return findings


@dataclass(frozen=True)
class IsolatedWorkspace:
    """Paths for one isolated suite workspace."""

    run_dir: Path
    workspace: Path
    src: Path
    results: Path
    agent_work: Path


def _resolve_workspace(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"workspace checkpoint does not exist: {candidate}")
    if (candidate / "src").is_dir():
        return candidate

    nested_workspace = candidate / "workspace"
    if (nested_workspace / "src").is_dir():
        return nested_workspace

    raise FileNotFoundError(
        "workspace checkpoint is missing src/. Pass either a workspace "
        f"directory or a run directory containing workspace/: {candidate}"
    )


def _write_workspace_readme(workspace: Path, *, source: str, migrated: bool = False) -> None:
    if migrated:
        body = (
            "This workspace was migrated to the current suite checkpoint "
            f"layout from {source}."
        )
    else:
        body = (
            "This workspace was generated for one suite trial. It contains "
            f"learned checkpoint state copied from {source}."
        )
    (workspace / "README.md").write_text(
        "# Isolated RAPID suite workspace\n\n"
        f"{body}\n\n"
        "Do not use it as the canonical source tree.\n",
        encoding="utf-8",
    )


def _copy_generated_test_scenes(source_workspace: Path, destination_workspace: Path) -> None:
    source = source_workspace / "test_scenes"
    if not source.is_dir():
        return

    destination = destination_workspace / "test_scenes"
    destination.mkdir(exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name in {"_common", "__init__.py", "__pycache__", "README.md"}:
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_tree(child, target)
        else:
            shutil.copy2(child, target)


def _copy_missing_canonical_infra(
    repo_root: Path,
    destination_src: Path,
) -> list[str]:
    canonical_src = repo_root / "workspace" / "src"
    copied: list[str] = []
    for package in (
        "instantiation_core",
        "instantiations",
        "primitive_core",
        "scene_io",
    ):
        source = canonical_src / package
        destination = destination_src / package
        if source.is_dir() and _copy_missing_tree(source, destination):
            copied.append(package)
    return copied


def _copy_invent_workspace_templates(repo_root: Path, workspace: Path) -> list[str]:
    template = repo_root / INVENT_WORKSPACE_TEMPLATE_REL
    if not template.is_dir():
        raise FileNotFoundError(f"missing invent workspace template: {template}")

    copied: list[str] = []
    for source in sorted(path for path in template.rglob("*") if path.is_file()):
        rel = source.relative_to(template)
        destination = workspace / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() == source.read_bytes():
            copied.append(rel.as_posix())
            continue
        shutil.copy2(source, destination)
        copied.append(rel.as_posix())
    return copied


def _write_workspace_codex_files(repo_root: Path, workspace: Path) -> tuple[Path, Path]:
    workspace_config = write_workspace_codex_config(workspace, repo_root=repo_root)
    codex_home_config = write_codex_home_bootstrap(
        run_codex_home_path(workspace),
        workspace=workspace,
        host_config=_host_codex_config(repo_root),
    )
    return workspace_config, codex_home_config


def _migrate_legacy_test_scenes(workspace: Path) -> list[str]:
    """Move old src/test_scenes generated scenes to workspace/test_scenes."""
    legacy = workspace / "src" / "test_scenes"
    if not legacy.is_dir():
        return []

    migrated: list[str] = []
    destination = workspace / "test_scenes"
    destination.mkdir(exist_ok=True)
    for child in sorted(legacy.iterdir()):
        if child.name in {"_common", "__init__.py", "__pycache__", "README.md"}:
            continue
        _copy_tree(child, destination / child.name) if child.is_dir() else shutil.copy2(
            child, destination / child.name
        )
        migrated.append(child.name)
    shutil.rmtree(legacy)
    _rewrite_text_tree(workspace, LEGACY_IMPORT_REWRITES)
    return migrated


def _remove_generated_test_scene_boilerplate(workspace: Path) -> list[str]:
    """Remove package boilerplate that is not part of generated validation scenes."""
    removed: list[str] = []
    for name in ("__init__.py",):
        path = workspace / "test_scenes" / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def create_isolated_workspace(
    run_dir: Path,
    *,
    repo_root: Path | None = None,
    from_workspace: Path | None = None,
) -> IsolatedWorkspace:
    """Create a clean candidate workspace for one suite trial."""
    repo_root = (repo_root or repo_root_from_env()).resolve()
    run_dir = run_dir.expanduser().resolve()
    workspace = run_dir / "workspace"
    src = workspace / "src"
    results = workspace / "results"
    agent_work = workspace / "agent_work"

    if workspace.exists():
        raise FileExistsError(
            f"candidate workspace already exists: {workspace}. "
            "Choose a new run directory or remove it explicitly."
        )

    pinned_runtime = os.environ.get("RAPID_RUNTIME_WORKSPACE", "").strip()
    pinned_hashes = (
        verify_runtime_manifest(Path(pinned_runtime)) if pinned_runtime else None
    )
    if from_workspace is None:
        source_workspace = _resolve_workspace(
            Path(pinned_runtime) if pinned_runtime else repo_root / "workspace"
        )
        findings = audit_zero_primitive_baseline(source_workspace / "src")
        if findings:
            formatted = "\n".join(f"  - src/{item}" for item in findings)
            raise RuntimeError(
                "refusing to create an isolated workspace from a non-empty "
                "canonical baseline. Start from that learned state explicitly "
                "with --from-workspace instead.\n"
                f"{formatted}"
            )
        source_label = str(source_workspace)
    else:
        source_workspace = _resolve_workspace(from_workspace)
        source_label = str(source_workspace)

    has_manifest = runtime_manifest_path(source_workspace).is_file()
    if has_manifest:
        verify_runtime_manifest(source_workspace)
    if pinned_hashes is not None:
        require_shared_source(source_workspace / "src", pinned_hashes)

    workspace.mkdir(parents=True)
    _copy_src_tree(source_workspace / "src", src)
    if not has_manifest and pinned_hashes is None:
        _copy_missing_canonical_infra(repo_root, src)
    _copy_generated_test_scenes(source_workspace, workspace)
    _copy_generated_test_scenes(source_workspace / "src", workspace)
    _migrate_legacy_test_scenes(workspace)
    _remove_generated_test_scene_boilerplate(workspace)
    _rewrite_text_tree(workspace, LEGACY_IMPORT_REWRITES)
    results.mkdir()
    agent_work.mkdir()
    _write_workspace_readme(workspace, source=source_label)
    _copy_invent_workspace_templates(repo_root, workspace)
    _write_workspace_codex_files(repo_root, workspace)
    if pinned_hashes is not None:
        require_shared_source(src, pinned_hashes)
    write_runtime_manifest(workspace)

    return IsolatedWorkspace(
        run_dir=run_dir,
        workspace=workspace,
        src=src,
        results=results,
        agent_work=agent_work,
    )


def _primitive_names(src: Path) -> list[str]:
    return [path.name for path in _concrete_dirs(src / "primitives")]


def _destination_group_exists(workspace: Path, primitive: str) -> bool:
    src = workspace / "src"
    return any(
        path.exists()
        for path in (
            src / "primitives" / primitive,
            src / "instantiations" / "primitives" / primitive,
            workspace / "test_scenes" / primitive,
        )
    )


def _source_group_paths(workspace: Path, primitive: str) -> dict[str, list[Path]]:
    src = workspace / "src"
    paths: dict[str, list[Path]] = {
        "primitives": [src / "primitives" / primitive],
        "instantiations": [
            src / "instantiations" / "primitives" / primitive,
        ],
        "test_scenes": [],
    }
    for root in (workspace / "test_scenes", src / "test_scenes"):
        candidate = root / primitive
        if candidate.exists():
            paths["test_scenes"].append(candidate)
    return paths


def _destination_group_same(
    destination_workspace: Path,
    source_workspace: Path,
    primitive: str,
) -> bool:
    src = destination_workspace / "src"
    destinations = {
        "primitives": src / "primitives" / primitive,
        "instantiations": (
            src / "instantiations" / "primitives" / primitive
        ),
        "test_scenes": destination_workspace / "test_scenes" / primitive,
    }
    for group, sources in _source_group_paths(source_workspace, primitive).items():
        for source in sources:
            if not source.exists():
                continue
            destination = destinations[group]
            if not _same_tree(destination, source):
                return False
    return True


def _next_renamed_primitive(destination_workspace: Path, primitive: str, source_id: str) -> str:
    base = f"{primitive}__from_{source_id}"
    candidate = base
    index = 2
    while _destination_group_exists(destination_workspace, candidate):
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _copy_primitive_group(
    source_workspace: Path,
    destination_workspace: Path,
    *,
    primitive: str,
    target_primitive: str,
) -> None:
    destination_src = destination_workspace / "src"
    destinations = {
        "primitives": destination_src / "primitives" / target_primitive,
        "instantiations": (
            destination_src
            / "instantiations"
            / "primitives"
            / target_primitive
        ),
        "test_scenes": destination_workspace / "test_scenes" / target_primitive,
    }
    for group, sources in _source_group_paths(source_workspace, primitive).items():
        for source in sources:
            if not source.exists():
                continue
            destination = destinations[group]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                _copy_tree(source, destination)
            else:
                shutil.copy2(source, destination)

    replacements = dict(LEGACY_IMPORT_REWRITES)
    replacements.update(
        _primitive_rename_replacements(primitive, target_primitive)
    )
    for path in destinations.values():
        _rewrite_text_tree(path, replacements)
    _rewrite_binding_manifest_program(
        destinations["instantiations"],
        program_kind="primitive",
        source_name=primitive,
        target_name=target_primitive,
    )


def _write_merge_index(
    workspace: Path,
    *,
    base_workspace: Path,
    source_records: list[dict[str, object]],
) -> None:
    lines = [
        f"base_workspace: {json.dumps(str(base_workspace))}",
        "sources:",
    ]
    for record in source_records:
        lines.extend(
            [
                f"  - path: {json.dumps(record['path'])}",
                f"    source_id: {json.dumps(record['source_id'])}",
                f"    copied: {json.dumps(record['copied'], sort_keys=True)}",
                f"    renamed: {json.dumps(record['renamed'], sort_keys=True)}",
                f"    deduplicated: {json.dumps(record['deduplicated'], sort_keys=True)}",
            ]
        )
    lines.append("")
    (workspace / "merge_index.yaml").write_text("\n".join(lines), encoding="utf-8")


def merge_workspaces(
    run_dir: Path,
    source_workspaces: Sequence[Path],
    *,
    repo_root: Path | None = None,
    base_workspace: Path | None = None,
) -> IsolatedWorkspace:
    """Create a workspace by merging learned state from source workspaces."""
    if not source_workspaces:
        raise ValueError("merge-workspaces requires at least one --source-workspace")

    repo_root = (repo_root or repo_root_from_env()).resolve()
    resolved_base = _resolve_workspace(
        base_workspace
        or Path(os.environ.get("RAPID_RUNTIME_WORKSPACE") or repo_root / "workspace")
    )
    if runtime_manifest_path(resolved_base).is_file():
        verify_runtime_manifest(resolved_base)
    expected_runtime = shared_source_hashes(resolved_base / "src")
    for source_arg in source_workspaces:
        source = _resolve_workspace(source_arg)
        if runtime_manifest_path(source).is_file():
            verify_runtime_manifest(source)
        require_shared_source(source / "src", expected_runtime)
    workspace_info = create_isolated_workspace(
        run_dir,
        repo_root=repo_root,
        from_workspace=resolved_base,
    )

    used_source_ids: set[str] = set()
    source_records: list[dict[str, object]] = []
    for source_arg in source_workspaces:
        source_workspace = _resolve_workspace(source_arg)
        source_id = _source_id(source_workspace, used_source_ids)
        record: dict[str, object] = {
            "path": str(source_workspace),
            "source_id": source_id,
            "copied": [],
            "renamed": [],
            "deduplicated": [],
        }

        for primitive in _primitive_names(source_workspace / "src"):
            if _destination_group_exists(workspace_info.workspace, primitive):
                if _destination_group_same(workspace_info.workspace, source_workspace, primitive):
                    record["deduplicated"].append({"primitive": primitive})  # type: ignore[index]
                    continue
                target_primitive = _next_renamed_primitive(
                    workspace_info.workspace,
                    primitive,
                    source_id,
                )
                _copy_primitive_group(
                    source_workspace,
                    workspace_info.workspace,
                    primitive=primitive,
                    target_primitive=target_primitive,
                )
                record["renamed"].append(  # type: ignore[index]
                    {"from": primitive, "to": target_primitive}
                )
            else:
                _copy_primitive_group(
                    source_workspace,
                    workspace_info.workspace,
                    primitive=primitive,
                    target_primitive=primitive,
                )
                record["copied"].append({"primitive": primitive})  # type: ignore[index]

        source_records.append(record)

    _migrate_legacy_test_scenes(workspace_info.workspace)
    _write_merge_index(
        workspace_info.workspace,
        base_workspace=resolved_base,
        source_records=source_records,
    )
    require_shared_source(workspace_info.src, expected_runtime)
    return workspace_info


def migrate_workspace_to_current_format(
    workspace: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Update one existing run workspace to the current checkpoint layout."""
    repo_root = (repo_root or repo_root_from_env()).resolve()
    workspace = _resolve_workspace(workspace)
    src = workspace / "src"

    copied_infra = _copy_missing_canonical_infra(repo_root, src)

    migrated_test_scenes = _migrate_legacy_test_scenes(workspace)
    removed_test_scene_boilerplate = _remove_generated_test_scene_boilerplate(workspace)

    removed_legacy_src: list[str] = []
    for name in sorted(LEGACY_SRC_EXCLUDE_NAMES - {"test_scenes"}):
        path = src / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed_legacy_src.append(name)

    _rewrite_text_tree(src, LEGACY_IMPORT_REWRITES)
    _rewrite_text_tree(workspace / "test_scenes", LEGACY_IMPORT_REWRITES)
    _write_workspace_readme(
        workspace,
        source="an existing historical run workspace",
        migrated=True,
    )
    copied_guidance = _copy_invent_workspace_templates(repo_root, workspace)
    codex_config, codex_home_config = _write_workspace_codex_files(repo_root, workspace)

    record: dict[str, object] = {
        "workspace": str(workspace),
        "copied_infra": copied_infra,
        "migrated_test_scenes": migrated_test_scenes,
        "removed_test_scene_boilerplate": removed_test_scene_boilerplate,
        "removed_legacy_src": removed_legacy_src,
        "updated_readme": True,
        "copied_guidance": copied_guidance,
        "codex_config": str(codex_config),
        "codex_home_config": str(codex_home_config),
    }
    lines = [
        "format: workspace_checkpoint_v2",
        f"workspace: {json.dumps(str(workspace))}",
        f"copied_infra: {json.dumps(copied_infra, sort_keys=True)}",
        f"migrated_test_scenes: {json.dumps(migrated_test_scenes, sort_keys=True)}",
        "removed_test_scene_boilerplate: "
        f"{json.dumps(removed_test_scene_boilerplate, sort_keys=True)}",
        f"removed_legacy_src: {json.dumps(removed_legacy_src, sort_keys=True)}",
        "updated_readme: true",
        f"copied_guidance: {json.dumps(copied_guidance, sort_keys=True)}",
        f"codex_config: {json.dumps(str(codex_config))}",
        f"codex_home_config: {json.dumps(str(codex_home_config))}",
        "",
    ]
    (workspace / "migration_index.yaml").write_text("\n".join(lines), encoding="utf-8")
    return record


def run_module_in_workspace(
    workspace: Path,
    module: str,
    module_args: Sequence[str] = (),
    *,
    repo_root: Path | None = None,
) -> int:
    """Run a Python module against a candidate ``workspace/src`` tree."""
    repo_root = (repo_root or repo_root_from_env()).resolve()
    workspace = workspace.expanduser().resolve()
    env = os.environ.copy()
    env["RAPID_REPO_ROOT"] = str(repo_root)
    env["RAPID_WORKSPACE"] = str(workspace)
    env["RAPID_ACTIVE_SRC_DIR"] = str(workspace / "src")
    env["RAPID_SRC_DIR"] = str(workspace / "src")
    cmd = [
        str(repo_root / "scripts" / "rapid-python"),
        "-m",
        module,
        *module_args,
    ]
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return int(completed.returncode)
