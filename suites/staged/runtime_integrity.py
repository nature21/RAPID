"""Shared-source identity across invention, library assembly, and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def shared_source_hashes(src: Path) -> dict[str, str]:
    """Hash infrastructure, including package files but excluding learned programs."""
    hashes = {}
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or rel.parts[0] == "instantiations":
            continue
        if rel.parts[0] in {"primitives", "strategies"} and len(rel.parts) > 2:
            continue
        hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def source_changes(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    return [
        name for name in sorted(expected.keys() | actual.keys())
        if expected.get(name) != actual.get(name)
    ]


def require_shared_source(src: Path, expected: Mapping[str, str]) -> None:
    changes = source_changes(expected, shared_source_hashes(src))
    if changes:
        raise RuntimeError(
            f"shared runtime mismatch in {src}: " + ", ".join(changes)
        )


def runtime_manifest_path(workspace: Path) -> Path:
    return workspace.parent / "runtime_manifest.json"


def write_runtime_manifest(workspace: Path) -> None:
    runtime_manifest_path(workspace).write_text(
        json.dumps(shared_source_hashes(workspace / "src"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_runtime_manifest(workspace: Path) -> dict[str, str]:
    expected = json.loads(runtime_manifest_path(workspace).read_text(encoding="utf-8"))
    require_shared_source(workspace / "src", expected)
    return expected
