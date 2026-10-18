"""Resolvers for immutable runtime assets shared by task scenes."""

from __future__ import annotations

import os
from pathlib import Path


FR3_TASK_V1_RELATIVE_PATH = Path("runtime_assets/fr3/fr3_task_v1")


def fr3_task_v1_root() -> Path:
    """Return the canonical task-tuned FR3 directory.

    Isolated suite launches provide ``RAPID_REPO_ROOT``.  The relative fallback
    keeps direct execution from the canonical repository convenient.
    """

    candidates: list[Path] = []
    repo_root = os.environ.get("RAPID_REPO_ROOT")
    if repo_root:
        candidates.append(Path(repo_root).expanduser() / FR3_TASK_V1_RELATIVE_PATH)
    candidates.append(Path(__file__).resolve().parents[3] / FR3_TASK_V1_RELATIVE_PATH)

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if (resolved / "robot.xml").is_file():
            return resolved

    searched = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "FR3 task runtime asset is unavailable; expected fr3_task_v1 at one "
        f"of: {searched}"
    )


def fr3_task_v1_xml() -> Path:
    """Return the standalone MuJoCo XML for the task-tuned FR3."""

    return fr3_task_v1_root() / "robot.xml"


__all__ = [
    "FR3_TASK_V1_RELATIVE_PATH",
    "fr3_task_v1_root",
    "fr3_task_v1_xml",
]
