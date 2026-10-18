"""Shared experiment directory selection for primitive-facing helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def experiment_root(repo_root: Path, value: str | Path | None = None) -> Path:
    if value is None:
        value = os.environ.get("RAPID_EXPERIMENT_ROOT") or "suite_runs"
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = repo_root if path.parts and path.parts[0] == "suite_runs" else repo_root / "suite_runs"
        path = base / path
    return path.resolve()


def add_experiment_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-root",
        help="Experiment directory under suite_runs, or an absolute path "
        "(default: RAPID_EXPERIMENT_ROOT or suite_runs)",
    )


def configure_experiment_root(args: argparse.Namespace, repo_root: Path) -> None:
    # Export the resolved root so child helper processes use the same directory.
    os.environ["RAPID_EXPERIMENT_ROOT"] = str(
        experiment_root(repo_root, args.experiment_root)
    )
