"""One-pass invention session with auditable source provenance."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from primitive_core.common import GLOBAL_MAX_OPTIMIZATION_BUDGET
from suites.staged.runtime_integrity import (
    runtime_manifest_path,
    shared_source_hashes,
    source_changes,
    verify_runtime_manifest,
)

from .codex_exec import (
    PRIMITIVES_DIR,
    SRC_DIR,
    STRATEGIES_DIR,
    CodexExecConfig,
    run_codex_exec,
)
from .instantiation_validation import digest_tree
from .invention_context import InventionKind


def _report_findings(
    kind: InventionKind,
    label: str,
    findings: Sequence[str],
    *,
    fatal: bool,
) -> None:
    prefix = f"[invent_{kind}_exec]"
    if not findings:
        print(f"{prefix} {label} passed")
        return
    disposition = "failed" if fatal else "advisory findings"
    print(f"{prefix} {label} {disposition}:", file=sys.stderr)
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    if fatal:
        raise SystemExit(1)


@dataclass(frozen=True)
class _ArtifactGuard:
    label: str
    paths: Mapping[str, Path]
    digests: Mapping[str, str]

    @classmethod
    def capture(cls, label: str, paths: Mapping[str, Path]) -> _ArtifactGuard:
        return cls(label, paths, {name: digest_tree(path) for name, path in paths.items()})

    def check(self, kind: InventionKind, *, fatal: bool = True) -> list[str]:
        findings = [
            f"changed tracked artifact: {name}"
            for name, path in self.paths.items()
            if digest_tree(path) != self.digests[name]
        ]
        _report_findings(kind, self.label, findings, fatal=fatal)
        return findings


def _tracked_source_paths(kind: InventionKind) -> dict[str, Path]:
    learned_root = PRIMITIVES_DIR if kind == "primitive" else STRATEGIES_DIR
    allowed = {learned_root.name, "instantiations", "__pycache__"}
    paths = {
        child.relative_to(SRC_DIR).as_posix(): child
        for child in SRC_DIR.iterdir()
        if child.name not in allowed
    }
    if learned_root.is_dir():
        paths.update(
            {
                child.relative_to(SRC_DIR).as_posix(): child
                for child in learned_root.iterdir()
                if child.name != "__pycache__"
            }
        )
    return paths


@dataclass(frozen=True)
class InventionSession:
    """Run one Codex phase over every requested invention example."""

    kind: InventionKind
    run_dir: Path
    config: CodexExecConfig
    iteration_ceiling: int

    def __post_init__(self) -> None:
        if self.iteration_ceiling < 1:
            raise ValueError("iteration_ceiling must be positive")
        if self.iteration_ceiling > GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters:
            raise ValueError(
                "iteration_ceiling exceeds GLOBAL_MAX_OPTIMIZATION_BUDGET"
            )

    @property
    def attempt_log(self) -> Path:
        return self.run_dir / "invention_attempts.jsonl"

    def _config(self, images: Sequence[Path]) -> CodexExecConfig:
        environment = dict(self.config.environment)
        environment["RAPID_INVENTION_ATTEMPT_LOG"] = str(self.attempt_log)
        environment["RAPID_OPTIMIZER_NUM_ROLLOUTS"] = str(
            _environment_ceiling(
                "RAPID_OPTIMIZER_NUM_ROLLOUTS",
                GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts,
            )
        )
        environment["RAPID_OPTIMIZER_MAX_ITERS"] = str(
            _environment_ceiling(
                "RAPID_OPTIMIZER_MAX_ITERS",
                self.iteration_ceiling,
            )
        )
        return replace(
            self.config,
            local_images=images,
            output_last_message=self.run_dir / "codex_final.md",
            environment=environment,
        )

    def run(self, prompt: str, images: Sequence[Path]) -> list[str]:
        if runtime_manifest_path(SRC_DIR.parent).is_file():
            verify_runtime_manifest(SRC_DIR.parent)
        shared_before = shared_source_hashes(SRC_DIR)
        provenance_guard = _ArtifactGuard.capture(
            "invention source provenance",
            _tracked_source_paths(self.kind),
        )
        try:
            run_codex_exec(prompt, self._config(images))
        finally:
            changes = source_changes(shared_before, shared_source_hashes(SRC_DIR))
            (self.run_dir / "runtime_integrity.json").write_text(
                json.dumps(
                    {"status": "failed" if changes else "passed",
                     "publishable": not changes, "changed_files": changes},
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            _report_findings(self.kind, "shared runtime integrity", changes, fatal=True)
            provenance_findings = provenance_guard.check(self.kind, fatal=False)
        return provenance_findings


def _environment_ceiling(name: str, ceiling: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ceiling
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name}={raw!r} must be a positive integer")
    return min(ceiling, value)


__all__ = ["InventionSession"]
