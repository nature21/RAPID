"""Fixed development probes for runtime interfaces and relational geometry."""

from __future__ import annotations

import hashlib
import json
import importlib
import inspect
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class ProbeIncomplete(RuntimeError):
    """A required physical probe has not reached its execution boundary."""


@contextmanager
def diagnostic_execution():
    keys = {"RAPID_INVENTION_ATTEMPT_LOG": None, "RAPID_OPTIMIZER_NUM_ROLLOUTS": "2", "RAPID_OPTIMIZER_MAX_ITERS": "1"}
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in keys.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with tempfile.TemporaryDirectory(prefix="rapid_probe_") as directory:
            yield ["--no-record", "--num-rollouts", "2", "--max-iters", "1",
                   "--metrics-json", f"{directory}/metrics.json",
                   "--state-trajectory-npz", f"{directory}/states.npz",
                   "--cartesian-goals-json", f"{directory}/goals.json"]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class GeometryProbe:
    """Measure equivalent physical observations and a non-equivalent control."""

    reference: Callable
    observed: Callable
    negative_control: Callable

    def run(self, case):
        reference = self.reference(case)
        observed = self.observed(case)
        control = self.negative_control(case)
        rotation = np.asarray(case.get("rotation", np.eye(3)), dtype=float)
        translation = np.asarray(case.get("translation", np.zeros(3)), dtype=float)
        findings = geometry_findings(reference, observed, rotation=rotation, translation=translation,
                                     atol=float(case.get("atol", 1e-7)))
        if not geometry_findings(reference, control, rotation=rotation, translation=translation,
                                 atol=float(case.get("atol", 1e-7))):
            findings.append("negative control did not distinguish an incorrect physical observation")
        return findings, {"reference": reference, "observed": observed, "negative_control": control}


@dataclass(frozen=True)
class SceneInterfaceProbe:
    manifest: Path

    def run(self, case):
        from .entrypoint import run_manifest_entrypoint
        with diagnostic_execution() as arguments:
            result = run_manifest_entrypoint(self.manifest, arguments)
        return [], {"manifest": str(self.manifest), "execution_completed": True,
                    "behavior_success": bool(result.metrics.get("success"))}


@dataclass(frozen=True)
class HandoffProbe:
    manifest: Path
    phase: str

    def run(self, case):
        from .handoff import run_handoff_probe
        from .manifest import load_binding_manifest
        binding = load_binding_manifest(self.manifest)
        directory = Path(os.environ["RAPID_WORKSPACE"]) / "agent_work/handoffs" / binding.program_name / binding.instance_name / self.phase
        try:
            latest = json.loads((directory / "latest.json").read_text())
        except FileNotFoundError as exc:
            raise ProbeIncomplete(f"no executed boundary for {binding.instance_name}/{self.phase}") from exc
        snapshot = directory / latest["snapshot"]
        if snapshot.parent != directory or snapshot.suffix != ".npz":
            raise ValueError("invalid handoff snapshot")
        evidence = run_handoff_probe(self.manifest, snapshot, case["configuration_perturbations"])
        return [finding for sample in evidence["samples"] for finding in sample["findings"]], evidence


def geometry_findings(
    reference: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    atol: float = 1e-7,
) -> list[str]:
    """Compare world positions/directions, intrinsic scalars and decisions.

    Callers decode contract-frame outputs to world coordinates before calling.
    A representation-only probe uses the identity transform. Physical layout
    probes must preserve the declared semantic and embodied constraints.
    """
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)):
        raise ValueError("comparison requires a finite rigid transform")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    findings: list[str] = []
    groups = {"positions", "directions", "scalars", "decisions"}
    if not reference or not any(reference.values()):
        return ["geometry observation is empty"]
    if set(reference) - groups or set(observed) != set(reference):
        return ["geometry observation groups differ or are unsupported"]
    for group, quantities in reference.items():
        actual = observed[group]
        if set(actual) != set(quantities):
            findings.append(f"{group}: observable names differ")
            continue
        for name, value in quantities.items():
            if group == "decisions":
                valid = isinstance(value, (bool, np.bool_)) and isinstance(actual[name], (bool, np.bool_)) and value == actual[name]
            else:
                expected = np.asarray(value, dtype=float)
                measured = np.asarray(actual[name], dtype=float)
                shape = (3,) if group in {"positions", "directions"} else ()
                valid = (expected.shape == measured.shape == shape
                         and np.all(np.isfinite(expected)) and np.all(np.isfinite(measured)))
                if valid:
                    if group in {"positions", "directions"}:
                        expected = rotation @ expected
                    if group == "positions":
                        expected = expected + translation
                    valid = np.allclose(expected, measured, atol=atol, rtol=0)
            if not valid:
                findings.append(f"{group}.{name}: physical observation differs")
    return findings


def freeze_validation_cases(path: Path, cases: Mapping[str, Mapping[str, Any]]) -> None:
    """Record probe inputs before tuning; refuse replacement on subsequent runs."""
    if not cases or any(not case.get("category") for case in cases.values()):
        raise ValueError("validation cases require names and categories")
    encoded = json.dumps(cases, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(encoded):
            raise ValueError("fixed validation cases cannot be replaced") from None


def source_fingerprint(paths: Sequence[Path]) -> str:
    """Fingerprint the learned program and its primitive dependencies."""
    digest = hashlib.sha256()
    for root in sorted(paths, key=str):
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for path in files:
            digest.update((root.name + "/" + str(path.relative_to(root) if root.is_dir() else path.name)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_validation_suite(
    plan_path: Path,
    probes: Mapping[str, GeometryProbe | SceneInterfaceProbe | HandoffProbe],
    *,
    source_paths: Sequence[Path],
    report_path: Path,
) -> dict[str, Any]:
    """Execute every frozen probe; retain failures without dropping hard cases.

    The host executes typed probes and evaluates their observations. A callback
    returning an empty findings list is not execution evidence. Behavior success
    remains separate from interface and handoff qualification.
    """
    plan_bytes = plan_path.read_bytes()
    cases = json.loads(plan_bytes)
    if not cases or set(probes) != set(cases):
        raise ValueError("probes must cover exactly the fixed validation cases")
    if not source_paths or any(not path.exists() for path in source_paths):
        raise ValueError("validation requires existing learned source paths")
    fingerprint = source_fingerprint(source_paths)
    probe_sources = {}
    for probe in probes.values():
        functions = ([probe.reference, probe.observed, probe.negative_control]
                     if isinstance(probe, GeometryProbe) else [type(probe)])
        for function in functions:
            try:
                filename = inspect.getsourcefile(function)
            except TypeError:
                filename = None
            if filename and Path(filename).is_file():
                probe_sources[str(Path(filename).resolve())] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    results = {}
    for name, case in cases.items():
        status = "passed"
        try:
            expected = {"representation": GeometryProbe, "layout": GeometryProbe,
                        "scene_interface": SceneInterfaceProbe, "handoff": HandoffProbe}[case["category"]]
            if not isinstance(probes[name], expected):
                raise TypeError(f"{case['category']} requires {expected.__name__}")
            findings, measurements = probes[name].run(case)
            if findings:
                status = "failed"
        except ProbeIncomplete as exc:
            status = "incomplete"
            findings = [str(exc)]
            measurements = {}
        except Exception as exc:
            status = "failed"
            findings = [f"{type(exc).__name__}: {exc}"]
            measurements = {}
        results[name] = {"category": case["category"], "status": status, "passed": status == "passed", "findings": findings,
                         "measurements": measurements}
    if source_fingerprint(source_paths) != fingerprint:
        raise ValueError("learned sources changed during validation")
    if plan_path.read_bytes() != plan_bytes:
        raise ValueError("fixed validation cases changed during validation")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
           for path, digest in probe_sources.items()):
        raise ValueError("probe sources changed during validation")
    coverage_findings = []
    if source_paths[0].parent.name == "strategies":
        program = importlib.import_module(f"strategies.{source_paths[0].name}.program").PHASE_PROGRAM
        from .manifest import load_binding_manifest
        instances = source_paths[0].parent.parent / "instantiations/strategies" / source_paths[0].name
        required = {(load_binding_manifest(path).instance_name, phase.name)
                    for path in instances.glob("example_*/binding.json") for phase in program.phases[1:]}
        covered = {(load_binding_manifest(probe.manifest).instance_name, probe.phase)
                   for probe in probes.values() if isinstance(probe, HandoffProbe)}
        if required - covered:
            coverage_findings.append("uncovered phase boundaries: " + ", ".join(
                f"{instance}/{phase}" for instance, phase in sorted(required - covered)))
        if not required and len(program.phases) > 1:
            coverage_findings.append("required example bindings are missing")
    passed = not coverage_findings and all(case["passed"] for case in results.values())
    status = "passed" if passed else (
        "failed" if any(case["status"] == "failed" for case in results.values()) else "incomplete")
    report = {
        "source_fingerprint": fingerprint,
        "cases_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "probe_sources": probe_sources,
        "coverage_findings": coverage_findings,
        "passed": passed,
        "status": status,
        "checks": results,
    }
    from .json_values import write_json_file
    write_json_file(report_path, report)
    return report
