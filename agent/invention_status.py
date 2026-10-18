"""Build and print machine-readable invention outcome summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from instantiation_core.qualification import source_fingerprint
from .codex_exec import AGENT_WORK_DIR, WORKSPACE
from .invention_context import InventionKind


AuditStatus = Literal["clean", "advisory", "unknown"]
BehaviorStatus = Literal["passed", "failed", "incomplete"]

_MIN_RETAINED_PRIMITIVE_ROLLOUTS = 32
_MIN_RETAINED_PRIMITIVE_ITERS = 6
_REQUIRED_PRIMITIVE_DIAGNOSTICS = (
    "cost_components",
    "progress_diagnostics",
    "mechanism_diagnostics",
)

_AUDIT_BOOLEAN_KEYS = {
    "audit_advisory_triggered",
    "transit_validation_advisory",
}
_ADVISORY_FALSE_KEYS = {
    "controller_state_handoff_valid",
    "retained_timing_valid",
    "transit_convergence_valid",
}


def canonical_status_path() -> Path:
    return AGENT_WORK_DIR / "invention_status.json"


def _behavior_value(payload: Mapping[str, Any]) -> bool | None:
    for container in (
        payload,
        payload.get("binding"),
        payload.get("metrics"),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("program_success", "success"):
            value = container.get(key)
            if isinstance(value, bool):
                return value
    return None


def _audit_findings(value: Any, *, path: str = "metrics") -> set[str]:
    findings: set[str] = set()
    if isinstance(value, Mapping):
        advisories = value.get("audit_advisories")
        if isinstance(advisories, Sequence) and not isinstance(
            advisories, (str, bytes)
        ):
            findings.update(f"{path}:{item}" for item in advisories if item)
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _AUDIT_BOOLEAN_KEYS and item is True:
                findings.add(item_path)
            if (
                (key.endswith("_audit_valid") or key in _ADVISORY_FALSE_KEYS)
                and item is False
            ):
                findings.add(item_path)
            if isinstance(item, (Mapping, list, tuple)):
                findings.update(_audit_findings(item, path=item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.update(_audit_findings(item, path=f"{path}[{index}]"))
    return findings


def _nested_numeric_values(value: Any, key: str) -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                values.extend(_nested_numeric_values(child, key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.extend(_nested_numeric_values(child, key))
    return values


def _diagnostic_mapping_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    finite_leaf = False
    stack = list(value.values())
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, bool):
            finite_leaf = True
        elif isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                return False
            finite_leaf = True
    return finite_leaf


def _primitive_evidence_findings(metrics: Mapping[str, Any]) -> list[str]:
    from primitive_core.evidence import primitive_evidence_findings
    findings = primitive_evidence_findings(metrics)
    rollouts = _nested_numeric_values(metrics, "num_rollouts")
    iterations = _nested_numeric_values(metrics, "requested_max_iters")
    if (
        not rollouts
        or max(rollouts) < _MIN_RETAINED_PRIMITIVE_ROLLOUTS
        or not iterations
        or max(iterations) < _MIN_RETAINED_PRIMITIVE_ITERS
    ):
        findings.append("retained primitive evidence is diagnostic-only")
    for key in _REQUIRED_PRIMITIVE_DIAGNOSTICS:
        if not _diagnostic_mapping_valid(metrics.get(key)):
            findings.append(f"retained metrics require finite {key}")
    effect_success = metrics.get("effect_success")
    mechanism_valid = metrics.get("mechanism_valid")
    program_success = metrics.get("success")
    if (
        isinstance(effect_success, bool)
        and isinstance(mechanism_valid, bool)
        and isinstance(program_success, bool)
        and program_success != (effect_success and mechanism_valid and
                                (not metrics.get("stability_required") or metrics.get("stability_valid") is True)
                                and metrics.get("transit", {}).get("transit_valid", True))
    ):
        findings.append("success disagrees with final effect, mechanism, stability or execution validity")
    return findings


def _aggregate_boolean_status(values: Sequence[bool | None]) -> BehaviorStatus:
    if not values or any(value is None for value in values):
        return "incomplete"
    if all(values):
        return "passed"
    return "failed"


def _example_status(
    metrics_path: Path,
    *,
    require_primitive_evidence: bool,
) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {
            "metrics_path": str(metrics_path),
            "behavior_success": None,
            "audit_status": "unknown",
            "audit_findings": [],
            "finding": "retained metrics are missing",
        }
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "metrics_path": str(metrics_path),
            "behavior_success": None,
            "audit_status": "unknown",
            "audit_findings": [],
            "finding": f"retained metrics are invalid: {exc}",
        }
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return {
            "metrics_path": str(metrics_path),
            "behavior_success": _behavior_value(payload),
            "audit_status": "unknown",
            "audit_findings": [],
            "finding": "retained metrics payload is missing",
        }
    audit_findings = sorted(_audit_findings(metrics))
    evidence_findings = (
        _primitive_evidence_findings(metrics) if require_primitive_evidence else []
    )
    behavior_success = _behavior_value(payload)
    return {
        "metrics_path": str(metrics_path),
        "behavior_success": behavior_success,
        "effect_success": metrics.get("effect_success", behavior_success),
        "mechanism_valid": metrics.get("mechanism_valid"),
        "retained_evidence_valid": not evidence_findings,
        "evidence_findings": evidence_findings,
        "audit_status": "advisory" if audit_findings else "clean",
        "audit_findings": audit_findings,
    }


def build_invention_status(
    *,
    kind: InventionKind,
    programs: Sequence[str],
    num_variants: int,
    workflow_completed: bool,
    validation_findings: Sequence[str] = (),
    workspace: Path = WORKSPACE,
) -> dict[str, Any]:
    """Summarize workflow, effect, mechanism, and audits independently."""

    examples: dict[str, dict[str, Any]] = {}
    for program in programs:
        for index in range(num_variants + 1):
            instance = f"example_{index}"
            key = f"{program}/{instance}"
            metrics_path = workspace / "results" / f"{program}_{instance}_metrics.json"
            examples[key] = _example_status(
                metrics_path,
                require_primitive_evidence=kind == "primitive",
            )

    behavior_values = [
        item["behavior_success"]
        if item.get("retained_evidence_valid", True)
        else None
        for item in examples.values()
    ]
    behavior_status = _aggregate_boolean_status(behavior_values)

    audit_values = [item["audit_status"] for item in examples.values()]
    if any(value == "advisory" for value in audit_values):
        audit_status: AuditStatus = "advisory"
    elif not audit_values or any(value == "unknown" for value in audit_values):
        audit_status = "unknown"
    else:
        audit_status = "clean"

    status: dict[str, Any] = {
        "schema": "rapid.invention_status",
        "schema_version": 2,
        "kind": kind,
        "programs": list(programs),
        "requested_examples_per_program": num_variants + 1,
        "workflow_completed": workflow_completed,
        "behavior_success": behavior_status == "passed",
        "behavior_status": behavior_status,
        "audit_status": audit_status,
        "examples": examples,
        "validation_findings": list(validation_findings),
        "qualification": {
            program: _qualification_status(kind, program, workspace)
            for program in programs
        },
    }
    if kind == "primitive":
        effect_values = [
            item.get("effect_success")
            if item.get("retained_evidence_valid", False)
            else None
            for item in examples.values()
        ]
        mechanism_values = [
            item.get("mechanism_valid")
            if item.get("retained_evidence_valid", False)
            else None
            for item in examples.values()
        ]
        effect_status = _aggregate_boolean_status(effect_values)
        mechanism_status = _aggregate_boolean_status(mechanism_values)
        status.update(
            {
                "effect_success": effect_status == "passed",
                "effect_status": effect_status,
                "mechanism_valid": mechanism_status == "passed",
                "mechanism_status": mechanism_status,
            }
        )
    return status


def _qualification_status(kind: str, program: str, workspace: Path) -> dict[str, Any]:
    """Keep fixed probe coverage separate from behavioral and audit outcomes."""
    directory = workspace / "agent_work" / "qualification" / program
    report_path = directory / "report.json"
    required = {"scene_interface", "representation", "layout"}
    if kind == "strategy":
        required.add("handoff")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        plan = (directory / "cases.json").read_bytes()
        cases = json.loads(plan)
        checks = report["checks"]
        if not report.get("probe_sources") or any(
            not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            for path, digest in report["probe_sources"].items()
        ):
            return {"status": "stale", "report_path": str(report_path)}
        package = "primitives" if kind == "primitive" else "strategies"
        sources = [workspace / "src" / package / program]
        if kind == "strategy":
            sources.append(workspace / "src" / "primitives")
        if report["source_fingerprint"] != source_fingerprint(sources):
            return {"status": "stale", "report_path": str(report_path)}
        if (report["cases_sha256"] != hashlib.sha256(plan).hexdigest()
                or set(cases) != set(checks)
                or any(cases[name]["category"] != checks[name]["category"] for name in cases)):
            return {"status": "stale", "report_path": str(report_path)}
        categories = {case["category"] for case in checks.values()}
        missing = sorted(required - categories)
        passed = bool(checks) and not missing and not report.get("coverage_findings") and all(
            check["passed"] is True and not check["findings"] and bool(check.get("measurements"))
            for check in checks.values()
        )
        return {
            "status": "passed" if passed else (
                "failed" if any(check.get("status") == "failed" for check in checks.values())
                else "incomplete"),
            "report_path": str(report_path),
            "missing_categories": missing,
            "checks": checks,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"status": "unknown", "report_path": str(report_path)}


def write_invention_status(
    payload: Mapping[str, Any],
    *paths: Path,
    announce: bool = True,
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    if announce:
        primitive_status = ""
        if payload.get("kind") == "primitive":
            primitive_status = (
                f" effect={payload['effect_status']}"
                f" mechanism={payload['mechanism_status']}"
            )
        print(
            "[invention-status] "
            f"workflow={'completed' if payload['workflow_completed'] else 'incomplete'} "
            f"behavior={payload['behavior_status']}"
            f"{primitive_status} audit={payload['audit_status']}"
        )


def initialize_invention_status(
    *,
    kind: InventionKind,
    num_variants: int,
    run_dir: Path,
) -> None:
    payload = build_invention_status(
        kind=kind,
        programs=(),
        num_variants=num_variants,
        workflow_completed=False,
    )
    write_invention_status(
        payload,
        canonical_status_path(),
        run_dir / "invention_status.json",
        announce=False,
    )


def finalize_invention_status(
    *,
    kind: InventionKind,
    programs: Sequence[str],
    num_variants: int,
    run_dir: Path,
    validation_findings: Sequence[str] = (),
    workflow_completed: bool = True,
) -> dict[str, Any]:
    payload = build_invention_status(
        kind=kind,
        programs=programs,
        num_variants=num_variants,
        workflow_completed=workflow_completed,
        validation_findings=validation_findings,
    )
    write_invention_status(
        payload,
        canonical_status_path(),
        run_dir / "invention_status.json",
    )
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Print tab-separated workflow, behavior, and audit status."
    )
    parser.add_argument("status_file", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.status_file.read_text(encoding="utf-8"))
    print(
        "\t".join(
            (
                "completed" if payload.get("workflow_completed") else "incomplete",
                str(payload.get("behavior_status", "unknown")),
                str(payload.get("audit_status", "unknown")),
            )
        )
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "build_invention_status",
    "canonical_status_path",
    "finalize_invention_status",
    "initialize_invention_status",
    "write_invention_status",
]
