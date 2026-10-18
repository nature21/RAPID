"""Final primitive outcome after retained-trajectory verification."""

from __future__ import annotations

from collections.abc import Mapping

from .common import CompileResult


def primitive_evidence_findings(metrics: Mapping) -> list[str]:
    findings = []
    for key in ("effect_success", "mechanism_valid", "stability_required"):
        if not isinstance(metrics.get(key), bool):
            findings.append(f"retained metrics require boolean {key}")
    if metrics.get("stability_required") is True and not isinstance(metrics.get("stability_valid"), bool):
        findings.append("required terminal stability was not measured")
    return findings


def finalize_primitive_result(result: CompileResult) -> CompileResult:
    """Derive final success without discarding unsuccessful physical evidence.

    Learned diagnostics supply effect_success, mechanism_valid, and whether
    the declared terminal semantics require stability. Required probes supply
    stability_valid. optimized_success remains the optimizer's effect result.
    Missing evidence is incomplete, never a successful primitive.
    """
    metrics = result.metrics
    findings = primitive_evidence_findings(metrics)
    metrics["evidence_findings"] = findings
    metrics["evidence_valid"] = not findings
    metrics["success"] = bool(
        not findings
        and metrics["effect_success"]
        and metrics["mechanism_valid"]
        and (not metrics["stability_required"] or metrics["stability_valid"])
        and metrics.get("transit", {}).get("transit_valid", True)
    )
    return result
