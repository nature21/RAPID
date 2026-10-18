"""Compact, JSON-safe evidence for frozen strategy deployment."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from instantiation_core.contracts import BindingContract


DEPLOY_RESPONSE_SCHEMA_VERSION = "rapid.deploy-binding-response/v1"
DEPLOY_CONTEXT_RESULT_PREFIX = "RAPID_DEPLOYMENT_CONTEXT_RESULT="
_RESPONSE_KEYS = frozenset(
    {"schema_version", "compatible", "roles", "diagnostics"}
)


@dataclass(frozen=True)
class DeploymentBindingDecision:
    """One validated semantic decision returned by the deployment agent."""

    compatible: bool
    roles: dict[str, str | list[str]]
    diagnostics: tuple[str, ...]


def _enum_suffix(value: object) -> str:
    text = str(getattr(value, "name", value)).lower()
    for marker in ("mjgeom_", "mjjnt_", "mjtrn_"):
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return text


def serialize_strategy_contract(contract: BindingContract) -> dict[str, object]:
    """Return only the declarative strategy-contract fields needed for binding."""

    if contract.program_kind != "strategy":
        raise ValueError("deployment requires a strategy binding contract")
    if contract.parameters:
        raise ValueError("deployment strategy contracts must be role-only")
    return {
        "program_name": contract.program_name,
        "roles": [
            {
                "role": requirement.role,
                "kind": requirement.kind.value,
                "description": requirement.description,
                "min_count": requirement.min_count,
                "max_count": requirement.max_count,
                "allowed_types": list(requirement.allowed_types),
                "required_body_joint_types": list(
                    requirement.required_body_joint_types
                ),
                "same_body_as": requirement.same_body_as,
            }
            for requirement in contract.roles
        ],
        "invariants": list(contract.invariants),
        "relations": [
            {
                "relation": relation.relation.value,
                "left_role": relation.left_role,
                "right_role": relation.right_role,
                "description": relation.description,
            }
            for relation in contract.relations
        ],
    }


def serialize_phase_program(program: object) -> list[dict[str, object]]:
    """Return phase semantics without serializing resolvers or executable code."""

    phases = getattr(program, "phases", None)
    if not isinstance(phases, Sequence):
        raise TypeError("strategy PHASE_PROGRAM must expose a phase sequence")
    result: list[dict[str, object]] = []
    for phase in phases:
        constraints = getattr(phase, "connecting_constraints", ())
        result.append(
            {
                "name": str(getattr(phase, "name")),
                "primitive": str(getattr(phase, "primitive")),
                "role_map": dict(getattr(phase, "role_map")),
                "connecting_constraints": [
                    {
                        "primitive_input": str(constraint.primitive_input),
                        "description": str(constraint.description),
                        "depends_on_roles": list(constraint.depends_on_roles),
                        "output_frame": str(constraint.output_frame),
                    }
                    for constraint in constraints
                ],
            }
        )
    return result


def compact_scene_inventory(
    inventory: Mapping[str, object],
    *,
    required_kinds: set[str],
    type_filters: Mapping[str, set[str]] | None = None,
) -> dict[str, object]:
    """Remove numeric runtime detail that does not help semantic role selection."""

    filters = {
        kind: {_enum_suffix(value) for value in values}
        for kind, values in (type_filters or {}).items()
    }

    def allowed(kind: str, raw_type: object) -> bool:
        accepted = filters.get(kind)
        return not accepted or _enum_suffix(raw_type) in accepted

    joints = {
        str(item.get("name")): {
            "name": str(item.get("name")),
            "type": _enum_suffix(item.get("type")),
        }
        for item in inventory.get("joints", [])
        if isinstance(item, Mapping)
    }
    result: dict[str, object] = {"summary": dict(inventory.get("summary", {}))}
    bodies: list[dict[str, object]] = []
    for raw in inventory.get("bodies", []):
        if not isinstance(raw, Mapping):
            continue
        bodies.append(
            {
                "name": raw.get("name"),
                "parent": raw.get("parent"),
                "joints": [
                    joints.get(str(name), {"name": str(name), "type": "unknown"})
                    for name in raw.get("joints", [])
                ],
            }
        )
    result["bodies"] = bodies

    if "geom" in required_kinds:
        result["geoms"] = [
            {
                "name": raw.get("name"),
                "body": raw.get("body"),
                "type": _enum_suffix(raw.get("type")),
                "size": raw.get("size"),
                "collision": bool(raw.get("contype") or raw.get("conaffinity")),
            }
            for raw in inventory.get("geoms", [])
            if isinstance(raw, Mapping) and allowed("geom", raw.get("type"))
        ]
    if "site" in required_kinds:
        result["sites"] = [
            {
                "name": raw.get("name"),
                "body": raw.get("body"),
                "type": _enum_suffix(raw.get("type")),
            }
            for raw in inventory.get("sites", [])
            if isinstance(raw, Mapping) and allowed("site", raw.get("type"))
        ]
    if "joint" in required_kinds:
        result["joints"] = [
            item for item in joints.values() if allowed("joint", item.get("type"))
        ]
    if "actuator" in required_kinds:
        result["actuators"] = [
            {
                "name": raw.get("name"),
                "transmission_type": _enum_suffix(raw.get("transmission_type")),
                "transmission_targets": raw.get("transmission_targets", []),
            }
            for raw in inventory.get("actuators", [])
            if isinstance(raw, Mapping)
            and allowed("actuator", raw.get("transmission_type"))
        ]
    return result


def extract_strategy_summary(markdown: str, *, maximum_chars: int = 7000) -> str:
    """Keep the strategy semantics relevant to goal and role compatibility."""

    text = markdown.strip()
    if not text:
        return ""
    sections = re.split(r"(?m)^##\s+", text)
    selected = [sections[0].strip()]
    wanted = (
        "inferred goal",
        "final success",
        "plan skeleton",
        "strategy binding roles",
        "known assumptions",
        "unsupported scenes",
    )
    for section in sections[1:]:
        heading = section.splitlines()[0].strip().lower() if section else ""
        if any(term in heading for term in wanted):
            selected.append("## " + section.strip())
    rendered = "\n\n".join(item for item in selected if item)
    return rendered[:maximum_chars]


def parse_deployment_binding_decision(text: str) -> DeploymentBindingDecision:
    """Parse the agent's strict response and reject executable or extra fields."""

    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced is not None:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"deployment response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("deployment response must be a JSON object")
    missing = sorted(_RESPONSE_KEYS - set(payload))
    unknown = sorted(set(payload) - _RESPONSE_KEYS)
    if missing or unknown:
        raise ValueError(
            "deployment response keys are invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != DEPLOY_RESPONSE_SCHEMA_VERSION:
        raise ValueError("deployment response schema_version is invalid")
    compatible = payload["compatible"]
    if not isinstance(compatible, bool):
        raise ValueError("deployment response compatible must be boolean")
    raw_roles = payload["roles"]
    if not isinstance(raw_roles, dict) or not all(
        isinstance(role, str) and role for role in raw_roles
    ):
        raise ValueError("deployment response roles must be a string-keyed object")
    roles: dict[str, str | list[str]] = {}
    for role, value in raw_roles.items():
        if isinstance(value, str) and value:
            roles[role] = value
        elif (
            isinstance(value, list)
            and all(isinstance(item, str) and item for item in value)
            and len(value) == len(set(value))
        ):
            roles[role] = list(value)
        else:
            raise ValueError(
                f"deployment response role {role!r} must be a name or unique name list"
            )
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, list) or not all(
        isinstance(item, str) and item.strip() for item in diagnostics
    ):
        raise ValueError("deployment response diagnostics must be non-empty strings")
    if compatible and diagnostics:
        raise ValueError("a compatible deployment response cannot contain diagnostics")
    if not compatible and (roles or not diagnostics):
        raise ValueError(
            "an incompatible deployment response requires empty roles and diagnostics"
        )
    return DeploymentBindingDecision(
        compatible=compatible,
        roles=roles,
        diagnostics=tuple(diagnostics),
    )


def build_deployment_prompt(
    *,
    strategy: str,
    scene_module: str,
    instance: str,
    goal: str,
    context: Mapping[str, object],
    has_view: bool,
) -> str:
    """Build a self-contained prompt that does not require source inspection."""

    view_clause = (
        "One initial scene view is attached; use it only to resolve semantic ambiguity."
        if has_view
        else "No rendered view is attached; rely on the supplied structural evidence."
    )
    snapshot = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"""
Use `$rapid-deploy-strategy` for this frozen deployment binding.

Read that skill once, then make the binding decision only from this prompt and
the attached view. Do not inspect workspace files or run commands beyond loading
the skill. Return the skill's exact JSON response and nothing else.

- Strategy: `{strategy}`
- Scene module: `{scene_module}`
- Instance: `{instance}`
- Natural-language goal: {goal}
- {view_clause}

Compact deployment evidence:

```json
{snapshot}
```
""".strip()


def build_deployment_correction_prompt(
    findings: Sequence[str],
) -> str:
    """Request the single permitted correction from the existing conversation."""

    rendered = json.dumps(list(findings), ensure_ascii=False)
    return f"""
The host validator rejected the proposed semantic role mapping with these
findings:

{rendered}

Using only the deployment evidence already supplied, return one corrected
`rapid.deploy-binding-response/v1` JSON object. Reconsider every role participating
in a failed relation, including a referenced body role; either endpoint may
be wrong. Keep unrelated assignments fixed and check the corrected mapping
against all connected constraints. Follow the skill's ownership rules rather
than repeating the rejected mapping. If the findings cannot be resolved from
the supplied evidence, return an incompatible response. Do not run tools or
add commentary.
""".strip()


__all__ = [
    "DEPLOY_CONTEXT_RESULT_PREFIX",
    "DEPLOY_RESPONSE_SCHEMA_VERSION",
    "DeploymentBindingDecision",
    "build_deployment_correction_prompt",
    "build_deployment_prompt",
    "compact_scene_inventory",
    "extract_strategy_summary",
    "parse_deployment_binding_decision",
    "serialize_phase_program",
    "serialize_strategy_contract",
]
