"""Strict, data-only scene binding manifests."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    BindingContract,
    SceneBinding,
    validate_binding_parameters,
)


BINDING_SCHEMA_VERSION = "rapid.binding/v1"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "program_kind",
        "program_name",
        "instance_name",
        "scene_module",
        "goal_text",
        "roles",
        "inputs",
    }
)

# Published for tooling and editor integration. Loading also performs these
# checks directly so validation has no optional jsonschema dependency.
BINDING_MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": BINDING_SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_TOP_LEVEL_KEYS),
    "properties": {
        "schema_version": {"const": BINDING_SCHEMA_VERSION},
        "program_kind": {"enum": ["primitive", "strategy"]},
        "program_name": {"type": "string", "minLength": 1},
        "instance_name": {"type": "string", "minLength": 1},
        "scene_module": {"type": "string", "minLength": 1},
        "goal_text": {"type": "string", "minLength": 1},
        "roles": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                ]
            },
        },
        "inputs": {"type": "object"},
    },
}


class ManifestError(ValueError):
    """Raised before simulation when a binding manifest is not data-only."""


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"binding manifest {key!r} must be a non-empty string")
    return value


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(
                f"binding manifest value at {path} must be finite"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ManifestError(f"binding manifest value at {path} is not valid JSON data")


def parse_binding_manifest(
    payload: object,
    *,
    contract: BindingContract | None = None,
) -> SceneBinding:
    """Parse a strict v1 manifest without importing executable binding code."""

    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ManifestError("binding manifest must be a JSON object")
    keys = set(payload)
    missing = sorted(_TOP_LEVEL_KEYS - keys)
    unknown = sorted(keys - _TOP_LEVEL_KEYS)
    if missing:
        raise ManifestError(f"binding manifest is missing keys: {missing}")
    if unknown:
        raise ManifestError(f"binding manifest has unknown keys: {unknown}")
    if payload["schema_version"] != BINDING_SCHEMA_VERSION:
        raise ManifestError(
            f"binding manifest schema_version must be {BINDING_SCHEMA_VERSION!r}"
        )
    kind = payload["program_kind"]
    if kind not in {"primitive", "strategy"}:
        raise ManifestError("binding manifest program_kind is invalid")
    roles = payload["roles"]
    if not isinstance(roles, dict) or not all(
        isinstance(name, str) and name for name in roles
    ):
        raise ManifestError("binding manifest roles must be a string-keyed object")
    normalized_roles: dict[str, str | tuple[str, ...]] = {}
    for role, value in roles.items():
        if isinstance(value, str) and value:
            normalized_roles[role] = value
        elif (
            isinstance(value, list)
            and all(isinstance(item, str) and item for item in value)
            and len(value) == len(set(value))
        ):
            normalized_roles[role] = tuple(value)
        else:
            raise ManifestError(
                f"binding manifest role {role!r} must be a name or unique name list"
            )
    inputs = payload["inputs"]
    if not isinstance(inputs, dict) or not all(
        isinstance(name, str) and name for name in inputs
    ):
        raise ManifestError("binding manifest inputs must be a string-keyed object")
    _validate_json_value(inputs, path="inputs")
    binding = SceneBinding(
        program_kind=kind,
        program_name=_require_text(payload, "program_name"),
        instance_name=_require_text(payload, "instance_name"),
        scene_module=_require_text(payload, "scene_module"),
        goal_text=_require_text(payload, "goal_text"),
        roles=normalized_roles,
        parameters=inputs,
    )
    if contract is not None:
        if binding.program_kind != contract.program_kind:
            raise ManifestError("manifest program_kind does not match the contract")
        if binding.program_name != contract.program_name:
            raise ManifestError("manifest program_name does not match the contract")
        declared = contract.parameters_by_name
        undeclared = sorted(set(binding.parameters) - set(declared))
        missing_inputs = sorted(
            requirement.name
            for requirement in contract.parameters
            if requirement.required and requirement.name not in binding.parameters
        )
        if undeclared:
            raise ManifestError(f"manifest has undeclared inputs: {undeclared}")
        if missing_inputs:
            raise ManifestError(
                f"manifest is missing required inputs: {missing_inputs}"
            )
        parameter_issues = validate_binding_parameters(contract, binding)
        if parameter_issues:
            raise ManifestError(
                "manifest primitive parameters violate the learned contract: "
                + "; ".join(issue.message for issue in parameter_issues)
            )
    return binding


def load_binding_manifest(
    path: str | Path,
    *,
    contract: BindingContract | None = None,
) -> SceneBinding:
    manifest_path = Path(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(
                    f"binding manifest contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read binding manifest {manifest_path}: {exc}") from exc
    return parse_binding_manifest(payload, contract=contract)


__all__ = [
    "BINDING_MANIFEST_SCHEMA",
    "BINDING_SCHEMA_VERSION",
    "ManifestError",
    "load_binding_manifest",
    "parse_binding_manifest",
]
