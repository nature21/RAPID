"""Build and validate deployment evidence inside the Codex permission sandbox."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from instantiation_core.contracts import (
    BindingContract,
    RoleRequirement,
    SceneBinding,
    resolve_binding_ownership,
    structural_role_candidates,
    validate_scene_binding,
)
from instantiation_core.scene import inventory_model, load_scene_module
from strategies.program import StrategyProgram

from .deployment_context import (
    DEPLOY_CONTEXT_RESULT_PREFIX,
    compact_scene_inventory,
    extract_strategy_summary,
    serialize_phase_program,
    serialize_strategy_contract,
)
from .instantiation_validation import (
    partition_program_contract_findings,
    validate_program_contract,
)


EXECUTION_POLICY_ENV = "RAPID_EXECUTION_POLICY"


def _load_strategy(strategy: str) -> tuple[object, BindingContract, StrategyProgram]:
    blocking, _advisories = partition_program_contract_findings(
        "strategy",
        validate_program_contract("strategy", strategy),
        execution_policy=os.environ.get(EXECUTION_POLICY_ENV, "strict").strip(),
    )
    if blocking:
        raise ValueError("invalid frozen strategy: " + "; ".join(blocking))
    contract_module = importlib.import_module(f"strategies.{strategy}.binding_contract")
    contract = getattr(contract_module, "BINDING_CONTRACT", None)
    if not isinstance(contract, BindingContract):
        raise TypeError("strategy must publish BINDING_CONTRACT")
    program_module = importlib.import_module(f"strategies.{strategy}.program")
    program = getattr(program_module, "PHASE_PROGRAM", None)
    if not isinstance(program, StrategyProgram):
        raise TypeError("strategy must publish PHASE_PROGRAM")
    return contract_module, contract, program


def build_context(strategy: str, scene_module: str) -> dict[str, object]:
    """Return validated, compact evidence for one model binding decision."""

    _contract_module, contract, program = _load_strategy(strategy)
    scene = load_scene_module(scene_module)
    model = scene.make_model()
    description_path = Path(
        importlib.import_module(f"strategies.{strategy}").__file__ or ""
    ).with_name("strategy_description.md")
    description = (
        description_path.read_text(encoding="utf-8")
        if description_path.is_file()
        else ""
    )
    requirements_by_kind: dict[str, list[RoleRequirement]] = {}
    for requirement in contract.roles:
        requirements_by_kind.setdefault(requirement.kind.value, []).append(
            requirement
        )
    type_filters = {
        kind: {
            allowed_type.lower()
            for requirement in requirements
            for allowed_type in requirement.allowed_types
        }
        for kind, requirements in requirements_by_kind.items()
        if requirements
        and all(requirement.allowed_types for requirement in requirements)
    }
    return {
        "strategy_contract": serialize_strategy_contract(contract),
        "strategy_summary": extract_strategy_summary(description),
        "phases": serialize_phase_program(program),
        "structural_role_candidates": structural_role_candidates(model, contract),
        "scene_inventory": compact_scene_inventory(
            inventory_model(model),
            required_kinds={requirement.kind.value for requirement in contract.roles},
            type_filters=type_filters,
        ),
    }


def validate_roles(
    *,
    strategy: str,
    scene_module: str,
    instance: str,
    goal: str,
    roles: dict[str, object],
) -> dict[str, object]:
    """Validate a proposed role mapping without creating an instantiation package."""

    contract_module, contract, _program = _load_strategy(strategy)
    try:
        binding = SceneBinding(
            program_kind="strategy",
            program_name=strategy,
            instance_name=instance,
            scene_module=scene_module,
            goal_text=goal,
            roles=roles,
            parameters={},
        )
        scene = load_scene_module(scene_module)
        model = scene.make_model()
        binding = resolve_binding_ownership(model, contract, binding)
        report = validate_scene_binding(
            model,
            contract,
            binding,
            invariant_checks=tuple(getattr(contract_module, "INVARIANT_CHECKS", ())),
        )
    except Exception as exc:
        return {"findings": [f"could not validate proposed roles: {exc}"]}
    return {
        "findings": [issue.message for issue in report.issues],
        "roles": {
            role: values[0] if len(values) == 1 else list(values)
            for role in binding.roles
            for values in (binding.role_values(role),)
        },
    }


def dispatch(request: dict[str, Any]) -> dict[str, object]:
    """Dispatch one deployment-context operation."""

    action = request.get("action")
    if action == "context":
        return {
            "context": build_context(
                str(request["strategy"]),
                str(request["scene_module"]),
            )
        }
    if action == "validate_roles":
        roles = request.get("roles")
        if not isinstance(roles, dict):
            raise TypeError("roles must be a JSON object")
        return validate_roles(
            strategy=str(request["strategy"]),
            scene_module=str(request["scene_module"]),
            instance=str(request["instance"]),
            goal=str(request["goal"]),
            roles=roles,
        )
    raise ValueError(f"unknown deployment context action: {action!r}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("deployment context worker expects one JSON request", file=sys.stderr)
        return 2
    try:
        request = json.loads(args[0])
        if not isinstance(request, dict):
            raise TypeError("deployment context request must be an object")
        result = {"ok": True, **dispatch(request)}
    except BaseException as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(DEPLOY_CONTEXT_RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_context", "dispatch", "main", "validate_roles"]
