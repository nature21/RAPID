"""Frozen typed phase programs for learned multi-primitive strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np
from primitive_core.policy import BoundExecutionContext, GripperOperation

from instantiation_core.contracts import (
    BindingContract,
    SceneBinding,
    validate_binding_parameters,
)
from primitive_core.common import (
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    OptimizationBudget,
)
from primitive_core.transit import TransitSpec


@dataclass(frozen=True)
class PhaseConstraintContext:
    """Live state available while resolving one primitive invocation."""

    model: mujoco.MjModel
    strategy_binding: SceneBinding
    phase_name: str
    phase_index: int
    state: np.ndarray
    controller_state: np.ndarray | None = None
    initial_action: np.ndarray | None = None
    initial_gripper_operation: GripperOperation = GripperOperation.MOVE

    def __post_init__(self) -> None:
        if self.strategy_binding.program_kind != "strategy":
            raise ValueError("phase constraints require a strategy binding")
        if not self.phase_name:
            raise ValueError("phase_name cannot be empty")
        if self.phase_index < 0:
            raise ValueError("phase_index cannot be negative")
        state = np.asarray(self.state, dtype=float).reshape(-1)
        if state.shape != (self.model.nq + self.model.nv,):
            raise ValueError("phase state must contain model.nq + model.nv values")
        if not np.all(np.isfinite(state)):
            raise ValueError("phase state must be finite")
        object.__setattr__(self, "state", state.copy())
        for field_name in ("controller_state", "initial_action"):
            value = getattr(self, field_name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size < 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{field_name} must be a finite vector")
            object.__setattr__(self, field_name, array.copy())
        try:
            operation = GripperOperation(self.initial_gripper_operation)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown initial gripper operation") from exc
        object.__setattr__(self, "initial_gripper_operation", operation)

    def bind_primitive(self, binding: SceneBinding) -> BoundExecutionContext:
        """Restore the live phase state and control handoff for a primitive."""

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.state[: self.model.nq]
        data.qvel[:] = self.state[self.model.nq :]
        mujoco.mj_forward(self.model, data)
        return BoundExecutionContext(
            model=self.model,
            data=data,
            binding=binding,
            initial_state=self.state,
            controller_state=self.controller_state,
            initial_action=self.initial_action,
            initial_gripper_operation=self.initial_gripper_operation,
        )


ConstraintResolver = Callable[[PhaseConstraintContext], Any]


@dataclass(frozen=True)
class ConnectingConstraint:
    """One late-bound relational input for a primitive phase."""

    primitive_input: str
    description: str
    depends_on_roles: tuple[str, ...]
    output_frame: str
    resolver: ConstraintResolver

    def __post_init__(self) -> None:
        if not self.primitive_input:
            raise ValueError("connecting-constraint primitive_input cannot be empty")
        if not self.description:
            raise ValueError("connecting-constraint description cannot be empty")
        if not self.output_frame:
            raise ValueError("connecting-constraint output_frame cannot be empty")
        dependencies = tuple(str(role) for role in self.depends_on_roles)
        if any(not role for role in dependencies):
            raise ValueError("connecting-constraint role names cannot be empty")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("connecting-constraint role names must be unique")
        if not callable(self.resolver):
            raise TypeError("connecting-constraint resolver must be callable")
        object.__setattr__(self, "depends_on_roles", dependencies)

    def resolve(self, context: PhaseConstraintContext) -> Any:
        """Evaluate the learned relation once at the current phase boundary."""

        return self.resolver(context)


@dataclass(frozen=True)
class PrimitivePhase:
    """One locally optimized primitive call with relationally derived inputs."""

    name: str
    primitive: str
    role_map: Mapping[str, str]
    connecting_constraints: tuple[ConnectingConstraint, ...] = ()
    optimization_budget: OptimizationBudget | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.primitive:
            raise ValueError("phase name and primitive cannot be empty")
        role_map = {str(key): str(value) for key, value in self.role_map.items()}
        if any(not key or not value for key, value in role_map.items()):
            raise ValueError("phase role maps cannot contain empty names")
        constraints = tuple(self.connecting_constraints)
        if any(not isinstance(item, ConnectingConstraint) for item in constraints):
            raise TypeError(
                "phase connecting_constraints must contain ConnectingConstraint values"
            )
        if self.optimization_budget is not None and not isinstance(
            self.optimization_budget,
            OptimizationBudget,
        ):
            raise TypeError(
                "phase optimization_budget must be an OptimizationBudget"
            )
        if (
            self.optimization_budget is not None
            and not GLOBAL_MAX_OPTIMIZATION_BUDGET.contains(
                self.optimization_budget
            )
        ):
            raise ValueError(
                "phase optimization_budget exceeds "
                "GLOBAL_MAX_OPTIMIZATION_BUDGET"
            )
        input_names = [item.primitive_input for item in constraints]
        if len(input_names) != len(set(input_names)):
            raise ValueError("phase connecting-constraint inputs must be unique")
        object.__setattr__(self, "role_map", MappingProxyType(role_map))
        object.__setattr__(self, "connecting_constraints", constraints)


@dataclass(frozen=True)
class InterPhaseTransit:
    """Typed transit intent between two named primitive phases."""

    after_phase: str
    before_phase: str
    spec: TransitSpec

    def __post_init__(self) -> None:
        if not self.after_phase or not self.before_phase:
            raise ValueError("transit phase names cannot be empty")
        if not isinstance(self.spec, TransitSpec):
            raise TypeError("inter-phase transit spec must be TransitSpec")


@dataclass(frozen=True)
class StrategyProgram:
    """Frozen phase order, role maps, connecting constraints, and transits."""

    phases: tuple[PrimitivePhase, ...]
    transits: tuple[InterPhaseTransit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "transits", tuple(self.transits))
        if not self.phases:
            raise ValueError("strategy program requires at least one phase")
        if any(not isinstance(phase, PrimitivePhase) for phase in self.phases):
            raise TypeError("strategy phases must contain PrimitivePhase values")
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("strategy phase names must be unique")
        expected_edges = set(zip(names, names[1:], strict=False))
        actual_edges = {
            (transit.after_phase, transit.before_phase)
            for transit in self.transits
        }
        if actual_edges - expected_edges:
            raise ValueError("transits may connect adjacent phases only")
        if len(actual_edges) != len(self.transits):
            raise ValueError("strategy transit edges must be unique")


def validate_strategy_program(
    program: StrategyProgram,
    strategy_contract: BindingContract,
    primitive_contracts: Mapping[str, BindingContract],
    primitive_reference_budgets: Mapping[str, OptimizationBudget] | None = None,
) -> tuple[str, ...]:
    """Validate role-only strategy composition against primitive interfaces."""

    findings: list[str] = []
    if strategy_contract.program_kind != "strategy":
        findings.append("strategy contract must have program_kind='strategy'")
    if strategy_contract.parameters:
        findings.append("strategy binding contracts must declare no parameters")
    strategy_roles = set(strategy_contract.roles_by_name)
    reference_budgets = primitive_reference_budgets
    for phase in program.phases:
        contract = primitive_contracts.get(phase.primitive)
        if contract is None:
            findings.append(
                f"phase {phase.name!r} references missing primitive {phase.primitive!r}"
            )
            continue
        if contract.program_kind != "primitive":
            findings.append(
                f"phase {phase.name!r} contract must have program_kind='primitive'"
            )
            continue
        if reference_budgets is not None:
            reference_budget = reference_budgets.get(phase.primitive)
            if not isinstance(reference_budget, OptimizationBudget):
                findings.append(
                    f"phase {phase.name!r} cannot validate its optimization "
                    f"budget because primitive {phase.primitive!r} has no "
                    "reference budget"
                )
            elif not GLOBAL_MAX_OPTIMIZATION_BUDGET.contains(reference_budget):
                findings.append(
                    f"primitive {phase.primitive!r} reference budget exceeds "
                    "GLOBAL_MAX_OPTIMIZATION_BUDGET"
                )
            elif (
                phase.optimization_budget is not None
                and not phase.optimization_budget.contains(reference_budget)
            ):
                findings.append(
                    f"phase {phase.name!r} optimization budget cannot be "
                    "smaller than its primitive reference budget"
                )
        required_roles = set(contract.roles_by_name)
        if set(phase.role_map) != required_roles:
            findings.append(
                f"phase {phase.name!r} role map must exactly cover "
                f"{sorted(required_roles)}"
            )
        unknown_strategy_roles = sorted(set(phase.role_map.values()) - strategy_roles)
        if unknown_strategy_roles:
            findings.append(
                f"phase {phase.name!r} maps unknown strategy roles "
                f"{unknown_strategy_roles}"
            )

        primitive_inputs = set(contract.parameters_by_name)
        constrained_inputs = {
            constraint.primitive_input
            for constraint in phase.connecting_constraints
        }
        undeclared_inputs = sorted(constrained_inputs - primitive_inputs)
        if undeclared_inputs:
            findings.append(
                f"phase {phase.name!r} constrains undeclared primitive inputs "
                f"{undeclared_inputs}"
            )
        missing_inputs = sorted(primitive_inputs - constrained_inputs)
        if missing_inputs:
            findings.append(
                f"phase {phase.name!r} is missing connecting constraints for "
                f"primitive inputs {missing_inputs}"
            )
        for constraint in phase.connecting_constraints:
            unknown_dependencies = sorted(
                set(constraint.depends_on_roles) - strategy_roles
            )
            if unknown_dependencies:
                findings.append(
                    f"phase {phase.name!r} constraint "
                    f"{constraint.primitive_input!r} depends on unknown strategy "
                    f"roles {unknown_dependencies}"
                )
    return tuple(findings)


def primitive_binding_for_phase(
    phase: PrimitivePhase,
    strategy_binding: SceneBinding,
    primitive_contract: BindingContract,
    *,
    context: PhaseConstraintContext,
) -> SceneBinding:
    """Resolve constraints and construct one ephemeral primitive invocation."""

    if primitive_contract.program_kind != "primitive":
        raise ValueError("phase contract must describe a primitive")
    if primitive_contract.program_name != phase.primitive:
        raise ValueError("phase primitive does not match its binding contract")
    if strategy_binding.parameters:
        raise ValueError("strategy bindings must not contain numeric parameters")
    if context.phase_name != phase.name:
        raise ValueError("constraint context does not match the primitive phase")

    roles: dict[str, str | tuple[str, ...]] = {}
    for primitive_role, strategy_role in phase.role_map.items():
        values = strategy_binding.role_values(strategy_role)
        if not values:
            requirement = primitive_contract.roles_by_name.get(primitive_role)
            if requirement is not None and requirement.min_count == 0:
                continue
            raise ValueError(
                f"strategy binding does not provide phase role {strategy_role!r}"
            )
        roles[primitive_role] = values[0] if len(values) == 1 else values

    inputs: dict[str, Any] = {}
    for constraint in phase.connecting_constraints:
        try:
            inputs[constraint.primitive_input] = constraint.resolve(context)
        except Exception as exc:
            raise ValueError(
                f"phase {phase.name!r} could not resolve primitive input "
                f"{constraint.primitive_input!r}: {exc}"
            ) from exc

    expected_inputs = set(primitive_contract.parameters_by_name)
    actual_inputs = set(inputs)
    if actual_inputs != expected_inputs:
        missing = sorted(expected_inputs - actual_inputs)
        unexpected = sorted(actual_inputs - expected_inputs)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"phase {phase.name!r} connecting constraints do not exactly cover "
            f"the primitive inputs: {', '.join(details)}"
        )

    binding = SceneBinding(
        program_kind="primitive",
        program_name=phase.primitive,
        instance_name=f"{strategy_binding.instance_name}__{phase.name}",
        scene_module=strategy_binding.scene_module,
        goal_text=strategy_binding.goal_text,
        roles=roles,
        parameters=inputs,
    )
    parameter_issues = validate_binding_parameters(primitive_contract, binding)
    if parameter_issues:
        raise ValueError(
            f"phase {phase.name!r} connecting constraints violate the primitive "
            "contract: "
            + "; ".join(issue.message for issue in parameter_issues)
        )
    return binding


__all__ = [
    "ConnectingConstraint",
    "ConstraintResolver",
    "InterPhaseTransit",
    "PrimitivePhase",
    "PhaseConstraintContext",
    "StrategyProgram",
    "primitive_binding_for_phase",
    "validate_strategy_program",
]
