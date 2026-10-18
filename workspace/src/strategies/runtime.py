"""Runtime helpers for composition strategy phase composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
import time
from typing import Any

import mujoco
import numpy as np

from instantiation_core.contracts import (
    BindingContract,
    SceneBinding,
    validate_scene_binding,
)
from primitive_core.common import CompileResult, ToolTrajectory
from primitive_core.fr3_transit import FR3TransitViolation
from primitive_core.policy import GripperOperation
from primitive_core.result_retention import capture_execution_results, retain_execution_result
from primitive_core.evidence import finalize_primitive_result
from strategies.program import (
    InterPhaseTransit,
    PrimitivePhase,
    PhaseConstraintContext,
    StrategyProgram,
    primitive_binding_for_phase,
    validate_strategy_program,
)


PhaseRunner = Callable[
    [PrimitivePhase, SceneBinding, PhaseConstraintContext],
    CompileResult,
]
TransitRunner = Callable[
    [InterPhaseTransit, PhaseConstraintContext],
    CompileResult,
]


EXECUTION_POLICY_ENV = "RAPID_EXECUTION_POLICY"
STRICT_EXECUTION_POLICY = "strict"
ATTEMPT_EXECUTION_POLICY = "attempt"
EXECUTION_POLICIES = (STRICT_EXECUTION_POLICY, ATTEMPT_EXECUTION_POLICY)


def _execution_policy() -> str:
    policy = os.environ.get(EXECUTION_POLICY_ENV, STRICT_EXECUTION_POLICY).strip()
    if policy not in EXECUTION_POLICIES:
        raise ValueError(
            f"{EXECUTION_POLICY_ENV} must be one of {EXECUTION_POLICIES}, got "
            f"{policy!r}"
        )
    return policy


def _recoverable_transit_result(
    exc: Exception,
    *,
    model: mujoco.MjModel,
    failed_name: str,
) -> CompileResult | None:
    """Return a finite rejected transit that attempt-mode may propagate.

    Transit planning and certification failures sometimes retain the physical
    controller rollout completed before the rejection.  Benchmark attempt
    mode may use that realized terminal state to give the next learned phase a
    chance to run.  Contract errors and failures without a concrete trajectory
    remain fatal.
    """

    if not isinstance(exc, FR3TransitViolation):
        return None
    result = getattr(exc, "result", None)
    if not isinstance(result, CompileResult):
        return None
    try:
        states = np.asarray(result.best_states, dtype=float)
        controls = np.asarray(result.trajectory.controls, dtype=float)
        positions = np.asarray(result.trajectory.positions, dtype=float)
        times = np.asarray(result.trajectory.times, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] != model.nq + model.nv
        or controls.ndim != 2
        or controls.shape[0] < 1
        or controls.shape[0] != states.shape[0]
        or positions.shape[0] != states.shape[0]
        or times.shape != (states.shape[0],)
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(controls))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(times))
    ):
        return None
    if result.controller_state is not None and not np.all(
        np.isfinite(np.asarray(result.controller_state, dtype=float))
    ):
        return None
    operations = result.trajectory.gripper_operations
    if operations is not None and np.asarray(operations).shape != (
        states.shape[0],
    ):
        return None
    result.metrics["attempt_continuation"] = {
        "accepted": True,
        "failed_segment": failed_name,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "retained_state_count": int(states.shape[0]),
    }
    return result


def _retain_partial_strategy_failure(
    exc: Exception,
    completed_results: Sequence[CompileResult],
    completed_names: Sequence[str],
    *,
    failed_name: str,
    dt: float,
    resolved_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Attach one beginning-to-failure trajectory to a strategy exception."""

    retained_results = list(completed_results)
    retained_names = list(completed_names)
    rejected = getattr(exc, "result", None)
    if isinstance(rejected, CompileResult):
        retained_results.append(rejected)
        retained_names.append(f"{failed_name}__rejected")
    if not retained_results:
        return
    try:
        partial = concat_phase_results(
            retained_results,
            dt,
            phase_names=retained_names,
        )
    except Exception as retention_error:
        exc.add_note(
            "retaining the beginning-to-failure strategy trajectory failed: "
            f"{retention_error}"
        )
        return
    attempt_continuations = [
        dict(item)
        for result in completed_results
        if isinstance(result.metrics.get("attempt_continuation"), Mapping)
        for item in (result.metrics["attempt_continuation"],)
    ]
    partial.metrics.update(
        {
            "success": False,
            "partial_strategy": True,
            "completed_segments": list(completed_names),
            "retained_segments": retained_names,
            "failed_segment": failed_name,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "resolved_primitive_inputs": {
                name: dict(values) for name, values in resolved_inputs.items()
            },
            "execution_policy": _execution_policy(),
            "attempt_continuations": attempt_continuations,
            "strict_would_abort": bool(attempt_continuations),
        }
    )
    setattr(exc, "result", partial)


def _result_controls(result: CompileResult) -> np.ndarray:
    controls = np.asarray(result.trajectory.controls, dtype=float)
    if controls.ndim != 2:
        raise ValueError(
            f"phase controls must be 2-D, got shape {controls.shape}."
        )
    return controls


def phase_final_state(result: CompileResult) -> np.ndarray:
    """Return a copy of the final physical state from a phase result."""

    states = np.asarray(result.best_states, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("phase result has no state trajectory.")
    return states[-1].copy()


def _result_final_action(result: CompileResult) -> np.ndarray:
    controls = _result_controls(result)
    if controls.shape[0] == 0:
        raise ValueError("phase result has no control trajectory.")
    return controls[-1].copy()


def _result_final_gripper_operation(result: CompileResult) -> GripperOperation:
    operations = result.trajectory.gripper_operations
    if operations is None:
        return GripperOperation.MOVE
    if operations.size == 0:
        raise ValueError("FR3 phase result has an empty gripper operation stream.")
    return GripperOperation(int(operations[-1]))


def _result_controller_state(result: CompileResult) -> np.ndarray | None:
    if result.controller_state is None:
        return None
    controller = np.asarray(result.controller_state, dtype=float).reshape(-1)
    if controller.size < 1 or not np.all(np.isfinite(controller)):
        raise ValueError("phase controller state must be a finite vector.")
    return controller.copy()


def concat_phase_results(
    phases: Sequence[CompileResult],
    dt: float,
    *,
    phase_names: Sequence[str] | None = None,
) -> CompileResult:
    """Concatenate optimized and simulated phase results."""

    phase_list = list(phases)
    if not phase_list:
        raise ValueError("at least one phase result is required.")
    if phase_names is not None and len(phase_names) != len(phase_list):
        raise ValueError("phase_names length must match phases length.")

    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    operations: list[np.ndarray | None] = []
    cost_history: list[np.ndarray] = []
    phase_metrics: dict[str, dict[str, Any]] = {}

    for idx, phase in enumerate(phase_list):
        phase_states = np.asarray(phase.best_states, dtype=float)
        phase_controls = _result_controls(phase)
        if phase_states.ndim != 2 or phase_states.shape[0] == 0:
            raise ValueError(f"phase {idx} has no state trajectory.")
        if phase_controls.shape[0] != phase_states.shape[0]:
            raise ValueError(
                f"phase {idx} has {phase_states.shape[0]} states but "
                f"{phase_controls.shape[0]} controls."
            )
        name = (
            phase_names[idx]
            if phase_names is not None
            else str(phase.metrics.get("phase") or f"phase_{idx}")
        )
        key = name if name not in phase_metrics else f"{name}_{idx}"
        phase_metrics[key] = dict(phase.metrics)
        states.append(phase_states)
        controls.append(phase_controls)
        phase_operations = phase.trajectory.gripper_operations
        if phase_operations is not None and phase_operations.shape != (
            phase_controls.shape[0],
        ):
            raise ValueError(f"phase {idx} has a misaligned operation stream.")
        operations.append(phase_operations)
        cost_history.append(np.asarray(phase.cost_history, dtype=float).reshape(-1))

    merged_states = np.concatenate(states, axis=0)
    merged_controls = np.concatenate(controls, axis=0)
    if any(value is None for value in operations) and any(
        value is not None for value in operations
    ):
        raise ValueError("strategy phases must consistently provide operations.")
    merged_operations = (
        None
        if operations[0] is None
        else np.concatenate([value for value in operations if value is not None])
    )
    phase_boundary_aperture_deltas = [
        abs(float(controls[index + 1][0, 6] - controls[index][-1, 6]))
        for index in range(len(controls) - 1)
        if controls[index].shape[1] >= 7 and controls[index + 1].shape[1] >= 7
    ]
    discontinuous_aperture_boundaries = [
        (index, delta)
        for index, delta in enumerate(phase_boundary_aperture_deltas)
        if delta > 1e-12
    ]
    if discontinuous_aperture_boundaries:
        raise ValueError(
            "strategy phases must preserve the terminal gripper command; "
            f"discontinuous boundaries: {discontinuous_aperture_boundaries}"
        )
    times = np.arange(merged_states.shape[0], dtype=float) * float(dt)
    nonempty_costs = [history for history in cost_history if history.size]
    merged_cost_history = (
        np.concatenate(nonempty_costs, axis=0)
        if nonempty_costs
        else np.empty((0,), dtype=float)
    )

    return CompileResult(
        trajectory=ToolTrajectory(
            times=times,
            positions=merged_controls.copy(),
            controls=merged_controls.copy(),
            gripper_operations=merged_operations,
        ),
        metrics={
            "num_phases": len(phase_list),
            "num_states": int(merged_states.shape[0]),
            "dt": float(dt),
            "phase_metrics": phase_metrics,
            "phase_boundary_aperture_deltas": phase_boundary_aperture_deltas,
            "maximum_phase_boundary_aperture_delta": max(
                phase_boundary_aperture_deltas,
                default=0.0,
            ),
        },
        best_states=merged_states,
        cost_history=merged_cost_history,
        initial_pose=dict(phase_list[0].initial_pose),
        controller_state=(
            None
            if phase_list[-1].controller_state is None
            else np.asarray(phase_list[-1].controller_state, dtype=float).copy()
        ),
    )


def execute_sequential_strategy(
    *,
    program: StrategyProgram,
    strategy_contract: BindingContract,
    primitive_contracts: Mapping[str, BindingContract],
    strategy_binding: SceneBinding,
    model: mujoco.MjModel,
    initial_state: np.ndarray,
    run_phase: PhaseRunner,
    dt: float,
    run_transit: TransitRunner | None = None,
    initial_controller_state: np.ndarray | None = None,
    initial_action: np.ndarray | None = None,
    initial_gripper_operation: GripperOperation = GripperOperation.MOVE,
    primitive_invariant_checks: Mapping[str, Sequence[Callable[..., Any]]] | None = None,
) -> CompileResult:
    """Resolve and execute a strategy as sequential local primitive problems.

    Connecting constraints are evaluated exactly once at each live phase
    boundary. The retained physical/controller state and terminal action from
    one phase or transit become the starting context for the next phase.
    """

    if not callable(run_phase):
        raise TypeError("run_phase must be callable")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    execution_policy = _execution_policy()
    findings = validate_strategy_program(
        program,
        strategy_contract,
        primitive_contracts,
    )
    if findings:
        raise ValueError("invalid strategy program: " + "; ".join(findings))
    validate_scene_binding(model, strategy_contract, strategy_binding).require_valid()

    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if state.shape != (model.nq + model.nv,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite model.nq + model.nv vector")
    state = state.copy()
    controller_state = (
        None
        if initial_controller_state is None
        else np.asarray(initial_controller_state, dtype=float).reshape(-1).copy()
    )
    action = (
        None
        if initial_action is None
        else np.asarray(initial_action, dtype=float).reshape(-1).copy()
    )
    operation = GripperOperation(initial_gripper_operation)
    if controller_state is not None and not np.all(np.isfinite(controller_state)):
        raise ValueError("initial_controller_state must be finite")
    if action is not None and (action.size < 1 or not np.all(np.isfinite(action))):
        raise ValueError("initial_action must be a finite vector")

    transits = {
        (transit.after_phase, transit.before_phase): transit
        for transit in program.transits
    }
    results: list[CompileResult] = []
    result_names: list[str] = []
    resolved_inputs: dict[str, dict[str, Any]] = {}
    primitive_phase_effect_success: dict[str, bool] = {}
    primitive_phase_success: dict[str, bool] = {}
    attempt_continuations: list[dict[str, Any]] = []
    invariant_checks = primitive_invariant_checks or {}
    from instantiation_core.handoff import (
        capture_boundary, current_replay, restore_boundary, HandoffComplete,
    )
    replay = current_replay()
    checkpoint_findings: list[str] = []

    for phase_index, phase in enumerate(program.phases):
        if replay is not None:
            if phase.name != replay["metadata"]["phase"]:
                continue
            state, controller_state, action, operation = restore_boundary(replay, model, strategy_binding)
        if phase_index > 0:
            try:
                capture_boundary(strategy_binding, phase.name, state, controller_state, action, operation, model=model)
            except (OSError, ValueError) as exc:
                checkpoint_findings.append(f"{phase.name}: {type(exc).__name__}: {exc}")
        if phase_index > 0:
            previous = program.phases[phase_index - 1]
            transit = transits.get((previous.name, phase.name))
            if transit is not None:
                if run_transit is None:
                    raise ValueError(
                        f"strategy transit {previous.name!r} -> {phase.name!r} "
                        "requires run_transit"
                    )
                transit_context = PhaseConstraintContext(
                    model=model,
                    strategy_binding=strategy_binding,
                    phase_name=phase.name,
                    phase_index=phase_index,
                    state=state,
                    controller_state=controller_state,
                    initial_action=action,
                    initial_gripper_operation=operation,
                )
                failed_name = f"{previous.name}__to__{phase.name}"
                try:
                    transit_result = run_transit(transit, transit_context)
                    if not isinstance(transit_result, CompileResult):
                        raise TypeError("run_transit must return CompileResult")
                except Exception as exc:
                    transit_result = (
                        _recoverable_transit_result(
                            exc,
                            model=model,
                            failed_name=failed_name,
                        )
                        if execution_policy == ATTEMPT_EXECUTION_POLICY
                        else None
                    )
                    if transit_result is None:
                        _retain_partial_strategy_failure(
                            exc,
                            results,
                            result_names,
                            failed_name=failed_name,
                            dt=dt,
                            resolved_inputs=resolved_inputs,
                        )
                        raise
                    attempt_continuations.append(
                        dict(transit_result.metrics["attempt_continuation"])
                    )
                results.append(transit_result)
                result_names.append(failed_name)
                state = phase_final_state(transit_result)
                controller_state = _result_controller_state(transit_result)
                action = _result_final_action(transit_result)
                operation = _result_final_gripper_operation(transit_result)

        context = PhaseConstraintContext(
            model=model,
            strategy_binding=strategy_binding,
            phase_name=phase.name,
            phase_index=phase_index,
            state=state,
            controller_state=controller_state,
            initial_action=action,
            initial_gripper_operation=operation,
        )
        phase_started = time.perf_counter()
        with capture_execution_results() as phase_results:
            try:
                primitive_contract = primitive_contracts[phase.primitive]
                primitive_binding = primitive_binding_for_phase(
                    phase, strategy_binding, primitive_contract, context=context,
                )
                validate_scene_binding(
                    model, primitive_contract, primitive_binding,
                    invariant_checks=tuple(invariant_checks.get(phase.primitive, ())),
                ).require_valid()
                resolved_inputs[phase.name] = dict(primitive_binding.parameters)
                phase_result = run_phase(phase, primitive_binding, context)
                if not isinstance(phase_result, CompileResult):
                    raise TypeError("run_phase must return CompileResult")
                retain_execution_result(phase_result)
                if "optimized_success" not in phase_result.metrics:
                    raise ValueError(f"primitive phase {phase.name!r} did not report optimized_success")
                finalize_primitive_result(phase_result)
                next_state = phase_final_state(phase_result)
                next_controller = _result_controller_state(phase_result)
                next_action = _result_final_action(phase_result)
                next_operation = _result_final_gripper_operation(phase_result)
            except Exception as exc:
                if not isinstance(getattr(exc, "result", None), CompileResult) and phase_results:
                    exc.result = phase_results[-1]
                if isinstance(getattr(exc, "result", None), CompileResult):
                    exc.result.metrics["phase_wall_seconds"] = time.perf_counter() - phase_started
                _retain_partial_strategy_failure(
                    exc,
                    results,
                    result_names,
                    failed_name=phase.name,
                    dt=dt,
                    resolved_inputs=resolved_inputs,
                )
                raise
        phase_result.metrics["phase_wall_seconds"] = time.perf_counter() - phase_started
        primitive_phase_effect_success[phase.name] = bool(
            phase_result.metrics.get("effect_success", False)
        )
        primitive_phase_success[phase.name] = phase_result.metrics["success"]
        results.append(phase_result)
        result_names.append(phase.name)
        state = next_state
        controller_state = next_controller
        action = next_action
        operation = next_operation
        if replay is not None:
            raise HandoffComplete(concat_phase_results(results, dt, phase_names=result_names))

    combined = concat_phase_results(results, dt, phase_names=result_names)
    combined.metrics["primitive_phase_order"] = [
        phase.name for phase in program.phases
    ]
    combined.metrics["handoff_checkpoint_findings"] = checkpoint_findings
    combined.metrics["execution_policy"] = execution_policy
    combined.metrics["attempt_continuations"] = attempt_continuations
    combined.metrics["strict_would_abort"] = bool(attempt_continuations)
    combined.metrics["local_primitive_optimization_count"] = len(program.phases)
    combined.metrics["resolved_primitive_inputs"] = resolved_inputs
    combined.metrics["primitive_phase_effect_success"] = (
        primitive_phase_effect_success
    )
    combined.metrics["all_primitive_phase_effects_success"] = all(
        primitive_phase_effect_success.values()
    )
    combined.metrics["primitive_phase_success"] = primitive_phase_success
    combined.metrics["all_primitive_phases_success"] = all(primitive_phase_success.values())
    retain_execution_result(combined)
    return combined


__all__ = [
    "ATTEMPT_EXECUTION_POLICY",
    "EXECUTION_POLICIES",
    "EXECUTION_POLICY_ENV",
    "PhaseRunner",
    "STRICT_EXECUTION_POLICY",
    "TransitRunner",
    "concat_phase_results",
    "execute_sequential_strategy",
    "phase_final_state",
]
