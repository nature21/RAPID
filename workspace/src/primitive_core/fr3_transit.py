"""Deterministic native-OSC transit before an optimized FR3 interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from primitive_core.common import CompileResult, ToolTrajectory
from primitive_core.fr3_osc import (
    FR3OSCAdapter,
    fr3_controller_state_from_observation,
)
from primitive_core.fr3_safety import DEFAULT_FR3_SAFETY_LIMITS
from primitive_core.native_fr3_osc import NativeFR3OSCRolloutBackend
from primitive_core.policy import GripperOperation
from primitive_core.result_retention import retain_execution_result


class FR3TransitViolation(RuntimeError):
    """A typed transit timed out or failed its semantic/runtime audit."""

    def __init__(
        self,
        message: str,
        *,
        result: CompileResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class FR3TransitResult:
    """Physical transit trajectory plus state needed by the next OSC rollout."""

    result: CompileResult
    controller_observations: np.ndarray
    controller_state: np.ndarray

    @property
    def terminal_state(self) -> np.ndarray:
        return np.asarray(self.result.best_states[-1], dtype=float).copy()

    @property
    def terminal_action(self) -> np.ndarray:
        return np.asarray(self.result.trajectory.controls[-1], dtype=float).copy()

    @property
    def terminal_gripper_operation(self) -> GripperOperation:
        operations = self.result.trajectory.gripper_operations
        if operations is None or operations.size == 0:
            raise ValueError("FR3 transit has no gripper operation stream")
        return GripperOperation(int(operations[-1]))


def _state_vector(model: mujoco.MjModel, state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=float).reshape(-1)
    expected = model.nq + model.nv
    if value.shape != (expected,):
        raise ValueError(f"initial state must have shape ({expected},)")
    if not np.all(np.isfinite(value)):
        raise ValueError("initial state must be finite")
    return value.copy()


def _unit_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm <= 1e-12:
        raise ValueError("quaternion must be a nonzero four-vector")
    return quaternion / norm


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    left = _unit_quaternion(first)
    right = _unit_quaternion(second)
    dot = float(left @ right)
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _unit_quaternion((1.0 - fraction) * left + fraction * right)
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return (
        np.sin((1.0 - fraction) * angle) / denominator * left
        + np.sin(fraction * angle) / denominator * right
    )


def _smooth_actions(
    adapter: FR3OSCAdapter,
    waypoints: np.ndarray,
    segment_durations: np.ndarray,
    timestep: float,
    settle_maximum_duration: float,
) -> np.ndarray:
    segments: list[np.ndarray] = [waypoints[0][None].copy()]
    for index, duration in enumerate(segment_durations):
        steps = max(1, int(np.ceil(float(duration) / timestep)))
        fractions = np.arange(1, steps + 1, dtype=float) / steps
        blends = fractions**3 * (10.0 + fractions * (-15.0 + 6.0 * fractions))
        start_position, start_quaternion = adapter.target_pose_world(waypoints[index])
        end_position, end_quaternion = adapter.target_pose_world(
            waypoints[index + 1]
        )
        gripper_start = float(waypoints[index, 6])
        gripper_end = float(waypoints[index + 1, 6])
        controls = np.empty((steps, adapter.action_dim), dtype=float)
        for step, blend in enumerate(blends):
            controls[step] = adapter.action_from_world_pose(
                (1.0 - blend) * start_position + blend * end_position,
                _slerp(start_quaternion, end_quaternion, float(blend)),
                gripper=(1.0 - blend) * gripper_start + blend * gripper_end,
            )
        segments.append(controls)
    settle_steps = int(np.ceil(settle_maximum_duration / timestep))
    if settle_steps:
        segments.append(np.tile(waypoints[-1], (settle_steps, 1)))
    return np.ascontiguousarray(np.concatenate(segments, axis=0))


def _aperture_diagnostics(
    model: mujoco.MjModel,
    adapter: FR3OSCAdapter,
    controls: np.ndarray,
    states: np.ndarray,
) -> dict[str, float | list[float]]:
    commanded = np.asarray(controls[:, 6], dtype=float)
    actual = adapter.gripper_apertures(model, states)
    return {
        "commanded_aperture_range": [
            float(np.min(commanded)),
            float(np.max(commanded)),
        ],
        "actual_aperture_range": [float(np.min(actual)), float(np.max(actual))],
        "commanded_aperture_total_variation": float(
            np.sum(np.abs(np.diff(commanded)))
        ),
        "actual_aperture_total_variation": float(
            np.sum(np.abs(np.diff(actual)))
        ),
        "maximum_aperture_tracking_error": float(
            np.max(np.abs(commanded - actual))
        ),
    }


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> float:
    error = np.empty(3, dtype=float)
    mujoco.mju_subQuat(error, _unit_quaternion(target), _unit_quaternion(actual))
    return float(np.linalg.norm(error))


def rollout_fr3_transit(
    model: mujoco.MjModel,
    initial_state: np.ndarray,
    adapter: FR3OSCAdapter,
    waypoint_actions: np.ndarray,
    segment_durations: Sequence[float],
    *,
    initial_controller_state: np.ndarray | None = None,
    initial_command_action: np.ndarray | None = None,
    gripper_operation: GripperOperation = GripperOperation.MOVE,
    settle_duration: float = 0.25,
    settle_maximum_duration: float | None = None,
    settle_stable_duration: float = 0.10,
    terminal_position_tolerance: float = 0.01,
    terminal_orientation_tolerance: float = 0.10,
    terminal_linear_speed_tolerance: float = 0.10,
    terminal_angular_speed_tolerance: float = 0.50,
    minimum_joint_margin: float = 0.05,
    label: str = "precontact_transit",
) -> FR3TransitResult:
    """Simulate one effect-free transit outside CEM/MPPI.

    The first waypoint must be the inherited command target, or the measured
    reset action when no preceding command exists. The remaining waypoints
    should be relationally derived collision-free clearance poses, ending at
    precontact. Learned programs should normally call the higher-level
    typed ``primitive_core.transit.execute_transit`` path, which centrally
    realizes stages and audits protected objects and contacts.
    """

    x0 = _state_vector(model, initial_state)
    waypoints = np.ascontiguousarray(waypoint_actions, dtype=np.float64)
    durations = np.asarray(tuple(segment_durations), dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != adapter.action_dim:
        raise ValueError("waypoint_actions must have shape (num_waypoints, 7)")
    if waypoints.shape[0] < 2:
        raise ValueError("transit requires at least two waypoints")
    if durations.shape != (waypoints.shape[0] - 1,):
        raise ValueError("segment_durations must contain one value per segment")
    if not np.all(np.isfinite(waypoints)) or not np.all(np.isfinite(durations)):
        raise ValueError("transit waypoints and durations must be finite")
    maximum_settle = (
        float(settle_duration) + 1.0
        if settle_maximum_duration is None
        else float(settle_maximum_duration)
    )
    if np.any(durations <= 0.0) or not np.isfinite(settle_duration):
        raise ValueError("transit durations must be positive and finite")
    if settle_duration < 0.0:
        raise ValueError("settle_duration cannot be negative")
    if not np.isfinite(maximum_settle) or maximum_settle < settle_duration:
        raise ValueError(
            "settle_maximum_duration must be finite and at least settle_duration"
        )
    if not np.isfinite(settle_stable_duration) or settle_stable_duration <= 0.0:
        raise ValueError("settle_stable_duration must be positive and finite")
    if maximum_settle < settle_stable_duration:
        raise ValueError(
            "settle_maximum_duration must cover the stable-duration window"
        )
    control_range = adapter.action_ctrlrange
    if np.any(waypoints < control_range[:, 0]) or np.any(
        waypoints > control_range[:, 1]
    ):
        raise ValueError("transit waypoint exceeds the FR3 action range")

    data = mujoco.MjData(model)
    data.qpos[:] = x0[: model.nq]
    data.qvel[:] = x0[model.nq :]
    mujoco.mj_forward(model, data)
    measured_start = adapter.action_from_data(model, data)
    expected_start = (
        measured_start
        if initial_command_action is None
        else np.asarray(initial_command_action, dtype=float)
    )
    if expected_start.shape != (adapter.action_dim,) or not np.all(
        np.isfinite(expected_start)
    ):
        raise ValueError("initial_command_action has the wrong shape or values")
    first_position, first_quaternion = adapter.target_pose_world(waypoints[0])
    expected_position, expected_quaternion = adapter.target_pose_world(expected_start)
    if float(np.linalg.norm(first_position - expected_position)) > 1e-6 or (
        _orientation_error(first_quaternion, expected_quaternion) > 1e-6
    ) or abs(float(waypoints[0, 6] - expected_start[6])) > 1e-6:
        raise ValueError(
            "first transit waypoint must equal the inherited command target"
        )

    timestep = float(model.opt.timestep)
    controls = _smooth_actions(
        adapter,
        waypoints,
        durations,
        timestep,
        maximum_settle,
    )
    try:
        operation = GripperOperation(gripper_operation)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown transit gripper operation") from exc
    if operation is GripperOperation.GRASP and not np.allclose(
        controls[:, 6], controls[0, 6], rtol=0.0, atol=1e-12
    ):
        raise ValueError("GRASP transit must preserve a constant aperture command")
    gripper_operations = np.full(
        controls.shape[0],
        int(operation),
        dtype=np.uint8,
    )
    moving_steps = int(
        sum(max(1, int(np.ceil(float(duration) / timestep))) for duration in durations)
    ) + 1
    backend = NativeFR3OSCRolloutBackend(
        model,
        num_threads=1,
        adapter=adapter,
        initial_controller_state=initial_controller_state,
        gripper_operations=gripper_operations,
    )
    try:
        output_states, output_sensors, _ = backend.rollout(
            x0[None],
            controls[None],
        )
    finally:
        backend.close()
    states = np.asarray(output_states[0], dtype=float)
    observation_dim = len(adapter.observation_names)
    observations = np.asarray(output_sensors[0, :, -observation_dim:], dtype=float)

    target_position, target_quaternion = adapter.target_pose_world(waypoints[-1])
    position_errors = np.linalg.norm(
        observations[:, :3] - target_position[None],
        axis=1,
    )
    orientation_errors = np.asarray(
        [
            _orientation_error(target_quaternion, observation[3:7])
            for observation in observations
        ],
        dtype=float,
    )
    linear_speeds = np.linalg.norm(observations[:, 7:10], axis=1)
    angular_speeds = np.linalg.norm(observations[:, 10:13], axis=1)
    actual_apertures = adapter.gripper_apertures(model, states)
    actual_aperture_speeds = np.abs(
        adapter.gripper_aperture_speeds(model, states)
    )
    terminal_gripper_position_tolerance = 0.05 * (
        adapter.gripper_range[1] - adapter.gripper_range[0]
    )
    terminal_gripper_speed_tolerance = (
        0.10 * DEFAULT_FR3_SAFETY_LIMITS.nominal_gripper_velocity
    )
    gripper_position_errors = np.abs(actual_apertures - waypoints[-1, 6])
    converged_steps = (
        (position_errors <= terminal_position_tolerance)
        & (orientation_errors <= terminal_orientation_tolerance)
        & (linear_speeds <= terminal_linear_speed_tolerance)
        & (angular_speeds <= terminal_angular_speed_tolerance)
        & (gripper_position_errors <= terminal_gripper_position_tolerance)
        & (actual_aperture_speeds <= terminal_gripper_speed_tolerance)
        & np.all(np.isfinite(observations), axis=1)
        & np.all(np.isfinite(states), axis=1)
    )
    minimum_settle_steps = int(np.ceil(settle_duration / timestep))
    stable_steps = max(1, int(np.ceil(settle_stable_duration / timestep)))
    earliest_end = max(
        moving_steps + minimum_settle_steps,
        moving_steps + stable_steps,
    )
    retained_steps: int | None = None
    for end in range(earliest_end, controls.shape[0] + 1):
        if np.all(converged_steps[end - stable_steps : end]):
            retained_steps = end
            break
    settled = retained_steps is not None
    if retained_steps is None:
        retained_steps = controls.shape[0]
    controls = controls[:retained_steps]
    states = states[:retained_steps]
    observations = observations[:retained_steps]

    terminal_position_error = float(
        np.linalg.norm(observations[-1, :3] - target_position)
    )
    terminal_orientation_error = _orientation_error(
        target_quaternion,
        observations[-1, 3:7],
    )
    terminal_linear_speed = float(np.linalg.norm(observations[-1, 7:10]))
    terminal_angular_speed = float(np.linalg.norm(observations[-1, 10:13]))
    terminal_gripper_position_error = abs(
        float(actual_apertures[retained_steps - 1] - waypoints[-1, 6])
    )
    terminal_gripper_speed = abs(
        float(actual_aperture_speeds[retained_steps - 1])
    )
    observed_minimum_joint_margin = float(np.min(observations[:, 26]))
    transit_convergence_valid = bool(
        settled
        and terminal_position_error <= terminal_position_tolerance
        and terminal_orientation_error <= terminal_orientation_tolerance
        and terminal_linear_speed <= terminal_linear_speed_tolerance
        and terminal_angular_speed <= terminal_angular_speed_tolerance
        and terminal_gripper_position_error <= terminal_gripper_position_tolerance
        and terminal_gripper_speed <= terminal_gripper_speed_tolerance
        and observed_minimum_joint_margin >= minimum_joint_margin
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(observations))
    )
    transit_valid = bool(
        np.all(np.isfinite(controls))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(observations))
    )
    terminal_controller_state = fr3_controller_state_from_observation(
        observations[-1]
    )
    result = CompileResult(
        trajectory=ToolTrajectory(
            times=np.arange(controls.shape[0], dtype=float) * timestep,
            positions=controls.copy(),
            controls=controls.copy(),
            gripper_operations=gripper_operations[:retained_steps].copy(),
        ),
        metrics={
            "phase": label,
            "transit_simulated": True,
            "transit_valid": transit_valid,
            "transit_convergence_valid": transit_convergence_valid,
            "controller_rollout_backend": "native_cpp_fr3_osc",
            "controller_state_handoff": bool(
                initial_controller_state is not None
            ),
            "num_steps": int(controls.shape[0]),
            "duration": float(controls.shape[0] * timestep),
            "moving_duration": float(moving_steps * timestep),
            "settle_minimum_duration": float(settle_duration),
            "settle_maximum_duration": maximum_settle,
            "settle_stable_duration": float(settle_stable_duration),
            "settle_used_duration": float(
                max(0, controls.shape[0] - moving_steps) * timestep
            ),
            "settle_converged": settled,
            "terminal_target_position": target_position.tolist(),
            "terminal_observed_position": observations[-1, :3].tolist(),
            "terminal_position_error": terminal_position_error,
            "terminal_orientation_error": terminal_orientation_error,
            "terminal_linear_speed": terminal_linear_speed,
            "terminal_angular_speed": terminal_angular_speed,
            "terminal_gripper_position_error": terminal_gripper_position_error,
            "terminal_gripper_position_tolerance": (
                terminal_gripper_position_tolerance
            ),
            "terminal_gripper_speed": terminal_gripper_speed,
            "terminal_gripper_speed_tolerance": terminal_gripper_speed_tolerance,
            "minimum_joint_margin": observed_minimum_joint_margin,
            "maximum_task_force": float(np.max(np.abs(observations[:, 13:16]))),
            "maximum_task_torque": float(np.max(np.abs(observations[:, 16:19]))),
            "maximum_joint_torque": float(np.max(np.abs(observations[:, 19:26]))),
        },
        best_states=states,
        cost_history=np.empty((0,), dtype=float),
        controller_state=terminal_controller_state.copy(),
    )
    result.metrics.update(_aperture_diagnostics(model, adapter, controls, states))
    return FR3TransitResult(
        result=result,
        controller_observations=observations,
        controller_state=terminal_controller_state,
    )


def prepend_fr3_transit(
    transit: FR3TransitResult,
    effect: CompileResult,
    *,
    require_valid: bool = True,
) -> CompileResult:
    """Prepend a simulated transit while preserving audit evidence and costs."""
    transit_states = np.asarray(transit.result.best_states, dtype=float)
    effect_states = np.asarray(effect.best_states, dtype=float)
    transit_controls = np.asarray(transit.result.trajectory.controls, dtype=float)
    effect_controls = np.asarray(effect.trajectory.controls, dtype=float)
    transit_positions = np.asarray(transit.result.trajectory.positions, dtype=float)
    effect_positions = np.asarray(effect.trajectory.positions, dtype=float)
    if transit_states.ndim != 2 or effect_states.ndim != 2:
        raise ValueError("transit and effect states must be two-dimensional")
    if transit_states.shape[1] != effect_states.shape[1]:
        raise ValueError("transit and effect state dimensions do not match")
    if transit_controls.shape[1:] != effect_controls.shape[1:]:
        raise ValueError("transit and effect action dimensions do not match")
    if transit_positions.shape[1:] != effect_positions.shape[1:]:
        raise ValueError("transit and effect position dimensions do not match")
    transit_times = np.asarray(transit.result.trajectory.times, dtype=float)
    effect_times = np.asarray(effect.trajectory.times, dtype=float)
    if transit_times.size < 2:
        raise ValueError("transit trajectory must contain at least two steps")
    timestep = float(transit_times[1] - transit_times[0])
    if effect_times.size >= 2 and not np.isclose(
        float(effect_times[1] - effect_times[0]), timestep
    ):
        raise ValueError("transit and effect timesteps do not match")

    states = np.concatenate((transit_states, effect_states), axis=0)
    controls = np.concatenate((transit_controls, effect_controls), axis=0)
    positions = np.concatenate((transit_positions, effect_positions), axis=0)
    transit_operations = transit.result.trajectory.gripper_operations
    effect_operations = effect.trajectory.gripper_operations
    if transit_operations is None or effect_operations is None:
        raise ValueError("FR3 transit and effect require gripper operation streams")
    operations = np.concatenate((transit_operations, effect_operations), axis=0)
    transit_metrics = transit.result.metrics
    metrics = dict(effect.metrics)
    effect_advisories = tuple(metrics.get("audit_advisories", ()))
    transit_advisories = tuple(transit_metrics.get("audit_advisories", ()))
    phase_boundary_aperture_delta = abs(
        float(effect_controls[0, 6] - transit_controls[-1, 6])
    )
    if require_valid and phase_boundary_aperture_delta > 1e-12:
        raise ValueError(
            "effect must inherit the transit terminal gripper command without "
            "a phase-boundary jump"
        )
    metrics.update(
        {
            "transit": dict(transit.result.metrics),
            "transit_simulated": True,
            "transit_duration": float(transit.result.metrics["duration"]),
            "effect_start_index": int(transit_states.shape[0]),
            "optimized_horizon": float(effect_states.shape[0] * timestep),
            "full_horizon": float(states.shape[0] * timestep),
            "phase_boundary_aperture_delta": phase_boundary_aperture_delta,
            "transit_validation_advisory": not bool(
                transit_metrics.get(
                    "transit_convergence_valid",
                    transit_metrics.get("transit_valid"),
                )
            )
            or bool(transit_metrics.get("audit_advisory_triggered", False)),
        }
    )
    whole_arm_values = tuple(
        bool(value)
        for value in (
            transit_metrics.get("whole_arm_audit_valid"),
            metrics.get("whole_arm_audit_valid"),
        )
        if value is not None
    )
    if whole_arm_values:
        metrics["whole_arm_audit_valid"] = all(whole_arm_values)
    convergence_values = tuple(
        bool(value)
        for value in (
            transit_metrics.get("transit_convergence_valid"),
            metrics.get("transit_convergence_valid"),
        )
        if value is not None
    )
    if convergence_values:
        metrics["transit_convergence_valid"] = all(convergence_values)
    metrics["self_collision_steps"] = int(
        transit_metrics.get("self_collision_steps", 0) or 0
    ) + int(metrics.get("self_collision_steps", 0) or 0)
    metrics["unapproved_contact_steps"] = int(
        transit_metrics.get("unapproved_robot_contact_steps", 0) or 0
    ) + int(metrics.get("unapproved_contact_steps", 0) or 0)
    metrics["permitted_fingertip_contact_steps"] = int(
        transit_metrics.get("permitted_fingertip_contact_steps", 0) or 0
    ) + int(metrics.get("permitted_fingertip_contact_steps", 0) or 0)
    for maximum_key in (
        "maximum_closed_stop_penetration",
        "maximum_finger_synchronization_error",
    ):
        maximum_values = tuple(
            float(value)
            for value in (
                transit_metrics.get(maximum_key),
                metrics.get(maximum_key),
            )
            if value is not None
        )
        if maximum_values:
            metrics[maximum_key] = max(maximum_values)
    for clearance_key in (
        "minimum_environment_clearance",
        "minimum_enforced_environment_clearance",
        "minimum_declared_proximity_clearance",
    ):
        clearance_values = tuple(
            float(value)
            for value in (
                transit_metrics.get(clearance_key),
                metrics.get(clearance_key),
            )
            if value is not None
        )
        if clearance_values:
            metrics[clearance_key] = min(clearance_values)
    metrics["audit_advisories"] = list(
        dict.fromkeys((*transit_advisories, *effect_advisories))
    )
    metrics["audit_advisory_triggered"] = bool(
        metrics["audit_advisories"]
        or transit_metrics.get("audit_advisory_triggered", False)
        or effect.metrics.get("audit_advisory_triggered", False)
    )
    if "success" in metrics:
        metrics["success"] = bool(
            metrics["success"] and transit_metrics.get("transit_valid")
        )
    result = CompileResult(
        trajectory=ToolTrajectory(
            times=np.arange(states.shape[0], dtype=float) * timestep,
            positions=positions,
            controls=controls,
            gripper_operations=operations,
        ),
        metrics=metrics,
        best_states=states,
        cost_history=np.asarray(effect.cost_history, dtype=float).copy(),
        initial_pose=dict(effect.initial_pose),
        controller_state=(
            None
            if effect.controller_state is None
            else np.asarray(effect.controller_state, dtype=float).copy()
        ),
    )
    retain_execution_result(result)
    return result


__all__ = [
    "FR3TransitResult",
    "FR3TransitViolation",
    "prepend_fr3_transit",
    "rollout_fr3_transit",
]
