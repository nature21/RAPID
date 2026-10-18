"""Fast collision-aware planning for contact-free FR3 transit phases.

Planning is performed in the seven arm joints, while controls and returned
waypoints remain Cartesian.  The planning model uses kinematics and MuJoCo's
native collision pipeline only; forward dynamics are reserved for the final
OSC rollout certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
import time
from typing import Any, Literal, Sequence

import mujoco
import numpy as np

from primitive_core.fr3_osc import FR3OSCAdapter
from primitive_core.fr3_safety import (
    AllowedProximityPair,
    DEFAULT_FR3_SAFETY_LIMITS,
    FR3SafetySpec,
    minimum_pose_segment_duration,
)


FR3_TRANSIT_PLANNER_ABI_VERSION = 5


class FR3TransitPlanningFailure(RuntimeError):
    """No collision-free Cartesian-executable transit was found in budget."""


@dataclass(frozen=True)
class FR3TransitPlanningConfig:
    """Runtime-owned limits for the kinematic planning layer."""

    # Planning and executed-state thresholds stay separate so release paths
    # can carry a tracking tube without weakening the retained-state audit.
    clearance: float = 0.010
    execution_clearance: float = 0.010
    joint_margin: float = 0.05
    damping: float = 0.05
    nullspace_gain: float = 0.50
    maximum_joint_step: float = 0.05
    propagation_step_size: float = 0.08
    minimum_control_steps: int = 1
    maximum_control_steps: int = 8
    maximum_linear_velocity: float = 0.18
    maximum_angular_velocity: float = 0.80
    goal_joint_tolerance: float = 0.025
    ik_position_tolerance: float = 0.004
    ik_orientation_tolerance: float = 0.04
    ik_maximum_iterations: int = 300
    planning_time: float = 2.0
    maximum_planning_time: float = 5.0
    maximum_state_checks: int = 150_000
    maximum_waypoints: int = 16
    interpolation_joint_step: float = 0.025
    # The free path keeps ``clearance`` against ordinary obstacles.  Only the
    # tool/next-target handoff may use this smaller, still contact-free margin.
    handoff_clearance: float = 0.002
    handoff_projection_step: float = 0.002
    handoff_projection_maximum_translation: float = 0.100
    # Handoff IK is only a seed for a collision-checked joint goal.  The
    # selected Cartesian command is reanchored to that goal's FK pose, so
    # these bounded residuals widen reachability without commanding an
    # unchecked exact pose.
    handoff_projection_ik_position_tolerance: float = 0.008
    handoff_projection_ik_orientation_tolerance: float = 0.100
    handoff_projection_candidate_budget: int = 512
    handoff_projection_batch_size: int = 32
    handoff_projection_refinement_seed_count: int = 8
    handoff_projection_goal_shortlist_size: int = 4
    release_minimum_translation: float = 0.015
    release_maximum_translation: float = 0.120
    release_maximum_rotation: float = 0.40
    release_candidate_samples: int = 256
    release_shortlist_size: int = 4
    release_diversity_joint_distance: float = 0.04
    release_tree_maximum_nodes: int = 2500
    release_path_tracking_reserve: float = 0.003
    release_goal_tracking_reserve: float = 0.020
    release_maximum_clearance_regression: float = 0.00025

    def __post_init__(self) -> None:
        positive = (
            "clearance",
            "execution_clearance",
            "damping",
            "maximum_joint_step",
            "propagation_step_size",
            "maximum_linear_velocity",
            "maximum_angular_velocity",
            "goal_joint_tolerance",
            "ik_position_tolerance",
            "ik_orientation_tolerance",
            "planning_time",
            "maximum_planning_time",
            "interpolation_joint_step",
            "handoff_clearance",
            "handoff_projection_step",
            "handoff_projection_maximum_translation",
            "handoff_projection_ik_position_tolerance",
            "handoff_projection_ik_orientation_tolerance",
            "release_minimum_translation",
            "release_maximum_translation",
            "release_maximum_rotation",
            "release_diversity_joint_distance",
            "release_path_tracking_reserve",
            "release_goal_tracking_reserve",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.joint_margin < 0.0 or self.nullspace_gain < 0.0:
            raise ValueError("joint_margin and nullspace_gain cannot be negative")
        if (
            not np.isfinite(self.release_maximum_clearance_regression)
            or self.release_maximum_clearance_regression < 0.0
        ):
            raise ValueError(
                "release maximum clearance regression must be finite and "
                "nonnegative"
            )
        if self.maximum_planning_time < self.planning_time:
            raise ValueError("maximum_planning_time must cover planning_time")
        for name in (
            "minimum_control_steps",
            "maximum_control_steps",
            "ik_maximum_iterations",
            "maximum_state_checks",
            "maximum_waypoints",
            "handoff_projection_candidate_budget",
            "handoff_projection_batch_size",
            "handoff_projection_refinement_seed_count",
            "handoff_projection_goal_shortlist_size",
            "release_candidate_samples",
            "release_shortlist_size",
            "release_tree_maximum_nodes",
        ):
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_control_steps < self.minimum_control_steps:
            raise ValueError("maximum_control_steps must cover minimum_control_steps")
        if self.maximum_waypoints < 2:
            raise ValueError("maximum_waypoints must be at least two")
        if (
            self.handoff_projection_batch_size
            > self.handoff_projection_candidate_budget
        ):
            raise ValueError(
                "handoff projection batch size cannot exceed its candidate "
                "budget"
            )
        if (
            self.handoff_projection_goal_shortlist_size
            > self.handoff_projection_candidate_budget
        ):
            raise ValueError(
                "handoff projection goal shortlist cannot exceed its "
                "candidate budget"
            )
        if (
            self.handoff_projection_refinement_seed_count
            > self.handoff_projection_candidate_budget
        ):
            raise ValueError(
                "handoff projection refinement seeds cannot exceed its "
                "candidate budget"
            )
        if self.handoff_clearance > self.clearance:
            raise ValueError("handoff_clearance cannot exceed path clearance")
        if self.release_maximum_translation < self.release_minimum_translation:
            raise ValueError(
                "release maximum translation must cover its minimum translation"
            )
        if (
            self.handoff_projection_step
            > self.handoff_projection_maximum_translation
        ):
            raise ValueError(
                "handoff projection maximum must cover at least one step"
            )


DEFAULT_FR3_TRANSIT_PLANNING_CONFIG = FR3TransitPlanningConfig()


@dataclass(frozen=True)
class FR3TransitPlan:
    waypoint_actions: np.ndarray
    segment_durations: tuple[float, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FR3ReleasePair:
    """One inherited robot/environment relation admissible during escape."""

    robot_geom_id: int
    environment_geom_id: int
    entry_state: Literal["contact", "near", "tracking"]
    initial_distance: float
    minimum_distance: float

    def __post_init__(self) -> None:
        if self.robot_geom_id < 0 or self.environment_geom_id < 0:
            raise ValueError("release geometry ids must be nonnegative")
        if self.robot_geom_id == self.environment_geom_id:
            raise ValueError("release geometry ids must be distinct")
        if self.entry_state not in {"contact", "near", "tracking"}:
            raise ValueError("release entry_state is invalid")
        for name in ("initial_distance", "minimum_distance"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"release {name} must be finite")
        if self.minimum_distance > self.initial_distance:
            raise ValueError(
                "release minimum distance cannot exceed its initial distance"
            )


@dataclass(frozen=True)
class _ReleaseCandidate:
    path: np.ndarray
    target_action: np.ndarray
    backend: str
    score: float
    translation: float
    rotation: float
    minimum_progress: float
    profile_samples: int


def _load_native_type() -> type:
    try:
        import mujoco_extensions.fr3_transit_planner as native
    except ImportError as exc:
        raise RuntimeError(
            "Collision-aware FR3 transits require the native planner extension. "
            "Build third_party/judo/mujoco_extensions with CMake."
        ) from exc
    if getattr(native, "ABI_VERSION", None) != FR3_TRANSIT_PLANNER_ABI_VERSION:
        raise RuntimeError(
            "The native FR3 transit planner ABI is stale; rebuild "
            "third_party/judo/mujoco_extensions."
        )
    return native.NativeFR3TransitKinematics


def _arm_joint_bounds(
    model: mujoco.MjModel,
    adapter: FR3OSCAdapter,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in adapter.config.arm_joint_names
        ],
        dtype=int,
    )
    if np.any(joint_ids < 0):
        raise ValueError("FR3 transit planner could not resolve all arm joints")
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=float)
    lower = ranges[:, 0] + margin
    upper = ranges[:, 1] - margin
    if np.any(lower >= upper):
        raise ValueError("FR3 planner joint margin removes an arm joint range")
    return lower, upper


def _native_config(
    adapter: FR3OSCAdapter,
    safety_spec: FR3SafetySpec,
    config: FR3TransitPlanningConfig,
    *,
    release_pairs: Sequence[FR3ReleasePair] = (),
    release_goal_clearance: float | None = None,
    planning_clearance: float | None = None,
) -> dict[str, Any]:
    clearance = (
        config.clearance
        if planning_clearance is None
        else float(planning_clearance)
    )
    return {
        "clearance": clearance,
        "release_goal_clearance": (
            clearance
            if release_goal_clearance is None
            else float(release_goal_clearance)
        ),
        "joint_margin": config.joint_margin,
        "damping": config.damping,
        "nullspace_gain": config.nullspace_gain,
        "maximum_joint_step": config.maximum_joint_step,
        "arm_joint_names": list(adapter.config.arm_joint_names),
        "nullspace_configuration": list(adapter.nullspace_configuration),
        "end_effector_site_name": adapter.config.end_effector_site_name,
        "base_body_name": adapter.config.base_body_name,
        "ignored_contact_pairs": [
            {
                "geom_ids_a": list(pair.geom_ids_a),
                "geom_ids_b": list(pair.geom_ids_b),
            }
            for pair in safety_spec.permitted_gripper_contact_pairs
        ],
        "proximity_pairs": [
            {
                "geom_ids_a": list(pair.geom_ids_a),
                "geom_ids_b": list(pair.geom_ids_b),
                "minimum_clearance": float(pair.minimum_clearance),
            }
            for pair in safety_spec.allowed_proximity_pairs
            if pair.minimum_clearance is not None
        ],
        "release_pairs": [
            {
                "robot_geom_id": pair.robot_geom_id,
                "environment_geom_id": pair.environment_geom_id,
                "minimum_clearance": pair.minimum_distance,
            }
            for pair in release_pairs
        ],
    }


def _pose_from_action(adapter: FR3OSCAdapter, action: np.ndarray) -> np.ndarray:
    position, quaternion = adapter.target_pose_world(action)
    return np.concatenate((position, quaternion))


@dataclass(frozen=True)
class _IKPoseResult:
    """One kinematic solve; collision validity is checked separately."""

    q: np.ndarray
    converged: bool
    position_error: float
    orientation_error: float


def _solve_pose(
    native: Any,
    seed: np.ndarray,
    pose: np.ndarray,
    config: FR3TransitPlanningConfig,
) -> _IKPoseResult:
    q, converged, position_error, orientation_error = native.solve_pose(
        np.ascontiguousarray(seed, dtype=np.float64),
        np.ascontiguousarray(pose, dtype=np.float64),
        config.ik_maximum_iterations,
        config.ik_position_tolerance,
        config.ik_orientation_tolerance,
    )
    return _IKPoseResult(
        q=np.asarray(q, dtype=float),
        converged=bool(converged),
        position_error=float(position_error),
        orientation_error=float(orientation_error),
    )


@dataclass(frozen=True)
class _HandoffPoseEvaluation:
    action: np.ndarray
    q: np.ndarray
    offset: np.ndarray
    orientation_change: float
    valid: bool
    clearance: float
    joint_margin: float
    joint_distance: float
    direct_path_valid: bool

    @property
    def translation(self) -> float:
        return float(np.linalg.norm(self.offset))


def _radical_inverse(index: int, base: int) -> float:
    """Return one deterministic Halton coordinate in ``[0, 1)``."""

    value = 0.0
    denominator = 1.0
    remainder = int(index)
    while remainder:
        remainder, digit = divmod(remainder, base)
        denominator *= base
        value += digit / denominator
    return value


def _quantize_handoff_offset(
    offset: np.ndarray,
    config: FR3TransitPlanningConfig,
) -> np.ndarray | None:
    """Quantize only radial distance; do not impose a Cartesian grid."""

    value = np.asarray(offset, dtype=float).reshape(3)
    distance = float(np.linalg.norm(value))
    if not np.isfinite(distance) or distance <= 1e-12:
        return None
    shell = max(1, int(round(distance / config.handoff_projection_step)))
    quantized_distance = min(
        config.handoff_projection_maximum_translation,
        shell * config.handoff_projection_step,
    )
    return value * (quantized_distance / distance)


def _low_discrepancy_handoff_offsets(
    config: FR3TransitPlanningConfig,
    count: int,
    *,
    start_index: int = 1,
) -> tuple[np.ndarray, ...]:
    """Sample translations throughout the ball without fixed search rays.

    Radius is distributed uniformly over the quantized shells rather than by
    volume.  This gives nearby handoffs the same sampling attention as distant
    ones.  The two angular coordinates use independent Halton bases, so the
    direction changes at every radius.
    """

    shell_count = int(
        math.floor(
            config.handoff_projection_maximum_translation
            / config.handoff_projection_step
        )
    )
    offsets: list[np.ndarray] = []
    for index in range(start_index, start_index + max(0, int(count))):
        radial_coordinate = _radical_inverse(index, 2)
        shell = min(
            shell_count,
            1 + int(math.floor(radial_coordinate * shell_count)),
        )
        z = 1.0 - 2.0 * _radical_inverse(index, 3)
        azimuth = 2.0 * math.pi * _radical_inverse(index, 5)
        planar = math.sqrt(max(0.0, 1.0 - z * z))
        direction = np.asarray(
            (
                planar * math.cos(azimuth),
                planar * math.sin(azimuth),
                z,
            ),
            dtype=float,
        )
        offsets.append(shell * config.handoff_projection_step * direction)
    offsets.sort(key=lambda value: float(np.linalg.norm(value)))
    return tuple(offsets)


def _refined_handoff_offsets(
    seeds: Sequence[_HandoffPoseEvaluation],
    config: FR3TransitPlanningConfig,
    count: int,
    *,
    start_index: int,
) -> tuple[np.ndarray, ...]:
    """Generate bounded local translations around promising native checks."""

    if not seeds or count <= 0:
        return ()
    offsets: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    attempts = 0
    maximum_attempts = max(32, int(count) * 12)
    while len(offsets) < count and attempts < maximum_attempts:
        seed = seeds[attempts % len(seeds)]
        sequence_index = start_index + attempts
        z = 1.0 - 2.0 * _radical_inverse(sequence_index, 3)
        azimuth = 2.0 * math.pi * _radical_inverse(sequence_index, 5)
        planar = math.sqrt(max(0.0, 1.0 - z * z))
        direction = np.asarray(
            (
                planar * math.cos(azimuth),
                planar * math.sin(azimuth),
                z,
            ),
            dtype=float,
        )
        refinement_shell = 1 + (
            (attempts // max(1, len(seeds))) % 5
        )
        raw = seed.offset + (
            refinement_shell
            * config.handoff_projection_step
            * direction
        )
        candidate = _quantize_handoff_offset(raw, config)
        attempts += 1
        if candidate is None:
            continue
        key = tuple(int(round(component * 1e9)) for component in candidate)
        if key in seen:
            continue
        seen.add(key)
        offsets.append(candidate)
    offsets.sort(key=lambda value: float(np.linalg.norm(value)))
    return tuple(offsets)


def _project_handoff_goal(
    native: Any,
    start_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    adapter: FR3OSCAdapter,
    desired_action: np.ndarray,
    orientation_candidates: Sequence[np.ndarray],
    config: FR3TransitPlanningConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    tuple[_HandoffPoseEvaluation, ...],
]:
    """Find a nearby collision-free handoff using native kinematics only.

    Every declared task-equivalent attitude is searched throughout the bounded
    translation ball.  In particular, failure to converge at the exact desired
    position does not rule out a nearby reachable position.  Kinematic
    convergence and collision validity remain separate so colliding solutions
    can guide local refinement without ever becoming planning goals.
    """

    desired_position, desired_orientation = adapter.target_pose_world(
        desired_action
    )
    attitudes: list[np.ndarray] = [desired_orientation]
    for raw in orientation_candidates:
        orientation = np.asarray(raw, dtype=float)
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
            raise ValueError(
                "handoff orientation candidates must be finite quaternions"
            )
        norm = float(np.linalg.norm(orientation))
        if norm <= 1e-12:
            raise ValueError("handoff orientation candidates cannot be zero")
        orientation = orientation / norm
        if any(
            abs(float(existing @ orientation)) >= 1.0 - 1e-12
            for existing in attitudes
        ):
            continue
        attitudes.append(orientation)
    if len(attitudes) > config.handoff_projection_candidate_budget:
        raise ValueError(
            "handoff attitude candidates exceed the projection candidate budget"
        )
    search_started = time.monotonic()
    seeds = (
        start_q,
        np.clip(np.asarray(adapter.nullspace_configuration), lower, upper),
        np.clip(
            0.5 * (start_q + np.asarray(adapter.nullspace_configuration)),
            lower,
            upper,
        ),
    )
    projection_config = replace(
        config,
        ik_position_tolerance=(
            config.handoff_projection_ik_position_tolerance
        ),
        ik_orientation_tolerance=(
            config.handoff_projection_ik_orientation_tolerance
        ),
    )
    candidate_count = 0
    global_candidate_count = 0
    refinement_candidate_count = 0
    ik_attempt_count = 0
    ik_converged_count = 0
    best_ik_result: _IKPoseResult | None = None
    evaluations: list[_HandoffPoseEvaluation] = []
    shortlist_count = 1
    exact_ik_attitude_count = 0

    def solve(action: np.ndarray) -> _HandoffPoseEvaluation | None:
        nonlocal candidate_count, ik_attempt_count, ik_converged_count
        nonlocal best_ik_result
        candidate_count += 1
        pose = _pose_from_action(adapter, action)
        solved_candidates: list[_HandoffPoseEvaluation] = []
        for seed in seeds:
            result = _solve_pose(native, seed, pose, projection_config)
            ik_attempt_count += 1
            if best_ik_result is None or max(
                result.position_error / projection_config.ik_position_tolerance,
                result.orientation_error
                / projection_config.ik_orientation_tolerance,
            ) < max(
                best_ik_result.position_error
                / projection_config.ik_position_tolerance,
                best_ik_result.orientation_error
                / projection_config.ik_orientation_tolerance,
            ):
                best_ik_result = result
            if not result.converged:
                continue
            ik_converged_count += 1
            realized_pose = np.asarray(
                native.pose(np.ascontiguousarray(result.q)),
                dtype=float,
            )
            realized_position = realized_pose[:3]
            realized_orientation = realized_pose[3:]
            offset = realized_position - desired_position
            realized_translation_limit = (
                config.handoff_projection_maximum_translation
                + projection_config.ik_position_tolerance
            )
            if (
                float(np.linalg.norm(offset))
                > realized_translation_limit + 1e-12
            ):
                continue
            orientation_change = 2.0 * float(
                np.arccos(
                    np.clip(
                        abs(
                            float(
                                desired_orientation @ realized_orientation
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            valid, clearance = native.check_state(
                np.ascontiguousarray(result.q)
            )
            direct_path_valid = bool(
                valid
                and _configuration_segment_valid(
                    native,
                    start_q,
                    result.q,
                    config,
                )
            )
            solved_candidates.append(
                _HandoffPoseEvaluation(
                    action=adapter.action_from_world_pose(
                        realized_position,
                        realized_orientation,
                        gripper=float(desired_action[6]),
                    ),
                    q=result.q,
                    offset=offset.copy(),
                    orientation_change=orientation_change,
                    valid=bool(valid),
                    clearance=float(clearance),
                    joint_margin=float(
                        np.min(
                            np.minimum(
                                result.q - lower,
                                upper - result.q,
                            )
                        )
                    ),
                    joint_distance=float(
                        np.linalg.norm(result.q - start_q)
                    ),
                    direct_path_valid=direct_path_valid,
                )
            )
        if not solved_candidates:
            return None
        return min(
            solved_candidates,
            key=lambda item: (
                not item.valid,
                not item.direct_path_valid,
                item.joint_distance,
                -item.clearance,
                -item.joint_margin,
            ),
        )

    def metrics_for(selected: _HandoffPoseEvaluation) -> dict[str, Any]:
        projected_position, _ = adapter.target_pose_world(selected.action)
        displacement = projected_position - desired_position
        translation = float(np.linalg.norm(displacement))
        return {
            "handoff_projection_required": bool(
                translation > 1e-12 or selected.orientation_change > 1e-12
            ),
            "handoff_desired_action": desired_action.tolist(),
            "handoff_projected_action": selected.action.tolist(),
            "handoff_projection_translation": translation,
            "handoff_projection_orientation": selected.orientation_change,
            "handoff_projection_direction": (
                displacement / translation
                if translation > 1e-12
                else np.zeros(3, dtype=float)
            ).tolist(),
            "handoff_projection_candidates": candidate_count,
            "handoff_projection_time": time.monotonic() - search_started,
            "handoff_projection_candidate_budget": (
                config.handoff_projection_candidate_budget
            ),
            "handoff_projection_realized_translation_limit": (
                config.handoff_projection_maximum_translation
                + projection_config.ik_position_tolerance
            ),
            "handoff_projection_global_candidates": global_candidate_count,
            "handoff_projection_refinement_candidates": (
                refinement_candidate_count
            ),
            "handoff_projection_ik_attempts": ik_attempt_count,
            "handoff_projection_ik_solutions": ik_converged_count,
            "handoff_projection_ik_position_tolerance": (
                projection_config.ik_position_tolerance
            ),
            "handoff_projection_ik_orientation_tolerance": (
                projection_config.ik_orientation_tolerance
            ),
            "handoff_projection_feasible_candidates": sum(
                evaluation.valid for evaluation in evaluations
            ),
            "handoff_projection_direct_path_candidates": sum(
                evaluation.direct_path_valid for evaluation in evaluations
            ),
            "handoff_projection_goal_shortlist_size": shortlist_count,
            "handoff_projection_selected_goal_rank": 1,
            "handoff_projection_search": (
                "task_equivalent_attitudes_then_low_discrepancy_translation_ball"
            ),
            "handoff_projection_orientation_fixed": len(attitudes) == 1,
            "handoff_orientation_candidates": len(attitudes),
            "handoff_exact_ik_orientation_candidates": (
                exact_ik_attitude_count
            ),
            "handoff_projection_aperture_fixed": True,
            "handoff_required_clearance": config.handoff_clearance,
            "handoff_projected_clearance": selected.clearance,
            "handoff_projected_joint_margin": selected.joint_margin,
            "handoff_projected_direct_path_valid": (
                selected.direct_path_valid
            ),
        }

    def feasible_key(
        evaluation: _HandoffPoseEvaluation,
    ) -> tuple[bool, float, float, float, float, float]:
        return (
            not evaluation.direct_path_valid,
            evaluation.translation,
            evaluation.orientation_change,
            -evaluation.clearance,
            evaluation.joint_distance,
            -evaluation.joint_margin,
        )

    for orientation in attitudes:
        action = adapter.action_from_world_pose(
            desired_position,
            orientation,
            gripper=float(desired_action[6]),
        )
        exact = solve(action)
        if exact is not None:
            evaluations.append(exact)
            exact_ik_attitude_count += 1
    exact_direct = [
        evaluation for evaluation in evaluations if evaluation.direct_path_valid
    ]
    if exact_direct:
        selected = min(exact_direct, key=feasible_key)
        # Prefer existing joint reserve over a smaller IK residual at a bound.
        # These bounds already include the configured planning joint margin.
        if selected.joint_margin <= 1e-12:
            interior = [item for item in exact_direct if item.joint_margin > 1e-12]
            if interior:
                selected = min(interior, key=feasible_key)
        return selected.action, selected.q, metrics_for(selected), (selected,)

    total_budget = config.handoff_projection_candidate_budget
    refinement_reserve = min(128, max(1, total_budget // 4))
    global_limit = max(
        0,
        (total_budget - candidate_count - refinement_reserve)
        // len(attitudes),
    )
    global_offsets = _low_discrepancy_handoff_offsets(
        config,
        global_limit,
    )
    feasible_found = False
    for start in range(0, len(global_offsets), config.handoff_projection_batch_size):
        batch = global_offsets[
            start : start + config.handoff_projection_batch_size
        ]
        for offset in batch:
            for orientation in attitudes:
                action = adapter.action_from_world_pose(
                    desired_position + offset,
                    orientation,
                    gripper=float(desired_action[6]),
                )
                evaluation = solve(action)
                global_candidate_count += 1
                if evaluation is not None:
                    evaluations.append(evaluation)
                    feasible_found = (
                        feasible_found or evaluation.direct_path_valid
                    )
        if feasible_found:
            break

    remaining_budget = min(
        refinement_reserve,
        total_budget - candidate_count,
    )
    if remaining_budget > 0:
        refinement_seeds = tuple(
            sorted(
                evaluations,
                key=lambda item: (
                    not item.valid,
                    item.translation if item.valid else -item.clearance,
                    -item.clearance,
                    -item.joint_margin,
                    item.joint_distance,
                ),
            )[: config.handoff_projection_refinement_seed_count]
        )
        if refinement_seeds:
            refinement_offsets = _refined_handoff_offsets(
                refinement_seeds,
                config,
                remaining_budget,
                start_index=global_limit + 1,
            )
        else:
            refinement_offsets = _low_discrepancy_handoff_offsets(
                config,
                remaining_budget,
                start_index=global_limit + 1,
            )
        for offset in refinement_offsets:
            for orientation in attitudes:
                if candidate_count >= total_budget:
                    break
                action = adapter.action_from_world_pose(
                    desired_position + offset,
                    orientation,
                    gripper=float(desired_action[6]),
                )
                evaluation = solve(action)
                refinement_candidate_count += 1
                if evaluation is not None:
                    evaluations.append(evaluation)

    feasible = [evaluation for evaluation in evaluations if evaluation.valid]
    if feasible:
        selected = min(feasible, key=feasible_key)
        shortlist: list[_HandoffPoseEvaluation] = [selected]
        ranking_views = (
            sorted(
                feasible,
                key=lambda item: (
                    item.joint_distance,
                    item.translation,
                    item.orientation_change,
                    -item.clearance,
                ),
            ),
            sorted(
                feasible,
                key=lambda item: (
                    -item.clearance,
                    item.joint_distance,
                    item.translation,
                ),
            ),
            sorted(
                feasible,
                key=lambda item: (
                    -item.joint_margin,
                    item.joint_distance,
                    item.translation,
                ),
            ),
        )
        for ranking in ranking_views:
            for candidate in ranking:
                if (
                    len(shortlist)
                    >= config.handoff_projection_goal_shortlist_size
                ):
                    break
                if any(
                    np.linalg.norm(candidate.q - existing.q)
                    <= config.goal_joint_tolerance
                    for existing in shortlist
                ):
                    continue
                shortlist.append(candidate)
            if len(shortlist) >= config.handoff_projection_goal_shortlist_size:
                break
        shortlist_count = len(shortlist)
        return (
            selected.action,
            selected.q,
            metrics_for(selected),
            tuple(shortlist),
        )

    best_clearance = max(
        (evaluation.clearance for evaluation in evaluations),
        default=-math.inf,
    )
    clearance_summary = (
        f"{best_clearance:.6f} m"
        if np.isfinite(best_clearance)
        else "unavailable"
    )
    if best_ik_result is None:
        residual_summary = "IK residual unavailable"
    else:
        residual_summary = (
            f"best IK residual {best_ik_result.position_error:.6f} m / "
            f"{best_ik_result.orientation_error:.6f} rad"
        )
    raise FR3TransitPlanningFailure(
        "the desired next-primitive initial pose could not be projected to a "
        "nearby collision-free handoff within "
        f"{config.handoff_projection_maximum_translation:.3f} m after "
        f"{candidate_count} candidate poses and {ik_attempt_count} IK attempts "
        f"({ik_converged_count} kinematically converged, "
        f"{sum(evaluation.valid for evaluation in evaluations)} "
        f"collision-free candidates, {residual_summary}, "
        f"best clearance {clearance_summary}, "
        f"{time.monotonic() - search_started:.3f} s)"
    )


def _configuration_segment_valid(
    native: Any,
    start: np.ndarray,
    end: np.ndarray,
    config: FR3TransitPlanningConfig,
) -> bool:
    steps = max(
        1,
        int(
            math.ceil(
                float(np.max(np.abs(end - start)))
                / config.interpolation_joint_step
            )
        ),
    )
    for fraction in np.arange(1, steps + 1, dtype=float) / steps:
        if native.state_checks() >= config.maximum_state_checks:
            return False
        if not native.is_valid(
            np.ascontiguousarray((1.0 - fraction) * start + fraction * end)
        ):
            return False
    return True


def _release_path_profile(
    native: Any,
    coarse_path: np.ndarray,
    release_pairs: Sequence[FR3ReleasePair],
    release_goal_clearance: float,
    config: FR3TransitPlanningConfig,
) -> tuple[np.ndarray, float, int] | None:
    """Densify and certify contact progress plus contact-free pair floors."""

    initial = np.asarray(
        [pair.initial_distance for pair in release_pairs],
        dtype=float,
    )
    minimum = np.asarray(
        [pair.minimum_distance for pair in release_pairs],
        dtype=float,
    )
    denominator = np.maximum(release_goal_clearance - initial, 1e-6)
    peak = initial.copy()
    progress_required = np.asarray(
        [pair.entry_state == "contact" for pair in release_pairs],
        dtype=bool,
    )
    dense = [np.asarray(coarse_path[0], dtype=float).copy()]
    minimum_progress = math.inf
    profile_samples = 0

    for start, end in zip(coarse_path[:-1], coarse_path[1:]):
        steps = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(end - start)))
                    / config.interpolation_joint_step
                )
            ),
        )
        for fraction in np.arange(1, steps + 1, dtype=float) / steps:
            if native.state_checks() >= config.maximum_state_checks:
                return None
            q = np.ascontiguousarray(
                (1.0 - fraction) * start + fraction * end,
                dtype=np.float64,
            )
            if not native.is_valid(q):
                return None
            distances = np.asarray(native.release_distances(q), dtype=float)
            if distances.shape != initial.shape or not np.all(
                np.isfinite(distances)
            ):
                return None
            if np.any(distances < minimum) or np.any(
                distances[progress_required]
                < peak[progress_required]
                - config.release_maximum_clearance_regression
            ):
                return None
            progress = np.clip((distances - initial) / denominator, -1.0, 1.0)
            minimum_progress = min(minimum_progress, float(np.min(progress)))
            peak = np.maximum(peak, distances)
            dense.append(np.asarray(q, dtype=float))
            profile_samples += 1

    if not dense or profile_samples < 1:
        return None
    return (
        np.asarray(dense, dtype=float),
        float(minimum_progress),
        profile_samples,
    )


def _guide_candidate(
    native: Any,
    start_q: np.ndarray,
    guide_actions: np.ndarray,
    adapter: FR3OSCAdapter,
    config: FR3TransitPlanningConfig,
) -> np.ndarray | None:
    configurations = [start_q.copy()]
    seed = start_q
    for action in guide_actions:
        result = _solve_pose(
            native,
            seed,
            _pose_from_action(adapter, action),
            config,
        )
        if not result.converged or not _configuration_segment_valid(
            native, seed, result.q, config
        ):
            return None
        configurations.append(result.q)
        seed = result.q
    return np.asarray(configurations, dtype=float)


def _planning_seed(value: int) -> int:
    """Validate one explicit positive signed 32-bit planner seed."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("planning seed must be an integer")
    seed = int(value)
    if seed < 1 or seed >= 2**31:
        raise ValueError("planning seed must be in [1, 2**31)")
    return seed


def _ompl_geometric_candidate(
    native: Any,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: FR3TransitPlanningConfig,
    *,
    deadline: float,
) -> np.ndarray | None:
    """Find a joint path whose retained controls are still Cartesian poses."""

    try:
        from ompl import base as ob
        from ompl import geometric as og
    except ImportError as exc:
        raise RuntimeError(
            "Collision-aware FR3 transits require the pinned OMPL Python bindings"
        ) from exc

    remaining = min(config.planning_time, deadline - time.monotonic())
    if remaining <= 0.0:
        return None
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    for index in range(7):
        bounds.setLow(index, float(lower[index]))
        bounds.setHigh(index, float(upper[index]))
    space.setBounds(bounds)
    setup = og.SimpleSetup(space)

    def valid(state: Any) -> bool:
        if time.monotonic() >= deadline:
            return False
        if native.state_checks() >= config.maximum_state_checks:
            return False
        q = np.ascontiguousarray([state[index] for index in range(7)])
        return bool(native.is_valid(q))

    setup.setStateValidityChecker(valid)
    start = space.allocState()
    goal = space.allocState()
    for index in range(7):
        start[index] = float(start_q[index])
        goal[index] = float(goal_q[index])
    setup.setStartAndGoalStates(start, goal)
    planner = og.RRTConnect(setup.getSpaceInformation())
    planner.setRange(0.20)
    setup.setPlanner(planner)
    status = setup.solve(remaining)
    if not bool(status) or not setup.haveExactSolutionPath():
        return None
    solution = setup.getSolutionPath()
    configurations = np.asarray(
        [
            [solution.getState(row)[column] for column in range(7)]
            for row in range(solution.getStateCount())
        ],
        dtype=float,
    )
    if configurations.shape[0] < 2:
        return None
    for left, right in zip(configurations[:-1], configurations[1:]):
        if not _configuration_segment_valid(native, left, right, config):
            return None
    configurations[0] = start_q
    configurations[-1] = goal_q
    return configurations


def _ompl_candidate(
    native: Any,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: FR3TransitPlanningConfig,
    *,
    deadline: float,
) -> np.ndarray | None:
    try:
        from ompl import base as ob
        from ompl import control as oc
    except ImportError as exc:
        raise RuntimeError(
            "Collision-aware FR3 transits require the pinned OMPL Python bindings"
        ) from exc

    remaining = min(config.planning_time, deadline - time.monotonic())
    if remaining <= 0.0:
        return None
    state_space = ob.RealVectorStateSpace(7)
    state_bounds = ob.RealVectorBounds(7)
    for index in range(7):
        state_bounds.setLow(index, float(lower[index]))
        state_bounds.setHigh(index, float(upper[index]))
    state_space.setBounds(state_bounds)
    control_space = oc.RealVectorControlSpace(state_space, 6)
    control_bounds = ob.RealVectorBounds(6)
    for index in range(3):
        control_bounds.setLow(index, -config.maximum_linear_velocity)
        control_bounds.setHigh(index, config.maximum_linear_velocity)
    for index in range(3, 6):
        control_bounds.setLow(index, -config.maximum_angular_velocity)
        control_bounds.setHigh(index, config.maximum_angular_velocity)
    control_space.setBounds(control_bounds)
    setup = oc.SimpleSetup(control_space)

    def valid(state: Any) -> bool:
        if time.monotonic() >= deadline:
            return False
        if native.state_checks() >= config.maximum_state_checks:
            return False
        q = np.ascontiguousarray([state[index] for index in range(7)])
        return bool(native.is_valid(q))

    def propagate(state: Any, control: Any, duration: float, result: Any) -> None:
        q = np.ascontiguousarray([state[index] for index in range(7)])
        twist = np.ascontiguousarray([control[index] for index in range(6)])
        propagated = np.asarray(native.propagate(q, twist, duration), dtype=float)
        for index in range(7):
            result[index] = float(propagated[index])

    setup.setStateValidityChecker(valid)
    setup.setStatePropagator(propagate)
    information = setup.getSpaceInformation()
    information.setPropagationStepSize(config.propagation_step_size)
    information.setMinMaxControlDuration(
        config.minimum_control_steps,
        config.maximum_control_steps,
    )
    start = state_space.allocState()
    goal = state_space.allocState()
    for index in range(7):
        start[index] = float(start_q[index])
        goal[index] = float(goal_q[index])
    setup.setStartAndGoalStates(start, goal, config.goal_joint_tolerance)
    planner = oc.RRT(information)
    planner.setGoalBias(0.20)
    planner.setIntermediateStates(True)
    setup.setPlanner(planner)
    status = setup.solve(remaining)
    if not bool(status) or not setup.haveExactSolutionPath():
        return None
    path = setup.getSolutionPath()
    configurations = np.asarray(
        [
            [path.getState(row)[column] for column in range(7)]
            for row in range(path.getStateCount())
        ],
        dtype=float,
    )
    if configurations.shape[0] < 2:
        return None
    if not _configuration_segment_valid(
        native, configurations[-1], goal_q, config
    ):
        return None
    configurations[-1] = goal_q
    return configurations


def _downsample_path(path: np.ndarray, maximum_waypoints: int) -> np.ndarray:
    if path.shape[0] <= maximum_waypoints:
        return path
    indices = np.unique(
        np.rint(np.linspace(0, path.shape[0] - 1, maximum_waypoints)).astype(int)
    )
    return path[indices]


def _actions_from_path(
    native: Any,
    path: np.ndarray,
    adapter: FR3OSCAdapter,
    start_action: np.ndarray,
    target_action: np.ndarray,
) -> np.ndarray:
    actions = [np.asarray(start_action, dtype=float).copy()]
    gripper = float(target_action[6])
    for q in path[1:-1]:
        pose = np.asarray(native.pose(np.ascontiguousarray(q)), dtype=float)
        actions.append(
            adapter.action_from_world_pose(pose[:3], pose[3:], gripper=gripper)
        )
    actions.append(np.asarray(target_action, dtype=float).copy())
    return np.asarray(actions, dtype=float)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        value = rng.normal(size=3)
        norm = float(np.linalg.norm(value))
        if np.isfinite(norm) and norm > 1e-9:
            return value / norm


def _rotated_quaternion(
    orientation: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    half = 0.5 * float(angle)
    delta = np.concatenate(([math.cos(half)], np.asarray(axis) * math.sin(half)))
    output = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(output, delta, orientation)
    norm = float(np.linalg.norm(output))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("release orientation sample produced an invalid quaternion")
    return output / norm


def _release_pose_samples(
    start_position: np.ndarray,
    start_orientation: np.ndarray,
    escape_biases: Sequence[np.ndarray],
    config: FR3TransitPlanningConfig,
    rng: np.random.Generator,
) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """Sample a continuous local SE(3) escape region.

    Contact normals and learned motion contribute only a probability bias.
    Every translational and rotational direction remains reachable.
    """

    biases: list[np.ndarray] = []
    for value in escape_biases:
        direction = np.asarray(value, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if np.isfinite(norm) and norm > 1e-9:
            biases.append(direction / norm)
    combined = (
        np.sum(np.asarray(biases), axis=0) if biases else np.zeros(3, dtype=float)
    )
    combined_norm = float(np.linalg.norm(combined))
    if combined_norm > 1e-9:
        combined = combined / combined_norm
    else:
        combined = np.asarray((0.0, 0.0, 1.0), dtype=float)

    directed = [combined]
    for bias in biases:
        if all(float(bias @ prior) < 0.995 for prior in directed):
            directed.append(bias)
    radii = np.unique(
        np.clip(
            np.asarray(
                (
                    config.release_minimum_translation,
                    0.5
                    * (
                        config.release_minimum_translation
                        + config.release_maximum_translation
                    ),
                    config.release_maximum_translation,
                )
            ),
            config.release_minimum_translation,
            config.release_maximum_translation,
        )
    )
    samples: list[tuple[np.ndarray, np.ndarray]] = [
        (
            start_position + float(radius) * direction,
            start_orientation.copy(),
        )
        for direction in directed
        for radius in radii
    ][: config.release_candidate_samples]
    while len(samples) < config.release_candidate_samples:
        direction = _random_unit_vector(rng)
        if biases and float(rng.random()) < 0.80:
            bias = biases[int(rng.integers(len(biases)))]
            direction = direction + float(rng.uniform(0.25, 1.50)) * bias
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-9:
                direction = direction / direction_norm
        radius_fraction = float(rng.random()) ** 1.5
        distance = config.release_minimum_translation + radius_fraction * (
            config.release_maximum_translation
            - config.release_minimum_translation
        )
        if float(rng.random()) < 0.65:
            orientation = start_orientation.copy()
        else:
            axis = _random_unit_vector(rng)
            angle = float(
                np.clip(
                    rng.normal(scale=0.45 * config.release_maximum_rotation),
                    -config.release_maximum_rotation,
                    config.release_maximum_rotation,
                )
            )
            orientation = _rotated_quaternion(start_orientation, axis, angle)
        samples.append((start_position + distance * direction, orientation))
    return samples


def _release_control_tree_candidates(
    native: Any,
    start_q: np.ndarray,
    start_action: np.ndarray,
    adapter: FR3OSCAdapter,
    release_pairs: Sequence[FR3ReleasePair],
    escape_biases: Sequence[np.ndarray],
    config: FR3TransitPlanningConfig,
    rng: np.random.Generator,
    *,
    deadline: float,
) -> list[_ReleaseCandidate]:
    """Grow a directed Cartesian-control tree toward the free-state region."""

    start_position, start_orientation = adapter.target_pose_world(start_action)
    initial_distances = np.asarray(
        [pair.initial_distance for pair in release_pairs], dtype=float
    )
    progress_required = np.asarray(
        [pair.entry_state == "contact" for pair in release_pairs],
        dtype=bool,
    )
    release_goal_clearance = (
        config.execution_clearance + config.release_goal_tracking_reserve
    )
    denominators = np.maximum(
        release_goal_clearance - initial_distances,
        1e-6,
    )
    biases: list[np.ndarray] = []
    for value in escape_biases:
        direction = np.asarray(value, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if np.isfinite(norm) and norm > 1e-9:
            biases.append(direction / norm)

    configurations: list[np.ndarray] = [start_q.copy()]
    parents = [-1]
    depths = [0]
    scores = [0.0]
    distance_vectors = [initial_distances.copy()]
    quantized = {
        tuple(np.rint(start_q / 0.015).astype(int).tolist())
    }
    goals: list[_ReleaseCandidate] = []

    while (
        len(configurations) < config.release_tree_maximum_nodes
        and time.monotonic() < deadline
        and native.state_checks() < config.maximum_state_checks
    ):
        if len(configurations) == 1 or float(rng.random()) < 0.72:
            top = np.argsort(np.asarray(scores))[-min(48, len(scores)) :]
            parent_index = int(top[int(rng.integers(len(top)))])
        else:
            parent_index = int(rng.integers(len(configurations)))
        parent_q = configurations[parent_index]

        linear_direction = _random_unit_vector(rng)
        if biases and float(rng.random()) < 0.75:
            bias = biases[int(rng.integers(len(biases)))]
            linear_direction = (
                linear_direction + float(rng.uniform(0.20, 1.75)) * bias
            )
            norm = float(np.linalg.norm(linear_direction))
            if norm > 1e-9:
                linear_direction = linear_direction / norm
        linear_speed = float(
            rng.uniform(0.25, 1.0) * config.maximum_linear_velocity
        )
        angular_velocity = np.zeros(3, dtype=float)
        if float(rng.random()) >= 0.25:
            angular_velocity = _random_unit_vector(rng) * float(
                rng.uniform(0.10, 1.0) * config.maximum_angular_velocity
            )
        twist = np.concatenate((linear_direction * linear_speed, angular_velocity))
        duration = float(rng.uniform(0.08, 0.32))
        child_q = np.asarray(
            native.propagate(
                np.ascontiguousarray(parent_q),
                np.ascontiguousarray(twist),
                duration,
            ),
            dtype=float,
        )
        if float(np.linalg.norm(child_q - parent_q)) < 1e-5:
            continue
        key = tuple(np.rint(child_q / 0.015).astype(int).tolist())
        if key in quantized:
            continue
        if not _configuration_segment_valid(native, parent_q, child_q, config):
            continue
        distances = np.asarray(
            native.release_distances(np.ascontiguousarray(child_q)), dtype=float
        )
        if np.any(
            distances[progress_required]
            < distance_vectors[parent_index][progress_required]
            - config.release_maximum_clearance_regression
        ):
            continue
        progress = np.clip(
            (distances - initial_distances) / denominators,
            -1.0,
            1.0,
        )
        score = float(np.min(progress) + 0.35 * np.mean(progress))
        score -= 0.002 * float(depths[parent_index] + 1)
        child_index = len(configurations)
        configurations.append(child_q)
        parents.append(parent_index)
        depths.append(depths[parent_index] + 1)
        scores.append(score)
        distance_vectors.append(distances)
        quantized.add(key)

        goal_valid, _ = native.check_release_goal(np.ascontiguousarray(child_q))
        if not goal_valid:
            continue
        indices: list[int] = []
        cursor = child_index
        while cursor >= 0:
            indices.append(cursor)
            cursor = parents[cursor]
        path = np.asarray(
            [configurations[index] for index in reversed(indices)], dtype=float
        )
        pose = np.asarray(native.pose(np.ascontiguousarray(child_q)), dtype=float)
        target_action = adapter.action_from_world_pose(
            pose[:3], pose[3:], gripper=float(start_action[6])
        )
        translation = float(np.linalg.norm(pose[:3] - start_position))
        rotation = 2.0 * float(
            np.arccos(
                np.clip(
                    abs(float(start_orientation @ pose[3:])),
                    0.0,
                    1.0,
                )
            )
        )
        path_length = float(
            np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        )
        profile = _release_path_profile(
            native,
            path,
            release_pairs,
            release_goal_clearance,
            config,
        )
        if profile is None:
            continue
        dense_path, minimum_progress, profile_samples = profile
        goals.append(
            _ReleaseCandidate(
                path=dense_path,
                target_action=target_action,
                backend="release_directed_control_tree",
                score=path_length + translation + 0.15 * rotation,
                translation=translation,
                rotation=rotation,
                minimum_progress=minimum_progress,
                profile_samples=profile_samples,
            )
        )
        if len(goals) >= 12 * config.release_shortlist_size:
            break
    return goals


def plan_fr3_release(
    model: mujoco.MjModel,
    initial_state: np.ndarray,
    adapter: FR3OSCAdapter,
    start_action: np.ndarray,
    release_pairs: Sequence[FR3ReleasePair],
    *,
    escape_biases: Sequence[np.ndarray] = (),
    config: FR3TransitPlanningConfig = DEFAULT_FR3_TRANSIT_PLANNING_CONFIG,
    seed: int,
    search_mode: Literal["direct", "expanded"] = "direct",
) -> tuple[FR3TransitPlan, ...]:
    """Find diverse fixed-aperture paths from inherited contact to free space.

    Search is entirely kinematic. Contact-equivalent pairs retain their
    entry-distance hysteresis, nearby pairs remain contact-free, and ordinary
    collision pairs carry a controller-tracking reserve. Returned candidates
    must finish with the full release clearance and are intended for a small
    controller-rollout shortlist.
    """

    inherited_pairs = tuple(release_pairs)
    if not inherited_pairs:
        raise ValueError("release planning requires inherited geometry pairs")
    if search_mode not in {"direct", "expanded"}:
        raise ValueError("release search_mode must be 'direct' or 'expanded'")
    state = np.asarray(initial_state, dtype=float)
    action = np.asarray(start_action, dtype=float)
    if state.shape != (model.nq + model.nv,) or not np.all(np.isfinite(state)):
        raise ValueError("FR3 release initial_state has the wrong shape or values")
    if action.shape != (adapter.action_dim,) or not np.all(np.isfinite(action)):
        raise ValueError("FR3 release start_action has the wrong shape or values")

    safety_spec = FR3SafetySpec.from_model(model, adapter)
    planning_clearance = (
        config.execution_clearance + config.release_path_tracking_reserve
    )
    release_goal_clearance = (
        config.execution_clearance + config.release_goal_tracking_reserve
    )
    planning_pairs = inherited_pairs
    entry_relaxed_pair_count = sum(
        pair.entry_state == "tracking" for pair in planning_pairs
    )
    native = _load_native_type()(
        model,
        np.ascontiguousarray(state[: model.nq]),
        _native_config(
            adapter,
            safety_spec,
            config,
            release_pairs=planning_pairs,
            release_goal_clearance=release_goal_clearance,
            planning_clearance=planning_clearance,
        ),
    )
    start_q = np.asarray(native.arm_configuration(), dtype=float)
    if not native.is_valid(np.ascontiguousarray(start_q)):
        physical_lower, physical_upper = _arm_joint_bounds(model, adapter, 0.0)
        margins = np.minimum(start_q - physical_lower, physical_upper - start_q)
        outside = np.flatnonzero(margins < config.joint_margin)
        if outside.size:
            details = ", ".join(
                f"{adapter.config.arm_joint_names[index]}: physical margin={margins[index]:.6f} rad"
                for index in outside
            )
            reason = "physical joint limit" if np.any(margins < 0) else "planning joint reserve"
            raise FR3TransitPlanningFailure(
                f"release entry violates {reason}; required planning margin={config.joint_margin:.6f} rad; "
                f"{details}; collision clearance was not evaluated"
            )
        _, minimum_clearance = native.check_state(
            np.ascontiguousarray(start_q)
        )
        entry_distances = np.asarray(
            native.release_distances(np.ascontiguousarray(start_q)),
            dtype=float,
        )
        violated_pairs = [
            (pair, float(distance))
            for pair, distance in zip(planning_pairs, entry_distances)
            if distance < pair.minimum_distance
        ]
        pair_details: list[str] = []
        for pair, distance in violated_pairs[:4]:
            robot_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                pair.robot_geom_id,
            )
            environment_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                pair.environment_geom_id,
            )
            pair_details.append(
                f"{robot_name} / {environment_name}: "
                f"{distance:.6f} m < {pair.minimum_distance:.6f} m"
            )
        pair_detail = "; ".join(pair_details)
        raise FR3TransitPlanningFailure(
            "release entry violates its captured geometry contract "
            f"(minimum clearance {float(minimum_clearance):.6f} m)"
            + (f": {pair_detail}" if pair_detail else "")
        )
    already_clear, _ = native.check_release_goal(np.ascontiguousarray(start_q))
    if already_clear:
        raise FR3TransitPlanningFailure(
            "release planning was requested from an already-clear configuration"
        )

    lower, upper = _arm_joint_bounds(model, adapter, config.joint_margin)
    start_position, start_orientation = adapter.target_pose_world(action)
    planner_seed = _planning_seed(seed)
    rng = np.random.default_rng(planner_seed)
    started = time.monotonic()
    deadline = started + config.maximum_planning_time
    sampling_deadline = started + 0.70 * config.maximum_planning_time
    native.reset_counters()
    release_ik_config = replace(
        config,
        ik_maximum_iterations=min(config.ik_maximum_iterations, 120),
    )
    goal_pool: list[tuple[np.ndarray, np.ndarray, float, float, float]] = []
    candidates: list[_ReleaseCandidate] = []
    samples_evaluated = 0
    ompl_seeded = False

    for position, orientation in _release_pose_samples(
        start_position,
        start_orientation,
        escape_biases,
        config,
        rng,
    ):
        if time.monotonic() >= sampling_deadline:
            break
        samples_evaluated += 1
        target_action = adapter.action_from_world_pose(
            position,
            orientation,
            gripper=float(action[6]),
        )
        result = _solve_pose(
            native,
            start_q,
            _pose_from_action(adapter, target_action),
            release_ik_config,
        )
        if not result.converged:
            continue
        goal_q = result.q
        goal_valid, goal_clearance = native.check_release_goal(
            np.ascontiguousarray(goal_q)
        )
        if not goal_valid:
            continue
        translation = float(np.linalg.norm(position - start_position))
        rotation = 2.0 * float(
            np.arccos(
                np.clip(abs(float(start_orientation @ orientation)), 0.0, 1.0)
            )
        )
        goal_pool.append(
            (
                goal_q.copy(),
                target_action.copy(),
                float(goal_clearance),
                translation,
                rotation,
            )
        )
        profile = _release_path_profile(
            native,
            np.stack((start_q, goal_q)),
            planning_pairs,
            release_goal_clearance,
            config,
        )
        if profile is not None:
            path, minimum_progress, profile_samples = profile
            score = (
                float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                + translation
                + 0.15 * rotation
            )
            candidates.append(
                _ReleaseCandidate(
                    path=path,
                    target_action=target_action.copy(),
                    backend="release_direct",
                    score=score,
                    translation=translation,
                    rotation=rotation,
                    minimum_progress=minimum_progress,
                    profile_samples=profile_samples,
                )
            )
        if len(candidates) >= 3 * config.release_shortlist_size:
            break

    if search_mode == "expanded" and time.monotonic() < deadline:
        candidates.extend(
            _release_control_tree_candidates(
                native,
                start_q,
                action,
                adapter,
                planning_pairs,
                escape_biases,
                config,
                rng,
                deadline=deadline,
            )
        )

    if len(candidates) < config.release_shortlist_size:
        goal_pool.sort(
            key=lambda item: (
                float(np.linalg.norm(item[0] - start_q)) + item[3] + 0.15 * item[4]
            )
        )
        for goal_q, target_action, _clearance, translation, rotation in goal_pool:
            if time.monotonic() >= deadline:
                break
            if any(
                float(np.linalg.norm(goal_q - item.path[-1])) < 0.02
                for item in candidates
            ):
                continue
            if not ompl_seeded:
                from ompl import util as ou

                ou.RNG.setSeed(planner_seed)
                ompl_seeded = True
            path = _ompl_geometric_candidate(
                native,
                start_q,
                goal_q,
                lower,
                upper,
                config,
                deadline=deadline,
            )
            if path is None:
                continue
            profile = _release_path_profile(
                native,
                path,
                planning_pairs,
                release_goal_clearance,
                config,
            )
            if profile is None:
                continue
            path, minimum_progress, profile_samples = profile
            score = (
                float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                + translation
                + 0.15 * rotation
            )
            candidates.append(
                _ReleaseCandidate(
                    path=path,
                    target_action=target_action.copy(),
                    backend="release_ompl_rrtconnect",
                    score=score,
                    translation=translation,
                    rotation=rotation,
                    minimum_progress=minimum_progress,
                    profile_samples=profile_samples,
                )
            )
            if len(candidates) >= 3 * config.release_shortlist_size:
                break

    if (
        search_mode == "direct"
        and len(candidates) < config.release_shortlist_size
        and time.monotonic() < deadline
    ):
        candidates.extend(
            _release_control_tree_candidates(
                native,
                start_q,
                action,
                adapter,
                planning_pairs,
                escape_biases,
                config,
                rng,
                deadline=deadline,
            )
        )

    if not candidates:
        raise FR3TransitPlanningFailure(
            "geometry-first release search found no path to full clearance"
        )
    candidates.sort(key=lambda item: item.score)
    candidate_directions: dict[int, np.ndarray] = {}
    for candidate in candidates:
        position, _ = adapter.target_pose_world(candidate.target_action)
        displacement = position - start_position
        norm = float(np.linalg.norm(displacement))
        candidate_directions[id(candidate)] = (
            displacement / norm
            if norm > 1e-9
            else np.zeros(3, dtype=float)
        )
    # Shortlisting is intentionally diverse in Cartesian escape direction.
    # Joint-distance-only diversity can still return variants of the same
    # locally jammed retreat, which wastes the expensive controller rollouts.
    selected: list[_ReleaseCandidate] = []
    for backend in (
        "release_direct",
        "release_directed_control_tree",
        "release_ompl_rrtconnect",
    ):
        backend_candidates = [
            candidate for candidate in candidates if candidate.backend == backend
        ]
        if backend_candidates:
            selected.append(min(backend_candidates, key=lambda item: item.score))
        if len(selected) >= config.release_shortlist_size:
            break
    remaining = [
        candidate
        for candidate in candidates
        if all(candidate is not chosen for chosen in selected)
    ]
    best_score = max(abs(float(candidates[0].score)), 1e-6)
    while remaining and len(selected) < config.release_shortlist_size:
        def diversity_value(
            candidate: _ReleaseCandidate,
        ) -> float:
            direction = candidate_directions[id(candidate)]
            directional = min(
                1.0
                - float(
                    np.clip(
                        direction @ candidate_directions[id(prior)],
                        -1.0,
                        1.0,
                    )
                )
                for prior in selected
            )
            joint = min(
                float(np.linalg.norm(candidate.path[-1] - prior.path[-1]))
                for prior in selected
            )
            joint = min(
                1.0,
                joint / config.release_diversity_joint_distance,
            )
            quality_penalty = 0.10 * max(
                0.0,
                float(candidate.score) / best_score - 1.0,
            )
            return directional + 0.25 * joint - quality_penalty

        chosen = max(remaining, key=diversity_value)
        selected.append(chosen)
        remaining = [
            candidate for candidate in remaining if candidate is not chosen
        ]

    elapsed = time.monotonic() - started
    release_limits = replace(
        DEFAULT_FR3_SAFETY_LIMITS,
        nominal_linear_velocity=min(
            DEFAULT_FR3_SAFETY_LIMITS.nominal_linear_velocity,
            config.maximum_linear_velocity,
        ),
        nominal_angular_velocity=min(
            DEFAULT_FR3_SAFETY_LIMITS.nominal_angular_velocity,
            config.maximum_angular_velocity,
        ),
    )
    plans: list[FR3TransitPlan] = []
    for rank, candidate in enumerate(selected, start=1):
        raw_path = candidate.path
        target_action = candidate.target_action
        target_position, _ = adapter.target_pose_world(target_action)
        displacement = target_position - start_position
        displacement_norm = float(np.linalg.norm(displacement))
        path = _downsample_path(raw_path, config.maximum_waypoints)
        minimum_progress = candidate.minimum_progress
        profile_samples = candidate.profile_samples
        if path.shape[0] < raw_path.shape[0]:
            profile = _release_path_profile(
                native,
                path,
                planning_pairs,
                release_goal_clearance,
                config,
            )
            if profile is None:
                continue
            path, minimum_progress, profile_samples = profile
        actions = _actions_from_path(native, path, adapter, action, target_action)
        actions[:, 6] = float(action[6])
        durations = tuple(
            max(
                0.10,
                minimum_pose_segment_duration(
                    adapter,
                    left,
                    right,
                    release_limits,
                ),
            )
            for left, right in zip(actions[:-1], actions[1:])
        )
        goal_valid, goal_clearance = native.check_release_goal(
            np.ascontiguousarray(path[-1])
        )
        if not goal_valid:
            continue
        plans.append(
            FR3TransitPlan(
                waypoint_actions=actions,
                segment_durations=durations,
                metrics={
                    "planning_backend": candidate.backend,
                    "planning_space": "fr3_joint_state_cartesian_controls",
                    "planning_dynamics": "kinematics_only",
                    "planning_clearance": planning_clearance,
                    "release_required_goal_clearance": release_goal_clearance,
                    "execution_clearance": config.execution_clearance,
                    "planning_time": elapsed,
                    "planning_state_checks": int(native.state_checks()),
                    "planning_raw_waypoints": int(raw_path.shape[0]),
                    "planning_waypoints": int(actions.shape[0]),
                    "planning_attempts": 1,
                    "planning_seed": planner_seed,
                    "release_geometry_search": True,
                    "release_search_mode": search_mode,
                    "release_path_tracking_reserve": (
                        config.release_path_tracking_reserve
                    ),
                    "release_goal_tracking_reserve": (
                        config.release_goal_tracking_reserve
                    ),
                    "release_maximum_linear_velocity": (
                        release_limits.nominal_linear_velocity
                    ),
                    "release_maximum_angular_velocity": (
                        release_limits.nominal_angular_velocity
                    ),
                    "release_aperture_fixed": True,
                    "release_aperture": float(action[6]),
                    "release_samples_evaluated": samples_evaluated,
                    "release_feasible_goals": len(goal_pool),
                    "release_feasible_paths": len(candidates),
                    "release_monitored_pair_count": len(planning_pairs),
                    "release_entry_relaxed_pair_count": (
                        entry_relaxed_pair_count
                    ),
                    "release_shortlist_rank": rank,
                    "release_shortlist_size": len(selected),
                    "release_path_score": candidate.score,
                    "release_path_minimum_progress": (
                        minimum_progress
                    ),
                    "release_path_profile_samples": profile_samples,
                    "release_translation": candidate.translation,
                    "release_translation_direction": (
                        displacement / max(displacement_norm, 1e-12)
                    ).tolist(),
                    "release_rotation": candidate.rotation,
                    "release_goal_clearance": float(goal_clearance),
                },
            )
        )
    if not plans:
        raise FR3TransitPlanningFailure(
            "geometry-first release candidates lost their terminal clearance"
        )
    return tuple(plans)


def plan_fr3_transit(
    model: mujoco.MjModel,
    initial_state: np.ndarray,
    adapter: FR3OSCAdapter,
    start_action: np.ndarray,
    target_action: np.ndarray,
    *,
    guide_actions: Sequence[np.ndarray] = (),
    handoff_orientation_candidates: Sequence[np.ndarray] = (),
    config: FR3TransitPlanningConfig = DEFAULT_FR3_TRANSIT_PLANNING_CONFIG,
    seed: int,
    handoff_tool_geom_ids: Sequence[int] = (),
    handoff_target_geom_ids: Sequence[int] = (),
) -> FR3TransitPlan:
    """Plan a collision-free path to a safe next-primitive handoff.

    ``target_action`` is the desired boundary pose and may itself be in
    contact.  When tool/target geometry is supplied, the planner projects that
    pose to the nearest sampled pose satisfying the smaller handoff clearance;
    declared task-equivalent attitudes are checked before translation, and all
    unrelated robot/environment pairs retain the global path clearance.
    """

    state = np.asarray(initial_state, dtype=float)
    if state.shape != (model.nq + model.nv,) or not np.all(np.isfinite(state)):
        raise ValueError("FR3 planning initial_state has the wrong shape or values")
    start_value = np.asarray(start_action, dtype=float)
    target_value = np.asarray(target_action, dtype=float)
    if start_value.shape != (adapter.action_dim,) or target_value.shape != (
        adapter.action_dim,
    ):
        raise ValueError("FR3 planning actions must have shape (7,)")
    if abs(float(start_value[6] - target_value[6])) > 1e-12:
        raise ValueError(
            "contact-free spatial planning requires fixed gripper aperture"
        )
    planner_seed = _planning_seed(seed)

    handoff_tool = tuple(map(int, handoff_tool_geom_ids))
    handoff_target = tuple(map(int, handoff_target_geom_ids))
    if bool(handoff_tool) != bool(handoff_target):
        raise ValueError(
            "handoff tool and target geom groups must be supplied together"
        )
    handoff_pairs = (
        (
            AllowedProximityPair(
                handoff_tool,
                handoff_target,
                minimum_clearance=config.handoff_clearance,
            ),
        )
        if handoff_tool
        else ()
    )
    safety_spec = FR3SafetySpec.from_model(
        model,
        adapter,
        allowed_proximity_pairs=handoff_pairs,
    )
    native_type = _load_native_type()
    native = native_type(
        model,
        np.ascontiguousarray(state[: model.nq]),
        _native_config(adapter, safety_spec, config),
    )
    start_q = np.asarray(native.arm_configuration(), dtype=float)
    if not native.is_valid(np.ascontiguousarray(start_q)):
        _, clearance = native.check_state(np.ascontiguousarray(start_q))
        raise FR3TransitPlanningFailure(
            "contact-free planning started before a certified release "
            f"(buffered clearance {float(clearance):.6f} m)"
        )

    lower, upper = _arm_joint_bounds(model, adapter, config.joint_margin)
    started = time.monotonic()
    handoff_goal_candidates: tuple[_HandoffPoseEvaluation, ...] = ()
    if handoff_pairs:
        (
            projected_target,
            goal_q,
            projection_metrics,
            handoff_goal_candidates,
        ) = _project_handoff_goal(
            native,
            start_q,
            lower,
            upper,
            adapter,
            target_value,
            handoff_orientation_candidates,
            config,
        )
    else:
        goal_pose = _pose_from_action(adapter, target_value)
        seeds = (
            start_q,
            np.clip(np.asarray(adapter.nullspace_configuration), lower, upper),
            np.clip(
                0.5
                * (
                    start_q
                    + np.asarray(adapter.nullspace_configuration)
                ),
                lower,
                upper,
            ),
        )
        goal_q = None
        kinematic_goal_found = False
        best_goal_clearance = -math.inf
        for ik_seed in seeds:
            result = _solve_pose(
                native,
                ik_seed,
                goal_pose,
                config,
            )
            if not result.converged:
                continue
            kinematic_goal_found = True
            valid, clearance = native.check_state(
                np.ascontiguousarray(result.q)
            )
            best_goal_clearance = max(best_goal_clearance, float(clearance))
            if valid:
                goal_q = result.q
                break
        if goal_q is None:
            detail = (
                f" (best clearance {best_goal_clearance:.6f} m)"
                if kinematic_goal_found and np.isfinite(best_goal_clearance)
                else ""
            )
            raise FR3TransitPlanningFailure(
                "the requested precontact pose has no collision-free FR3 "
                + (
                    "configuration"
                    if kinematic_goal_found
                    else "kinematic solution"
                )
                + detail
            )
        projected_target = target_value.copy()
        projection_metrics = {
            "handoff_projection_required": False,
            "handoff_desired_action": target_value.tolist(),
            "handoff_projected_action": target_value.tolist(),
            "handoff_projection_translation": 0.0,
            "handoff_projection_orientation": 0.0,
            "handoff_projection_candidates": 0,
            "handoff_required_clearance": config.clearance,
        }

    guide = np.asarray(tuple(guide_actions), dtype=float)
    if guide.size == 0:
        guide = projected_target[None]
    elif guide.ndim != 2 or guide.shape[1] != adapter.action_dim:
        raise ValueError("guide_actions must contain seven-channel actions")
    elif np.allclose(guide[-1], target_value, rtol=0.0, atol=1e-12):
        guide = guide.copy()
        guide[-1] = projected_target
    if not np.allclose(guide[-1], projected_target, rtol=0.0, atol=1e-12):
        guide = np.concatenate((guide, projected_target[None]), axis=0)

    deadline = started + config.maximum_planning_time
    native.reset_counters()
    path = _guide_candidate(native, start_q, guide, adapter, config)
    method = "typed_guide"
    planning_attempts = 0
    if path is None and bool(
        projection_metrics.get("handoff_projected_direct_path_valid", False)
    ):
        path = np.stack((start_q, goal_q))
        method = "projected_direct_handoff"
    if path is None:
        from ompl import util as ou

        native.reset_counters()
        ou.RNG.setSeed(planner_seed)
        planning_goals = (
            tuple(
                (candidate.action, candidate.q, candidate)
                for candidate in handoff_goal_candidates
            )
            if handoff_goal_candidates
            else ((projected_target, goal_q, None),)
        )

        def activate_goal(index: int) -> None:
            nonlocal projected_target, goal_q
            action, configuration, evaluation = planning_goals[index]
            projected_target = np.asarray(action, dtype=float).copy()
            goal_q = np.asarray(configuration, dtype=float).copy()
            if handoff_pairs:
                projection_metrics[
                    "handoff_projection_selected_goal_rank"
                ] = index + 1
            if evaluation is None:
                return
            displacement = evaluation.offset
            translation = evaluation.translation
            projection_metrics.update(
                {
                    "handoff_projected_action": evaluation.action.tolist(),
                    "handoff_projection_translation": translation,
                    "handoff_projection_orientation": (
                        evaluation.orientation_change
                    ),
                    "handoff_projection_direction": (
                        displacement / translation
                        if translation > 1e-12
                        else np.zeros(3, dtype=float)
                    ).tolist(),
                    "handoff_projected_clearance": evaluation.clearance,
                    "handoff_projected_joint_margin": evaluation.joint_margin,
                    "handoff_projected_direct_path_valid": (
                        evaluation.direct_path_valid
                    ),
                }
            )

        for goal_index, (_, candidate_q, _) in enumerate(planning_goals):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            goals_left = len(planning_goals) - goal_index
            attempt_config = replace(
                config,
                planning_time=min(
                    config.planning_time,
                    remaining / goals_left,
                ),
            )
            planning_attempts += 1
            path = _ompl_geometric_candidate(
                native,
                start_q,
                candidate_q,
                lower,
                upper,
                attempt_config,
                deadline=deadline,
            )
            if path is not None:
                activate_goal(goal_index)
                break
        method = "ompl_geometric_rrtconnect"
        if path is None:
            method = "ompl_control_rrt"
            for goal_index, (_, candidate_q, _) in enumerate(planning_goals):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                goals_left = len(planning_goals) - goal_index
                attempt_config = replace(
                    config,
                    planning_time=min(
                        config.planning_time,
                        remaining / goals_left,
                    ),
                )
                planning_attempts += 1
                path = _ompl_candidate(
                    native,
                    start_q,
                    candidate_q,
                    lower,
                    upper,
                    attempt_config,
                    deadline=deadline,
                )
                if path is not None:
                    activate_goal(goal_index)
                    break
                if native.state_checks() >= config.maximum_state_checks:
                    break
    if path is None:
        handoff_detail = (
            "; handoff projection selected_direct_path="
            f"{projection_metrics.get('handoff_projected_direct_path_valid')}, "
            "direct_candidates="
            f"{projection_metrics.get('handoff_projection_direct_path_candidates')}, "
            "feasible_candidates="
            f"{projection_metrics.get('handoff_projection_feasible_candidates')}"
            if handoff_pairs
            else ""
        )
        raise FR3TransitPlanningFailure(
            "OMPL exhausted the FR3 contact-free planning budget"
            + handoff_detail
        )
    raw_waypoints = int(path.shape[0])
    path = _downsample_path(path, config.maximum_waypoints)
    actions = _actions_from_path(
        native,
        path,
        adapter,
        start_value,
        projected_target,
    )
    durations = tuple(
        max(0.10, minimum_pose_segment_duration(adapter, left, right))
        for left, right in zip(actions[:-1], actions[1:])
    )
    elapsed = time.monotonic() - started
    return FR3TransitPlan(
        waypoint_actions=actions,
        segment_durations=durations,
        metrics={
            "planning_backend": method,
            "planning_space": "fr3_joint_state_cartesian_controls",
            "planning_dynamics": "kinematics_only",
            "planning_clearance": config.clearance,
            "execution_clearance": config.execution_clearance,
            "planning_time": elapsed,
            "planning_state_checks": int(native.state_checks()),
            "planning_raw_waypoints": raw_waypoints,
            "planning_waypoints": int(actions.shape[0]),
            "planning_attempts": planning_attempts,
            "planning_seed": planner_seed,
            **projection_metrics,
        },
    )


def check_fr3_transit_configuration(
    model: mujoco.MjModel,
    state: np.ndarray,
    adapter: FR3OSCAdapter,
    *,
    config: FR3TransitPlanningConfig = DEFAULT_FR3_TRANSIT_PLANNING_CONFIG,
) -> tuple[bool, float]:
    """Check one full-scene configuration with the native 10 mm buffer."""

    value = np.asarray(state, dtype=float)
    if value.shape != (model.nq + model.nv,) or not np.all(np.isfinite(value)):
        raise ValueError("FR3 planning state has the wrong shape or values")
    safety_spec = FR3SafetySpec.from_model(model, adapter)
    native = _load_native_type()(
        model,
        np.ascontiguousarray(value[: model.nq]),
        _native_config(adapter, safety_spec, config),
    )
    q = np.asarray(native.arm_configuration(), dtype=float)
    valid, clearance = native.check_state(np.ascontiguousarray(q))
    return bool(valid), float(clearance)


__all__ = [
    "DEFAULT_FR3_TRANSIT_PLANNING_CONFIG",
    "FR3ReleasePair",
    "FR3TransitPlan",
    "FR3TransitPlanningConfig",
    "FR3TransitPlanningFailure",
    "check_fr3_transit_configuration",
    "plan_fr3_release",
    "plan_fr3_transit",
]
