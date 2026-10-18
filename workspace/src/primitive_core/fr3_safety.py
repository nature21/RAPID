"""Runtime-owned FR3 motion timing and retained trajectory audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

import mujoco
import numpy as np

if TYPE_CHECKING:
    from primitive_core.common import CompileResult
    from primitive_core.fr3_osc import FR3OSCAdapter


@dataclass(frozen=True)
class FR3SafetyLimits:
    """Versioned timing and clearance values used outside the OSC."""

    version: str = "fr3-motion-audit/v4"
    minimum_environment_clearance: float = 0.005
    # MuJoCo contact generation and geom-distance queries can disagree by a
    # few floating-point ulps at a threshold.  This tolerance is zero-scale by
    # default and is enlarged only for the bounded inherited-contact release
    # window, where controller tracking can introduce sub-millimetre chatter.
    geometry_numerical_tolerance: float = 0.0
    nominal_linear_velocity: float = 0.35
    nominal_angular_velocity: float = 0.80
    nominal_linear_acceleration: float = 1.50
    nominal_angular_acceleration: float = 3.00
    nominal_gripper_velocity: float = 0.02
    maximum_closed_stop_penetration: float = 0.00025
    maximum_finger_synchronization_error: float = 0.00025

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("FR3 motion/audit profile version cannot be empty")
        positive = (
            "nominal_linear_velocity",
            "nominal_angular_velocity",
            "nominal_linear_acceleration",
            "nominal_angular_acceleration",
            "nominal_gripper_velocity",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not np.isfinite(self.minimum_environment_clearance)
            or self.minimum_environment_clearance < 0.0
        ):
            raise ValueError(
                "minimum_environment_clearance must be finite and nonnegative"
            )
        for name in (
            "geometry_numerical_tolerance",
            "maximum_closed_stop_penetration",
            "maximum_finger_synchronization_error",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


DEFAULT_FR3_SAFETY_LIMITS = FR3SafetyLimits()
_MINIMUM_JERK_VELOCITY_FACTOR = 1.875
_MINIMUM_JERK_ACCELERATION_FACTOR = 5.773502691896258


@dataclass(frozen=True)
class AllowedContactPair:
    """One phase-specific contact relation permitted by learned intent."""

    geom_ids_a: tuple[int, ...]
    geom_ids_b: tuple[int, ...]

    def __post_init__(self) -> None:
        left = tuple(self.geom_ids_a)
        right = tuple(self.geom_ids_b)
        if not left or not right:
            raise ValueError("allowed contact geom groups cannot be empty")
        if any(not isinstance(value, Integral) or value < 0 for value in left + right):
            raise ValueError("allowed contact geom ids must be nonnegative integers")
        object.__setattr__(self, "geom_ids_a", tuple(map(int, left)))
        object.__setattr__(self, "geom_ids_b", tuple(map(int, right)))

    def matches(self, first: int, second: int) -> bool:
        return (first in self.geom_ids_a and second in self.geom_ids_b) or (
            second in self.geom_ids_a and first in self.geom_ids_b
        )


@dataclass(frozen=True)
class AllowedProximityPair:
    """A contact-free pair with an optional phase-specific clearance.

    ``minimum_clearance=None`` retains the historical diagnostic-only
    exemption.  A finite value keeps the pair out of the general environment
    buffer while still enforcing its own smaller, non-contact margin.  This is
    used for a transit's final precontact handoff, never for the contact-rich
    primitive that follows it.
    """

    geom_ids_a: tuple[int, ...]
    geom_ids_b: tuple[int, ...]
    minimum_clearance: float | None = None

    def __post_init__(self) -> None:
        left = tuple(self.geom_ids_a)
        right = tuple(self.geom_ids_b)
        if not left or not right:
            raise ValueError("allowed proximity geom groups cannot be empty")
        if any(not isinstance(value, Integral) or value < 0 for value in left + right):
            raise ValueError("allowed proximity geom ids must be nonnegative integers")
        object.__setattr__(self, "geom_ids_a", tuple(map(int, left)))
        object.__setattr__(self, "geom_ids_b", tuple(map(int, right)))
        if self.minimum_clearance is not None:
            clearance = float(self.minimum_clearance)
            if not np.isfinite(clearance) or clearance < 0.0:
                raise ValueError(
                    "proximity-pair minimum_clearance must be finite and "
                    "nonnegative"
                )
            object.__setattr__(self, "minimum_clearance", clearance)

    def matches(self, first: int, second: int) -> bool:
        return (first in self.geom_ids_a and second in self.geom_ids_b) or (
            second in self.geom_ids_a and first in self.geom_ids_b
        )


def _descends_from(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return False


def _has_articulated_ancestor(
    model: mujoco.MjModel,
    body_id: int,
    ancestor_id: int,
) -> bool:
    current = int(body_id)
    while current > 0:
        if int(model.body_jntnum[current]) > 0:
            return True
        if current == ancestor_id:
            break
        current = int(model.body_parentid[current])
    return False


def _collision_enabled(model: mujoco.MjModel, geom_id: int) -> bool:
    return bool(
        int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
    )


@dataclass(frozen=True)
class FR3SafetySpec:
    """Resolved geometry and permitted contacts for retained auditing."""

    robot_geom_ids: tuple[int, ...]
    moving_robot_geom_ids: tuple[int, ...]
    allowed_contact_pairs: tuple[AllowedContactPair, ...] = ()
    allowed_proximity_pairs: tuple[AllowedProximityPair, ...] = ()
    limits: FR3SafetyLimits = DEFAULT_FR3_SAFETY_LIMITS
    permitted_gripper_contact_pairs: tuple[AllowedContactPair, ...] = ()
    gripper_qpos_addresses: tuple[int, int] | None = None
    gripper_joint_lower_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        robot = tuple(map(int, self.robot_geom_ids))
        moving = tuple(map(int, self.moving_robot_geom_ids))
        if not robot or not moving:
            raise ValueError("FR3 safety requires robot collision geoms")
        if any(value < 0 for value in robot + moving):
            raise ValueError("FR3 safety geom ids must be nonnegative")
        if not set(moving).issubset(robot):
            raise ValueError("moving robot geoms must be a subset of robot geoms")
        object.__setattr__(self, "robot_geom_ids", tuple(dict.fromkeys(robot)))
        object.__setattr__(
            self,
            "moving_robot_geom_ids",
            tuple(dict.fromkeys(moving)),
        )
        object.__setattr__(
            self,
            "allowed_contact_pairs",
            tuple(self.allowed_contact_pairs),
        )
        object.__setattr__(
            self,
            "allowed_proximity_pairs",
            tuple(self.allowed_proximity_pairs),
        )
        object.__setattr__(
            self,
            "permitted_gripper_contact_pairs",
            tuple(self.permitted_gripper_contact_pairs),
        )
        if any(
            not isinstance(pair, AllowedContactPair)
            for pair in self.allowed_contact_pairs
        ):
            raise TypeError("allowed_contact_pairs must contain AllowedContactPair")
        if any(
            not isinstance(pair, AllowedProximityPair)
            for pair in self.allowed_proximity_pairs
        ):
            raise TypeError(
                "allowed_proximity_pairs must contain AllowedProximityPair"
            )
        if any(
            not isinstance(pair, AllowedContactPair)
            for pair in self.permitted_gripper_contact_pairs
        ):
            raise TypeError(
                "permitted_gripper_contact_pairs must contain AllowedContactPair"
            )
        if (self.gripper_qpos_addresses is None) != (
            self.gripper_joint_lower_limits is None
        ):
            raise ValueError(
                "gripper addresses and lower limits must be supplied together"
            )
        if self.gripper_qpos_addresses is not None:
            addresses = tuple(map(int, self.gripper_qpos_addresses))
            lower = tuple(map(float, self.gripper_joint_lower_limits or ()))
            if len(addresses) != 2 or len(lower) != 2:
                raise ValueError("FR3 safety requires two gripper joint entries")
            if any(value < 0 for value in addresses) or not np.all(
                np.isfinite(lower)
            ):
                raise ValueError("invalid gripper joint audit metadata")
            object.__setattr__(self, "gripper_qpos_addresses", addresses)
            object.__setattr__(self, "gripper_joint_lower_limits", lower)
        if not isinstance(self.limits, FR3SafetyLimits):
            raise TypeError("limits must be FR3SafetyLimits")

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        adapter: "FR3OSCAdapter",
        *,
        allowed_contact_pairs: Sequence[AllowedContactPair] = (),
        allowed_proximity_pairs: Sequence[AllowedProximityPair] = (),
        limits: FR3SafetyLimits = DEFAULT_FR3_SAFETY_LIMITS,
    ) -> "FR3SafetySpec":
        base_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            adapter.config.base_body_name,
        )
        if base_id < 0:
            raise ValueError("FR3 safety could not resolve the configured base body")
        robot: list[int] = []
        moving: list[int] = []
        for geom_id in range(model.ngeom):
            if not _collision_enabled(model, geom_id):
                continue
            body_id = int(model.geom_bodyid[geom_id])
            if not _descends_from(model, body_id, base_id):
                continue
            robot.append(geom_id)
            if _has_articulated_ancestor(model, body_id, base_id):
                moving.append(geom_id)
        finger_geom_groups: list[tuple[int, ...]] = []
        for body_name in ("left_finger", "right_finger"):
            finger_body = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if finger_body < 0:
                raise ValueError(f"FR3 safety could not resolve {body_name}")
            finger_geom_groups.append(
                tuple(
                    geom_id
                    for geom_id in robot
                    if _descends_from(
                        model,
                        int(model.geom_bodyid[geom_id]),
                        finger_body,
                    )
                )
            )
        if any(not group for group in finger_geom_groups):
            raise ValueError("FR3 safety requires collision geoms on both fingers")
        address_pairs = adapter.gripper_joint_address_pairs(model)
        gripper_joint_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in adapter.config.gripper_joint_names
        ]
        spec = cls(
            robot_geom_ids=tuple(robot),
            moving_robot_geom_ids=tuple(moving),
            allowed_contact_pairs=tuple(allowed_contact_pairs),
            allowed_proximity_pairs=tuple(allowed_proximity_pairs),
            limits=limits,
            permitted_gripper_contact_pairs=(
                AllowedContactPair(
                    finger_geom_groups[0],
                    finger_geom_groups[1],
                ),
            ),
            gripper_qpos_addresses=tuple(pair[0] for pair in address_pairs),
            gripper_joint_lower_limits=tuple(
                float(model.jnt_range[joint_id, 0])
                for joint_id in gripper_joint_ids
            ),
        )
        for pair in (*spec.allowed_contact_pairs, *spec.allowed_proximity_pairs):
            for geom_id in pair.geom_ids_a + pair.geom_ids_b:
                if geom_id >= model.ngeom:
                    raise ValueError(
                        f"allowed contact/proximity geom id {geom_id} exceeds "
                        f"model.ngeom={model.ngeom}"
                    )
            if not (
                set(pair.geom_ids_a).intersection(spec.robot_geom_ids)
                or set(pair.geom_ids_b).intersection(spec.robot_geom_ids)
            ):
                raise ValueError(
                    "an allowed contact/proximity pair must include an FR3 geom"
                )
        return spec


class FR3SafetyViolation(RuntimeError):
    """A retained FR3 trajectory failed motion timing or whole-arm auditing."""

    def __init__(
        self,
        message: str,
        *,
        result: "CompileResult | None" = None,
    ) -> None:
        super().__init__(message)
        self.result = result


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return 2.0 * math.acos(float(np.clip(abs(left @ right), 0.0, 1.0)))


def minimum_pose_segment_duration(
    adapter: "FR3OSCAdapter",
    start: np.ndarray,
    end: np.ndarray,
    limits: FR3SafetyLimits = DEFAULT_FR3_SAFETY_LIMITS,
) -> float:
    """Return a conservative minimum for a quintic pose/gripper segment."""

    start_position, start_quaternion = adapter.target_pose_world(start)
    end_position, end_quaternion = adapter.target_pose_world(end)
    distance = float(np.linalg.norm(end_position - start_position))
    euler_delta = (np.asarray(end[3:6]) - np.asarray(start[3:6]) + np.pi) % (
        2.0 * np.pi
    ) - np.pi
    angle = max(
        _quaternion_distance(start_quaternion, end_quaternion),
        float(np.linalg.norm(euler_delta)),
    )
    aperture = abs(float(end[6] - start[6]))
    # Maximum derivatives of 10s^3 - 15s^4 + 6s^5 on [0, 1].
    return max(
        _MINIMUM_JERK_VELOCITY_FACTOR
        * distance
        / limits.nominal_linear_velocity,
        math.sqrt(
            _MINIMUM_JERK_ACCELERATION_FACTOR
            * distance
            / limits.nominal_linear_acceleration
        ),
        _MINIMUM_JERK_VELOCITY_FACTOR
        * angle
        / limits.nominal_angular_velocity,
        math.sqrt(
            _MINIMUM_JERK_ACCELERATION_FACTOR
            * angle
            / limits.nominal_angular_acceleration
        ),
        _MINIMUM_JERK_VELOCITY_FACTOR
        * aperture
        / limits.nominal_gripper_velocity,
        0.0,
    )


def minimum_pose_trajectory_duration(
    adapter: "FR3OSCAdapter",
    knots: np.ndarray,
    knot_fractions: np.ndarray,
    limits: FR3SafetyLimits = DEFAULT_FR3_SAFETY_LIMITS,
) -> float:
    """Return the total horizon needed by every normalized knot interval."""

    actions = np.asarray(knots, dtype=float)
    fractions = np.asarray(knot_fractions, dtype=float)
    required = 0.0
    for index in range(actions.shape[0] - 1):
        interval = float(fractions[index + 1] - fractions[index])
        segment = minimum_pose_segment_duration(
            adapter,
            actions[index],
            actions[index + 1],
            limits,
        )
        required = max(required, segment / interval)
    return required


def _interpolate_pose_action(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    delta[3:6] = (delta[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    action = np.asarray(start, dtype=float) + float(fraction) * delta
    action[3:6] = (action[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    return action


def _maximum_minimum_jerk_displacement(
    duration: float,
    velocity: float,
    acceleration: float | None = None,
) -> float:
    maximum = duration * velocity / _MINIMUM_JERK_VELOCITY_FACTOR
    if acceleration is not None:
        maximum = min(
            maximum,
            duration**2
            * acceleration
            / _MINIMUM_JERK_ACCELERATION_FACTOR,
        )
    # Stay just inside the analytical boundary so ordinary floating-point
    # roundoff does not send every limited segment through fallback bisection.
    return maximum * (1.0 - 1e-12)


def _limit_pose_action_for_duration(
    adapter: "FR3OSCAdapter",
    start: np.ndarray,
    requested_end: np.ndarray,
    duration: float,
    limits: FR3SafetyLimits,
) -> np.ndarray:
    delta = np.asarray(requested_end, dtype=float) - np.asarray(start, dtype=float)
    delta[3:6] = (delta[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    limited = np.asarray(start, dtype=float).copy()

    distance = float(np.linalg.norm(delta[:3]))
    maximum_distance = _maximum_minimum_jerk_displacement(
        duration,
        limits.nominal_linear_velocity,
        limits.nominal_linear_acceleration,
    )
    translation_scale = min(1.0, maximum_distance / max(distance, 1e-15))
    limited[:3] += translation_scale * delta[:3]

    start_quaternion = adapter.target_pose_world(start)[1]
    end_quaternion = adapter.target_pose_world(requested_end)[1]
    angle = max(
        _quaternion_distance(start_quaternion, end_quaternion),
        float(np.linalg.norm(delta[3:6])),
    )
    maximum_angle = _maximum_minimum_jerk_displacement(
        duration,
        limits.nominal_angular_velocity,
        limits.nominal_angular_acceleration,
    )
    rotation_scale = min(1.0, maximum_angle / max(angle, 1e-15))
    limited[3:6] += rotation_scale * delta[3:6]
    limited[3:6] = (limited[3:6] + np.pi) % (2.0 * np.pi) - np.pi

    aperture = abs(float(delta[6]))
    maximum_aperture = _maximum_minimum_jerk_displacement(
        duration,
        limits.nominal_gripper_velocity,
    )
    gripper_scale = min(1.0, maximum_aperture / max(aperture, 1e-15))
    limited[6] += gripper_scale * delta[6]

    if minimum_pose_segment_duration(adapter, start, limited, limits) <= duration:
        return limited

    # Euler interpolation is not exactly a quaternion geodesic. In the rare
    # case where the analytic component limits remain too aggressive, retain
    # the largest endpoint verified safe by a short fallback bisection.
    lower = 0.0
    upper = 1.0
    feasible_end = np.asarray(start, dtype=float).copy()
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        candidate_end = _interpolate_pose_action(start, limited, middle)
        candidate_required = minimum_pose_segment_duration(
            adapter,
            start,
            candidate_end,
            limits,
        )
        if candidate_required <= duration:
            lower = middle
            feasible_end = candidate_end
        else:
            upper = middle
    return feasible_end


def enforce_pose_trajectory_timing(
    adapter: "FR3OSCAdapter",
    knots: np.ndarray,
    knot_fractions: np.ndarray,
    horizon: float,
    limits: FR3SafetyLimits = DEFAULT_FR3_SAFETY_LIMITS,
) -> tuple[np.ndarray, int]:
    """Project sampled FR3 knots into the timing envelope for ``horizon``.

    The leading dimensions of ``knots`` are treated as independent trajectories.
    Each later knot is moved toward the preceding, already-projected knot until
    its minimum-jerk segment respects the shared linear, angular, acceleration,
    and gripper limits. The input is never mutated.
    """

    values = np.asarray(knots, dtype=float)
    fractions = np.asarray(knot_fractions, dtype=float).reshape(-1)
    if values.ndim < 2 or values.shape[-1] != adapter.action_dim:
        raise ValueError("FR3 timing knots must end in (num_knots, action_dim)")
    if values.shape[-2] < 2 or fractions.shape != (values.shape[-2],):
        raise ValueError("FR3 timing requires one fraction per knot")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(fractions))
        or not np.isfinite(horizon)
        or horizon <= 0.0
    ):
        raise ValueError("FR3 timing inputs must be finite with a positive horizon")
    if (
        not np.isclose(fractions[0], 0.0)
        or not np.isclose(fractions[-1], 1.0)
        or np.any(np.diff(fractions) <= 0.0)
    ):
        raise ValueError(
            "FR3 timing fractions must be strictly increasing and span [0, 1]"
        )

    projected = values.copy()
    trajectories = projected.reshape((-1, values.shape[-2], values.shape[-1]))
    limited_segments = 0
    for trajectory in trajectories:
        for index in range(trajectory.shape[0] - 1):
            start = trajectory[index].copy()
            requested_end = trajectory[index + 1].copy()
            available = float(
                horizon * (fractions[index + 1] - fractions[index])
            )
            required = minimum_pose_segment_duration(
                adapter,
                start,
                requested_end,
                limits,
            )
            if required <= available:
                continue

            trajectory[index + 1] = _limit_pose_action_for_duration(
                adapter,
                start,
                requested_end,
                available,
                limits,
            )
            limited_segments += 1

    return np.ascontiguousarray(projected), limited_segments


def _collision_pair_enabled(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool(
        int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
        or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])
    )


@dataclass(frozen=True)
class FR3TrajectorySafetyAudit:
    valid: bool
    self_collision_steps: int
    unapproved_contact_steps: int
    minimum_environment_clearance: float
    minimum_declared_proximity_clearance: float = math.inf
    minimum_declared_proximity_margin: float = math.inf
    declared_proximity_clearance_valid: bool = True
    minimum_environment_clearance_pair: tuple[str, str] | None = None
    minimum_environment_clearance_step: int | None = None
    dominant_unapproved_contact_pair: tuple[str, str] | None = None
    permitted_fingertip_contact_steps: int = 0
    maximum_closed_stop_penetration: float = 0.0
    maximum_finger_synchronization_error: float = 0.0
    geometry_numerical_tolerance: float = 0.0

    @property
    def minimum_enforced_environment_clearance(self) -> float:
        """Explicit name for the clearance-enforced pair minimum."""

        return self.minimum_environment_clearance


def audit_fr3_trajectory(
    model: mujoco.MjModel,
    states: np.ndarray,
    spec: FR3SafetySpec,
    *,
    clearance_stride: int = 1,
) -> FR3TrajectorySafetyAudit:
    """Exactly replay one retained state trajectory for whole-arm safety."""

    if clearance_stride < 1:
        raise ValueError("clearance_stride must be positive")
    state_values = np.asarray(states, dtype=float)
    if state_values.ndim != 2 or state_values.shape[1] != model.nq + model.nv:
        raise ValueError("FR3 safety audit states have the wrong shape")
    if not np.all(np.isfinite(state_values)):
        return FR3TrajectorySafetyAudit(
            valid=False,
            self_collision_steps=0,
            unapproved_contact_steps=0,
            minimum_environment_clearance=-math.inf,
            minimum_declared_proximity_clearance=-math.inf,
            minimum_declared_proximity_margin=-math.inf,
            declared_proximity_clearance_valid=False,
        )

    robot = set(spec.robot_geom_ids)
    moving = tuple(spec.moving_robot_geom_ids)
    external = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in robot and _collision_enabled(model, geom_id)
    )
    clearance_pairs = tuple(
        (robot_geom, environment_geom)
        for robot_geom in moving
        for environment_geom in external
        if _collision_pair_enabled(model, robot_geom, environment_geom)
        and not any(
            pair.matches(robot_geom, environment_geom)
            for pair in spec.allowed_contact_pairs
        )
        and not any(
            pair.matches(robot_geom, environment_geom)
            for pair in spec.allowed_proximity_pairs
        )
    )
    proximity_pairs = tuple(
        (
            robot_geom,
            environment_geom,
            tuple(
                pair
                for pair in spec.allowed_proximity_pairs
                if pair.matches(robot_geom, environment_geom)
            ),
        )
        for robot_geom in moving
        for environment_geom in external
        if _collision_pair_enabled(model, robot_geom, environment_geom)
        and any(
            pair.matches(robot_geom, environment_geom)
            for pair in spec.allowed_proximity_pairs
        )
    )
    data = mujoco.MjData(model)
    self_collision_steps = 0
    unapproved_contact_steps = 0
    unapproved_contact_pairs: Counter[tuple[str, str]] = Counter()
    minimum_clearance = math.inf
    minimum_clearance_pair: tuple[str, str] | None = None
    minimum_clearance_step: int | None = None
    minimum_proximity_clearance = math.inf
    minimum_proximity_margin = math.inf
    proximity_clearance_valid = True
    permitted_fingertip_contact_steps = 0
    maximum_stop_penetration = 0.0
    maximum_finger_mismatch = 0.0
    distance_limit = float(spec.limits.minimum_environment_clearance)
    geometry_tolerance = float(spec.limits.geometry_numerical_tolerance)
    for step, state in enumerate(state_values):
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq :]
        # Retained-state certification needs positions and contacts, not
        # dynamics.  This avoids the much more expensive full forward pass.
        mujoco.mj_fwdKinematics(model, data)
        mujoco.mj_collision(model, data)
        self_contact = False
        permitted_fingertip_contact = False
        unapproved = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            # MuJoCo can retain a contact record at a positive distance and
            # can report tiny negative distances from solver/geometry
            # roundoff.  Clearance checks below remain authoritative; count a
            # collision only when penetration exceeds the phase's explicitly
            # bounded tolerance.
            if float(contact.dist) > -geometry_tolerance:
                continue
            first = int(contact.geom1)
            second = int(contact.geom2)
            first_robot = first in robot
            second_robot = second in robot
            if first_robot and second_robot:
                if any(
                    pair.matches(first, second)
                    for pair in spec.permitted_gripper_contact_pairs
                ):
                    permitted_fingertip_contact = True
                else:
                    self_contact = True
            elif first_robot or second_robot:
                if not any(
                    pair.matches(first, second)
                    for pair in spec.allowed_contact_pairs
                ):
                    unapproved = True
                    first_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, first
                    ) or f"geom_{first}"
                    second_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, second
                    ) or f"geom_{second}"
                    pair_names = tuple(sorted((first_name, second_name)))
                    unapproved_contact_pairs[pair_names] += 1
        self_collision_steps += int(self_contact)
        permitted_fingertip_contact_steps += int(permitted_fingertip_contact)
        unapproved_contact_steps += int(unapproved)

        if spec.gripper_qpos_addresses is not None:
            finger_positions = state[
                np.asarray(spec.gripper_qpos_addresses, dtype=int)
            ]
            lower_limits = np.asarray(
                spec.gripper_joint_lower_limits,
                dtype=float,
            )
            maximum_stop_penetration = max(
                maximum_stop_penetration,
                float(np.max(np.maximum(lower_limits - finger_positions, 0.0))),
            )
            maximum_finger_mismatch = max(
                maximum_finger_mismatch,
                abs(float(finger_positions[0] - finger_positions[1])),
            )

        if step % clearance_stride == 0 or step == state_values.shape[0] - 1:
            for first, second in clearance_pairs:
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        first,
                        second,
                        distance_limit,
                        None,
                    )
                )
                if distance < minimum_clearance:
                    minimum_clearance = distance
                    first_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, first
                    ) or f"geom_{first}"
                    second_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, second
                    ) or f"geom_{second}"
                    minimum_clearance_pair = tuple(
                        sorted((first_name, second_name))
                    )
                    minimum_clearance_step = step
            for first, second, declarations in proximity_pairs:
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        first,
                        second,
                        distance_limit,
                        None,
                    )
                )
                minimum_proximity_clearance = min(
                    minimum_proximity_clearance,
                    distance,
                )
                required_clearances = tuple(
                    float(pair.minimum_clearance)
                    for pair in declarations
                    if pair.minimum_clearance is not None
                )
                if required_clearances:
                    required = max(required_clearances)
                    minimum_proximity_margin = min(
                        minimum_proximity_margin,
                        distance - required,
                    )
                    if distance < required - geometry_tolerance:
                        proximity_clearance_valid = False

    if not clearance_pairs:
        minimum_clearance = math.inf
    valid = bool(
        self_collision_steps == 0
        and unapproved_contact_steps == 0
        and minimum_clearance >= distance_limit - geometry_tolerance
        and proximity_clearance_valid
        and maximum_stop_penetration
        <= spec.limits.maximum_closed_stop_penetration
        and maximum_finger_mismatch
        <= spec.limits.maximum_finger_synchronization_error
    )
    return FR3TrajectorySafetyAudit(
        valid=valid,
        self_collision_steps=self_collision_steps,
        unapproved_contact_steps=unapproved_contact_steps,
        minimum_environment_clearance=minimum_clearance,
        minimum_declared_proximity_clearance=minimum_proximity_clearance,
        minimum_declared_proximity_margin=minimum_proximity_margin,
        declared_proximity_clearance_valid=proximity_clearance_valid,
        minimum_environment_clearance_pair=minimum_clearance_pair,
        minimum_environment_clearance_step=minimum_clearance_step,
        dominant_unapproved_contact_pair=(
            unapproved_contact_pairs.most_common(1)[0][0]
            if unapproved_contact_pairs
            else None
        ),
        permitted_fingertip_contact_steps=permitted_fingertip_contact_steps,
        maximum_closed_stop_penetration=maximum_stop_penetration,
        maximum_finger_synchronization_error=maximum_finger_mismatch,
        geometry_numerical_tolerance=geometry_tolerance,
    )


__all__ = [
    "AllowedContactPair",
    "AllowedProximityPair",
    "DEFAULT_FR3_SAFETY_LIMITS",
    "FR3SafetyLimits",
    "FR3SafetySpec",
    "FR3SafetyViolation",
    "FR3TrajectorySafetyAudit",
    "audit_fr3_trajectory",
    "enforce_pose_trajectory_timing",
    "minimum_pose_segment_duration",
    "minimum_pose_trajectory_duration",
]
