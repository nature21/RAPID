"""Typed relational transit policies and their shared FR3 realization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from numbers import Integral
from typing import Literal, Mapping, TypeAlias

import mujoco
import numpy as np

from primitive_core.common import CompileResult, ToolTrajectory
from primitive_core.profiling import timed_execution
from primitive_core.fr3_osc import FR3_CONTROLLER_STATE_NAMES, FR3OSCAdapter
from primitive_core.fr3_safety import (
    DEFAULT_FR3_SAFETY_LIMITS,
    AllowedContactPair,
    AllowedProximityPair,
    FR3SafetySpec,
    audit_fr3_trajectory,
    minimum_pose_segment_duration,
)
from primitive_core.fr3_transit import (
    FR3TransitResult,
    FR3TransitViolation,
    rollout_fr3_transit,
)
from primitive_core.fr3_transit_planner import (
    DEFAULT_FR3_TRANSIT_PLANNING_CONFIG,
    FR3ReleasePair,
    FR3TransitPlan,
    FR3TransitPlanningFailure,
    check_fr3_transit_configuration,
    plan_fr3_release,
    plan_fr3_transit,
)
from primitive_core.policy import BoundExecutionContext, GripperOperation


AxisName = Literal["x", "y", "z"]


# Release is a runtime-owned controller operation, not a learned transit
# duration.  The inherited window remains the strategy's semantic permission
# to release contact; these values give the hardware-parity equilibrium filter
# enough time to realize that permission without weakening the certificate.
_MINIMUM_RELEASE_MOVEMENT_DURATION = 1.25
_MINIMUM_RELEASE_CONTACT_TIMEOUT = 1.50
_RELEASE_SETTLE_MAXIMUM_DURATION = 0.75
_RELEASE_STABILIZATION_MAXIMUM_DURATION = 2.00
_RELEASE_STABLE_SEPARATION_DURATION = 0.050
_RELEASE_STABLE_SEPARATION_GAP = 0.0005
_RELEASE_PENETRATION_HYSTERESIS = 0.00025
_RELEASE_CONTACT_DISTANCE_EPSILON = 0.0001
_RELEASE_CLEARANCE_NUMERICAL_TOLERANCE = 0.00005
_MAXIMUM_RELEASE_CONTROLLER_ROLLOUTS = 8
_MAXIMUM_RELEASE_REPLANS = 1

# A frozen random-stream identity, not a public schema or project name.
# Renaming this changes physical release/handoff paths for the same user seed.
_TRANSIT_SEED_DOMAIN_V1 = "nppl.fr3_transit.v1"


def _derive_transit_planning_seed(
    transit_seed: int,
    transit_index: int,
    stream: Literal["release", "handoff"],
    attempt: int = 0,
) -> int:
    """Derive a name-independent seed for one planner stream."""

    for name, value in (
        ("transit_seed", transit_seed),
        ("transit_index", transit_index),
        ("attempt", attempt),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if stream not in {"release", "handoff"}:
        raise ValueError("unknown transit planner stream")
    payload = (
        f"{_TRANSIT_SEED_DOMAIN_V1}:{int(transit_seed)}:{int(transit_index)}:"
        f"{stream}:{int(attempt)}"
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return 1 + int.from_bytes(digest[:4], "little") % (2**31 - 2)


def _reanchor_release_controller_state(
    controller_state: np.ndarray | None,
    adapter: FR3OSCAdapter,
    measured_action: np.ndarray,
) -> np.ndarray | None:
    """Reset the OSC equilibrium pose at release entry without changing grip."""

    if controller_state is None:
        return None
    state = np.asarray(controller_state, dtype=float).copy()
    if state.shape != (len(FR3_CONTROLLER_STATE_NAMES),):
        raise ValueError("FR3 release controller state has the wrong shape")
    indices = {
        name: index for index, name in enumerate(FR3_CONTROLLER_STATE_NAMES)
    }
    position, orientation = adapter.target_pose_world(measured_action)
    for name, value in zip(
        ("filtered_target_x", "filtered_target_y", "filtered_target_z"),
        position,
    ):
        state[indices[name]] = float(value)
    for name, value in zip(
        (
            "filtered_target_qw",
            "filtered_target_qx",
            "filtered_target_qy",
            "filtered_target_qz",
        ),
        orientation,
    ):
        state[indices[name]] = float(value)
    for name, value in zip(
        (
            "latched_x",
            "latched_y",
            "latched_z",
            "latched_roll",
            "latched_pitch",
            "latched_yaw",
            "latched_gripper",
        ),
        measured_action,
    ):
        state[indices[name]] = float(value)
    state[indices["command_phase"]] = 0.0
    state[indices["gripper_command"]] = float(measured_action[6])
    return state


def _unit_quaternion(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("orientation must be a finite quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion cannot be zero")
    return quaternion / norm


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(output, left, right)
    return _unit_quaternion(output)


@dataclass(frozen=True)
class RoleFrame:
    """One learned semantic feature resolved from the current model state."""

    position: np.ndarray
    orientation: np.ndarray
    half_extents: np.ndarray
    geom_ids: tuple[int, ...] = ()
    body_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        extents = np.asarray(self.half_extents, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("role-frame position must be a finite three-vector")
        if extents.shape != (3,) or not np.all(np.isfinite(extents)):
            raise ValueError("role-frame half_extents must be a finite three-vector")
        if np.any(extents < 0.0):
            raise ValueError("role-frame half_extents cannot be negative")
        if any(int(item) < 0 for item in (*self.geom_ids, *self.body_ids)):
            raise ValueError("role-frame MuJoCo ids cannot be negative")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "orientation", _unit_quaternion(self.orientation))
        object.__setattr__(self, "half_extents", extents.copy())
        object.__setattr__(self, "geom_ids", tuple(int(item) for item in self.geom_ids))
        object.__setattr__(self, "body_ids", tuple(int(item) for item in self.body_ids))

    def axis(self, name: AxisName) -> np.ndarray:
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, self.orientation)
        index = {"x": 0, "y": 1, "z": 2}[name]
        return rotation.reshape(3, 3)[:, index].copy()

    def support_radius(self, direction: np.ndarray) -> float:
        axes = np.column_stack((self.axis("x"), self.axis("y"), self.axis("z")))
        return float(np.abs(direction @ axes) @ self.half_extents)


@dataclass(frozen=True)
class AxisRef:
    """A signed local axis of a semantic role."""

    role: str
    axis: AxisName
    sign: Literal[-1, 1] = 1

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("axis role cannot be empty")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        if self.sign not in {-1, 1}:
            raise ValueError("axis sign must be -1 or 1")


@dataclass(frozen=True)
class ScaleExpr:
    """A frozen coefficient times a live semantic object dimension."""

    role: str
    axis: AxisName
    coefficient: float

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("scale-expression role cannot be empty")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("scale-expression axis must be x, y, or z")
        if not np.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError(
                "scale-expression coefficient must be finite and nonnegative"
            )

    def resolve(self, frames: Mapping[str, RoleFrame]) -> float:
        frame = _frame(frames, self.role)
        index = {"x": 0, "y": 1, "z": 2}[self.axis]
        value = self.coefficient * 2.0 * frame.half_extents[index]
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("resolved scale expression must be finite and nonnegative")
        return float(value)


@dataclass(frozen=True)
class Withdraw:
    distance: ScaleExpr
    direction: AxisRef
    duration: float


@dataclass(frozen=True)
class RaiseToClearance:
    reference_role: str
    clearance: ScaleExpr
    direction: AxisRef
    duration: float


@dataclass(frozen=True)
class Reorient:
    """Set one task attitude and declare any discrete equivalent offsets."""

    target_role: str
    quaternion_offset: tuple[float, float, float, float]
    duration: float
    equivalent_quaternion_offsets: tuple[
        tuple[float, float, float, float], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.target_role:
            raise ValueError("reorientation target role cannot be empty")
        primary = tuple(
            float(item) for item in _unit_quaternion(self.quaternion_offset)
        )
        equivalents = tuple(
            tuple(float(item) for item in _unit_quaternion(offset))
            for offset in self.equivalent_quaternion_offsets
        )
        object.__setattr__(self, "quaternion_offset", primary)
        object.__setattr__(self, "equivalent_quaternion_offsets", equivalents)


@dataclass(frozen=True)
class Approach:
    target_role: str
    standoff: ScaleExpr
    direction: AxisRef
    duration: float


@dataclass(frozen=True)
class ConfigureGripper:
    """Set a physical aperture in free space before spatial approach."""

    aperture_fraction: float
    duration: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.aperture_fraction)
            or not 0.0 <= self.aperture_fraction <= 1.0
        ):
            raise ValueError("aperture_fraction must be finite and within [0, 1]")


@dataclass(frozen=True)
class Settle:
    """Request a precontact convergence-observation window.

    ``duration`` is the minimum hold, ``stable_duration`` is the consecutive
    convergence window, and ``maximum_duration`` caps the direct rollout. The
    higher-level hardware-parity transit runtime may lengthen that cap. A
    finite rollout that reaches the cap without convergence is retained as an
    advisory rather than rejected.
    """

    duration: float
    maximum_duration: float | None = None
    stable_duration: float = 0.10

    def __post_init__(self) -> None:
        maximum = (
            float(self.duration) + 1.0
            if self.maximum_duration is None
            else float(self.maximum_duration)
        )
        if not np.isfinite(maximum) or maximum < float(self.duration):
            raise ValueError("settle maximum_duration must be finite and >= duration")
        if not np.isfinite(self.stable_duration) or self.stable_duration <= 0.0:
            raise ValueError("settle stable_duration must be positive and finite")
        if maximum < self.stable_duration:
            raise ValueError(
                "settle maximum_duration must cover the stable-duration window"
            )
        object.__setattr__(self, "maximum_duration", maximum)


TransitStage: TypeAlias = (
    Withdraw | RaiseToClearance | Reorient | ConfigureGripper | Approach | Settle
)


@dataclass(frozen=True)
class InheritedReleaseWindow:
    """Authorize a bounded release of contact inherited from the prior phase.

    ``maximum_duration`` is the learned requested contact-release window. The
    hardware-parity runtime may lengthen the effective timeout to cover its
    fixed controller profile and reports that effective value in metrics.
    """

    maximum_duration: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_duration) or self.maximum_duration <= 0.0:
            raise ValueError("inherited release duration must be positive and finite")


@dataclass(frozen=True)
class TransitSpec:
    """Frozen learned transit intent; no waypoints or binding callbacks."""

    stages: tuple[TransitStage, ...]
    tool_role: str
    target_role: str
    protected_roles: tuple[str, ...] = ()
    protected_drift_limit: ScaleExpr | None = None
    inherited_release_window: InheritedReleaseWindow | None = None
    label: str = "precontact_transit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "protected_roles", tuple(self.protected_roles))
        if not self.stages:
            raise ValueError("transit spec must contain at least one stage")
        if not self.tool_role or not self.target_role or not self.label:
            raise ValueError(
                "transit tool role, target role, and label cannot be empty"
            )
        if self.tool_role == self.target_role:
            raise ValueError("transit tool and target roles must differ")
        if (
            self.tool_role in self.protected_roles
            or self.target_role in self.protected_roles
        ):
            raise ValueError("protected roles must exclude tool and target roles")
        settle_indices = [
            index
            for index, stage in enumerate(self.stages)
            if isinstance(stage, Settle)
        ]
        if settle_indices != [len(self.stages) - 1]:
            raise ValueError("every transit must end with exactly one Settle stage")
        moving = [stage for stage in self.stages if not isinstance(stage, Settle)]
        if not moving:
            raise ValueError("transit spec requires at least one moving stage")
        for stage in self.stages:
            if not np.isfinite(stage.duration) or stage.duration <= 0.0:
                raise ValueError("transit stage durations must be positive and finite")
        approach_indices = [
            index
            for index, stage in enumerate(self.stages)
            if isinstance(stage, Approach)
        ]
        configuration_indices = [
            index
            for index, stage in enumerate(self.stages)
            if isinstance(stage, ConfigureGripper)
        ]
        if len(configuration_indices) > 1:
            raise ValueError("a transit may contain at most one ConfigureGripper")
        if (
            approach_indices
            and configuration_indices
            and max(configuration_indices) >= min(approach_indices)
        ):
            raise ValueError(
                "ConfigureGripper must complete before the first Approach"
            )
        if self.protected_roles and self.protected_drift_limit is None:
            raise ValueError(
                "protected roles require a dimension-relative drift limit"
            )


@dataclass(frozen=True)
class TransitAudit:
    valid: bool
    forbidden_contact_steps: int
    inherited_contact_steps: int
    maximum_protected_drift: float
    aperture_motion_during_target_contact_steps: int = 0
    maximum_aperture_motion_during_target_contact: float = 0.0
    commanded_aperture_motion_during_target_contact_steps: int = 0
    maximum_commanded_aperture_motion_during_target_contact: float = 0.0


@dataclass(frozen=True)
class _InheritedGeometryPair:
    """One exact robot-environment relation inherited at phase entry."""

    robot_geom_id: int
    environment_geom_id: int
    initial_distance: float
    entry_state: Literal["contact", "near"]
    escape_directions: tuple[tuple[float, float, float], ...] = ()

    @property
    def key(self) -> tuple[int, int]:
        return (self.robot_geom_id, self.environment_geom_id)

    @property
    def contact_equivalent(self) -> bool:
        return self.entry_state == "contact"


@dataclass(frozen=True)
class _InheritedReleaseAudit:
    """Pair-level release certificate for a trajectory beginning in contact."""

    valid: bool
    contacts_released: bool
    separation_step: int | None
    separation_time: float | None
    last_contact_step: int | None
    terminal_separation_duration: float
    inherited_contact_steps: int
    new_contact_steps: int
    recontact_steps: int
    penetration_worsening_steps: int
    maximum_penetration_regression: float
    minimum_inherited_distance: float
    minimum_inherited_pair: tuple[int, int] | None
    pair_minimum_distances: tuple[tuple[tuple[int, int], float], ...]
    maximum_protected_drift: float
    first_violation_kind: str | None
    first_violation_step: int | None
    first_violation_pair: tuple[int, int] | None


def _classify_release_pair(
    *,
    contacting: bool,
    distance: float,
    required_clearance: float,
) -> Literal["contact", "near", "clear"]:
    """Classify phase-entry geometry with one planner/auditor tolerance."""

    if contacting or distance <= _RELEASE_CONTACT_DISTANCE_EPSILON:
        return "contact"
    if distance < required_clearance:
        return "near"
    return "clear"


def _frame(frames: Mapping[str, RoleFrame], role: str) -> RoleFrame:
    try:
        frame = frames[role]
    except KeyError as exc:
        raise ValueError(f"transit references unresolved role {role!r}") from exc
    if not isinstance(frame, RoleFrame):
        raise TypeError(f"transit role {role!r} must resolve to RoleFrame")
    return frame


def _axis(frames: Mapping[str, RoleFrame], reference: AxisRef) -> np.ndarray:
    return float(reference.sign) * _frame(frames, reference.role).axis(reference.axis)


def realize_transit_waypoints(
    adapter: FR3OSCAdapter,
    measured_start_action: np.ndarray,
    frames: Mapping[str, RoleFrame],
    spec: TransitSpec,
) -> tuple[np.ndarray, tuple[float, ...], Settle]:
    """Realize typed intent into OSC actions using only learned stage code."""

    action = np.asarray(measured_start_action, dtype=float).copy()
    if action.shape != (adapter.action_dim,):
        raise ValueError("measured start action has the wrong dimension")
    position, orientation = adapter.target_pose_world(action)
    gripper = float(action[6])
    waypoints = [action.copy()]
    durations: list[float] = []
    settle: Settle | None = None
    for stage in spec.stages:
        if isinstance(stage, Settle):
            settle = stage
            continue
        if isinstance(stage, Withdraw):
            position = (
                position
                + _axis(frames, stage.direction)
                * stage.distance.resolve(frames)
            )
        elif isinstance(stage, RaiseToClearance):
            direction = _axis(frames, stage.direction)
            reference = _frame(frames, stage.reference_role)
            desired = (
                float(direction @ reference.position)
                + reference.support_radius(direction)
                + max(
                    stage.clearance.resolve(frames),
                    DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.clearance,
                )
            )
            rise = max(0.0, desired - float(direction @ position))
            position = position + direction * rise
        elif isinstance(stage, Reorient):
            reference = _frame(frames, stage.target_role)
            orientation = _multiply_quaternions(
                reference.orientation,
                _unit_quaternion(stage.quaternion_offset),
            )
        elif isinstance(stage, ConfigureGripper):
            gripper = adapter.gripper_aperture(stage.aperture_fraction)
        elif isinstance(stage, Approach):
            direction = _axis(frames, stage.direction)
            target = _frame(frames, stage.target_role)
            standoff = stage.standoff.resolve(frames)
            position = (
                target.position
                - direction
                * (
                    target.support_radius(direction)
                    + standoff
                )
            )
        else:  # pragma: no cover - guards future stage additions
            raise TypeError(f"unsupported transit stage: {type(stage).__name__}")
        action = adapter.action_from_world_pose(position, orientation, gripper=gripper)
        waypoints.append(action)
        minimum_duration = minimum_pose_segment_duration(
            adapter,
            waypoints[-2],
            waypoints[-1],
        )
        durations.append(max(float(stage.duration), minimum_duration))
    assert settle is not None
    return np.asarray(waypoints), tuple(durations), settle


def _equivalent_handoff_orientations(
    frames: Mapping[str, RoleFrame],
    spec: TransitSpec,
) -> tuple[np.ndarray, ...]:
    reorientation = next(
        (
            stage
            for stage in reversed(spec.stages)
            if isinstance(stage, Reorient)
        ),
        None,
    )
    if reorientation is None:
        return ()
    reference = _frame(frames, reorientation.target_role)
    return tuple(
        _multiply_quaternions(
            reference.orientation,
            np.asarray(offset, dtype=float),
        )
        for offset in reorientation.equivalent_quaternion_offsets
    )


def _contact_between(data: mujoco.MjData, left: set[int], right: set[int]) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        if (first in left and second in right) or (
            second in left and first in right
        ):
            return True
    return False


def _minimum_pair_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left: set[int],
    right: set[int],
    cutoff: float,
) -> float:
    minimum = float(cutoff)
    for first in left:
        for second in right:
            if first == second:
                continue
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    first,
                    second,
                    float(cutoff),
                    None,
                )
            )
            minimum = min(minimum, distance)
    return minimum


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"geom_{geom_id}"
    )


def _robot_environment_pair(
    first: int,
    second: int,
    robot_ids: set[int],
) -> tuple[int, int] | None:
    if first in robot_ids and second not in robot_ids:
        return (first, second)
    if second in robot_ids and first not in robot_ids:
        return (second, first)
    return None


def _collision_pair_enabled(
    model: mujoco.MjModel,
    first: int,
    second: int,
) -> bool:
    return bool(
        int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
        or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])
    )


def _snapshot_release_entry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    safety: FR3SafetySpec,
    execution_clearance: float,
    planning_clearance: float,
) -> tuple[tuple[_InheritedGeometryPair, ...], tuple[FR3ReleasePair, ...]]:
    """Capture the complete geometry contract for one release entry."""

    robot_ids = set(safety.robot_geom_ids)
    moving_ids = set(safety.moving_robot_geom_ids)
    contact_pairs: set[tuple[int, int]] = set()
    escape_directions: dict[tuple[int, int], list[np.ndarray]] = {}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        pair = _robot_environment_pair(first, second, robot_ids)
        if pair is None:
            continue
        robot_geom, environment_geom = pair
        if robot_geom not in moving_ids:
            continue
        if float(contact.dist) <= _RELEASE_CONTACT_DISTANCE_EPSILON:
            contact_pairs.add(pair)
        normal = np.asarray(contact.frame[:3], dtype=float)
        # MuJoCo's contact normal points from geom1 to geom2. Orient it away
        # from the environment for the robot geometry.
        away = -normal if first == robot_geom else normal
        escape_directions.setdefault(pair, []).append(away.copy())

    inherited: list[_InheritedGeometryPair] = []
    tracking: list[FR3ReleasePair] = []
    distance_cutoff = (
        planning_clearance + _RELEASE_CLEARANCE_NUMERICAL_TOLERANCE
    )
    for robot_geom in sorted(moving_ids):
        for environment_geom in range(model.ngeom):
            pair = (robot_geom, environment_geom)
            if (
                environment_geom in robot_ids
                or not (
                    int(model.geom_contype[environment_geom])
                    or int(model.geom_conaffinity[environment_geom])
                )
                or not _collision_pair_enabled(
                    model,
                    robot_geom,
                    environment_geom,
                )
            ):
                continue
            from_to = np.empty(6, dtype=float)
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    robot_geom,
                    environment_geom,
                    distance_cutoff,
                    from_to,
                )
            )
            if distance >= planning_clearance:
                continue
            contacting = pair in contact_pairs
            entry_state = _classify_release_pair(
                contacting=contacting,
                distance=distance,
                required_clearance=execution_clearance,
            )
            if entry_state == "clear":
                tracking.append(
                    FR3ReleasePair(
                        robot_geom_id=robot_geom,
                        environment_geom_id=environment_geom,
                        entry_state="tracking",
                        initial_distance=distance,
                        minimum_distance=execution_clearance,
                    )
                )
                continue
            directions = escape_directions.setdefault(pair, [])
            closest_point_delta = from_to[:3] - from_to[3:]
            if np.all(np.isfinite(closest_point_delta)) and (
                np.linalg.norm(closest_point_delta) > 1e-9
            ):
                directions.append(closest_point_delta.copy())
            inherited.append(
                _InheritedGeometryPair(
                    robot_geom_id=robot_geom,
                    environment_geom_id=environment_geom,
                    initial_distance=distance,
                    entry_state=entry_state,
                    escape_directions=tuple(
                        tuple(float(value) for value in direction)
                        for direction in directions
                    ),
                )
            )
    return tuple(inherited), tuple(tracking)


def _audit_inherited_release(
    context: BoundExecutionContext,
    frames: Mapping[str, RoleFrame],
    spec: TransitSpec,
    states: np.ndarray,
    safety: FR3SafetySpec,
    inherited_pairs: tuple[_InheritedGeometryPair, ...],
    *,
    contact_timeout: float,
) -> _InheritedReleaseAudit:
    """Certify separation, non-worsening, and absence of new contact."""

    model = context.model
    state_values = np.asarray(states, dtype=float)
    robot_ids = set(safety.robot_geom_ids)
    contacting_keys = {
        pair.key for pair in inherited_pairs if pair.contact_equivalent
    }
    protected_body_ids: set[int] = set()
    for role in spec.protected_roles:
        frame = _frame(frames, role)
        protected_body_ids.update(frame.body_ids)
        protected_body_ids.update(
            int(model.geom_bodyid[geom_id]) for geom_id in frame.geom_ids
        )

    data = mujoco.MjData(model)

    def set_state(state: np.ndarray) -> None:
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq :]
        mujoco.mj_fwdKinematics(model, data)
        mujoco.mj_collision(model, data)

    set_state(context.initial_state)
    protected_starts = {
        body_id: np.asarray(data.xpos[body_id], dtype=float).copy()
        for body_id in protected_body_ids
    }
    stable_steps_required = max(
        1,
        int(
            np.ceil(
                _RELEASE_STABLE_SEPARATION_DURATION / float(model.opt.timestep)
            )
        ),
    )
    timeout_step = max(1, int(np.floor(contact_timeout / model.opt.timestep)))
    stable_steps = 0
    separation_step: int | None = 0 if not contacting_keys else None
    last_contact_step: int | None = None
    inherited_contact_steps = 0
    new_contact_steps = 0
    recontact_steps = 0
    penetration_worsening_steps = 0
    maximum_penetration_regression = 0.0
    minimum_distance = min(
        (pair.initial_distance for pair in inherited_pairs),
        default=float("inf"),
    )
    minimum_pair = min(
        inherited_pairs,
        key=lambda pair: pair.initial_distance,
        default=None,
    )
    pair_minimum_distances = {
        pair.key: pair.initial_distance for pair in inherited_pairs
    }
    maximum_drift = 0.0
    first_violation_kind: str | None = None
    first_violation_step: int | None = None
    first_violation_pair: tuple[int, int] | None = None

    def record_violation(
        kind: str,
        step: int,
        pair: tuple[int, int] | None,
    ) -> None:
        nonlocal first_violation_kind, first_violation_step
        nonlocal first_violation_pair
        if first_violation_kind is None:
            first_violation_kind = kind
            first_violation_step = step
            first_violation_pair = pair

    for step, state in enumerate(state_values, start=1):
        set_state(state)
        active_pairs: set[tuple[int, int]] = set()
        penetrating_pairs: set[tuple[int, int]] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > _RELEASE_CONTACT_DISTANCE_EPSILON:
                continue
            pair = _robot_environment_pair(
                int(contact.geom1),
                int(contact.geom2),
                robot_ids,
            )
            if pair is not None:
                active_pairs.add(pair)
                if float(contact.dist) < -_RELEASE_PENETRATION_HYSTERESIS:
                    penetrating_pairs.add(pair)

        inherited_contacts = active_pairs & contacting_keys
        # A release begins next to contact and the impedance controller can
        # chatter by a fraction of a millimetre.  Preserve the strict planner
        # goal and terminal clearance, but require a genuinely new penetration
        # beyond the bounded hysteresis before rejecting the whole release.
        new_contacts = penetrating_pairs - contacting_keys
        if inherited_contacts:
            inherited_contact_steps += 1
        if new_contacts:
            new_contact_steps += 1
            record_violation("new_contact", step, sorted(new_contacts)[0])

        all_separated = True
        worsening_pairs: list[tuple[int, int]] = []
        for pair in inherited_pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    pair.robot_geom_id,
                    pair.environment_geom_id,
                    DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.execution_clearance,
                    None,
                )
            )
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_pair = pair
            pair_minimum_distances[pair.key] = min(
                pair_minimum_distances[pair.key],
                distance,
            )
            if pair.contact_equivalent:
                pair_separated = bool(
                    distance >= _RELEASE_STABLE_SEPARATION_GAP
                    and pair.key not in active_pairs
                )
                all_separated = all_separated and pair_separated
                minimum_allowed = (
                    pair.initial_distance - _RELEASE_PENETRATION_HYSTERESIS
                )
                maximum_penetration_regression = max(
                    maximum_penetration_regression,
                    pair.initial_distance - distance,
                )
            else:
                minimum_allowed = -_RELEASE_PENETRATION_HYSTERESIS
            if distance < minimum_allowed:
                worsening_pairs.append(pair.key)
        if worsening_pairs:
            penetration_worsening_steps += 1
            record_violation(
                "clearance_regression",
                step,
                sorted(worsening_pairs)[0],
            )

        if separation_step is not None and not all_separated:
            recontact_steps += 1

        if contacting_keys:
            if all_separated:
                stable_steps += 1
            else:
                stable_steps = 0
                last_contact_step = step
            if separation_step is None and stable_steps >= stable_steps_required:
                separation_step = step - stable_steps_required + 1

        for body_id, start in protected_starts.items():
            maximum_drift = max(
                maximum_drift,
                float(np.linalg.norm(np.asarray(data.xpos[body_id]) - start)),
            )

    terminal_separation_duration = float(stable_steps * model.opt.timestep)
    contacts_released = bool(
        not contacting_keys
        or (
            separation_step is not None
            and separation_step <= timeout_step
            and stable_steps >= stable_steps_required
        )
    )
    drift_limit = (
        spec.protected_drift_limit.resolve(frames)
        if spec.protected_drift_limit is not None
        else 0.0
    )
    drift_valid = not protected_starts or maximum_drift <= drift_limit
    if not drift_valid:
        record_violation("protected_drift", len(state_values), None)
    if not contacts_released:
        record_violation("separation_timeout", len(state_values), None)
    return _InheritedReleaseAudit(
        valid=bool(
            contacts_released
            and new_contact_steps == 0
            and penetration_worsening_steps == 0
            and drift_valid
        ),
        contacts_released=contacts_released,
        separation_step=separation_step,
        separation_time=(
            None
            if separation_step is None
            else float(separation_step * model.opt.timestep)
        ),
        last_contact_step=last_contact_step,
        terminal_separation_duration=terminal_separation_duration,
        inherited_contact_steps=inherited_contact_steps,
        new_contact_steps=new_contact_steps,
        recontact_steps=recontact_steps,
        penetration_worsening_steps=penetration_worsening_steps,
        maximum_penetration_regression=maximum_penetration_regression,
        minimum_inherited_distance=minimum_distance,
        minimum_inherited_pair=(
            None if minimum_pair is None else minimum_pair.key
        ),
        pair_minimum_distances=tuple(pair_minimum_distances.items()),
        maximum_protected_drift=maximum_drift,
        first_violation_kind=first_violation_kind,
        first_violation_step=first_violation_step,
        first_violation_pair=first_violation_pair,
    )


def _pair_escape_direction(
    model: mujoco.MjModel,
    states: np.ndarray,
    safety: FR3SafetySpec,
    pair: tuple[int, int] | None,
    step: int | None,
) -> np.ndarray | None:
    """Return the measured robot-away-from-environment direction."""

    if pair is None:
        return None
    robot_geom, environment_geom = pair
    if robot_geom not in set(safety.robot_geom_ids):
        robot_geom, environment_geom = environment_geom, robot_geom
    if robot_geom not in set(safety.robot_geom_ids):
        return None
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        return None
    sample = 1 if step is None else step
    state = values[min(max(sample - 1, 0), values.shape[0] - 1)]
    data = mujoco.MjData(model)
    data.qpos[:] = state[: model.nq]
    data.qvel[:] = state[model.nq :]
    mujoco.mj_fwdKinematics(model, data)
    from_to = np.empty(6, dtype=float)
    mujoco.mj_geomDistance(
        model,
        data,
        robot_geom,
        environment_geom,
        DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.execution_clearance,
        from_to,
    )
    direction = from_to[:3] - from_to[3:]
    norm = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or norm <= 1e-9:
        return None
    return direction / norm


def _release_feedback_direction(
    model: mujoco.MjModel,
    states: np.ndarray,
    safety: FR3SafetySpec,
    audit: _InheritedReleaseAudit,
) -> np.ndarray | None:
    return _pair_escape_direction(
        model,
        states,
        safety,
        (
            audit.first_violation_pair
            if audit.first_violation_pair is not None
            else audit.minimum_inherited_pair
        ),
        (
            audit.first_violation_step
            if audit.first_violation_step is not None
            else np.asarray(states).shape[0]
        ),
    )


def _whole_arm_feedback_direction(
    model: mujoco.MjModel,
    states: np.ndarray,
    safety: FR3SafetySpec,
    audit: object,
) -> np.ndarray | None:
    pair_names = getattr(audit, "minimum_environment_clearance_pair", None)
    if pair_names is None:
        return None
    pair = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in pair_names
    )
    if len(pair) != 2 or min(pair) < 0:
        return None
    return _pair_escape_direction(
        model,
        states,
        safety,
        pair,
        getattr(audit, "minimum_environment_clearance_step", None),
    )


def _terminal_release_feedback_direction(
    model: mujoco.MjModel,
    states: np.ndarray,
    safety: FR3SafetySpec,
    release_pairs: tuple[FR3ReleasePair, ...],
) -> np.ndarray | None:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or not release_pairs:
        return None
    data = mujoco.MjData(model)
    data.qpos[:] = values[-1, : model.nq]
    data.qvel[:] = values[-1, model.nq :]
    mujoco.mj_fwdKinematics(model, data)
    pair = min(
        release_pairs,
        key=lambda item: float(
            mujoco.mj_geomDistance(
                model,
                data,
                item.robot_geom_id,
                item.environment_geom_id,
                DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.execution_clearance,
                None,
            )
        ),
    )
    return _pair_escape_direction(
        model,
        values,
        safety,
        (pair.robot_geom_id, pair.environment_geom_id),
        values.shape[0],
    )


def _release_is_stationary(
    model: mujoco.MjModel,
    adapter: FR3OSCAdapter,
    candidate: FR3TransitResult,
) -> bool:
    stable_steps = max(1, int(np.ceil(0.10 / float(model.opt.timestep))))
    observations = candidate.controller_observations[-stable_steps:]
    states = candidate.result.best_states[-stable_steps:]
    return bool(
        observations.shape[0] >= stable_steps
        and np.all(np.linalg.norm(observations[:, 7:10], axis=1) <= 0.010)
        and np.all(np.linalg.norm(observations[:, 10:13], axis=1) <= 0.050)
        and np.all(
            np.abs(adapter.gripper_aperture_speeds(model, states)) <= 0.002
        )
    )


def _stabilize_release_candidate(
    model: mujoco.MjModel,
    adapter: FR3OSCAdapter,
    candidate: FR3TransitResult,
    *,
    label: str,
) -> FR3TransitResult:
    terminal_data = mujoco.MjData(model)
    terminal_data.qpos[:] = candidate.terminal_state[: model.nq]
    terminal_data.qvel[:] = candidate.terminal_state[model.nq :]
    mujoco.mj_forward(model, terminal_data)
    hold_action = adapter.action_from_data(model, terminal_data)
    hold_action[6] = candidate.terminal_action[6]
    controller_state = _reanchor_release_controller_state(
        candidate.controller_state,
        adapter,
        hold_action,
    )
    stabilization = rollout_fr3_transit(
        model,
        candidate.terminal_state,
        adapter,
        np.stack((hold_action, hold_action)),
        (float(model.opt.timestep),),
        initial_controller_state=controller_state,
        initial_command_action=hold_action,
        gripper_operation=GripperOperation.MOVE,
        settle_duration=0.10,
        settle_maximum_duration=_RELEASE_STABILIZATION_MAXIMUM_DURATION,
        settle_stable_duration=0.10,
        terminal_position_tolerance=0.020,
        terminal_orientation_tolerance=0.080,
        terminal_linear_speed_tolerance=0.010,
        terminal_angular_speed_tolerance=0.050,
        label=f"{label}_stabilization",
    )
    combined = _concatenate_transit_phases(
        [candidate, stabilization],
        label,
        float(model.opt.timestep),
    )
    combined.result.metrics.update(
        {
            "transit_valid": bool(
                candidate.result.metrics.get("transit_valid", False)
                and stabilization.result.metrics.get("transit_valid", False)
            ),
            "transit_convergence_valid": bool(
                stabilization.result.metrics.get(
                    "transit_convergence_valid",
                    False,
                )
            ),
            "release_stabilization_used": True,
            "release_stabilization_duration": stabilization.result.metrics[
                "duration"
            ],
        }
    )
    return combined


def _prioritize_release_plans(
    plans: list[FR3TransitPlan],
    feedback_direction: np.ndarray | None,
) -> None:
    """Move candidates aligned with the observed escape normal to the front."""

    if feedback_direction is None:
        return

    def key(plan: FR3TransitPlan) -> tuple[float, int]:
        direction = np.asarray(
            plan.metrics.get("release_translation_direction", np.zeros(3)),
            dtype=float,
        )
        alignment = (
            float(direction @ feedback_direction)
            if direction.shape == (3,) and np.all(np.isfinite(direction))
            else -1.0
        )
        return (-alignment, int(plan.metrics.get("release_shortlist_rank", 0)))

    plans.sort(key=key)


def audit_transit(
    context: BoundExecutionContext,
    frames: Mapping[str, RoleFrame],
    spec: TransitSpec,
    states: np.ndarray,
    *,
    adapter: FR3OSCAdapter | None = None,
    controls: np.ndarray | None = None,
) -> TransitAudit:
    """Audit the single retained transit, including inherited target contact."""

    tool_ids = set(_frame(frames, spec.tool_role).geom_ids)
    if not tool_ids:
        raise ValueError("transit tool role must resolve at least one geom id")
    target_ids = set(_frame(frames, spec.target_role).geom_ids)
    if not target_ids:
        raise ValueError("transit target role must resolve at least one geom id")
    protected_ids = {
        geom_id
        for role in spec.protected_roles
        for geom_id in _frame(frames, role).geom_ids
    }
    data = mujoco.MjData(context.model)

    def set_state(state: np.ndarray) -> None:
        data.qpos[:] = state[: context.model.nq]
        data.qvel[:] = state[context.model.nq :]
        mujoco.mj_fwdKinematics(context.model, data)
        mujoco.mj_collision(context.model, data)

    set_state(context.initial_state)
    target_contact_inherited = bool(
        target_ids and _contact_between(data, tool_ids, target_ids)
    )
    protected_body_ids: set[int] = set()
    for role in spec.protected_roles:
        frame = _frame(frames, role)
        protected_body_ids.update(frame.body_ids)
        protected_body_ids.update(
            int(context.model.geom_bodyid[geom_id]) for geom_id in frame.geom_ids
        )
    protected_starts = {
        body_id: np.asarray(data.xpos[body_id], dtype=float).copy()
        for body_id in protected_body_ids
    }
    release_closed = False
    no_contact_steps = 0
    inherited_steps = 0
    forbidden_steps = 0
    maximum_drift = 0.0
    aperture_motion_steps = 0
    maximum_aperture_motion = 0.0
    commanded_aperture_motion_steps = 0
    maximum_commanded_aperture_motion = 0.0
    if adapter is not None:
        previous_aperture = float(
            adapter.gripper_apertures(
                context.model,
                context.initial_state[None],
            )[0]
        )
        aperture_motion_tolerance = max(
            1e-12,
            1e-6 * (adapter.gripper_range[1] - adapter.gripper_range[0]),
        )
        command_values = (
            None if controls is None else np.asarray(controls, dtype=float)
        )
        if command_values is not None and command_values.shape != (
            len(states),
            adapter.action_dim,
        ):
            raise ValueError("transit audit controls must align with retained states")
        previous_command = (
            float(context.initial_action[6])
            if context.initial_action is not None
            else float(adapter.action_from_data(context.model, data)[6])
        )
    step_limit = (
        int(
            np.floor(
                spec.inherited_release_window.maximum_duration
                / context.model.opt.timestep
            )
        )
        if spec.inherited_release_window is not None
        else 0
    )
    stable_release_steps = max(
        1,
        int(np.ceil(0.050 / context.model.opt.timestep)),
    )
    for step, state in enumerate(np.asarray(states, dtype=float), start=1):
        set_state(state)
        if protected_ids and _contact_between(data, tool_ids, protected_ids):
            forbidden_steps += 1
        if target_ids:
            contact = _contact_between(data, tool_ids, target_ids)
            if adapter is not None:
                current_aperture = float(
                    adapter.gripper_apertures(context.model, state[None])[0]
                )
                aperture_motion = abs(current_aperture - previous_aperture)
                if contact and aperture_motion > aperture_motion_tolerance:
                    aperture_motion_steps += 1
                    maximum_aperture_motion = max(
                        maximum_aperture_motion,
                        aperture_motion,
                    )
                if command_values is not None:
                    command_motion = abs(
                        float(command_values[step - 1, 6]) - previous_command
                    )
                    if contact and command_motion > aperture_motion_tolerance:
                        commanded_aperture_motion_steps += 1
                        maximum_commanded_aperture_motion = max(
                            maximum_commanded_aperture_motion,
                            command_motion,
                        )
            allowed = bool(
                contact
                and target_contact_inherited
                and not release_closed
                and spec.inherited_release_window is not None
                and step <= step_limit
            )
            if contact and not allowed:
                forbidden_steps += 1
            if allowed:
                inherited_steps += 1
            if target_contact_inherited:
                if contact:
                    no_contact_steps = 0
                else:
                    no_contact_steps += 1
                    if no_contact_steps >= stable_release_steps:
                        release_closed = True
        if adapter is not None:
            previous_aperture = float(
                adapter.gripper_apertures(context.model, state[None])[0]
            )
            if command_values is not None:
                previous_command = float(command_values[step - 1, 6])
        for body_id, start in protected_starts.items():
            maximum_drift = max(
                maximum_drift,
                float(np.linalg.norm(np.asarray(data.xpos[body_id]) - start)),
            )
    drift_limit = (
        spec.protected_drift_limit.resolve(frames)
        if spec.protected_drift_limit is not None
        else 0.0
    )
    drift_valid = not protected_starts or maximum_drift <= drift_limit
    return TransitAudit(
        valid=(
            forbidden_steps == 0
            and drift_valid
            and commanded_aperture_motion_steps == 0
        ),
        forbidden_contact_steps=forbidden_steps,
        inherited_contact_steps=inherited_steps,
        maximum_protected_drift=maximum_drift,
        aperture_motion_during_target_contact_steps=aperture_motion_steps,
        maximum_aperture_motion_during_target_contact=maximum_aperture_motion,
        commanded_aperture_motion_during_target_contact_steps=(
            commanded_aperture_motion_steps
        ),
        maximum_commanded_aperture_motion_during_target_contact=(
            maximum_commanded_aperture_motion
        ),
    )


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = _unit_quaternion(first)
    right = _unit_quaternion(second)
    return 2.0 * float(np.arccos(np.clip(abs(float(left @ right)), 0.0, 1.0)))


def _rollout_free_gripper_configuration(
    model: mujoco.MjModel,
    adapter: FR3OSCAdapter,
    initial_state: np.ndarray,
    controller_state: np.ndarray | None,
    initial_action: np.ndarray,
    aperture: float,
    *,
    duration: float,
    label: str,
) -> FR3TransitResult:
    target = np.asarray(initial_action, dtype=float).copy()
    target[6] = float(aperture)
    return rollout_fr3_transit(
        model,
        initial_state,
        adapter,
        np.stack((initial_action, target)),
        (max(float(model.opt.timestep), float(duration)),),
        initial_controller_state=controller_state,
        initial_command_action=initial_action,
        gripper_operation=GripperOperation.MOVE,
        settle_duration=0.20,
        # A release can hand this phase a controller filter that is still
        # unloading its old contact equilibrium. Retain early on convergence,
        # but grant the same controller-scale settling budget as spatial
        # transit instead of rejecting a safe gripper operation after 0.8 s.
        settle_maximum_duration=4.00,
        settle_stable_duration=0.10,
        # This phase changes aperture rather than spatial intent.  Preserve
        # the inherited compliant equilibrium while still requiring it to be
        # stationary while reporting any collision finding as an advisory.
        terminal_position_tolerance=0.060,
        terminal_orientation_tolerance=0.200,
        label=label,
    )


def _whole_arm_metrics(audit: object) -> dict[str, object]:
    return {
        "whole_arm_audit_valid": audit.valid,
        "self_collision_steps": audit.self_collision_steps,
        "unapproved_robot_contact_steps": audit.unapproved_contact_steps,
        "minimum_environment_clearance": audit.minimum_environment_clearance,
        "minimum_enforced_environment_clearance": (
            audit.minimum_enforced_environment_clearance
        ),
        "minimum_environment_clearance_pair": (
            audit.minimum_environment_clearance_pair
        ),
        "minimum_environment_clearance_step": (
            audit.minimum_environment_clearance_step
        ),
        "minimum_declared_proximity_clearance": (
            audit.minimum_declared_proximity_clearance
        ),
        "minimum_declared_proximity_margin": (
            audit.minimum_declared_proximity_margin
        ),
        "declared_proximity_clearance_valid": (
            audit.declared_proximity_clearance_valid
        ),
        "dominant_unapproved_contact_pair": (
            audit.dominant_unapproved_contact_pair
        ),
        "permitted_fingertip_contact_steps": (
            audit.permitted_fingertip_contact_steps
        ),
        "maximum_closed_stop_penetration": audit.maximum_closed_stop_penetration,
        "maximum_finger_synchronization_error": (
            audit.maximum_finger_synchronization_error
        ),
        "geometry_numerical_tolerance": getattr(
            audit,
            "geometry_numerical_tolerance",
            0.0,
        ),
    }


def _audit_free_phase(
    model: mujoco.MjModel,
    phase: FR3TransitResult,
    safety: FR3SafetySpec,
) -> bool:
    """Record final-OSC validation findings without rejecting finite motion."""

    audit = audit_fr3_trajectory(model, phase.result.best_states, safety)
    metrics = phase.result.metrics
    metrics.update(_whole_arm_metrics(audit))
    if not metrics.get("transit_valid", False):
        raise FR3TransitViolation(
            f"{metrics.get('phase', 'FR3 transit phase')} produced a non-finite "
            "OSC rollout",
            result=phase.result,
        )
    findings: list[str] = []
    if not metrics.get("transit_convergence_valid", False):
        findings.append("transit_convergence_or_joint_margin")
    if not audit.valid:
        findings.append("whole_arm_clearance_or_contact")
    existing = metrics.get("audit_advisories", ()) or ()
    metrics["audit_advisories"] = list(
        dict.fromkeys((*existing, *findings))
    )
    metrics["audit_advisory_triggered"] = bool(
        metrics["audit_advisories"]
    )
    metrics["final_osc_rollout_advisory"] = {
        "triggered": bool(findings),
        "findings": findings,
        "convergence_valid": bool(
            metrics.get("transit_convergence_valid", False)
        ),
        "whole_arm_audit_valid": bool(audit.valid),
    }
    return not findings


def _concatenate_transit_phases(
    phases: list[FR3TransitResult],
    label: str,
    timestep: float,
) -> FR3TransitResult:
    if not phases:
        raise ValueError("FR3 transit requires at least one physical phase")
    controls = np.concatenate(
        [np.asarray(phase.result.trajectory.controls) for phase in phases], axis=0
    )
    positions = np.concatenate(
        [np.asarray(phase.result.trajectory.positions) for phase in phases], axis=0
    )
    states = np.concatenate(
        [np.asarray(phase.result.best_states) for phase in phases], axis=0
    )
    observations = np.concatenate(
        [np.asarray(phase.controller_observations) for phase in phases], axis=0
    )
    operations = np.concatenate(
        [
            np.asarray(phase.result.trajectory.gripper_operations, dtype=np.uint8)
            for phase in phases
        ]
    )
    boundary_deltas = [
        abs(
            float(
                phases[index].result.trajectory.controls[0, 6]
                - phases[index - 1].result.trajectory.controls[-1, 6]
            )
        )
        for index in range(1, len(phases))
    ]
    rollout_backends = tuple(
        phase.result.metrics.get("controller_rollout_backend")
        for phase in phases
    )
    controller_rollout_backend = (
        rollout_backends[0]
        if rollout_backends
        and rollout_backends[0] is not None
        and all(value == rollout_backends[0] for value in rollout_backends)
        else None
    )
    result = CompileResult(
        trajectory=ToolTrajectory(
            times=np.arange(controls.shape[0], dtype=float) * float(timestep),
            positions=positions,
            controls=controls,
            gripper_operations=operations,
        ),
        metrics={
            "phase": label,
            "transit_simulated": True,
            "duration": float(controls.shape[0] * timestep),
            "num_steps": int(controls.shape[0]),
            "phase_boundary_aperture_delta": max(boundary_deltas, default=0.0),
            "controller_rollout_backend": controller_rollout_backend,
            "transit_phases": [dict(phase.result.metrics) for phase in phases],
        },
        best_states=states,
        cost_history=np.empty((0,), dtype=float),
        controller_state=phases[-1].controller_state.copy(),
    )
    return FR3TransitResult(
        result=result,
        controller_observations=observations,
        controller_state=phases[-1].controller_state.copy(),
    )


def _retain_transit_failure(
    exc: Exception,
    accepted_phases: list[FR3TransitResult],
    rejected_phase: FR3TransitResult | None,
    *,
    label: str,
    timestep: float,
) -> None:
    """Attach accepted internal phases and the rejected phase to ``exc``."""

    phases = [*accepted_phases]
    if rejected_phase is not None:
        phases.append(rejected_phase)
    if not phases:
        return
    try:
        partial = _concatenate_transit_phases(phases, label, timestep)
    except Exception as retention_error:
        exc.add_note(
            "retaining the beginning-to-failure transit trajectory failed: "
            f"{retention_error}"
        )
        return
    partial.result.metrics.update(
        {
            "success": False,
            "partial_transit": True,
            "accepted_transit_phase_count": len(accepted_phases),
            "rejected_transit_phase_retained": rejected_phase is not None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
    )
    setattr(exc, "result", partial.result)


@timed_execution("transit")
def execute_transit(
    context: BoundExecutionContext,
    adapter: FR3OSCAdapter,
    frames: Mapping[str, RoleFrame],
    spec: TransitSpec,
    *,
    transit_seed: int,
    transit_index: int,
) -> FR3TransitResult:
    """Release, plan, physically audit, and hand off one FR3 transit."""

    model = context.model
    start_data = mujoco.MjData(model)
    start_data.qpos[:] = context.initial_state[: model.nq]
    start_data.qvel[:] = context.initial_state[model.nq :]
    mujoco.mj_fwdKinematics(model, start_data)
    mujoco.mj_collision(model, start_data)
    measured_start = adapter.action_from_data(model, start_data)
    inherited_start = (
        measured_start
        if context.initial_action is None
        else np.asarray(context.initial_action, dtype=float)
    )
    if inherited_start.shape != (adapter.action_dim,):
        raise ValueError("transit initial_action must match the FR3 action dimension")
    nominal_waypoints, nominal_durations, settle = realize_transit_waypoints(
        adapter,
        inherited_start,
        frames,
        spec,
    )
    tool_ids = set(_frame(frames, spec.tool_role).geom_ids)
    target_ids = set(_frame(frames, spec.target_role).geom_ids)
    if not tool_ids or not target_ids:
        raise ValueError("transit tool and target roles require collision geoms")
    required_clearance = (
        DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.execution_clearance
    )
    limits = replace(
        DEFAULT_FR3_SAFETY_LIMITS,
        minimum_environment_clearance=(
            DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.execution_clearance
        ),
    )
    free_safety = FR3SafetySpec.from_model(model, adapter, limits=limits)
    handoff_proximity = AllowedProximityPair(
        tuple(sorted(tool_ids)),
        tuple(sorted(target_ids)),
        minimum_clearance=(
            DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.handoff_clearance
        ),
    )
    handoff_safety = FR3SafetySpec.from_model(
        model,
        adapter,
        allowed_proximity_pairs=(handoff_proximity,),
        limits=limits,
    )
    release_planning_clearance = (
        required_clearance
        + DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.release_path_tracking_reserve
    )
    inherited_pairs, entry_tracking_pairs = _snapshot_release_entry(
        model,
        start_data,
        free_safety,
        required_clearance,
        release_planning_clearance,
    )
    if inherited_pairs and spec.inherited_release_window is None:
        raise FR3TransitViolation(
            "transition begins with a robot/environment contact or "
            "sub-clearance relation without an inherited release window"
        )
    inherited_contact_exemptions = tuple(
        AllowedContactPair(
            (pair.robot_geom_id,),
            (pair.environment_geom_id,),
        )
        for pair in inherited_pairs
        if pair.contact_equivalent
    )
    inherited_proximity_exemptions = tuple(
        AllowedProximityPair(
            (pair.robot_geom_id,),
            (pair.environment_geom_id,),
            minimum_clearance=0.0,
        )
        for pair in inherited_pairs
        if not pair.contact_equivalent
    )
    tracking_reserve = (
        DEFAULT_FR3_TRANSIT_PLANNING_CONFIG.release_path_tracking_reserve
    )
    tracking_proximity_exemptions = tuple(
        AllowedProximityPair(
            (pair.robot_geom_id,),
            (pair.environment_geom_id,),
            minimum_clearance=max(
                0.0,
                pair.initial_distance - tracking_reserve,
            ),
        )
        for pair in entry_tracking_pairs
    )
    release_safety: FR3SafetySpec | None = None

    phases: list[FR3TransitResult] = []
    phase_state = np.asarray(context.initial_state, dtype=float)
    phase_controller = context.controller_state
    phase_action = inherited_start.copy()
    release_metrics: dict[str, object] | None = None
    inherited_release_audit: _InheritedReleaseAudit | None = None
    configuration_metrics: dict[str, object] | None = None

    if inherited_pairs:
        assert spec.inherited_release_window is not None
        initial_target_clearance = _minimum_pair_distance(
            model,
            start_data,
            tool_ids,
            target_ids,
            required_clearance,
        )
        release_limits = replace(
            limits,
            minimum_environment_clearance=(
                required_clearance - _RELEASE_CLEARANCE_NUMERICAL_TOLERANCE
            ),
            geometry_numerical_tolerance=_RELEASE_PENETRATION_HYSTERESIS,
        )
        release_safety = FR3SafetySpec.from_model(
            model,
            adapter,
            allowed_contact_pairs=inherited_contact_exemptions,
            allowed_proximity_pairs=(
                inherited_proximity_exemptions
                + tracking_proximity_exemptions
            ),
            limits=release_limits,
        )
        # Contact-rich primitives intentionally finish with a penetrated OSC
        # equilibrium target. Geometry, however, starts at the measured robot
        # pose. First move that equilibrium target back to the measured pose
        # (with fixed aperture), then plan the spatial escape from the same
        # measured anchor. This removes residual contact force without an
        # instantaneous controller-command jump.
        release_start_action = measured_start.copy()
        release_start_action[6] = phase_action[6]
        release_controller_state = _reanchor_release_controller_state(
            phase_controller,
            adapter,
            release_start_action,
        )
        release_constraints = tuple(
            FR3ReleasePair(
                robot_geom_id=pair.robot_geom_id,
                environment_geom_id=pair.environment_geom_id,
                entry_state=pair.entry_state,
                initial_distance=pair.initial_distance,
                minimum_distance=(
                    pair.initial_distance - _RELEASE_PENETRATION_HYSTERESIS
                    if pair.contact_equivalent
                    else 0.0
                ),
            )
            for pair in inherited_pairs
        ) + entry_tracking_pairs
        escape_biases: list[np.ndarray] = []
        start_position, _ = adapter.target_pose_world(measured_start)
        for action in nominal_waypoints[1:]:
            position, _ = adapter.target_pose_world(action)
            if np.linalg.norm(position - start_position) > 1e-6:
                escape_biases.append(position - start_position)
                break
        escape_biases.extend(
            np.asarray(direction, dtype=float)
            for pair in inherited_pairs
            for direction in pair.escape_directions
        )
        try:
            release_seed = _derive_transit_planning_seed(
                transit_seed,
                transit_index,
                "release",
            )
            release_plans = plan_fr3_release(
                model,
                phase_state,
                adapter,
                release_start_action,
                release_constraints,
                escape_biases=escape_biases,
                seed=release_seed,
                search_mode="direct",
            )
            for release_plan in release_plans:
                release_plan.metrics.update(
                    {
                        "transit_seed": int(transit_seed),
                        "transit_index": int(transit_index),
                        "planning_seed_stream": "release",
                        "planning_seed_attempt": 0,
                    }
                )
        except FR3TransitPlanningFailure as exc:
            raise FR3TransitViolation(str(exc)) from exc
        minimum_contact_timeout = max(
            spec.inherited_release_window.maximum_duration,
            _MINIMUM_RELEASE_CONTACT_TIMEOUT,
        )
        release: FR3TransitResult | None = None
        last_candidate: FR3TransitResult | None = None
        candidate_rejections: list[dict[str, object]] = []
        pending_release_plans = list(release_plans)
        attempt = 0
        replan_count = 0
        feedback_direction: np.ndarray | None = None
        while attempt < _MAXIMUM_RELEASE_CONTROLLER_ROLLOUTS:
            if not pending_release_plans:
                if (
                    feedback_direction is None
                    or replan_count >= _MAXIMUM_RELEASE_REPLANS
                ):
                    break
                replan_count += 1
                retry_biases = [feedback_direction, *escape_biases]
                try:
                    pending_release_plans = list(
                        plan_fr3_release(
                            model,
                            phase_state,
                            adapter,
                            release_start_action,
                            release_constraints,
                            escape_biases=retry_biases,
                            seed=_derive_transit_planning_seed(
                                transit_seed,
                                transit_index,
                                "release",
                                replan_count,
                            ),
                            search_mode="expanded",
                        )
                    )
                except FR3TransitPlanningFailure:
                    break
                for plan in pending_release_plans:
                    plan.metrics.update(
                        {
                            "release_replan_round": replan_count,
                            "transit_seed": int(transit_seed),
                            "transit_index": int(transit_index),
                            "planning_seed_stream": "release",
                            "planning_seed_attempt": replan_count,
                        }
                    )
                if not pending_release_plans:
                    break
            attempt += 1
            candidate_plan = pending_release_plans.pop(0)
            candidate_durations = np.asarray(
                candidate_plan.segment_durations,
                dtype=float,
            )
            candidate_durations *= 1.0 + 0.5 * replan_count
            minimum_movement_duration = max(
                float(model.opt.timestep),
                spec.inherited_release_window.maximum_duration,
                _MINIMUM_RELEASE_MOVEMENT_DURATION,
            )
            if float(np.sum(candidate_durations)) < minimum_movement_duration:
                candidate_durations *= minimum_movement_duration / float(
                    np.sum(candidate_durations)
                )
            contact_timeout = max(
                minimum_contact_timeout,
                float(np.sum(candidate_durations)),
            )
            candidate = rollout_fr3_transit(
                model,
                phase_state,
                adapter,
                candidate_plan.waypoint_actions,
                tuple(float(value) for value in candidate_durations),
                initial_controller_state=release_controller_state,
                initial_command_action=release_start_action,
                gripper_operation=GripperOperation.MOVE,
                settle_duration=0.05,
                settle_maximum_duration=_RELEASE_SETTLE_MAXIMUM_DURATION,
                settle_stable_duration=0.10,
                terminal_position_tolerance=0.020,
                # Release owns no task-relative terminal orientation. Once a
                # stationary, fully clear state is certified, the following
                # command is reanchored to the measured pose before planning.
                terminal_orientation_tolerance=0.080,
                terminal_linear_speed_tolerance=0.010,
                terminal_angular_speed_tolerance=0.050,
                label=f"{spec.label}_release_candidate_{attempt}",
            )
            candidate.result.metrics.update(candidate_plan.metrics)
            last_candidate = candidate
            release_stationary = _release_is_stationary(
                model,
                adapter,
                candidate,
            )
            release_audit = _audit_inherited_release(
                context,
                frames,
                spec,
                candidate.result.best_states,
                free_safety,
                inherited_pairs,
                contact_timeout=contact_timeout,
            )
            whole = audit_fr3_trajectory(
                model,
                candidate.result.best_states,
                release_safety,
            )
            clear, terminal_clearance = check_fr3_transit_configuration(
                model,
                candidate.terminal_state,
                adapter,
            )
            if (
                candidate.result.metrics.get("transit_valid", False)
                and release_audit.valid
                and whole.valid
                and clear
                and not release_stationary
            ):
                candidate = _stabilize_release_candidate(
                    model,
                    adapter,
                    candidate,
                    label=f"{spec.label}_release_candidate_{attempt}",
                )
                candidate.result.metrics.update(candidate_plan.metrics)
                last_candidate = candidate
                release_stationary = _release_is_stationary(
                    model,
                    adapter,
                    candidate,
                )
                release_audit = _audit_inherited_release(
                    context,
                    frames,
                    spec,
                    candidate.result.best_states,
                    free_safety,
                    inherited_pairs,
                    contact_timeout=contact_timeout,
                )
                whole = audit_fr3_trajectory(
                    model,
                    candidate.result.best_states,
                    release_safety,
                )
                clear, terminal_clearance = check_fr3_transit_configuration(
                    model,
                    candidate.terminal_state,
                    adapter,
                )
            pair_minimum_by_key = dict(release_audit.pair_minimum_distances)
            candidate.result.metrics.update(
                {
                    "release_attempt": attempt,
                    "release_initial_target_clearance": initial_target_clearance,
                    "release_terminal_clearance": terminal_clearance,
                    "release_semantic_valid": release_audit.valid,
                    "release_whole_arm_valid": whole.valid,
                    "release_terminal_clear": clear,
                    "release_stationary_valid": release_stationary,
                    "release_contact_separation_valid": (
                        release_audit.contacts_released
                    ),
                    "release_clearance_acquisition_valid": clear,
                    "release_effective_movement_duration": float(
                        np.sum(candidate_durations)
                    ),
                    "release_initial_controller_reanchored": True,
                    "release_effective_contact_timeout": contact_timeout,
                    "release_inherited_pair_count": len(inherited_pairs),
                    "release_inherited_contact_pair_count": len(
                        inherited_contact_exemptions
                    ),
                    "release_entry_pair_states": {
                        "contact": sum(
                            pair.contact_equivalent for pair in inherited_pairs
                        ),
                        "near": sum(
                            not pair.contact_equivalent
                            for pair in inherited_pairs
                        ),
                        "tracking": len(entry_tracking_pairs),
                    },
                    "release_entry_contract": [
                        {
                            "robot_geom": _geom_name(
                                model,
                                pair.robot_geom_id,
                            ),
                            "environment_geom": _geom_name(
                                model,
                                pair.environment_geom_id,
                            ),
                            "entry_state": pair.entry_state,
                            "initial_distance": pair.initial_distance,
                            "minimum_distance": pair.minimum_distance,
                        }
                        for pair in release_constraints
                    ],
                    "release_forbidden_contact_steps": (
                        release_audit.new_contact_steps
                        + release_audit.penetration_worsening_steps
                    ),
                    "release_inherited_contact_steps": (
                        release_audit.inherited_contact_steps
                    ),
                    "release_separation_step": release_audit.separation_step,
                    "release_separation_time": release_audit.separation_time,
                    "release_last_contact_step": (
                        release_audit.last_contact_step
                    ),
                    "release_terminal_separation_duration": (
                        release_audit.terminal_separation_duration
                    ),
                    "release_new_contact_steps": release_audit.new_contact_steps,
                    "release_recontact_steps": release_audit.recontact_steps,
                    "release_penetration_worsening_steps": (
                        release_audit.penetration_worsening_steps
                    ),
                    "release_maximum_penetration_regression": (
                        release_audit.maximum_penetration_regression
                    ),
                    "release_stabilization_used": bool(
                        candidate.result.metrics.get(
                            "release_stabilization_used",
                            False,
                        )
                    ),
                    "release_replan_round": replan_count,
                    "release_minimum_inherited_distance": (
                        release_audit.minimum_inherited_distance
                    ),
                    "release_minimum_inherited_pair": (
                        None
                        if release_audit.minimum_inherited_pair is None
                        else [
                            _geom_name(model, geom_id)
                            for geom_id in release_audit.minimum_inherited_pair
                        ]
                    ),
                    "release_pair_audits": [
                        {
                            "robot_geom": _geom_name(
                                model,
                                pair.robot_geom_id,
                            ),
                            "environment_geom": _geom_name(
                                model,
                                pair.environment_geom_id,
                            ),
                            "entry_state": pair.entry_state,
                            "initial_distance": pair.initial_distance,
                            "minimum_distance": pair_minimum_by_key[pair.key],
                        }
                        for pair in inherited_pairs
                    ],
                    "release_maximum_protected_drift": (
                        release_audit.maximum_protected_drift
                    ),
                    "release_first_violation_kind": (
                        release_audit.first_violation_kind
                    ),
                    "release_first_violation_step": (
                        release_audit.first_violation_step
                    ),
                    "release_first_violation_pair": (
                        None
                        if release_audit.first_violation_pair is None
                        else [
                            _geom_name(model, geom_id)
                            for geom_id in release_audit.first_violation_pair
                        ]
                    ),
                    **_whole_arm_metrics(whole),
                }
            )
            if (
                candidate.result.metrics.get("transit_valid", False)
                and release_stationary
                and release_audit.valid
                and whole.valid
                and clear
            ):
                release = candidate
                inherited_release_audit = release_audit
                break
            rejection = {
                "attempt": attempt,
                "shortlist_rank": candidate_plan.metrics.get(
                    "release_shortlist_rank"
                ),
                "planning_backend": candidate_plan.metrics.get(
                    "planning_backend"
                ),
                "finite": bool(
                    candidate.result.metrics.get("transit_valid", False)
                ),
                "converged": bool(
                    candidate.result.metrics.get(
                        "transit_convergence_valid",
                        False,
                    )
                ),
                "stationary": release_stationary,
                "contacts_released": release_audit.contacts_released,
                "semantic_valid": release_audit.valid,
                "whole_arm_valid": whole.valid,
                "terminal_clear": clear,
                "terminal_clearance": terminal_clearance,
                "first_violation_kind": release_audit.first_violation_kind,
                "first_violation_step": release_audit.first_violation_step,
                "first_violation_pair": candidate.result.metrics[
                    "release_first_violation_pair"
                ],
                "new_contact_steps": release_audit.new_contact_steps,
                "recontact_steps": release_audit.recontact_steps,
                "clearance_regression_steps": (
                    release_audit.penetration_worsening_steps
                ),
                "maximum_penetration_regression": (
                    release_audit.maximum_penetration_regression
                ),
                "last_contact_step": release_audit.last_contact_step,
                "terminal_separation_duration": (
                    release_audit.terminal_separation_duration
                ),
                "stabilization_used": bool(
                    candidate.result.metrics.get(
                        "release_stabilization_used",
                        False,
                    )
                ),
                "replan_round": replan_count,
                "minimum_environment_clearance": (
                    whole.minimum_environment_clearance
                ),
                "minimum_environment_clearance_pair": (
                    whole.minimum_environment_clearance_pair
                ),
            }
            candidate_rejections.append(rejection)
            if not release_audit.valid:
                feedback_direction = _release_feedback_direction(
                    model,
                    candidate.result.best_states,
                    free_safety,
                    release_audit,
                )
            elif not whole.valid:
                feedback_direction = _whole_arm_feedback_direction(
                    model,
                    candidate.result.best_states,
                    free_safety,
                    whole,
                )
            elif not clear:
                feedback_direction = _terminal_release_feedback_direction(
                    model,
                    candidate.result.best_states,
                    free_safety,
                    release_constraints,
                )
            else:
                feedback_direction = None
            _prioritize_release_plans(
                pending_release_plans,
                feedback_direction,
            )
        if release is None:
            if last_candidate is not None:
                last_candidate.result.metrics["release_candidate_rejections"] = (
                    candidate_rejections
                )
            raise FR3TransitViolation(
                "failed to certify a safe, stationary terminal release after "
                f"{len(candidate_rejections)} controller rollouts",
                result=(None if last_candidate is None else last_candidate.result),
            )
        release.result.metrics["release_candidate_rejections"] = (
            candidate_rejections
        )
        phases.append(release)
        phase_state = release.terminal_state
        phase_controller = release.controller_state
        terminal_data = mujoco.MjData(model)
        terminal_data.qpos[:] = phase_state[: model.nq]
        terminal_data.qvel[:] = phase_state[model.nq :]
        mujoco.mj_fwdKinematics(model, terminal_data)
        measured_release_action = adapter.action_from_data(model, terminal_data)
        commanded_release_action = release.terminal_action
        release_reanchor_distance = float(
            np.linalg.norm(
                adapter.target_pose_world(commanded_release_action)[0]
                - adapter.target_pose_world(measured_release_action)[0]
            )
        )
        # Once the physical state owns both certificates, re-anchor subsequent
        # commands to that measured clear pose. Preserve the commanded aperture
        # so the internal phase boundary remains continuous.
        measured_release_action[6] = commanded_release_action[6]
        phase_action = measured_release_action
        release.result.metrics.update(
            {
                "release_command_reanchored": True,
                "release_command_reanchor_distance": release_reanchor_distance,
            }
        )
        release_metrics = dict(release.result.metrics)
    desired_aperture = float(nominal_waypoints[-1, 6])
    configuration_stage = next(
        (stage for stage in spec.stages if isinstance(stage, ConfigureGripper)),
        None,
    )
    actual_aperture = float(
        adapter.gripper_apertures(model, phase_state[None])[0]
    )
    aperture_tolerance = 0.05 * (
        adapter.gripper_range[1] - adapter.gripper_range[0]
    )
    if abs(actual_aperture - desired_aperture) > aperture_tolerance:
        configured_action = phase_action.copy()
        configured_action[6] = desired_aperture
        configuration_duration = minimum_pose_segment_duration(
            adapter, phase_action, configured_action
        )
        if configuration_stage is not None:
            configuration_duration = max(
                configuration_duration,
                configuration_stage.duration,
            )
        configuration = _rollout_free_gripper_configuration(
            model,
            adapter,
            phase_state,
            phase_controller,
            phase_action,
            desired_aperture,
            duration=configuration_duration,
            label=f"{spec.label}_gripper_configuration",
        )
        _audit_free_phase(model, configuration, free_safety)
        phases.append(configuration)
        phase_state = configuration.terminal_state
        phase_controller = configuration.controller_state
        terminal_data = mujoco.MjData(model)
        terminal_data.qpos[:] = phase_state[: model.nq]
        terminal_data.qvel[:] = phase_state[model.nq :]
        mujoco.mj_fwdKinematics(model, terminal_data)
        measured_configuration_action = adapter.action_from_data(
            model, terminal_data
        )
        commanded_configuration_action = configuration.terminal_action
        commanded_position, commanded_orientation = adapter.target_pose_world(
            commanded_configuration_action
        )
        measured_position, measured_orientation = adapter.target_pose_world(
            measured_configuration_action
        )
        measured_configuration_action[6] = commanded_configuration_action[6]
        phase_action = measured_configuration_action
        configuration.result.metrics.update(
            {
                "configuration_command_reanchored": True,
                "configuration_command_reanchor_distance": float(
                    np.linalg.norm(commanded_position - measured_position)
                ),
                "configuration_command_reanchor_orientation": (
                    _quaternion_distance(
                        commanded_orientation,
                        measured_orientation,
                    )
                ),
            }
        )
        configuration_metrics = dict(configuration.result.metrics)

    guide_actions: list[np.ndarray] = []
    guide_durations: list[float] = []
    previous_position, previous_orientation = adapter.target_pose_world(
        nominal_waypoints[0]
    )
    for action, duration in zip(nominal_waypoints[1:], nominal_durations):
        position, orientation = adapter.target_pose_world(action)
        spatial = bool(
            np.linalg.norm(position - previous_position) > 1e-9
            or _quaternion_distance(orientation, previous_orientation) > 1e-9
        )
        if spatial:
            guide = np.asarray(action, dtype=float).copy()
            guide[6] = desired_aperture
            guide_actions.append(guide)
            guide_durations.append(float(duration))
        previous_position = position
        previous_orientation = orientation
    target_action = np.asarray(nominal_waypoints[-1], dtype=float).copy()
    target_action[6] = desired_aperture

    def rollout_plan(candidate_plan: FR3TransitPlan) -> FR3TransitResult:
        candidate = rollout_fr3_transit(
            model,
            phase_state,
            adapter,
            candidate_plan.waypoint_actions,
            candidate_plan.segment_durations,
            initial_controller_state=phase_controller,
            initial_command_action=phase_action,
            gripper_operation=GripperOperation.MOVE,
            settle_duration=settle.duration,
            # The hardware-parity equilibrium filter has an approximately
            # one-second time constant. Give the advisory check enough time to
            # observe convergence instead of reporting an avoidable miss.
            settle_maximum_duration=max(
                float(settle.maximum_duration),
                float(settle.duration) + 4.0,
            ),
            settle_stable_duration=settle.stable_duration,
            # The Cartesian equilibrium remains compliant even in free space
            # because joint-limit and nullspace torques can offset the task
            # target. Collision safety is audited on realized states below;
            # this tolerance only decides when the stationary handoff settles.
            terminal_position_tolerance=0.040,
            label=spec.label,
        )
        candidate.result.metrics.update(candidate_plan.metrics)
        return candidate

    try:
        plan = plan_fr3_transit(
            model,
            phase_state,
            adapter,
            phase_action,
            target_action,
            guide_actions=guide_actions,
            handoff_orientation_candidates=_equivalent_handoff_orientations(
                frames,
                spec,
            ),
            seed=_derive_transit_planning_seed(
                transit_seed,
                transit_index,
                "handoff",
            ),
            handoff_tool_geom_ids=tuple(sorted(tool_ids)),
            handoff_target_geom_ids=tuple(sorted(target_ids)),
        )
        plan.metrics.update(
            {
                "transit_seed": int(transit_seed),
                "transit_index": int(transit_index),
                "planning_seed_stream": "handoff",
            }
        )
        if (
            plan.metrics["planning_backend"] == "typed_guide"
            and len(guide_durations) == len(plan.segment_durations)
        ):
            plan = replace(
                plan,
                segment_durations=tuple(
                    max(planned, learned)
                    for planned, learned in zip(
                        plan.segment_durations,
                        guide_durations,
                    )
                ),
            )
    except FR3TransitPlanningFailure as exc:
        failure = FR3TransitViolation(str(exc))
        _retain_transit_failure(
            failure,
            phases,
            None,
            label=spec.label,
            timestep=float(model.opt.timestep),
        )
        raise failure from exc
    spatial = rollout_plan(plan)
    _audit_free_phase(model, spatial, handoff_safety)
    phases.append(spatial)

    result = _concatenate_transit_phases(phases, spec.label, model.opt.timestep)
    raw_audit = audit_transit(
        context,
        frames,
        spec,
        result.result.best_states,
        adapter=adapter,
        controls=result.result.trajectory.controls,
    )
    if inherited_release_audit is not None:
        drift_limit = (
            spec.protected_drift_limit.resolve(frames)
            if spec.protected_drift_limit is not None
            else 0.0
        )
        drift_valid = bool(
            not spec.protected_roles
            or raw_audit.maximum_protected_drift <= drift_limit
        )
        release_forbidden_steps = (
            inherited_release_audit.new_contact_steps
            + inherited_release_audit.penetration_worsening_steps
        )
        audit = replace(
            raw_audit,
            valid=bool(inherited_release_audit.valid and drift_valid),
            forbidden_contact_steps=release_forbidden_steps,
            inherited_contact_steps=(
                inherited_release_audit.inherited_contact_steps
            ),
        )
    else:
        audit = raw_audit
    # Each phase has already recorded its stricter final-OSC findings above.
    # The combined diagnostic must retain the release exemptions at the
    # beginning; otherwise a valid inherited contact would be mislabeled.
    combined_safety = (
        release_safety if release_safety is not None else handoff_safety
    )
    whole_arm_audit = audit_fr3_trajectory(
        model,
        result.result.best_states,
        combined_safety,
    )
    controls = np.asarray(result.result.trajectory.controls, dtype=float)
    states = np.asarray(result.result.best_states, dtype=float)
    commanded_aperture = controls[:, 6]
    actual_aperture = adapter.gripper_apertures(model, states)
    metrics = result.result.metrics
    transit_convergence_valid = all(
        bool(
            phase.result.metrics.get(
                "release_stationary_valid",
                phase.result.metrics.get(
                    "transit_convergence_valid",
                    False,
                ),
            )
        )
        for phase in phases
    )
    finite_execution = bool(
        all(
            phase.result.metrics.get("transit_valid", False)
            for phase in phases
        )
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(controls))
        and np.all(np.isfinite(result.controller_observations))
        and np.all(np.isfinite(result.controller_state))
    )
    phase_advisories = tuple(
        advisory
        for phase in phases
        for advisory in phase.result.metrics.get("audit_advisories", ())
    )
    final_osc_advisories: list[str] = []
    if not transit_convergence_valid:
        final_osc_advisories.append("transit_convergence_or_joint_margin")
    if not audit.valid:
        final_osc_advisories.append("semantic_transit")
    if not whole_arm_audit.valid:
        final_osc_advisories.append("whole_arm_clearance_or_contact")
    projection_advisories = (
        ("next_primitive_initial_knot_requires_safe_handoff_projection",)
        if plan.metrics.get("handoff_projection_required", False)
        else ()
    )
    audit_advisories = list(
        dict.fromkeys(
            (
                *projection_advisories,
                *phase_advisories,
                *final_osc_advisories,
            )
        )
    )
    metrics.update(
        {
            "transit_policy": "collision_aware_release_handoff_v4",
            "transit_seed": int(transit_seed),
            "transit_index": int(transit_index),
            "transit_valid": finite_execution,
            "transit_convergence_valid": transit_convergence_valid,
            "transit_audit_valid": audit.valid,
            "forbidden_contact_steps": audit.forbidden_contact_steps,
            "inherited_contact_steps": audit.inherited_contact_steps,
            "maximum_protected_drift": audit.maximum_protected_drift,
            "whole_arm_audit_valid": whole_arm_audit.valid,
            "self_collision_steps": whole_arm_audit.self_collision_steps,
            "unapproved_robot_contact_steps": whole_arm_audit.unapproved_contact_steps,
            "minimum_environment_clearance": (
                whole_arm_audit.minimum_environment_clearance
            ),
            "minimum_enforced_environment_clearance": (
                whole_arm_audit.minimum_enforced_environment_clearance
            ),
            "minimum_declared_proximity_clearance": (
                whole_arm_audit.minimum_declared_proximity_clearance
            ),
            "minimum_declared_proximity_margin": (
                whole_arm_audit.minimum_declared_proximity_margin
            ),
            "declared_proximity_clearance_valid": (
                whole_arm_audit.declared_proximity_clearance_valid
            ),
            "permitted_fingertip_contact_steps": (
                whole_arm_audit.permitted_fingertip_contact_steps
            ),
            "maximum_closed_stop_penetration": (
                whole_arm_audit.maximum_closed_stop_penetration
            ),
            "maximum_finger_synchronization_error": (
                whole_arm_audit.maximum_finger_synchronization_error
            ),
            "commanded_aperture_range": [
                float(np.min(commanded_aperture)),
                float(np.max(commanded_aperture)),
            ],
            "actual_aperture_range": [
                float(np.min(actual_aperture)),
                float(np.max(actual_aperture)),
            ],
            "commanded_aperture_total_variation": float(
                np.sum(np.abs(np.diff(commanded_aperture)))
            ),
            "actual_aperture_total_variation": float(
                np.sum(np.abs(np.diff(actual_aperture)))
            ),
            "maximum_aperture_tracking_error": float(
                np.max(np.abs(commanded_aperture - actual_aperture))
            ),
            "aperture_motion_during_target_contact_steps": (
                audit.aperture_motion_during_target_contact_steps
            ),
            "maximum_aperture_motion_during_target_contact": (
                audit.maximum_aperture_motion_during_target_contact
            ),
            "commanded_aperture_motion_during_target_contact_steps": (
                audit.commanded_aperture_motion_during_target_contact_steps
            ),
            "maximum_commanded_aperture_motion_during_target_contact": (
                audit.maximum_commanded_aperture_motion_during_target_contact
            ),
            "release": release_metrics,
            "planning": dict(plan.metrics),
            "audit_advisories": audit_advisories,
            "audit_advisory_triggered": bool(audit_advisories),
            "final_osc_rollout_advisory": {
                "triggered": bool(final_osc_advisories),
                "findings": final_osc_advisories,
                "convergence_valid": transit_convergence_valid,
                "semantic_audit_valid": bool(audit.valid),
                "whole_arm_audit_valid": bool(whole_arm_audit.valid),
            },
            "initial_knot_advisory": (
                {
                    "kind": "collision_free_handoff_projection",
                    "desired_action": plan.metrics["handoff_desired_action"],
                    "projected_action": plan.metrics["handoff_projected_action"],
                    "translation": plan.metrics[
                        "handoff_projection_translation"
                    ],
                    "orientation": plan.metrics[
                        "handoff_projection_orientation"
                    ],
                    "required_clearance": plan.metrics[
                        "handoff_required_clearance"
                    ],
                    "suggestion": (
                        "make the first nominal knot a collision-free "
                        "precontact knot and begin contact acquisition in a "
                        "later primitive knot"
                    ),
                }
                if plan.metrics.get("handoff_projection_required", False)
                else None
            ),
        }
    )
    if configuration_metrics is not None:
        metrics["gripper_configuration"] = configuration_metrics
    if not metrics["transit_valid"]:
        raise FR3TransitViolation(
            "FR3 transit produced a non-finite execution",
            result=result.result,
        )
    return result


__all__ = [
    "Approach",
    "AxisRef",
    "ConfigureGripper",
    "InheritedReleaseWindow",
    "RaiseToClearance",
    "Reorient",
    "RoleFrame",
    "ScaleExpr",
    "Settle",
    "TransitAudit",
    "TransitSpec",
    "TransitStage",
    "Withdraw",
    "audit_transit",
    "execute_transit",
    "realize_transit_waypoints",
]
