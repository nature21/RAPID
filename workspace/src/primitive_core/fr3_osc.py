"""Cartesian impedance operational-space control for FR3 rollouts.

The optimizer-facing Cartesian action order is
``x, y, z, roll, pitch, yaw, gripper``. The first six values describe an
equilibrium pose in the FR3 base frame. They are converted to bounded joint
torques inside the native MuJoCo controller; joint-space commands are never
exposed to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import mujoco
import numpy as np


FR3_ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
FR3_ARM_ACTUATORS = FR3_ARM_JOINTS
FR3_GRIPPER_ACTUATOR = "fr3_hand"
FR3_GRIPPER_JOINTS = ("finger_joint1", "finger_joint2")
FR3_CARTESIAN_ACTION_DIM = 7
FR3_ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
FR3_HAND_SHORT_AXIS_LOCAL = (1.0, 0.0, 0.0)
FR3_HAND_JAW_AXIS_LOCAL = (0.0, 1.0, 0.0)
FR3_HAND_LONG_AXIS_LOCAL = (0.0, 0.0, 1.0)
FR3_CONTROLLER_STATE_NAMES = (
    *(f"tau_{index}" for index in range(1, 8)),
    "filtered_target_x",
    "filtered_target_y",
    "filtered_target_z",
    "filtered_target_qw",
    "filtered_target_qx",
    "filtered_target_qy",
    "filtered_target_qz",
    *(f"null_q_{index}" for index in range(1, 8)),
    *(f"latched_{name}" for name in FR3_ACTION_NAMES),
    "command_phase",
    "gripper_command",
    "gripper_mode",
    "gripper_stall_count",
)
FR3_OSC_OBSERVATION_NAMES = (
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qw",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "ee_wx",
    "ee_wy",
    "ee_wz",
    "wrench_fx",
    "wrench_fy",
    "wrench_fz",
    "wrench_tx",
    "wrench_ty",
    "wrench_tz",
    "tau_1",
    "tau_2",
    "tau_3",
    "tau_4",
    "tau_5",
    "tau_6",
    "tau_7",
    "minimum_joint_margin",
    "filtered_target_x",
    "filtered_target_y",
    "filtered_target_z",
    "filtered_target_qw",
    "filtered_target_qx",
    "filtered_target_qy",
    "filtered_target_qz",
    "null_q_1",
    "null_q_2",
    "null_q_3",
    "null_q_4",
    "null_q_5",
    "null_q_6",
    "null_q_7",
    "latched_x",
    "latched_y",
    "latched_z",
    "latched_roll",
    "latched_pitch",
    "latched_yaw",
    "latched_gripper",
    "command_phase",
    "gripper_command",
    "gripper_mode",
    "gripper_stall_count",
)


def fr3_controller_state_from_observation(observation: np.ndarray) -> np.ndarray:
    """Extract all state required for an exact controller phase handoff."""

    values = np.asarray(observation, dtype=float)
    if values.shape[-1] != len(FR3_OSC_OBSERVATION_NAMES):
        raise ValueError(
            "FR3 controller observation has the wrong trailing dimension"
        )
    indices = [
        FR3_OSC_OBSERVATION_NAMES.index(name)
        for name in FR3_CONTROLLER_STATE_NAMES
    ]
    state = np.asarray(values[..., indices], dtype=float)
    if not np.all(np.isfinite(state)):
        raise ValueError("FR3 controller state must be finite")
    return np.ascontiguousarray(state)


def _as_tuple(
    values: np.ndarray | tuple[float, ...],
    length: int,
    name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_quaternion(values: np.ndarray | tuple[float, ...]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain four values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion cannot have zero norm")
    return quaternion / norm


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _xyz_euler_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Invert MuJoCo's lower-case ``xyz`` Euler convention."""

    rotation = _quaternion_matrix(_normalize_quaternion(quaternion))
    pitch = math.asin(float(np.clip(rotation[0, 2], -1.0, 1.0)))
    cosine = math.cos(pitch)
    if abs(cosine) > 1e-8:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    else:
        # At gimbal lock choose yaw=0 and preserve the observable combined angle.
        yaw = 0.0
        roll = math.atan2(float(rotation[2, 1]), float(rotation[1, 1]))
    return np.asarray((roll, pitch, yaw), dtype=float)


def _named_id(
    model: mujoco.MjModel,
    kind: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"FR3 OSC model is missing required {kind.name} {name!r}")
    return object_id


def validate_fr3_osc_model(
    model: mujoco.MjModel,
    config: "FR3OSCConfig | None" = None,
) -> None:
    """Validate the physical actuator and frame contract required by FR3 OSC."""

    cfg = config or FR3OSCConfig()
    if model.nu < 8:
        raise ValueError(
            f"FR3 OSC requires seven arm actuators and one gripper; model.nu={model.nu}"
        )
    _named_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
    _named_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        cfg.end_effector_site_name,
    )
    arm_actuator_ids: list[int] = []
    for joint_name, actuator_name in zip(
        cfg.arm_joint_names,
        cfg.arm_actuator_names,
        strict=True,
    ):
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"FR3 arm joint {joint_name!r} must be a hinge")
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"FR3 arm actuator {actuator_name!r} must use joint transmission"
            )
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(
                f"FR3 arm actuator {actuator_name!r} does not target {joint_name!r}"
            )
        arm_actuator_ids.append(actuator_id)
    if len(set(arm_actuator_ids)) != 7:
        raise ValueError("FR3 OSC arm actuator names must resolve to seven actuators")
    if arm_actuator_ids != list(range(FR3_CARTESIAN_ACTION_DIM)):
        raise ValueError(
            "FR3 OSC requires the seven named arm actuators to occupy actuator "
            "slots 0 through 6 so Cartesian actions can be transported safely"
        )
    gripper_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        cfg.gripper_actuator_name,
    )
    if gripper_id in arm_actuator_ids:
        raise ValueError("FR3 gripper actuator must be distinct from arm actuators")
    if model.actuator_trntype[gripper_id] != mujoco.mjtTrn.mjTRN_TENDON:
        raise ValueError("FR3 gripper actuator must use a tendon transmission")
    tendon_id = int(model.actuator_trnid[gripper_id, 0])
    if tendon_id < 0:
        raise ValueError("FR3 gripper actuator must target a tendon")
    for joint_name in cfg.gripper_joint_names:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError(f"FR3 gripper joint {joint_name!r} must be a slide")


@dataclass(frozen=True)
class FR3OSCConfig:
    """Compliance, timing, and safety parameters for FR3 operational-space control."""

    profile: str = "hardware_parity"
    translational_stiffness: tuple[float, float, float] = (200.0, 200.0, 200.0)
    rotational_stiffness: tuple[float, float, float] = (30.0, 30.0, 30.0)
    translational_damping: tuple[float, float, float] = (
        28.284271247461902,
        28.284271247461902,
        28.284271247461902,
    )
    rotational_damping: tuple[float, float, float] = (
        10.954451150103322,
        10.954451150103322,
        10.954451150103322,
    )
    nullspace_stiffness: float = 0.5
    nullspace_damping: float = 1.4142135623730951
    joint_limit_stiffness: float = 12.0
    joint_limit_margin: float = 0.20
    nullspace_damping_regularizer: float = 0.05
    maximum_task_force: float = 60.0
    maximum_task_torque: float = 10.0
    maximum_torque_rate: float = 1000.0
    controller_rate_hz: float = 1000.0
    command_rate_hz: float = 50.0
    equilibrium_filter_alpha: float = 0.005
    torque_lowpass_cutoff_hz: float = 100.0
    libfranka_rate_limiting: bool = True
    gripper_grasp_force: float = 40.0
    gripper_maximum_force: float = 60.0
    gripper_move_stiffness: float = 10_000.0
    gripper_move_damping: float = 130.0
    gripper_move_speed: float = 0.02
    gripper_stall_velocity: float = 0.001
    gripper_stall_position_error: float = 0.001
    gripper_command_tolerance: float = 1e-6
    gripper_stall_steps: int = 5
    arm_joint_names: tuple[str, ...] = FR3_ARM_JOINTS
    arm_actuator_names: tuple[str, ...] = FR3_ARM_ACTUATORS
    gripper_actuator_name: str = FR3_GRIPPER_ACTUATOR
    gripper_joint_names: tuple[str, str] = FR3_GRIPPER_JOINTS
    end_effector_site_name: str = "grasp_site"
    base_body_name: str = "fr3"
    use_implicitfast: bool = True

    @classmethod
    def hardware_parity(cls, **overrides: object) -> "FR3OSCConfig":
        """Construct the default franka_ros Cartesian-impedance profile."""

        return cls(**overrides)

    @classmethod
    def legacy_native(cls, **overrides: object) -> "FR3OSCConfig":
        """Construct the pre-parity RAPID native OSC profile."""

        values: dict[str, object] = {
            "profile": "legacy_native",
            "translational_stiffness": (300.0, 300.0, 300.0),
            "rotational_stiffness": (25.0, 25.0, 25.0),
            "translational_damping": (77.0, 77.0, 77.0),
            "rotational_damping": (10.0, 10.0, 10.0),
            "nullspace_stiffness": 5.0,
            "nullspace_damping": 2.0,
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if self.profile not in {"hardware_parity", "legacy_native"}:
            raise ValueError("profile must be 'hardware_parity' or 'legacy_native'")
        object.__setattr__(
            self,
            "translational_stiffness",
            _as_tuple(self.translational_stiffness, 3, "translational_stiffness"),
        )
        object.__setattr__(
            self,
            "rotational_stiffness",
            _as_tuple(self.rotational_stiffness, 3, "rotational_stiffness"),
        )
        object.__setattr__(
            self,
            "translational_damping",
            _as_tuple(self.translational_damping, 3, "translational_damping"),
        )
        object.__setattr__(
            self,
            "rotational_damping",
            _as_tuple(self.rotational_damping, 3, "rotational_damping"),
        )
        object.__setattr__(self, "arm_joint_names", tuple(self.arm_joint_names))
        object.__setattr__(self, "arm_actuator_names", tuple(self.arm_actuator_names))
        object.__setattr__(self, "gripper_joint_names", tuple(self.gripper_joint_names))
        if len(self.arm_joint_names) != 7 or len(self.arm_actuator_names) != 7:
            raise ValueError("FR3 OSC requires seven arm joints and seven arm actuators")
        if len(self.gripper_joint_names) != 2:
            raise ValueError("FR3 OSC requires two gripper joints")
        positive_scalars = (
            "nullspace_damping_regularizer",
            "maximum_task_force",
            "maximum_task_torque",
            "maximum_torque_rate",
            "controller_rate_hz",
            "command_rate_hz",
            "torque_lowpass_cutoff_hz",
            "gripper_grasp_force",
            "gripper_maximum_force",
            "gripper_move_stiffness",
            "gripper_move_damping",
            "gripper_move_speed",
            "gripper_stall_velocity",
            "gripper_stall_position_error",
            "gripper_command_tolerance",
        )
        for field_name in positive_scalars:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.gripper_grasp_force > self.gripper_maximum_force:
            raise ValueError(
                "gripper_grasp_force cannot exceed gripper_maximum_force"
            )
        if (
            isinstance(self.gripper_stall_steps, bool)
            or int(self.gripper_stall_steps) != self.gripper_stall_steps
            or self.gripper_stall_steps < 1
        ):
            raise ValueError("gripper_stall_steps must be a positive integer")
        if self.command_rate_hz > self.controller_rate_hz:
            raise ValueError("command_rate_hz cannot exceed controller_rate_hz")
        if not 0.0 < self.equilibrium_filter_alpha <= 1.0:
            raise ValueError("equilibrium_filter_alpha must be within (0, 1]")
        for field_name in (
            "nullspace_stiffness",
            "nullspace_damping",
            "joint_limit_stiffness",
            "joint_limit_margin",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class FR3OSCAdapter:
    """Portable seven-channel Cartesian action contract for FR3 rollouts."""

    frame_position_world: tuple[float, float, float]
    frame_quaternion_world: tuple[float, float, float, float]
    nullspace_configuration: tuple[float, float, float, float, float, float, float]
    gripper_range: tuple[float, float]
    config: FR3OSCConfig = FR3OSCConfig()

    action_dim: ClassVar[int] = FR3_CARTESIAN_ACTION_DIM
    name: ClassVar[str] = "fr3_cartesian_impedance_osc"
    observation_names: ClassVar[tuple[str, ...]] = FR3_OSC_OBSERVATION_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_position_world",
            _as_tuple(self.frame_position_world, 3, "frame_position_world"),
        )
        frame_quaternion = _normalize_quaternion(self.frame_quaternion_world)
        object.__setattr__(
            self,
            "frame_quaternion_world",
            tuple(float(value) for value in frame_quaternion),
        )
        object.__setattr__(
            self,
            "nullspace_configuration",
            _as_tuple(self.nullspace_configuration, 7, "nullspace_configuration"),
        )
        gripper_range = _as_tuple(self.gripper_range, 2, "gripper_range")
        if gripper_range[0] >= gripper_range[1]:
            raise ValueError("gripper_range must be increasing")
        object.__setattr__(self, "gripper_range", gripper_range)

    @classmethod
    def from_model_data(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        config: FR3OSCConfig | None = None,
    ) -> "FR3OSCAdapter":
        """Capture the fixed FR3 base frame and home nullspace posture."""

        cfg = config or FR3OSCConfig()
        if data.model is not model:
            raise ValueError("data must belong to model")
        validate_fr3_osc_model(model, cfg)
        base_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        joint_ids = np.asarray(
            [
                _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in cfg.arm_joint_names
            ],
            dtype=int,
        )
        qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
        joint_ranges = np.asarray(model.jnt_range[joint_ids], dtype=float)
        nullspace_configuration = np.asarray(data.qpos[qpos_addresses], dtype=float)
        if not np.all(np.isfinite(nullspace_configuration)):
            raise ValueError("FR3 reset joint configuration must be finite")
        # Preserve the physical handoff state. If a retained phase ends just
        # outside a joint limit, give OSC an in-range nullspace recovery target
        # instead of rejecting the next phase before it can be evaluated.
        nullspace_configuration = np.clip(
            nullspace_configuration,
            joint_ranges[:, 0],
            joint_ranges[:, 1],
        )
        gripper_actuator = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            cfg.gripper_actuator_name,
        )
        gripper_range = tuple(
            float(value) for value in model.actuator_ctrlrange[gripper_actuator]
        )
        return cls(
            frame_position_world=tuple(float(value) for value in data.xpos[base_id]),
            frame_quaternion_world=tuple(float(value) for value in data.xquat[base_id]),
            nullspace_configuration=tuple(
                float(value) for value in nullspace_configuration
            ),
            gripper_range=gripper_range,
            config=cfg,
        )

    @property
    def action_ctrlrange(self) -> np.ndarray:
        return np.asarray(
            [
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-math.pi, math.pi),
                (-math.pi, math.pi),
                (-math.pi, math.pi),
                self.gripper_range,
            ],
            dtype=float,
        )

    def gripper_aperture(self, aperture_fraction: float) -> float:
        """Resolve a normalized aperture against the physical joint range."""

        fraction = float(aperture_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("aperture_fraction must be finite and within [0, 1]")
        lower, upper = self.gripper_range
        return float(lower + fraction * (upper - lower))

    def gripper_joint_addresses(self, model: mujoco.MjModel) -> tuple[int, int]:
        """Return the first finger addresses for legacy diagnostics."""

        return self.gripper_joint_address_pairs(model)[0]

    def gripper_joint_address_pairs(
        self,
        model: mujoco.MjModel,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return qpos/qvel addresses for both symmetric finger joints."""

        pairs: list[tuple[int, int]] = []
        for name in self.config.gripper_joint_names:
            joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
                raise ValueError("FR3 gripper joints must be scalar slides")
            pairs.append(
                (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
            )
        return pairs[0], pairs[1]

    def gripper_apertures(
        self,
        model: mujoco.MjModel,
        states: np.ndarray,
    ) -> np.ndarray:
        """Return the per-finger aperture represented by the fixed tendon."""

        values = np.asarray(states, dtype=float)
        addresses = [pair[0] for pair in self.gripper_joint_address_pairs(model)]
        return np.mean(values[..., addresses], axis=-1)

    def gripper_aperture_speeds(
        self,
        model: mujoco.MjModel,
        states: np.ndarray,
    ) -> np.ndarray:
        """Return fixed-tendon velocity from packed qpos/qvel state arrays."""

        values = np.asarray(states, dtype=float)
        addresses = [
            model.nq + pair[1] for pair in self.gripper_joint_address_pairs(model)
        ]
        return np.mean(values[..., addresses], axis=-1)

    @property
    def frame_rotation_world(self) -> np.ndarray:
        return _quaternion_matrix(np.asarray(self.frame_quaternion_world))

    def target_pose_world(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode one Cartesian action into a world-frame site pose."""

        value = np.asarray(action, dtype=float)
        if value.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        local_quaternion = np.empty(4, dtype=float)
        mujoco.mju_euler2Quat(local_quaternion, value[3:6], "xyz")
        world_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            world_quaternion,
            np.asarray(self.frame_quaternion_world),
            local_quaternion,
        )
        world_position = (
            np.asarray(self.frame_position_world)
            + self.frame_rotation_world @ value[:3]
        )
        return world_position, world_quaternion / np.linalg.norm(world_quaternion)

    def action_from_world_pose(
        self,
        position_world: np.ndarray,
        quaternion_world: np.ndarray,
        *,
        gripper: float,
    ) -> np.ndarray:
        """Encode a world-frame pose using the FR3 Cartesian action order."""

        local_position = self.frame_rotation_world.T @ (
            np.asarray(position_world, dtype=float)
            - np.asarray(self.frame_position_world)
        )
        inverse_frame = np.asarray(self.frame_quaternion_world) * np.asarray(
            (1.0, -1.0, -1.0, -1.0)
        )
        local_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            local_quaternion,
            inverse_frame,
            _normalize_quaternion(quaternion_world),
        )
        euler = _xyz_euler_from_quaternion(local_quaternion)
        action = np.concatenate((local_position, euler, (float(gripper),)))
        return np.clip(
            action,
            self.action_ctrlrange[:, 0],
            self.action_ctrlrange[:, 1],
        )

    def action_from_data(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        gripper: float | None = None,
    ) -> np.ndarray:
        site_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.config.end_effector_site_name,
        )
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        if gripper is None:
            addresses = self.gripper_joint_address_pairs(model)
            aperture = float(np.mean([data.qpos[pair[0]] for pair in addresses]))
        else:
            aperture = float(gripper)
        return self.action_from_world_pose(
            data.site_xpos[site_id],
            quaternion,
            gripper=aperture,
        )

__all__ = [
    "FR3_ACTION_NAMES",
    "FR3_ARM_ACTUATORS",
    "FR3_ARM_JOINTS",
    "FR3_CARTESIAN_ACTION_DIM",
    "FR3_GRIPPER_ACTUATOR",
    "FR3_GRIPPER_JOINTS",
    "FR3_HAND_JAW_AXIS_LOCAL",
    "FR3_HAND_LONG_AXIS_LOCAL",
    "FR3_HAND_SHORT_AXIS_LOCAL",
    "FR3_OSC_OBSERVATION_NAMES",
    "FR3OSCAdapter",
    "FR3OSCConfig",
    "validate_fr3_osc_model",
]
