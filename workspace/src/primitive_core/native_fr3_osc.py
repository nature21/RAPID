"""Native C++ rollout backend for the shared FR3 OSC action contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from judo.utils.rollout_backend import RolloutBackend

if TYPE_CHECKING:
    from primitive_core.fr3_osc import FR3OSCAdapter
    from primitive_core.rollout_control import RolloutTelemetrySpec

from primitive_core.fr3_osc import FR3_CONTROLLER_STATE_NAMES
from primitive_core.fr3_osc import (
    FR3_CARTESIAN_ACTION_DIM,
    FR3_OSC_OBSERVATION_NAMES,
)
from primitive_core.policy import GripperOperation


FR3_NATIVE_ABI_VERSION = 2


def _native_rebuild_error(module: Any, detail: str) -> RuntimeError:
    loaded = getattr(
        module,
        "NATIVE_MODULE_PATH",
        getattr(module, "__file__", "<unknown module path>"),
    )
    return RuntimeError(
        f"Incompatible native FR3 OSC extension loaded from {loaded}: {detail}. "
        "Rebuild it with `conda run -n rapid cmake -B "
        "third_party/judo/mujoco_extensions/build -S "
        "third_party/judo/mujoco_extensions && conda run -n rapid cmake "
        "--build third_party/judo/mujoco_extensions/build -j8`."
    )


def _load_native_types() -> tuple[type, type]:
    try:
        import mujoco_extensions.fr3_osc_rollout as native_module
    except ImportError as exc:
        raise RuntimeError(
            "Native FR3 OSC rollouts require the Judo C++ extension. Build it "
            "with `conda run -n rapid cmake -B "
            "third_party/judo/mujoco_extensions/build -S "
            "third_party/judo/mujoco_extensions && conda run -n rapid cmake "
            "--build third_party/judo/mujoco_extensions/build -j8`."
        ) from exc
    _validate_native_abi(native_module)
    NativeRolloutType = native_module.NativeFR3OSCRollout
    NativeControllerType = native_module.NativeFR3OSCController
    return NativeRolloutType, NativeControllerType


def _validate_native_abi(native_module: Any) -> None:
    expected = {
        "ABI_VERSION": FR3_NATIVE_ABI_VERSION,
        "ACTION_DIM": FR3_CARTESIAN_ACTION_DIM,
        "CONTROLLER_STATE_DIM": len(FR3_CONTROLLER_STATE_NAMES),
        "OBSERVATION_DIM": len(FR3_OSC_OBSERVATION_NAMES),
        "GRIPPER_OPERATION_MOVE": int(GripperOperation.MOVE),
        "GRIPPER_OPERATION_GRASP": int(GripperOperation.GRASP),
    }
    for name, value in expected.items():
        actual = getattr(native_module, name, None)
        if actual != value:
            raise _native_rebuild_error(
                native_module,
                f"expected {name}={value}, found {actual!r}",
            )


def _native_config(adapter: FR3OSCAdapter) -> dict[str, Any]:
    config = adapter.config
    control_range = adapter.action_ctrlrange
    return {
        "profile": config.profile,
        "frame_position_world": list(adapter.frame_position_world),
        "frame_quaternion_world": list(adapter.frame_quaternion_world),
        "nullspace_configuration": list(adapter.nullspace_configuration),
        "gripper_range": list(adapter.gripper_range),
        "stiffness": list(
            config.translational_stiffness + config.rotational_stiffness
        ),
        "damping": list(config.translational_damping + config.rotational_damping),
        "action_lower": control_range[:, 0].tolist(),
        "action_upper": control_range[:, 1].tolist(),
        "nullspace_stiffness": config.nullspace_stiffness,
        "nullspace_damping": config.nullspace_damping,
        "joint_limit_stiffness": config.joint_limit_stiffness,
        "joint_limit_margin": config.joint_limit_margin,
        "nullspace_damping_regularizer": config.nullspace_damping_regularizer,
        "maximum_task_force": config.maximum_task_force,
        "maximum_task_torque": config.maximum_task_torque,
        "maximum_torque_rate": config.maximum_torque_rate,
        "controller_rate_hz": config.controller_rate_hz,
        "command_rate_hz": config.command_rate_hz,
        "equilibrium_filter_alpha": config.equilibrium_filter_alpha,
        "torque_lowpass_cutoff_hz": config.torque_lowpass_cutoff_hz,
        "libfranka_rate_limiting": config.libfranka_rate_limiting,
        "gripper_grasp_force": config.gripper_grasp_force,
        "gripper_maximum_force": config.gripper_maximum_force,
        "gripper_move_stiffness": config.gripper_move_stiffness,
        "gripper_move_damping": config.gripper_move_damping,
        "gripper_move_speed": config.gripper_move_speed,
        "gripper_stall_velocity": config.gripper_stall_velocity,
        "gripper_stall_position_error": config.gripper_stall_position_error,
        "gripper_command_tolerance": config.gripper_command_tolerance,
        "gripper_stall_steps": config.gripper_stall_steps,
        "arm_joint_names": list(config.arm_joint_names),
        "arm_actuator_names": list(config.arm_actuator_names),
        "gripper_actuator_name": config.gripper_actuator_name,
        "gripper_joint_names": list(config.gripper_joint_names),
        "end_effector_site_name": config.end_effector_site_name,
        "use_implicitfast": config.use_implicitfast,
    }


def _native_telemetry_config(
    model: mujoco.MjModel,
    telemetry: RolloutTelemetrySpec | None,
) -> dict[str, Any] | None:
    if telemetry is None:
        return None
    for pair in telemetry.contact_pairs:
        for geom_id in pair.geom_ids_a + pair.geom_ids_b:
            if geom_id >= model.ngeom:
                raise ValueError(
                    f"contact telemetry geom id {geom_id} is outside model.ngeom="
                    f"{model.ngeom}"
                )
    return {
        "contact_pairs": [
            {
                "name": pair.name,
                "geom_ids_a": list(pair.geom_ids_a),
                "geom_ids_b": list(pair.geom_ids_b),
            }
            for pair in telemetry.contact_pairs
        ],
        "include_maximum_penetration": telemetry.include_maximum_penetration,
    }


class NativeFR3OSCRolloutBackend(RolloutBackend):
    """Run one FR3 Cartesian OSC trajectory per native MuJoCo worker."""

    implementation_name = "native_cpp_fr3_osc"

    def __init__(
        self,
        model: mujoco.MjModel,
        num_threads: int,
        adapter: FR3OSCAdapter,
        initial_controller_state: np.ndarray | None = None,
        telemetry: RolloutTelemetrySpec | None = None,
        gripper_operations: np.ndarray | None = None,
    ) -> None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        if adapter.name != "fr3_cartesian_impedance_osc":
            raise ValueError(
                "NativeFR3OSCRolloutBackend only supports the shared FR3 OSC adapter"
            )
        self.num_threads = num_threads
        self.model = model
        self.adapter = adapter
        self.telemetry = telemetry
        operations = None
        if gripper_operations is not None:
            operations = np.ascontiguousarray(gripper_operations, dtype=np.uint8)
            if operations.ndim != 1 or np.any(
                operations > int(GripperOperation.GRASP)
            ):
                raise ValueError(
                    "gripper_operations must be a one-dimensional "
                    "MOVE/GRASP stream"
                )
        self.gripper_operations = operations
        self.initial_controller_state = self._validate_controller_state(
            initial_controller_state
        )
        self._native_type = _load_native_types()[0]
        self._native: Any | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        self._native = self._native_type(
            self.model,
            self.num_threads,
            _native_config(self.adapter),
            _native_telemetry_config(self.model, self.telemetry),
        )

    @property
    def telemetry_names(self) -> tuple[str, ...]:
        if self.telemetry is None:
            return ()
        return self.telemetry.observation_names

    @staticmethod
    def _validate_controller_state(
        state: np.ndarray | None,
    ) -> np.ndarray | None:
        if state is None:
            return None
        value = np.ascontiguousarray(state, dtype=np.float64)
        expected_shape = (len(FR3_CONTROLLER_STATE_NAMES),)
        if value.shape != expected_shape:
            raise ValueError(
                f"initial_controller_state must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("initial_controller_state must be finite")
        return value.copy()

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: object = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        del last_policy_output
        if self._native is None:
            raise RuntimeError("native FR3 OSC rollout backend is closed")
        if x0.ndim == 1:
            x0 = np.tile(x0, (self.num_threads, 1))
        states = np.ascontiguousarray(x0, dtype=np.float64)
        actions = np.ascontiguousarray(controls, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError("x0 must have shape (batch, nq + nv)")
        if actions.ndim != 3:
            raise ValueError("controls must have shape (batch, steps, action_dim)")
        if states.shape[0] != self.num_threads or actions.shape[0] != self.num_threads:
            raise ValueError(
                "native controller rollout batch size must equal num_threads"
            )
        if states.shape[1] != self.model.nq + self.model.nv:
            raise ValueError(
                f"x0 must have {self.model.nq + self.model.nv} state values; "
                f"got {states.shape[1]}"
            )
        if actions.shape[-1] != self.adapter.action_dim:
            raise ValueError(
                f"expected action dimension {self.adapter.action_dim}; "
                f"got {actions.shape[-1]}"
            )
        operations = self.gripper_operations
        if operations is None:
            operations = np.zeros(actions.shape[1], dtype=np.uint8)
        if actions.shape[1] != operations.size:
            raise ValueError(
                "gripper operation stream length must match rollout steps"
            )
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError("rollout states and actions must be finite")
        controller_states = None
        if self.initial_controller_state is not None:
            controller_states = np.tile(
                self.initial_controller_state,
                (self.num_threads, 1),
            )
        output_states, output_sensors = self._native.rollout(
            states,
            actions,
            operations,
            controller_states,
        )
        return (
            np.asarray(output_states),
            np.asarray(output_sensors),
            None,
        )

    def update(self, num_threads: int) -> None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        self.num_threads = num_threads
        self._rebuild()

    def close(self) -> None:
        self._native = None


class NativeFR3OSCController:
    """Advance one visible MuJoCo world with the native FR3 OSC."""

    implementation_name = "native_cpp_fr3_osc"

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        adapter: FR3OSCAdapter,
    ) -> None:
        if data.model is not model:
            raise ValueError("data must belong to model")
        if adapter.name != "fr3_cartesian_impedance_osc":
            raise ValueError(
                "NativeFR3OSCController only supports the shared FR3 OSC adapter"
            )
        self.model = model
        self.data = data
        self.adapter = adapter
        native_type = _load_native_types()[1]
        self._native = native_type(model, data, _native_config(adapter))

    def step(
        self,
        action: np.ndarray,
        gripper_operation: GripperOperation | int = GripperOperation.MOVE,
    ) -> np.ndarray:
        """Apply one Cartesian target and advance one MuJoCo physics step."""

        value = np.ascontiguousarray(action, dtype=np.float64)
        if value.shape != (self.adapter.action_dim,):
            raise ValueError(f"action must have shape ({self.adapter.action_dim},)")
        if not np.all(np.isfinite(value)):
            raise ValueError("action must be finite")
        try:
            operation = GripperOperation(gripper_operation)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown gripper operation") from exc
        observation = np.asarray(self._native.step(value, int(operation)))
        expected_shape = (len(self.adapter.observation_names),)
        if observation.shape != expected_shape:
            raise RuntimeError(
                f"native OSC returned {observation.shape}; expected {expected_shape}"
            )
        return observation

    def reset(self) -> None:
        """Reset torque-rate state after the simulation state is restored."""

        self._native.reset()

    @property
    def controller_state(self) -> np.ndarray:
        """Return the torque-rate state needed to continue this trajectory."""

        state = np.asarray(self._native.controller_state(), dtype=np.float64)
        if state.shape != (len(FR3_CONTROLLER_STATE_NAMES),) or not np.all(
            np.isfinite(state)
        ):
            raise RuntimeError("native OSC returned an invalid controller state")
        return state.copy()

    def restore_controller_state(self, state: np.ndarray) -> None:
        """Restore torque-rate state when continuing a simulated trajectory."""

        value = NativeFR3OSCRolloutBackend._validate_controller_state(state)
        assert value is not None
        self._native.restore_controller_state(value)


__all__ = ["NativeFR3OSCController", "NativeFR3OSCRolloutBackend"]
