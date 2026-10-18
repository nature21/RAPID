"""Interactively inspect an FR3 scene with OSC control or kinematic dragging.

Use ``scripts/rapid-view SCENE`` for OSC control, or add
``--controller kinematic`` for direct IK dragging. Scene references may be
short names, Python modules, task directories, or paths to scene.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib
import importlib.util
import math
from pathlib import Path
from queue import SimpleQueue
import time
from types import ModuleType
from typing import Any

import mujoco
import numpy as np

from primitive_core.fr3_osc import FR3OSCAdapter, FR3OSCConfig
from primitive_core.native_fr3_osc import NativeFR3OSCController
from primitive_core.policy import GripperOperation


FR3_ARM_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FR3_HAND_ACTUATOR = "fr3_hand"


@dataclass(frozen=True)
class InitialState:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray

    @classmethod
    def capture(cls, data: mujoco.MjData) -> InitialState:
        return cls(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            act=data.act.copy(),
            ctrl=data.ctrl.copy(),
            mocap_pos=data.mocap_pos.copy(),
            mocap_quat=data.mocap_quat.copy(),
        )

    def restore(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.act[:] = self.act
        data.ctrl[:] = self.ctrl
        data.mocap_pos[:] = self.mocap_pos
        data.mocap_quat[:] = self.mocap_quat
        mujoco.mj_forward(model, data)


class FR3Controller:
    """Kinematic end-effector dragging and FR3 actuator control."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        body_name: str,
        site_name: str | None,
    ) -> None:
        self.model = model
        self.arm_actuator_ids = np.asarray(
            [
                _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in FR3_ARM_NAMES
            ],
            dtype=int,
        )
        arm_joint_ids = np.asarray(
            [
                _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FR3_ARM_NAMES
            ],
            dtype=int,
        )
        self.arm_qpos_addresses = np.asarray(
            model.jnt_qposadr[arm_joint_ids],
            dtype=int,
        )
        self.arm_dof_addresses = np.asarray(
            model.jnt_dofadr[arm_joint_ids],
            dtype=int,
        )
        self.arm_joint_ranges = np.asarray(model.jnt_range[arm_joint_ids], dtype=float)
        self.end_effector_body_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        self.local_point = np.zeros(3, dtype=float)
        if site_name:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                rotation = data.xmat[self.end_effector_body_id].reshape(3, 3)
                self.local_point = rotation.T @ (
                    data.site_xpos[site_id] - data.xpos[self.end_effector_body_id]
                )
        self.hand_actuator_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            FR3_HAND_ACTUATOR,
        )
        finger_joint_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "finger_joint1",
        )
        self.gripper_range = np.asarray(
            model.jnt_range[finger_joint_id],
            dtype=float,
        )

    def hold_current_arm_pose(self, data: mujoco.MjData) -> None:
        targets = data.qpos[self.arm_qpos_addresses]
        limits = self.model.actuator_ctrlrange[self.arm_actuator_ids]
        data.ctrl[self.arm_actuator_ids] = np.clip(
            targets,
            limits[:, 0],
            limits[:, 1],
        )

    def drag_to(
        self,
        data: mujoco.MjData,
        target_body_pos: np.ndarray,
        target_body_quat: np.ndarray,
        *,
        iterations: int = 8,
    ) -> None:
        """Move the FR3 joints toward a MuJoCo perturbation target pose."""

        target_rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(target_rotation_flat, target_body_quat)
        target_point = (
            np.asarray(target_body_pos, dtype=float)
            + target_rotation_flat.reshape(3, 3) @ self.local_point
        )
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        rotation_error = np.zeros(3, dtype=float)

        for _ in range(iterations):
            body_rotation = data.xmat[self.end_effector_body_id].reshape(3, 3)
            current_point = (
                data.xpos[self.end_effector_body_id]
                + body_rotation @ self.local_point
            )
            mujoco.mj_jac(
                self.model,
                data,
                jacp,
                jacr,
                current_point,
                self.end_effector_body_id,
            )
            mujoco.mju_subQuat(
                rotation_error,
                target_body_quat,
                data.xquat[self.end_effector_body_id],
            )
            # mju_subQuat returns the angular correction in the body's local
            # frame, while MuJoCo's rotational Jacobian is world-aligned.
            world_rotation_error = body_rotation @ rotation_error
            error = np.concatenate(
                (target_point - current_point, world_rotation_error)
            )
            if float(np.linalg.norm(error)) < 1e-5:
                break

            jacobian = np.vstack((jacp, jacr))[:, self.arm_dof_addresses]
            damping = 2e-2
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6),
                error,
            )
            delta = np.clip(delta, -0.12, 0.12)
            qpos = data.qpos[self.arm_qpos_addresses] + delta
            data.qpos[self.arm_qpos_addresses] = np.clip(
                qpos,
                self.arm_joint_ranges[:, 0],
                self.arm_joint_ranges[:, 1],
            )
            mujoco.mj_forward(self.model, data)

        data.qvel[self.arm_dof_addresses] = 0.0
        self.hold_current_arm_pose(data)

    def set_gripper(self, data: mujoco.MjData, *, open_: bool) -> None:
        aperture = self.gripper_range[1 if open_ else 0]
        actuator_limits = self.model.actuator_ctrlrange[self.hand_actuator_id]
        data.ctrl[self.hand_actuator_id] = np.clip(
            aperture,
            actuator_limits[0],
            actuator_limits[1],
        )


def _native_reset_detected(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: FR3Controller,
    initial_state: InitialState,
) -> bool:
    """Detect a native viewer reset to qpos0 with cleared controls."""

    actuator_ids = controller.arm_actuator_ids
    qpos_addresses = controller.arm_qpos_addresses
    controls_cleared = bool(
        np.allclose(data.ctrl[actuator_ids], 0.0, rtol=0.0, atol=1e-12)
    )
    at_model_reference = bool(
        np.allclose(
            data.qpos[qpos_addresses],
            model.qpos0[qpos_addresses],
            rtol=0.0,
            atol=1e-12,
        )
    )
    home_differs_from_reference = not bool(
        np.allclose(
            initial_state.qpos[qpos_addresses],
            model.qpos0[qpos_addresses],
            rtol=0.0,
            atol=1e-12,
        )
    )
    return controls_cleared or (
        at_model_reference and home_differs_from_reference
    )


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"scene does not contain required {kind.name} {name!r}")
    return object_id


def _enable_fr3_gravity_compensation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[int, ...]:
    """Cancel gravity for the FR3 while preserving scene gravity."""

    root_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3")
    compensated = {root_id}
    for body_id in range(1, model.nbody):
        if int(model.body_parentid[body_id]) in compensated:
            compensated.add(body_id)
    body_ids = tuple(sorted(compensated))
    model.body_gravcomp[list(body_ids)] = 1.0
    # The compiled model initially has no compensated bodies. Refresh derived
    # constants so MuJoCo enables its gravity-compensation computation.
    # mj_setConst resets MjData as a side effect, so preserve the scene's home
    # keyframe state and nonzero controls around it.
    state = InitialState.capture(data)
    mujoco.mj_setConst(model, data)
    state.restore(model, data)
    return body_ids


def _scene_module_name(scene_ref: str) -> str:
    value = scene_ref.strip().replace("\\", "/").strip("/")
    if not value:
        raise ValueError("scene reference cannot be empty")
    if value.endswith("/scene.py"):
        value = value[: -len("/scene.py")]
    elif value.endswith(".scene"):
        return value
    elif value.endswith(".py"):
        value = value[:-3]

    value = value.replace("/", ".")
    if value.startswith("invention_real2sim."):
        value = f"tasks.{value}"
    elif not value.startswith("tasks."):
        value = f"tasks.invention_real2sim.{value}"
    return f"{value}.scene"


def _load_scene(scene_ref: str) -> ModuleType:
    path = Path(scene_ref).expanduser()
    if path.is_dir():
        path = path / "scene.py"
    if path.is_file():
        module_name = f"_view_scene_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load scene file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_name = _scene_module_name(scene_ref)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"could not load {scene_ref!r} as {module_name!r}; pass a reference "
            "such as 'invention_real2sim/demo_2_scene' or a path to scene.py"
        ) from exc


def _make_model_and_data(scene: ModuleType) -> tuple[mujoco.MjModel, mujoco.MjData]:
    make_model = getattr(scene, "make_model", None)
    if not callable(make_model):
        raise AttributeError(f"{scene.__name__} does not define make_model()")
    model = make_model()

    make_data = getattr(scene, "make_data", None)
    data = make_data(model) if callable(make_data) else mujoco.MjData(model)
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("scene make_model()/make_data() returned unexpected values")
    mujoco.mj_forward(model, data)
    return model, data


def _apply_camera(viewer: Any, scene: ModuleType) -> None:
    camera = getattr(scene, "DEFAULT_CAMERA", {}) or {}
    if camera.get("camera_lookat") is not None:
        viewer.cam.lookat[:] = np.asarray(camera["camera_lookat"], dtype=float)
    if camera.get("camera_distance") is not None:
        viewer.cam.distance = float(camera["camera_distance"])
    if camera.get("camera_azimuth") is not None:
        viewer.cam.azimuth = float(camera["camera_azimuth"])
    if camera.get("camera_elevation") is not None:
        viewer.cam.elevation = float(camera["camera_elevation"])


def _select_end_effector(
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str,
    site_name: str | None,
) -> None:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    local_point = np.zeros(3, dtype=float)
    if site_name:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            rotation = data.xmat[body_id].reshape(3, 3)
            local_point = rotation.T @ (data.site_xpos[site_id] - data.xpos[body_id])

    viewer.perturb.select = body_id
    viewer.perturb.localpos[:] = local_point
    viewer.perturb.refpos[:] = data.xpos[body_id]
    viewer.perturb.refquat[:] = data.xquat[body_id]
    viewer.perturb.refselpos[:] = data.xpos[body_id] + data.xmat[body_id].reshape(
        3, 3
    ) @ local_point
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_PERTFORCE)] = 1
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_PERTOBJ)] = 1


def _print_kinematic_help(scene_ref: str, body_name: str) -> None:
    print(
        f"\nScene: {scene_ref}\n"
        f"End effector: {body_name} (preselected)\n\n"
        "Mouse\n"
        "  Ctrl + right drag        move end effector in the vertical plane\n"
        "  Ctrl + Shift + right drag  move it in the horizontal plane\n"
        "  Ctrl + left drag         rotate it\n"
        "  Left double-click        select a different body\n\n"
        "Keys\n"
        "  Enter    toggle gripper open / closed\n"
        "  Backspace  reset scene to the FR3 home pose\n"
        "  Space    pause / resume physics (dragging still works)\n"
        "  F1       MuJoCo viewer help\n"
    )


def _run_kinematic(args: argparse.Namespace) -> int:
    scene = _load_scene(args.scene)
    model, data = _make_model_and_data(scene)
    _enable_fr3_gravity_compensation(model, data)
    mujoco.mj_forward(model, data)
    controller = FR3Controller(
        model,
        data,
        body_name=args.end_effector_body,
        site_name=args.end_effector_site or None,
    )
    initial_state = InitialState.capture(data)

    paused = bool(args.paused)
    reset_requested = False
    gripper_open = True
    gripper_toggle_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal paused, reset_requested, gripper_toggle_requested
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == " ":
            paused = not paused
            print("paused" if paused else "running")
        elif keycode == 259:  # GLFW_KEY_BACKSPACE; same action as the native UI.
            reset_requested = True
        elif keycode == 257:  # GLFW_KEY_ENTER; unused by the native viewer.
            gripper_toggle_requested = True

    viewer_module = importlib.import_module("mujoco.viewer")
    _print_kinematic_help(args.scene, args.end_effector_body)

    with viewer_module.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            _apply_camera(viewer, scene)
            _select_end_effector(
                viewer,
                model,
                data,
                body_name=args.end_effector_body,
                site_name=args.end_effector_site or None,
            )
        viewer.sync()

        while viewer.is_running():
            with viewer.lock():
                should_reset = reset_requested or _native_reset_detected(
                    model,
                    data,
                    controller,
                    initial_state,
                )
                if should_reset:
                    reset_requested = False
                    initial_state.restore(model, data)
                    gripper_open = True
                    _select_end_effector(
                        viewer,
                        model,
                        data,
                        body_name=args.end_effector_body,
                        site_name=args.end_effector_site or None,
                    )

            if gripper_toggle_requested:
                gripper_toggle_requested = False
                gripper_open = not gripper_open
                controller.set_gripper(data, open_=gripper_open)

            data.xfrc_applied[:] = 0.0
            with viewer.lock():
                drag_active = bool(
                    viewer.perturb.active or viewer.perturb.active2
                )
                dragging_end_effector = (
                    drag_active
                    and int(viewer.perturb.select)
                    == controller.end_effector_body_id
                )
                target_pos = viewer.perturb.refpos.copy()
                target_quat = viewer.perturb.refquat.copy()
                if drag_active and not dragging_end_effector and not paused:
                    mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)

            if dragging_end_effector:
                controller.drag_to(data, target_pos, target_quat)

            if not paused:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep) if not paused else 1.0 / 60.0)
    return 0


_PRESET_FACTORS = {"soft": 1.0 / 3.0, "default": 1.0, "stiff": 2.0}
_PRESET_ORDER = tuple(_PRESET_FACTORS)
_STEP_PRESETS = (
    (0.001, math.radians(0.5)),
    (0.005, math.radians(2.0)),
    (0.010, math.radians(5.0)),
    (0.020, math.radians(10.0)),
    (0.050, math.radians(20.0)),
)


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    result[1:] *= -1.0
    result /= np.linalg.norm(result)
    return result


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(result, left, right)
    return result / np.linalg.norm(result)


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    values = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(values, quaternion)
    return values.reshape(3, 3)


class EndEffectorTransform:
    """Rigid transform between the selected hand body and controlled site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        body_name: str,
        site_name: str,
    ) -> None:
        self.body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        self.site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self.body_id < 0:
            raise ValueError(f"scene does not contain body {body_name!r}")
        if self.site_id < 0:
            raise ValueError(f"scene does not contain site {site_name!r}")

        body_rotation = data.xmat[self.body_id].reshape(3, 3)
        self.site_position_body = body_rotation.T @ (
            data.site_xpos[self.site_id] - data.xpos[self.body_id]
        )
        site_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(site_quaternion, data.site_xmat[self.site_id])
        self.site_quaternion_body = _quaternion_product(
            _quaternion_inverse(data.xquat[self.body_id]),
            site_quaternion,
        )

    def site_pose_from_body_pose(
        self,
        body_position: np.ndarray,
        body_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = _quaternion_rotation(body_quaternion)
        position = (
            np.asarray(body_position, dtype=float)
            + rotation @ self.site_position_body
        )
        quaternion = _quaternion_product(
            np.asarray(body_quaternion, dtype=float),
            self.site_quaternion_body,
        )
        return position, quaternion

    def body_pose_from_site_pose(
        self,
        site_position: np.ndarray,
        site_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_quaternion = _quaternion_product(
            np.asarray(site_quaternion, dtype=float),
            _quaternion_inverse(self.site_quaternion_body),
        )
        body_position = (
            np.asarray(site_position, dtype=float)
            - _quaternion_rotation(body_quaternion) @ self.site_position_body
        )
        return body_position, body_quaternion


def _scaled_config(base: FR3OSCConfig, factor: float) -> FR3OSCConfig:
    if factor <= 0.0:
        raise ValueError("compliance scale must be positive")
    damping_factor = math.sqrt(factor)
    return replace(
        base,
        translational_stiffness=tuple(
            factor * value for value in base.translational_stiffness
        ),
        rotational_stiffness=tuple(
            factor * value for value in base.rotational_stiffness
        ),
        translational_damping=tuple(
            damping_factor * value for value in base.translational_damping
        ),
        rotational_damping=tuple(
            damping_factor * value for value in base.rotational_damping
        ),
    )


def _clip_target(
    action: np.ndarray,
    adapter: FR3OSCAdapter,
    home_action: np.ndarray,
    workspace_radius: float | None,
) -> np.ndarray:
    result = np.clip(
        np.asarray(action, dtype=float),
        adapter.action_ctrlrange[:, 0],
        adapter.action_ctrlrange[:, 1],
    )
    if workspace_radius is not None:
        result[:3] = np.clip(
            result[:3],
            home_action[:3] - workspace_radius,
            home_action[:3] + workspace_radius,
        )
    return result


def _key_command(keycode: int) -> str | None:
    return {
        32: "pause",  # GLFW_KEY_SPACE; same semantic as native viewer.
        259: "reset",  # GLFW_KEY_BACKSPACE; same semantic as native viewer.
        71: "grasp",  # GLFW_KEY_G
        320: "gripper-move",  # GLFW_KEY_KP_0
        321: "roll-",  # GLFW_KEY_KP_1
        322: "y-",  # GLFW_KEY_KP_2
        323: "z-",  # GLFW_KEY_KP_3
        324: "x-",  # GLFW_KEY_KP_4
        325: "hold",  # GLFW_KEY_KP_5
        326: "x+",  # GLFW_KEY_KP_6
        327: "roll+",  # GLFW_KEY_KP_7
        328: "y+",  # GLFW_KEY_KP_8
        329: "z+",  # GLFW_KEY_KP_9
        330: "cycle-step",  # GLFW_KEY_KP_DECIMAL
        331: "yaw-",  # GLFW_KEY_KP_DIVIDE
        332: "yaw+",  # GLFW_KEY_KP_MULTIPLY
        333: "pitch-",  # GLFW_KEY_KP_SUBTRACT
        334: "pitch+",  # GLFW_KEY_KP_ADD
        335: "cycle-preset",  # GLFW_KEY_KP_ENTER
        336: "select",  # GLFW_KEY_KP_EQUAL
    }.get(keycode)


def _make_config(args: argparse.Namespace) -> FR3OSCConfig:
    return FR3OSCConfig(
        translational_stiffness=(args.translation_stiffness,) * 3,
        rotational_stiffness=(args.rotation_stiffness,) * 3,
        translational_damping=(args.translation_damping,) * 3,
        rotational_damping=(args.rotation_damping,) * 3,
        maximum_task_force=args.maximum_force,
        maximum_task_torque=args.maximum_torque,
        maximum_torque_rate=args.maximum_torque_rate,
    )


def _status_line(
    observation: np.ndarray,
    target_action: np.ndarray,
    adapter: FR3OSCAdapter,
    preset: str,
    requested_operation: GripperOperation,
) -> str:
    target_position, _ = adapter.target_pose_world(target_action)
    error_mm = 1000.0 * np.linalg.norm(target_position - observation[:3])
    force = observation[13:16]
    torque = observation[19:26]
    return (
        f"{preset:7s}  target={np.round(target_position, 3)}  "
        f"error={error_mm:6.1f} mm  Fcmd={np.round(force, 1)} N  "
        f"max|tau|={np.max(np.abs(torque)):5.1f} Nm  "
        f"joint_margin={observation[26]:.3f} rad  "
        f"gripper={requested_operation.name}/mode{int(round(observation[50]))}"
    )


def _print_osc_help(args: argparse.Namespace, config: FR3OSCConfig) -> None:
    print(
        f"\nInteractive scene (OSC): {args.scene}\n"
        "The target is an equilibrium pose; the arm is never moved by IK.\n\n"
        "Mouse\n"
        "  Ctrl + right drag          move target in the vertical plane\n"
        "  Ctrl + Shift + right drag  move target in the horizontal plane\n"
        "  Ctrl + left drag           rotate target\n"
        "  Double-click another body, then Ctrl-drag to disturb it\n\n"
        "Numeric keypad (enable Num Lock)\n"
        "  KP 6/4      +x/-x        KP 8/2      +y/-y\n"
        "  KP 9/3      +z/-z        KP 7/1      +/-roll\n"
        "  KP +/-      +/-pitch     KP *//      +/-yaw\n"
        "  KP decimal  cycle keyboard step size\n"
        "  KP Enter    cycle soft/default/stiff compliance\n"
        "  KP 0        toggle open/closed with MOVE\n"
        "  G           close explicitly with GRASP\n"
        "  KP 5        hold the current measured pose\n"
        "  KP =        reselect the end effector for mouse control\n\n"
        "Shared viewer keys\n"
        "  Backspace  reset robot, objects, target, and controller\n"
        "  Space      pause/resume physics\n"
        "  F1         native MuJoCo viewer help\n\n"
        f"Default Kp={config.translational_stiffness} N/m, "
        f"Kr={config.rotational_stiffness} Nm/rad, "
        f"force cap={config.maximum_task_force} N/axis\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        default="invention_real2sim/flip_scene",
        help=("Scene shorthand, Python module, directory, or scene.py path "
              "(default: invention_real2sim/flip_scene)."),
    )
    parser.add_argument(
        "--controller", choices=("osc", "kinematic"), default="osc",
        help="OSC torque control (default) or kinematic IK dragging.",
    )
    parser.add_argument("--end-effector-body", default="hand")
    parser.add_argument("--end-effector-site", default="grasp_site")
    parser.add_argument("--translation-stiffness", type=float, default=300.0)
    parser.add_argument("--rotation-stiffness", type=float, default=25.0)
    parser.add_argument("--translation-damping", type=float, default=35.0)
    parser.add_argument("--rotation-damping", type=float, default=10.0)
    parser.add_argument("--maximum-force", type=float, default=60.0)
    parser.add_argument("--maximum-torque", type=float, default=10.0)
    parser.add_argument("--maximum-torque-rate", type=float, default=1000.0)
    parser.add_argument("--position-step", type=float, default=0.01)
    parser.add_argument(
        "--angle-step-degrees",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--workspace-radius",
        type=float,
        default=None,
        help=(
            "Optional per-axis target displacement allowed from home in meters; "
            "disabled by default."
        ),
    )
    parser.add_argument("--telemetry-hz", type=float, default=5.0)
    parser.add_argument("--paused", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "translation_stiffness",
        "rotation_stiffness",
        "translation_damping",
        "rotation_damping",
        "maximum_force",
        "maximum_torque",
        "maximum_torque_rate",
        "position_step",
        "angle_step_degrees",
        "telemetry_hz",
    )
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.workspace_radius is not None and (
        not math.isfinite(args.workspace_radius) or args.workspace_radius <= 0.0
    ):
        raise ValueError("--workspace-radius must be positive when provided")


def _set_target_marker(
    viewer: Any,
    transform: EndEffectorTransform,
    adapter: FR3OSCAdapter,
    target_action: np.ndarray,
) -> None:
    site_position, site_quaternion = adapter.target_pose_world(target_action)
    body_position, body_quaternion = transform.body_pose_from_site_pose(
        site_position,
        site_quaternion,
    )
    viewer.perturb.refpos[:] = body_position
    viewer.perturb.refquat[:] = body_quaternion
    viewer.perturb.refselpos[:] = site_position


def _run_osc(args: argparse.Namespace) -> int:
    _validate_args(args)
    scene = _load_scene(args.scene)
    model, data = _make_model_and_data(scene)
    base_config = _make_config(args)
    adapter = FR3OSCAdapter.from_model_data(model, data, config=base_config)
    home_action = adapter.action_from_data(model, data)
    target_action = home_action.copy()
    transform = EndEffectorTransform(
        model,
        data,
        body_name=args.end_effector_body,
        site_name=args.end_effector_site,
    )
    initial_state = InitialState.capture(data)
    controller = NativeFR3OSCController(model, data, adapter)
    mujoco.mj_forward(model, data)

    events: SimpleQueue[str] = SimpleQueue()
    paused = bool(args.paused)
    preset = "default"
    position_step = float(args.position_step)
    angle_step = math.radians(args.angle_step_degrees)
    step_preset_index = min(
        range(len(_STEP_PRESETS)),
        key=lambda index: abs(_STEP_PRESETS[index][0] - position_step),
    )
    gripper_open = bool(target_action[6] > np.mean(adapter.gripper_range))
    gripper_operation = GripperOperation.MOVE
    last_observation: np.ndarray | None = None

    def key_callback(keycode: int) -> None:
        command = _key_command(keycode)
        if command is not None:
            events.put(command)

    viewer_module = importlib.import_module("mujoco.viewer")
    _print_osc_help(args, base_config)
    next_telemetry = time.monotonic()

    viewer = viewer_module.launch_passive(
        model,
        data,
        key_callback=key_callback,
    )
    loop_error: BaseException | None = None
    try:
        with viewer.lock():
            _apply_camera(viewer, scene)
            _select_end_effector(
                viewer,
                model,
                data,
                body_name=args.end_effector_body,
                site_name=args.end_effector_site,
            )
            _set_target_marker(viewer, transform, adapter, target_action)
        viewer.sync()

        while viewer.is_running():
            started = time.monotonic()
            reselect = False
            update_marker = False
            while not events.empty():
                command = events.get()
                if command == "pause":
                    paused = not paused
                    print("\npaused" if paused else "\nrunning")
                elif command == "reset":
                    initial_state.restore(model, data)
                    controller.reset()
                    target_action = home_action.copy()
                    gripper_open = True
                    gripper_operation = GripperOperation.MOVE
                    last_observation = None
                    reselect = True
                    update_marker = True
                elif command == "gripper-move":
                    gripper_open = not gripper_open
                    target_action[6] = adapter.gripper_range[1 if gripper_open else 0]
                    gripper_operation = GripperOperation.MOVE
                elif command == "grasp":
                    gripper_open = False
                    target_action[6] = adapter.gripper_range[0]
                    gripper_operation = GripperOperation.GRASP
                elif command == "hold":
                    target_action = adapter.action_from_data(
                        model,
                        data,
                        gripper=target_action[6],
                    )
                    update_marker = True
                elif command == "select":
                    reselect = True
                    update_marker = True
                elif command == "cycle-preset":
                    preset = _PRESET_ORDER[
                        (_PRESET_ORDER.index(preset) + 1) % len(_PRESET_ORDER)
                    ]
                    config = _scaled_config(
                        base_config,
                        _PRESET_FACTORS[preset],
                    )
                    adapter = replace(adapter, config=config)
                    controller = NativeFR3OSCController(model, data, adapter)
                    print(
                        f"\n{preset} compliance: "
                        f"Kp={config.translational_stiffness}, "
                        f"Dp={config.translational_damping}"
                    )
                elif command == "cycle-step":
                    step_preset_index = (step_preset_index + 1) % len(
                        _STEP_PRESETS
                    )
                    position_step, angle_step = _STEP_PRESETS[step_preset_index]
                    print(
                        f"\nsteps: {position_step:.3f} m, "
                        f"{math.degrees(angle_step):.1f} deg"
                    )
                else:
                    channel_and_sign = {
                        "x+": (0, 1.0),
                        "x-": (0, -1.0),
                        "y+": (1, 1.0),
                        "y-": (1, -1.0),
                        "z+": (2, 1.0),
                        "z-": (2, -1.0),
                        "roll+": (3, 1.0),
                        "roll-": (3, -1.0),
                        "pitch+": (4, 1.0),
                        "pitch-": (4, -1.0),
                        "yaw+": (5, 1.0),
                        "yaw-": (5, -1.0),
                    }
                    channel, sign = channel_and_sign[command]
                    increment = position_step if channel < 3 else angle_step
                    target_action[channel] += sign * increment
                    target_action = _clip_target(
                        target_action,
                        adapter,
                        home_action,
                        args.workspace_radius,
                    )
                    update_marker = True

            data.xfrc_applied[:] = 0.0
            with viewer.lock():
                if reselect:
                    _select_end_effector(
                        viewer,
                        model,
                        data,
                        body_name=args.end_effector_body,
                        site_name=args.end_effector_site,
                    )
                drag_active = bool(viewer.perturb.active or viewer.perturb.active2)
                selected_body = int(viewer.perturb.select)
                dragging_end_effector = (
                    drag_active and selected_body == transform.body_id
                )
                body_target_position = viewer.perturb.refpos.copy()
                body_target_quaternion = viewer.perturb.refquat.copy()
                if drag_active and not dragging_end_effector and not paused:
                    mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)
                if update_marker and not dragging_end_effector:
                    _set_target_marker(viewer, transform, adapter, target_action)

            if dragging_end_effector:
                site_position, site_quaternion = transform.site_pose_from_body_pose(
                    body_target_position,
                    body_target_quaternion,
                )
                target_action = adapter.action_from_world_pose(
                    site_position,
                    site_quaternion,
                    gripper=target_action[6],
                )
                target_action = _clip_target(
                    target_action,
                    adapter,
                    home_action,
                    args.workspace_radius,
                )

            if not paused:
                last_observation = controller.step(
                    target_action,
                    gripper_operation,
                )

            viewer.sync()
            now = time.monotonic()
            if last_observation is not None and now >= next_telemetry:
                print(
                    "\r"
                    + _status_line(
                        last_observation,
                        target_action,
                        adapter,
                        preset,
                        gripper_operation,
                    ),
                    end="",
                    flush=True,
                )
                next_telemetry = now + 1.0 / args.telemetry_hz

            desired_period = float(model.opt.timestep) if not paused else 1.0 / 60.0
            time.sleep(max(0.0, desired_period - (time.monotonic() - started)))
    except BaseException as exc:
        loop_error = exc
    finally:
        try:
            viewer.close()
        except BaseException:
            if loop_error is None:
                raise

    if loop_error is not None:
        raise loop_error

    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.controller == "kinematic":
        return _run_kinematic(args)
    return _run_osc(args)


if __name__ == "__main__":
    raise SystemExit(main())
