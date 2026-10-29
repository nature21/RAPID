"""Bake resting object poses into a staged Real2Sim runtime scene."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def settle_scene(
    scene_xml: Path,
    *,
    minimum_seconds: float = 2.0,
    maximum_seconds: float = 10.0,
    stable_seconds: float = 0.25,
    max_linear_speed: float = 0.001,
    max_angular_speed: float = 0.01,
) -> dict[str, object]:
    """Simulate unforced objects, then persist poses once motion stays small.

    Operates on the staged bundle after collision canonicalization and any
    validated repair. The source reconstruction and repair hashes stay intact.
    A timeout or invalid simulation leaves the XML untouched.
    """

    values = (
        minimum_seconds,
        maximum_seconds,
        stable_seconds,
        max_linear_speed,
        max_angular_speed,
    )
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("settling durations and speed limits must be finite and positive")
    if maximum_seconds < max(minimum_seconds, stable_seconds):
        raise ValueError(
            "maximum settling duration must cover minimum and stable durations"
        )

    tree = ET.parse(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = {body.get("name"): body for body in tree.findall("./worldbody/body")}
    objects: list[tuple[str, int, int, int]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        name = model.body(body_id).name
        if name not in bodies:
            raise ValueError(f"cannot save settled pose of non-root object {name!r}")
        objects.append(
            (name, body_id, int(model.jnt_qposadr[joint_id]),
             int(model.jnt_dofadr[joint_id]))
        )

    if not objects:
        return {"status": "ok", "duration_seconds": 0.0, "objects": {}}

    timestep = float(model.opt.timestep)
    minimum_steps = math.ceil(minimum_seconds / timestep)
    maximum_steps = math.ceil(maximum_seconds / timestep)
    required_stable_steps = math.ceil(stable_seconds / timestep)
    stable_steps = 0
    for step in range(1, maximum_steps + 1):
        mujoco.mj_step(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or any(warning.number for warning in data.warning)
        ):
            raise RuntimeError("scene settling failed: invalid state or MuJoCo warning")
        slow = all(
            np.linalg.norm(data.qvel[dof : dof + 3]) <= max_linear_speed
            and np.linalg.norm(data.qvel[dof + 3 : dof + 6]) <= max_angular_speed
            for _, _, _, dof in objects
        )
        stable_steps = stable_steps + 1 if slow else 0
        if step >= minimum_steps and stable_steps >= required_stable_steps:
            break
    else:
        raise RuntimeError(
            f"scene did not settle within {maximum_seconds:g} simulation seconds"
        )

    # mj_step's derived body transforms can lag the integrated qpos by one step.
    mujoco.mj_forward(model, data)
    poses = {}
    for name, body_id, _, _ in objects:
        body = bodies[name]
        for attribute in ("axisangle", "euler", "xyaxes", "zaxis"):
            body.attrib.pop(attribute, None)
        position = data.xpos[body_id]
        quaternion = data.xquat[body_id]
        body.set("pos", _vector(position))
        body.set("quat", _vector(quaternion))
        poses[name] = {"position": position.tolist(), "quaternion": quaternion.tolist()}

    # Explicit keyframes must reset the objects to the same resting poses.
    for key_id, key in enumerate(tree.findall("./keyframe/key")):
        qpos = model.key_qpos[key_id].copy()
        qvel = model.key_qvel[key_id].copy()
        for _, _, address, dof in objects:
            qpos[address : address + 7] = data.qpos[address : address + 7]
            qvel[dof : dof + 6] = 0.0
        key.set("qpos", _vector(qpos))
        if key.get("qvel") is not None:
            key.set("qvel", _vector(qvel))

    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="unicode")
    return {
        "status": "ok",
        "duration_seconds": step * timestep,
        "stable_seconds": stable_seconds,
        "max_linear_speed_meters_per_second": max_linear_speed,
        "max_angular_speed_radians_per_second": max_angular_speed,
        "objects": poses,
    }


def _vector(values: np.ndarray) -> str:
    return " ".join(format(float(value), ".17g") for value in values)
