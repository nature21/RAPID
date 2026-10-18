"""Shared MuJoCo state-layout introspection for scene instantiations."""

from __future__ import annotations

import mujoco
import numpy as np


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def body_joint_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Return the joint ids attached to ``body_name``."""

    body_id = model.body(body_name).id
    start = int(model.body_jntadr[body_id])
    num = int(model.body_jntnum[body_id])
    return list(range(start, start + num))


def _axis_idx(axis_char: str) -> int:
    try:
        return _AXIS_INDEX[axis_char]
    except KeyError as exc:
        raise ValueError(
            f"axis_char must be one of {sorted(_AXIS_INDEX)}; got {axis_char!r}."
        ) from exc


def find_slide(model: mujoco.MjModel, body_name: str, axis_char: str) -> int:
    """Return the qpos address of a body's slide joint along ``axis_char``."""

    qpos, _ = find_slide_dof(model, body_name, axis_char)
    return qpos


def find_slide_dof(
    model: mujoco.MjModel,
    body_name: str,
    axis_char: str,
) -> tuple[int, int]:
    """Return qpos/dof addresses for a body's slide joint along an axis."""

    axis = _axis_idx(axis_char)
    for joint_id in body_joint_ids(model, body_name):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        if abs(np.asarray(model.jnt_axis[joint_id])[axis]) > 0.5:
            return (
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
    raise ValueError(f"Body {body_name!r} has no slide joint along {axis_char}.")


def find_hinge(model: mujoco.MjModel, body_name: str, axis_char: str) -> int:
    """Return the qpos address of a body's hinge joint about ``axis_char``."""

    axis = _axis_idx(axis_char)
    for joint_id in body_joint_ids(model, body_name):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if abs(np.asarray(model.jnt_axis[joint_id])[axis]) > 0.5:
            return int(model.jnt_qposadr[joint_id])
    raise ValueError(f"Body {body_name!r} has no hinge joint about {axis_char}.")


def find_freejoint(model: mujoco.MjModel, body_name: str) -> int:
    """Return the id of a body's unique free joint."""

    free = [
        joint_id
        for joint_id in body_joint_ids(model, body_name)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free) != 1:
        raise ValueError(
            f"Body {body_name!r} must have exactly one freejoint; got {len(free)}."
        )
    return free[0]


def freejoint_qpos_dof(
    model: mujoco.MjModel,
    body_name: str,
) -> tuple[int, int]:
    """Return qpos/dof addresses for a body's unique free joint."""

    joint_id = find_freejoint(model, body_name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def build_initial_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Return fresh forwarded MuJoCo data at the model reset state."""

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return data


def build_initial_world_state(model: mujoco.MjModel) -> np.ndarray:
    """Return concatenated reset-state qpos and qvel."""

    data = build_initial_data(model)
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


__all__ = [
    "body_joint_ids",
    "build_initial_data",
    "build_initial_world_state",
    "find_freejoint",
    "find_hinge",
    "find_slide",
    "find_slide_dof",
    "freejoint_qpos_dof",
]
