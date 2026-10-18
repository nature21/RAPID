"""Scene loading and deterministic MuJoCo model inventory."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

import mujoco
import numpy as np


class InventorySummary(TypedDict):
    model: str
    nq: int
    nv: int
    nu: int
    nbody: int
    ngeom: int
    nsite: int
    njnt: int
    timestep: float


class ModelInventory(TypedDict):
    """JSON-safe typed inventory supplied to a binding agent."""

    summary: InventorySummary
    bodies: list[dict[str, Any]]
    geoms: list[dict[str, Any]]
    sites: list[dict[str, Any]]
    joints: list[dict[str, Any]]
    actuators: list[dict[str, Any]]


def _normalize_scene_module(value: str) -> str:
    module = value.strip()
    if not module:
        raise ValueError("scene module cannot be empty")
    if module.endswith(".scene"):
        return module
    candidate = f"{module}.scene"
    try:
        if importlib.util.find_spec(candidate) is not None:
            return candidate
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass
    return module


def load_scene_module(value: str) -> ModuleType:
    """Import a task scene package or explicit ``.scene`` module."""

    module_name = _normalize_scene_module(value)
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "make_model", None)):
        raise AttributeError(f"{module_name!r} does not expose make_model()")
    return module


@dataclass(frozen=True)
class SceneSource:
    """Importable deterministic source for a concrete MuJoCo scene."""

    module: str

    def load(self) -> ModuleType:
        return load_scene_module(self.module)

    def make_model(self) -> mujoco.MjModel:
        return self.load().make_model()

    def make_spec(self) -> mujoco.MjSpec:
        module = self.load()
        make_spec = getattr(module, "make_spec", None)
        if not callable(make_spec):
            raise AttributeError(f"{module.__name__!r} does not expose make_spec()")
        return make_spec()

    @property
    def default_camera(self) -> dict[str, Any]:
        return dict(getattr(self.load(), "DEFAULT_CAMERA", {}) or {})


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or f"<unnamed:{index}>"


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _actuator_transmission_targets(
    model: mujoco.MjModel,
    actuator_id: int,
) -> list[dict[str, Any]]:
    transmission = mujoco.mjtTrn(int(model.actuator_trntype[actuator_id]))
    ids = [int(item) for item in model.actuator_trnid[actuator_id]]
    if transmission in {
        mujoco.mjtTrn.mjTRN_JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT,
        mujoco.mjtTrn.mjTRN_SLIDERCRANK,
    }:
        kinds = [mujoco.mjtObj.mjOBJ_JOINT]
    elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
        kinds = [mujoco.mjtObj.mjOBJ_TENDON]
    elif transmission == mujoco.mjtTrn.mjTRN_SITE:
        kinds = [mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_SITE]
    elif transmission == mujoco.mjtTrn.mjTRN_BODY:
        kinds = [mujoco.mjtObj.mjOBJ_BODY]
    else:
        kinds = []
    targets: list[dict[str, Any]] = []
    for obj_type, target_id in zip(kinds, ids, strict=False):
        if target_id < 0:
            continue
        targets.append(
            {
                "object_type": _enum_name(obj_type),
                "id": target_id,
                "name": _name(model, obj_type, target_id),
            }
        )
    return targets


def inventory_model(model: mujoco.MjModel) -> ModelInventory:
    """Return a JSON-safe inventory of the names and structure of ``model``."""

    bodies: list[dict[str, Any]] = []
    for body_id in range(model.nbody):
        joint_start = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        joints = [
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(joint_start, joint_start + joint_count)
        ]
        parent_id = int(model.body_parentid[body_id])
        bodies.append(
            {
                "id": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "parent": (
                    None
                    if body_id == 0
                    else _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                ),
                "joints": joints,
                "pos": _array(model.body_pos[body_id]),
                "quat": _array(model.body_quat[body_id]),
            }
        )

    geoms: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        geoms.append(
            {
                "id": geom_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "type": _enum_name(mujoco.mjtGeom(int(model.geom_type[geom_id]))),
                "size": _array(model.geom_size[geom_id]),
                "pos": _array(model.geom_pos[geom_id]),
                "quat": _array(model.geom_quat[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )

    sites: list[dict[str, Any]] = []
    for site_id in range(model.nsite):
        body_id = int(model.site_bodyid[site_id])
        sites.append(
            {
                "id": site_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "type": _enum_name(mujoco.mjtGeom(int(model.site_type[site_id]))),
                "size": _array(model.site_size[site_id]),
                "pos": _array(model.site_pos[site_id]),
                "quat": _array(model.site_quat[site_id]),
            }
        )

    joints: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        joints.append(
            {
                "id": joint_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "type": _enum_name(mujoco.mjtJoint(int(model.jnt_type[joint_id]))),
                "axis": _array(model.jnt_axis[joint_id]),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
            }
        )

    actuators: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        actuators.append(
            {
                "id": actuator_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                "transmission_type": _enum_name(
                    mujoco.mjtTrn(int(model.actuator_trntype[actuator_id]))
                ),
                "transmission_ids": _array(model.actuator_trnid[actuator_id]),
                "transmission_targets": _actuator_transmission_targets(
                    model,
                    actuator_id,
                ),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "control_range": _array(model.actuator_ctrlrange[actuator_id]),
            }
        )

    return {
        "summary": {
            "model": model.names[: int(model.nnames)].split(b"\0", 1)[0].decode(
                errors="replace"
            ),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
            "njnt": int(model.njnt),
            "timestep": float(model.opt.timestep),
        },
        "bodies": bodies,
        "geoms": geoms,
        "sites": sites,
        "joints": joints,
        "actuators": actuators,
    }


def write_scene_inventory(
    scene_ref: str,
    output_path: str | Path,
) -> Path:
    """Compile ``scene_ref`` and write its deterministic inventory as JSON."""

    source = SceneSource(scene_ref)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory_model(source.make_model()), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "SceneSource",
    "InventorySummary",
    "ModelInventory",
    "inventory_model",
    "load_scene_module",
    "write_scene_inventory",
]
