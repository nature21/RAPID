"""MuJoCo task wrapper for sanitized real2sim scene bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PusherConfig:
    """Configuration for the Cartesian pusher added to real2sim scenes."""

    name: str = "pusher"
    radius: float = 0.018
    mass: float = 1.0
    clearance: float = 0.03
    initial_xyz: Vec3 | None = None
    ctrl_min: Vec3 | None = None
    ctrl_max: Vec3 | None = None
    side_preference: tuple[str, ...] = ("+x", "-x", "-y", "+y")
    xy_margin: float = 0.08
    z_clearance: float = 0.05
    min_z_range: float = 0.18
    kp_xy: float = 1000.0
    kv_xy: float = 100.0
    kp_z: float = 1000.0
    kv_z: float = 100.0
    damping_xy: float = 1.0
    damping_z: float = 1.0
    force_range: tuple[float, float] = (-200.0, 200.0)
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.25, 1.0)
    site_rgba: tuple[float, float, float, float] = (0.1, 0.4, 1.0, 1.0)
    friction: tuple[float, float, float] = (1.2, 0.02, 0.002)


@dataclass(frozen=True, slots=True)
class Real2SimTaskConfig:
    """Configuration for one packaged real2sim task scene."""

    scene_dir: str | Path
    task_name: str | None = None
    target_body: str | None = None
    pusher: PusherConfig = field(default_factory=PusherConfig)


@dataclass(frozen=True, slots=True)
class _ObjectInfo:
    body: str
    safe_name: str
    name: str
    has_freejoint: bool
    bounds: np.ndarray


@dataclass(frozen=True, slots=True)
class _PusherLayout:
    home_xyz: np.ndarray
    ctrl_min: np.ndarray
    ctrl_max: np.ndarray
    target: _ObjectInfo
    scene_bounds: np.ndarray


def make_xml(config: Real2SimTaskConfig) -> str:
    """Return an augmented MJCF XML string for ``config``."""

    scene_dir = _scene_dir(config)
    root = _load_scene_root(scene_dir)
    root.set("model", config.task_name or f"real2sim_{scene_dir.name}")
    _absolutize_asset_files(root, scene_dir)

    labels = _load_label_metadata(scene_dir)
    objects = _objects_from_scene_root(root, labels)
    target = _select_target_object(config, objects)
    scene_bounds = _scene_bounds(objects)
    layout = _build_pusher_layout(config.pusher, target, objects, scene_bounds)
    _add_pusher(root, config.pusher, layout)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def make_model(config: Real2SimTaskConfig) -> mujoco.MjModel:
    """Return the compiled MuJoCo model for ``config``."""

    return mujoco.MjModel.from_xml_string(make_xml(config))


def make_spec(config: Real2SimTaskConfig) -> mujoco.MjSpec:
    """Return the parsed MuJoCo spec for browser replay."""

    return mujoco.MjSpec.from_string(make_xml(config))


def make_default_camera(config: Real2SimTaskConfig) -> dict[str, Any]:
    """Return the captured RGB-D camera or a scene-framing fallback."""

    scene_dir = _scene_dir(config)
    root = _load_scene_root(scene_dir)
    _absolutize_asset_files(root, scene_dir)
    objects = _objects_from_scene_root(root, _load_label_metadata(scene_dir))
    metadata = _load_scene_metadata(scene_dir)
    calibrated = _calibrated_default_camera(metadata, objects)
    if calibrated is None and not metadata:
        calibrated = _bundled_default_camera(scene_dir, objects)
    if calibrated is not None:
        return calibrated
    if not objects:
        return {
            "camera_lookat": (0.0, 0.0, 0.05),
            "camera_distance": 0.8,
            "camera_azimuth": 270.0,
            "camera_elevation": -55.0,
        }

    bounds = _scene_bounds(objects)
    center = bounds.mean(axis=0)
    span = np.maximum(bounds[1] - bounds[0], 0.25)
    distance = float(max(0.65, 1.35 * np.linalg.norm(span[:2])))
    return {
        "camera_lookat": (
            float(center[0]),
            float(center[1]),
            float(max(0.04, center[2])),
        ),
        "camera_distance": distance,
        "camera_azimuth": 270.0,
        "camera_elevation": -55.0,
    }


def _calibrated_default_camera(
    metadata: dict[str, Any],
    objects: list[_ObjectInfo],
) -> dict[str, Any] | None:
    """Convert a captured OpenCV camera pose to MuJoCo free-camera controls."""

    depth_input = metadata.get("depth_input")
    camera = metadata.get("camera")
    if not isinstance(depth_input, dict) or not isinstance(camera, dict):
        return None
    if depth_input.get("provider") != "rgbd_scene_directory":
        return None
    if str(camera.get("world_frame_mode", "")).replace("-", "_") != (
        "camera_extrinsics"
    ):
        return None

    return _opencv_pose_default_camera(
        camera.get("T_world_from_camera_opencv"),
        objects,
    )


def _bundled_default_camera(
    scene_dir: Path,
    objects: list[_ObjectInfo],
) -> dict[str, Any] | None:
    """Recover the calibrated view when a failed run has no metadata.json."""

    camera_npz = scene_dir / "depth_camera.npz"
    if not camera_npz.is_file():
        return None
    try:
        with np.load(camera_npz, allow_pickle=False) as camera:
            raw_transform = camera["c2w_opencv"]
    except (KeyError, OSError, ValueError):
        return None
    return _opencv_pose_default_camera(raw_transform, objects)


def _opencv_pose_default_camera(
    raw_transform: Any,
    objects: list[_ObjectInfo],
) -> dict[str, Any] | None:
    """Convert one OpenCV camera-to-world pose to MuJoCo free-camera controls."""

    try:
        transform = np.asarray(raw_transform, dtype=float)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None

    position = transform[:3, 3]
    forward = transform[:3, 2]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-9:
        return None
    forward = forward / forward_norm

    distance = 1.0
    if objects:
        center = _scene_bounds(objects).mean(axis=0)
        projected_distance = float((center - position) @ forward)
        if np.isfinite(projected_distance) and projected_distance > 0.1:
            distance = projected_distance
    lookat = position + distance * forward
    azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    elevation = float(
        np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0)))
    )
    return {
        "camera_lookat": tuple(float(value) for value in lookat),
        "camera_distance": float(distance),
        "camera_azimuth": azimuth,
        "camera_elevation": elevation,
    }


def _scene_dir(config: Real2SimTaskConfig) -> Path:
    return Path(config.scene_dir).expanduser().resolve()


def _load_scene_root(scene_dir: Path) -> ET.Element:
    scene_xml = scene_dir / "scene.xml"
    if not scene_xml.is_file():
        raise FileNotFoundError(f"real2sim scene.xml not found: {scene_xml}")
    return ET.parse(scene_xml).getroot()


def _load_label_metadata(scene_dir: Path) -> dict[str, dict[str, str]]:
    metadata = _load_scene_metadata(scene_dir)

    labels: dict[str, dict[str, str]] = {}
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        return labels
    for item in objects:
        if not isinstance(item, dict):
            continue
        safe_name = str(item.get("safe_name", "")).strip()
        body = str(item.get("body", safe_name)).strip()
        name = str(item.get("name", safe_name or body)).strip()
        if not body:
            continue
        labels[body] = {
            "safe_name": safe_name or body,
            "name": name or safe_name or body,
        }
    return labels


def _load_scene_metadata(scene_dir: Path) -> dict[str, Any]:
    metadata_json = scene_dir / "metadata.json"
    if not metadata_json.is_file():
        return {}
    try:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _absolutize_asset_files(root: ET.Element, scene_dir: Path) -> None:
    for elem in root.findall(".//*[@file]"):
        value = str(elem.get("file", "")).strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            elem.set("file", str((scene_dir / path).resolve()))


def _objects_from_scene_root(
    root: ET.Element,
    labels: dict[str, dict[str, str]],
) -> list[_ObjectInfo]:
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    objects: list[_ObjectInfo] = []
    for body_id in range(1, model.nbody):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body:
            continue
        bounds = _body_mesh_bounds(model, data, body_id)
        if bounds is None:
            continue
        label = labels.get(body, {})
        objects.append(
            _ObjectInfo(
                body=body,
                safe_name=label.get("safe_name", body),
                name=label.get("name", body),
                has_freejoint=_body_has_freejoint(model, body_id),
                bounds=bounds,
            )
        )
    return objects


def _body_has_freejoint(model: mujoco.MjModel, body_id: int) -> bool:
    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    for jnt_id in range(jnt_adr, jnt_adr + jnt_num):
        if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return True
    return False


def _body_mesh_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray | None:
    all_points: list[np.ndarray] = []
    collision_points: list[np.ndarray] = []
    geom_ids = np.nonzero(model.geom_bodyid == body_id)[0]
    for geom_id in geom_ids:
        points = _geom_points(model, data, int(geom_id))
        if points is None or points.size == 0:
            continue
        all_points.append(points)
        if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
            collision_points.append(points)

    point_sets = collision_points or all_points
    if not point_sets:
        return None
    points = np.vstack(point_sets)
    if not np.isfinite(points).all():
        return None
    return np.stack([points.min(axis=0), points.max(axis=0)])


def _geom_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray | None:
    geom_type = int(model.geom_type[geom_id])
    mesh_id = int(model.geom_dataid[geom_id])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    origin = data.geom_xpos[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0:
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
        return (rotation @ vertices.T).T + origin

    extents = _geom_local_extents(model, geom_id)
    if extents is None:
        return None
    corners = np.array(
        [
            [sx * extents[0], sy * extents[1], sz * extents[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return (rotation @ corners.T).T + origin


def _geom_local_extents(model: mujoco.MjModel, geom_id: int) -> np.ndarray | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.array([size[0], size[0], size[0]], dtype=float)
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        return np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    return size[:3].astype(float)


def _select_target_object(
    config: Real2SimTaskConfig,
    objects: list[_ObjectInfo],
) -> _ObjectInfo:
    if not objects:
        raise ValueError("real2sim scene has no usable object geometry")

    if config.target_body is not None:
        target_body = str(config.target_body).strip()
        for obj in objects:
            if target_body in {obj.body, obj.safe_name, obj.name}:
                return obj
        raise ValueError(f"target_body {config.target_body!r} not found in scene")

    for obj in objects:
        if obj.has_freejoint:
            return obj
    return objects[0]


def _scene_bounds(objects: list[_ObjectInfo]) -> np.ndarray:
    if not objects:
        return np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.2]], dtype=float)
    bounds = np.asarray([obj.bounds for obj in objects], dtype=float)
    return np.stack([bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)])


def _build_pusher_layout(
    pusher: PusherConfig,
    target: _ObjectInfo,
    objects: list[_ObjectInfo],
    scene_bounds: np.ndarray,
) -> _PusherLayout:
    home_xyz = (
        np.asarray(pusher.initial_xyz, dtype=float)
        if pusher.initial_xyz is not None
        else _auto_pusher_home(pusher, target, objects)
    )
    if home_xyz.shape != (3,) or not np.isfinite(home_xyz).all():
        raise ValueError(f"invalid pusher initial_xyz: {pusher.initial_xyz!r}")

    ctrl_min, ctrl_max = _pusher_ctrl_range(pusher, home_xyz, scene_bounds)
    return _PusherLayout(
        home_xyz=home_xyz,
        ctrl_min=ctrl_min,
        ctrl_max=ctrl_max,
        target=target,
        scene_bounds=scene_bounds,
    )


def _auto_pusher_home(
    pusher: PusherConfig,
    target: _ObjectInfo,
    objects: list[_ObjectInfo],
) -> np.ndarray:
    target_bounds = target.bounds
    center = target_bounds.mean(axis=0)
    z = max(float(pusher.radius), float(target_bounds[0, 2] + pusher.radius))
    gap = float(pusher.radius + pusher.clearance)
    candidates = {
        "+x": np.array([target_bounds[1, 0] + gap, center[1], z], dtype=float),
        "-x": np.array([target_bounds[0, 0] - gap, center[1], z], dtype=float),
        "+y": np.array([center[0], target_bounds[1, 1] + gap, z], dtype=float),
        "-y": np.array([center[0], target_bounds[0, 1] - gap, z], dtype=float),
    }

    ordered = [side for side in pusher.side_preference if side in candidates]
    ordered.extend(side for side in candidates if side not in ordered)
    for side in ordered:
        xyz = candidates[side]
        if _pusher_clearance_score(xyz, pusher.radius, target, objects) > 0.0:
            return xyz
    return max(
        candidates.values(),
        key=lambda xyz: _pusher_clearance_score(xyz, pusher.radius, target, objects),
    )


def _pusher_clearance_score(
    xyz: np.ndarray,
    radius: float,
    target: _ObjectInfo,
    objects: list[_ObjectInfo],
) -> float:
    scores: list[float] = []
    for obj in objects:
        if obj.body == target.body:
            continue
        dx = max(obj.bounds[0, 0] - xyz[0], 0.0, xyz[0] - obj.bounds[1, 0])
        dy = max(obj.bounds[0, 1] - xyz[1], 0.0, xyz[1] - obj.bounds[1, 1])
        scores.append(float(np.hypot(dx, dy) - radius))
    return min(scores) if scores else 1.0


def _pusher_ctrl_range(
    pusher: PusherConfig,
    home_xyz: np.ndarray,
    scene_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if pusher.ctrl_min is not None or pusher.ctrl_max is not None:
        if pusher.ctrl_min is None or pusher.ctrl_max is None:
            raise ValueError("pusher ctrl_min and ctrl_max must be provided together")
        ctrl_min = np.asarray(pusher.ctrl_min, dtype=float)
        ctrl_max = np.asarray(pusher.ctrl_max, dtype=float)
    else:
        margin = float(pusher.xy_margin)
        ctrl_min = np.array(
            [
                scene_bounds[0, 0] - margin - home_xyz[0],
                scene_bounds[0, 1] - margin - home_xyz[1],
                0.0,
            ],
            dtype=float,
        )
        ctrl_max = np.array(
            [
                scene_bounds[1, 0] + margin - home_xyz[0],
                scene_bounds[1, 1] + margin - home_xyz[1],
                max(
                    float(pusher.min_z_range),
                    scene_bounds[1, 2] + float(pusher.z_clearance) - home_xyz[2],
                ),
            ],
            dtype=float,
        )
    if ctrl_min.shape != (3,) or ctrl_max.shape != (3,):
        raise ValueError("pusher ctrl_min and ctrl_max must be length-3 vectors")
    if not np.isfinite(ctrl_min).all() or not np.isfinite(ctrl_max).all():
        raise ValueError("pusher control ranges must be finite")
    if np.any(ctrl_min >= ctrl_max):
        raise ValueError(f"invalid pusher control range: {ctrl_min} >= {ctrl_max}")
    return ctrl_min, ctrl_max


def _add_pusher(
    root: ET.Element,
    pusher: PusherConfig,
    layout: _PusherLayout,
) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    _ensure_name_available(root, pusher.name)
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": pusher.name, "pos": _fmt_vec(layout.home_xyz)},
    )
    _add_slide_joint(
        body,
        name=f"{pusher.name}_x",
        axis="1 0 0",
        range_values=(layout.ctrl_min[0], layout.ctrl_max[0]),
        damping=pusher.damping_xy,
    )
    _add_slide_joint(
        body,
        name=f"{pusher.name}_y",
        axis="0 1 0",
        range_values=(layout.ctrl_min[1], layout.ctrl_max[1]),
        damping=pusher.damping_xy,
    )
    _add_slide_joint(
        body,
        name=f"{pusher.name}_z",
        axis="0 0 1",
        range_values=(layout.ctrl_min[2], layout.ctrl_max[2]),
        damping=pusher.damping_z,
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": pusher.name,
            "type": "sphere",
            "size": _fmt_float(pusher.radius),
            "mass": _fmt_float(pusher.mass),
            "rgba": _fmt_vec(pusher.rgba),
            "friction": _fmt_vec(pusher.friction),
            "condim": "4",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"{pusher.name}_site",
            "pos": "0 0 0",
            "size": "0.004",
            "rgba": _fmt_vec(pusher.site_rgba),
        },
    )

    actuator = _get_or_create_section(root, "actuator")
    force_range = _fmt_vec(pusher.force_range)
    _add_position_actuator(
        actuator,
        name=f"act_{pusher.name}_x",
        joint=f"{pusher.name}_x",
        kp=pusher.kp_xy,
        kv=pusher.kv_xy,
        ctrl_range=(layout.ctrl_min[0], layout.ctrl_max[0]),
        force_range=force_range,
    )
    _add_position_actuator(
        actuator,
        name=f"act_{pusher.name}_y",
        joint=f"{pusher.name}_y",
        kp=pusher.kp_xy,
        kv=pusher.kv_xy,
        ctrl_range=(layout.ctrl_min[1], layout.ctrl_max[1]),
        force_range=force_range,
    )
    _add_position_actuator(
        actuator,
        name=f"act_{pusher.name}_z",
        joint=f"{pusher.name}_z",
        kp=pusher.kp_z,
        kv=pusher.kv_z,
        ctrl_range=(layout.ctrl_min[2], layout.ctrl_max[2]),
        force_range=force_range,
    )

    sensor = _get_or_create_section(root, "sensor")
    ET.SubElement(
        sensor,
        "framepos",
        {
            "name": f"trace_{pusher.name}",
            "objtype": "site",
            "objname": f"{pusher.name}_site",
        },
    )


def _ensure_name_available(root: ET.Element, name: str) -> None:
    for elem in root.iter():
        if elem.get("name") == name:
            raise ValueError(f"MJCF already contains an element named {name!r}")


def _add_slide_joint(
    body: ET.Element,
    *,
    name: str,
    axis: str,
    range_values: tuple[float, float],
    damping: float,
) -> None:
    ET.SubElement(
        body,
        "joint",
        {
            "name": name,
            "type": "slide",
            "axis": axis,
            "limited": "true",
            "range": _fmt_vec(range_values),
            "damping": _fmt_float(damping),
        },
    )


def _add_position_actuator(
    actuator: ET.Element,
    *,
    name: str,
    joint: str,
    kp: float,
    kv: float,
    ctrl_range: tuple[float, float],
    force_range: str,
) -> None:
    ET.SubElement(
        actuator,
        "position",
        {
            "name": name,
            "joint": joint,
            "kp": _fmt_float(kp),
            "kv": _fmt_float(kv),
            "ctrlrange": _fmt_vec(ctrl_range),
            "ctrllimited": "true",
            "forcerange": force_range,
        },
    )


def _get_or_create_section(root: ET.Element, tag: str) -> ET.Element:
    existing = root.find(tag)
    if existing is not None:
        return existing

    section = ET.Element(tag)
    if tag == "actuator":
        sensor = root.find("sensor")
        if sensor is not None:
            root.insert(list(root).index(sensor), section)
        else:
            root.append(section)
    elif tag == "sensor":
        root.append(section)
    else:
        root.append(section)
    return section


def _fmt_vec(values: Any) -> str:
    return " ".join(_fmt_float(value) for value in np.asarray(values).reshape(-1))


def _fmt_float(value: Any) -> str:
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.8g}"
