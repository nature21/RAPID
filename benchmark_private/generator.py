"""Materialize portable benchmark cases from the eight reference scenes."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .assets import (
    asset_alignment_quaternion,
    asset_record,
    hope_upright_quaternion,
    load_asset_manifest,
    matrix_quaternion,
    realistic_half_extents,
)
from .materials import (
    SimulationMaterial,
    appearance_material,
    effective_box_mass,
    material_texture_rgb,
)
from .physics import elasticity_solref
from .primitives import (
    is_primitive_object,
    primitive_half_extents,
    primitive_inertia,
    primitive_mass,
    primitive_preset,
)
from .strategies import generator_for
from .specs import (
    BENCHMARK_VERSION,
    DISPLAY_NAMES,
    STRATEGY_2_GRIPPER_MAX_WIDTH_M,
    STRATEGY_CONTRACTS,
    StrategyContract,
    VariantParameters,
    safe_appearance_name,
    strategy_2_manipulation_metrics,
)
from .strategy_common import ContextResult, SceneEditor


FR3_HOME_JOINT_QPOS: dict[str, float] = {
    "fr3_joint1": -0.0030637685314503847,
    "fr3_joint2": -0.7778934457717781,
    "fr3_joint3": 0.01585533499243391,
    "fr3_joint4": -2.3693353790401894,
    "fr3_joint5": -0.004060005459262234,
    "fr3_joint6": 1.554097367289531,
    "fr3_joint7": 0.7704814906387026,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
}


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _vec(raw: str | None, default: Iterable[float]) -> np.ndarray:
    if raw is None:
        return np.asarray(tuple(default), dtype=np.float64)
    return np.asarray([float(item) for item in raw.split()], dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    value = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _canonical_upright_box_rotation(
    half_extents: np.ndarray,
    yaw_radians: float,
    *,
    lateral_rank: int,
    vertical_rank: int,
) -> np.ndarray:
    """Map ranked box axes into an upright semantic yaw frame."""

    order = np.argsort(np.asarray(half_extents), kind="stable")
    vertical_index = int(order[vertical_rank])
    lateral_index = int(order[lateral_rank])
    axis_index = next(
        index
        for index in range(3)
        if index not in {vertical_index, lateral_index}
    )
    lateral = np.asarray(
        (math.cos(yaw_radians), math.sin(yaw_radians), 0.0),
        dtype=np.float64,
    )
    axis = np.asarray(
        (-math.sin(yaw_radians), math.cos(yaw_radians), 0.0),
        dtype=np.float64,
    )
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[:, lateral_index] = lateral
    rotation[:, axis_index] = axis
    rotation[:, vertical_index] = (0.0, 0.0, 1.0)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, axis_index] *= -1.0
    return rotation


def _box_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
    hx, hy, hz = half_extents
    return np.asarray(
        (
            mass * (hy * hy + hz * hz) / 3.0,
            mass * (hx * hx + hz * hz) / 3.0,
            mass * (hx * hx + hy * hy) / 3.0,
        )
    )


def _object_material(appearance: str) -> SimulationMaterial:
    if is_primitive_object(appearance):
        return primitive_preset(appearance).material
    return appearance_material(appearance)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise KeyError(f"reference scene is missing body {name!r}")
    return body


def _floor_height(root: ET.Element) -> float:
    floor = root.find(".//geom[@name='floor']")
    if floor is None:
        raise KeyError("reference scene is missing floor geom")
    return float(_vec(floor.get("pos"), (0.0, 0.0, 0.0))[2])


def _normalize_floor_height(root: ET.Element) -> float:
    """Translate the task world so the work surface is exactly world z=0."""

    floor_z = _floor_height(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("reference scene is missing worldbody")
    for element in list(worldbody):
        if element.get("pos") is None:
            continue
        position = _vec(element.get("pos"), (0.0, 0.0, 0.0))
        position[2] -= floor_z
        if element.get("name") == "floor":
            position[2] = 0.0
        element.set("pos", _fmt(position))
    if abs(_floor_height(root)) > 1e-12:
        raise AssertionError("floor-height normalization failed")
    return floor_z


def _rest_body_on_plane(
    body: ET.Element,
    collision: ET.Element,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> None:
    body_pos = _vec(body.get("pos"), (0.0, 0.0, 0.0))
    body_quat = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    geom_pos = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
    geom_quat = _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
    size = _vec(collision.get("size"), (0.01, 0.01, 0.01))
    body_rotation = _quat_matrix(body_quat)
    collision_rotation = body_rotation @ _quat_matrix(geom_quat)
    normal = plane_normal / np.linalg.norm(plane_normal)
    if normal[2] < 0.0:
        normal = -normal
    support_radius = float(np.abs(normal @ collision_rotation) @ size)
    rotated_local_center = body_rotation @ geom_pos
    # Solve n dot (body_pos + local_center - n * radius - plane_point) = 0
    # for the world-z component of body_pos.
    known = np.asarray((body_pos[0], body_pos[1], 0.0)) + rotated_local_center
    body_pos[2] = (
        float(normal @ plane_point)
        + support_radius
        - float(normal @ known)
    ) / float(normal[2]) + 1e-4
    body.set("pos", _fmt(body_pos))


def _static_box_support_plane(
    root: ET.Element,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    body = _find_body(root, body_name)
    collision = next(
        (geom for geom in body.findall("geom") if geom.get("type") == "box" and geom.get("group") == "3"),
        None,
    )
    if collision is None:
        raise ValueError(f"support body {body_name!r} has no primitive collision box")
    body_pos = _vec(body.get("pos"), (0.0, 0.0, 0.0))
    body_quat = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    geom_pos = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
    geom_quat = _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
    size = _vec(collision.get("size"), (0.01, 0.01, 0.01))
    body_rotation = _quat_matrix(body_quat)
    collision_rotation = body_rotation @ _quat_matrix(geom_quat)
    center = body_pos + body_rotation @ geom_pos
    vertical_axis = int(np.argmax(np.abs(collision_rotation[2])))
    normal = collision_rotation[:, vertical_axis].copy()
    if normal[2] < 0.0:
        normal = -normal
    point = center + normal * size[vertical_axis]
    return point, normal


def _remove_target_visuals(root: ET.Element, body: ET.Element) -> None:
    removed_meshes: set[str] = set()
    removed_materials: set[str] = set()
    for geom in list(body.findall("geom")):
        if geom.get("type") != "mesh":
            continue
        if geom.get("mesh"):
            removed_meshes.add(str(geom.get("mesh")))
        if geom.get("material"):
            removed_materials.add(str(geom.get("material")))
        body.remove(geom)

    used_meshes = {
        str(geom.get("mesh"))
        for geom in root.findall(".//geom[@mesh]")
        if geom.get("mesh")
    }
    used_materials = {
        str(geom.get("material"))
        for geom in root.findall(".//geom[@material]")
        if geom.get("material")
    }
    asset = root.find("asset")
    if asset is None:
        raise ValueError("reference scene is missing asset section")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") in removed_meshes - used_meshes:
            asset.remove(mesh)
    for material in list(asset.findall("material")):
        if material.get("name") in removed_materials - used_materials:
            asset.remove(material)


def _copy_hope_asset(
    asset_root: Path,
    case_bundle: Path,
    appearance: str,
) -> tuple[str, str, dict[str, Any]]:
    record = asset_record(asset_root, appearance)
    safe = str(record["safe_name"])
    source_dir = asset_root / safe
    mesh_target = case_bundle / "meshes" / f"hope_{safe}.obj"
    texture_target = case_bundle / "textures" / f"hope_{safe}.png"
    mesh_target.parent.mkdir(parents=True, exist_ok=True)
    texture_target.parent.mkdir(parents=True, exist_ok=True)
    if not mesh_target.exists():
        shutil.copy2(source_dir / "visual.obj", mesh_target)
    if not texture_target.exists():
        shutil.copy2(source_dir / "texture.png", texture_target)
    return (
        f"meshes/{mesh_target.name}",
        f"textures/{texture_target.name}",
        record,
    )


def _add_hope_visual(
    root: ET.Element,
    body: ET.Element,
    *,
    role_name: str,
    appearance: str,
    half_extents: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
    asset_root: Path,
    case_bundle: Path,
) -> None:
    asset = root.find("asset")
    if asset is None:
        raise ValueError("scene is missing asset section")
    mesh_file, texture_file, _ = _copy_hope_asset(asset_root, case_bundle, appearance)
    physical_material = appearance_material(appearance)
    safe_asset = safe_appearance_name(appearance)
    texture_name = f"{role_name}_{safe_asset}_texture"
    material_name = f"{role_name}_{safe_asset}_material"
    mesh_name = f"{role_name}_{safe_asset}_mesh"
    ET.SubElement(
        asset,
        "texture",
        name=texture_name,
        type="2d",
        file=texture_file,
    )
    ET.SubElement(
        asset,
        "material",
        name=material_name,
        texture=texture_name,
        texuniform="false",
        specular=f"{physical_material.specular:.6g}",
        shininess=f"{physical_material.shininess:.6g}",
        reflectance=f"{physical_material.reflectance:.6g}",
    )
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        file=mesh_file,
        scale=_fmt(half_extents),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{role_name}_visual",
        type="mesh",
        mesh=mesh_name,
        material=material_name,
        pos=_fmt(local_pos),
        quat=_fmt(local_quat),
        contype="0",
        conaffinity="0",
        density="0",
        group="2",
    )


def _add_primitive_visual(
    root: ET.Element,
    body: ET.Element,
    *,
    role_name: str,
    appearance: str,
    half_extents: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
    case_bundle: Path,
) -> None:
    preset = primitive_preset(appearance)
    material = preset.material
    asset = root.find("asset")
    if asset is None:
        raise ValueError("scene is missing asset section")
    texture_name = f"{role_name}_{material.key}_texture"
    material_name = f"{role_name}_{material.key}_material"
    texture_target = (
        case_bundle / "textures" / f"primitive_{role_name}_{material.key}.png"
    )
    texture_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        material_texture_rgb(
            material,
            seed=sum(role_name.encode("utf-8")) + 65537,
        )
    ).save(texture_target)
    ET.SubElement(
        asset,
        "texture",
        name=texture_name,
        type="2d",
        file=f"textures/{texture_target.name}",
    )
    ET.SubElement(
        asset,
        "material",
        name=material_name,
        texture=texture_name,
        texuniform="true",
        texrepeat="2 2",
        rgba=_fmt((1.0, 1.0, 1.0, material.opacity)),
        specular=f"{material.specular:.6g}",
        shininess=f"{material.shininess:.6g}",
        reflectance=f"{material.reflectance:.6g}",
    )
    visual_size = (
        _fmt(half_extents)
        if preset.shape == "box"
        else _fmt((half_extents[0], half_extents[2]))
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{role_name}_visual",
        type=preset.shape,
        size=visual_size,
        material=material_name,
        pos=_fmt(local_pos),
        quat=_fmt(local_quat),
        contype="0",
        conaffinity="0",
        density="0",
        group="2",
    )


def _rename_target(
    root: ET.Element,
    contract: StrategyContract,
    parameters: VariantParameters,
    *,
    support_body: str | None,
    target_center_xy: tuple[float, float] | None,
    target_yaw_radians: float | None,
    target_canonical_upright: bool,
    target_lateral_rank: int,
    target_vertical_rank: int,
    target_half_extents_override: tuple[float, float, float] | None,
    asset_root: Path,
    case_bundle: Path,
) -> tuple[str, str, np.ndarray]:
    body = _find_body(root, contract.target_body)
    collision = next(
        (geom for geom in body.findall("geom") if geom.get("type") == "box" and geom.get("group") == "3"),
        None,
    )
    if collision is None:
        raise ValueError(f"target body {contract.target_body!r} has no primitive collision box")
    inertial = body.find("inertial")
    freejoint = body.find("freejoint")
    if inertial is None or freejoint is None:
        raise ValueError(f"target body {contract.target_body!r} must be dynamic")

    base_size = _vec(collision.get("size"), (0.01, 0.01, 0.01))
    primitive = is_primitive_object(parameters.appearance)
    if primitive:
        preset = primitive_preset(parameters.appearance)
        if not preset.target_eligible or preset.shape != "box":
            raise ValueError(
                "benchmark targets require a target-eligible box preset: "
                f"{parameters.appearance}"
            )
        half_extents = primitive_half_extents(
            preset,
            parameters.size_scale_xyz,
            broad=True,
        )
    else:
        appearance_record = asset_record(asset_root, parameters.appearance)
        half_extents = realistic_half_extents(
            appearance_record,
            parameters.size_scale_xyz,
        )
    if target_half_extents_override is not None:
        half_extents = np.asarray(
            target_half_extents_override,
            dtype=np.float64,
        )
        if half_extents.shape != (3,) or np.any(half_extents <= 0.0):
            raise ValueError(
                "target_half_extents_override must contain three positive values"
            )
    if target_canonical_upright:
        collision_pos = np.zeros(3, dtype=np.float64)
        collision_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    else:
        collision_pos = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
        reference_quat = _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
        collision_quat = _quat_multiply(
            reference_quat,
            asset_alignment_quaternion(half_extents, base_size),
        )
    new_safe_name = safe_appearance_name(parameters.appearance)
    physical_material = _object_material(parameters.appearance)
    if not new_safe_name.endswith(("box", "carton")):
        display = DISPLAY_NAMES[parameters.appearance].lower()
        if "box" in display:
            new_safe_name += "_box"
        elif "carton" in display:
            new_safe_name += "_carton"

    _remove_target_visuals(root, body)
    old_name = str(body.get("name"))
    body.set("name", new_safe_name)
    freejoint.set("name", f"{new_safe_name}_joint")
    collision.set("name", f"{new_safe_name}_collision")
    collision.set("pos", _fmt(collision_pos))
    collision.set("size", _fmt(half_extents))
    collision.set("quat", _fmt(collision_quat))
    friction = np.asarray(
        (
            physical_material.sliding_friction * parameters.friction_scale,
            physical_material.torsional_friction,
            physical_material.rolling_friction,
        )
    )
    collision.set("friction", _fmt(friction))
    solref = _vec(collision.get("solref"), (0.01, 1.0))
    solref[:] = elasticity_solref(float(solref[0]), parameters.restitution)
    collision.set("solref", _fmt(solref))

    mass = (
        primitive_mass(
            preset,
            half_extents,
            mass_scale=parameters.mass_scale,
        )
        if primitive
        else effective_box_mass(
            physical_material,
            half_extents,
            scale=parameters.mass_scale,
        )
    )
    inertial.set("mass", f"{mass:.9g}")
    inertial.set("pos", _fmt(collision_pos))
    inertial.set("quat", _fmt(collision_quat))
    inertial.set(
        "diaginertia",
        _fmt(
            primitive_inertia(preset, half_extents, mass)
            if primitive
            else _box_inertia(mass, half_extents)
        ),
    )
    site = body.find("site")
    if site is not None:
        site.set("name", f"{new_safe_name}_site")

    plane_point, plane_normal = (
        _static_box_support_plane(root, support_body)
        if support_body is not None
        else (np.asarray((0.0, 0.0, _floor_height(root))), np.asarray((0.0, 0.0, 1.0)))
    )
    yaw_value = (
        float(target_yaw_radians)
        if target_yaw_radians is not None
        else parameters.yaw_radians
    )
    if target_canonical_upright:
        if target_center_xy is None:
            raise ValueError("canonical target placement requires an explicit center")
        if not np.allclose(
            plane_normal / np.linalg.norm(plane_normal),
            np.asarray((0.0, 0.0, 1.0)),
            atol=1e-8,
        ):
            raise ValueError("canonical target placement requires a horizontal support")
        final_body_quat = matrix_quaternion(
            _canonical_upright_box_rotation(
                half_extents,
                yaw_value,
                lateral_rank=target_lateral_rank,
                vertical_rank=target_vertical_rank,
            )
        )
    else:
        yaw_axis = plane_normal / np.linalg.norm(plane_normal)
        body_quat = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        yaw_quat = np.asarray(
            (
                math.cos(yaw_value / 2),
                *(yaw_axis * math.sin(yaw_value / 2)),
            )
        )
        final_body_quat = _quat_multiply(yaw_quat, body_quat)
    body.set("quat", _fmt(final_body_quat))
    body_pos = _vec(body.get("pos"), (0.0, 0.0, 0.0))
    if target_center_xy is None:
        body_pos[:2] += np.asarray(parameters.translation_xy)
    else:
        rotated_collision_pos = _quat_matrix(final_body_quat) @ collision_pos
        body_pos[:2] = (
            np.asarray(target_center_xy, dtype=np.float64)
            - rotated_collision_pos[:2]
        )
    body.set("pos", _fmt(body_pos))
    _rest_body_on_plane(body, collision, plane_point, plane_normal)

    if primitive:
        _add_primitive_visual(
            root,
            body,
            role_name=new_safe_name,
            appearance=parameters.appearance,
            half_extents=half_extents,
            local_pos=collision_pos,
            local_quat=collision_quat,
            case_bundle=case_bundle,
        )
    else:
        _add_hope_visual(
            root,
            body,
            role_name=new_safe_name,
            appearance=parameters.appearance,
            half_extents=half_extents,
            local_pos=collision_pos,
            local_quat=collision_quat,
            asset_root=asset_root,
            case_bundle=case_bundle,
        )
    return old_name, new_safe_name, half_extents


def _collision_footprints_xy(root: ET.Element) -> list[tuple[np.ndarray, np.ndarray]]:
    """Conservative AABBs for existing task collision primitives."""

    footprints: list[tuple[np.ndarray, np.ndarray]] = []
    for body in root.findall(".//worldbody/body"):
        body_position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
        body_rotation = _quat_matrix(
            _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        )
        for geom in body.findall("geom"):
            if geom.get("group") != "3":
                continue
            geom_position = _vec(geom.get("pos"), (0.0, 0.0, 0.0))
            center = body_position + body_rotation @ geom_position
            geom_type = geom.get("type", "sphere")
            raw_size = _vec(geom.get("size"), (0.01,))
            if geom_type == "box" and raw_size.shape == (3,):
                geom_rotation = body_rotation @ _quat_matrix(
                    _vec(geom.get("quat"), (1.0, 0.0, 0.0, 0.0))
                )
                half = np.abs(geom_rotation[:2]) @ raw_size
            elif geom_type in {"cylinder", "capsule"} and raw_size.size >= 1:
                half = np.asarray((raw_size[0], raw_size[0]), dtype=np.float64)
            elif geom_type == "sphere" and raw_size.size >= 1:
                half = np.asarray((raw_size[0], raw_size[0]), dtype=np.float64)
            else:
                continue
            footprints.append((center[:2], half))
    return footprints


def _body_collision_footprint_xy(
    root: ET.Element,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the planar AABB of one top-level task collision body."""

    body = _find_body(root, body_name)
    collision = next(
        (
            geom
            for geom in body.findall("geom")
            if geom.get("group") == "3" and geom.get("type") == "box"
        ),
        None,
    )
    if collision is None:
        raise ValueError(f"body {body_name!r} has no box collision footprint")
    body_position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
    body_rotation = _quat_matrix(
        _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    )
    collision_position = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
    collision_rotation = body_rotation @ _quat_matrix(
        _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
    )
    center = body_position + body_rotation @ collision_position
    half = np.abs(collision_rotation[:2]) @ _vec(
        collision.get("size"),
        (0.01, 0.01, 0.01),
    )
    return center[:2], half


def _select_distractor_xy(
    root: ET.Element,
    *,
    desired_xy: tuple[float, float],
    footprint_half: np.ndarray,
    protected_regions: tuple[tuple[np.ndarray, np.ndarray], ...],
    manipulation_clearance: float = 0.09,
) -> tuple[float, float]:
    """Choose the nearest visible, collision-free distractor staging point."""

    occupied = _collision_footprints_xy(root)
    grid = [
        (float(x), float(y))
        for y in (
            -0.60,
            -0.54,
            -0.48,
            -0.44,
            -0.40,
            -0.32,
            -0.24,
            0.58,
            0.52,
            0.46,
            0.40,
            0.32,
            0.24,
        )
        for x in np.linspace(0.16, 1.10, 17)
    ]
    candidates = [desired_xy, *sorted(
        grid,
        key=lambda value: (
            (value[0] - desired_xy[0]) ** 2 + (value[1] - desired_xy[1]) ** 2,
            value[1],
            value[0],
        ),
    )]
    lower = np.asarray((0.10, -0.68), dtype=np.float64)
    upper = np.asarray((1.16, 0.64), dtype=np.float64)
    clearance = 0.018
    for candidate in candidates:
        center = np.asarray(candidate, dtype=np.float64)
        if np.any(center - footprint_half < lower) or np.any(
            center + footprint_half > upper
        ):
            continue
        if any(
            bool(
                np.all(
                    np.abs(center - other_center)
                    < footprint_half + other_half + clearance
                )
            )
            for other_center, other_half in occupied
        ):
            continue
        if any(
            bool(
                np.all(
                    np.abs(center - protected_center)
                    < footprint_half
                    + protected_half
                    + manipulation_clearance
                )
            )
            for protected_center, protected_half in protected_regions
        ):
            continue
        return float(center[0]), float(center[1])
    raise ValueError(
        f"no collision-free distractor slot near {desired_xy} for half footprint "
        f"{footprint_half.tolist()}"
    )


def _add_distractors(
    root: ET.Element,
    contract: StrategyContract,
    parameters: VariantParameters,
    *,
    target_body: str,
    context_layout: dict[str, Any],
    asset_root: Path,
    case_bundle: Path,
) -> list[dict[str, str]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene is missing worldbody")
    floor_z = _floor_height(root)
    slots = contract.distractor_slots_xy
    protected_center, protected_half = _body_collision_footprint_xy(
        root,
        target_body,
    )
    protected_regions: list[tuple[np.ndarray, np.ndarray]] = [
        (protected_center, protected_half)
    ]
    manipulation_waypoints = context_layout.get(
        "manipulation_waypoints_xy",
        (),
    )
    if isinstance(manipulation_waypoints, (list, tuple)):
        parsed_waypoints = [
            np.asarray(value, dtype=np.float64)
            for value in manipulation_waypoints
            if np.asarray(value).shape == (2,)
        ]
        for start, end in zip(parsed_waypoints, parsed_waypoints[1:]):
            protected_regions.append(
                (
                    0.5 * (start + end),
                    protected_half + 0.5 * np.abs(end - start),
                )
            )
    for raw_region in context_layout.get(
        "manipulation_exclusion_regions",
        (),
    ):
        if not isinstance(raw_region, dict):
            continue
        center = np.asarray(raw_region.get("center_xy", ()), dtype=np.float64)
        half = np.asarray(
            raw_region.get("half_extents_xy", ()),
            dtype=np.float64,
        )
        if center.shape == (2,) and half.shape == (2,) and np.all(half > 0.0):
            protected_regions.append((center, half))
    manipulation_clearance = float(
        context_layout.get("distractor_manipulation_clearance", 0.09)
    )
    if (
        not math.isfinite(manipulation_clearance)
        or manipulation_clearance < 0.0
    ):
        raise ValueError(
            "distractor_manipulation_clearance must be a finite nonnegative value"
        )
    if contract.strategy_id == 2:
        wall_flip = context_layout.get("wall_flip", {})
        wall_center = np.asarray(
            wall_flip.get("wall_center_xy", protected_center),
            dtype=np.float64,
        )
        if wall_center.shape == (2,):
            # Protect the complete target-to-wall sweep, not only the initial
            # target footprint.  Distractors are static obstacles, so placing
            # one in this region would make the demonstrated push/flip path
            # invalid even though there is no reset-time overlap.
            protected_regions.append(
                (
                    0.5 * (protected_center + wall_center),
                    0.5 * np.abs(wall_center - protected_center)
                    + protected_half,
                )
            )
        wall_normal = np.asarray(
            wall_flip.get("wall_normal_xy", (1.0, 0.0)),
            dtype=np.float64,
        )
        wall_axis = np.asarray(
            wall_flip.get("wall_axis_xy", (0.0, 1.0)),
            dtype=np.float64,
        )
        if wall_normal.shape == (2,) and wall_axis.shape == (2,):
            # Keep visual context close to the task when the complete layout
            # rotates through +/-90 degrees. Fixed world-frame slots can fall
            # outside the deployment camera or cluster on the sweep path.
            relative_offsets = (
                (-0.30, 0.05),
                (0.30, 0.05),
                (-0.30, -0.12),
            )
            slots = tuple(
                tuple(
                    float(value)
                    for value in (
                        protected_center
                        + axis_offset * wall_axis
                        + normal_offset * wall_normal
                    )
                )
                for axis_offset, normal_offset in relative_offsets
            )
    context_layout["distractor_exclusion_regions"] = [
        {
            "center_xy": center.tolist(),
            "half_extents_xy": half.tolist(),
        }
        for center, half in protected_regions
    ]
    context_layout["distractor_motion"] = "static_noncausal"
    context_layout["distractor_slots_xy"] = [list(slot) for slot in slots]
    records: list[dict[str, str]] = []
    for index, appearance in enumerate(parameters.distractor_appearances[:3]):
        base_safe = safe_appearance_name(appearance)
        safe_name = f"{base_safe}_{index + 1}"
        rng = np.random.default_rng(parameters.seed + 1009 * (index + 1))
        primitive = is_primitive_object(appearance)
        local_quat = np.asarray((1.0, 0.0, 0.0, 0.0))
        if primitive:
            preset = primitive_preset(appearance)
            jitter = tuple(float(value) for value in rng.uniform(0.78, 1.22, 3))
            size = primitive_half_extents(preset, jitter, broad=True)
            collision_type = preset.shape
            visual_source = "material_primitive"
        else:
            asset = asset_record(asset_root, appearance)
            uniform_scale = float(rng.uniform(0.75, 1.25))
            size = realistic_half_extents(
                asset,
                (uniform_scale, uniform_scale, uniform_scale),
            )
            collision_type = "box"
            visual_source = "hope_asset"
            local_quat = hope_upright_quaternion()
        desired_xy = slots[index]
        base_yaw = float(rng.uniform(-math.pi, math.pi))
        yaw_candidates = (
            (base_yaw,)
            if collision_type != "box"
            else (
                base_yaw,
                base_yaw + math.pi / 2.0,
                base_yaw + math.pi / 4.0,
                base_yaw - math.pi / 4.0,
            )
        )
        placement: tuple[float, float, float, np.ndarray] | None = None
        for yaw in yaw_candidates:
            quaternion = np.asarray(
                (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            )
            if collision_type == "box":
                rotation = _quat_matrix(quaternion) @ _quat_matrix(local_quat)
                footprint_half = np.abs(rotation[:2]) @ size
            else:
                footprint_half = np.asarray((size[0], size[0]), dtype=np.float64)
            try:
                x, y = _select_distractor_xy(
                    root,
                    desired_xy=desired_xy,
                    footprint_half=footprint_half,
                    protected_regions=tuple(protected_regions),
                    manipulation_clearance=manipulation_clearance,
                )
            except ValueError:
                continue
            placement = (x, y, yaw, quaternion)
            break
        if placement is None:
            raise ValueError(
                f"no safe pose for distractor {appearance!r} near {desired_xy}"
            )
        x, y, yaw, quaternion = placement
        if collision_type == "box":
            collision_rotation = _quat_matrix(quaternion) @ _quat_matrix(local_quat)
            support_height = float(np.abs(collision_rotation[2]) @ size)
        else:
            support_height = float(size[2])
        body = ET.SubElement(
            worldbody,
            "body",
            name=safe_name,
            pos=_fmt((x, y, floor_z + support_height + 1e-4)),
            quat=_fmt(quaternion),
        )
        physical_material = _object_material(appearance)
        if primitive:
            _add_primitive_visual(
                root,
                body,
                role_name=safe_name,
                appearance=appearance,
                half_extents=size,
                local_pos=np.zeros(3),
                local_quat=local_quat,
                case_bundle=case_bundle,
            )
        else:
            _add_hope_visual(
                root,
                body,
                role_name=safe_name,
                appearance=appearance,
                half_extents=size,
                local_pos=np.zeros(3),
                local_quat=local_quat,
                asset_root=asset_root,
                case_bundle=case_bundle,
            )
        collision_size = (
            _fmt(size)
            if collision_type == "box"
            else _fmt((size[0], size[2]))
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{safe_name}_collision",
            type=collision_type,
            size=collision_size,
            quat=_fmt(local_quat),
            density="0",
            group="3",
            rgba="0.55 0.55 0.55 0.18",
            condim="4",
            friction=_fmt(
                (
                    physical_material.sliding_friction,
                    physical_material.torsional_friction,
                    physical_material.rolling_friction,
                )
            ),
            solref=_fmt(elasticity_solref(0.01, physical_material.restitution)),
            solimp="0.9 0.95 0.001",
        )
        ET.SubElement(body, "site", name=f"{safe_name}_site", pos="0 0 0")
        records.append(
            {
                "body": safe_name,
                "name": DISPLAY_NAMES[appearance],
                "safe_name": safe_name,
                "material": physical_material.key,
                "visual_source": visual_source,
                "motion": "static",
            }
        )
    return records


def _update_metadata(
    metadata_path: Path,
    *,
    object_updates: list[dict[str, str]],
    distractors: list[dict[str, str]],
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    objects = list(metadata.get("objects", []))
    for update in object_updates:
        found = False
        for item in objects:
            if item.get("body") == update["old_body"]:
                item.update(
                    {
                        "body": update["new_body"],
                        "name": update["name"],
                        "safe_name": update["new_body"],
                    }
                )
                if update.get("material"):
                    item["material"] = update["material"]
                found = True
                break
        if not found:
            raise KeyError(
                f"metadata does not describe context body {update['old_body']!r}"
            )
    objects.extend(distractors)
    metadata["objects"] = objects
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prune_unreferenced_assets(bundle: Path, root: ET.Element) -> None:
    referenced = {
        Path(str(item.get("file"))).as_posix()
        for item in root.findall("./asset/*[@file]")
        if item.get("file")
    }
    for dirname in ("meshes", "textures"):
        directory = bundle / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file():
                rel = path.relative_to(bundle).as_posix()
                if rel not in referenced:
                    path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()


def _scene_module_text(strategy_id: int, case_index: int, camera: dict[str, Any]) -> str:
    package = f"benchmark_strategy_{strategy_id}_case_{case_index:03d}_scene"
    return f'''"""Sealed benchmark scene for strategy {strategy_id}, case {case_index:03d}."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from scene_io.runtime_assets import fr3_task_v1_xml


_PACKAGE_NAME = {package!r}
_PACKAGE_DIR = Path(__file__).resolve().parent
SCENE_DIR = _PACKAGE_DIR / "bundle"
SCENE_XML = SCENE_DIR / "scene.xml"
FR3_XML = fr3_task_v1_xml()
CONFIG: dict[str, Any] = {{
    "scene_dir": SCENE_DIR,
    "scene_xml": SCENE_XML,
    "fr3_xml": FR3_XML,
    "task_name": f"sim_{{_PACKAGE_NAME}}",
}}

DEFAULT_CAMERA: dict[str, Any] = {camera!r}
FR3_HOME_JOINT_QPOS: dict[str, float] = {FR3_HOME_JOINT_QPOS!r}
FR3_BASE_POS = (0.0, 0.0, 0.0)
FR3_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
FR3_MOUNT_FRAME = "fr3_mount"


def make_spec() -> mujoco.MjSpec:
    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    robot = mujoco.MjSpec.from_file(str(FR3_XML))
    home = robot.add_key(name="home")
    home.qpos = list(FR3_HOME_JOINT_QPOS.values())
    home.ctrl = [FR3_HOME_JOINT_QPOS[f"fr3_joint{{index}}"] for index in range(1, 8)] + [FR3_HOME_JOINT_QPOS["finger_joint1"]]
    mount = scene.worldbody.add_frame(name=FR3_MOUNT_FRAME, pos=FR3_BASE_POS, quat=FR3_BASE_QUAT)
    scene.attach(robot, prefix="", frame=mount)
    return scene


def make_model() -> mujoco.MjModel:
    return make_spec().compile()


def make_data(model: mujoco.MjModel | None = None) -> mujoco.MjData:
    model = make_model() if model is None else model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise KeyError("missing FR3 home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return data


def launch_viewer() -> None:
    import mujoco.viewer
    model = make_model()
    mujoco.viewer.launch(model, make_data(model))


__all__ = ["CONFIG", "DEFAULT_CAMERA", "FR3_BASE_POS", "FR3_BASE_QUAT", "FR3_HOME_JOINT_QPOS", "FR3_XML", "SCENE_DIR", "launch_viewer", "make_data", "make_model", "make_spec"]


if __name__ == "__main__":
    launch_viewer()
'''


def _replay_text(strategy_id: int, case_index: int) -> str:
    module = f"tasks.sim_benchmark.strategy_{strategy_id}.case_{case_index:03d}_scene.scene"
    return f'''"""Replay / record entry points for sealed benchmark case {case_index:03d}."""

from scene_io.playback import make_playback
from {module} import DEFAULT_CAMERA, make_spec

playback = make_playback(
    spec_builder=make_spec,
    default_camera=DEFAULT_CAMERA,
    default_output="sim_benchmark_strategy_{strategy_id}_case_{case_index:03d}_replay.mp4",
)

replay_browser = playback.browser
replay_mujoco = playback.mujoco
record_video = playback.record
'''


def materialize_case(
    repo_root: Path,
    asset_root: Path,
    contract: StrategyContract,
    parameters: VariantParameters,
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Create one standalone public case and return its private latent record."""

    source_case = (
        repo_root
        / "tasks"
        / "sim_benchmark"
        / contract.key
        / "case_000_scene"
    )
    destination = destination or source_case.parent / parameters.case_name
    if not source_case.is_dir():
        raise FileNotFoundError(f"reference case does not exist: {source_case}")
    load_asset_manifest(asset_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source_case / "bundle", temporary / "bundle", dirs_exist_ok=True)
        bundle = temporary / "bundle"
        xml_path = bundle / "scene.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root.set(
            "model",
            f"sim_benchmark_strategy_{contract.strategy_id}_case_{parameters.case_index:03d}_scene",
        )
        source_floor_z = _normalize_floor_height(root)
        editor = SceneEditor(
            root,
            parameters,
            asset_root=asset_root,
            case_bundle=bundle,
        )
        editor.layout_record["world_frame"] = {
            "source_floor_z": source_floor_z,
            "floor_z": 0.0,
            "robot_base_z": 0.0,
        }
        fixture_offsets: tuple[int, ...]
        if contract.strategy_id == 7:
            fixture_offsets = (0, 3)
        elif contract.fixture_bodies:
            fixture_offsets = (0,)
        else:
            fixture_offsets = ()
        editor.apply_work_surface_material(fixture_offsets=fixture_offsets)
        editor.apply_lighting_variation()
        context: ContextResult = generator_for(contract.strategy_id)(editor, contract)
        old_target, new_target, target_half_extents = _rename_target(
            root,
            contract,
            parameters,
            support_body=context.support_body,
            target_center_xy=context.target_center_xy,
            target_yaw_radians=context.target_yaw_radians,
            target_canonical_upright=context.target_canonical_upright,
            target_lateral_rank=context.target_lateral_rank,
            target_vertical_rank=context.target_vertical_rank,
            target_half_extents_override=context.target_half_extents_override,
            asset_root=asset_root,
            case_bundle=bundle,
        )
        if contract.strategy_id == 2:
            target_body_element = _find_body(root, new_target)
            target_inertial = target_body_element.find("inertial")
            target_collision = next(
                geom
                for geom in target_body_element.findall("geom")
                if geom.get("group") == "3"
            )
            if target_inertial is None:
                raise ValueError("strategy 2 target is missing inertial properties")
            target_friction = float(
                _vec(target_collision.get("friction"), (0.0,))[0]
            )
            manipulation = strategy_2_manipulation_metrics(
                2.0 * target_half_extents,
                mass_kg=float(target_inertial.get("mass", "nan")),
                sliding_friction=target_friction,
                gripper_max_width_m=STRATEGY_2_GRIPPER_MAX_WIDTH_M,
            )
            context.layout_record["wall_flip"]["manipulation_feasibility"] = (
                manipulation
            )
            if not manipulation["passed"]:
                failed = [
                    name
                    for name, passed in manipulation["checks"].items()
                    if not passed
                ]
                raise ValueError(
                    "strategy 2 target is not wall-flip/grasp feasible: "
                    + ", ".join(failed)
                )
        distractors = _add_distractors(
            root,
            contract,
            parameters,
            target_body=new_target,
            context_layout=context.layout_record,
            asset_root=asset_root,
            case_bundle=bundle,
        )
        _prune_unreferenced_assets(bundle, root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="unicode", xml_declaration=False)
        with xml_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")

        _update_metadata(
            bundle / "metadata.json",
            object_updates=[
                *context.object_updates,
                {
                    "old_body": old_target,
                    "new_body": new_target,
                    "name": DISPLAY_NAMES[parameters.appearance],
                    "material": _object_material(parameters.appearance).key,
                },
            ],
            distractors=distractors,
        )
        reference_module = json.loads(
            json.dumps(
                __import__(
                    f"tasks.sim_benchmark.strategy_{contract.strategy_id}.case_000_scene.scene",
                    fromlist=["DEFAULT_CAMERA"],
                ).DEFAULT_CAMERA
            )
        )
        fixed_default_camera = bool(
            context.layout_record.get("fixed_default_camera", False)
        )
        camera_distance = reference_module.get("camera_distance")
        target_longest_dimension = float(2.0 * np.max(target_half_extents))
        camera_zoom_scale = 1.0
        if (
            not fixed_default_camera
            and camera_distance is not None
            and target_longest_dimension < 0.10
        ):
            camera_zoom_scale = float(
                np.clip(target_longest_dimension / 0.10, 0.70, 1.0)
            )
            reference_module["camera_distance"] = (
                float(camera_distance) * camera_zoom_scale
            )
        camera_lookat = reference_module.get("camera_lookat")
        if camera_lookat is not None and len(camera_lookat) == 3:
            if not fixed_default_camera and context.target_center_xy is not None:
                camera_lookat[0] = float(context.target_center_xy[0])
                camera_lookat[1] = float(context.target_center_xy[1])
            camera_lookat[2] = max(
                0.04,
                float(camera_lookat[2]) - source_floor_z,
            )
        camera_override = context.layout_record.get("camera_override", {})
        for key in (
            "camera_distance",
            "camera_azimuth",
            "camera_elevation",
        ):
            if key in camera_override:
                reference_module[key] = float(camera_override[key])
        goal = contract.goal_template.format(
            target=DISPLAY_NAMES[parameters.appearance],
            **context.goal_values,
        )
        (temporary / "__init__.py").write_text("", encoding="utf-8")
        (temporary / "scene.py").write_text(
            _scene_module_text(contract.strategy_id, parameters.case_index, reference_module),
            encoding="utf-8",
        )
        (temporary / "replay.py").write_text(
            _replay_text(contract.strategy_id, parameters.case_index), encoding="utf-8"
        )
        (temporary / "goal.txt").write_text(goal + "\n", encoding="utf-8")
        public_record = {
            "benchmark_version": BENCHMARK_VERSION,
            "case_id": f"strategy_{contract.strategy_id}/case_{parameters.case_index:03d}",
            "goal_file": "goal.txt",
            "scored": True,
        }
        (temporary / "benchmark_case.json").write_text(
            json.dumps(public_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        entity_renames = dict(context.entity_renames)
        entity_renames[old_target] = new_target

        def remap_private(value: Any) -> Any:
            if isinstance(value, str):
                for source in sorted(entity_renames, key=len, reverse=True):
                    if value == source or value.startswith(f"{source}_"):
                        return entity_renames[source] + value[len(source) :]
                return value
            if isinstance(value, list):
                return [remap_private(item) for item in value]
            if isinstance(value, tuple):
                return tuple(remap_private(item) for item in value)
            if isinstance(value, dict):
                return {key: remap_private(item) for key, item in value.items()}
            return value

        return {
            "benchmark_version": BENCHMARK_VERSION,
            "strategy_id": contract.strategy_id,
            "case_index": parameters.case_index,
            "parameters": parameters.to_json(),
            "authoritative_roles": {
                "target": new_target,
                "fixtures": list(context.fixture_bodies),
                "secondary_dynamic": list(context.secondary_bodies),
                "support": context.support_body,
                "distractors": [item["safe_name"] for item in distractors],
            },
            "entity_renames": entity_renames,
            "context_layout": context.layout_record,
            "target_representation": {
                "visual_source": (
                    "material_primitive"
                    if is_primitive_object(parameters.appearance)
                    else "hope_asset"
                ),
                "source_asset": (
                    None
                    if is_primitive_object(parameters.appearance)
                    else parameters.appearance
                ),
                "primitive_preset": (
                    parameters.appearance
                    if is_primitive_object(parameters.appearance)
                    else None
                ),
                "full_dimensions_m": (2.0 * target_half_extents).tolist(),
                "collision_shape": "box",
            },
            "material_assignments": {
                "target": _object_material(parameters.appearance).key,
                "fixtures": {
                    name: record["key"]
                    for name, record in context.layout_record[
                        "fixture_materials"
                    ].items()
                },
                "secondary_dynamic": {
                    name: record["key"]
                    for name, record in context.layout_record[
                        "object_materials"
                    ].items()
                },
                "distractors": {
                    item["safe_name"]: item["material"] for item in distractors
                },
                "surfaces": {
                    name: record["key"]
                    for name, record in context.layout_record[
                        "surface_materials"
                    ].items()
                },
            },
            "relation_checks": remap_private(list(context.relation_checks)),
            "phases": list(contract.phases),
            "goal": goal,
            "target_half_extents": target_half_extents.tolist(),
            "deployment_camera": {
                "camera": reference_module,
                "target_longest_dimension_m": target_longest_dimension,
                "adaptive_zoom_scale": camera_zoom_scale,
            },
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
            continue
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["materialize_case", "tree_sha256"]
