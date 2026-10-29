"""MuJoCo scene export for real2sim reconstruction bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class _CollisionBox:
    center: np.ndarray
    half_extents: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class _VisualAsset:
    name: str
    file: str
    rgba: list[float]


def write_mujoco_scene(scene_xml: Path, objects: list[Any], config: Any) -> None:
    dynamic_objects = bool(config.dynamic_objects)
    collision_boxes = _prepare_collision_boxes(scene_xml, objects)

    root = ET.Element("mujoco", {"model": "real2sim_single_image"})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "meshdir": ".", "strippath": "false"},
    )
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "Euler"},
    )
    _add_visual_defaults(root)

    asset = ET.SubElement(root, "asset")
    _add_grid_material(asset, config)
    for obj in objects:
        for visual_asset in _visual_mesh_assets(obj):
            ET.SubElement(
                asset,
                "mesh",
                {"name": visual_asset.name, "file": visual_asset.file},
            )
            _upsert_visual_material(
                asset,
                _visual_material_name(visual_asset),
                visual_asset.rgba,
            )

    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        _contact_attributes_from_config(config),
    )

    worldbody = ET.SubElement(root, "worldbody")
    configured_floor_z = getattr(config, "support_floor_z", None)
    support_floor_z = (
        float(configured_floor_z)
        if configured_floor_z is not None
        else (
            0.0
            if str(getattr(config, "world_frame_mode", "")).replace("-", "_")
            == "support_surface"
            else None
        )
    )
    floor_z, plane_size, center = _scene_floor(objects, floor_z=support_floor_z)
    _add_lights(worldbody, center)
    floor_attrs = {
        "name": "floor",
        "type": "plane",
        "pos": _fmt_vec([0.0, 0.0, floor_z]),
        "size": _fmt_vec([plane_size, plane_size, 0.01]),
        "material": "grid",
    }
    floor_attrs.update(
        _contact_attributes_from_physics(
            getattr(config, "support_surface_physics", None),
            config,
        )
    )
    ET.SubElement(worldbody, "geom", floor_attrs)

    for idx, obj in enumerate(objects):
        object_dynamic = dynamic_objects and not bool(getattr(obj, "is_static", False))
        _add_reconstructed_body(
            worldbody,
            obj,
            idx,
            dynamic=object_dynamic,
            verticalize=bool(getattr(obj, "is_static", False)),
            collidable=True,
            mass=_object_mass(obj, config),
            contact_attrs=_contact_attributes_from_object(obj, config),
            collision_box=collision_boxes.get(obj.safe_name),
        )

    ET.indent(root, space="  ")
    scene_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def normalize_mujoco_lighting(scene_xml: Path) -> None:
    """Update an existing real2sim MJCF file to the current lighting profile."""

    tree = ET.parse(scene_xml)
    root = tree.getroot()
    _add_visual_defaults(root)
    _update_grid_material(root, _load_support_surface_physics(scene_xml))
    _normalize_visual_materials(root, _load_object_appearance_hints(scene_xml))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {scene_xml}")

    center = _xml_scene_center(worldbody)
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    _add_lights(worldbody, center)

    ET.indent(root, space="  ")
    scene_xml.write_text(
        ET.tostring(root, encoding="unicode") + "\n",
        encoding="utf-8",
    )


class InitialObjectPenetrationError(ValueError):
    """The generated scene starts with deeply intersecting reconstructed objects."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        pair = report.get("maximum_penetration_pair") or ["unknown", "unknown"]
        super().__init__(
            "initial reconstructed-object penetration exceeds limit: "
            f"{report['maximum_penetration_meters']:.6g} m > "
            f"{report['threshold_meters']:.6g} m "
            f"({pair[0]} vs {pair[1]})"
        )


def object_collision_geom_ids(
    model: Any, object_safe_names: list[str] | tuple[str, ...] | None,
) -> set[int]:
    """Include every active collision part of the named reconstructed bodies."""

    names = set(object_safe_names) if object_safe_names is not None else {
        model.body(int(model.geom_bodyid[index])).name
        for index in range(model.ngeom)
        if model.geom(index).name.endswith("_collision")
    }
    return {
        index for index in range(model.ngeom)
        if model.body(int(model.geom_bodyid[index])).name in names
        and (model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
    }


def compile_mujoco_scene(
    scene_xml: Path,
    *,
    object_safe_names: list[str] | tuple[str, ...] | None = None,
    max_initial_object_penetration_meters: float = 0.03,
) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_geom_ids = object_collision_geom_ids(model, object_safe_names)

    contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first_id = int(contact.geom[0])
        second_id = int(contact.geom[1])
        if first_id not in object_geom_ids or second_id not in object_geom_ids:
            continue
        contacts.append(
            {
                "geoms": [model.geom(first_id).name, model.geom(second_id).name],
                "penetration_meters": max(0.0, -float(contact.dist)),
                "normal_world": np.asarray(contact.frame, dtype=float)[:3].tolist(),
                "position_world": np.asarray(contact.pos, dtype=float).tolist(),
            }
        )
    contacts.sort(key=lambda item: float(item["penetration_meters"]), reverse=True)
    maximum = float(contacts[0]["penetration_meters"]) if contacts else 0.0
    threshold = max(0.0, float(max_initial_object_penetration_meters))
    report = {
        "status": "ok" if maximum <= threshold else "failed",
        "total_contact_count": int(data.ncon),
        "object_contact_count": len(contacts),
        "maximum_penetration_meters": maximum,
        "maximum_penetration_pair": contacts[0]["geoms"] if contacts else None,
        "threshold_meters": threshold,
        "deepest_object_contacts": contacts[:10],
    }
    if maximum > threshold:
        raise InitialObjectPenetrationError(report)
    return report


def _add_grid_material(asset: ET.Element, config: Any | None = None) -> None:
    rgb1, rgb2, reflectance = _floor_material_values(
        getattr(config, "support_surface_physics", None) if config is not None else None
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": rgb1,
            "rgb2": rgb2,
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "grid",
            "texture": "grid",
            "texrepeat": "4 4",
            "texuniform": "true",
            "reflectance": reflectance,
        },
    )


def _update_grid_material(root: ET.Element, support_physics: Any) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    rgb1, rgb2, reflectance = _floor_material_values(support_physics)
    texture = asset.find("./texture[@name='grid']")
    if texture is not None:
        texture.set("rgb1", rgb1)
        texture.set("rgb2", rgb2)
    material = asset.find("./material[@name='grid']")
    if material is not None:
        material.set("reflectance", reflectance)


def _normalize_visual_materials(
    root: ET.Element,
    appearance_hints: dict[str, str],
) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        return

    for body in worldbody.findall(".//body"):
        body_name = body.get("name", "")
        appearance_text = appearance_hints.get(body_name, body_name)
        for geom in body.findall("geom"):
            if geom.get("group") != "2" or geom.get("type") != "mesh":
                continue
            material_name = geom.get("material")
            if not material_name:
                mesh_name = geom.get("mesh") or geom.get("name") or "visual"
                material_name = f"{mesh_name}_mat"
                geom.set("material", material_name)
            rgba = _geom_visual_rgba(root, geom)
            adjusted_rgba = _adjust_visual_rgba(rgba, appearance_text)
            _upsert_visual_material(asset, material_name, adjusted_rgba)
            geom.attrib.pop("rgba", None)


def _geom_visual_rgba(root: ET.Element, geom: ET.Element) -> list[float]:
    material_name = geom.get("material")
    if material_name:
        material = root.find(f"./asset/material[@name='{material_name}']")
        if material is not None:
            rgba_text = material.get("rgba")
            if rgba_text:
                rgba = _parse_float_vector(rgba_text)
                if rgba.size >= 3:
                    return _rgba4(rgba)

    rgba_text = geom.get("rgba")
    if rgba_text:
        rgba = _parse_float_vector(rgba_text)
        if rgba.size >= 3:
            return _rgba4(rgba)
    return [1.0, 1.0, 1.0, 1.0]


def _load_object_appearance_hints(scene_xml: Path) -> dict[str, str]:
    metadata_json = scene_xml.parent / "metadata.json"
    if not metadata_json.is_file():
        return {}
    try:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    hints: dict[str, str] = {}
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        return hints
    for item in objects:
        if not isinstance(item, dict):
            continue
        safe_name = str(item.get("safe_name", "")).strip()
        if not safe_name:
            continue
        hints[safe_name] = _appearance_text_from_mapping(item)
    return hints


def _appearance_text_from_mapping(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, ""))
        for key in (
            "name",
            "description",
            "physical_material",
            "physical_reason",
            "safe_name",
        )
    ).lower()


def _load_support_surface_physics(scene_xml: Path) -> dict[str, Any]:
    metadata_json = scene_xml.parent / "metadata.json"
    if not metadata_json.is_file():
        return {}
    try:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    mujoco_meta = metadata.get("mujoco")
    if isinstance(mujoco_meta, dict):
        physics = mujoco_meta.get("support_surface_physics")
        if isinstance(physics, dict):
            return physics
    return {}


def _floor_material_values(support_physics: Any) -> tuple[str, str, str]:
    physics = support_physics if isinstance(support_physics, dict) else {}
    text = " ".join(
        str(physics.get(key, ""))
        for key in ("object_name", "material", "reason")
    ).lower()
    if any(word in text for word in ("black", "dark", "matte")):
        return "0.035 0.035 0.035", "0.10 0.10 0.10", ".035"
    if any(word in text for word in ("wood", "cardboard", "tan", "brown")):
        return ".58 .48 .36", ".42 .34 .24", ".06"
    if any(word in text for word in ("white", "light", "paper", "foam")):
        return ".72 .72 .70", ".56 .56 .54", ".04"
    return ".58 .58 .56", ".44 .44 .42", ".04"


def _add_visual_defaults(root: ET.Element) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        insert_at = 0
        for index, child in enumerate(list(root)):
            if child.tag in {"compiler", "option"}:
                insert_at = index + 1
        root.insert(insert_at, visual)

    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.40 0.40 0.40")
    headlight.set("diffuse", "0.48 0.48 0.48")
    headlight.set("specular", "0.08 0.08 0.08")


def _add_lights(worldbody: ET.Element, center: np.ndarray) -> None:
    center = np.asarray(center, dtype=float).reshape(3)
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key",
            "pos": _fmt_vec([center[0] - 1.2, center[1] - 1.0, center[2] + 2.2]),
            "dir": "1 1 -2.4",
            "diffuse": "0.50 0.50 0.50",
            "specular": "0.06 0.06 0.06",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill",
            "pos": _fmt_vec([center[0] + 1.1, center[1] + 0.9, center[2] + 1.6]),
            "dir": "-1 -0.8 -1.8",
            "diffuse": "0.28 0.28 0.28",
            "specular": "0.02 0.02 0.02",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "overhead",
            "pos": _fmt_vec([center[0], center[1], center[2] + 2.0]),
            "dir": "0 0 -1",
            "diffuse": "0.16 0.16 0.16",
            "specular": "0.01 0.01 0.01",
        },
    )


def _upsert_visual_material(
    asset: ET.Element,
    name: str,
    rgba: list[float],
) -> None:
    material = asset.find(f"./material[@name='{name}']")
    if material is None:
        material = ET.SubElement(asset, "material", {"name": name})
    material.set("rgba", _fmt_vec(_rgba4(np.asarray(rgba, dtype=float))))
    material.set("specular", "0")
    material.set("shininess", "0")
    material.set("reflectance", "0")


def _visual_material_name(visual_asset: _VisualAsset) -> str:
    return f"{visual_asset.name}_mat"


def _add_reconstructed_body(
    worldbody: ET.Element,
    obj: Any,
    index: int,
    *,
    dynamic: bool,
    verticalize: bool,
    collidable: bool,
    mass: float,
    contact_attrs: dict[str, str],
    collision_box: _CollisionBox | None,
) -> None:
    body_pos = np.asarray(obj.body_pos, dtype=float)
    body_quat: np.ndarray | None = None
    if verticalize:
        body_pos, body_quat = _verticalized_box_body_pose(
            body_pos,
            np.eye(3, dtype=float),
            collision_box,
        )
    body_attrs = {"name": obj.safe_name, "pos": _fmt_vec(body_pos)}
    if body_quat is not None:
        body_attrs["quat"] = _fmt_vec(body_quat)
    body = ET.SubElement(
        worldbody,
        "body",
        body_attrs,
    )
    if dynamic:
        ET.SubElement(body, "freejoint", {"name": f"{obj.safe_name}_joint"})
        ET.SubElement(
            body,
            "inertial",
            {
                "mass": _fmt_float(mass),
                "diaginertia": _fmt_vec(
                    _box_diagonal_inertia(collision_box.half_extents, mass)
                ),
                "pos": _fmt_vec(collision_box.center),
                "quat": _fmt_vec(collision_box.quat),
            },
        )

    for visual_index, visual_asset in enumerate(_visual_mesh_assets(obj)):
        suffix = "" if visual_index == 0 else f"_{visual_index}"
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.safe_name}_visual{suffix}",
                "type": "mesh",
                "mesh": visual_asset.name,
                "material": _visual_material_name(visual_asset),
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "2",
            },
        )

    if collidable and collision_box is not None:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.safe_name}_collision",
                "type": "box",
                "pos": _fmt_vec(collision_box.center),
                "quat": _fmt_vec(collision_box.quat),
                "size": _fmt_vec(collision_box.half_extents),
                "density": "0",
                "group": "3",
                "rgba": "0.55 0.55 0.55 0.18",
                **contact_attrs,
            },
        )

    ET.SubElement(body, "site", {"name": f"{obj.safe_name}_site", "pos": "0 0 0"})


def _visual_mesh_assets(obj: Any) -> list[_VisualAsset]:
    specs = getattr(obj, "visual_mesh_files", None)
    if not specs:
        specs = [{"file": obj.mesh_file, "rgba": [1.0, 1.0, 1.0, 1.0]}]
    assets: list[_VisualAsset] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            continue
        file = str(spec.get("file", "")).strip()
        if not file:
            continue
        rgba = np.asarray(spec.get("rgba", [1.0, 1.0, 1.0, 1.0]), dtype=float)
        if rgba.size < 4:
            rgba = np.pad(rgba.reshape(-1), (0, 4 - rgba.size), constant_values=1.0)
        rgba = np.asarray(
            _adjust_visual_rgba(rgba[:4], _appearance_text_from_object(obj)),
            dtype=float,
        )
        name = f"{obj.safe_name}_mesh" if index == 0 else f"{obj.safe_name}_mesh_{index}"
        assets.append(_VisualAsset(name=name, file=file, rgba=rgba[:4].tolist()))
    if not assets:
        assets.append(
            _VisualAsset(
                name=f"{obj.safe_name}_mesh",
                file=str(obj.mesh_file),
                rgba=[1.0, 1.0, 1.0, 1.0],
            )
        )
    return assets


def _appearance_text_from_object(obj: Any) -> str:
    return " ".join(
        str(getattr(obj, key, ""))
        for key in (
            "name",
            "description",
            "physical_material",
            "physical_reason",
            "safe_name",
        )
    ).lower()


def _adjust_visual_rgba(rgba: Any, appearance_text: str) -> list[float]:
    values = _rgba4(np.asarray(rgba, dtype=float))
    rgb = np.clip(np.asarray(values[:3], dtype=float), 0.0, 1.0)
    alpha = float(np.clip(values[3], 0.0, 1.0))

    if _is_corrugated_cardboard_appearance(appearance_text):
        rgb = _adjust_corrugated_cardboard_rgb(rgb)

    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha]


def _is_corrugated_cardboard_appearance(text: str) -> bool:
    text = text.lower()
    positive = ("plain cardboard", "corrugated", "tan", "brown", "shipping box")
    negative = ("cheez", "snack", "orange", "printed", "coated", "retail")
    return any(term in text for term in positive) and not any(
        term in text for term in negative
    )


def _adjust_corrugated_cardboard_rgb(rgb: np.ndarray) -> np.ndarray:
    base = np.asarray([0.46, 0.34, 0.20], dtype=float)
    luma = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    saturation = float(rgb.max(initial=0.0) - rgb.min(initial=0.0))
    if luma > 0.50:
        rgb = 0.70 * base + 0.30 * rgb
    elif saturation < 0.16 and luma > 0.38:
        rgb = 0.55 * base + 0.45 * rgb
    luma = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    if luma > 0.43:
        rgb = rgb * (0.43 / luma)
    return np.clip(rgb, [0.16, 0.11, 0.06], [0.52, 0.42, 0.28])


def _object_mass(obj: Any, config: Any) -> float:
    mass = _optional_positive_float(getattr(obj, "physical_mass_kg", None))
    if mass is None:
        mass = _optional_positive_float(getattr(config, "object_mass", None))
    return float(np.clip(0.2 if mass is None else mass, 0.001, 100.0))


def _contact_attributes_from_config(config: Any) -> dict[str, str]:
    physics = {
        "sliding_friction": getattr(config, "default_sliding_friction", 0.7),
        "torsional_friction": getattr(config, "default_torsional_friction", 0.01),
        "rolling_friction": getattr(config, "default_rolling_friction", 0.001),
        "restitution": getattr(config, "default_restitution", 0.05),
    }
    return _contact_attributes_from_physics(physics, config)


def _contact_attributes_from_object(obj: Any, config: Any) -> dict[str, str]:
    physics = {
        "sliding_friction": getattr(obj, "physical_sliding_friction", None),
        "torsional_friction": getattr(obj, "physical_torsional_friction", None),
        "rolling_friction": getattr(obj, "physical_rolling_friction", None),
        "restitution": getattr(obj, "physical_restitution", None),
    }
    return _contact_attributes_from_physics(physics, config)


def _contact_attributes_from_physics(physics: Any, config: Any) -> dict[str, str]:
    physics = physics if isinstance(physics, dict) else {}
    sliding = _physics_float(
        physics,
        "sliding_friction",
        getattr(config, "default_sliding_friction", 0.7),
    )
    torsional = _physics_float(
        physics,
        "torsional_friction",
        getattr(config, "default_torsional_friction", 0.01),
    )
    rolling = _physics_float(
        physics,
        "rolling_friction",
        getattr(config, "default_rolling_friction", 0.001),
    )
    restitution = _physics_float(
        physics,
        "restitution",
        getattr(config, "default_restitution", 0.05),
    )

    sliding = float(np.clip(sliding, 0.03, 3.0))
    torsional = float(np.clip(torsional, 0.0, 0.2))
    rolling = float(np.clip(rolling, 0.0, 0.1))
    restitution = float(np.clip(restitution, 0.0, 1.0))
    damping_ratio = float(np.clip(1.0 - 0.7 * restitution, 0.2, 1.0))

    return {
        "condim": "4",
        "friction": _fmt_vec([sliding, torsional, rolling]),
        "solref": _fmt_vec([0.01, damping_ratio]),
        "solimp": "0.9 0.95 0.001",
    }


def _physics_float(physics: dict[str, Any], key: str, default: float) -> float:
    value = physics.get(key)
    if value is None and key == "sliding_friction":
        value = physics.get("dynamic_friction")
    if value is None and key == "sliding_friction":
        friction = physics.get("friction")
        if isinstance(friction, (list, tuple)) and friction:
            value = friction[0]
    if value is None:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return out


def _optional_positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _prepare_collision_boxes(
    scene_xml: Path,
    objects: list[Any],
) -> dict[str, _CollisionBox]:
    return {
        obj.safe_name: _fit_3d_obb(scene_xml, obj)
        for obj in objects
    }


def _fit_3d_obb(
    scene_xml: Path,
    obj: Any,
) -> _CollisionBox:
    """Fit a full 3D oriented bounding box to an object's local mesh."""

    import trimesh

    source_mesh_file = str(
        getattr(obj, "obb_source_mesh_file", "")
        or getattr(obj, "collision_mesh_file", "")
        or obj.mesh_file
    )
    mesh_path = _resolve_scene_asset(scene_xml, source_mesh_file)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"reconstructed mesh not found: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError(f"reconstructed mesh has insufficient vertices: {mesh_path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"reconstructed mesh has non-finite vertices: {mesh_path}")

    return _obb_3d_from_vertices(vertices)


def _obb_3d_from_vertices(
    vertices: np.ndarray,
) -> _CollisionBox:
    """Fit a tight 3D OBB to the complete vertex set."""

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError("an oriented collision box requires at least three 3D vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("oriented collision box vertices must all be finite")

    return _tight_obb_3d(vertices)


def _tight_obb_3d(vertices: np.ndarray) -> _CollisionBox:
    import trimesh

    try:
        transform, _ = trimesh.bounds.oriented_bounds(
            vertices,
            angle_digits=2,
            ordered=True,
        )
        rotation = np.asarray(transform[:3, :3], dtype=float).T
    except Exception:
        rotation = _principal_axes_3d(vertices)

    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0

    projected = vertices @ rotation
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    half_extents = np.maximum((upper - lower) / 2.0, 5e-5)
    center = ((lower + upper) / 2.0) @ rotation.T
    quat = _rotation_matrix_to_quat(rotation)
    return _CollisionBox(center=center, half_extents=half_extents, quat=quat)


def _principal_axes_3d(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    return rotation


def _rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Return MuJoCo's wxyz quaternion for a proper 3D rotation matrix."""

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    import trimesh

    quat = np.asarray(trimesh.transformations.quaternion_from_matrix(matrix), dtype=float)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    import trimesh

    return np.asarray(
        trimesh.transformations.quaternion_matrix(quat)[:3, :3],
        dtype=float,
    )


def _verticalized_box_body_pose(
    body_pos: np.ndarray,
    body_rotation: np.ndarray,
    collision_box: _CollisionBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimally rotate one box axis onto world Z and preserve its bottom."""

    box_rotation = _quat_to_rotation_matrix(collision_box.quat)
    world_box_rotation = body_rotation @ box_rotation
    vertical_axis = int(np.argmax(np.abs(world_box_rotation[2, :])))
    source_axis = world_box_rotation[:, vertical_axis]
    target_axis = np.array(
        [0.0, 0.0, 1.0 if source_axis[2] >= 0.0 else -1.0],
        dtype=float,
    )
    correction = _rotation_between_vectors(source_axis, target_axis)

    current_center = body_pos + body_rotation @ collision_box.center
    current_bottom = float(
        current_center[2]
        - np.sum(
            np.abs(world_box_rotation[2, :]) * collision_box.half_extents
        )
    )

    corrected_body_rotation = correction @ body_rotation
    corrected_box_rotation = corrected_body_rotation @ box_rotation
    corrected_local_center = corrected_body_rotation @ collision_box.center
    corrected_local_bottom = float(
        corrected_local_center[2]
        - np.sum(
            np.abs(corrected_box_rotation[2, :]) * collision_box.half_extents
        )
    )
    corrected_body_pos = np.asarray(body_pos, dtype=float).copy()
    corrected_body_pos[2] = current_bottom - corrected_local_bottom
    return corrected_body_pos, _rotation_matrix_to_quat(corrected_body_rotation)


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        return np.eye(3, dtype=float)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + skew + skew @ skew * (
        (1.0 - cosine) / (sine * sine)
    )


def _resolve_scene_asset(scene_xml: Path, asset_file: str | Path) -> Path:
    asset_path = Path(asset_file)
    if asset_path.is_absolute():
        return asset_path
    return scene_xml.parent / asset_path


def _box_diagonal_inertia(half_extents: Any, mass: float) -> np.ndarray:
    half_extents = np.asarray(half_extents, dtype=float)
    if half_extents.shape != (3,) or not np.isfinite(half_extents).all():
        extents = np.full(3, 1e-3, dtype=float)
    else:
        extents = np.maximum(2.0 * half_extents, 1e-3)

    x, y, z = extents
    inertia = np.array(
        [
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        ],
        dtype=float,
    )
    return np.maximum(inertia, 1e-8)


def _scene_floor(
    objects: list[Any],
    *,
    floor_z: float | None = None,
) -> tuple[float, float, np.ndarray]:
    if not objects:
        return 0.0 if floor_z is None else float(floor_z), 1.0, np.zeros(3, dtype=float)

    bounds = np.asarray([obj.world_bounds for obj in objects], dtype=float)
    mins = bounds[:, 0, :].min(axis=0)
    maxs = bounds[:, 1, :].max(axis=0)
    floor_z = float(mins[2]) if floor_z is None else float(floor_z)
    center = (mins + maxs) / 2.0
    span_xy = np.maximum(maxs[:2] - mins[:2], 0.5)
    plane_size = float(max(1.0, np.max(span_xy) * 1.5))
    return floor_z, plane_size, center


def _xml_scene_center(worldbody: ET.Element) -> np.ndarray:
    points: list[np.ndarray] = []
    for elem in worldbody.iter():
        pos_text = elem.get("pos")
        if not pos_text:
            continue
        try:
            pos = np.fromstring(pos_text, sep=" ", dtype=float)
        except ValueError:
            continue
        if pos.shape == (3,) and np.isfinite(pos).all():
            points.append(pos)
    if not points:
        return np.zeros(3, dtype=float)
    return np.asarray(points, dtype=float).mean(axis=0)


def _parse_float_vector(text: str) -> np.ndarray:
    try:
        return np.fromstring(text, sep=" ", dtype=float)
    except ValueError:
        return np.asarray([], dtype=float)


def _rgba4(values: Any) -> list[float]:
    rgba = np.asarray(values, dtype=float).reshape(-1)
    if rgba.size < 4:
        rgba = np.pad(rgba, (0, 4 - rgba.size), constant_values=1.0)
    return np.clip(rgba[:4], 0.0, 1.0).tolist()


def _fmt_vec(values: Any) -> str:
    return " ".join(_fmt_float(value) for value in np.asarray(values).reshape(-1))


def _fmt_float(value: Any) -> str:
    value = float(value)
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.8g}"
