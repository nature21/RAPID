"""Shared XML editing utilities for private per-strategy generators."""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .assets import (
    asset_alignment_quaternion,
    asset_record,
    matrix_quaternion,
    realistic_half_extents,
)
from .materials import (
    MATERIALS,
    SimulationMaterial,
    appearance_material,
    compatible_work_surface_material,
    effective_box_mass,
    material_texture_rgb,
    structural_material,
    work_surface_texture_rgb,
)
from .physics import elasticity_solref
from .primitives import (
    is_primitive_object,
    primitive_half_extents,
    primitive_inertia,
    primitive_mass,
    primitive_preset,
)
from .specs import DISPLAY_NAMES, VariantParameters, safe_appearance_name


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


def _box_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
    hx, hy, hz = half_extents
    return np.asarray(
        (
            mass * (hy * hy + hz * hz) / 3.0,
            mass * (hx * hx + hz * hz) / 3.0,
            mass * (hx * hx + hy * hy) / 3.0,
        )
    )


def _apply_contact_material(
    collision: ET.Element,
    material: SimulationMaterial,
) -> None:
    collision.set(
        "friction",
        _fmt(
            (
                material.sliding_friction,
                material.torsional_friction,
                material.rolling_friction,
            )
        ),
    )
    solref = _vec(collision.get("solref"), (0.01, 1.0))
    collision.set(
        "solref",
        _fmt(elasticity_solref(float(solref[0]), material.restitution)),
    )


def _appearance_body_name(appearance: str) -> str:
    safe = safe_appearance_name(appearance)
    display = DISPLAY_NAMES[appearance].lower()
    if not safe.endswith(("box", "carton")):
        if "box" in display:
            safe += "_box"
        elif "carton" in display:
            safe += "_carton"
    return safe


def planar_axes(yaw_radians: float) -> tuple[np.ndarray, np.ndarray]:
    """Return lateral and forward axes for a z-up task frame."""

    lateral = np.asarray(
        (math.cos(yaw_radians), math.sin(yaw_radians)),
        dtype=np.float64,
    )
    forward = np.asarray(
        (-math.sin(yaw_radians), math.cos(yaw_radians)),
        dtype=np.float64,
    )
    return lateral, forward


def semantic_box_rotation(
    half_extents: np.ndarray,
    *,
    lateral: np.ndarray,
    forward: np.ndarray,
    lateral_rank: int,
    vertical_rank: int,
) -> tuple[np.ndarray, float, float, float]:
    """Map ranked box dimensions into a reconstruction-free task frame."""

    order = np.argsort(np.asarray(half_extents), kind="stable")
    lateral_index = int(order[lateral_rank])
    vertical_index = int(order[vertical_rank])
    forward_index = next(
        index
        for index in range(3)
        if index not in {lateral_index, vertical_index}
    )
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[:, lateral_index] = (lateral[0], lateral[1], 0.0)
    rotation[:, forward_index] = (forward[0], forward[1], 0.0)
    rotation[:, vertical_index] = (0.0, 0.0, 1.0)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, forward_index] *= -1.0
    return (
        rotation,
        float(half_extents[lateral_index]),
        float(half_extents[forward_index]),
        float(half_extents[vertical_index]),
    )


@dataclass(frozen=True)
class ContextResult:
    fixture_bodies: tuple[str, ...]
    secondary_bodies: tuple[str, ...]
    support_body: str | None
    entity_renames: dict[str, str]
    goal_values: dict[str, str]
    relation_checks: tuple[dict[str, Any], ...]
    object_updates: tuple[dict[str, str], ...]
    layout_record: dict[str, Any]
    target_center_xy: tuple[float, float] | None = None
    target_yaw_radians: float | None = None
    target_canonical_upright: bool = False
    target_lateral_rank: int = 1
    target_vertical_rank: int = 0
    target_half_extents_override: tuple[float, float, float] | None = None


class SceneEditor:
    """Mutable scene editor used only by the sealed benchmark generator."""

    def __init__(
        self,
        root: ET.Element,
        parameters: VariantParameters,
        *,
        asset_root: Path,
        case_bundle: Path,
    ) -> None:
        self.root = root
        self.parameters = parameters
        self.asset_root = asset_root
        self.case_bundle = case_bundle
        self.entity_renames: dict[str, str] = {}
        self.object_updates: list[dict[str, str]] = []
        self.layout_record: dict[str, Any] = {
            "poses": {},
            "fixture_materials": {},
            "object_materials": {},
            "object_geometry": {},
            "surface_materials": {},
        }

    @property
    def values(self) -> tuple[float, ...]:
        return self.parameters.context_layout

    def body(self, name: str) -> ET.Element:
        body = self.root.find(f".//body[@name='{name}']")
        if body is None:
            raise KeyError(f"reference scene is missing body {name!r}")
        return body

    def collision(self, body_name: str) -> ET.Element:
        body = self.body(body_name)
        collision = next(
            (
                geom
                for geom in body.findall("geom")
                if geom.get("type") == "box" and geom.get("group") == "3"
            ),
            None,
        )
        if collision is None:
            raise ValueError(f"body {body_name!r} has no primitive collision box")
        return collision

    def appearance_half_extents(
        self,
        appearance: str,
        size_scale_xyz: tuple[float, float, float],
        *,
        broad_primitive_scale: bool,
    ) -> np.ndarray:
        if is_primitive_object(appearance):
            return primitive_half_extents(
                primitive_preset(appearance),
                size_scale_xyz,
                broad=broad_primitive_scale,
            )
        return realistic_half_extents(
            asset_record(self.asset_root, appearance),
            size_scale_xyz,
        )

    def collision_world_geometry(
        self,
        body_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = self.body(body_name)
        collision = self.collision(body_name)
        body_position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
        body_rotation = _quat_matrix(
            _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        )
        geom_position = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
        geom_rotation = body_rotation @ _quat_matrix(
            _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
        )
        center = body_position + body_rotation @ geom_position
        half_extents = _vec(collision.get("size"), (0.01, 0.01, 0.01))
        return center, geom_rotation, half_extents

    def resize_static_canonical_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
    ) -> None:
        """Resize a static canonical box collision and its matching visuals."""

        body = self.body(body_name)
        if body.find("freejoint") is not None:
            raise ValueError(
                f"cannot resize dynamic body {body_name!r} as a static fixture"
            )
        size = np.asarray(half_extents, dtype=np.float64)
        if size.shape != (3,) or np.any(size <= 0.0):
            raise ValueError(f"invalid box half-extents: {half_extents!r}")
        encoded = _fmt(size)
        self.collision(body_name).set("size", encoded)
        for geom in body.findall("geom"):
            if geom.get("type") == "box" and geom.get("group") == "2":
                geom.set("size", encoded)
        self.layout_record.setdefault("resized_static_boxes", {})[
            body_name
        ] = {"half_extents": size.tolist()}

    def place_collision_center_yaw(
        self,
        name: str,
        *,
        center_xy: tuple[float, float],
        yaw_radians: float,
    ) -> None:
        body = self.body(name)
        body_quaternion = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        delta = np.asarray(
            (
                math.cos(yaw_radians / 2.0),
                0.0,
                0.0,
                math.sin(yaw_radians / 2.0),
            ),
            dtype=np.float64,
        )
        new_quaternion = _quat_multiply(delta, body_quaternion)
        body.set("quat", _fmt(new_quaternion))
        body_rotation = _quat_matrix(new_quaternion)
        collision = self.collision(name)
        collision_position = _vec(
            collision.get("pos"), (0.0, 0.0, 0.0)
        )
        body_position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
        rotated_collision_position = body_rotation @ collision_position
        body_position[:2] = np.asarray(center_xy) - rotated_collision_position[:2]
        body.set("pos", _fmt(body_position))
        if body.find("freejoint") is not None:
            self.rest_on_floor(name)
        self._record_pose(name, body)

    def place_box_collision_on_floor(
        self,
        name: str,
        *,
        center_xy: tuple[float, float],
        world_rotation: np.ndarray,
    ) -> None:
        """Place a box with a prescribed world frame and a flush bottom face."""

        desired_rotation = np.asarray(world_rotation, dtype=np.float64)
        if desired_rotation.shape != (3, 3):
            raise ValueError("world_rotation must be a 3x3 matrix")
        if not np.allclose(
            desired_rotation.T @ desired_rotation,
            np.eye(3),
            atol=1e-8,
        ):
            raise ValueError("world_rotation must be orthonormal")
        if not math.isclose(
            float(np.linalg.det(desired_rotation)),
            1.0,
            abs_tol=1e-8,
        ):
            raise ValueError("world_rotation must be a proper rotation")

        body = self.body(name)
        collision = self.collision(name)
        local_position = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
        local_rotation = _quat_matrix(
            _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
        )
        body_rotation = desired_rotation @ local_rotation.T
        body.set("quat", _fmt(matrix_quaternion(body_rotation)))

        floor = self.root.find(".//geom[@name='floor']")
        if floor is None:
            raise KeyError("reference scene is missing floor")
        floor_z = float(_vec(floor.get("pos"), (0.0, 0.0, 0.0))[2])
        half_extents = _vec(collision.get("size"), (0.01, 0.01, 0.01))
        support_height = float(np.abs(desired_rotation[2]) @ half_extents)
        collision_center = np.asarray(
            (center_xy[0], center_xy[1], floor_z + support_height + 1e-4),
            dtype=np.float64,
        )
        body_position = collision_center - body_rotation @ local_position
        body.set("pos", _fmt(body_position))
        self.layout_record.setdefault("canonical_supports", {})[name] = {
            "support": "floor",
            "support_z": floor_z,
            "clearance": 1e-4,
        }
        self._record_pose(name, body)

    def _record_pose(self, name: str, body: ET.Element) -> None:
        self.layout_record["poses"][name] = {
            "pos": _vec(body.get("pos"), (0.0, 0.0, 0.0)).tolist(),
            "quat": _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0)).tolist(),
        }

    def move_yaw(self, name: str, dx: float, dy: float, yaw_degrees: float) -> None:
        body = self.body(name)
        position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
        position[:2] += (dx, dy)
        body.set("pos", _fmt(position))
        quaternion = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        yaw = math.radians(yaw_degrees)
        delta = np.asarray((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        body.set("quat", _fmt(_quat_multiply(delta, quaternion)))
        if body.find("freejoint") is not None:
            self.rest_on_floor(name)
        self._record_pose(name, body)

    def rigid_xy_transform(
        self,
        names: Iterable[str],
        *,
        center_xy: tuple[float, float],
        dx: float,
        dy: float,
        yaw_degrees: float,
    ) -> None:
        angle = math.radians(yaw_degrees)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        center = np.asarray(center_xy, dtype=np.float64)
        for name in names:
            body = self.body(name)
            position = _vec(body.get("pos"), (0.0, 0.0, 0.0))
            transformed = center + rotation @ (position[:2] - center) + (dx, dy)
            delta = transformed - position[:2]
            self.move_yaw(name, float(delta[0]), float(delta[1]), yaw_degrees)

    def rest_on_floor(self, name: str) -> None:
        body = self.body(name)
        collision = self.collision(name)
        floor = self.root.find(".//geom[@name='floor']")
        if floor is None:
            raise KeyError("reference scene is missing floor")
        floor_z = float(_vec(floor.get("pos"), (0.0, 0.0, 0.0))[2])
        body_pos = _vec(body.get("pos"), (0.0, 0.0, 0.0))
        body_quat = _vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        geom_pos = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
        geom_quat = _vec(collision.get("quat"), (1.0, 0.0, 0.0, 0.0))
        size = _vec(collision.get("size"), (0.01, 0.01, 0.01))
        body_rotation = _quat_matrix(body_quat)
        collision_rotation = body_rotation @ _quat_matrix(geom_quat)
        support_radius = float(np.abs(collision_rotation[2]) @ size)
        local_center_z = float((body_rotation @ geom_pos)[2])
        body_pos[2] = floor_z + support_radius - local_center_z + 1e-4
        body.set("pos", _fmt(body_pos))

    def _rename_body_entities(self, old_name: str, new_name: str) -> ET.Element:
        body = self.body(old_name)
        for element in body.iter():
            name = element.get("name")
            if name == old_name or (name and name.startswith(f"{old_name}_")):
                renamed = new_name + name[len(old_name) :]
                self.entity_renames[name] = renamed
                element.set("name", renamed)
        self.entity_renames[old_name] = new_name
        return body

    def apply_fixture_material(
        self,
        old_name: str,
        *,
        palette_offset: int = 0,
        canonical_box_visual: bool = False,
    ) -> tuple[str, str]:
        material = structural_material(
            self.parameters.fixture_material_index + palette_offset
        )
        new_name = old_name.replace("white", material.key)
        if new_name == old_name:
            new_name = f"{material.key}_{old_name}"
        body = self.body(old_name)
        asset = self.root.find("asset")
        if asset is None:
            raise ValueError("scene is missing asset section")
        fixture_scale = float(self.parameters.fixture_scale)
        collision = self.collision(old_name)
        for geom in body.findall("geom"):
            if geom.get("pos") is not None:
                geom.set(
                    "pos",
                    _fmt(fixture_scale * _vec(geom.get("pos"), (0.0, 0.0, 0.0))),
                )
            if geom.get("type") == "mesh" and geom.get("mesh"):
                mesh = asset.find(f"mesh[@name='{geom.get('mesh')}']")
                if mesh is not None:
                    mesh.set(
                        "scale",
                        _fmt(
                            fixture_scale
                            * _vec(mesh.get("scale"), (1.0, 1.0, 1.0))
                        ),
                    )
        collision.set(
            "size",
            _fmt(fixture_scale * _vec(collision.get("size"), (0.01, 0.01, 0.01))),
        )
        for site in body.findall("site"):
            if site.get("pos") is not None:
                site.set(
                    "pos",
                    _fmt(fixture_scale * _vec(site.get("pos"), (0.0, 0.0, 0.0))),
                )
        self.rest_on_floor(old_name)
        texture_name = f"{new_name}_{material.key}_surface"
        texture_path = (
            self.case_bundle / "textures" / f"material_{new_name}_{material.key}.png"
        )
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            material_texture_rgb(material, seed=self.parameters.seed)
        ).save(texture_path)
        ET.SubElement(
            asset,
            "texture",
            name=texture_name,
            type="2d",
            file=f"textures/{texture_path.name}",
        )
        if canonical_box_visual:
            visual_material_name = f"{new_name}_{material.key}_material"
            ET.SubElement(
                asset,
                "material",
                name=visual_material_name,
                texture=texture_name,
                texuniform="true",
                texrepeat="2 2",
                rgba=_fmt((1.0, 1.0, 1.0, material.opacity)),
                specular=f"{material.specular:.6g}",
                shininess=f"{material.shininess:.6g}",
                reflectance=f"{material.reflectance:.6g}",
            )
            self._remove_mesh_visuals(body)
            ET.SubElement(
                body,
                "geom",
                name=f"{old_name}_visual",
                type="box",
                size=collision.get("size", "0.01 0.01 0.01"),
                material=visual_material_name,
                pos=collision.get("pos", "0 0 0"),
                quat=collision.get("quat", "1 0 0 0"),
                contype="0",
                conaffinity="0",
                density="0",
                group="2",
            )
        else:
            for geom in body.findall("geom"):
                if geom.get("type") != "mesh":
                    continue
                material_name = geom.get("material")
                visual_material = (
                    asset.find(f"material[@name='{material_name}']")
                    if material_name
                    else None
                )
                if visual_material is not None:
                    visual_material.set(
                        "rgba", _fmt((1.0, 1.0, 1.0, material.opacity))
                    )
                    visual_material.set("texture", texture_name)
                    visual_material.set("texuniform", "true")
                    visual_material.set("texrepeat", "2 2")
                    visual_material.set("specular", f"{material.specular:.6g}")
                    visual_material.set("shininess", f"{material.shininess:.6g}")
                    visual_material.set("reflectance", f"{material.reflectance:.6g}")
                else:
                    geom.set("rgba", _fmt((*material.base_rgb, material.opacity)))
        _apply_contact_material(collision, material)
        self._rename_body_entities(old_name, new_name)
        descriptor = " ".join(
            material.display_name if token == "white" else token
            for token in old_name.split("_")
        )
        self.object_updates.append(
            {
                "old_body": old_name,
                "new_body": new_name,
                "name": descriptor,
                "material": material.key,
            }
        )
        self.layout_record["fixture_materials"][new_name] = {
            **material.to_json(),
            "fixture_scale": fixture_scale,
        }
        self._record_pose(new_name, body)
        return new_name, descriptor

    def apply_work_surface_material(
        self,
        *,
        fixture_offsets: tuple[int, ...] = (0,),
    ) -> str:
        """Replace the checker floor with a compatible physical work surface."""

        fixtures = tuple(
            structural_material(self.parameters.fixture_material_index + offset)
            for offset in fixture_offsets
        )
        material = compatible_work_surface_material(
            fixtures,
            seed=self.parameters.seed,
        )
        floor = self.root.find(".//geom[@name='floor']")
        asset = self.root.find("asset")
        if floor is None or asset is None:
            raise ValueError("scene is missing floor geom or asset section")
        old_material_name = floor.get("material")
        texture_name = f"work_surface_{material.key}_texture"
        material_name = f"work_surface_{material.key}_material"
        texture_path = (
            self.case_bundle / "textures" / f"work_surface_{material.key}.png"
        )
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            work_surface_texture_rgb(material, seed=self.parameters.seed + 104729)
        ).save(texture_path)
        ET.SubElement(
            asset,
            "texture",
            name=texture_name,
            type="2d",
            file=f"textures/{texture_path.name}",
        )
        ET.SubElement(
            asset,
            "material",
            name=material_name,
            texture=texture_name,
            texuniform="true",
            texrepeat=(
                "2 2" if material.surface_model == "wood_grain" else "4 4"
            ),
            rgba=_fmt((1.0, 1.0, 1.0, material.opacity)),
            specular=f"{material.specular:.6g}",
            shininess=f"{material.shininess:.6g}",
            reflectance=f"{material.reflectance:.6g}",
        )
        floor.set("material", material_name)
        _apply_contact_material(floor, material)

        if old_material_name and not self.root.findall(
            f".//geom[@material='{old_material_name}']"
        ):
            old_material = asset.find(f"material[@name='{old_material_name}']")
            if old_material is not None:
                old_texture_name = old_material.get("texture")
                asset.remove(old_material)
                if old_texture_name and not asset.findall(
                    f"material[@texture='{old_texture_name}']"
                ):
                    old_texture = asset.find(f"texture[@name='{old_texture_name}']")
                    if old_texture is not None:
                        asset.remove(old_texture)
        self.layout_record["surface_materials"]["floor"] = material.to_json()
        return material.key

    def apply_lighting_variation(self) -> None:
        """Apply bounded warm/cool and intensity variation to scene lighting."""

        intensity = float(self.parameters.lighting_intensity)
        temperature = float(self.parameters.lighting_temperature)
        tint = np.asarray(
            (1.0 + 0.07 * temperature, 1.0, 1.0 - 0.09 * temperature),
            dtype=np.float64,
        )
        lights: dict[str, dict[str, Any]] = {}
        for index, light in enumerate(self.root.findall("./worldbody/light")):
            for attribute in ("diffuse", "specular"):
                if light.get(attribute) is None:
                    continue
                color = _vec(light.get(attribute), (1.0, 1.0, 1.0))
                color = np.clip(color * intensity * tint, 0.0, 1.0)
                light.set(attribute, _fmt(color))
            position = _vec(light.get("pos"), (0.0, 0.0, 2.0))
            position += np.asarray(
                (
                    0.04 * self.values[(2 * index) % len(self.values)],
                    0.04 * self.values[(2 * index + 1) % len(self.values)],
                    0.06 * self.values[(2 * index + 2) % len(self.values)],
                )
            )
            light.set("pos", _fmt(position))
            lights[str(light.get("name", f"light_{index}"))] = {
                "pos": position.tolist(),
                "diffuse": _vec(light.get("diffuse"), (0.0, 0.0, 0.0)).tolist(),
            }
        headlight = self.root.find("./visual/headlight")
        if headlight is not None:
            for attribute in ("ambient", "diffuse", "specular"):
                if headlight.get(attribute) is None:
                    continue
                color = _vec(headlight.get(attribute), (1.0, 1.0, 1.0))
                headlight.set(
                    attribute,
                    _fmt(np.clip(color * intensity * tint, 0.0, 1.0)),
                )
        self.layout_record["lighting"] = {
            "intensity": intensity,
            "temperature": temperature,
            "lights": lights,
        }

    def _remove_mesh_visuals(self, body: ET.Element) -> None:
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh":
                body.remove(geom)

    def _unique_body_name(self, preferred: str) -> str:
        if self.root.find(f".//body[@name='{preferred}']") is None:
            return preferred
        suffix = 2
        while self.root.find(f".//body[@name='{preferred}_{suffix}']") is not None:
            suffix += 1
        return f"{preferred}_{suffix}"

    def _add_hope_visual(
        self,
        body: ET.Element,
        *,
        role_name: str,
        appearance: str,
        half_extents: np.ndarray,
        local_pos: np.ndarray,
        local_quat: np.ndarray,
    ) -> None:
        record = asset_record(self.asset_root, appearance)
        physical_material = appearance_material(appearance)
        safe_asset = str(record["safe_name"])
        source = self.asset_root / safe_asset
        mesh_target = self.case_bundle / "meshes" / f"hope_{safe_asset}.obj"
        texture_target = self.case_bundle / "textures" / f"hope_{safe_asset}.png"
        mesh_target.parent.mkdir(parents=True, exist_ok=True)
        texture_target.parent.mkdir(parents=True, exist_ok=True)
        if not mesh_target.exists():
            shutil.copy2(source / "visual.obj", mesh_target)
        if not texture_target.exists():
            shutil.copy2(source / "texture.png", texture_target)
        asset = self.root.find("asset")
        if asset is None:
            raise ValueError("scene is missing asset section")
        texture_name = f"{role_name}_{safe_asset}_texture"
        material_name = f"{role_name}_{safe_asset}_material"
        mesh_name = f"{role_name}_{safe_asset}_mesh"
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
            texuniform="false",
            specular=f"{physical_material.specular:.6g}",
            shininess=f"{physical_material.shininess:.6g}",
            reflectance=f"{physical_material.reflectance:.6g}",
        )
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=f"meshes/{mesh_target.name}",
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
        self,
        body: ET.Element,
        *,
        role_name: str,
        appearance: str,
        half_extents: np.ndarray,
        local_pos: np.ndarray,
        local_quat: np.ndarray,
    ) -> None:
        preset = primitive_preset(appearance)
        material = MATERIALS[preset.material_key]
        asset = self.root.find("asset")
        if asset is None:
            raise ValueError("scene is missing asset section")
        texture_name = f"{role_name}_{preset.material_key}_texture"
        material_name = f"{role_name}_{preset.material_key}_material"
        texture_path = (
            self.case_bundle
            / "textures"
            / f"primitive_{role_name}_{preset.material_key}.png"
        )
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            material_texture_rgb(
                material,
                seed=self.parameters.seed + sum(role_name.encode("utf-8")),
            )
        ).save(texture_path)
        ET.SubElement(
            asset,
            "texture",
            name=texture_name,
            type="2d",
            file=f"textures/{texture_path.name}",
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

    def reskin_dynamic(
        self,
        old_name: str,
        appearance: str,
        *,
        canonical_box: bool = False,
        half_extents_override: np.ndarray | None = None,
    ) -> tuple[str, str]:
        body = self.body(old_name)
        collision = self.collision(old_name)
        reference_half_extents = _vec(
            collision.get("size"), (0.01, 0.01, 0.01)
        )
        primitive = is_primitive_object(appearance)
        if primitive:
            preset = primitive_preset(appearance)
            if preset.shape != "box":
                raise ValueError(
                    f"task-relevant primitive must use a box collider: {appearance}"
                )
            half_extents = primitive_half_extents(preset, broad=False)
            physical_material = preset.material
        else:
            record = asset_record(self.asset_root, appearance)
            half_extents = realistic_half_extents(record)
            physical_material = appearance_material(appearance)
        if half_extents_override is not None:
            half_extents = np.asarray(
                half_extents_override,
                dtype=np.float64,
            )
            if half_extents.shape != (3,) or np.any(half_extents <= 0.0):
                raise ValueError(
                    f"invalid dynamic box half-extents: {half_extents_override!r}"
                )
        if canonical_box:
            local_pos = np.zeros(3, dtype=np.float64)
            local_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        else:
            local_pos = _vec(collision.get("pos"), (0.0, 0.0, 0.0))
            reference_quat = _vec(
                collision.get("quat"), (1.0, 0.0, 0.0, 0.0)
            )
            local_quat = _quat_multiply(
                reference_quat,
                asset_alignment_quaternion(half_extents, reference_half_extents),
            )
        collision.set("pos", _fmt(local_pos))
        collision.set("size", _fmt(half_extents))
        collision.set("quat", _fmt(local_quat))
        self._remove_mesh_visuals(body)
        new_name = self._unique_body_name(_appearance_body_name(appearance))
        body = self._rename_body_entities(old_name, new_name)
        if primitive:
            self._add_primitive_visual(
                body,
                role_name=new_name,
                appearance=appearance,
                half_extents=half_extents,
                local_pos=local_pos,
                local_quat=local_quat,
            )
        else:
            self._add_hope_visual(
                body,
                role_name=new_name,
                appearance=appearance,
                half_extents=half_extents,
                local_pos=local_pos,
                local_quat=local_quat,
            )
        _apply_contact_material(collision, physical_material)
        inertial = body.find("inertial")
        if inertial is not None:
            mass = (
                primitive_mass(preset, half_extents)
                if primitive
                else effective_box_mass(physical_material, half_extents)
            )
            inertial.set("mass", f"{mass:.9g}")
            inertial.set("pos", _fmt(local_pos))
            inertial.set("quat", _fmt(local_quat))
            inertia = (
                primitive_inertia(preset, half_extents, mass)
                if primitive
                else _box_inertia(mass, half_extents)
            )
            inertial.set("diaginertia", _fmt(inertia))
        self.rest_on_floor(new_name)
        descriptor = DISPLAY_NAMES[appearance]
        self.object_updates.append(
            {
                "old_body": old_name,
                "new_body": new_name,
                "name": descriptor,
                "material": physical_material.key,
            }
        )
        self._record_pose(new_name, body)
        self.layout_record["object_materials"][new_name] = physical_material.to_json()
        self.layout_record["object_geometry"][new_name] = {
            "visual_source": "material_primitive" if primitive else "hope_asset",
            "source_asset": None if primitive else appearance,
            "primitive_preset": appearance if primitive else None,
            "full_dimensions_m": (2.0 * half_extents).tolist(),
            "collision_shape": "box",
        }
        return new_name, descriptor

    def reskin_static(
        self,
        old_name: str,
        appearance: str,
        *,
        canonical_box: bool = False,
        half_extents_override: np.ndarray | None = None,
    ) -> tuple[str, str]:
        """Reskin a reference free body, then weld it into the world."""

        new_name, descriptor = self.reskin_dynamic(
            old_name,
            appearance,
            canonical_box=canonical_box,
            half_extents_override=half_extents_override,
        )
        body = self.body(new_name)
        if body.findall("joint"):
            raise ValueError(
                f"cannot weld articulated body {new_name!r} into the world"
            )
        freejoint = body.find("freejoint")
        if freejoint is not None:
            body.remove(freejoint)
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)

        material = self.layout_record["object_materials"].pop(new_name)
        self.layout_record["fixture_materials"][new_name] = material
        return new_name, descriptor


def distance_check(
    left: str,
    right: str,
    minimum: float,
    maximum: float,
) -> dict[str, Any]:
    return {
        "kind": "distance_xy",
        "left": left,
        "right": right,
        "minimum": minimum,
        "maximum": maximum,
    }


def run_strategy_cli(strategy_id: int) -> int:
    """Make each strategy module independently executable for debugging."""

    import sys

    from .cli import main

    return main(["generate", "--strategy", str(strategy_id), *sys.argv[1:]])


__all__ = [
    "ContextResult",
    "SceneEditor",
    "distance_check",
    "run_strategy_cli",
]
