"""Check visual/collision envelopes and fit multipart visuals to a corrected box."""

from __future__ import annotations

import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real2sim.adapter import _geom_points
from real2sim.mujoco_export import _obb_3d_from_vertices, _quat_to_rotation_matrix


def _object_geometry(model, data, name: str):
    body_id = model.body(name).id
    ids = np.flatnonzero(model.geom_bodyid == body_id)
    collisions = [int(index) for index in ids
                  if model.geom_contype[index] or model.geom_conaffinity[index]]
    visuals = [int(index) for index in ids if index not in collisions
               and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH]
    return body_id, visuals, collisions


def visual_collision_alignment(
    scene_xml: Path, names: list[str], *, absolute_tolerance: float = 0.01,
    relative_tolerance: float = 0.15,
) -> dict:
    """Measure envelope disagreement in each object's collision coordinate frame.

    This checks geometric consistency, not alignment to the real image. Both
    remain necessary: a matching pair of visual/collision models can be wrong.
    """

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objects = {}
    failed = []
    for name in names:
        _, visuals, collisions = _object_geometry(model, data, name)
        if not visuals:
            objects[name] = {"status": "skipped", "reason": "no separate visual mesh"}
            continue
        primary = model.geom(f"{name}_collision").id
        rotation = data.geom_xmat[primary].reshape(3, 3)
        origin = data.geom_xpos[primary]
        visual_points = np.vstack([_geom_points(model, data, index) for index in visuals])
        collision_points = [_geom_points(model, data, index) for index in collisions]
        if not collision_points or any(points is None for points in collision_points):
            objects[name] = {"status": "skipped", "reason": "unsupported collision geometry"}
            continue
        visual_local = (visual_points - origin) @ rotation
        collision_local = (np.vstack(collision_points) - origin) @ rotation
        visual_bounds = np.array([visual_local.min(axis=0), visual_local.max(axis=0)])
        collision_bounds = np.array([collision_local.min(axis=0), collision_local.max(axis=0)])
        tolerance = np.maximum(absolute_tolerance, relative_tolerance * np.ptp(collision_bounds, axis=0))
        errors = np.abs(visual_bounds - collision_bounds)
        mismatch = bool(np.any(errors > tolerance))
        if mismatch:
            failed.append(name)
        objects[name] = {
            "status": "failed" if mismatch else "ok",
            "visual_bounds_in_collision_frame": visual_bounds.tolist(),
            "collision_bounds_in_collision_frame": collision_bounds.tolist(),
            "face_errors_meters": errors.tolist(),
            "tolerance_per_axis_meters": tolerance.tolist(),
            "maximum_face_error_meters": float(errors.max()),
        }
    return {"status": "failed" if failed else "ok", "failed_objects": failed, "objects": objects}


def _closest_box_axes(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Resolve OBB axis/sign ambiguity without accidentally turning meshes 90°."""

    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            axes = source[:, permutation] * np.asarray(signs)
            if np.linalg.det(axes) > 0:
                candidates.append((float(np.linalg.norm(axes - target)), axes, permutation))
    _, axes, permutation = min(candidates, key=lambda item: item[0])
    return axes, permutation


def _vector(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _write_mesh(model, mesh_id: int, vertices: np.ndarray, path: Path) -> None:
    """Write compiled triangle topology and UVs with transformed vertices."""

    face_start = int(model.mesh_faceadr[mesh_id])
    face_end = face_start + int(model.mesh_facenum[mesh_id])
    faces = model.mesh_face[face_start:face_end]
    texture_start = int(model.mesh_texcoordadr[mesh_id])
    texture_count = int(model.mesh_texcoordnum[mesh_id])
    with path.open("w", encoding="utf-8") as output:
        for vertex in vertices:
            output.write(f"v {_vector(vertex)}\n")
        if texture_start >= 0 and texture_count:
            for uv in model.mesh_texcoord[texture_start:texture_start + texture_count]:
                output.write(f"vt {_vector(uv)}\n")
        for offset, face in enumerate(faces):
            uv_indices = model.mesh_facetexcoord[face_start + offset]
            if texture_start >= 0 and texture_count and np.all(uv_indices >= 0):
                refs = [f"{int(vertex) + 1}/{int(uv) + 1}" for vertex, uv in zip(face, uv_indices)]
            else:
                refs = [str(int(vertex) + 1) for vertex in face]
            output.write("f " + " ".join(refs) + "\n")


def fit_visuals_to_collision(scene_xml: Path, names: list[str]) -> dict:
    """Fit each object's entire visual assembly to its trusted collision OBB.

    All color/texture parts use one affine transform. Bodies, collisions and
    physical inertia retain their current state. Use only when the collision
    box is already a good representation of the real object.
    """

    tree = ET.parse(scene_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("scene has no visual mesh assets")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = visual_collision_alignment(scene_xml, names)
    mesh_dir = scene_xml.parent / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    existing_assets = {item.get("name") for item in asset}
    outputs = []
    bodies = {body.get("name"): body for body in root.findall("./worldbody/body")}
    for name in names:
        body_id, visuals, collisions = _object_geometry(model, data, name)
        if not visuals:
            continue
        if len(collisions) != 1 or model.geom_type[collisions[0]] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"{name}: automatic fitting requires one box; align compound geometry manually")
        collision = collisions[0]
        target_rotation = data.geom_xmat[collision].reshape(3, 3)
        target_center = data.geom_xpos[collision]
        parts = [_geom_points(model, data, index) for index in visuals]
        fitted = _obb_3d_from_vertices(np.vstack(parts))
        axes, permutation = _closest_box_axes(_quat_to_rotation_matrix(fitted.quat), target_rotation)
        half_extents = np.asarray(fitted.half_extents)[list(permutation)]
        scale = model.geom_size[collision, :3] / half_extents
        body_rotation = data.xmat[body_id].reshape(3, 3)
        body = bodies[name]
        geoms = {geom.get("name"): geom for geom in body.findall("geom")}
        # Freeze previously inferred inertia before changing visual geometry.
        if body.find("inertial") is None and model.body_mass[body_id] > 0:
            ET.SubElement(body, "inertial", {
                "mass": _vector([model.body_mass[body_id]]),
                "pos": _vector(model.body_ipos[body_id]),
                "quat": _vector(model.body_iquat[body_id]),
                "diaginertia": _vector(model.body_inertia[body_id]),
            })
        for index, points in zip(visuals, parts):
            world = ((points - fitted.center) @ axes * scale) @ target_rotation.T + target_center
            local = (world - data.xpos[body_id]) @ body_rotation
            geom = geoms[model.geom(index).name]
            base = f"{name}_aligned_visual_{index}"
            unique = base
            suffix = 1
            while unique in existing_assets or (mesh_dir / f"{unique}.obj").exists():
                unique = f"{base}_{suffix}"
                suffix += 1
            existing_assets.add(unique)
            path = mesh_dir / f"{unique}.obj"
            _write_mesh(model, int(model.geom_dataid[index]), local, path)
            outputs.append(path.relative_to(scene_xml.parent).as_posix())
            ET.SubElement(asset, "mesh", {"name": unique, "file": outputs[-1], "scale": "1 1 1"})
            geom.set("mesh", unique)
            for attribute in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
                geom.attrib.pop(attribute, None)
            geom.set("pos", "0 0 0")
            geom.set("quat", "1 0 0 0")
    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="unicode")
    updated = mujoco.MjModel.from_xml_path(str(scene_xml))
    for field in ("qpos0", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
        if not np.allclose(getattr(model, field), getattr(updated, field), atol=1e-10):
            raise ValueError(f"visual fitting unexpectedly changed physical state: {field}")
    after = visual_collision_alignment(scene_xml, names)
    return {"before": before, "after": after, "written_meshes": outputs}
