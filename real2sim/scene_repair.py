"""Codex-agent fallback for unhealthy generated Real2Sim MJCF scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from real2sim.mujoco_export import (
    InitialObjectPenetrationError,
    compile_mujoco_scene,
    object_collision_geom_ids,
)
from real2sim.scene_bundle import SceneBundleSnapshot, scene_asset_hashes


DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_TURNS = 12
DEFAULT_STABILITY_SECONDS = 2.0
DEFAULT_MAX_HORIZONTAL_DRIFT_METERS = 0.04
DEFAULT_MAX_VERTICAL_DRIFT_METERS = 0.03
DEFAULT_MAX_ROTATION_DRIFT_DEGREES = 12.0
DEFAULT_MAX_FINAL_LINEAR_SPEED = 0.05
DEFAULT_MAX_FINAL_ANGULAR_SPEED = 0.5
SKILL_RELATIVE_PATH = Path(
    "real2sim/.agents/skills/repair-real2sim-scene/SKILL.md"
)


class SceneRepairAgentError(RuntimeError):
    """The scene-repair agent could not produce a validated scene."""


class SceneHealthError(SceneRepairAgentError):
    """The MJCF compiles but fails a physical scene-health check."""

    def __init__(self, message: str, report: dict[str, Any]):
        self.report = report
        super().__init__(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fmt(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _object_safe_names(root: ET.Element) -> list[str]:
    names: list[str] = []
    for body in root.findall("./worldbody/body"):
        name = str(body.get("name", "")).strip()
        if not name:
            continue
        if body.find(f"./geom[@name='{name}_collision']") is not None:
            names.append(name)
    return names


def _validate_edit_scope(before_xml: bytes, after_xml: bytes) -> None:
    """Allow object/mesh reconstruction while preserving the calibrated world."""

    before = ET.fromstring(before_xml)
    after = ET.fromstring(after_xml)
    before_names = set(_object_safe_names(before))
    after_names = set(_object_safe_names(after))
    if not after_names:
        raise SceneRepairAgentError("repair must retain at least one scene object")

    def signature(element: ET.Element) -> tuple:
        return (
            element.tag,
            sorted(element.attrib.items()),
            (element.text or "").strip(),
            tuple(signature(child) for child in element),
        )

    # Existing bodies retain their motion model; geometry may be replaced.
    old_bodies = {body.get("name"): body for body in before.findall("./worldbody/body")}
    new_bodies = {body.get("name"): body for body in after.findall("./worldbody/body")}
    for name in before_names & after_names:
        old, new = old_bodies[name], new_bodies[name]
        old_joints = [signature(item) for item in old if item.tag in {"joint", "freejoint"}]
        new_joints = [signature(item) for item in new if item.tag in {"joint", "freejoint"}]
        if old_joints != new_joints:
            raise SceneRepairAgentError(f"repair changed motion model for {name!r}")
        for attribute in ("gravcomp", "mocap", "childclass"):
            if old.get(attribute) != new.get(attribute):
                raise SceneRepairAgentError(f"repair changed {attribute} for {name!r}")
        old_geoms = {geom.get("name"): geom for geom in old.findall("geom")}
        for geom in new.findall("geom"):
            previous = old_geoms.get(geom.get("name"))
            if previous is None:
                continue
            for attribute in (
                "contype", "conaffinity", "friction", "solref",
                "solimp", "condim", "margin", "gap",
            ):
                if previous.get(attribute) != geom.get(attribute):
                    raise SceneRepairAgentError(f"repair changed contact setting {attribute}")

    # Existing shared materials/textures remain fixed; new object-specific
    # materials/textures can accompany added or replacement meshes.
    before_assets = before.find("asset")
    after_assets = after.find("asset")
    if before_assets is not None:
        for asset in before_assets:
            if asset.tag == "mesh":
                continue
            matches = [] if after_assets is None else [
                item for item in after_assets
                if item.tag == asset.tag and item.get("name") == asset.get("name")
            ]
            if len(matches) != 1 or signature(matches[0]) != signature(asset):
                raise SceneRepairAgentError("repair changed an existing non-mesh asset")

    for root, names in ((before, before_names), (after, after_names)):
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise SceneRepairAgentError("scene has no worldbody")
        for body in list(worldbody.findall("body")):
            if body.get("name") in names:
                if body.find("body") is not None:
                    raise SceneRepairAgentError("repaired objects must be root bodies without child bodies")
                worldbody.remove(body)
        asset = root.find("asset")
        if asset is not None:
            for item in list(asset):
                if item.tag in {"mesh", "material", "texture"}:
                    asset.remove(item)
            if not len(asset) and not asset.attrib:
                root.remove(asset)
    if signature(before) != signature(after):
        raise SceneRepairAgentError(
            "repair changed protected scene structure, floor, camera, or physics settings"
        )


def _validate_object_collisions(scene_xml: Path, names: list[str]) -> None:
    """Require physical collision geometry even when object meshes are replaced."""

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    for name in names:
        geom = model.geom(f"{name}_collision")
        if (
            model.geom_contype[geom.id] == 0
            or model.geom_conaffinity[geom.id] == 0
            or model.body(int(model.geom_bodyid[geom.id])).name != name
        ):
            raise SceneRepairAgentError(f"object {name!r} must retain active collision geometry")


def _validate_floor_clearance(
    scene_xml: Path,
    object_safe_names: list[str],
    *,
    max_floor_penetration_meters: float = 0.01,
) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    object_geom_ids = object_collision_geom_ids(model, object_safe_names)

    contacts: list[dict[str, Any]] = []
    if floor_id >= 0:
        for index in range(data.ncon):
            contact = data.contact[index]
            first_id = int(contact.geom[0])
            second_id = int(contact.geom[1])
            ids = {first_id, second_id}
            if floor_id not in ids or not (ids & object_geom_ids):
                continue
            object_id = next(iter(ids & object_geom_ids))
            contacts.append(
                {
                    "geom": model.geom(object_id).name,
                    "penetration_meters": max(0.0, -float(contact.dist)),
                }
            )
    contacts.sort(key=lambda item: float(item["penetration_meters"]), reverse=True)
    maximum = float(contacts[0]["penetration_meters"]) if contacts else 0.0
    return {
        "maximum_floor_penetration_meters": maximum,
        "floor_contact_count": len(contacts),
        "floor_penetration_threshold_meters": max_floor_penetration_meters,
        "status": (
            "ok" if maximum <= max_floor_penetration_meters else "failed"
        ),
    }


def _quaternion_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.dot(first, second))
    return float(math.degrees(2.0 * math.acos(np.clip(abs(dot), -1.0, 1.0))))


def _validate_dynamic_stability(
    scene_xml: Path,
    object_safe_names: list[str],
    *,
    duration_seconds: float = DEFAULT_STABILITY_SECONDS,
    max_horizontal_drift_meters: float = DEFAULT_MAX_HORIZONTAL_DRIFT_METERS,
    max_vertical_drift_meters: float = DEFAULT_MAX_VERTICAL_DRIFT_METERS,
    max_rotation_drift_degrees: float = DEFAULT_MAX_ROTATION_DRIFT_DEGREES,
    max_final_linear_speed: float = DEFAULT_MAX_FINAL_LINEAR_SPEED,
    max_final_angular_speed: float = DEFAULT_MAX_FINAL_ANGULAR_SPEED,
) -> dict[str, Any]:
    """Advance the unforced scene and reject objects that topple or disperse."""

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    for name in object_safe_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise SceneRepairAgentError(f"repaired scene is missing body {name!r}")
        joint_count = int(model.body_jntnum[body_id])
        joint_address = int(model.body_jntadr[body_id])
        if joint_count != 1 or joint_address < 0:
            continue
        if int(model.jnt_type[joint_address]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        initial[name] = (
            joint_address,
            np.asarray(data.xpos[body_id], dtype=float).copy(),
            np.asarray(data.xquat[body_id], dtype=float).copy(),
        )

    step_count = max(1, int(math.ceil(duration_seconds / model.opt.timestep)))
    for _ in range(step_count):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or any(warning.number for warning in data.warning)):
        report = {
            "status": "failed",
            "duration_seconds": float(step_count * model.opt.timestep),
            "step_count": step_count,
            "reason": "simulation produced invalid state or MuJoCo warnings",
            "failed_objects": list(initial),
            "objects": {},
        }
        return report

    thresholds = {
        "max_horizontal_drift_meters": max_horizontal_drift_meters,
        "max_vertical_drift_meters": max_vertical_drift_meters,
        "max_rotation_drift_degrees": max_rotation_drift_degrees,
        "max_final_linear_speed_meters_per_second": max_final_linear_speed,
        "max_final_angular_speed_radians_per_second": max_final_angular_speed,
    }
    objects: dict[str, dict[str, Any]] = {}
    failed_objects: list[str] = []
    for name, (joint_id, initial_pos, initial_quat) in initial.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        final_pos = np.asarray(data.xpos[body_id], dtype=float).copy()
        final_quat = np.asarray(data.xquat[body_id], dtype=float).copy()
        delta = final_pos - initial_pos
        dof_address = int(model.jnt_dofadr[joint_id])
        velocity = np.asarray(data.qvel[dof_address : dof_address + 6], dtype=float)
        metrics = {
            "initial_position": initial_pos.tolist(),
            "final_position": final_pos.tolist(),
            "position_delta": delta.tolist(),
            "horizontal_drift_meters": float(np.linalg.norm(delta[:2])),
            "vertical_drift_meters": abs(float(delta[2])),
            "rotation_drift_degrees": _quaternion_distance_degrees(
                initial_quat,
                final_quat,
            ),
            "final_linear_speed_meters_per_second": float(
                np.linalg.norm(velocity[:3])
            ),
            "final_angular_speed_radians_per_second": float(
                np.linalg.norm(velocity[3:])
            ),
        }
        failures = [
            key
            for key, threshold_key in (
                ("horizontal_drift_meters", "max_horizontal_drift_meters"),
                ("vertical_drift_meters", "max_vertical_drift_meters"),
                ("rotation_drift_degrees", "max_rotation_drift_degrees"),
                (
                    "final_linear_speed_meters_per_second",
                    "max_final_linear_speed_meters_per_second",
                ),
                (
                    "final_angular_speed_radians_per_second",
                    "max_final_angular_speed_radians_per_second",
                ),
            )
            if float(metrics[key]) > float(thresholds[threshold_key])
        ]
        metrics["failed_metrics"] = failures
        if failures:
            failed_objects.append(name)
        objects[name] = metrics

    return {
        "status": "failed" if failed_objects else "ok",
        "duration_seconds": float(step_count * model.opt.timestep),
        "step_count": step_count,
        "thresholds": thresholds,
        "failed_objects": failed_objects,
        "objects": objects,
    }


def validate_repaired_scene(
    scene_dir: str | Path,
    *,
    before_xml: bytes | None = None,
    object_safe_names: list[str] | tuple[str, ...] | None = None,
    max_initial_object_penetration_meters: float = 0.03,
) -> dict[str, Any]:
    """Validate edit scope, MuJoCo compilation, and initial scene health."""

    directory = Path(scene_dir).expanduser().resolve()
    scene_xml = directory / "scene.xml"
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Real2Sim scene is missing: {scene_xml}")

    after_xml = scene_xml.read_bytes()
    if before_xml is not None:
        _validate_edit_scope(before_xml, after_xml)

    parsed = ET.fromstring(after_xml)
    # The final XML is authoritative after additions/removals, even when
    # callers still hold the original reconstruction's object list.
    names = _object_safe_names(parsed)
    if before_xml is None and object_safe_names is not None:
        missing = set(object_safe_names) - set(names)
        if missing:
            raise SceneRepairAgentError(f"scene is missing expected objects: {sorted(missing)}")
    if not names:
        raise SceneRepairAgentError(
            "repaired scene has no reconstructed-object collision geoms"
        )

    report = compile_mujoco_scene(
        scene_xml,
        object_safe_names=names,
        max_initial_object_penetration_meters=(
            max_initial_object_penetration_meters
        ),
    )
    initial_contact_status = report["status"]
    _validate_object_collisions(scene_xml, names)
    floor = _validate_floor_clearance(scene_xml, names)
    stability = _validate_dynamic_stability(scene_xml, names)
    report["initial_contact_status"] = initial_contact_status
    report["floor_clearance"] = floor
    report["stability"] = stability
    report["validated_object_safe_names"] = names
    if floor["status"] != "ok":
        report["status"] = "failed"
        report["failure_kind"] = "floor_penetration"
        raise SceneHealthError(
            "scene health check failed: reconstructed objects penetrate the floor "
            f"by {floor['maximum_floor_penetration_meters']:.6g} m",
            report,
        )
    if stability["status"] != "ok":
        report["status"] = "failed"
        report["failure_kind"] = "dynamic_stability"
        failed = ", ".join(stability["failed_objects"]) or "unknown objects"
        raise SceneHealthError(
            "scene collapses during the unforced settling rollout; failed objects: "
            f"{failed}",
            report,
        )
    from real2sim.visual_alignment import visual_collision_alignment

    alignment = visual_collision_alignment(scene_xml, names)
    report["visual_collision_alignment"] = alignment
    if alignment["status"] != "ok":
        report["status"] = "failed"
        report["failure_kind"] = "visual_collision_alignment"
        raise SceneHealthError(
            "visual meshes disagree with collision geometry: "
            + ", ".join(alignment["failed_objects"]),
            report,
        )
    return report


def render_source_camera(
    scene_dir: str | Path,
    output_path: str | Path,
    *,
    step_count: int = 0,
    geometry: str = "visual",
) -> Path:
    """Render scene.xml from the calibrated source RGB camera."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio.v2 as imageio
    import mujoco

    directory = Path(scene_dir).expanduser().resolve()
    scene_xml = directory / "scene.xml"
    camera_npz = directory / "depth_camera.npz"
    input_image = directory / "input.png"
    if not camera_npz.is_file():
        raise FileNotFoundError(f"camera bundle is missing: {camera_npz}")
    if not input_image.is_file():
        raise FileNotFoundError(f"processing image is missing: {input_image}")

    with np.load(camera_npz) as camera:
        c2w = np.asarray(camera["c2w_opencv"], dtype=float)
        intrinsics = np.asarray(camera["intrinsics"], dtype=float)
    if c2w.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("depth_camera.npz has invalid camera matrices")

    with Image.open(input_image) as image:
        width, height = image.size
    root = ET.parse(scene_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(directory))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {scene_xml}")
    camera_name = "real2sim_agent_source_camera"
    old_camera = worldbody.find(f"./camera[@name='{camera_name}']")
    if old_camera is not None:
        worldbody.remove(old_camera)

    camera_rotation = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, camera_rotation.reshape(-1))
    fy = float(intrinsics[1, 1])
    if not math.isfinite(fy) or fy <= 0:
        raise ValueError("camera fy must be finite and positive")
    fovy = math.degrees(2.0 * math.atan(float(height) / (2.0 * fy)))
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": camera_name,
            "mode": "fixed",
            "pos": _fmt(c2w[:3, 3]),
            "quat": _fmt(quaternion),
            "fovy": f"{fovy:.9g}",
        },
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, int(step_count))):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    option = mujoco.MjvOption()
    if geometry not in {"visual", "collision", "overlay"}:
        raise ValueError(f"unknown geometry view: {geometry}")
    names = _object_safe_names(root)
    collision_ids = object_collision_geom_ids(model, names)
    for index in range(model.ngeom):
        if model.body(int(model.geom_bodyid[index])).name not in names:
            continue
        model.geom_group[index] = 3 if index in collision_ids else 2
        if index in collision_ids and geometry != "visual":
            model.geom_matid[index] = -1
            model.geom_rgba[index] = [0.1, 0.7, 1.0, 0.3 if geometry == "overlay" else 1.0]
    option.geomgroup[2] = geometry != "collision"
    option.geomgroup[3] = geometry != "visual"
    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        renderer.update_scene(data, camera=camera_name, scene_option=option)
        pixels = renderer.render()
    finally:
        renderer.close()

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output, pixels)
    return output


def _build_agent_prompt(
    *,
    scene_dir: Path,
    source_image: Path,
    skill_path: Path,
    error_summary: str,
    threshold: float,
    max_turns: int,
) -> str:
    repo_root = _repo_root()
    return f"""Use $repair-real2sim-scene at {skill_path}.

Repair the generated Real2Sim scene in {scene_dir}. The source image is
{source_image}; it is attached to this request. The generated MJCF is
{scene_dir / 'scene.xml'}. The failed preflight report is
{scene_dir / 'physics_preflight.json'}.

Failure: {error_summary}

Align the complete scene as closely as the evidence supports to the real camera
image, while keeping it physically healthy. You may edit scene.xml and existing
scene mesh assets, reshape or replace meshes, add missed objects, and remove
false-positive or duplicate objects. Write new mesh/material/texture files under
{scene_dir / 'meshes'}. Compare silhouettes, dimensions, object count, relative
placement, occlusion, appearance, and support contacts from the calibrated camera.
Do not stop at merely passing the physical checks.
After collision edits, complete a visual alignment pass. The validator now
checks visual/collision envelopes as well as physics. A pixel-identical render
before and after collision repair does not establish that the mesh matches the
corrected object's geometry. Inspect the visual/collision overlay and the real
RGB, and transform all visual color/texture parts of an object together.

Keep stable names for retained objects. Each object must be a direct worldbody
body with an active geom named <body_name>_collision; additional collision geoms
may represent compound shapes. New movable objects need a freejoint. Preserve
the calibrated camera/world/floor, global physics settings, and retained objects'
motion/contact settings. Keep raw RGB, depth, masks, and repository code unchanged.
The launcher updates metadata from the final XML, backs up scene and mesh files,
and restores them if independent validation fails.

Use this repository command for deterministic validation:

{repo_root / 'scripts/rapid-python'} -m real2sim.scene_repair validate \
  --scene-dir {scene_dir} --baseline {scene_dir / 'agent_repair/scene.before.xml'} \
  --max-initial-object-penetration {threshold:.9g}

Use this command to render the current XML from the calibrated source camera:

{repo_root / 'scripts/rapid-python'} -m real2sim.scene_repair render \
  --scene-dir {scene_dir} --output {scene_dir / 'agent_repair/source_camera_after.png'}

Inspect visual/collision agreement with:

{repo_root / 'scripts/rapid-python'} -m real2sim.scene_repair render \
  --scene-dir {scene_dir} --geometry overlay \
  --output {scene_dir / 'agent_repair/source_camera_overlay.png'}

When the collision box already matches the real object's pose and dimensions,
fit ALL visual parts to it while preserving the body's physics with:

{repo_root / 'scripts/rapid-python'} -m real2sim.scene_repair align-visuals \
  --scene-dir {scene_dir} --objects BODY_NAME [BODY_NAME ...]

This helper fits one affine transform per object to a single box. Review its
result against RGB; compound collision shapes or non-box objects require manual
mesh correction. Do not shrink an already correct collision to hide mesh error.

Also render after the required two-second unforced stability rollout:

{repo_root / 'scripts/rapid-python'} -m real2sim.scene_repair render \
  --scene-dir {scene_dir} --steps 400 \
  --output {scene_dir / 'agent_repair/source_camera_settled.png'}

Finish after validation passes and you have inspected both calibrated renders
against the real image. Explain the visual evidence for added/deleted objects
and mesh changes, and disclose any remaining alignment errors or uncertain
occluded geometry. The launcher requests at most about
{max_turns} substantial inspection/edit/validation cycles.
"""


def _write_agent_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def synchronize_scene_metadata(directory: Path) -> dict[str, Any]:
    """Publish the final object inventory; retain perception results as history."""

    import mujoco

    from real2sim.adapter import _body_mesh_bounds

    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    original = metadata.setdefault("reconstruction_objects", metadata.get("objects", []))
    labels = {
        str(item.get("safe_name", item.get("body", ""))): item
        for item in original if isinstance(item, dict)
    }
    root = ET.parse(directory / "scene.xml").getroot()
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mesh_files = {mesh.get("name"): mesh.get("file") for mesh in root.findall("./asset/mesh")}
    bodies = {body.get("name"): body for body in root.findall("./worldbody/body")}
    objects = []
    for name in _object_safe_names(root):
        body_id = model.body(name).id
        body = bodies[name]
        bounds = _body_mesh_bounds(model, data, body_id)
        objects.append({
            "body": name, "safe_name": name,
            "name": labels.get(name, {}).get("name", name.replace("_", " ")),
            "body_pos": data.xpos[body_id].tolist(),
            "body_quat": data.xquat[body_id].tolist(),
            "world_bounds": None if bounds is None else bounds.tolist(),
            "is_static": int(model.body_jntnum[body_id]) == 0,
            "mesh_files": sorted({
                mesh_files[geom.get("mesh")]
                for geom in body.findall("geom") if mesh_files.get(geom.get("mesh"))
            }),
            "pose_source": "scene_repair",
        })
    metadata["objects"] = objects
    if isinstance(metadata.get("mujoco"), dict):
        metadata["mujoco"]["collision_geometry"] = "scene_xml"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _run_repair_process(
    command: list[str], directory: Path, prompt: str, events_path: Path,
) -> tuple[str | None, int]:
    """Run the repair process, terminating it before rollback on interruption."""

    process = subprocess.Popen(
        command,
        cwd=directory,
        env={**os.environ, "RAPID_SCENE_REPAIR_AGENT": "1"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()

        thread_id: str | None = None
        assert process.stdout is not None
        with events_path.open("w", encoding="utf-8") as events:
            for line in process.stdout:
                events.write(line)
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    print(f"[real2sim.agent] {text}")
                    continue
                if isinstance(event, dict) and event.get("thread_id"):
                    thread_id = str(event["thread_id"])
                event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
                item = event.get("item") if isinstance(event, dict) else None
                if event_type in {
                    "thread.started",
                    "turn.completed",
                    "turn.failed",
                    "error",
                }:
                    print(f"[real2sim.agent] {event_type}")
                elif event_type == "item.started" and isinstance(item, dict):
                    if item.get("type") == "command_execution":
                        command_text = " ".join(str(item.get("command", "")).split())
                        print(f"[real2sim.agent] command: {command_text[:220]}")

        returncode = process.wait()
        return thread_id, returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def repair_scene_with_agent(
    scene_dir: str | Path,
    *,
    source_image: str | Path | None = None,
    error: BaseException | str | None = None,
    object_safe_names: list[str] | tuple[str, ...] | None = None,
    max_initial_object_penetration_meters: float = 0.03,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    """Launch Codex to repair one generated scene, then independently validate it."""

    directory = Path(scene_dir).expanduser().resolve()
    scene_xml = directory / "scene.xml"
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Real2Sim scene is missing: {scene_xml}")
    image_path = (
        Path(source_image).expanduser().resolve()
        if source_image is not None
        else (
            directory / "input_original.png"
            if (directory / "input_original.png").is_file()
            else directory / "input.png"
        )
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Real2Sim source image is missing: {image_path}")

    skill_path = _repo_root() / SKILL_RELATIVE_PATH
    if not skill_path.is_file():
        raise FileNotFoundError(
            f"scene-repair agent definition is missing: {skill_path}"
        )
    codex = shutil.which("codex")
    if codex is None:
        raise SceneRepairAgentError(
            "`codex` CLI was not found on PATH; cannot run Real2Sim agent fallback"
        )

    work_dir = directory / "agent_repair"
    work_dir.mkdir(parents=True, exist_ok=True)
    before_xml = scene_xml.read_bytes()
    before_path = work_dir / "scene.before.xml"
    before_path.write_bytes(before_xml)
    error_summary = "unknown scene validation failure" if error is None else str(error)
    try:
        validate_repaired_scene(
            directory,
            object_safe_names=object_safe_names,
            max_initial_object_penetration_meters=(
                max_initial_object_penetration_meters
            ),
        )
    except (InitialObjectPenetrationError, SceneHealthError) as baseline_error:
        error_summary = str(baseline_error)
        (directory / "physics_preflight.json").write_text(
            json.dumps(baseline_error.report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report_path = work_dir / "report.json"
    _write_agent_report(
        report_path,
        {
            "agent": "repair-real2sim-scene",
            "status": "running",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "source_image": str(image_path),
            "scene_xml": str(scene_xml),
            "before_sha256": _sha256(before_xml),
            "initial_error": error_summary,
        },
    )
    prompt = _build_agent_prompt(
        scene_dir=directory,
        source_image=image_path,
        skill_path=skill_path,
        error_summary=error_summary,
        threshold=max_initial_object_penetration_meters,
        max_turns=max(1, int(max_turns)),
    )
    (work_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    render_error: str | None = None
    settled_render_error: str | None = None
    try:
        render_source_camera(directory, work_dir / "source_camera_before.png")
    except Exception as exc:  # Rendering is useful evidence, not a repair gate.
        render_error = f"{type(exc).__name__}: {exc}"
    try:
        render_source_camera(
            directory,
            work_dir / "source_camera_before_settled.png",
            step_count=400,
        )
    except Exception as exc:  # Rendering is useful evidence, not a repair gate.
        settled_render_error = f"{type(exc).__name__}: {exc}"

    final_path = work_dir / "final.md"
    events_path = work_dir / "events.jsonl"
    command = [
        codex,
        "exec",
        "--json",
        "--model",
        model,
        "--cd",
        str(directory),
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        'model_reasoning_summary="concise"',
        "-c",
        'service_tier="fast"',
        "-c",
        "features.fast_mode=true",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "--image",
        str(image_path),
        "--output-last-message",
        str(final_path),
        "-",
    ]

    print(f"[real2sim] scene-repair agent: {skill_path}")
    print(f"[real2sim] scene-repair image: {image_path}")
    snapshot = SceneBundleSnapshot(directory, work_dir / "bundle.before")
    before_assets = scene_asset_hashes(directory)
    try:
        thread_id, returncode = _run_repair_process(command, directory, prompt, events_path)
    except BaseException as exc:
        snapshot.restore()
        _write_agent_report(report_path, {
            "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
            "before_sha256": _sha256(before_xml),
        })
        raise
    base_report: dict[str, Any] = {
        "agent": "repair-real2sim-scene",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "thread_id": thread_id,
        "source_image": str(image_path),
        "scene_xml": str(scene_xml),
        "before_sha256": _sha256(before_xml),
        "initial_error": error_summary,
        "source_camera_before_render_error": render_error,
        "source_camera_before_settled_render_error": settled_render_error,
        "codex_returncode": returncode,
    }

    if returncode != 0:
        failed_xml = scene_xml.read_bytes() if scene_xml.is_file() else b""
        if failed_xml and failed_xml != before_xml:
            (work_dir / "scene.failed.xml").write_bytes(failed_xml)
        snapshot.restore()
        base_report.update({"status": "failed", "reason": "Codex exited nonzero"})
        _write_agent_report(report_path, base_report)
        raise SceneRepairAgentError(
            f"scene-repair agent exited with status {returncode}; see {events_path}"
        )

    try:
        snapshot.validate_asset_locations()
        validation = validate_repaired_scene(
            directory,
            before_xml=before_xml,
            object_safe_names=object_safe_names,
            max_initial_object_penetration_meters=(
                max_initial_object_penetration_meters
            ),
        )
        after_assets = scene_asset_hashes(directory)
        before_names = set(_object_safe_names(ET.fromstring(before_xml)))
        after_names = set(validation["validated_object_safe_names"])
        base_report["object_changes"] = {
            "added": sorted(after_names - before_names),
            "removed": sorted(before_names - after_names),
            "retained": sorted(before_names & after_names),
        }
        base_report["before_asset_sha256"] = before_assets
        base_report["after_asset_sha256"] = after_assets
        base_report["changed_assets"] = sorted(
            path for path in before_assets.keys() | after_assets.keys()
            if before_assets.get(path) != after_assets.get(path)
        )
        synchronize_scene_metadata(directory)
    except Exception as exc:
        failed_xml = scene_xml.read_bytes() if scene_xml.is_file() else b""
        if failed_xml:
            (work_dir / "scene.failed.xml").write_bytes(failed_xml)
        snapshot.restore()
        base_report.update(
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "failed_sha256": _sha256(failed_xml) if failed_xml else None,
            }
        )
        _write_agent_report(report_path, base_report)
        raise SceneRepairAgentError(
            f"scene-repair agent produced an invalid scene: {exc}; see {report_path}"
        ) from exc

    after_xml = scene_xml.read_bytes()
    after_render_error: str | None = None
    after_settled_render_error: str | None = None
    try:
        render_source_camera(directory, work_dir / "source_camera_after.png")
    except Exception as exc:  # Keep a valid repair even if EGL rendering fails.
        after_render_error = f"{type(exc).__name__}: {exc}"
    try:
        render_source_camera(
            directory,
            work_dir / "source_camera_settled.png",
            step_count=400,
        )
    except Exception as exc:  # Keep a valid repair even if EGL rendering fails.
        after_settled_render_error = f"{type(exc).__name__}: {exc}"
    base_report.update(
        {
            "status": "ok",
            "after_sha256": _sha256(after_xml),
            "source_camera_after_render_error": after_render_error,
            "source_camera_settled_render_error": after_settled_render_error,
            "validation": validation,
        }
    )
    _write_agent_report(report_path, base_report)
    (directory / "physics_preflight.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[real2sim] scene-repair agent validated scene: "
        f"max penetration={validation['maximum_penetration_meters']:.6g} m"
    )
    return validation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate the Real2Sim scene-repair agent fallback."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--scene-dir", type=Path, required=True)
    validate.add_argument("--baseline", type=Path, default=None)
    validate.add_argument(
        "--max-initial-object-penetration",
        type=float,
        default=0.03,
    )

    render = subparsers.add_parser("render")
    render.add_argument("--scene-dir", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--steps", type=int, default=0)
    render.add_argument("--geometry", choices=("visual", "collision", "overlay"), default="visual")

    align = subparsers.add_parser("align-visuals")
    align.add_argument("--scene-dir", type=Path, required=True)
    align.add_argument("--objects", nargs="+", required=True)

    repair = subparsers.add_parser("repair")
    repair.add_argument("--scene-dir", type=Path, required=True)
    repair.add_argument("--source-image", type=Path, default=None)
    repair.add_argument("--error", default=None)
    repair.add_argument("--model", default=DEFAULT_MODEL)
    repair.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
    )
    repair.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    repair.add_argument(
        "--max-initial-object-penetration",
        type=float,
        default=0.03,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            baseline = args.baseline.read_bytes() if args.baseline is not None else None
            report = validate_repaired_scene(
                args.scene_dir,
                before_xml=baseline,
                max_initial_object_penetration_meters=(
                    max(0.0, args.max_initial_object_penetration)
                ),
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "render":
            print(
                render_source_camera(
                    args.scene_dir,
                    args.output,
                    step_count=max(0, args.steps),
                    geometry=args.geometry,
                )
            )
        elif args.command == "align-visuals":
            from real2sim.visual_alignment import fit_visuals_to_collision

            directory = args.scene_dir.expanduser().resolve()
            snapshot = SceneBundleSnapshot(directory, directory / "agent_repair/visual_fit.before")
            try:
                report = fit_visuals_to_collision(directory / "scene.xml", args.objects)
                if report["after"]["status"] != "ok":
                    raise SceneRepairAgentError("visual fitting did not satisfy the alignment check")
            except BaseException:
                snapshot.restore()
                raise
            (directory / "agent_repair/visual_fit.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            report = repair_scene_with_agent(
                args.scene_dir,
                source_image=args.source_image,
                error=args.error,
                max_initial_object_penetration_meters=(
                    max(0.0, args.max_initial_object_penetration)
                ),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_turns=max(1, args.max_turns),
            )
            print(json.dumps(report, indent=2, sort_keys=True))
    except (InitialObjectPenetrationError, SceneHealthError) as exc:
        if args.command == "validate":
            report_path = args.scene_dir.expanduser().resolve() / "physics_preflight.json"
            report_path.write_text(
                json.dumps(exc.report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(exc.report, indent=2, sort_keys=True), file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
