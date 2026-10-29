"""Package real2sim outputs as sanitized task scene modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from real2sim.adapter import Real2SimTaskConfig, make_default_camera
from real2sim.mujoco_export import (
    _box_diagonal_inertia,
    _fmt_vec,
    _obb_3d_from_vertices,
    _quat_to_rotation_matrix,
    _verticalized_box_body_pose,
    compile_mujoco_scene,
)


OBJ_MTL_RE = re.compile(r"^\s*mtllib\s+(.+?)\s*$")
MTL_TEXTURE_KEYS = {
    "map_Ka",
    "map_Kd",
    "map_Ks",
    "map_Ke",
    "map_Ns",
    "map_d",
    "map_bump",
    "map_Bump",
    "bump",
    "disp",
    "decal",
    "refl",
    "norm",
}

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
DEFAULT_FR3_BASE_POS = (0.0, 0.0, 0.0)
DEFAULT_FR3_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _snake_identifier(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if not name:
        raise ValueError(f"could not derive a task name from {value!r}")
    if name[0].isdigit():
        name = f"scene_{name}"
    if not name.isidentifier():
        raise ValueError(f"not a valid Python package name: {name!r}")
    return name


def _validate_source(source_dir: Path) -> None:
    scene_xml = source_dir / "scene.xml"
    if not scene_xml.is_file():
        raise FileNotFoundError(f"real2sim output is missing: {scene_xml}")
    from real2sim.scene_repair import _object_safe_names

    object_names = _object_safe_names(ET.parse(scene_xml).getroot())
    threshold = 0.03
    metadata_path = source_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            threshold = float(
                metadata.get("mujoco", {}).get(
                    "max_initial_object_penetration_meters",
                    threshold,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    compile_mujoco_scene(
        scene_xml,
        object_safe_names=object_names,
        max_initial_object_penetration_meters=threshold,
    )


def _copy_runtime_bundle(
    source_dir: Path,
    bundle_dir: Path,
    *,
    task_name: str,
) -> dict[str, object]:
    camera = make_default_camera(Real2SimTaskConfig(scene_dir=source_dir))
    tmp_dir = bundle_dir.with_name(f".{bundle_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        scene_root = ET.parse(source_dir / "scene.xml").getroot()
        scene_root.set("model", task_name)
        _replace_mesh_collisions_with_obbs(scene_root, source_dir)
        asset_map = _collect_referenced_assets(source_dir, scene_root)
        _write_scene_xml(scene_root, tmp_dir / "scene.xml")
        _copy_asset_map(asset_map, tmp_dir)
        from real2sim.settle_scene import settle_scene

        print(f"[real2sim] settling scene before packaging: {task_name}", flush=True)
        settling = settle_scene(tmp_dir / "scene.xml")
        (tmp_dir / "settling.json").write_text(
            json.dumps(settling, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[real2sim] saved resting poses for {len(settling['objects'])} objects "
            f"after {settling['duration_seconds']:.3f}s of simulation"
        )
        _write_label_metadata(source_dir, scene_root, tmp_dir / "metadata.json")
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        tmp_dir.rename(bundle_dir)
        return camera
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _replace_mesh_collisions_with_obbs(
    scene_root: ET.Element,
    source_dir: Path,
) -> None:
    """Canonicalize reconstructed object collisions before packaging."""

    # A successful repair has independently validated the final object inventory
    # and geometry, including edited mesh assets and compound collision shapes.
    # Refitting from the original reconstruction mesh here would silently undo
    # repaired collision sizes/local poses (and static-body corrections).
    if _has_validated_scene_repair(source_dir):
        return

    asset = scene_root.find("asset")
    worldbody = scene_root.find("worldbody")
    if asset is None or worldbody is None:
        return
    mesh_assets = {
        str(mesh.get("name")): mesh
        for mesh in asset.findall("mesh")
        if mesh.get("name") and mesh.get("file")
    }
    compiler = scene_root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    obb_source_files = _obb_source_files(source_dir / "metadata.json")

    for body in worldbody.findall("./body"):
        body_name = str(body.get("name", ""))
        geoms = list(body.findall("geom"))
        collision_geoms = [geom for geom in geoms if _is_collision_geom(geom)]
        if not collision_geoms:
            continue

        source_file = obb_source_files.get(body_name)
        if source_file is not None:
            source_mesh = trimesh.load_mesh(source_dir / source_file, process=False)
            point_sets = [np.asarray(source_mesh.vertices, dtype=float)]
        else:
            mesh_geoms = [geom for geom in geoms if geom.get("type") == "mesh"]
            mesh_collision_geoms = [
                geom for geom in mesh_geoms if geom in collision_geoms
            ]
            if not mesh_collision_geoms:
                continue
            visual_geoms = [geom for geom in mesh_geoms if geom not in collision_geoms]
            source_geoms = visual_geoms or mesh_collision_geoms
            point_sets = [
                points
                for geom in source_geoms
                if (points := _mesh_geom_vertices(geom, mesh_assets, source_dir, mesh_dir))
                is not None
            ]
        if not point_sets:
            raise ValueError(f"cannot fit collision OBB for body {body_name!r}")
        collision_box = _obb_3d_from_vertices(np.vstack(point_sets))

        collision = collision_geoms[0]
        for key in ("mesh", "fromto", "axisangle", "euler", "xyaxes", "zaxis"):
            collision.attrib.pop(key, None)
        collision.set("type", "box")
        collision.set("pos", _fmt_vec(collision_box.center))
        collision.set("quat", _fmt_vec(collision_box.quat))
        collision.set("size", _fmt_vec(collision_box.half_extents))
        for extra in collision_geoms[1:]:
            body.remove(extra)

        if body.find("joint") is None and body.find("freejoint") is None:
            body_pos = _float_vector(body.get("pos"), default=[0.0, 0.0, 0.0])
            body_quat = _float_vector(
                body.get("quat"),
                default=[1.0, 0.0, 0.0, 0.0],
            )
            corrected_pos, corrected_quat = _verticalized_box_body_pose(
                body_pos,
                _quat_to_rotation_matrix(body_quat),
                collision_box,
            )
            for key in ("axisangle", "euler", "xyaxes", "zaxis"):
                body.attrib.pop(key, None)
            body.set("pos", _fmt_vec(corrected_pos))
            body.set("quat", _fmt_vec(corrected_quat))

        inertial = body.find("inertial")
        if inertial is not None:
            mass = float(inertial.get("mass", "0.2"))
            inertial.attrib.pop("fullinertia", None)
            inertial.set("pos", _fmt_vec(collision_box.center))
            inertial.set("quat", _fmt_vec(collision_box.quat))
            inertial.set(
                "diaginertia",
                _fmt_vec(_box_diagonal_inertia(collision_box.half_extents, mass)),
            )

    used_meshes = {
        str(geom.get("mesh"))
        for geom in worldbody.findall(".//geom[@mesh]")
        if geom.get("mesh")
    }
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used_meshes:
            asset.remove(mesh)


def _has_validated_scene_repair(source_dir: Path) -> bool:
    report_path = source_dir / "agent_repair" / "report.json"
    scene_xml = source_dir / "scene.xml"
    if not report_path.is_file() or not scene_xml.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scene_sha256 = hashlib.sha256(scene_xml.read_bytes()).hexdigest()
        if "after_asset_sha256" in report:
            from real2sim.scene_bundle import scene_asset_hashes

            if report["after_asset_sha256"] != scene_asset_hashes(source_dir):
                return False
    except (OSError, ValueError, ET.ParseError):
        return False
    validation = report.get("validation")
    return (
        report.get("status") == "ok"
        and report.get("after_sha256") == scene_sha256
        and isinstance(validation, dict)
        and validation.get("status") == "ok"
    )


def _obb_source_files(metadata_json: Path) -> dict[str, Path]:
    if not metadata_json.is_file():
        return {}
    try:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        return {}

    sources: dict[str, Path] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or item.get("safe_name") or "").strip()
        source = str(
            item.get("obb_source_mesh_file")
            or item.get("collision_mesh_file")
            or ""
        ).strip()
        if body and source:
            sources[body] = Path(source)
    return sources


def _is_collision_geom(geom: ET.Element) -> bool:
    try:
        contype = int(geom.get("contype", "1"))
        conaffinity = int(geom.get("conaffinity", "1"))
    except ValueError:
        return True
    return contype != 0 or conaffinity != 0


def _mesh_geom_vertices(
    geom: ET.Element,
    mesh_assets: dict[str, ET.Element],
    source_dir: Path,
    mesh_dir: Path,
) -> np.ndarray | None:
    mesh_asset = mesh_assets.get(str(geom.get("mesh", "")))
    if mesh_asset is None:
        return None
    file_value = str(mesh_asset.get("file", "")).strip()
    if not file_value:
        return None
    mesh_path = Path(file_value)
    if not mesh_path.is_absolute():
        mesh_path = source_dir / mesh_dir / mesh_path
    loaded = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(loaded.vertices, dtype=float)
    if vertices.size == 0:
        return None

    scale = _float_vector(mesh_asset.get("scale"), default=[1.0, 1.0, 1.0])
    vertices = vertices * scale
    quat = _float_vector(geom.get("quat"), default=[1.0, 0.0, 0.0, 0.0])
    rotation = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
    pos = _float_vector(geom.get("pos"), default=[0.0, 0.0, 0.0])
    return vertices @ rotation.T + pos


def _float_vector(value: str | None, *, default: list[float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    parsed = np.fromstring(value, sep=" ", dtype=float)
    if parsed.shape != (len(default),) or not np.isfinite(parsed).all():
        raise ValueError(f"invalid MJCF vector: {value!r}")
    return parsed


def _collect_referenced_assets(
    source_dir: Path,
    scene_root: ET.Element,
) -> dict[Path, Path]:
    assets: dict[Path, Path] = {}
    pending: list[tuple[Path, Path]] = []

    for elem in scene_root.findall(".//*[@file]"):
        value = str(elem.get("file", "")).strip()
        if not value:
            continue
        source_path, bundle_rel = _resolve_asset_reference(source_dir, value)
        elem.set("file", bundle_rel.as_posix())
        pending.append((source_path, bundle_rel))

    while pending:
        source_path, bundle_rel = pending.pop()
        source_path, bundle_rel = _resolve_asset_reference(source_dir, str(source_path))
        if source_path in assets:
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"referenced asset not found: {source_path}")
        assets[source_path] = bundle_rel
        suffix = source_path.suffix.lower()
        if suffix == ".obj":
            pending.extend(_obj_material_references(source_path, bundle_rel))
        elif suffix == ".mtl":
            pending.extend(_mtl_texture_references(source_path, bundle_rel))

    return assets


def _resolve_asset_reference(source_dir: Path, value: str) -> tuple[Path, Path]:
    path = Path(value)
    source_path = (path if path.is_absolute() else source_dir / path).resolve()
    try:
        bundle_rel = source_path.relative_to(source_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"asset reference escapes real2sim output directory: {value}"
        ) from exc
    return source_path, Path(bundle_rel.as_posix())


def _obj_material_references(obj_path: Path, obj_bundle_rel: Path) -> list[tuple[Path, Path]]:
    references: list[tuple[Path, Path]] = []
    try:
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return references
    for line in lines:
        match = OBJ_MTL_RE.match(line)
        if not match:
            continue
        for raw in match.group(1).split():
            source = (obj_path.parent / raw).resolve()
            bundle_rel = obj_bundle_rel.parent / raw
            references.append((source, bundle_rel))
    return references


def _mtl_texture_references(mtl_path: Path, mtl_bundle_rel: Path) -> list[tuple[Path, Path]]:
    references: list[tuple[Path, Path]] = []
    try:
        lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return references
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2 or parts[0] not in MTL_TEXTURE_KEYS:
            continue
        texture = _last_mtl_path_token(parts[1:])
        if texture is None:
            continue
        source = (mtl_path.parent / texture).resolve()
        bundle_rel = mtl_bundle_rel.parent / texture
        references.append((source, bundle_rel))
    return references


def _last_mtl_path_token(tokens: list[str]) -> str | None:
    for token in reversed(tokens):
        if token.startswith("-"):
            continue
        return token
    return None


def _write_scene_xml(scene_root: ET.Element, output_path: Path) -> None:
    ET.indent(scene_root, space="  ")
    output_path.write_text(
        ET.tostring(scene_root, encoding="unicode") + "\n",
        encoding="utf-8",
    )


def _relativize_asset_files(root: ET.Element, scene_dir: Path) -> None:
    scene_dir = scene_dir.resolve()
    for elem in root.findall(".//*[@file]"):
        value = str(elem.get("file", "")).strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            continue
        try:
            elem.set("file", path.resolve().relative_to(scene_dir).as_posix())
        except ValueError:
            elem.set("file", path.as_posix())


def _copy_asset_map(asset_map: dict[Path, Path], bundle_dir: Path) -> None:
    for source, rel in sorted(asset_map.items(), key=lambda item: item[1].as_posix()):
        dest = bundle_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def _write_label_metadata(
    source_dir: Path,
    scene_root: ET.Element,
    output_path: Path,
) -> None:
    labels = _labels_from_metadata(source_dir / "metadata.json")
    bodies = _scene_object_bodies(scene_root)
    objects = []
    for body in bodies:
        label = labels.get(body, {})
        safe_name = str(label.get("safe_name", body)).strip() or body
        name = str(label.get("name", safe_name)).strip() or safe_name
        objects.append({"body": body, "safe_name": safe_name, "name": name})
    output_path.write_text(
        json.dumps({"objects": objects}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _labels_from_metadata(metadata_json: Path) -> dict[str, dict[str, str]]:
    if not metadata_json.is_file():
        return {}
    try:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

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


def _scene_object_bodies(scene_root: ET.Element) -> list[str]:
    bodies: list[str] = []
    for body in scene_root.findall("./worldbody/body"):
        name = str(body.get("name", "")).strip()
        if not name:
            continue
        if body.find("geom") is None and body.find(".//geom") is None:
            continue
        bodies.append(name)
    return bodies


def _scene_py(
    package_name: str,
    default_camera: dict[str, object],
) -> str:
    return f'''"""Task scene for the packaged real2sim {package_name} bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from scene_io.runtime_assets import fr3_task_v1_xml


_PACKAGE_NAME = {package_name!r}
_PACKAGE_DIR = Path(__file__).resolve().parent
SCENE_DIR = _PACKAGE_DIR / "bundle"
SCENE_XML = SCENE_DIR / "scene.xml"
FR3_XML = fr3_task_v1_xml()
CONFIG: dict[str, Any] = {{
    "scene_dir": SCENE_DIR,
    "scene_xml": SCENE_XML,
    "fr3_xml": FR3_XML,
    "task_name": f"real2sim_{{_PACKAGE_NAME}}",
}}

DEFAULT_CAMERA: dict[str, Any] = {_camera_literal(default_camera)}
FR3_HOME_JOINT_QPOS: dict[str, float] = {FR3_HOME_JOINT_QPOS!r}
FR3_BASE_POS = {DEFAULT_FR3_BASE_POS!r}
FR3_BASE_QUAT = {DEFAULT_FR3_BASE_QUAT!r}
FR3_MOUNT_FRAME = "fr3_mount"


def make_spec() -> mujoco.MjSpec:
    """Compose the reconstruction and standalone FR3 submodel."""

    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    robot = mujoco.MjSpec.from_file(str(FR3_XML))
    home = robot.add_key(name="home")
    home.qpos = list(FR3_HOME_JOINT_QPOS.values())
    home.ctrl = [
        FR3_HOME_JOINT_QPOS[f"fr3_joint{{index}}"] for index in range(1, 8)
    ] + [FR3_HOME_JOINT_QPOS["finger_joint1"]]
    mount = scene.worldbody.add_frame(
        name=FR3_MOUNT_FRAME,
        pos=FR3_BASE_POS,
        quat=FR3_BASE_QUAT,
    )
    # Passing the empty prefix preserves the canonical FR3 names. Omitting the
    # argument makes MjSpec add a leading slash to every attached name.
    scene.attach(robot, prefix="", frame=mount)
    return scene


def make_model() -> mujoco.MjModel:
    """Return the compiled reconstruction-plus-FR3 model."""

    return make_spec().compile()


def make_data(model: mujoco.MjModel | None = None) -> mujoco.MjData:
    """Return data with the FR3 at home and its gripper open."""

    model = make_model() if model is None else model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise KeyError("missing FR3 home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return data


def launch_viewer() -> None:
    """Open this composed scene in MuJoCo's interactive viewer."""

    import mujoco.viewer

    model = make_model()
    mujoco.viewer.launch(model, make_data(model))


__all__ = [
    "CONFIG",
    "DEFAULT_CAMERA",
    "FR3_BASE_POS",
    "FR3_BASE_QUAT",
    "FR3_HOME_JOINT_QPOS",
    "FR3_XML",
    "SCENE_DIR",
    "launch_viewer",
    "make_data",
    "make_model",
    "make_spec",
]


if __name__ == "__main__":
    launch_viewer()
'''


def _camera_literal(camera: dict[str, object]) -> str:
    values = dict(camera)
    lookat = values.get("camera_lookat")
    if lookat is not None:
        values["camera_lookat"] = tuple(lookat)  # type: ignore[arg-type]
    return repr(values)


def _replay_py(package_name: str) -> str:
    return f'''"""Replay / record entry points for the real2sim {package_name} task."""

from __future__ import annotations

from scene_io.playback import make_playback
from tasks.invention_real2sim.{package_name}.scene import DEFAULT_CAMERA, make_spec


playback = make_playback(
    spec_builder=make_spec,
    default_camera=DEFAULT_CAMERA,
    default_output="real2sim_{package_name}_replay.mp4",
)

replay_browser = playback.browser
replay_mujoco = playback.mujoco
record_video = playback.record
'''


def _init_py(package_name: str) -> str:
    return f'''"""Packaged real2sim task scene: {package_name}."""

from __future__ import annotations
'''


def package_scene(
    source_dir: str | Path,
    *,
    name: str | None = None,
    tasks_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Create or refresh a sanitized task scene from a real2sim output dir."""

    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"real2sim output directory not found: {source}")
    _validate_source(source)

    package_name = _snake_identifier(name or source.name)
    root = (
        Path(tasks_root)
        if tasks_root is not None
        else _repo_root() / "tasks" / "invention_real2sim"
    )
    root = root.expanduser().resolve()
    dest = root / package_name
    bundle = dest / "bundle"

    if dest.exists() and not force:
        raise FileExistsError(
            f"task package already exists: {dest}\n"
            "pass --force to refresh its generated wrapper and runtime bundle"
        )
    if dest.exists():
        if dest == source or source.is_relative_to(dest):
            raise ValueError(
                f"refusing to replace task directory containing its source: {dest}"
            )
        shutil.rmtree(dest)
    root.mkdir(parents=True, exist_ok=True)
    root_init = root / "__init__.py"
    if not root_init.exists():
        root_init.write_text(
            '"""Sanitized task scenes generated from real2sim outputs."""\n',
            encoding="utf-8",
        )
    dest.mkdir(parents=True, exist_ok=True)
    task_name = f"real2sim_{package_name}"
    default_camera = _copy_runtime_bundle(source, bundle, task_name=task_name)
    (dest / "__init__.py").write_text(_init_py(package_name), encoding="utf-8")
    (dest / "scene.py").write_text(
        _scene_py(package_name, default_camera),
        encoding="utf-8",
    )
    (dest / "replay.py").write_text(_replay_py(package_name), encoding="utf-8")
    return dest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package generated results/real2sim/<case> directories as "
            "sanitized tasks.invention_real2sim.<case> scene modules."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              ./scripts/rapid-real2sim results/real2sim/demo5_scene
              ./scripts/rapid-real2sim frame.png --name demo5_scene

            Then launch agents with:
              --env-module tasks.invention_real2sim.demo5_scene
            """
        ),
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        type=Path,
        help="Generated real2sim output directory, e.g. results/real2sim/demo5_scene.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Task package name. Defaults to a snake_case version of source_dir name.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Destination real2sim task root. Defaults to tasks/invention_real2sim.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing packaged task wrapper and runtime bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.name is not None and len(args.source_dirs) != 1:
        parser.error("--name can only be used with one source directory")

    try:
        destinations = [
            package_scene(
                source_dir,
                name=args.name,
                tasks_root=args.tasks_root,
                force=args.force,
            )
            for source_dir in args.source_dirs
        ]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for dest in destinations:
        module = f"tasks.invention_real2sim.{dest.name}"
        rel_dest = dest
        try:
            rel_dest = dest.relative_to(_repo_root())
        except ValueError:
            pass
        print(f"[real2sim.package] wrote {rel_dest}")
        print(f"[real2sim.package] env module: {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
