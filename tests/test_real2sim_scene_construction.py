"""Regression tests for tool-free Real2Sim scene construction."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from real2sim.mujoco_export import (
    InitialObjectPenetrationError,
    _obb_3d_from_vertices,
    _quat_to_rotation_matrix,
    compile_mujoco_scene,
    write_mujoco_scene,
)
from real2sim.schema import PipelineConfig
from real2sim.workflow import package_reconstruction
from scene_io.runtime_assets import fr3_task_v1_root, fr3_task_v1_xml


def _arbitrary_rotation() -> np.ndarray:
    roll, pitch, yaw = 0.31, -0.42, 0.47
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def _rotated_box_mesh() -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
    rotation = _arbitrary_rotation()
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
    mesh.apply_translation([0.03, -0.04, 0.052])
    return mesh, np.asarray(mesh.vertices, dtype=float).copy()


def _object(mesh_file: str, vertices: np.ndarray, *, is_static: bool = False):
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    return SimpleNamespace(
        name="test object",
        safe_name="test_object",
        description="",
        mesh_file=mesh_file,
        obb_source_mesh_file=mesh_file,
        visual_mesh_files=[],
        body_pos=[0.0, 0.0, 0.0],
        local_bounds=bounds.tolist(),
        world_bounds=bounds.tolist(),
        is_static=is_static,
    )


class RuntimeAssetResolverTest(unittest.TestCase):
    def test_uses_explicit_repo_root_for_isolated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            robot_xml = repo_root / "runtime_assets/fr3/fr3_task_v1/robot.xml"
            robot_xml.parent.mkdir(parents=True)
            robot_xml.write_text("<mujoco/>\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"RAPID_REPO_ROOT": str(repo_root)},
                clear=False,
            ):
                self.assertEqual(fr3_task_v1_xml(), robot_xml.resolve())

    def test_direct_repository_fallback_needs_no_environment_variable(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAPID_REPO_ROOT", None)
            self.assertEqual(
                fr3_task_v1_root(),
                Path(__file__).resolve().parents[1]
                / "runtime_assets/fr3/fr3_task_v1",
            )


class OrientedCollisionBoxTest(unittest.TestCase):
    def test_collision_box_is_tight_3d_obb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, vertices = _rotated_box_mesh()
            mesh_path = root / "object.obj"
            mesh.export(mesh_path)
            scene_xml = root / "scene.xml"

            write_mujoco_scene(
                scene_xml,
                [_object(mesh_path.name, vertices)],
                PipelineConfig(verbose=False),
            )

            xml_root = ET.parse(scene_xml).getroot()
            collision = xml_root.find(
                "./worldbody/body[@name='test_object']/geom[@name='test_object_collision']"
            )
            self.assertIsNotNone(collision)
            assert collision is not None
            self.assertEqual(collision.get("type"), "box")
            self.assertIsNone(collision.get("mesh"))
            self.assertFalse(
                any(
                    "collision_mesh" in (asset.get("name") or "")
                    for asset in xml_root.findall("./asset/mesh")
                )
            )

            model = mujoco.MjModel.from_xml_path(str(scene_xml))
            geom_id = model.geom("test_object_collision").id
            rotation = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            center = np.asarray(model.geom_pos[geom_id], dtype=float)
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            projected = (vertices - center) @ rotation

            np.testing.assert_allclose(projected.min(axis=0), -size, atol=1e-8)
            np.testing.assert_allclose(projected.max(axis=0), size, atol=1e-8)
            self.assertLess(float(np.max(np.abs(rotation[2, :]))), 0.99)
            np.testing.assert_allclose(
                np.sort(2.0 * size), np.sort([0.4, 0.2, 0.1]), atol=1e-8
            )

            body_id = model.body("test_object").id
            np.testing.assert_allclose(model.body_ipos[body_id], center, atol=1e-10)
            self.assertEqual(model.body_jntnum[body_id], 1)

    def test_board_obb_includes_nail_protrusions(self) -> None:
        board = trimesh.creation.box(extents=[1.2, 0.18, 0.02])
        vertices = np.asarray(board.vertices)
        faces = np.asarray(board.faces)
        for _ in range(3):
            vertices, faces = trimesh.remesh.subdivide(vertices, faces)

        point_sets = [vertices]
        nail_positions = (
            (-0.42, -0.04),
            (-0.12, 0.05),
            (0.21, -0.03),
            (0.45, 0.04),
        )
        for x, y in nail_positions:
            nail = trimesh.creation.cylinder(radius=0.012, height=0.05, sections=16)
            nail.apply_translation([x, y, 0.035])
            point_sets.append(np.asarray(nail.vertices))

        rotation = _arbitrary_rotation()
        center = np.array([0.08, -0.03, 0.12])
        vertices_with_nails = np.vstack(point_sets) @ rotation.T + center
        collision_box = _obb_3d_from_vertices(vertices_with_nails)

        np.testing.assert_allclose(
            np.sort(2.0 * collision_box.half_extents),
            np.sort([1.2, 0.18, 0.07]),
            atol=2e-3,
        )
        np.testing.assert_allclose(
            collision_box.center,
            center + rotation @ np.array([0.0, 0.0, 0.025]),
            atol=2e-3,
        )

    def test_box_obb_uses_both_broad_faces(self) -> None:
        # Model two dense, slightly uneven broad faces plus the four side faces.
        # Collision fitting must always retain the complete vertex set.
        y = np.linspace(-0.06, 0.06, 10)
        z = np.linspace(-0.086, 0.086, 16)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        ripple = 0.002 * np.sin(yy * 40.0) * np.cos(zz * 30.0)
        broad_faces = [
            np.column_stack(
                [
                    np.full(yy.size, side * 0.0205) + ripple.ravel(),
                    yy.ravel(),
                    zz.ravel(),
                ]
            )
            for side in (-1.0, 1.0)
        ]
        side_points: list[np.ndarray] = []
        x = np.linspace(-0.0225, 0.0225, 5)
        for coordinate, extent, axis in (
            (-0.06, 0.086, 1),
            (0.06, 0.086, 1),
            (-0.086, 0.06, 2),
            (0.086, 0.06, 2),
        ):
            other = np.linspace(-extent, extent, 10)
            points = np.zeros((len(x) * len(other), 3), dtype=float)
            points[:, 0] = np.repeat(x, len(other))
            points[:, axis] = coordinate
            points[:, 3 - axis] = np.tile(other, len(x))
            side_points.append(points)

        vertices = np.vstack([*broad_faces, *side_points])
        rotation = _arbitrary_rotation()
        vertices = vertices @ rotation.T
        collision_box = _obb_3d_from_vertices(vertices)

        np.testing.assert_allclose(
            np.sort(2.0 * collision_box.half_extents),
            np.sort([0.045, 0.12, 0.172]),
            atol=5e-3,
        )

    def test_fixed_objects_still_receive_box_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, vertices = _rotated_box_mesh()
            mesh_path = root / "object.obj"
            mesh.export(mesh_path)
            scene_xml = root / "scene.xml"
            original_box = _obb_3d_from_vertices(vertices)
            original_rotation = _quat_to_rotation_matrix(original_box.quat)
            expected_angle = np.arccos(
                np.max(np.abs(original_rotation[2, :]))
            )
            original_bottom = float(
                original_box.center[2]
                - np.sum(
                    np.abs(original_rotation[2, :])
                    * original_box.half_extents
                )
            )

            write_mujoco_scene(
                scene_xml,
                [_object(mesh_path.name, vertices, is_static=True)],
                PipelineConfig(dynamic_objects=False, verbose=False),
            )

            model = mujoco.MjModel.from_xml_path(str(scene_xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            body_id = model.body("test_object").id
            geom_id = model.geom("test_object_collision").id
            self.assertEqual(model.body_jntnum[body_id], 0)
            self.assertEqual(
                model.geom_type[geom_id],
                mujoco.mjtGeom.mjGEOM_BOX,
            )
            world_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            self.assertAlmostEqual(
                float(np.max(np.abs(world_rotation[2, :]))),
                1.0,
                places=10,
            )
            corrected_bottom = float(
                data.geom_xpos[geom_id, 2]
                - np.sum(
                    np.abs(world_rotation[2, :]) * model.geom_size[geom_id]
                )
            )
            self.assertAlmostEqual(corrected_bottom, original_bottom, places=7)
            actual_angle = 2.0 * np.arccos(abs(float(model.body_quat[body_id, 0])))
            self.assertAlmostEqual(actual_angle, expected_angle, places=7)


class InitialObjectPenetrationTest(unittest.TestCase):
    def test_rejects_deep_object_overlap_but_ignores_floor_contact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
            first_path = root / "first.obj"
            second_path = root / "second.obj"
            mesh.export(first_path)
            mesh.export(second_path)
            vertices = np.asarray(mesh.vertices)
            first = _object(first_path.name, vertices)
            first.name = "first"
            first.safe_name = "first"
            second = _object(second_path.name, vertices)
            second.name = "second"
            second.safe_name = "second"
            scene_xml = root / "scene.xml"
            write_mujoco_scene(
                scene_xml,
                [first, second],
                PipelineConfig(verbose=False),
            )

            with self.assertRaises(InitialObjectPenetrationError) as raised:
                compile_mujoco_scene(
                    scene_xml,
                    object_safe_names=["first", "second"],
                    max_initial_object_penetration_meters=0.03,
                )

            self.assertGreater(
                raised.exception.report["maximum_penetration_meters"],
                0.03,
            )
            self.assertNotIn(
                "floor",
                raised.exception.report["maximum_penetration_pair"],
            )


class SharedFR3PackagingTest(unittest.TestCase):
    def test_packaging_uses_shared_fr3_at_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            mesh, _ = _rotated_box_mesh()
            mesh.export(source / "object.obj")
            (source / "scene.xml").write_text(
                """<mujoco model="source">
  <compiler meshdir="."/>
  <asset>
    <mesh name="object_visual_mesh" file="object.obj"/>
    <mesh name="object_collision_mesh" file="object.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="object">
      <freejoint/>
      <inertial mass="0.2" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
      <geom name="object_visual" type="mesh" mesh="object_visual_mesh"
            contype="0" conaffinity="0"/>
      <geom name="object_collision" type="box" size="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )
            (source / "metadata.json").write_text(
                """{
  "objects": [
    {
      "safe_name": "object",
      "obb_source_mesh_file": "object.obj"
    }
  ]
}
""",
                encoding="utf-8",
            )

            tasks_root = root / "tasks"
            tasks_root.mkdir()
            result = package_reconstruction(
                source,
                name="tool_free",
                tasks_root=tasks_root,
            )
            destination = result.task_dir
            self.assertEqual(result.module, "tasks.invention_real2sim.tool_free")
            self.assertFalse(hasattr(result, "end_effector"))

            scene_xml = destination / "bundle" / "scene.xml"
            packaged = ET.parse(scene_xml).getroot()
            body_names = {body.get("name") for body in packaged.findall(".//body")}
            self.assertIn("object", body_names)
            self.assertNotIn("fr3", body_names)
            self.assertFalse((destination / "bundle" / "fr3").exists())
            fr3_root = fr3_task_v1_root()
            self.assertTrue((fr3_root / "LICENSE").is_file())
            self.assertIsNone(ET.parse(fr3_root / "defaults.xml").find("sensor"))
            self.assertEqual(
                len(list((fr3_root / "meshes").iterdir())),
                50,
            )
            self.assertEqual(packaged.get("model"), "real2sim_tool_free")
            collision = packaged.find(
                "./worldbody/body[@name='object']/geom[@name='object_collision']"
            )
            self.assertIsNotNone(collision)
            assert collision is not None
            self.assertEqual(collision.get("type"), "box")
            self.assertIsNone(collision.get("mesh"))
            np.testing.assert_allclose(
                np.sort(np.fromstring(collision.get("size", ""), sep=" ") * 2.0),
                np.sort([0.4, 0.2, 0.1]),
                atol=1e-8,
            )
            self.assertIsNone(
                packaged.find("./asset/mesh[@name='object_collision_mesh']")
            )
            model = mujoco.MjModel.from_xml_path(str(scene_xml))
            self.assertEqual(model.nu, 0)
            expected_home = [
                -0.0030637685314503847,
                -0.7778934457717781,
                0.01585533499243391,
                -2.3693353790401894,
                -0.004060005459262234,
                1.554097367289531,
                0.7704814906387026,
            ]
            for mesh in ET.parse(fr3_root / "assets.xml").findall("./asset/mesh"):
                asset_path = (fr3_root / str(mesh.get("file"))).resolve()
                self.assertTrue(asset_path.is_relative_to(fr3_root))
                self.assertTrue(asset_path.is_file())

            module_spec = importlib.util.spec_from_file_location(
                "_standalone_fr3_scene_test",
                destination / "scene.py",
            )
            assert module_spec is not None and module_spec.loader is not None
            scene_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(scene_module)
            self.assertEqual(scene_module.FR3_XML, fr3_root / "robot.xml")
            model = scene_module.make_model()
            module_data = scene_module.make_data(model)
            np.testing.assert_allclose(scene_module.FR3_BASE_POS, [0.0, 0.0, 0.0])
            np.testing.assert_allclose(scene_module.FR3_BASE_QUAT, [1.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(
                module_data.body("fr3").xpos,
                [0.0, 0.0, 0.0],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                module_data.body("fr3").xquat,
                [1.0, 0.0, 0.0, 0.0],
                atol=1e-12,
            )
            self.assertGreaterEqual(model.body("hand").id, 0)
            self.assertEqual(model.nu, 8)
            self.assertEqual(model.nsensor, 0)
            self.assertEqual(
                [model.actuator(index).name for index in range(model.nu)],
                [
                    "fr3_joint1",
                    "fr3_joint2",
                    "fr3_joint3",
                    "fr3_joint4",
                    "fr3_joint5",
                    "fr3_joint6",
                    "fr3_joint7",
                    "fr3_hand",
                ],
            )
            np.testing.assert_allclose(
                model.actuator_gainprm[:, 0],
                [4500, 4500, 3500, 3500, 2000, 2000, 2000, 10000],
            )
            self.assertEqual(model.nkey, 1)
            self.assertEqual(model.key(0).name, "home")
            reset_data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, reset_data, 0)
            for index, expected in enumerate(expected_home, start=1):
                joint_id = model.joint(f"fr3_joint{index}").id
                qpos_address = model.jnt_qposadr[joint_id]
                self.assertAlmostEqual(
                    model.key_qpos[0, qpos_address],
                    expected,
                )
                self.assertAlmostEqual(
                    reset_data.qpos[qpos_address],
                    expected,
                )
                self.assertAlmostEqual(
                    module_data.qpos[qpos_address],
                    expected,
                )
                self.assertAlmostEqual(module_data.ctrl[index - 1], expected)
                self.assertAlmostEqual(model.key_ctrl[0, index - 1], expected)
            np.testing.assert_allclose(reset_data.ctrl, module_data.ctrl)
            for finger_name in ("finger_joint1", "finger_joint2"):
                finger_id = model.joint(finger_name).id
                self.assertAlmostEqual(
                    module_data.qpos[model.jnt_qposadr[finger_id]],
                    0.04,
                )
            self.assertAlmostEqual(module_data.ctrl[7], 0.04)
            finger_gap = np.linalg.norm(
                module_data.body("left_finger").xpos
                - module_data.body("right_finger").xpos
            )
            self.assertAlmostEqual(float(finger_gap), 0.08, places=8)
            for side in ("left", "right"):
                for index in range(6):
                    geom_name = f"{side}_finger_collision_{index}"
                    np.testing.assert_allclose(
                        model.geom_friction[model.geom(geom_name).id],
                        (1.5, 0.02, 0.002),
                    )
            np.testing.assert_allclose(
                model.geom_friction[model.geom("hand_collision").id],
                [0.5, 0.005, 0.0001],
            )


class Real2SimBenchmarkPlaceholderTest(unittest.TestCase):
    def test_placeholders_match_corresponding_demo_scenes(self) -> None:
        tasks_root = Path(__file__).resolve().parents[1] / "tasks"
        shared_fr3 = fr3_task_v1_root() / "robot.xml"

        for index in range(1, 9):
            with self.subTest(strategy=index):
                original_dir = tasks_root / "invention_real2sim" / f"demo_{index}_scene"
                benchmark_dir = (
                    tasks_root
                    / "sim_benchmark"
                    / f"strategy_{index}"
                    / "case_000_scene"
                )
                self.assertFalse((benchmark_dir / "bundle" / "fr3").exists())
                self.assertEqual(
                    (benchmark_dir / "bundle" / "metadata.json").read_bytes(),
                    (original_dir / "bundle" / "metadata.json").read_bytes(),
                )

                original_xml = ET.parse(original_dir / "bundle" / "scene.xml").getroot()
                benchmark_xml = ET.parse(benchmark_dir / "bundle" / "scene.xml").getroot()
                self.assertEqual(
                    benchmark_xml.get("model"),
                    f"sim_benchmark_strategy_{index}_case_000_scene",
                )
                original_xml.set("model", "normalized")
                benchmark_xml.set("model", "normalized")
                self.assertEqual(
                    ET.tostring(benchmark_xml),
                    ET.tostring(original_xml),
                )

                original = importlib.import_module(
                    f"tasks.invention_real2sim.demo_{index}_scene.scene"
                )
                benchmark = importlib.import_module(
                    "tasks.sim_benchmark."
                    f"strategy_{index}.case_000_scene.scene"
                )
                self.assertEqual(benchmark.FR3_XML, shared_fr3)
                original_model = original.make_model()
                benchmark_model = benchmark.make_model()
                self.assertEqual(benchmark_model.nbody, original_model.nbody)
                self.assertEqual(benchmark_model.ngeom, original_model.ngeom)
                self.assertEqual(benchmark_model.nu, original_model.nu)
                self.assertEqual(benchmark_model.nkey, original_model.nkey)

class FR3OriginTest(unittest.TestCase):
    def test_package_always_places_fr3_at_world_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "scene.xml").write_text(
                '<mujoco><worldbody><geom type="plane" size="1 1 0.01"/>'
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            tasks_root = root / "tasks"
            tasks_root.mkdir()

            result = package_reconstruction(
                source,
                name="calibrated",
                tasks_root=tasks_root,
            )

            scene_text = (result.task_dir / "scene.py").read_text(encoding="utf-8")
            self.assertIn("FR3_BASE_POS = (0.0, 0.0, 0.0)", scene_text)
            self.assertIn("FR3_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)", scene_text)


if __name__ == "__main__":
    unittest.main()
