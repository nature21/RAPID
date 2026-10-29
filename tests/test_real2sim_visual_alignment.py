"""Visual fitting must keep multipart geometry and physical state consistent."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from real2sim.scene_repair import SceneHealthError, validate_repaired_scene
from real2sim.visual_alignment import fit_visuals_to_collision, visual_collision_alignment


class VisualAlignmentTest(unittest.TestCase):
    def _scene(self, directory: Path) -> Path:
        meshes = directory / "meshes"
        meshes.mkdir()
        for name, offset in (("left", -0.05), ("right", 0.05)):
            mesh = trimesh.creation.box(extents=[0.1, 0.08, 0.12])
            mesh.apply_translation([offset, 0, 0])
            mesh.apply_transform(trimesh.transformations.rotation_matrix(0.2, [0, 0, 1]))
            mesh.apply_translation([0.06, 0, 0])
            # Include explicit UV indices to verify they survive the mesh rewrite.
            with (meshes / f"{name}.obj").open("w") as output:
                for index, vertex in enumerate(mesh.vertices):
                    output.write("v " + " ".join(map(str, vertex)) + "\n")
                    output.write(f"vt {index / len(mesh.vertices)} 0.5\n")
                for face in mesh.faces:
                    output.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in face) + "\n")
        path = directory / "scene.xml"
        path.write_text('''<mujoco>
  <asset>
    <mesh name="left" file="meshes/left.obj"/>
    <mesh name="right" file="meshes/right.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="object" pos="0 0 0.05">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
      <geom name="object_collision" type="box" size="0.05 0.04 0.05"/>
      <geom name="left_visual" type="mesh" mesh="left" contype="0" conaffinity="0"
            density="0" group="2" rgba="1 0 0 1"/>
      <geom name="right_visual" type="mesh" mesh="right" contype="0" conaffinity="0"
            density="0" group="2" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>''')
        return path

    def test_physical_success_does_not_accept_displaced_visuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._scene(directory)
            with self.assertRaises(SceneHealthError) as raised:
                validate_repaired_scene(directory)
            report = raised.exception.report
            self.assertEqual(report["failure_kind"], "visual_collision_alignment")
            self.assertEqual(report["stability"]["status"], "ok")
            self.assertEqual(report["visual_collision_alignment"]["failed_objects"], ["object"])

    def test_fit_preserves_physics_uvs_and_relative_color_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self._scene(directory)
            old_xml = path.read_bytes()
            old = mujoco.MjModel.from_xml_path(str(path))
            old_meshes = [(directory / f"meshes/{name}.obj").read_bytes() for name in ("left", "right")]
            report = fit_visuals_to_collision(path, ["object"])
            self.assertEqual(report["before"]["status"], "failed")
            self.assertEqual(report["after"]["status"], "ok")
            self.assertEqual(validate_repaired_scene(directory, before_xml=old_xml)["status"], "ok")
            new = mujoco.MjModel.from_xml_path(str(path))
            for field in ("qpos0", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
                np.testing.assert_allclose(getattr(new, field), getattr(old, field), atol=1e-10)
            old_collision = ET.fromstring(old_xml).find(".//geom[@name='object_collision']")
            new_collision = ET.parse(path).find(".//geom[@name='object_collision']")
            self.assertEqual(old_collision.attrib, new_collision.attrib)
            data = mujoco.MjData(new)
            mujoco.mj_forward(new, data)
            from real2sim.adapter import _geom_points
            left = _geom_points(new, data, new.geom("left_visual").id)
            right = _geom_points(new, data, new.geom("right_visual").id)
            # One assembly transform keeps the colored halves on opposite sides.
            self.assertLess(left[:, 0].mean(), right[:, 0].mean())
            self.assertAlmostEqual(left[:, 0].max(), right[:, 0].min(), delta=1e-6)
            for name, original in zip(("left", "right"), old_meshes):
                self.assertEqual((directory / f"meshes/{name}.obj").read_bytes(), original)
                old_id = old.geom(f"{name}_visual").dataid[0]
                new_id = new.geom(f"{name}_visual").dataid[0]
                old_start, new_start = old.mesh_texcoordadr[old_id], new.mesh_texcoordadr[new_id]
                count = old.mesh_texcoordnum[old_id]
                self.assertGreater(count, 0)
                np.testing.assert_allclose(
                    old.mesh_texcoord[old_start:old_start + count],
                    new.mesh_texcoord[new_start:new_start + count],
                )

    def test_obb_axis_permutations_do_not_rotate_parts_by_ninety_degrees(self) -> None:
        from real2sim.visual_alignment import _closest_box_axes
        axes = np.eye(3)[:, [2, 0, 1]]
        result, permutation = _closest_box_axes(axes, np.eye(3))
        np.testing.assert_allclose(result, np.eye(3))
        self.assertEqual(permutation, (1, 2, 0))
