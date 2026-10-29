"""End-to-end repair acceptance and rollback using a simulated agent editor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import trimesh

from real2sim.package_scene import _copy_runtime_bundle, _has_validated_scene_repair
from real2sim.scene_bundle import SceneBundleSnapshot
from real2sim.scene_repair import SceneRepairAgentError, repair_scene_with_agent


class SceneRepairAssetsTest(unittest.TestCase):
    def _source(self, root: Path) -> bytes:
        (root / "meshes").mkdir()
        trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(root / "meshes/part.obj")
        (root / "meshes/unused.obj").write_text("original unused mesh")
        (root / "input.png").touch()
        (root / "metadata.json").write_text(json.dumps({"objects": [
            {"safe_name": "kept", "name": "Original label"},
            {"safe_name": "removed", "name": "False detection"},
        ]}))
        xml = b'''<mujoco>
  <asset><mesh name="part" file="meshes/part.obj"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="kept" pos="-0.2 0 0.05"><freejoint/>
      <geom name="visual" type="mesh" mesh="part" contype="0" conaffinity="0"/>
      <geom name="kept_collision" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="removed" pos="0.2 0 0.05"><freejoint/>
      <geom name="removed_collision" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>'''
        (root / "scene.xml").write_bytes(xml)
        return xml

    def _repair(self, root: Path, edit, *, exit_code: int = 0):
        def run(*_):
            edit()
            return "test-thread", exit_code

        with (
            mock.patch("real2sim.scene_repair.shutil.which", return_value="/test/codex"),
            mock.patch("real2sim.scene_repair.render_source_camera"),
            mock.patch("real2sim.scene_repair._run_repair_process", side_effect=run),
        ):
            return repair_scene_with_agent(root, object_safe_names=["kept", "removed"])

    def test_mesh_only_edit_is_accepted_and_asset_change_invalidates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self._source(root)
            self._repair(root, lambda: trimesh.creation.box(
                extents=[0.08, 0.1, 0.1],
            ).export(root / "meshes/part.obj"))
            self.assertEqual((root / "scene.xml").read_bytes(), before)
            self.assertTrue(_has_validated_scene_repair(root))
            report = json.loads((root / "agent_repair/report.json").read_text())
            self.assertEqual(report["changed_assets"], ["meshes/part.obj"])
            with (root / "meshes/part.obj").open("a") as mesh:
                mesh.write("\n# changed after validation\n")
            self.assertFalse(_has_validated_scene_repair(root))

    def test_add_remove_and_replacement_mesh_survive_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self._source(root)

            def edit():
                trimesh.creation.box(extents=[0.08, 0.1, 0.1]).export(root / "meshes/new.obj")
                (root / "scene.xml").write_bytes(
                    before.replace(b"removed", b"added").replace(b"meshes/part.obj", b"meshes/new.obj")
                )
                (root / "meshes/part.obj").unlink()

            report = self._repair(root, edit)
            self.assertEqual(report["validated_object_safe_names"], ["kept", "added"])
            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual([item["safe_name"] for item in metadata["objects"]], ["kept", "added"])
            self.assertEqual(metadata["objects"][0]["name"], "Original label")
            bundle = root / "task_bundle"
            _copy_runtime_bundle(root, bundle, task_name="test")
            self.assertEqual((bundle / "meshes/new.obj").read_bytes(), (root / "meshes/new.obj").read_bytes())
            self.assertFalse((bundle / "meshes/part.obj").exists())
            self.assertEqual(
                {body.get("name") for body in ET.parse(bundle / "scene.xml").findall("./worldbody/body")},
                {"kept", "added"},
            )
            changes = json.loads((root / "agent_repair/report.json").read_text())["object_changes"]
            self.assertEqual(changes["added"], ["added"])
            self.assertEqual(changes["removed"], ["removed"])

    def test_failure_restores_modified_deleted_assets_and_metadata(self) -> None:
        for failure in ("physics", "exit", "interrupt"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                before = self._source(root)
                old_mesh = (root / "meshes/part.obj").read_bytes()
                old_metadata = (root / "metadata.json").read_bytes()

                def edit():
                    trimesh.creation.box(extents=[0.08, 0.1, 0.1]).export(root / "meshes/part.obj")
                    (root / "meshes/new.obj").write_text("candidate asset")
                    (root / "meshes/unused.obj").unlink()
                    (root / "metadata.json").write_text("{}")
                    (root / "scene.xml").write_bytes(before.replace(b'0.2 0 0.05', b'-0.2 0 0.05'))
                    if failure == "interrupt":
                        raise KeyboardInterrupt

                expected = KeyboardInterrupt if failure == "interrupt" else SceneRepairAgentError
                with self.assertRaises(expected):
                    self._repair(root, edit, exit_code=1 if failure == "exit" else 0)
                self.assertEqual((root / "scene.xml").read_bytes(), before)
                self.assertEqual((root / "meshes/part.obj").read_bytes(), old_mesh)
                self.assertEqual((root / "meshes/unused.obj").read_text(), "original unused mesh")
                self.assertEqual((root / "metadata.json").read_bytes(), old_metadata)
                self.assertFalse((root / "meshes/new.obj").exists())
                self.assertTrue((root / "agent_repair/failed_bundle/meshes/new.obj").is_file())

    def test_snapshot_tracks_obj_material_and_texture_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_text('<mujoco><asset><mesh name="m" file="mesh.obj"/></asset></mujoco>')
            (root / "mesh.obj").write_text("mtllib material.mtl\n")
            (root / "material.mtl").write_text("map_Kd texture.png\n")
            (root / "texture.png").write_bytes(b"original texture")
            snapshot = SceneBundleSnapshot(root, root / "agent_repair/bundle.before")
            (root / "material.mtl").unlink()
            (root / "texture.png").write_bytes(b"edited texture")
            snapshot.restore()
            self.assertEqual((root / "material.mtl").read_text(), "map_Kd texture.png\n")
            self.assertEqual((root / "texture.png").read_bytes(), b"original texture")
