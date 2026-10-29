"""Physics regressions for saving settled Real2Sim task poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real2sim.package_scene import _copy_runtime_bundle, _has_validated_scene_repair
from real2sim.settle_scene import settle_scene


SCENE = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="support" pos="0 0 0.05">
      <geom type="box" size="0.2 0.2 0.05"/>
    </body>
    <body name="object" pos="0 0 0.18" euler="3 0 0">
      <freejoint/>
      <geom name="object_collision" type="box" size="0.05 0.04 0.05"/>
    </body>
  </worldbody>
  <keyframe><key name="initial" qpos="0 0 0.18 1 0 0 0" qvel="1 0 0 0 0 0"/></keyframe>
</mujoco>
"""


class SettleSceneTest(unittest.TestCase):
    def test_saved_pose_and_keyframe_restart_at_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.xml"
            scene.write_text(SCENE, encoding="utf-8")
            report = settle_scene(scene)
            self.assertGreaterEqual(report["duration_seconds"], 2.0)
            root = ET.parse(scene)
            self.assertEqual(root.find("./worldbody/body[@name='support']").get("pos"),
                             "0 0 0.05")
            self.assertIsNone(root.find("./worldbody/body[@name='object']").get("euler"))
            model = mujoco.MjModel.from_xml_path(str(scene))
            for reset_to_keyframe in (False, True):
                with self.subTest(reset_to_keyframe=reset_to_keyframe):
                    data = mujoco.MjData(model)
                    if reset_to_keyframe:
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                    self.assertAlmostEqual(data.qpos[2], 0.15, delta=0.0005)
                    np.testing.assert_allclose(data.qvel, 0.0)
                    initial = data.qpos.copy()
                    for _ in range(round(1.0 / model.opt.timestep)):
                        mujoco.mj_step(model, data)
                    np.testing.assert_allclose(data.qpos, initial, atol=0.0005)

    def test_timeout_does_not_save_a_falling_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.xml"
            xml = '<mujoco><worldbody><body name="falling"><freejoint/>' \
                  '<geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
            scene.write_text(xml, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not settle"):
                settle_scene(scene, minimum_seconds=0.1, maximum_seconds=0.2,
                             stable_seconds=0.05)
            self.assertEqual(scene.read_text(encoding="utf-8"), xml)

    def test_packaging_settles_after_repair_without_invalidating_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "scene.xml").write_text(SCENE, encoding="utf-8")
            repair = source / "agent_repair"
            repair.mkdir()
            (repair / "report.json").write_text(json.dumps({
                "status": "ok", "after_sha256": hashlib.sha256(SCENE.encode()).hexdigest(),
                "validation": {"status": "ok"},
            }), encoding="utf-8")
            # A refit would try to read this missing original mesh, undoing repair.
            (source / "metadata.json").write_text(json.dumps({"objects": [{
                "safe_name": "object", "obb_source_mesh_file": "missing.obj",
            }]}), encoding="utf-8")
            bundle = Path(tmp) / "bundle"
            _copy_runtime_bundle(source, bundle, task_name="settled")
            self.assertTrue(_has_validated_scene_repair(source))
            self.assertEqual((source / "scene.xml").read_text(encoding="utf-8"), SCENE)
            root = ET.parse(bundle / "scene.xml")
            body = root.find("./worldbody/body[@name='object']")
            self.assertAlmostEqual(float(body.get("pos").split()[2]), 0.15, delta=0.0005)
            self.assertEqual(body.find("geom").get("size"), "0.05 0.04 0.05")
            self.assertEqual(json.loads((bundle / "settling.json").read_text())["status"], "ok")
