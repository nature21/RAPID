"""Regression tests for the Real2Sim Codex scene-repair fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from real2sim.mujoco_export import InitialObjectPenetrationError
from real2sim.pipeline import _validate_scene_with_agent_fallback
from real2sim.package_scene import (
    _has_validated_scene_repair,
    _replace_mesh_collisions_with_obbs,
)
from real2sim.scene_repair import (
    SceneHealthError,
    SceneRepairAgentError,
    _validate_edit_scope,
    validate_repaired_scene,
)
from real2sim.schema import PipelineConfig


def _scene_xml(second_x: float = 0.0) -> bytes:
    return f"""<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="first" pos="-0.2 0 0.05">
      <freejoint/>
      <geom name="first_collision" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="second" pos="{second_x} 0 0.05">
      <freejoint/>
      <geom name="second_collision" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
""".encode()


class SceneRepairScopeTest(unittest.TestCase):
    def test_allows_pose_and_size_attributes(self) -> None:
        before = b"""<mujoco><asset><mesh name="part" file="part.obj"/></asset>
<worldbody><body name="object" pos="0 0 0"><freejoint/>
<inertial mass="1" diaginertia="1 1 1"/>
<geom name="object_visual" type="mesh" mesh="part"/>
<geom name="object_collision" type="box" size="1 1 1"/>
</body></worldbody></mujoco>"""
        after = b"""<mujoco><asset>
<mesh name="part" file="part.obj" scale="1 2 1"/></asset>
<worldbody><body name="object" pos="1 0 0" quat="1 0 0 0"><freejoint/>
<inertial mass="1" diaginertia="2 3 4"/>
<geom name="object_visual" type="mesh" mesh="part" pos="0 0 0"/>
<geom name="object_collision" type="box" size="1 2 1" quat="1 0 0 0"/>
</body></worldbody></mujoco>"""

        _validate_edit_scope(before, after)

    def test_allows_mesh_replacement_but_rejects_untracked_bodies(self) -> None:
        before = b"""<mujoco><asset><mesh name="part" file="part.obj"/></asset>
<worldbody><body name="object"><geom name="object_collision" type="box"
size="1 1 1"/></body></worldbody></mujoco>"""
        changed_path = before.replace(b'file="part.obj"', b'file="other.obj"')
        _validate_edit_scope(before, changed_path)

        changed_structure = before.replace(
            b"</worldbody>",
            b'<body name="extra"/></worldbody>',
        )
        with self.assertRaisesRegex(SceneRepairAgentError, "protected scene"):
            _validate_edit_scope(before, changed_structure)

    def test_preserves_calibrated_world_and_existing_dynamics(self) -> None:
        before = _scene_xml(second_x=0.2)
        for after in (
            before.replace(b'<freejoint/>', b'', 1),
            before.replace(b'name="floor"', b'name="floor" pos="0 0 -1"'),
            before.replace(b'<worldbody>', b'<option gravity="0 0 0"/><worldbody>'),
            before.replace(b'name="first_collision"', b'name="first_collision" contype="0"'),
        ):
            with self.subTest(after=after), self.assertRaises(SceneRepairAgentError):
                _validate_edit_scope(before, after)

    def test_validates_added_objects_instead_of_deleted_original_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = _scene_xml(second_x=0.2)
            after = before.replace(b'second', b'added')
            (root / "scene.xml").write_bytes(after)
            report = validate_repaired_scene(
                root, before_xml=before, object_safe_names=["first", "second"],
            )
            self.assertEqual(report["validated_object_safe_names"], ["first", "added"])
            self.assertIn("added", report["stability"]["objects"])
            # An added floating object must still fail the independent rollout.
            (root / "scene.xml").write_bytes(after.replace(b'0.2 0 0.05', b'0.2 0 0.5'))
            with self.assertRaises(SceneHealthError):
                validate_repaired_scene(root, before_xml=before, object_safe_names=["first", "second"])

    def test_checks_extra_collision_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = _scene_xml(second_x=0.2)
            after = before.replace(
                b'<geom name="first_collision"',
                b'<geom name="extra_collision_part" type="box" size="0.05 0.05 0.05" pos="0.4 0 0"/>'
                b'<geom name="first_collision"',
            )
            (root / "scene.xml").write_bytes(after)
            with self.assertRaises(InitialObjectPenetrationError):
                validate_repaired_scene(root, before_xml=before)

    def test_validates_a_scoped_healthy_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = _scene_xml(second_x=-0.2)
            (root / "scene.xml").write_bytes(_scene_xml(second_x=0.2))

            report = validate_repaired_scene(
                root,
                before_xml=before,
                object_safe_names=["first", "second"],
                max_initial_object_penetration_meters=0.03,
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["maximum_penetration_meters"], 0.0)
        self.assertEqual(report["stability"]["status"], "ok")

    def test_rejects_a_scene_that_collapses_after_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_text(
                """<mujoco>
<worldbody><geom name="floor" type="plane" size="1 1 0.01"/>
<body name="falling" pos="0 0 0.5"><freejoint/>
<geom name="falling_collision" type="box" size="0.05 0.05 0.05"/>
</body></worldbody></mujoco>""",
                encoding="utf-8",
            )

            with self.assertRaises(SceneHealthError) as raised:
                validate_repaired_scene(
                    root,
                    object_safe_names=["falling"],
                )

        report = raised.exception.report
        self.assertEqual(report["failure_kind"], "dynamic_stability")
        self.assertIn("falling", report["stability"]["failed_objects"])
        self.assertGreater(
            report["stability"]["objects"]["falling"][
                "vertical_drift_meters"
            ],
            0.03,
        )

    def test_packaging_preserves_validated_agent_pose_and_collision_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_bytes = b"<mujoco/>"
            (root / "scene.xml").write_bytes(scene_bytes)
            repair_dir = root / "agent_repair"
            repair_dir.mkdir()
            (repair_dir / "report.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "after_sha256": hashlib.sha256(scene_bytes).hexdigest(),
                        "validation": {"status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            scene_root = ET.fromstring(
                """<mujoco><asset/>
<worldbody><body name="object" pos="1 2 3">
<geom name="object_collision" type="box" size="0.1 0.2 0.3"/>
</body></worldbody></mujoco>"""
            )

            _replace_mesh_collisions_with_obbs(scene_root, root)

        body = scene_root.find("./worldbody/body[@name='object']")
        collision = scene_root.find(
            "./worldbody/body[@name='object']/geom[@name='object_collision']"
        )
        assert body is not None and collision is not None
        self.assertEqual(body.get("pos"), "1 2 3")
        self.assertEqual(collision.get("size"), "0.1 0.2 0.3")

    def test_packaging_does_not_trust_a_stale_repair_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_bytes(b"<mujoco/>")
            repair_dir = root / "agent_repair"
            repair_dir.mkdir()
            (repair_dir / "report.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "after_sha256": "stale",
                        "validation": {"status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(_has_validated_scene_repair(root))


class PipelineAgentFallbackTest(unittest.TestCase):
    def test_deep_penetration_invokes_agent_and_returns_repaired_report(self) -> None:
        failure = {
            "status": "failed",
            "maximum_penetration_meters": 0.1,
            "threshold_meters": 0.03,
            "maximum_penetration_pair": ["first_collision", "second_collision"],
        }
        repaired = {
            "status": "ok",
            "maximum_penetration_meters": 0.0,
            "threshold_meters": 0.03,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_xml = root / "scene.xml"
            scene_xml.write_bytes(_scene_xml())
            source_image = root / "input.png"
            source_image.touch()
            config = PipelineConfig(
                use_scene_repair_agent=True,
                scene_repair_agent_model="test-model",
                scene_repair_agent_reasoning_effort="medium",
                scene_repair_agent_max_turns=7,
                verbose=False,
            )
            objects = [
                SimpleNamespace(safe_name="first"),
                SimpleNamespace(safe_name="second"),
            ]
            with (
                mock.patch(
                    "real2sim.scene_repair.validate_repaired_scene",
                    side_effect=InitialObjectPenetrationError(failure),
                ),
                mock.patch(
                    "real2sim.scene_repair.repair_scene_with_agent",
                    return_value=repaired,
                ) as repair,
            ):
                report = _validate_scene_with_agent_fallback(
                    scene_xml=scene_xml,
                    reconstructed=objects,
                    config=config,
                )

            saved_failure = json.loads(
                (root / "physics_preflight.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report, repaired)
        self.assertEqual(saved_failure, failure)
        repair.assert_called_once_with(
            root,
            source_image=source_image,
            error=mock.ANY,
            object_safe_names=["first", "second"],
            max_initial_object_penetration_meters=0.03,
            model="test-model",
            reasoning_effort="medium",
            max_turns=7,
        )

    def test_dynamic_stability_failure_invokes_agent(self) -> None:
        failure = {
            "status": "failed",
            "failure_kind": "dynamic_stability",
            "stability": {
                "status": "failed",
                "failed_objects": ["first"],
            },
        }
        repaired = {
            "status": "ok",
            "stability": {"status": "ok", "failed_objects": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_xml = root / "scene.xml"
            scene_xml.write_bytes(_scene_xml())
            source_image = root / "input.png"
            source_image.touch()
            with (
                mock.patch(
                    "real2sim.scene_repair.validate_repaired_scene",
                    side_effect=SceneHealthError("scene collapses", failure),
                ),
                mock.patch(
                    "real2sim.scene_repair.repair_scene_with_agent",
                    return_value=repaired,
                ) as repair,
            ):
                report = _validate_scene_with_agent_fallback(
                    scene_xml=scene_xml,
                    reconstructed=[SimpleNamespace(safe_name="first")],
                    config=PipelineConfig(
                        use_scene_repair_agent=True,
                        verbose=False,
                    ),
                )

        self.assertEqual(report, repaired)
        repair.assert_called_once()
        self.assertIsInstance(repair.call_args.kwargs["error"], SceneHealthError)

    def test_no_agent_flag_preserves_original_failure(self) -> None:
        failure = {
            "status": "failed",
            "maximum_penetration_meters": 0.1,
            "threshold_meters": 0.03,
            "maximum_penetration_pair": ["first_collision", "second_collision"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_xml = Path(tmp) / "scene.xml"
            scene_xml.write_bytes(_scene_xml())
            with (
                mock.patch(
                    "real2sim.scene_repair.validate_repaired_scene",
                    side_effect=InitialObjectPenetrationError(failure),
                ),
                self.assertRaises(InitialObjectPenetrationError),
            ):
                _validate_scene_with_agent_fallback(
                    scene_xml=scene_xml,
                    reconstructed=[SimpleNamespace(safe_name="first")],
                    config=PipelineConfig(
                        use_scene_repair_agent=False,
                        verbose=False,
                    ),
                )


class AgentFallbackCliTest(unittest.TestCase):
    def test_cli_enables_agent_fallback_by_default_and_can_disable_it(self) -> None:
        from real2sim.cli import build_arg_parser, pipeline_config_from_args

        parser = build_arg_parser()
        default = pipeline_config_from_args(parser.parse_args(["frame.png"]))
        disabled = pipeline_config_from_args(
            parser.parse_args(["frame.png", "--no-agent-fallback"])
        )

        self.assertTrue(default.use_scene_repair_agent)
        self.assertFalse(disabled.use_scene_repair_agent)


if __name__ == "__main__":
    unittest.main()
