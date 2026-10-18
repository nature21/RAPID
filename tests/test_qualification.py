from __future__ import annotations

import ast
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from agent.instantiation_validation import scene_interface_findings
from agent.invention_status import _qualification_status
from instantiation_core.qualification import (
    freeze_validation_cases, geometry_findings, run_validation_suite,
    GeometryProbe, SceneInterfaceProbe, HandoffProbe, ProbeIncomplete,
)


class QualificationTests(unittest.TestCase):
    def test_handoff_coverage_requires_every_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/strategies/probe"
            source.mkdir(parents=True)
            (source / "program.py").write_text("VERSION = 1\n")
            manifests = []
            for index in range(2):
                path = root / f"src/instantiations/strategies/probe/example_{index}/binding.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "schema_version": "rapid.binding/v1", "program_kind": "strategy",
                    "program_name": "probe", "instance_name": f"example_{index}",
                    "scene_module": "test_scenes.probe.scene", "goal_text": "test",
                    "roles": {}, "inputs": {},
                }))
                manifests.append(path)
            plan = root / "cases.json"
            freeze_validation_cases(plan, {"boundary": {"category": "handoff"}})
            program = SimpleNamespace(PHASE_PROGRAM=SimpleNamespace(
                phases=[SimpleNamespace(name="first"), SimpleNamespace(name="second")]))
            with patch("instantiation_core.qualification.importlib.import_module", return_value=program), \
                 patch.object(HandoffProbe, "run", return_value=([], {"executed": True})):
                report = run_validation_suite(
                    plan, {"boundary": HandoffProbe(manifests[0], "second")},
                    source_paths=[source], report_path=root / "report.json")
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("example_1/second", report["coverage_findings"][0])

    def test_layout_transform_and_wrong_frame_negative_control(self) -> None:
        reference = {"positions": {"goal": [1., 2., 0.]}, "directions": {"axis": [1., 0., 0.]}, "scalars": {"width": .2}, "decisions": {"fits": True}}
        observed = {"positions": {"goal": [-1., 1., 0.]}, "directions": {"axis": [0., 1., 0.]}, "scalars": {"width": .2}, "decisions": {"fits": True}}
        rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertEqual(geometry_findings(reference, observed, rotation=rotation, translation=np.array([1., 0., 0.])), [])
        self.assertTrue(geometry_findings(reference, reference, rotation=rotation))

    def test_changed_feasibility_and_nonfinite_observations_fail(self) -> None:
        self.assertTrue(geometry_findings({"decisions": {"fits": True}}, {"decisions": {"fits": False}}))
        self.assertTrue(geometry_findings({"scalars": {"extent": np.nan}}, {"scalars": {"extent": np.nan}}))
        self.assertTrue(geometry_findings({}, {}))
        with self.assertRaises(ValueError):
            geometry_findings({}, {}, rotation=np.diag([-1., 1., 1.]))

    def test_development_metadata_is_not_a_runtime_requirement(self) -> None:
        for source in ["scene.VARIANT_SPEC.to_dict()", "getattr(scene, 'VARIANT_SPEC')", "from scene import VARIANT_SPEC as spec"]:
            self.assertTrue(scene_interface_findings(ast.parse(source), Path("main.py")))
        self.assertEqual(scene_interface_findings(ast.parse("model = scene.make_model()\ndata = scene.make_data(model)"), Path("main.py")), [])

    def test_missing_boundary_is_incomplete_and_keeps_other_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/primitives/probe"
            source.mkdir(parents=True)
            (source / "program.py").write_text("VERSION = 1\n")
            plan = root / "cases.json"
            cases = {"boundary": {"category": "handoff"}, "interface": {"category": "scene_interface"}}
            freeze_validation_cases(plan, cases)
            probes = {"boundary": HandoffProbe(root / "binding.json", "second"),
                      "interface": SceneInterfaceProbe(root / "binding.json")}
            with patch.object(HandoffProbe, "run", side_effect=ProbeIncomplete("no executed boundary")), \
                 patch("instantiation_core.entrypoint.run_manifest_entrypoint",
                       return_value=SimpleNamespace(metrics={"success": False})):
                report = run_validation_suite(plan, probes, source_paths=[source], report_path=root / "report.json")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["checks"]["boundary"]["status"], "incomplete")
            self.assertTrue(report["checks"]["interface"]["passed"])
            self.assertFalse(report["checks"]["interface"]["measurements"]["behavior_success"])

    def test_fixed_cases_failure_retention_and_source_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "src/primitives/test_primitive"
            source.mkdir(parents=True)
            (source / "features.py").write_text("scale = 1\n")
            directory = workspace / "agent_work/qualification/test_primitive"
            plan = directory / "cases.json"
            cases = {category: {"category": category, "seed": 7} for category in ("scene_interface", "representation", "layout")}
            freeze_validation_cases(plan, cases)
            freeze_validation_cases(plan, cases)
            with self.assertRaises(ValueError):
                freeze_validation_cases(plan, {"easier": {"category": "layout"}})
            probes = {name: GeometryProbe(lambda case: {"scalars": {"width": .2}},
                                          lambda case: {"scalars": {"width": .2}},
                                          lambda case: {"scalars": {"width": .3}}) for name in cases}
            probes["scene_interface"] = SceneInterfaceProbe(workspace / "binding.json")
            with patch("instantiation_core.entrypoint.run_manifest_entrypoint",
                       return_value=SimpleNamespace(metrics={"success": False})):
                run_validation_suite(plan, probes, source_paths=[source], report_path=directory / "report.json")
            self.assertEqual(_qualification_status("primitive", "test_primitive", workspace)["status"], "passed")
            (source / "features.py").write_text("scale = 2\n")
            self.assertEqual(_qualification_status("primitive", "test_primitive", workspace)["status"], "stale")
            probes["representation"] = lambda case: []
            with patch("instantiation_core.entrypoint.run_manifest_entrypoint",
                       return_value=SimpleNamespace(metrics={"success": False})):
                report = run_validation_suite(plan, probes, source_paths=[source], report_path=directory / "report.json")
            self.assertFalse(report["passed"])
            self.assertIn("requires GeometryProbe", report["checks"]["representation"]["findings"][0])
            self.assertEqual(set(report["checks"]), set(cases))
            self.assertEqual(_qualification_status("primitive", "test_primitive", workspace)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
