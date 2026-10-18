"""Regressions for physical evidence and retained execution boundaries."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from instantiation_core.handoff import capture_boundary, run_handoff_probe, restore_boundary
from instantiation_core.metric_artifacts import externalize_metric_arrays
from primitive_core.evidence import finalize_primitive_result
from primitive_core.geometry import contact_pairs, geom_distance_within
from primitive_core.profiling import capture_timings, execution_timer
from primitive_core.result_retention import retain_execution_result
from strategies.runtime import execute_sequential_strategy
from tests.test_strategy_program import _model, _result, _two_phase_fixture


class ExecutionEvidenceTests(unittest.TestCase):
    def test_distance_cutoff_does_not_imply_contact(self):
        model = _model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertIsNone(geom_distance_within(model, data, 0, 1, .01))
        self.assertEqual(contact_pairs(data, [0], [1]), ())
        data.qpos[0] = .275
        mujoco.mj_forward(model, data)
        self.assertLessEqual(geom_distance_within(model, data, 0, 1, .01), 0)
        self.assertTrue(contact_pairs(data, [0], [1]))

    def test_final_success_requires_all_declared_evidence(self):
        result = _result(np.zeros(2), .1, .1)
        for updates, expected in [
            ({"mechanism_valid": False}, False),
            ({"mechanism_valid": True, "stability_required": True}, False),
            ({"stability_valid": False}, False),
            ({"stability_valid": True}, True),
            ({"transit": {"transit_valid": False}}, False),
        ]:
            result.metrics.update(updates)
            finalize_primitive_result(result)
            self.assertIs(result.metrics["success"], expected)
            self.assertTrue(result.metrics["optimized_success"])
        self.assertEqual(result.best_states.shape, (2, 2))

    def test_numeric_arrays_round_trip_without_mutating_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = np.arange(3000, dtype=float).reshape(1000, 3)
            metrics = {"metrics": {"phase_metrics": {"first": {"positions": source.tolist(), "success": True}}}}
            output = Path(tmp) / "metrics.json"
            packed = externalize_metric_arrays(metrics, output)
            output.write_text(json.dumps(packed))
            pointer = packed["metrics"]["phase_metrics"]["first"]["positions"]
            with np.load(output.parent / packed["metric_arrays"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive[pointer["array_key"]], source)
            self.assertIsInstance(metrics["metrics"]["phase_metrics"]["first"]["positions"], list)
            self.assertLess(output.stat().st_size, 1000)

    def test_failed_stage_still_records_elapsed_time(self):
        with capture_timings() as timings:
            with self.assertRaises(ValueError), execution_timer("geometry"):
                raise ValueError("bad geometry")
        self.assertEqual(timings["geometry"]["calls"], 1)
        self.assertGreaterEqual(timings["geometry"]["wall_seconds"], 0)

    def execute(self, run_phase, run_transit):
        strategy, primitive, program, binding = _two_phase_fixture()
        model = _model()
        return execute_sequential_strategy(
            program=program, strategy_contract=strategy,
            primitive_contracts={"effect": primitive}, strategy_binding=binding,
            model=model, initial_state=np.zeros(model.nq + model.nv),
            run_phase=run_phase, run_transit=run_transit, dt=.01,
        )

    @patch.dict(os.environ, {"RAPID_EXECUTION_POLICY": "attempt"})
    def test_postprocessing_failure_keeps_prefix_and_current_rollout(self):
        def phase(spec, binding, context):
            result = _result(context.state, .1 if spec.name == "first" else .3, .1)
            retain_execution_result(result)
            if spec.name == "second":
                raise ValueError("postprocessing failed")
            return result
        with self.assertRaisesRegex(ValueError, "postprocessing") as caught:
            self.execute(phase, lambda spec, ctx: _result(ctx.state, .2, .2))
        result = caught.exception.result
        self.assertAlmostEqual(result.best_states[0, 0], 0)
        self.assertAlmostEqual(result.best_states[-1, 0], .3)
        self.assertEqual(result.metrics["execution_policy"], "attempt")
        self.assertFalse(result.metrics["strict_would_abort"])
        self.assertEqual(result.metrics["retained_segments"], ["first", "first__to__second", "second__rejected"])

    def test_parameter_resolution_failure_keeps_completed_prefix(self):
        from strategies.program import primitive_binding_for_phase
        def resolve(phase, *args, **kwargs):
            if phase.name == "second":
                raise ValueError("cannot resolve width")
            return primitive_binding_for_phase(phase, *args, **kwargs)
        with patch("strategies.runtime.primitive_binding_for_phase", side_effect=resolve):
            with self.assertRaisesRegex(ValueError, "resolve width") as caught:
                self.execute(lambda spec, binding, ctx: _result(ctx.state, .1, .1),
                             lambda spec, ctx: _result(ctx.state, .2, .2))
        self.assertAlmostEqual(caught.exception.result.best_states[0, 0], 0)
        self.assertAlmostEqual(caught.exception.result.best_states[-1, 0], .2)

    def test_effect_and_mechanism_outcomes_stay_separate_across_phases(self):
        def phase(spec, binding, ctx):
            result = _result(ctx.state, .1, .1)
            result.metrics["mechanism_valid"] = spec.name == "second"
            return result
        result = self.execute(phase, lambda spec, ctx: _result(ctx.state, .2, .2))
        self.assertTrue(result.metrics["all_primitive_phase_effects_success"])
        self.assertFalse(result.metrics["all_primitive_phases_success"])
        self.assertEqual(result.metrics["primitive_phase_success"], {"first": False, "second": True})

    def test_handoff_replays_real_boundary_and_rejects_stale_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "src"
            package = source / "strategies/two_phases"
            package.mkdir(parents=True)
            (source / "primitives").mkdir()
            implementation = package / "program.py"
            implementation.write_text("VERSION = 1\n")
            _, _, _, binding = _two_phase_fixture()
            starts = []
            def phase(spec, binding, ctx):
                starts.append((spec.name, ctx.state.copy(), ctx.initial_action.copy()))
                result = _result(ctx.state, .4, .4)
                result.metrics["transit"] = {"transit_valid": True, "transit_convergence_valid": True}
                return result
            def transit(spec, ctx):
                result = _result(ctx.state, ctx.state[0], .2)
                result.metrics.update(transit_valid=True, transit_convergence_valid=True)
                return result
            env = {"RAPID_WORKSPACE": tmp, "RAPID_ACTIVE_SRC_DIR": str(source),
                   "RAPID_INVENTION_ATTEMPT_LOG": str(workspace / "attempts.jsonl")}
            with patch.dict(os.environ, env):
                capture_boundary(binding, "second", np.array([.123, .045]), np.array([.7]), np.array([.6]), 0, model=_model())
                directory = workspace / "agent_work/handoffs/two_phases/test/second"
                snapshot = directory / json.loads((directory / "latest.json").read_text())["snapshot"]
                with patch("instantiation_core.entrypoint.run_manifest_entrypoint",
                           side_effect=lambda *args: self.execute(phase, transit)):
                    report = run_handoff_probe(workspace / "binding.json", snapshot, [[0.], [.001]])
                self.assertEqual([item[0] for item in starts], ["second", "second"])
                np.testing.assert_allclose([item[1][0] for item in starts], [.123, .124])
                np.testing.assert_allclose([item[1][1] for item in starts], [.045, .045])
                self.assertTrue(all(not sample["findings"] for sample in report["samples"]))
                changed_model = _model()
                changed_model.geom_size[0, 0] *= 2
                with np.load(snapshot, allow_pickle=False) as archive:
                    replay = {"metadata": json.loads(str(archive["metadata"])),
                              "arrays": {}, "perturbation": [0.]}
                with self.assertRaisesRegex(ValueError, "physics or binding changed"):
                    restore_boundary(replay, changed_model, binding)
                implementation.write_text("VERSION = 2\n")
                with self.assertRaisesRegex(ValueError, "stale"):
                    run_handoff_probe(workspace / "binding.json", snapshot, [[0.]])


if __name__ == "__main__":
    unittest.main()
