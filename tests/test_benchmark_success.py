from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from benchmark_private.success import (
    evaluate_benchmark_success,
    rescore_retained_benchmark_evaluation,
)
from suites.staged.benchmark_eval import _records_returncode
from suites.staged.benchmark_runner import _summary


def _strategy_2_metrics(
    *,
    jaw_a: float = 1.0,
    jaw_b: float = 1.0,
    clearance: float = 0.02,
) -> dict[str, object]:
    return {
        "metrics": {
            "final_psi": {
                "jaw_a_late_contact_fraction": jaw_a,
                "jaw_b_late_contact_fraction": jaw_b,
                "minimum_late_support_clearance": clearance,
                "success": False,
            }
        },
        "program_success": False,
    }


_STORAGE_XML = """
<mujoco model="benchmark_storage">
  <option gravity="0 0 0" timestep="0.01"/>
  <worldbody>
    <body name="target" pos="0.02 0 0.03">
      <freejoint/>
      <geom name="target_collision" type="box" size="0.05 0.08 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _pickup_xml(*, platform: bool) -> str:
    support = (
        '<body name="platform" pos="0 0 0.05">'
        '<geom name="platform_collision" type="box" size="0.2 0.2 0.05"/>'
        "</body>"
        if platform
        else '<geom name="floor" type="plane" size="1 1 0.01"/>'
    )
    target_z = 0.13 if platform else 0.04
    return f"""
<mujoco model="benchmark_pickup">
  <option gravity="0 0 0" timestep="0.01"/>
  <worldbody>
    {support}
    <body name="target" pos="0 0 {target_z}">
      <freejoint/>
      <geom name="target_collision" type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="left_finger" pos="-0.024 0 {target_z}">
      <geom name="left_collision" type="box" size="0.005 0.02 0.02"/>
    </body>
    <body name="right_finger" pos="0.024 0 {target_z}">
      <geom name="right_collision" type="box" size="0.005 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_static_trajectory(path: Path, model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = np.concatenate((data.qpos, data.qvel))
    np.savez_compressed(path, states=np.tile(state, (4, 1)))


def _storage_manifest(strategy_id: int) -> dict[str, object]:
    keys = {
        1: ("corridor", "available_store_depth"),
        5: ("upright_slot", "available_depth"),
        6: ("planar_slot", "available_depth"),
    }
    layout_key, depth_key = keys[strategy_id]
    return {
        "authoritative_roles": {"target": "target", "support": None},
        "context_layout": {
            layout_key: {
                "entrance_xy": [0.0, 0.0],
                "axis_xy": [1.0, 0.0],
                "lateral_xy": [0.0, 1.0],
                "clear_width": 0.10,
                depth_key: 0.30,
            }
        },
    }


class BenchmarkSuccessTests(unittest.TestCase):
    def test_strategy_two_uses_bilateral_grasp_and_one_centimeter_clearance(self) -> None:
        evaluation = evaluate_benchmark_success(2, _strategy_2_metrics())
        assert evaluation is not None
        self.assertTrue(evaluation["success"])
        self.assertEqual(
            evaluation["thresholds"]["minimum_support_clearance_m"],
            0.01,
        )

    def test_strategy_two_rejects_one_missing_jaw_or_insufficient_clearance(self) -> None:
        missing_jaw = evaluate_benchmark_success(
            2,
            _strategy_2_metrics(jaw_b=0.59),
        )
        low_clearance = evaluate_benchmark_success(
            2,
            _strategy_2_metrics(clearance=0.0099),
        )
        assert missing_jaw is not None and low_clearance is not None
        self.assertFalse(missing_jaw["success"])
        self.assertFalse(low_clearance["success"])

    def test_missing_measurements_fail_closed(self) -> None:
        evaluation = evaluate_benchmark_success(2, {"metrics": {}})
        assert evaluation is not None
        self.assertFalse(evaluation["measurements_available"])
        self.assertFalse(evaluation["success"])

    def test_all_pickup_strategies_use_the_same_simple_terminal_rule(self) -> None:
        metric_shapes = {
            2: {
                "jaw_a_late_contact_fraction": 0.8,
                "jaw_b_late_contact_fraction": 0.8,
                "minimum_late_support_clearance": 0.011,
            },
            3: {
                "measurements": {
                    "jaw_a_late_contact_fraction": 0.8,
                    "jaw_b_late_contact_fraction": 0.8,
                    "minimum_late_lift_progress": 0.011,
                }
            },
            4: {
                "diagnostics": {
                    "jaw_a_late_contact_fraction": 0.8,
                    "jaw_b_late_contact_fraction": 0.8,
                    "minimum_late_target_bottom_clearance_m": 0.011,
                }
            },
            7: {
                "diagnostics": {
                    "jaw_a_late_contact_fraction": 0.8,
                    "jaw_b_late_contact_fraction": 0.8,
                    "minimum_late_lift_height": 0.011,
                }
            },
        }
        for strategy_id, final_psi in metric_shapes.items():
            with self.subTest(strategy=strategy_id):
                evaluation = evaluate_benchmark_success(
                    strategy_id,
                    {"metrics": {"final_psi": final_psi}},
                )
                assert evaluation is not None
                self.assertTrue(evaluation["success"])
                self.assertEqual(
                    set(evaluation["conjuncts"]),
                    {"terminal_bilateral_grasp", "terminal_support_clearance"},
                )

    def test_trajectory_pickup_uses_floor_or_authoritative_platform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for strategy_id, platform in ((2, False), (8, True)):
                with self.subTest(strategy=strategy_id):
                    model = mujoco.MjModel.from_xml_string(
                        _pickup_xml(platform=platform)
                    )
                    path = Path(tmp) / f"pickup_{strategy_id}.npz"
                    _write_static_trajectory(path, model)
                    support = "platform" if platform else None
                    evaluation = evaluate_benchmark_success(
                        strategy_id,
                        {},
                        state_trajectory_path=path,
                        private_manifest={
                            "authoritative_roles": {
                                "target": "target",
                                "support": support,
                            },
                            "context_layout": {
                                "world_frame": {"floor_z": 0.0}
                            },
                        },
                        model=model,
                    )
                    assert evaluation is not None
                    self.assertTrue(evaluation["success"], evaluation)
                    self.assertEqual(evaluation["source"], "state_trajectory")

    def test_all_storage_strategies_use_terminal_geometric_containment(self) -> None:
        model = mujoco.MjModel.from_xml_string(_STORAGE_XML)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "storage.npz"
            _write_static_trajectory(path, model)
            for strategy_id in (1, 5, 6):
                with self.subTest(strategy=strategy_id):
                    evaluation = evaluate_benchmark_success(
                        strategy_id,
                        {},
                        state_trajectory_path=path,
                        private_manifest=_storage_manifest(strategy_id),
                        model=model,
                    )
                    assert evaluation is not None
                    self.assertTrue(evaluation["success"], evaluation)
                    self.assertEqual(
                        set(evaluation["conjuncts"]),
                        {
                            "center_past_entrance",
                            "center_before_back_boundary",
                            "center_laterally_contained",
                        },
                    )
                    self.assertAlmostEqual(
                        evaluation["measurements"]["target_center_depth_m"],
                        0.02,
                    )

    def test_retained_storage_decision_is_rescored_without_simulation(self) -> None:
        evaluation = rescore_retained_benchmark_evaluation(
            5,
            {
                "evaluator": "strategy_5_terminal_storage_center_entry_v2",
                "source": "state_trajectory",
                "success": False,
                "measurements": {
                    "target_depth_min_m": -0.04504243719635648,
                    "target_depth_max_m": 0.14797149666380952,
                    "target_lateral_min_m": -0.06936869401184781,
                    "target_lateral_max_m": 0.03071875991168265,
                },
                "thresholds": {
                    "available_depth_m": 0.224184565,
                    "lateral_half_width_m": 0.0411187475,
                    "containment_tolerance_m": 0.005,
                },
            },
        )
        assert evaluation is not None
        self.assertTrue(evaluation["success"])
        self.assertEqual(
            evaluation["evaluator"],
            "strategy_5_terminal_storage_center_region_v3",
        )
        self.assertEqual(
            evaluation["conjuncts"]["center_past_entrance"],
            True,
        )
        self.assertTrue(
            evaluation["conjuncts"]["center_laterally_contained"]
        )

    def test_trajectory_evaluation_fails_closed_when_artifact_is_missing(self) -> None:
        evaluation = evaluate_benchmark_success(
            6,
            {},
            state_trajectory_path="missing.npz",
            private_manifest=_storage_manifest(6),
            model=mujoco.MjModel.from_xml_string(_STORAGE_XML),
        )
        assert evaluation is not None
        self.assertFalse(evaluation["success"])
        self.assertFalse(evaluation["measurements_available"])

    def test_summary_prefers_benchmark_success_and_retains_disagreement(self) -> None:
        records = [
            {
                "benchmark_success": True,
                "program_success": False,
                "target_bound": True,
                "binding_valid": True,
                "authoritative_scene_role_coverage": 1.0,
                "bucket": "diagnostic",
                "diagnostic_dimension": "size",
                "stress_axis": None,
            }
        ]
        summary = _summary(records)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["strategy_predicate_successes"], 0)
        self.assertEqual(summary["benchmark_strategy_disagreements"], 1)

    def test_strict_return_code_uses_benchmark_success(self) -> None:
        records = [
            {
                "case_index": 1,
                "binding_valid": True,
                "benchmark_success": True,
                "program_success": False,
            }
        ]
        self.assertEqual(
            _records_returncode(records, selected=(1,), strict=True),
            0,
        )


if __name__ == "__main__":
    unittest.main()
