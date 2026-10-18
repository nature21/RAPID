from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.play_cartesian_goals import (
    gripper_command_events,
    issue_gripper_event,
    load_document,
)


def _payload() -> dict:
    return {
        "schema": "rapid.fr3_cartesian_goals",
        "version": 2,
        "sample_rate_hz": 50.0,
        "time_semantics": "zero_order_hold",
        "pose_frame": "fr3_link0",
        "controlled_frame": "fr3_hand_tcp",
        "quaternion_order": "xyzw",
        "controller": {},
        "gripper": {
            "width_semantic": "total",
            "range_m": [0.0, 0.08],
            "move_speed_m_s": 0.04,
            "grasp_force_n": 40.0,
            "maximum_force_n": 60.0,
            "grasp_epsilon_inner_m": 0.005,
            "grasp_epsilon_outer_policy": "remaining_physical_opening",
        },
        "goals": [
            {
                "position_m": [0.0, 0.0, 0.3],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_width_m": 0.08,
                "gripper_operation": "move",
            },
            {
                "position_m": [0.01, 0.0, 0.3],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_width_m": 0.04,
                "gripper_operation": "grasp",
            },
            {
                "position_m": [0.02, 0.0, 0.3],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_width_m": 0.02,
                "gripper_operation": "grasp",
            },
        ],
    }


class CartesianGoalPlayerTests(unittest.TestCase):
    def _load(self, payload: object) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goals.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_document(path)

    def test_rejects_legacy_and_incomplete_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "document object"):
            self._load([])
        payload = _payload()
        del payload["goals"][0]["gripper_operation"]
        with self.assertRaisesRegex(ValueError, "requires"):
            self._load(payload)

    def test_rejects_invalid_width_and_increasing_grasp(self) -> None:
        payload = _payload()
        payload["goals"][0]["gripper_width_m"] = 0.09
        with self.assertRaisesRegex(ValueError, "range_m"):
            self._load(payload)
        payload = _payload()
        payload["goals"][2]["gripper_width_m"] = 0.06
        with self.assertRaisesRegex(ValueError, "GRASP width increases"):
            self._load(payload)

    def test_events_preserve_explicit_semantics_and_terminal_width(self) -> None:
        document = self._load(_payload())
        self.assertEqual(
            gripper_command_events(document["goals"]),
            {0: ("move", 0.08), 1: ("grasp", 0.02)},
        )

    def test_hardware_arguments_come_from_document(self) -> None:
        gripper = mock.Mock()
        parameters = self._load(_payload())["gripper"]
        issue_gripper_event(gripper, "grasp", 0.02, parameters)
        gripper.grasp_async.assert_called_once_with(
            0.02,
            0.04,
            40.0,
            0.005,
            0.06,
        )
        issue_gripper_event(gripper, "move", 0.08, parameters)
        gripper.move_async.assert_called_once_with(0.08, 0.04)


if __name__ == "__main__":
    unittest.main()
