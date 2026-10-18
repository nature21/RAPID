"""Tests for the interactive FR3 scene viewer control logic."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from tasks.invention_real2sim.demo_1_scene.scene import make_data, make_model
from vistools.view_scene import (
    FR3Controller,
    InitialState,
    _enable_fr3_gravity_compensation,
    _native_reset_detected,
    _load_scene,
    _make_model_and_data,
    main,
)


class SceneViewerInputTests(unittest.TestCase):
    def test_supported_scene_references_preserve_model_and_home_state(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference_model = make_model()
        reference_data = make_data(reference_model)
        for scene_ref in (
            "demo_1_scene",
            "invention_real2sim/demo_1_scene",
            "tasks.invention_real2sim.demo_1_scene",
            "tasks.invention_real2sim.demo_1_scene.scene",
            str(root / "tasks/invention_real2sim/demo_1_scene"),
            str(root / "tasks/invention_real2sim/demo_1_scene/scene.py"),
        ):
            with self.subTest(scene=scene_ref):
                model, data = _make_model_and_data(_load_scene(scene_ref))
                self.assertEqual(model.nq, reference_model.nq)
                np.testing.assert_array_equal(data.qpos, reference_data.qpos)
                np.testing.assert_array_equal(data.ctrl, reference_data.ctrl)

    def test_kinematic_mode_does_not_enter_osc_runtime(self) -> None:
        with (
            mock.patch("vistools.view_scene._run_kinematic", return_value=0) as run,
            mock.patch("vistools.view_scene._run_osc") as osc,
        ):
            self.assertEqual(main(["invention_real2sim/demo_1_scene", "--controller", "kinematic", "--paused"]), 0)
            osc.assert_not_called()
            args = run.call_args.args[0]
            self.assertEqual(args.scene, "invention_real2sim/demo_1_scene")
            self.assertTrue(args.paused)

    def test_default_mode_retains_osc_scene_and_gain_options(self) -> None:
        with (
            mock.patch("vistools.view_scene._run_osc", return_value=0) as run,
            mock.patch("vistools.view_scene._run_kinematic") as kinematic,
        ):
            self.assertEqual(main(["--translation-stiffness", "250"]), 0)
            kinematic.assert_not_called()
            args = run.call_args.args[0]
            self.assertEqual(args.scene, "invention_real2sim/flip_scene")
            self.assertEqual(args.translation_stiffness, 250)


class FR3ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.controller = FR3Controller(
            self.model,
            self.data,
            body_name="hand",
            site_name="grasp_site",
        )

    def test_drag_to_tracks_cartesian_target_and_updates_controls(self) -> None:
        initial_site_pos = self.data.site("grasp_site").xpos.copy()
        target_body_pos = self.data.body("hand").xpos.copy()
        target_body_pos[0] += 0.03
        target_body_quat = self.data.body("hand").xquat.copy()

        self.controller.drag_to(
            self.data,
            target_body_pos,
            target_body_quat,
            iterations=20,
        )

        expected_site_pos = initial_site_pos + np.array([0.03, 0.0, 0.0])
        np.testing.assert_allclose(
            self.data.site("grasp_site").xpos,
            expected_site_pos,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            self.data.ctrl[self.controller.arm_actuator_ids],
            self.data.qpos[self.controller.arm_qpos_addresses],
        )

    def test_drag_to_tracks_orientation_target(self) -> None:
        target_body_pos = self.data.body("hand").xpos.copy()
        initial_quat = self.data.body("hand").xquat.copy()
        local_yaw = np.array([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
        target_body_quat = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(target_body_quat, initial_quat, local_yaw)

        self.controller.drag_to(
            self.data,
            target_body_pos,
            target_body_quat,
            iterations=20,
        )

        rotation_error = np.empty(3, dtype=float)
        mujoco.mju_subQuat(
            rotation_error,
            target_body_quat,
            self.data.body("hand").xquat,
        )
        self.assertLess(float(np.linalg.norm(rotation_error)), 1e-4)
        np.testing.assert_allclose(
            self.data.body("hand").xpos,
            target_body_pos,
            atol=1e-4,
        )

    def test_initial_state_restores_home_pose_and_nonzero_controls(self) -> None:
        initial_state = InitialState.capture(self.data)
        self.data.qpos[self.controller.arm_qpos_addresses] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        initial_state.restore(self.model, self.data)

        np.testing.assert_allclose(self.data.qpos, initial_state.qpos)
        np.testing.assert_allclose(self.data.ctrl, initial_state.ctrl)
        self.assertFalse(np.allclose(self.data.ctrl[:7], 0.0))

    def test_enabling_gravity_compensation_preserves_home_state(self) -> None:
        initial_state = InitialState.capture(self.data)
        initial_site_pos = self.data.site("grasp_site").xpos.copy()

        _enable_fr3_gravity_compensation(self.model, self.data)

        np.testing.assert_allclose(self.data.qpos, initial_state.qpos)
        np.testing.assert_allclose(self.data.ctrl, initial_state.ctrl)
        np.testing.assert_allclose(
            self.data.site("grasp_site").xpos,
            initial_site_pos,
        )

    def test_native_zero_state_is_detected_and_restored(self) -> None:
        initial_state = InitialState.capture(self.data)
        self.assertFalse(
            _native_reset_detected(
                self.model,
                self.data,
                self.controller,
                initial_state,
            )
        )

        mujoco.mj_resetData(self.model, self.data)
        self.assertTrue(
            _native_reset_detected(
                self.model,
                self.data,
                self.controller,
                initial_state,
            )
        )
        initial_state.restore(self.model, self.data)
        self.assertFalse(
            _native_reset_detected(
                self.model,
                self.data,
                self.controller,
                initial_state,
            )
        )


if __name__ == "__main__":
    unittest.main()
