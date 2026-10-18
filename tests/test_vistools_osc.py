"""Unit tests for the non-GUI OSC playground helpers."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from primitive_core.fr3_osc import (
    FR3_CONTROLLER_STATE_NAMES,
    FR3_OSC_OBSERVATION_NAMES,
    FR3OSCAdapter,
    FR3OSCConfig,
    fr3_controller_state_from_observation,
)
from primitive_core.native_fr3_osc import (
    NativeFR3OSCController,
    NativeFR3OSCRolloutBackend,
)
from primitive_core.policy import GripperOperation
from vistools.view_scene import (
    EndEffectorTransform,
    _clip_target,
    _key_command,
    _scaled_config,
    main as playground_main,
)
from tasks.invention_real2sim.push_scene.scene import make_data, make_model

try:
    from mujoco_extensions.fr3_osc_rollout import NativeFR3OSCController as _Native
except ImportError:
    _NATIVE_FR3_OSC_AVAILABLE = False
else:
    _NATIVE_FR3_OSC_AVAILABLE = _Native is not None


class OSCPlaygroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)

    def test_hand_site_transform_round_trips_current_pose(self) -> None:
        transform = EndEffectorTransform(
            self.model,
            self.data,
            body_name="hand",
            site_name="grasp_site",
        )
        site_position, site_quaternion = transform.site_pose_from_body_pose(
            self.data.body("hand").xpos,
            self.data.body("hand").xquat,
        )
        np.testing.assert_allclose(
            site_position,
            self.data.site("grasp_site").xpos,
            atol=1e-12,
        )
        actual_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            actual_quaternion,
            self.data.site("grasp_site").xmat,
        )
        self.assertAlmostEqual(abs(float(site_quaternion @ actual_quaternion)), 1.0)

        body_position, body_quaternion = transform.body_pose_from_site_pose(
            site_position,
            site_quaternion,
        )
        np.testing.assert_allclose(
            body_position,
            self.data.body("hand").xpos,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            abs(float(body_quaternion @ self.data.body("hand").xquat)),
            1.0,
        )

    def test_target_is_limited_around_home_and_by_action_range(self) -> None:
        home = self.adapter.action_from_data(self.model, self.data)
        target = home.copy()
        target[:3] += (1.0, -1.0, 0.5)
        target[3:6] = (10.0, -10.0, 10.0)
        target[6] = -1.0

        clipped = _clip_target(target, self.adapter, home, 0.2)

        np.testing.assert_allclose(clipped[:3], home[:3] + (0.2, -0.2, 0.2))
        np.testing.assert_allclose(clipped[3:6], (np.pi, -np.pi, np.pi))
        self.assertEqual(clipped[6], self.adapter.gripper_range[0])

    def test_target_has_no_home_relative_limit_by_default(self) -> None:
        home = self.adapter.action_from_data(self.model, self.data)
        target = home.copy()
        target[2] -= 0.6

        clipped = _clip_target(target, self.adapter, home, None)

        self.assertAlmostEqual(clipped[2], home[2] - 0.6)

    def test_compliance_presets_scale_damping_by_square_root(self) -> None:
        base = FR3OSCConfig()
        scaled = _scaled_config(base, 0.25)

        np.testing.assert_allclose(scaled.translational_stiffness, 50.0)
        np.testing.assert_allclose(scaled.rotational_stiffness, 7.5)
        np.testing.assert_allclose(
            scaled.translational_damping,
            np.asarray(base.translational_damping) * 0.5,
        )
        np.testing.assert_allclose(
            scaled.rotational_damping,
            np.asarray(base.rotational_damping) * 0.5,
        )
        self.assertEqual(scaled.maximum_task_force, base.maximum_task_force)

    def test_keymap_exposes_cartesian_and_controller_commands(self) -> None:
        self.assertEqual(_key_command(326), "x+")  # Numpad 6
        self.assertEqual(_key_command(334), "pitch+")  # Numpad +
        self.assertEqual(_key_command(335), "cycle-preset")  # Numpad Enter
        self.assertEqual(_key_command(325), "hold")  # Numpad 5
        self.assertEqual(_key_command(320), "gripper-move")  # Numpad 0
        self.assertEqual(_key_command(ord("G")), "grasp")
        self.assertIsNone(_key_command(ord("w")))
        self.assertIsNone(_key_command(ord("3")))
        self.assertIsNone(_key_command(999))

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_interactive_controller_preserves_mjcf_joint_frictionloss(self) -> None:
        arm_dof_addresses = np.asarray(
            [
                self.model.jnt_dofadr[self.model.joint(f"fr3_joint{index}").id]
                for index in range(1, 8)
            ],
            dtype=int,
        )
        expected = self.model.dof_frictionloss[arm_dof_addresses].copy()
        self.assertTrue(np.all(expected > 0.0))

        NativeFR3OSCController(
            self.model,
            self.data,
            self.adapter,
        )

        np.testing.assert_array_equal(
            self.model.dof_frictionloss[arm_dof_addresses],
            expected,
        )

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_interactive_native_step_matches_native_rollout(self) -> None:
        rollout_model = make_model()
        rollout_data = make_data(rollout_model)
        rollout_adapter = FR3OSCAdapter.from_model_data(
            rollout_model,
            rollout_data,
        )
        initial_state = np.concatenate((rollout_data.qpos, rollout_data.qvel))
        action = rollout_adapter.action_from_data(rollout_model, rollout_data)
        action[0] += 0.015
        actions = np.tile(action, (1, 80, 1))
        operations = np.zeros(80, dtype=np.uint8)
        operations[40:] = int(GripperOperation.GRASP)
        rollout = NativeFR3OSCRolloutBackend(
            rollout_model,
            1,
            rollout_adapter,
            gripper_operations=operations,
        )
        try:
            expected_states, expected_observations, _ = rollout.rollout(
                initial_state[None],
                actions,
            )
        finally:
            rollout.close()

        live_model = make_model()
        live_data = make_data(live_model)
        live_adapter = FR3OSCAdapter.from_model_data(live_model, live_data)
        controller = NativeFR3OSCController(live_model, live_data, live_adapter)
        live_states = []
        live_observations = []
        for step_action, operation in zip(actions[0], operations, strict=True):
            live_observations.append(
                controller.step(step_action, GripperOperation(int(operation)))
            )
            live_states.append(np.concatenate((live_data.qpos, live_data.qvel)))

        self.assertEqual(
            np.asarray(live_observations).shape,
            (80, len(FR3_OSC_OBSERVATION_NAMES)),
        )
        np.testing.assert_allclose(live_states, expected_states[0], atol=1e-12)
        np.testing.assert_allclose(
            live_observations,
            expected_observations[0],
            atol=1e-12,
        )

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_native_gripper_grasps_holds_force_and_releases_with_move(self) -> None:
        config = FR3OSCConfig(
            command_rate_hz=1.0 / self.model.opt.timestep,
            gripper_stall_steps=3,
            gripper_move_speed=10.0,
        )
        adapter = FR3OSCAdapter.from_model_data(
            self.model,
            self.data,
            config=config,
        )
        close_action = adapter.action_from_data(self.model, self.data)
        close_action[6] = adapter.gripper_range[0]
        controller = NativeFR3OSCController(
            self.model,
            self.data,
            adapter,
        )
        gripper_actuator = self.model.actuator(config.gripper_actuator_name).id
        finger_joints = (
            self.model.joint("finger_joint1"),
            self.model.joint("finger_joint2"),
        )
        gripper_mode_index = FR3_OSC_OBSERVATION_NAMES.index("gripper_mode")

        def pin_blocked_fingers() -> None:
            for joint in finger_joints:
                self.data.qpos[joint.qposadr] = 0.02
                self.data.qvel[joint.dofadr] = 0.0
            mujoco.mj_forward(self.model, self.data)

        modes = []
        for _ in range(config.gripper_stall_steps):
            pin_blocked_fingers()
            modes.append(
                controller.step(close_action, GripperOperation.GRASP)[
                    gripper_mode_index
                ]
            )

        np.testing.assert_array_equal(modes, (1.0, 1.0, 2.0))
        self.assertAlmostEqual(
            self.data.ctrl[gripper_actuator],
            -config.gripper_grasp_force,
        )
        for joint in finger_joints:
            self.assertAlmostEqual(
                self.data.qfrc_actuator[joint.dofadr],
                -0.5 * config.gripper_grasp_force,
                places=6,
            )
        held_state = controller.controller_state
        self.assertEqual(
            held_state[FR3_CONTROLLER_STATE_NAMES.index("gripper_mode")],
            2.0,
        )
        controller.reset()
        controller.restore_controller_state(held_state)
        np.testing.assert_array_equal(controller.controller_state, held_state)

        release_action = close_action.copy()
        release_action[6] = adapter.gripper_range[1]
        pin_blocked_fingers()
        release_observation = controller.step(
            release_action,
            GripperOperation.MOVE,
        )

        self.assertEqual(release_observation[gripper_mode_index], 0.0)
        self.assertGreater(self.data.ctrl[gripper_actuator], 0.0)

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_move_closing_never_enters_force_hold(self) -> None:
        config = FR3OSCConfig(
            command_rate_hz=1.0 / self.model.opt.timestep,
            gripper_stall_steps=3,
            gripper_move_speed=10.0,
        )
        adapter = FR3OSCAdapter.from_model_data(self.model, self.data, config=config)
        action = adapter.action_from_data(self.model, self.data)
        action[6] = adapter.gripper_range[0]
        controller = NativeFR3OSCController(self.model, self.data, adapter)
        mode_index = FR3_OSC_OBSERVATION_NAMES.index("gripper_mode")
        for _ in range(2 * config.gripper_stall_steps):
            for name in config.gripper_joint_names:
                joint = self.model.joint(name)
                self.data.qpos[joint.qposadr] = 0.02
                self.data.qvel[joint.dofadr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            observation = controller.step(action, GripperOperation.MOVE)
            self.assertEqual(observation[mode_index], 0.0)

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_move_gain_maps_one_millimeter_error_to_ten_newtons(self) -> None:
        config = FR3OSCConfig(
            command_rate_hz=1.0 / self.model.opt.timestep,
            gripper_move_speed=10.0,
        )
        adapter = FR3OSCAdapter.from_model_data(self.model, self.data, config=config)
        action = adapter.action_from_data(self.model, self.data)
        action[6] -= 0.001
        controller = NativeFR3OSCController(self.model, self.data, adapter)
        controller.step(action, GripperOperation.MOVE)
        actuator = self.model.actuator(config.gripper_actuator_name).id
        self.assertAlmostEqual(self.data.ctrl[actuator], -10.0, places=6)

    def test_controller_error_closes_viewer_before_propagation(self) -> None:
        class FailingController:
            def __init__(self, *_args, **_kwargs):
                pass

            def step(self, *_args, **_kwargs):
                raise RuntimeError("controller failed")

        class FakeViewer:
            def __init__(self):
                self.closed = False
                self.perturb = SimpleNamespace(
                    active=0,
                    active2=0,
                    select=-1,
                    refpos=np.zeros(3),
                    refquat=np.asarray((1.0, 0.0, 0.0, 0.0)),
                    refselpos=np.zeros(3),
                )

            def lock(self):
                return nullcontext()

            def sync(self):
                return None

            def is_running(self):
                return True

            def close(self):
                self.closed = True

        viewer = FakeViewer()
        viewer_module = SimpleNamespace(
            launch_passive=lambda *_args, **_kwargs: viewer
        )
        with (
            mock.patch("vistools.view_scene._load_scene", return_value=object()),
            mock.patch(
                "vistools.view_scene._make_model_and_data",
                return_value=(self.model, self.data),
            ),
            mock.patch("vistools.view_scene.NativeFR3OSCController", FailingController),
            mock.patch("vistools.view_scene._apply_camera"),
            mock.patch("vistools.view_scene._select_end_effector"),
            mock.patch("vistools.view_scene._set_target_marker"),
            mock.patch("vistools.view_scene._print_osc_help"),
            mock.patch("vistools.view_scene.importlib.import_module", return_value=viewer_module),
        ):
            with self.assertRaisesRegex(RuntimeError, "controller failed"):
                playground_main(["invention_real2sim/push_scene"])
        self.assertTrue(viewer.closed)

    def test_stale_abi_fails_before_viewer_launch(self) -> None:
        import_module = mock.Mock()
        with (
            mock.patch("vistools.view_scene._load_scene", return_value=object()),
            mock.patch(
                "vistools.view_scene._make_model_and_data",
                return_value=(self.model, self.data),
            ),
            mock.patch(
                "vistools.view_scene.NativeFR3OSCController",
                side_effect=RuntimeError("stale ABI; rebuild"),
            ),
            mock.patch("vistools.view_scene.importlib.import_module", import_module),
        ):
            with self.assertRaisesRegex(RuntimeError, "stale ABI"):
                playground_main(["invention_real2sim/push_scene"])
        import_module.assert_not_called()

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_interactive_native_reset_clears_torque_rate_state(self) -> None:
        initial_qpos = self.data.qpos.copy()
        initial_qvel = self.data.qvel.copy()
        action = self.adapter.action_from_data(self.model, self.data)
        action[0] += 0.02
        controller = NativeFR3OSCController(
            self.model,
            self.data,
            self.adapter,
        )
        first_observation = controller.step(action)
        np.testing.assert_array_equal(
            controller.controller_state,
            fr3_controller_state_from_observation(first_observation),
        )
        saved_controller_state = controller.controller_state
        controller.reset()
        reset_state = controller.controller_state
        np.testing.assert_array_equal(reset_state[:7], np.zeros(7))
        self.assertEqual(reset_state.shape, (len(FR3_CONTROLLER_STATE_NAMES),))
        controller.restore_controller_state(saved_controller_state)
        np.testing.assert_array_equal(
            controller.controller_state,
            saved_controller_state,
        )
        for _ in range(10):
            controller.step(action)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = initial_qpos
        self.data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.model, self.data)
        controller.reset()
        reset_observation = controller.step(action)

        np.testing.assert_array_equal(reset_observation, first_observation)


if __name__ == "__main__":
    unittest.main()
