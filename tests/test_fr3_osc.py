"""Tests for Cartesian-action FR3 operational-space control rollouts."""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from primitive_core.common import PrimitiveCompilerConfig
from primitive_core.compile import PrimitiveCompileSpec, compile_primitive
from primitive_core.fr3_osc import (
    FR3_ACTION_NAMES,
    FR3_CARTESIAN_ACTION_DIM,
    FR3_CONTROLLER_STATE_NAMES,
    FR3_HAND_JAW_AXIS_LOCAL,
    FR3_HAND_LONG_AXIS_LOCAL,
    FR3_HAND_SHORT_AXIS_LOCAL,
    FR3_OSC_OBSERVATION_NAMES,
    FR3OSCAdapter,
    FR3OSCConfig,
    fr3_controller_state_from_observation,
)
from primitive_core.fr3_safety import minimum_pose_trajectory_duration
from primitive_core.native_fr3_osc import (
    FR3_NATIVE_ABI_VERSION,
    NativeFR3OSCRolloutBackend,
    _validate_native_abi,
)
from primitive_core.policy import GripperIntent, GripperOperation
from primitive_core.rollout_control import (
    ContactPairTelemetry,
    RolloutTelemetrySpec,
)
from tasks.invention_real2sim.push_scene.scene import make_data, make_model

try:
    from mujoco_extensions.fr3_osc_rollout import NativeFR3OSCRollout as _Native
except ImportError:
    _NATIVE_FR3_OSC_AVAILABLE = False
else:
    _NATIVE_FR3_OSC_AVAILABLE = _Native is not None

_FR3_SCENE_NAMES = (
    "demo_1_scene",
    "demo_2_scene",
    "demo_3_scene",
    "demo_5_scene",
    "demo_7_scene",
    "demo_8_scene",
    "demo_6_scene",
    "flip_scene",
    "grasp_scene",
    "press_move_scene",
    "push_scene",
)


class FR3OSCActionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)

    def test_cartesian_action_contract(self) -> None:
        self.assertEqual(FR3_CARTESIAN_ACTION_DIM, 7)
        self.assertEqual(
            FR3_ACTION_NAMES,
            ("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
        )
        np.testing.assert_allclose(
            self.adapter.action_ctrlrange,
            (
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (0.0, 0.04),
            ),
        )

    def test_default_profile_matches_real_cartesian_controller(self) -> None:
        config = FR3OSCConfig()

        self.assertEqual(config.profile, "hardware_parity")
        self.assertEqual(config.translational_stiffness, (200.0,) * 3)
        self.assertEqual(config.rotational_stiffness, (30.0,) * 3)
        np.testing.assert_allclose(
            config.translational_damping,
            (2.0 * np.sqrt(200.0),) * 3,
        )
        np.testing.assert_allclose(
            config.rotational_damping,
            (2.0 * np.sqrt(30.0),) * 3,
        )
        self.assertEqual(config.nullspace_stiffness, 0.5)
        self.assertAlmostEqual(config.nullspace_damping, np.sqrt(2.0))
        self.assertEqual(config.controller_rate_hz, 1000.0)
        self.assertEqual(config.command_rate_hz, 50.0)
        self.assertEqual(config.equilibrium_filter_alpha, 0.005)
        self.assertEqual(config.torque_lowpass_cutoff_hz, 100.0)
        self.assertTrue(config.libfranka_rate_limiting)
        self.assertEqual(config.gripper_grasp_force, 40.0)
        self.assertEqual(config.gripper_maximum_force, 60.0)
        self.assertEqual(config.gripper_move_stiffness, 10_000.0)
        self.assertEqual(config.gripper_move_damping, 130.0)
        self.assertEqual(config.gripper_move_speed, 0.02)
        self.assertEqual(config.gripper_stall_velocity, 0.001)
        self.assertEqual(config.gripper_stall_position_error, 0.001)
        self.assertEqual(config.gripper_stall_steps, 5)

    def test_gripper_actuator_is_symmetric_fixed_tendon(self) -> None:
        actuator = self.model.actuator("fr3_hand").id
        self.assertEqual(
            self.model.actuator_trntype[actuator],
            mujoco.mjtTrn.mjTRN_TENDON,
        )
        np.testing.assert_allclose(self.model.wrap_prm, (0.5, 0.5))
        self.assertEqual(self.model.tendon_num[0], 2)

    def test_hand_frame_axes_match_collision_geometry_and_jaw_motion(self) -> None:
        hand_geom = self.model.geom("hand_collision").id
        mesh_id = int(self.model.geom_dataid[hand_geom])
        vertex_start = int(self.model.mesh_vertadr[mesh_id])
        vertex_count = int(self.model.mesh_vertnum[mesh_id])
        extents = np.ptp(
            self.model.mesh_vert[
                vertex_start : vertex_start + vertex_count
            ],
            axis=0,
        )

        self.assertEqual(int(np.argmin(extents)), 0)
        np.testing.assert_array_equal(FR3_HAND_SHORT_AXIS_LOCAL, (1.0, 0.0, 0.0))
        np.testing.assert_array_equal(FR3_HAND_JAW_AXIS_LOCAL, (0.0, 1.0, 0.0))
        np.testing.assert_array_equal(FR3_HAND_LONG_AXIS_LOCAL, (0.0, 0.0, 1.0))
        finger_joint = self.model.joint("finger_joint1").id
        np.testing.assert_array_equal(
            self.model.jnt_axis[finger_joint],
            FR3_HAND_JAW_AXIS_LOCAL,
        )

    def test_native_abi_metadata_matches_public_contract(self) -> None:
        import mujoco_extensions.fr3_osc_rollout as native

        self.assertEqual(native.ABI_VERSION, FR3_NATIVE_ABI_VERSION)
        self.assertEqual(native.ACTION_DIM, 7)
        self.assertEqual(native.CONTROLLER_STATE_DIM, 32)
        self.assertEqual(native.OBSERVATION_DIM, 52)

    def test_stale_native_abi_fails_with_loaded_path_and_rebuild_command(self) -> None:
        stale = SimpleNamespace(
            __file__="/tmp/stale_fr3_osc.so",
            ABI_VERSION=1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "stale_fr3_osc.so.*Rebuild",
        ):
            _validate_native_abi(stale)

    def test_gripper_force_configuration_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            FR3OSCConfig(gripper_grasp_force=61.0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            FR3OSCConfig(gripper_stall_steps=0)

    def test_legacy_profile_retains_pre_parity_behavior_parameters(self) -> None:
        config = FR3OSCConfig.legacy_native()

        self.assertEqual(config.profile, "legacy_native")
        self.assertEqual(config.translational_stiffness, (300.0,) * 3)
        self.assertEqual(config.rotational_stiffness, (25.0,) * 3)
        self.assertEqual(config.translational_damping, (77.0,) * 3)
        self.assertEqual(config.rotational_damping, (10.0,) * 3)
        self.assertEqual(config.nullspace_stiffness, 5.0)
        self.assertEqual(config.nullspace_damping, 2.0)

    def test_current_site_pose_round_trips_through_cartesian_action(self) -> None:
        action = self.adapter.action_from_data(self.model, self.data)
        position, quaternion = self.adapter.target_pose_world(action)

        np.testing.assert_allclose(
            position,
            self.data.site("grasp_site").xpos,
            atol=1e-10,
        )
        current_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            current_quaternion,
            self.data.site("grasp_site").xmat,
        )
        orientation_error = np.empty(3, dtype=float)
        mujoco.mju_subQuat(
            orientation_error,
            quaternion,
            current_quaternion,
        )
        np.testing.assert_allclose(orientation_error, 0.0, atol=1e-10)

    def test_out_of_range_handoff_uses_in_range_nullspace_recovery_target(
        self,
    ) -> None:
        joint_name = self.adapter.config.arm_joint_names[0]
        joint_id = self.model.joint(joint_name).id
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        lower = float(self.model.jnt_range[joint_id, 0])
        self.data.qpos[qpos_address] = lower - 0.01
        mujoco.mj_forward(self.model, self.data)

        recovered = FR3OSCAdapter.from_model_data(self.model, self.data)

        self.assertAlmostEqual(recovered.nullspace_configuration[0], lower)
        self.assertAlmostEqual(self.data.qpos[qpos_address], lower - 0.01)

    def test_rollout_telemetry_contract_is_explicit_and_ordered(self) -> None:
        telemetry = RolloutTelemetrySpec(
            contact_pairs=(
                ContactPairTelemetry("robot_fixture", (1, 2), (3,)),
            ),
            include_maximum_penetration=True,
        )

        self.assertEqual(
            telemetry.observation_names,
            ("robot_fixture", "maximum_penetration"),
        )
        self.assertEqual(telemetry.observation_slice(5), slice(5, 7))

    def test_empty_rollout_telemetry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            RolloutTelemetrySpec()


@unittest.skipUnless(
    _NATIVE_FR3_OSC_AVAILABLE,
    "native FR3 OSC extension has not been built",
)
class FR3OSCRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)
        self.x0 = np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    def _rollout(
        self,
        action: np.ndarray,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        try:
            states, sensors, _ = backend.rollout(
                self.x0[None],
                np.tile(action, (1, steps, 1)),
            )
        finally:
            backend.close()
        return states, sensors

    def test_holding_home_pose_is_compliant_and_stable(self) -> None:
        action = self.adapter.action_from_data(self.model, self.data)
        initial_position = self.data.site("grasp_site").xpos.copy()

        states, sensors = self._rollout(action, steps=200)

        self.assertEqual(states.shape, (1, 200, self.model.nq + self.model.nv))
        self.assertEqual(
            sensors.shape,
            (1, 200, len(FR3_OSC_OBSERVATION_NAMES)),
        )
        self.assertLess(
            float(np.linalg.norm(sensors[0, -1, :3] - initial_position)),
            0.002,
        )
        self.assertGreater(float(sensors[0, -1, 26]), 0.1)

    def test_cartesian_target_tracks_without_exposing_joint_actions(self) -> None:
        action = self.adapter.action_from_data(self.model, self.data)
        action[0] += 0.02
        target_position, _ = self.adapter.target_pose_world(action)
        initial_position = self.data.site("grasp_site").xpos.copy()

        _states, sensors = self._rollout(action, steps=500)

        self.assertLess(
            float(np.linalg.norm(sensors[0, -1, :3] - target_position)),
            float(np.linalg.norm(initial_position - target_position)),
        )
        self.assertGreater(
            float(np.linalg.norm(sensors[0, -1, :3] - initial_position)),
            0.001,
        )
        torques = sensors[0, :, 19:26]
        torque_limits = np.asarray((87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0))
        self.assertTrue(np.all(np.abs(torques) <= torque_limits + 1e-10))
        maximum_delta = (
            self.adapter.config.maximum_torque_rate * self.model.opt.timestep
        )
        initial_and_commanded = np.vstack((np.zeros(7), torques))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(initial_and_commanded, axis=0)))),
            maximum_delta + 1e-10,
        )

    def test_controller_backend_does_not_mutate_source_scene(self) -> None:
        original_gain = self.model.actuator_gainprm.copy()
        original_bias = self.model.actuator_biasprm.copy()
        original_friction = self.model.dof_frictionloss.copy()

        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        backend.close()

        np.testing.assert_array_equal(self.model.actuator_gainprm, original_gain)
        np.testing.assert_array_equal(self.model.actuator_biasprm, original_bias)
        np.testing.assert_array_equal(self.model.dof_frictionloss, original_friction)

    def test_parallel_candidates_receive_their_own_cartesian_targets(self) -> None:
        hold = self.adapter.action_from_data(self.model, self.data)
        move = hold.copy()
        move[0] += 0.02
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=2,
            adapter=self.adapter,
        )
        try:
            _states, sensors, _ = backend.rollout(
                np.tile(self.x0, (2, 1)),
                np.stack(
                    (
                        np.tile(hold, (200, 1)),
                        np.tile(move, (200, 1)),
                    )
                ),
            )
        finally:
            backend.close()

        self.assertGreater(
            float(np.linalg.norm(sensors[1, -1, :3] - sensors[0, -1, :3])),
            0.001,
        )

    def test_large_pose_error_generates_bounded_force(self) -> None:
        action = self.adapter.action_from_data(self.model, self.data)
        action[0] += 1.0

        _states, sensors = self._rollout(action, steps=3)

        forces = sensors[0, :, 13:16]
        self.assertGreater(float(np.max(np.abs(forces))), 10.0)
        self.assertTrue(
            np.all(
                np.abs(forces)
                <= self.adapter.config.maximum_task_force + 1e-10
            )
        )

    def test_shared_compiler_optimizes_seven_actions_for_eight_actuators(self) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        config = PrimitiveCompilerConfig(
            num_rollouts=2,
            max_parallel_rollouts=2,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            sigma_init=0.001,
            sigma_min=0.0001,
            sigma_max=0.01,
            horizon=0.02,
            optimize_initial_pose=False,
        )
        nominal = np.tile(initial_action, (config.num_nodes, 1))
        all_geom_ids = tuple(range(self.model.ngeom))
        telemetry = RolloutTelemetrySpec(
            contact_pairs=(
                ContactPairTelemetry(
                    "all_contact_count",
                    all_geom_ids,
                    all_geom_ids,
                ),
            ),
            include_maximum_penetration=True,
        )

        def cost(
            states: np.ndarray,
            actions: np.ndarray,
            controller_observations: np.ndarray,
        ) -> np.ndarray:
            del states, actions
            osc_observations = controller_observations[
                ..., -len(FR3_OSC_OBSERVATION_NAMES) :
            ]
            position_error = (
                osc_observations[:, -1, :3]
                - self.data.site("grasp_site").xpos
            )
            return np.sum(position_error**2, axis=1)

        result, best_knots = compile_primitive(
            PrimitiveCompileSpec(
                model=self.model,
                initial_state=self.x0,
                nominal_knots=nominal,
                sigma_init=np.full_like(nominal, 0.001),
                cost_fn=cost,
                action_ctrlrange=self.adapter.action_ctrlrange,
                rollout_control=self.adapter,
                rollout_telemetry=telemetry,
                fixed_first_action=True,
                gripper_intent=GripperIntent.HOLD,
                gripper_operations=np.full(2, GripperOperation.MOVE),
            ),
            config,
            verbose=False,
        )

        self.assertEqual(self.model.nu, 8)
        self.assertEqual(best_knots.shape, (2, 7))
        self.assertEqual(result.trajectory.controls.shape[1], 7)
        self.assertEqual(result.metrics["action_dim"], 7)
        self.assertEqual(
            result.metrics["low_level_controller"],
            "fr3_cartesian_impedance_osc",
        )
        self.assertEqual(
            result.metrics["controller_rollout_backend"],
            "native_cpp_fr3_osc",
        )
        self.assertFalse(result.metrics["controller_state_handoff"])
        self.assertIn("nominal_cost", result.metrics)
        self.assertIn("optimized_cost", result.metrics)
        self.assertIn("optimizer_improvement", result.metrics)
        self.assertIn("nominal_to_optimized_control_rms", result.metrics)
        self.assertIn("nominal_to_optimized_control_max", result.metrics)
        self.assertEqual(
            result.metrics["rollout_telemetry_names"],
            ["all_contact_count", "maximum_penetration"],
        )
        self.assertEqual(
            result.controller_state.shape, (len(FR3_CONTROLLER_STATE_NAMES),)
        )
        self.assertTrue(np.all(np.isfinite(result.controller_state)))

    def test_compiler_expands_unsafe_short_horizon_with_minimum_jerk(self) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        final_action = initial_action.copy()
        final_action[0] += 0.04
        nominal = np.stack((initial_action, final_action))
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            sigma_min=1e-6,
            sigma_max=1e-5,
            horizon=0.02,
            optimize_initial_pose=False,
        )

        result, _ = compile_primitive(
            PrimitiveCompileSpec(
                model=self.model,
                initial_state=self.x0,
                nominal_knots=nominal,
                sigma_init=np.zeros_like(nominal),
                cost_fn=lambda states, controls, observations: np.linalg.norm(
                    observations[:, -1, :3] - final_action[None, :3],
                    axis=1,
                ),
                action_ctrlrange=self.adapter.action_ctrlrange,
                rollout_control=self.adapter,
                fixed_first_action=True,
                gripper_intent=GripperIntent.HOLD,
                gripper_operations=np.full(2, GripperOperation.MOVE),
            ),
            config,
            verbose=False,
        )

        self.assertGreater(result.metrics["effective_horizon"], 0.02)
        self.assertEqual(result.metrics["control_interpolation"], "minimum_jerk")
        self.assertGreaterEqual(
            result.metrics["effective_horizon"],
            result.metrics["runtime_minimum_horizon"],
        )

    def test_compiler_projects_sampled_knots_into_the_effective_horizon(
        self,
    ) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        nominal = np.stack((initial_action, initial_action))
        sampled = nominal.copy()
        sampled[1, 0] += 0.30

        class FixedOptimizer:
            sigma = np.zeros_like(nominal)

            def sample_control_knots(self, _nominal: np.ndarray) -> np.ndarray:
                return sampled[None].copy()

            def update_nominal_knots(
                self,
                candidates: np.ndarray,
                _rewards: np.ndarray,
            ) -> np.ndarray:
                return candidates[0].copy()

        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            horizon=0.20,
            optimize_initial_pose=False,
        )
        with mock.patch(
            "primitive_core.compile._build_optimizer",
            return_value=FixedOptimizer(),
        ):
            result, best_knots = compile_primitive(
                PrimitiveCompileSpec(
                    model=self.model,
                    initial_state=self.x0,
                    nominal_knots=nominal,
                    sigma_init=np.zeros_like(nominal),
                    cost_fn=lambda states, controls, observations: (
                        -controls[:, -1, 0]
                    ),
                    action_ctrlrange=self.adapter.action_ctrlrange,
                    rollout_control=self.adapter,
                    fixed_first_action=True,
                    gripper_intent=GripperIntent.HOLD,
                    gripper_operations=np.full(2, GripperOperation.MOVE),
                ),
                config,
                verbose=False,
            )

        self.assertEqual(result.metrics["timing_limited_candidate_segments"], 1)
        self.assertTrue(result.metrics["retained_timing_valid"])
        self.assertLess(best_knots[1, 0], sampled[1, 0])
        self.assertLessEqual(
            minimum_pose_trajectory_duration(
                self.adapter,
                best_knots,
                np.asarray((0.0, 1.0)),
            ),
            result.metrics["effective_horizon"] + 1e-10,
        )

    def test_compiler_rejects_optimizer_control_of_the_initial_fr3_target(
        self,
    ) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        nominal = np.stack((initial_action, initial_action))
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            horizon=0.20,
            optimize_initial_pose=False,
        )

        with self.assertRaisesRegex(ValueError, "fixed_first_action=True"):
            compile_primitive(
                PrimitiveCompileSpec(
                    model=self.model,
                    initial_state=self.x0,
                    nominal_knots=nominal,
                    sigma_init=np.zeros_like(nominal),
                    cost_fn=lambda states, controls, observations: np.zeros(1),
                    action_ctrlrange=self.adapter.action_ctrlrange,
                    rollout_control=self.adapter,
                    gripper_intent=GripperIntent.HOLD,
                    gripper_operations=np.full(2, GripperOperation.MOVE),
                ),
                config,
                verbose=False,
            )

    def test_compiler_requires_explicit_fr3_gripper_intent(self) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        nominal = np.stack((initial_action, initial_action))
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            horizon=0.02,
            optimize_initial_pose=False,
        )

        with self.assertRaisesRegex(ValueError, "must declare GripperIntent"):
            compile_primitive(
                PrimitiveCompileSpec(
                    model=self.model,
                    initial_state=self.x0,
                    nominal_knots=nominal,
                    sigma_init=np.zeros_like(nominal),
                    cost_fn=lambda states, controls, observations: np.zeros(1),
                    action_ctrlrange=self.adapter.action_ctrlrange,
                    rollout_control=self.adapter,
                    fixed_first_action=True,
                ),
                config,
                verbose=False,
            )

    def test_cem_and_mppi_keep_actuating_gripper_profile_deterministic(self) -> None:
        initial_action = self.adapter.action_from_data(self.model, self.data)
        final_action = initial_action.copy()
        final_action[6] = self.adapter.gripper_aperture(0.25)
        nominal = np.stack((initial_action, final_action))

        for optimizer_name in ("cem", "mppi"):
            with self.subTest(optimizer=optimizer_name):
                config = PrimitiveCompilerConfig(
                    optimizer=optimizer_name,
                    num_rollouts=2,
                    max_parallel_rollouts=2,
                    num_nodes=2,
                    num_elites=1,
                    max_iters=1,
                    sigma_min=0.005,
                    sigma_max=0.01,
                    horizon=0.05,
                    optimize_initial_pose=False,
                )
                result, best_knots = compile_primitive(
                    PrimitiveCompileSpec(
                        model=self.model,
                        initial_state=self.x0,
                        nominal_knots=nominal,
                        sigma_init=np.full_like(nominal, 0.01),
                        cost_fn=lambda states, controls, observations: np.zeros(
                            states.shape[0]
                        ),
                        action_ctrlrange=self.adapter.action_ctrlrange,
                        rollout_control=self.adapter,
                        fixed_first_action=True,
                        gripper_intent=GripperIntent.ACTUATE,
                        gripper_operations=np.full(2, GripperOperation.GRASP),
                    ),
                    config,
                    verbose=False,
                )

                np.testing.assert_array_equal(best_knots[:, 6], nominal[:, 6])
                np.testing.assert_allclose(
                    result.trajectory.controls[:, 6],
                    result.trajectory.positions[:, 6],
                    atol=0.0,
                )
                self.assertEqual(result.metrics["gripper_intent"], "actuate")
                np.testing.assert_array_equal(
                    result.trajectory.gripper_operations,
                    np.full(
                        result.trajectory.times.shape,
                        GripperOperation.GRASP,
                        dtype=np.uint8,
                    ),
                )
                self.assertTrue(
                    result.metrics["gripper_optimizer_channel_fixed"]
                )


@unittest.skipUnless(
    _NATIVE_FR3_OSC_AVAILABLE,
    "native FR3 OSC extension has not been built",
)
class NativeFR3OSCRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)
        self.x0 = np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    def _varying_actions(self, steps: int = 160) -> np.ndarray:
        hold = self.adapter.action_from_data(self.model, self.data)
        first = np.tile(hold, (steps, 1))
        first[30:, 0] += 0.02
        first[80:, 1] -= 0.012
        first[120:, 5] += 0.04
        second = np.tile(hold, (steps, 1))
        second[20:, 2] += 0.015
        second[70:, 0] -= 0.018
        second[110:, 4] -= 0.035
        return np.stack((first, second))

    def test_large_pose_error_is_saturated_without_halting_rollout(self) -> None:
        action = self.adapter.action_from_data(self.model, self.data)
        action[0] += 1.0
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        try:
            states, observations, _ = backend.rollout(
                self.x0[None],
                np.tile(action, (1, 40, 1)),
            )
        finally:
            backend.close()

        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.all(np.isfinite(observations)))
        self.assertGreater(
            float(np.linalg.norm(states[0, -1] - self.x0)),
            0.0,
        )

    def test_native_varying_targets_are_finite_and_candidate_specific(self) -> None:
        actions = self._varying_actions()
        initial_states = np.tile(self.x0, (2, 1))
        native_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=2,
            adapter=self.adapter,
        )
        try:
            native_states, native_sensors, _ = native_backend.rollout(
                initial_states,
                actions,
            )
        finally:
            native_backend.close()

        self.assertTrue(np.all(np.isfinite(native_states)))
        self.assertTrue(np.all(np.isfinite(native_sensors)))
        self.assertGreater(
            float(
                np.linalg.norm(
                    native_sensors[1, -1, :3] - native_sensors[0, -1, :3]
                )
            ),
            0.001,
        )

    def test_hardware_profile_latches_input_at_exactly_50_hz(self) -> None:
        self.assertAlmostEqual(self.model.opt.timestep, 0.005)
        hold = self.adapter.action_from_data(self.model, self.data)
        actions = np.tile(hold, (9, 1))
        actions[:, 0] += np.arange(actions.shape[0]) * 0.001
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        try:
            _states, observations, _ = backend.rollout(
                self.x0[None],
                actions[None],
            )
        finally:
            backend.close()

        latched_x = observations[
            0, :, FR3_OSC_OBSERVATION_NAMES.index("latched_x")
        ]
        np.testing.assert_array_equal(latched_x[:4], np.full(4, actions[0, 0]))
        np.testing.assert_array_equal(latched_x[4:8], np.full(4, actions[4, 0]))
        self.assertEqual(latched_x[8], actions[8, 0])

    def test_reported_user_torque_excludes_mujoco_gravity_compensation(self) -> None:
        hold = self.adapter.action_from_data(self.model, self.data)
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        try:
            _states, observations, _ = backend.rollout(
                self.x0[None],
                hold[None, None],
            )
        finally:
            backend.close()

        np.testing.assert_allclose(observations[0, 0, 19:26], 0.0, atol=1e-12)

    def test_native_uses_general_base_rotation_for_task_force(self) -> None:
        frame_quaternion = np.empty(4, dtype=float)
        mujoco.mju_euler2Quat(
            frame_quaternion,
            np.asarray((0.2, -0.3, 0.4)),
            "xyz",
        )
        rotated_adapter = FR3OSCAdapter(
            frame_position_world=self.adapter.frame_position_world,
            frame_quaternion_world=tuple(frame_quaternion),
            nullspace_configuration=self.adapter.nullspace_configuration,
            gripper_range=self.adapter.gripper_range,
            config=self.adapter.config,
        )
        action = rotated_adapter.action_from_data(self.model, self.data)
        action[0] += 0.01
        native_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=rotated_adapter,
        )
        try:
            _native_states, native_sensors, _ = native_backend.rollout(
                self.x0[None],
                action[None, None, :],
            )
        finally:
            native_backend.close()

        expected_force = (
            rotated_adapter.frame_rotation_world[:, 0]
            * 0.01
            * np.asarray(rotated_adapter.config.translational_stiffness)
            * (1.0 - (1.0 - rotated_adapter.config.equilibrium_filter_alpha) ** 4)
        )
        np.testing.assert_allclose(
            native_sensors[0, 0, 13:16],
            expected_force,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_native_rollouts_reset_controller_state_and_preserve_model(self) -> None:
        original_gain = self.model.actuator_gainprm.copy()
        original_bias = self.model.actuator_biasprm.copy()
        original_friction = self.model.dof_frictionloss.copy()
        actions = self._varying_actions(80)
        initial_states = np.tile(self.x0, (2, 1))
        backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=2,
            adapter=self.adapter,
        )
        try:
            first_states, first_sensors, _ = backend.rollout(initial_states, actions)
            second_states, second_sensors, _ = backend.rollout(initial_states, actions)
        finally:
            backend.close()

        np.testing.assert_array_equal(first_states, second_states)
        np.testing.assert_array_equal(first_sensors, second_sensors)
        np.testing.assert_array_equal(self.model.actuator_gainprm, original_gain)
        np.testing.assert_array_equal(self.model.actuator_biasprm, original_bias)
        np.testing.assert_array_equal(self.model.dof_frictionloss, original_friction)

    def test_native_rollout_restores_torque_rate_state_across_phases(self) -> None:
        actions = self._varying_actions(120)[:1]

        def rollout(
            initial_state: np.ndarray,
            phase_actions: np.ndarray,
            controller_state: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            backend = NativeFR3OSCRolloutBackend(
                self.model,
                num_threads=1,
                adapter=self.adapter,
                initial_controller_state=controller_state,
            )
            try:
                states, observations, _ = backend.rollout(
                    initial_state[None],
                    phase_actions,
                )
            finally:
                backend.close()
            return states, observations

        full_states, full_observations = rollout(self.x0, actions)
        first_states, first_observations = rollout(self.x0, actions[:, :55])
        controller_state = fr3_controller_state_from_observation(
            first_observations[0, -1]
        )
        second_states, second_observations = rollout(
            first_states[0, -1],
            actions[:, 55:],
            controller_state,
        )

        np.testing.assert_array_equal(
            np.concatenate((first_states, second_states), axis=1),
            full_states,
        )
        np.testing.assert_array_equal(
            np.concatenate((first_observations, second_observations), axis=1),
            full_observations,
        )

    def test_opt_in_contact_telemetry_matches_post_step_state_replay(self) -> None:
        all_geom_ids = tuple(range(self.model.ngeom))
        telemetry = RolloutTelemetrySpec(
            contact_pairs=(
                ContactPairTelemetry(
                    "all_contact_count",
                    all_geom_ids,
                    all_geom_ids,
                ),
            ),
            include_maximum_penetration=True,
        )
        actions = self._varying_actions(40)[:1]

        plain_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        telemetry_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
            telemetry=telemetry,
        )
        try:
            plain_states, plain_sensors, _ = plain_backend.rollout(
                self.x0[None],
                actions,
            )
            telemetry_states, telemetry_sensors, _ = telemetry_backend.rollout(
                self.x0[None],
                actions,
            )
        finally:
            plain_backend.close()
            telemetry_backend.close()

        np.testing.assert_array_equal(telemetry_states, plain_states)
        np.testing.assert_array_equal(
            telemetry_sensors[..., : self.model.nsensordata],
            plain_sensors[..., : self.model.nsensordata],
        )
        np.testing.assert_array_equal(
            telemetry_sensors[..., -len(FR3_OSC_OBSERVATION_NAMES) :],
            plain_sensors[..., -len(FR3_OSC_OBSERVATION_NAMES) :],
        )
        self.assertEqual(
            telemetry_sensors.shape[-1],
            plain_sensors.shape[-1] + len(telemetry.observation_names),
        )

        expected = []
        replay_data = mujoco.MjData(self.model)
        for state in telemetry_states[0]:
            replay_data.qpos[:] = state[: self.model.nq]
            replay_data.qvel[:] = state[self.model.nq :]
            mujoco.mj_forward(self.model, replay_data)
            contact_count = sum(
                int(replay_data.contact[index].geom[0] >= 0)
                * int(replay_data.contact[index].geom[1] >= 0)
                for index in range(replay_data.ncon)
            )
            maximum_penetration = max(
                (
                    -float(replay_data.contact[index].dist)
                    for index in range(replay_data.ncon)
                ),
                default=0.0,
            )
            expected.append((contact_count, max(0.0, maximum_penetration)))

        np.testing.assert_allclose(
            telemetry.view(telemetry_sensors, self.model.nsensordata)[0],
            np.asarray(expected),
            rtol=0.0,
            atol=1e-12,
        )

class FR3OSCSceneIntegrationTests(unittest.TestCase):
    def test_demo_eight_corridor_boundaries_are_static(self) -> None:
        scene = importlib.import_module("tasks.invention_real2sim.demo_6_scene.scene")
        model = scene.make_model()
        for body_name in (
            "brown_cardboard_box",
            "white_product_box",
        ):
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            self.assertGreaterEqual(body_id, 0)
            self.assertEqual(int(model.body_jntnum[body_id]), 0)

        target_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "blue_product_box",
        )
        self.assertGreaterEqual(target_id, 0)
        self.assertEqual(int(model.body_jntnum[target_id]), 1)

    def test_real2sim_arm_scenes_share_the_osc_action_contract(self) -> None:
        for scene_name in _FR3_SCENE_NAMES:
            with self.subTest(scene=scene_name):
                scene = importlib.import_module(f"tasks.invention_real2sim.{scene_name}.scene")
                model = scene.make_model()
                data = scene.make_data(model)
                adapter = FR3OSCAdapter.from_model_data(model, data)

                self.assertEqual(model.nu, 8)
                self.assertEqual(adapter.action_from_data(model, data).shape, (7,))

    @unittest.skipUnless(
        _NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_native_backend_rolls_out_every_real2sim_arm_scene(self) -> None:
        for scene_name in _FR3_SCENE_NAMES:
            with self.subTest(scene=scene_name):
                scene = importlib.import_module(f"tasks.invention_real2sim.{scene_name}.scene")
                model = scene.make_model()
                data = scene.make_data(model)
                adapter = FR3OSCAdapter.from_model_data(model, data)
                initial_state = np.concatenate((data.qpos, data.qvel))
                action = adapter.action_from_data(model, data)
                backend = NativeFR3OSCRolloutBackend(model, 1, adapter)
                try:
                    states, sensors, _ = backend.rollout(
                        initial_state[None],
                        np.tile(action, (1, 20, 1)),
                    )
                finally:
                    backend.close()

                self.assertEqual(states.shape, (1, 20, model.nq + model.nv))
                self.assertEqual(
                    sensors.shape[-1],
                    model.nsensordata + len(FR3_OSC_OBSERVATION_NAMES),
                )
                self.assertTrue(np.all(np.isfinite(states)))
                self.assertTrue(np.all(np.isfinite(sensors)))


if __name__ == "__main__":
    unittest.main()
