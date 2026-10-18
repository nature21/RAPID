from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from primitive_core.common import PrimitiveCompilerConfig
from primitive_core.compile import PrimitiveCompileSpec, compile_primitive
from primitive_core.fr3_osc import FR3_OSC_OBSERVATION_NAMES
from primitive_core.fr3_safety import (
    AllowedContactPair,
    AllowedProximityPair,
    FR3SafetyLimits,
    FR3SafetySpec,
    audit_fr3_trajectory,
    enforce_pose_trajectory_timing,
    minimum_pose_trajectory_duration,
)
from primitive_core.policy import GripperIntent, GripperOperation


class _PoseAdapter:
    action_dim = 7

    @staticmethod
    def target_pose_world(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_euler2Quat(quaternion, np.asarray(action[3:6]), "xyz")
        return np.asarray(action[:3], dtype=float), quaternion


class RetainedFR3SafetyAuditTests(unittest.TestCase):
    @staticmethod
    def _contact_model() -> tuple[mujoco.MjModel, np.ndarray, int, int]:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.01" gravity="0 0 0"/>
              <worldbody>
                <geom name="floor" type="plane" size="1 1 0.1"/>
                <body name="moving" pos="0 0 0.04">
                  <joint name="lift" type="slide" axis="0 0 1"/>
                  <geom name="robot_sphere" type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state = np.concatenate((data.qpos, data.qvel))
        return (
            model,
            state,
            model.geom("robot_sphere").id,
            model.geom("floor").id,
        )

    def test_unapproved_contact_fails_and_declared_contact_passes(self) -> None:
        model, state, robot_geom, floor_geom = self._contact_model()
        unsafe_spec = FR3SafetySpec((robot_geom,), (robot_geom,))
        unsafe = audit_fr3_trajectory(model, state[None], unsafe_spec)

        self.assertFalse(unsafe.valid)
        self.assertGreater(unsafe.unapproved_contact_steps, 0)
        self.assertEqual(
            unsafe.dominant_unapproved_contact_pair,
            ("floor", "robot_sphere"),
        )

        allowed_spec = FR3SafetySpec(
            (robot_geom,),
            (robot_geom,),
            (AllowedContactPair((robot_geom,), (floor_geom,)),),
        )
        allowed = audit_fr3_trajectory(model, state[None], allowed_spec)
        self.assertTrue(allowed.valid)

    def test_declared_proximity_relaxes_clearance_but_never_contact(self) -> None:
        model, state, robot_geom, floor_geom = self._contact_model()
        contact_free_state = state.copy()
        contact_free_state[0] = 0.014  # 4 mm sphere-to-plane separation.

        ordinary = audit_fr3_trajectory(
            model,
            contact_free_state[None],
            FR3SafetySpec((robot_geom,), (robot_geom,)),
        )
        self.assertFalse(ordinary.valid)
        self.assertAlmostEqual(ordinary.minimum_environment_clearance, 0.004)

        proximity_spec = FR3SafetySpec(
            (robot_geom,),
            (robot_geom,),
            allowed_proximity_pairs=(
                AllowedProximityPair((robot_geom,), (floor_geom,)),
            ),
        )
        proximity = audit_fr3_trajectory(
            model,
            contact_free_state[None],
            proximity_spec,
        )
        self.assertTrue(proximity.valid)
        self.assertEqual(proximity.minimum_environment_clearance, np.inf)
        self.assertAlmostEqual(
            proximity.minimum_declared_proximity_clearance,
            0.004,
        )

        contact = audit_fr3_trajectory(model, state[None], proximity_spec)
        self.assertFalse(contact.valid)
        self.assertGreater(contact.unapproved_contact_steps, 0)

    def test_proximity_declaration_does_not_relax_other_robot_links(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.01" gravity="0 0 0"/>
              <worldbody>
                <geom name="floor" type="plane" size="1 1 0.1"/>
                <body name="moving" pos="0 0 0.04">
                  <joint name="lift" type="slide" axis="0 0 1"/>
                  <geom name="tool" type="sphere" size="0.05"/>
                  <geom name="other_link" pos="0.2 0 0"
                        type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        data.qpos[0] = 0.014
        mujoco.mj_forward(model, data)
        state = np.concatenate((data.qpos, data.qvel))
        tool = model.geom("tool").id
        other_link = model.geom("other_link").id
        floor = model.geom("floor").id
        audit = audit_fr3_trajectory(
            model,
            state[None],
            FR3SafetySpec(
                (tool, other_link),
                (tool, other_link),
                allowed_proximity_pairs=(
                    AllowedProximityPair((tool,), (floor,)),
                ),
            ),
        )

        self.assertFalse(audit.valid)
        self.assertAlmostEqual(audit.minimum_environment_clearance, 0.004)
        self.assertAlmostEqual(
            audit.minimum_declared_proximity_clearance,
            0.004,
        )

    def test_geometry_tolerance_allows_only_bounded_shallow_penetration(
        self,
    ) -> None:
        model, state, robot_geom, floor_geom = self._contact_model()
        limits = FR3SafetyLimits(geometry_numerical_tolerance=0.00025)
        spec = FR3SafetySpec(
            (robot_geom,),
            (robot_geom,),
            allowed_proximity_pairs=(
                AllowedProximityPair(
                    (robot_geom,),
                    (floor_geom,),
                    minimum_clearance=0.0,
                ),
            ),
            limits=limits,
        )
        shallow = state.copy()
        shallow[0] = 0.00985  # 0.15 mm penetration.
        deep = state.copy()
        deep[0] = 0.00960  # 0.40 mm penetration.

        shallow_audit = audit_fr3_trajectory(model, shallow[None], spec)
        deep_audit = audit_fr3_trajectory(model, deep[None], spec)

        self.assertTrue(shallow_audit.valid)
        self.assertAlmostEqual(
            shallow_audit.geometry_numerical_tolerance,
            0.00025,
        )
        self.assertFalse(deep_audit.valid)
        self.assertGreater(deep_audit.unapproved_contact_steps, 0)
        self.assertFalse(deep_audit.declared_proximity_clearance_valid)

    def test_hold_allows_passive_aperture_deflection_with_constant_command(
        self,
    ) -> None:
        model, initial_state, robot_geom, floor_geom = self._contact_model()

        class FakeAdapter:
            name = "fr3_cartesian_impedance_osc"
            action_dim = 7
            observation_names = FR3_OSC_OBSERVATION_NAMES
            action_ctrlrange = np.asarray(
                [(-1.0, 1.0)] * 6 + [(0.0, 0.04)],
                dtype=float,
            )
            gripper_range = (0.0, 0.04)

            @staticmethod
            def target_pose_world(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return np.asarray(action[:3]), np.asarray((1.0, 0.0, 0.0, 0.0))

            @staticmethod
            def gripper_joint_addresses(
                _model: mujoco.MjModel,
            ) -> tuple[int, int]:
                return 0, 0

            @staticmethod
            def gripper_apertures(
                _model: mujoco.MjModel,
                states: np.ndarray,
            ) -> np.ndarray:
                return np.asarray(states)[..., 0]

        class FakeBackend:
            implementation_name = "native_cpp_fr3_osc"

            def rollout(
                self,
                x0: np.ndarray,
                controls: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, None]:
                states = np.repeat(x0[:, None, :], controls.shape[1], axis=1)
                if controls.shape[1] > 1:
                    states[:, 1:, 0] = 0.001
                observations = np.zeros(
                    (
                        controls.shape[0],
                        controls.shape[1],
                        len(FR3_OSC_OBSERVATION_NAMES),
                    ),
                    dtype=float,
                )
                return states, observations, None

            @staticmethod
            def close() -> None:
                return None

        adapter = FakeAdapter()
        nominal = np.zeros((2, 7), dtype=float)
        safety = FR3SafetySpec(
            (robot_geom,),
            (robot_geom,),
            allowed_contact_pairs=(
                AllowedContactPair((robot_geom,), (floor_geom,)),
            ),
        )
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            sigma_min=1e-4,
            sigma_max=1e-3,
            horizon=0.02,
            optimize_initial_pose=False,
        )

        retained_audit = SimpleNamespace(
            valid=False,
            minimum_environment_clearance=-0.001,
            minimum_enforced_environment_clearance=-0.001,
            minimum_declared_proximity_clearance=float("inf"),
            self_collision_steps=0,
            dominant_self_collision_pair=None,
            unapproved_contact_steps=1,
            dominant_unapproved_contact_pair=("floor", "robot_sphere"),
        )
        with (
            mock.patch(
                "primitive_core.compile._make_rollout_backend",
                return_value=(FakeBackend(), 1),
            ),
            mock.patch(
                "primitive_core.compile.audit_fr3_trajectory",
                return_value=retained_audit,
            ),
        ):
            result, _ = compile_primitive(
                PrimitiveCompileSpec(
                    model=model,
                    initial_state=initial_state,
                    nominal_knots=nominal,
                    sigma_init=np.full_like(nominal, 0.001),
                    cost_fn=lambda states, controls, observations: np.zeros(1),
                    action_ctrlrange=adapter.action_ctrlrange,
                    rollout_control=adapter,
                    rollout_safety=safety,
                    fixed_first_action=True,
                    gripper_intent=GripperIntent.HOLD,
                    gripper_operations=np.full(2, GripperOperation.MOVE),
                ),
                config,
                verbose=False,
            )

        self.assertTrue(
            result.metrics["retained_gripper_contact_audit_valid"]
        )
        self.assertFalse(result.metrics["whole_arm_audit_valid"])
        self.assertTrue(result.metrics["audit_advisory_triggered"])
        self.assertIn(
            "whole_arm_clearance_or_contact",
            result.metrics["audit_advisories"],
        )
        self.assertEqual(
            result.metrics[
                "commanded_aperture_motion_during_target_contact_steps"
            ],
            0,
        )
        self.assertGreater(
            result.metrics[
                "actual_aperture_motion_during_target_contact_steps"
            ],
            0,
        )

    def test_opposing_fingers_are_permitted_but_drift_is_not(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option gravity="0 0 0"/>
              <worldbody>
                <body name="left"><joint name="left_q" type="slide"
                  axis="1 0 0" range="0 .04"/>
                  <geom name="left_pad" type="sphere" size=".01"/></body>
                <body name="right"><joint name="right_q" type="slide"
                  axis="1 0 0" range="0 .04"/>
                  <geom name="right_pad" type="sphere" size=".01"/></body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        data.qpos[:] = -0.0001
        mujoco.mj_forward(model, data)
        state = np.concatenate((data.qpos, data.qvel))
        left = model.geom("left_pad").id
        right = model.geom("right_pad").id
        permitted = AllowedContactPair((left,), (right,))
        spec = FR3SafetySpec(
            (left, right),
            (left, right),
            permitted_gripper_contact_pairs=(permitted,),
            gripper_qpos_addresses=(0, 1),
            gripper_joint_lower_limits=(0.0, 0.0),
        )
        audit = audit_fr3_trajectory(model, state[None], spec)
        self.assertTrue(audit.valid)
        self.assertEqual(audit.permitted_fingertip_contact_steps, 1)
        self.assertAlmostEqual(audit.maximum_closed_stop_penetration, 0.0001)

        ordinary = audit_fr3_trajectory(
            model,
            state[None],
            FR3SafetySpec((left, right), (left, right)),
        )
        self.assertFalse(ordinary.valid)
        self.assertEqual(ordinary.self_collision_steps, 1)

        drifted = state.copy()
        drifted[:2] = (0.0, 0.001)
        drift_audit = audit_fr3_trajectory(model, drifted[None], spec)
        self.assertFalse(drift_audit.valid)
        self.assertAlmostEqual(
            drift_audit.maximum_finger_synchronization_error,
            0.001,
        )

    def test_sampled_knots_are_projected_into_the_horizon_timing_envelope(
        self,
    ) -> None:
        adapter = _PoseAdapter()
        requested = np.zeros((2, 7), dtype=float)
        requested[1, 0] = 0.30
        requested[1, 5] = 0.80
        fractions = np.asarray((0.0, 1.0))
        horizon = 0.20

        projected, limited_segments = enforce_pose_trajectory_timing(
            adapter,
            requested,
            fractions,
            horizon,
        )

        self.assertEqual(limited_segments, 1)
        self.assertGreater(
            minimum_pose_trajectory_duration(adapter, requested, fractions),
            horizon,
        )
        self.assertLessEqual(
            minimum_pose_trajectory_duration(adapter, projected, fractions),
            horizon + 1e-10,
        )
        self.assertLess(projected[1, 0], requested[1, 0])
        np.testing.assert_array_equal(requested[1, (0, 5)], (0.30, 0.80))


if __name__ == "__main__":
    unittest.main()
