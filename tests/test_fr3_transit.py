"""Tests for deterministic native-OSC precontact transit."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from instantiation_core import SceneBinding
from primitive_core.common import (
    CompileResult,
    PrimitiveCompilerConfig,
    ToolTrajectory,
)
from primitive_core.compile import PrimitiveCompileSpec, compile_primitive
from primitive_core.fr3_osc import (
    FR3_CONTROLLER_STATE_NAMES,
    FR3OSCAdapter,
    fr3_controller_state_from_observation,
)
from primitive_core.fr3_transit import prepend_fr3_transit, rollout_fr3_transit
from primitive_core.policy import (
    BoundExecutionContext,
    GripperIntent,
    GripperOperation,
)
from primitive_core.transit import (
    _audit_free_phase,
    Approach,
    AxisRef,
    ConfigureGripper,
    RoleFrame,
    ScaleExpr,
    Settle,
    TransitSpec,
    execute_transit,
)
from primitive_core.native_fr3_osc import NativeFR3OSCRolloutBackend
from tasks.invention_real2sim.push_scene.scene import make_data, make_model

try:
    from mujoco_extensions.fr3_osc_rollout import NativeFR3OSCRollout as _Native
except ImportError:
    _NATIVE_FR3_OSC_AVAILABLE = False
else:
    _NATIVE_FR3_OSC_AVAILABLE = _Native is not None


@unittest.skipUnless(
    _NATIVE_FR3_OSC_AVAILABLE,
    "native FR3 OSC extension has not been built",
)
class FR3TransitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))
        self.start = self.adapter.action_from_data(self.model, self.data)

    def _transit(self):
        clearance = self.start.copy()
        clearance[2] += 0.008
        precontact = clearance.copy()
        precontact[0] += 0.012
        return rollout_fr3_transit(
            self.model,
            self.x0,
            self.adapter,
            np.stack((self.start, clearance, precontact)),
            (0.25, 0.25),
            settle_duration=0.30,
        )

    def test_transit_is_physical_and_preserves_source_model(self) -> None:
        original_gain = self.model.actuator_gainprm.copy()
        original_bias = self.model.actuator_biasprm.copy()
        original_friction = self.model.dof_frictionloss.copy()

        transit = self._transit()

        self.assertTrue(transit.result.metrics["transit_simulated"])
        self.assertTrue(transit.result.metrics["transit_valid"])
        self.assertLess(
            transit.result.metrics["terminal_position_error"],
            0.02,
        )
        self.assertGreaterEqual(
            transit.result.metrics["settle_used_duration"],
            transit.result.metrics["settle_minimum_duration"],
        )
        self.assertLessEqual(
            transit.result.metrics["settle_used_duration"],
            transit.result.metrics["settle_maximum_duration"],
        )
        self.assertEqual(
            transit.result.metrics["controller_rollout_backend"],
            "native_cpp_fr3_osc",
        )
        np.testing.assert_array_equal(
            transit.result.trajectory.gripper_operations,
            np.zeros(transit.result.best_states.shape[0], dtype=np.uint8),
        )
        self.assertEqual(transit.terminal_state.shape, (self.model.nq + self.model.nv,))
        np.testing.assert_array_equal(
            transit.controller_state,
            fr3_controller_state_from_observation(
                transit.controller_observations[-1]
            ),
        )
        np.testing.assert_array_equal(
            transit.result.controller_state,
            transit.controller_state,
        )
        np.testing.assert_array_equal(self.model.actuator_gainprm, original_gain)
        np.testing.assert_array_equal(self.model.actuator_biasprm, original_bias)
        np.testing.assert_array_equal(self.model.dof_frictionloss, original_friction)

    def test_constant_aperture_transit_can_preserve_grasp(self) -> None:
        target = self.start.copy()
        target[2] += 0.004
        transit = rollout_fr3_transit(
            self.model,
            self.x0,
            self.adapter,
            np.stack((self.start, target)),
            (0.15,),
            gripper_operation=GripperOperation.GRASP,
            settle_duration=0.10,
        )
        np.testing.assert_array_equal(
            transit.result.trajectory.gripper_operations,
            np.ones(transit.result.best_states.shape[0], dtype=np.uint8),
        )

    def test_finite_nonconverged_transit_is_retained_as_advisory(self) -> None:
        target = self.start.copy()
        target[0] += 0.012

        transit = rollout_fr3_transit(
            self.model,
            self.x0,
            self.adapter,
            np.stack((self.start, target)),
            (0.25,),
            settle_duration=0.10,
            minimum_joint_margin=100.0,
        )

        self.assertTrue(transit.result.metrics["transit_valid"])
        self.assertFalse(transit.result.metrics["transit_convergence_valid"])

    def test_final_osc_collision_and_convergence_findings_are_advisory(
        self,
    ) -> None:
        transit = self._transit()
        transit.result.metrics["transit_convergence_valid"] = False
        collision_audit = SimpleNamespace(
            valid=False,
            self_collision_steps=0,
            unapproved_contact_steps=3,
            minimum_environment_clearance=-0.001,
            minimum_enforced_environment_clearance=-0.001,
            minimum_environment_clearance_pair=("hand", "obstacle"),
            minimum_environment_clearance_step=5,
            minimum_declared_proximity_clearance=0.0,
            minimum_declared_proximity_margin=-0.002,
            declared_proximity_clearance_valid=False,
            dominant_unapproved_contact_pair=("hand", "obstacle"),
            permitted_fingertip_contact_steps=0,
            maximum_closed_stop_penetration=0.0,
            maximum_finger_synchronization_error=0.0,
        )

        with patch(
            "primitive_core.transit.audit_fr3_trajectory",
            return_value=collision_audit,
        ):
            clean = _audit_free_phase(self.model, transit, object())

        self.assertFalse(clean)
        self.assertTrue(transit.result.metrics["transit_valid"])
        self.assertTrue(transit.result.metrics["audit_advisory_triggered"])
        self.assertEqual(
            transit.result.metrics["audit_advisories"],
            [
                "transit_convergence_or_joint_margin",
                "whole_arm_clearance_or_contact",
            ],
        )
        self.assertEqual(
            transit.result.metrics["final_osc_rollout_advisory"]["findings"],
            transit.result.metrics["audit_advisories"],
        )

    def test_transit_handoff_and_prepend_match_continuous_rollout(self) -> None:
        transit = self._transit()
        effect_actions = np.tile(transit.terminal_action, (40, 1))
        effect_actions[:, 1] += np.linspace(0.0, 0.006, 40)
        effect_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
            initial_controller_state=transit.controller_state,
        )
        try:
            effect_states, effect_observations, _ = effect_backend.rollout(
                transit.terminal_state[None],
                effect_actions[None],
            )
        finally:
            effect_backend.close()
        timestep = float(self.model.opt.timestep)
        effect = CompileResult(
            trajectory=ToolTrajectory(
                times=np.arange(40) * timestep,
                positions=effect_actions.copy(),
                controls=effect_actions.copy(),
                gripper_operations=np.zeros(40, dtype=np.uint8),
            ),
            metrics={"success": True, "best_cost": 1.0},
            best_states=effect_states[0],
            cost_history=np.asarray((1.0,)),
            controller_state=fr3_controller_state_from_observation(
                effect_observations[0, -1]
            ),
        )
        transit.result.metrics.update(
            {
                "transit_convergence_valid": False,
                "whole_arm_audit_valid": False,
                "self_collision_steps": 0,
                "unapproved_robot_contact_steps": 2,
                "permitted_fingertip_contact_steps": 3,
                "maximum_closed_stop_penetration": 0.0002,
                "maximum_finger_synchronization_error": 0.0001,
                "audit_advisories": ["whole_arm_clearance_or_contact"],
                "audit_advisory_triggered": True,
            }
        )
        effect.metrics.update(
            {
                "transit_convergence_valid": True,
                "whole_arm_audit_valid": True,
                "self_collision_steps": 0,
                "unapproved_contact_steps": 0,
                "permitted_fingertip_contact_steps": 4,
                "maximum_closed_stop_penetration": 0.0001,
                "maximum_finger_synchronization_error": 0.0002,
                "audit_advisories": [],
                "audit_advisory_triggered": False,
            }
        )

        combined = prepend_fr3_transit(transit, effect)
        full_actions = np.concatenate(
            (transit.result.trajectory.controls, effect_actions),
            axis=0,
        )
        full_backend = NativeFR3OSCRolloutBackend(
            self.model,
            num_threads=1,
            adapter=self.adapter,
        )
        try:
            full_states, _full_observations, _ = full_backend.rollout(
                self.x0[None],
                full_actions[None],
            )
        finally:
            full_backend.close()

        np.testing.assert_allclose(combined.best_states, full_states[0], atol=1e-12)
        np.testing.assert_array_equal(combined.trajectory.controls, full_actions)
        self.assertEqual(
            combined.metrics["effect_start_index"],
            transit.result.best_states.shape[0],
        )
        self.assertTrue(combined.metrics["success"])
        self.assertFalse(combined.metrics["transit_convergence_valid"])
        self.assertFalse(combined.metrics["whole_arm_audit_valid"])
        self.assertEqual(combined.metrics["unapproved_contact_steps"], 2)
        self.assertEqual(combined.metrics["permitted_fingertip_contact_steps"], 7)
        self.assertEqual(combined.metrics["maximum_closed_stop_penetration"], 0.0002)
        self.assertEqual(
            combined.metrics["maximum_finger_synchronization_error"],
            0.0002,
        )
        self.assertTrue(combined.metrics["audit_advisory_triggered"])
        self.assertEqual(
            combined.metrics["audit_advisories"],
            ["whole_arm_clearance_or_contact"],
        )
        self.assertEqual(combined.cost_history.tolist(), [1.0])
        np.testing.assert_array_equal(
            combined.controller_state,
            effect.controller_state,
        )

        discontinuous_controls = effect_actions.copy()
        discontinuous_controls[0, 6] -= 0.001
        discontinuous = CompileResult(
            trajectory=ToolTrajectory(
                times=effect.trajectory.times,
                positions=discontinuous_controls.copy(),
                controls=discontinuous_controls,
                gripper_operations=np.zeros(40, dtype=np.uint8),
            ),
            metrics=dict(effect.metrics),
            best_states=effect.best_states,
            cost_history=effect.cost_history,
            controller_state=effect.controller_state,
        )
        with self.assertRaisesRegex(ValueError, "phase-boundary jump"):
            prepend_fr3_transit(transit, discontinuous)

    def test_compiler_reports_transit_controller_state_handoff(self) -> None:
        transit = self._transit()
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            sigma_init=0.0001,
            sigma_min=0.00001,
            sigma_max=0.001,
            horizon=0.02,
            optimize_initial_pose=False,
        )
        nominal = np.tile(transit.terminal_action, (config.num_nodes, 1))

        def cost(
            states: np.ndarray,
            actions: np.ndarray,
            observations: np.ndarray,
        ) -> np.ndarray:
            del states, actions
            return np.sum(observations[:, -1, 7:13] ** 2, axis=1)

        result, _ = compile_primitive(
            PrimitiveCompileSpec(
                model=self.model,
                initial_state=transit.terminal_state,
                nominal_knots=nominal,
                sigma_init=np.full_like(nominal, 0.0001),
                cost_fn=cost,
                action_ctrlrange=self.adapter.action_ctrlrange,
                rollout_control=self.adapter,
                initial_controller_state=transit.controller_state,
                fixed_first_action=True,
                gripper_intent=GripperIntent.HOLD,
                gripper_operations=np.full(2, GripperOperation.MOVE),
            ),
            config,
            verbose=False,
        )

        self.assertTrue(result.metrics["controller_state_handoff"])
        self.assertEqual(
            result.controller_state.shape, (len(FR3_CONTROLLER_STATE_NAMES),)
        )
        self.assertTrue(result.metrics["gripper_optimizer_channel_fixed"])

    def test_gripper_configuration_settles_before_handoff(self) -> None:
        configured = self.start.copy()
        configured[6] = self.adapter.gripper_aperture(0.25)
        transit = rollout_fr3_transit(
            self.model,
            self.x0,
            self.adapter,
            np.stack((self.start, configured)),
            (1.0,),
            settle_duration=0.20,
        )

        self.assertTrue(transit.result.metrics["settle_converged"])
        self.assertLessEqual(
            transit.result.metrics["terminal_gripper_position_error"],
            transit.result.metrics["terminal_gripper_position_tolerance"],
        )
        self.assertLessEqual(
            transit.result.metrics["terminal_gripper_speed"],
            transit.result.metrics["terminal_gripper_speed_tolerance"],
        )
        gripper_qpos, gripper_dof = self.adapter.gripper_joint_addresses(
            self.model
        )
        self.assertAlmostEqual(
            transit.result.metrics["terminal_gripper_position_error"],
            abs(
                transit.result.best_states[-1, gripper_qpos]
                - configured[6]
            ),
        )
        self.assertAlmostEqual(
            transit.result.metrics["terminal_gripper_speed"],
            abs(transit.result.best_states[-1, self.model.nq + gripper_dof]),
        )
        self.assertAlmostEqual(
            transit.result.trajectory.controls[0, 6],
            self.start[6],
        )

    def test_typed_configuration_finishes_before_approach(self) -> None:
        site = self.data.site("grasp_site")
        site_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(site_quaternion, site.xmat)
        tool_geom_ids = tuple(
            self.model.geom(name).id
            for name in (
                "hand_collision",
                "left_finger_collision_0",
                "right_finger_collision_0",
            )
        )
        target_name = next(
            name
            for name in ("cereal_box", "cheez_it_box")
            if mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            >= 0
        )
        target_geom = self.model.geom(f"{target_name}_collision").id
        frames = {
            "tool": RoleFrame(
                position=site.xpos.copy(),
                orientation=site_quaternion,
                half_extents=np.asarray((0.01, 0.01, 0.01)),
                geom_ids=tool_geom_ids,
            ),
            "target": RoleFrame(
                position=self.data.body(target_name).xpos.copy(),
                orientation=self.data.body(target_name).xquat.copy(),
                half_extents=np.asarray((0.03, 0.02, 0.05)),
                geom_ids=(target_geom,),
            ),
            "precontact": RoleFrame(
                # Approach realization reserves the runtime-owned 1 cm
                # clearance buffer; whole-tool geometry is checked natively.
                position=(
                    site.xpos.copy()
                    + np.asarray(site.xmat).reshape(3, 3)[:, 0] * 0.01
                ),
                orientation=site_quaternion,
                half_extents=np.zeros(3),
                geom_ids=(target_geom,),
            ),
        }
        spec = TransitSpec(
            stages=(
                ConfigureGripper(0.75, 0.35),
                Approach(
                    "precontact",
                    ScaleExpr("precontact", "x", 0.0),
                    AxisRef("precontact", "x"),
                    0.10,
                ),
                Settle(0.10),
            ),
            tool_role="tool",
            target_role="target",
            label="configure_then_approach",
        )
        binding = SceneBinding(
            program_kind="primitive",
            program_name="test",
            instance_name="test",
            scene_module="tasks.invention_real2sim.push_scene.scene",
            goal_text="Configure then approach.",
            roles={},
        )
        context = BoundExecutionContext(
            self.model,
            self.data,
            binding,
            self.x0,
        )

        transit = execute_transit(
            context,
            self.adapter,
            frames,
            spec,
            transit_seed=42,
            transit_index=0,
        )

        self.assertTrue(transit.result.metrics["transit_valid"])
        self.assertEqual(
            transit.result.metrics["controller_rollout_backend"],
            "native_cpp_fr3_osc",
        )
        self.assertIn("gripper_configuration", transit.result.metrics)
        self.assertEqual(transit.result.metrics["phase_boundary_aperture_delta"], 0.0)
        self.assertAlmostEqual(
            transit.result.trajectory.controls[-1, 6],
            self.adapter.gripper_aperture(0.75),
        )
        configuration_metrics = transit.result.metrics["gripper_configuration"]
        np.testing.assert_array_equal(
            transit.result.trajectory.gripper_operations,
            np.zeros(transit.result.best_states.shape[0], dtype=np.uint8),
        )
        self.assertLessEqual(
            configuration_metrics["terminal_gripper_position_error"],
            configuration_metrics["terminal_gripper_position_tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
