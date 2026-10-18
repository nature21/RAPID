"""Focused tests for collision-aware FR3 transition planning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
import unittest

import mujoco
import numpy as np

from instantiation_core import SceneBinding
from primitive_core.fr3_osc import FR3OSCAdapter
from primitive_core.fr3_safety import (
    AllowedProximityPair,
    FR3SafetySpec,
    audit_fr3_trajectory,
)
from primitive_core.fr3_transit_planner import (
    FR3ReleasePair,
    FR3TransitPlan,
    FR3TransitPlanningConfig,
    FR3TransitPlanningFailure,
    _arm_joint_bounds,
    _native_config,
    _project_handoff_goal,
    _release_path_profile,
    check_fr3_transit_configuration,
    plan_fr3_transit,
    plan_fr3_release,
)
from primitive_core.policy import BoundExecutionContext, GripperOperation
from primitive_core.transit import (
    AxisRef,
    InheritedReleaseWindow,
    RoleFrame,
    ScaleExpr,
    Settle,
    TransitSpec,
    Withdraw,
    _InheritedGeometryPair,
    _audit_inherited_release,
    _classify_release_pair,
    _derive_transit_planning_seed,
    _prioritize_release_plans,
    _snapshot_release_entry,
    execute_transit,
)
from tasks.invention_real2sim.push_scene.scene import make_data, make_model

try:
    from mujoco_extensions.fr3_transit_planner import (
        NativeFR3TransitKinematics as _Native,
    )
except ImportError:
    _NATIVE_TRANSIT_AVAILABLE = False
else:
    _NATIVE_TRANSIT_AVAILABLE = _Native is not None


@unittest.skipUnless(
    _NATIVE_TRANSIT_AVAILABLE,
    "native FR3 transit planner extension has not been built",
)
class FR3TransitPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.data = make_data(self.model)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)
        self.state = np.concatenate((self.data.qpos, self.data.qvel))
        self.start = self.adapter.action_from_data(self.model, self.data)

    def _movable_target_name(self) -> str:
        return next(
            name
            for name in ("cheez_it_box", "cereal_box")
            if mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            >= 0
        )

    def test_release_reports_joint_limits_separately_from_collision(self) -> None:
        lower, upper = _arm_joint_bounds(self.model, self.adapter, 0.0)
        pair = FR3ReleasePair(0, 1, "contact", 0.0, -0.001)
        for margin, reason in ((.02, "planning joint reserve"), (-.001, "physical joint limit")):
            with self.subTest(margin=margin):
                q = (lower + upper) / 2
                q[0] = lower[0] + margin
                native = mock.Mock()
                native.arm_configuration.return_value = q
                native.is_valid.return_value = False
                with mock.patch("primitive_core.fr3_transit_planner._load_native_type", return_value=lambda *args: native):
                    with self.assertRaisesRegex(FR3TransitPlanningFailure, reason) as caught:
                        plan_fr3_release(self.model, self.state, self.adapter, self.start, (pair,), seed=0)
                self.assertIn("collision clearance was not evaluated", str(caught.exception))
                native.check_state.assert_not_called()

    def test_native_pose_convergence_is_separate_from_collision(self) -> None:
        config = FR3TransitPlanningConfig()
        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        native = _Native(
            self.model,
            np.ascontiguousarray(self.data.qpos),
            _native_config(self.adapter, safety, config),
        )
        lower, upper = _arm_joint_bounds(
            self.model,
            self.adapter,
            config.joint_margin,
        )
        rng = np.random.default_rng(7)
        colliding = None
        for _ in range(16):
            candidate = rng.uniform(lower, upper)
            valid, clearance = native.check_state(
                np.ascontiguousarray(candidate)
            )
            if not valid and np.isfinite(clearance):
                colliding = candidate
                break
        self.assertIsNotNone(colliding)
        assert colliding is not None

        target = native.pose(np.ascontiguousarray(colliding))
        q, converged, position_error, orientation_error = native.solve_pose(
            np.ascontiguousarray(colliding),
            np.ascontiguousarray(target),
            config.ik_maximum_iterations,
            config.ik_position_tolerance,
            config.ik_orientation_tolerance,
        )

        self.assertTrue(converged)
        self.assertLessEqual(position_error, config.ik_position_tolerance)
        self.assertLessEqual(orientation_error, config.ik_orientation_tolerance)
        valid, _ = native.check_state(np.ascontiguousarray(q))
        self.assertFalse(valid)

    def test_handoff_command_is_reanchored_to_checked_fk_pose(self) -> None:
        desired_position, desired_orientation = self.adapter.target_pose_world(
            self.start
        )
        realized_position = desired_position + np.asarray((0.005, 0.0, 0.0))
        realized_pose = np.concatenate(
            (realized_position, desired_orientation)
        )
        start_q = np.asarray(
            [
                self.data.qpos[
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                ]
                for name in self.adapter.config.arm_joint_names
            ],
            dtype=float,
        )
        lower, upper = _arm_joint_bounds(
            self.model,
            self.adapter,
            FR3TransitPlanningConfig().joint_margin,
        )

        class FakeNative:
            def __init__(self) -> None:
                self._checks = 0

            def solve_pose(self, *_args):
                return start_q.copy(), True, 0.005, 0.0

            def pose(self, _q):
                return realized_pose.copy()

            def check_state(self, _q):
                self._checks += 1
                return True, 0.01

            def is_valid(self, _q):
                self._checks += 1
                return True

            def state_checks(self):
                return self._checks

        projected, goal_q, metrics, candidates = _project_handoff_goal(
            FakeNative(),
            start_q,
            lower,
            upper,
            self.adapter,
            self.start,
            (),
            FR3TransitPlanningConfig(),
        )

        projected_position, projected_orientation = (
            self.adapter.target_pose_world(projected)
        )
        np.testing.assert_allclose(projected_position, realized_position)
        self.assertGreater(
            abs(float(projected_orientation @ desired_orientation)),
            1.0 - 1e-12,
        )
        np.testing.assert_allclose(goal_q, start_q)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(metrics["handoff_projection_translation"], 0.005)
        self.assertAlmostEqual(
            metrics["handoff_projection_ik_position_tolerance"],
            0.008,
        )

    def test_handoff_avoids_joint_boundary_when_direct_alternative_exists(self) -> None:
        config = FR3TransitPlanningConfig()
        lower, upper = _arm_joint_bounds(
            self.model, self.adapter, config.joint_margin
        )
        start_q = (lower + upper) / 2
        position, orientation = self.adapter.target_pose_world(self.start)
        alternate = np.empty(4)
        mujoco.mju_mulQuat(alternate, orientation, np.asarray((0., 0., 0., 1.)))

        # The alternate attitude has a smaller FK position residual and wins
        # the usual ranking. Only a viable alternative to a boundary solution
        # should change that choice.
        scenarios = (
            ("boundary", 0.2, 0.0, True, True, True),
            ("interior", 0.2, 0.02, True, True, False),
            ("both_at_boundary", 0.0, 0.0, True, True, False),
            ("alternative_collides", 0.2, 0.0, False, True, False),
            ("alternative_path_blocked", 0.2, 0.0, True, False, False),
        )
        for name, primary_margin, alternate_margin, valid, direct, use_primary in scenarios:
            with self.subTest(name=name):
                primary_q = start_q.copy()
                primary_q[0] = lower[0] + primary_margin
                alternate_q = start_q.copy()
                alternate_q[0] = upper[0] - alternate_margin

                class FakeNative:
                    def solve_pose(self, _seed, pose, *_args):
                        primary = abs(float(pose[3:] @ orientation)) > 1 - 1e-12
                        q = primary_q if primary else alternate_q
                        return q.copy(), True, 0.005 if primary else 0.001, 0.0

                    def pose(self, q):
                        primary = np.array_equal(q, primary_q)
                        offset = np.asarray((0.005 if primary else 0.001, 0., 0.))
                        return np.concatenate((
                            position + offset,
                            orientation if primary else alternate,
                        ))

                    def check_state(self, q):
                        collision_free = valid if np.array_equal(q, primary_q) else True
                        return collision_free, 0.01 if collision_free else -0.01

                def segment_valid(_native, _start, goal, _config):
                    return direct if np.array_equal(goal, primary_q) else True

                with mock.patch(
                    "primitive_core.fr3_transit_planner._configuration_segment_valid",
                    side_effect=segment_valid,
                ):
                    action, goal_q, metrics, candidates = _project_handoff_goal(
                        FakeNative(), start_q, lower, upper, self.adapter,
                        self.start, (alternate,), config,
                    )

                expected_q = primary_q if use_primary else alternate_q
                np.testing.assert_array_equal(goal_q, expected_q)
                projected_position, projected_orientation = self.adapter.target_pose_world(action)
                np.testing.assert_allclose(
                    projected_position,
                    position + np.asarray((0.005 if use_primary else 0.001, 0., 0.)),
                )
                expected_orientation = orientation if use_primary else alternate
                self.assertAlmostEqual(abs(float(projected_orientation @ expected_orientation)), 1.0)
                self.assertAlmostEqual(
                    metrics["handoff_projected_joint_margin"],
                    primary_margin if use_primary else alternate_margin,
                )
                self.assertEqual(metrics["handoff_projection_global_candidates"], 0)
                self.assertEqual(len(candidates), 1)

    def test_transit_planner_seed_uses_only_runtime_owned_inputs(self) -> None:
        seed = _derive_transit_planning_seed(42, 1, "release")

        self.assertEqual(
            seed,
            _derive_transit_planning_seed(42, 1, "release"),
        )
        self.assertNotEqual(
            seed,
            _derive_transit_planning_seed(42, 2, "release"),
        )
        self.assertNotEqual(
            seed,
            _derive_transit_planning_seed(42, 1, "handoff"),
        )
        self.assertNotEqual(
            seed,
            _derive_transit_planning_seed(42, 1, "release", 1),
        )
        self.assertGreaterEqual(seed, 1)
        self.assertLess(seed, 2**31)

    def test_release_entry_classification_uses_one_contact_tolerance(self) -> None:
        self.assertEqual(
            _classify_release_pair(
                contacting=True,
                distance=0.002,
                required_clearance=0.010,
            ),
            "contact",
        )
        self.assertEqual(
            _classify_release_pair(
                contacting=False,
                distance=0.00005,
                required_clearance=0.010,
            ),
            "contact",
        )
        self.assertEqual(
            _classify_release_pair(
                contacting=False,
                distance=0.001,
                required_clearance=0.010,
            ),
            "near",
        )
        self.assertEqual(
            _classify_release_pair(
                contacting=False,
                distance=0.010,
                required_clearance=0.010,
            ),
            "clear",
        )

    def test_release_entry_snapshot_captures_tracking_pairs(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="robot" pos="0 0 0">
                  <joint name="slide" type="slide" axis="1 0 0"
                         range="-0.1 0.1"/>
                  <geom name="robot_geom" type="sphere" size="0.01"/>
                </body>
                <geom name="environment_geom" type="sphere" size="0.01"
                      pos="0.0315 0 0"/>
              </worldbody>
            </mujoco>
            """
        )
        robot = model.geom("robot_geom").id
        environment = model.geom("environment_geom").id
        safety = FR3SafetySpec(
            robot_geom_ids=(robot,),
            moving_robot_geom_ids=(robot,),
        )
        state = np.zeros(model.nq + model.nv)
        data = mujoco.MjData(model)
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq :]
        mujoco.mj_forward(model, data)

        inherited, tracking = _snapshot_release_entry(
            model,
            data,
            safety,
            0.010,
            0.013,
        )

        self.assertEqual(inherited, ())
        self.assertEqual(len(tracking), 1)
        self.assertEqual(tracking[0].environment_geom_id, environment)
        self.assertGreaterEqual(tracking[0].minimum_distance, 0.010)
        self.assertLess(
            tracking[0].minimum_distance,
            tracking[0].initial_distance,
        )

    def test_release_entry_snapshot_captures_all_subclearance_pairs(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="robot" pos="0 0 0">
                  <joint name="slide" type="slide" axis="1 0 0"
                         range="-0.1 0.1"/>
                  <geom name="robot_geom" type="sphere" size="0.01"/>
                </body>
                <geom name="environment_geom" type="sphere" size="0.01"
                      pos="0.025 0 0"/>
              </worldbody>
            </mujoco>
            """
        )
        robot = model.geom("robot_geom").id
        environment = model.geom("environment_geom").id
        safety = FR3SafetySpec(
            robot_geom_ids=(robot,),
            moving_robot_geom_ids=(robot,),
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        inherited, tracking = _snapshot_release_entry(
            model,
            data,
            safety,
            0.010,
            0.013,
        )

        self.assertEqual(tracking, ())
        self.assertEqual(len(inherited), 1)
        self.assertEqual(inherited[0].environment_geom_id, environment)
        self.assertEqual(inherited[0].entry_state, "near")
        self.assertAlmostEqual(inherited[0].initial_distance, 0.005)

    def test_release_certificate_uses_terminal_stable_separation(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.005"/>
              <worldbody>
                <body name="robot">
                  <joint name="slide" type="slide" axis="1 0 0"
                         range="-0.1 0.1"/>
                  <geom name="robot_geom" type="sphere" size="0.01"/>
                </body>
                <body name="target" pos="0.019 0 0">
                  <geom name="target_geom" type="sphere" size="0.01"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        robot = model.geom("robot_geom").id
        target = model.geom("target_geom").id
        target_body = model.body("target").id
        initial_state = np.zeros(model.nq + model.nv)
        binding = SceneBinding(
            program_kind="primitive",
            program_name="release_test",
            instance_name="release_test",
            scene_module="tasks.example.scene",
            goal_text="Release the target.",
            roles={},
        )
        frames = {
            "tool": RoleFrame(
                np.zeros(3),
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                np.ones(3),
                (robot,),
            ),
            "target": RoleFrame(
                data.body(target_body).xpos.copy(),
                data.body(target_body).xquat.copy(),
                np.ones(3),
                (target,),
                (target_body,),
            ),
        }
        spec = TransitSpec(
            stages=(
                Withdraw(
                    ScaleExpr("target", "x", 1.0),
                    AxisRef("target", "x"),
                    0.1,
                ),
                Settle(0.1),
            ),
            tool_role="tool",
            target_role="target",
            inherited_release_window=InheritedReleaseWindow(1.5),
            label="release_test",
        )
        positions = np.concatenate(
            (
                np.asarray((0.0,)),
                np.full(12, -0.002),
                np.asarray((0.0,)),
                np.full(24, -0.012),
            )
        )
        states = np.zeros((positions.size, model.nq + model.nv))
        states[:, 0] = positions
        safety = FR3SafetySpec(
            robot_geom_ids=(robot,),
            moving_robot_geom_ids=(robot,),
        )
        audit = _audit_inherited_release(
            BoundExecutionContext(
                model,
                data,
                binding,
                initial_state,
            ),
            frames,
            spec,
            states,
            safety,
            (
                _InheritedGeometryPair(
                    robot_geom_id=robot,
                    environment_geom_id=target,
                    initial_distance=-0.001,
                    entry_state="contact",
                ),
            ),
            contact_timeout=1.5,
        )

        self.assertTrue(audit.valid)
        self.assertTrue(audit.contacts_released)
        self.assertGreater(audit.recontact_steps, 0)
        self.assertGreaterEqual(audit.terminal_separation_duration, 0.10)
        self.assertEqual(audit.penetration_worsening_steps, 0)

    def test_release_near_pair_uses_bounded_penetration_hysteresis(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.005"/>
              <worldbody>
                <body name="robot">
                  <joint name="slide" type="slide" axis="1 0 0"
                         range="-0.1 0.1"/>
                  <geom name="robot_geom" type="sphere" size="0.01"/>
                </body>
                <body name="target" pos="0.0202 0 0">
                  <geom name="target_geom" type="sphere" size="0.01"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        robot = model.geom("robot_geom").id
        target = model.geom("target_geom").id
        target_body = model.body("target").id
        initial_state = np.zeros(model.nq + model.nv)
        binding = SceneBinding(
            program_kind="primitive",
            program_name="release_test",
            instance_name="release_test",
            scene_module="tasks.example.scene",
            goal_text="Release the target.",
            roles={},
        )
        frames = {
            "tool": RoleFrame(
                np.zeros(3),
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                np.ones(3),
                (robot,),
            ),
            "target": RoleFrame(
                data.body(target_body).xpos.copy(),
                data.body(target_body).xquat.copy(),
                np.ones(3),
                (target,),
                (target_body,),
            ),
        }
        spec = TransitSpec(
            stages=(
                Withdraw(
                    ScaleExpr("target", "x", 1.0),
                    AxisRef("target", "x"),
                    0.1,
                ),
                Settle(0.1),
            ),
            tool_role="tool",
            target_role="target",
            inherited_release_window=InheritedReleaseWindow(1.5),
            label="release_test",
        )
        context = BoundExecutionContext(
            model,
            data,
            binding,
            initial_state,
        )
        safety = FR3SafetySpec(
            robot_geom_ids=(robot,),
            moving_robot_geom_ids=(robot,),
        )
        inherited = (
            _InheritedGeometryPair(
                robot_geom_id=robot,
                environment_geom_id=target,
                initial_distance=0.0002,
                entry_state="near",
            ),
        )

        shallow = initial_state[None].copy()
        shallow[0, 0] = 0.00035  # 0.15 mm penetration.
        deep = initial_state[None].copy()
        deep[0, 0] = 0.00050  # 0.30 mm penetration.
        shallow_audit = _audit_inherited_release(
            context,
            frames,
            spec,
            shallow,
            safety,
            inherited,
            contact_timeout=1.5,
        )
        deep_audit = _audit_inherited_release(
            context,
            frames,
            spec,
            deep,
            safety,
            inherited,
            contact_timeout=1.5,
        )

        self.assertTrue(shallow_audit.valid)
        self.assertEqual(shallow_audit.new_contact_steps, 0)
        self.assertEqual(shallow_audit.penetration_worsening_steps, 0)
        self.assertFalse(deep_audit.valid)
        self.assertEqual(deep_audit.new_contact_steps, 1)
        self.assertEqual(deep_audit.penetration_worsening_steps, 1)

    def test_release_path_profile_rejects_clearance_backtracking(self) -> None:
        class FakeNative:
            def __init__(self) -> None:
                self.checks = 0

            def state_checks(self):
                return self.checks

            def is_valid(self, _q):
                self.checks += 1
                return True

            def release_distances(self, q):
                return np.asarray((float(q[0]),))

        pair = FR3ReleasePair(0, 1, "contact", 0.0, -0.001)
        config = FR3TransitPlanningConfig(
            interpolation_joint_step=1.0,
            release_maximum_clearance_regression=0.01,
        )
        rising = np.zeros((2, 7))
        rising[1, 0] = 0.03
        backtracking = np.zeros((3, 7))
        backtracking[1, 0] = 0.03
        backtracking[2, 0] = 0.015

        self.assertIsNotNone(
            _release_path_profile(FakeNative(), rising, (pair,), 0.02, config)
        )
        self.assertIsNone(
            _release_path_profile(
                FakeNative(),
                backtracking,
                (pair,),
                0.02,
                config,
            )
        )

    def test_release_feedback_prioritizes_aligned_candidate(self) -> None:
        def plan(
            rank: int,
            direction: tuple[float, float, float],
        ) -> FR3TransitPlan:
            return FR3TransitPlan(
                waypoint_actions=np.zeros((2, 7)),
                segment_durations=(1.0,),
                metrics={
                    "release_shortlist_rank": rank,
                    "release_translation_direction": direction,
                },
            )

        plans = [
            plan(2, (1.0, 0.0, 0.0)),
            plan(3, (0.0, 1.0, 0.0)),
        ]

        _prioritize_release_plans(plans, np.asarray((0.0, 1.0, 0.0)))

        self.assertEqual(plans[0].metrics["release_shortlist_rank"], 3)

    def test_native_planning_avoids_python_simulation_and_preserves_model(self) -> None:
        original_margin = self.model.geom_margin.copy()
        target = self.start.copy()
        target[2] += 0.015

        with mock.patch.object(
            mujoco,
            "mj_forward",
            side_effect=AssertionError("planner called Python mj_forward"),
        ), mock.patch.object(
            mujoco,
            "mj_step",
            side_effect=AssertionError("planner called Python mj_step"),
        ):
            plan = plan_fr3_transit(
                self.model,
                self.state,
                self.adapter,
                self.start,
                target,
                guide_actions=(target,),
                seed=11,
            )

        self.assertEqual(plan.metrics["planning_backend"], "typed_guide")
        self.assertEqual(plan.metrics["planning_dynamics"], "kinematics_only")
        self.assertEqual(plan.metrics["planning_clearance"], 0.010)
        self.assertEqual(plan.metrics["execution_clearance"], 0.010)
        np.testing.assert_allclose(plan.waypoint_actions[0], self.start)
        np.testing.assert_allclose(plan.waypoint_actions[-1], target)
        np.testing.assert_array_equal(self.model.geom_margin, original_margin)
        valid, clearance = check_fr3_transit_configuration(
            self.model,
            self.state,
            self.adapter,
        )
        self.assertTrue(valid)
        self.assertGreaterEqual(clearance, 0.010)

    def test_colliding_typed_guide_falls_back_to_ompl(self) -> None:
        target = self.start.copy()
        target[2] += 0.015
        obstacle_body = self._movable_target_name()
        colliding_guide = target.copy()
        colliding_guide[:3] = self.data.body(obstacle_body).xpos

        plan = plan_fr3_transit(
            self.model,
            self.state,
            self.adapter,
            self.start,
            target,
            guide_actions=(colliding_guide, target),
            config=FR3TransitPlanningConfig(
                planning_time=2.0,
                maximum_planning_time=5.0,
            ),
            seed=13,
        )

        self.assertEqual(
            plan.metrics["planning_backend"],
            "ompl_geometric_rrtconnect",
        )
        self.assertGreater(plan.metrics["planning_state_checks"], 0)
        self.assertLessEqual(plan.metrics["planning_time"], 5.1)
        np.testing.assert_allclose(plan.waypoint_actions[-1], target)

    def test_contact_goal_uses_low_discrepancy_translation_search(self) -> None:
        desired = self.start.copy()
        desired[2] += 0.015
        desired_position, _ = self.adapter.target_pose_world(desired)
        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        tool_geoms = safety.moving_robot_geom_ids[-2:]
        target_name = self._movable_target_name()
        target_geoms = (self.model.geom(f"{target_name}_collision").id,)

        class FakeNative:
            received_config = None

            def __init__(self, model, qpos, config):
                type(self).received_config = config
                self._checks = 0
                self._q = np.asarray(
                    [
                        qpos[
                            model.jnt_qposadr[
                                mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_JOINT,
                                    name,
                                )
                            ]
                        ]
                        for name in config["arm_joint_names"]
                    ],
                    dtype=float,
                )

            def arm_configuration(self):
                return self._q.copy()

            def is_valid(self, _q):
                self._checks += 1
                return True

            def check_state(self, _q):
                self._checks += 1
                offset = abs(float(_q[0] - self._q[0]))
                return offset >= 0.003, 0.002 if offset >= 0.003 else -0.001

            def solve_pose(
                self,
                _seed,
                pose,
                _iterations,
                _position_tolerance,
                _orientation_tolerance,
            ):
                self._last_pose = np.asarray(pose, dtype=float).copy()
                offset = np.asarray(pose[:3]) - desired_position
                q = self._q.copy()
                q[0] += float(np.linalg.norm(offset))
                return q, True, 0.0, 0.0

            def pose(self, _q):
                return self._last_pose.copy()

            def reset_counters(self):
                self._checks = 0

            def state_checks(self):
                return self._checks

        with mock.patch(
            "primitive_core.fr3_transit_planner._load_native_type",
            return_value=FakeNative,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._guide_candidate",
            return_value=None,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._ompl_geometric_candidate",
            side_effect=AssertionError("direct handoff called OMPL"),
        ), mock.patch.object(
            mujoco,
            "mj_forward",
            side_effect=AssertionError("planner called Python mj_forward"),
        ), mock.patch.object(
            mujoco,
            "mj_step",
            side_effect=AssertionError("planner called Python mj_step"),
        ):
            plan = plan_fr3_transit(
                self.model,
                self.state,
                self.adapter,
                self.start,
                desired,
                guide_actions=(desired,),
                seed=17,
                handoff_tool_geom_ids=tool_geoms,
                handoff_target_geom_ids=target_geoms,
            )

        self.assertTrue(plan.metrics["handoff_projection_required"])
        self.assertEqual(
            plan.metrics["planning_backend"],
            "projected_direct_handoff",
        )
        self.assertAlmostEqual(
            plan.metrics["handoff_projection_translation"],
            0.004,
            places=9,
        )
        projected_position, _ = self.adapter.target_pose_world(
            plan.waypoint_actions[-1]
        )
        displacement = projected_position - desired_position
        self.assertAlmostEqual(float(np.linalg.norm(displacement)), 0.004)
        self.assertTrue(np.all(np.abs(displacement) > 1e-6))
        _, desired_orientation = self.adapter.target_pose_world(desired)
        _, projected_orientation = self.adapter.target_pose_world(
            plan.waypoint_actions[-1]
        )
        self.assertAlmostEqual(
            abs(float(desired_orientation @ projected_orientation)),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(plan.waypoint_actions[-1, 6], desired[6])
        self.assertEqual(
            plan.metrics["handoff_projection_search"],
            "task_equivalent_attitudes_then_low_discrepancy_translation_ball",
        )
        self.assertTrue(plan.metrics["handoff_projection_orientation_fixed"])
        self.assertTrue(plan.metrics["handoff_projection_aperture_fixed"])
        self.assertLessEqual(
            plan.metrics["handoff_projection_candidates"],
            plan.metrics["handoff_projection_candidate_budget"],
        )
        self.assertEqual(
            plan.metrics["handoff_projection_candidate_budget"],
            512,
        )
        self.assertAlmostEqual(
            FR3TransitPlanningConfig().handoff_projection_maximum_translation,
            0.100,
        )
        self.assertAlmostEqual(
            FR3TransitPlanningConfig().handoff_projection_ik_position_tolerance,
            0.008,
        )
        self.assertAlmostEqual(
            FR3TransitPlanningConfig().handoff_projection_ik_orientation_tolerance,
            0.100,
        )
        proximity = FakeNative.received_config["proximity_pairs"]
        self.assertEqual(len(proximity), 1)
        self.assertEqual(proximity[0]["minimum_clearance"], 0.002)

    def test_handoff_searches_translation_when_exact_pose_has_no_ik(self) -> None:
        desired = self.start.copy()
        desired[0] += 0.01
        desired_position, _ = self.adapter.target_pose_world(desired)
        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        tool_geoms = safety.moving_robot_geom_ids[-2:]
        target_name = self._movable_target_name()
        target_geoms = (self.model.geom(f"{target_name}_collision").id,)

        class FakeNative:
            solve_calls = 0

            def __init__(self, model, qpos, config):
                self._checks = 0
                self._q = np.asarray(
                    [
                        qpos[
                            model.jnt_qposadr[
                                mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_JOINT,
                                    name,
                                )
                            ]
                        ]
                        for name in config["arm_joint_names"]
                    ],
                    dtype=float,
                )

            def arm_configuration(self):
                return self._q.copy()

            def is_valid(self, _q):
                self._checks += 1
                return True

            def check_state(self, _q):
                self._checks += 1
                return True, 0.01

            def solve_pose(
                self,
                _seed,
                pose,
                _iterations,
                _position_tolerance,
                _orientation_tolerance,
            ):
                type(self).solve_calls += 1
                self._last_pose = np.asarray(pose, dtype=float).copy()
                offset = np.asarray(pose[:3]) - desired_position
                distance = float(np.linalg.norm(offset))
                q = self._q.copy()
                q[0] += distance
                converged = distance >= 0.003
                return q, converged, 0.0 if converged else 0.01, 0.0

            def pose(self, _q):
                return self._last_pose.copy()

            def reset_counters(self):
                self._checks = 0

            def state_checks(self):
                return self._checks

        FakeNative.solve_calls = 0
        with mock.patch(
            "primitive_core.fr3_transit_planner._load_native_type",
            return_value=FakeNative,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._guide_candidate",
            return_value=None,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._ompl_geometric_candidate",
            side_effect=AssertionError("projected direct handoff called OMPL"),
        ):
            plan = plan_fr3_transit(
                self.model,
                self.state,
                self.adapter,
                self.start,
                desired,
                seed=19,
                handoff_tool_geom_ids=tool_geoms,
                handoff_target_geom_ids=target_geoms,
            )

        self.assertGreater(FakeNative.solve_calls, 3)
        self.assertEqual(
            plan.metrics["handoff_exact_ik_orientation_candidates"],
            0,
        )
        self.assertGreater(plan.metrics["handoff_projection_translation"], 0.0)
        self.assertEqual(
            plan.metrics["planning_backend"],
            "projected_direct_handoff",
        )

    def test_handoff_ompl_tries_diverse_collision_free_goals(self) -> None:
        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        tool_geoms = safety.moving_robot_geom_ids[-2:]
        target_name = self._movable_target_name()
        target_geoms = (self.model.geom(f"{target_name}_collision").id,)
        arm_q = np.asarray(
            [
                self.data.qpos[
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                ]
                for name in self.adapter.config.arm_joint_names
            ],
            dtype=float,
        )
        first_q = arm_q.copy()
        first_q[0] += 0.08
        second_q = arm_q.copy()
        second_q[1] += 0.08
        first_action = self.start.copy()
        first_action[0] += 0.02
        second_action = self.start.copy()
        second_action[1] += 0.02

        def evaluation(action, q, offset):
            return SimpleNamespace(
                action=action,
                q=q,
                offset=np.asarray(offset, dtype=float),
                orientation_change=0.0,
                clearance=0.01,
                joint_margin=0.5,
                direct_path_valid=False,
                translation=float(np.linalg.norm(offset)),
            )

        candidates = (
            evaluation(first_action, first_q, (0.02, 0.0, 0.0)),
            evaluation(second_action, second_q, (0.0, 0.02, 0.0)),
        )
        projection_metrics = {
            "handoff_projection_required": True,
            "handoff_projected_direct_path_valid": False,
            "handoff_projection_direct_path_candidates": 0,
            "handoff_projection_feasible_candidates": 2,
            "handoff_projection_goal_shortlist_size": 2,
            "handoff_projection_selected_goal_rank": 1,
        }

        class FakeNative:
            def __init__(self, *_args):
                self._checks = 0

            def arm_configuration(self):
                return arm_q.copy()

            def is_valid(self, _q):
                self._checks += 1
                return True

            def reset_counters(self):
                self._checks = 0

            def state_checks(self):
                return self._checks

        geometric_paths = (
            None,
            np.stack((arm_q, second_q)),
        )
        with mock.patch(
            "ompl.util.RNG.setSeed",
        ), mock.patch(
            "primitive_core.fr3_transit_planner._load_native_type",
            return_value=FakeNative,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._project_handoff_goal",
            return_value=(
                first_action,
                first_q,
                projection_metrics,
                candidates,
            ),
        ), mock.patch(
            "primitive_core.fr3_transit_planner._guide_candidate",
            return_value=None,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._ompl_geometric_candidate",
            side_effect=geometric_paths,
        ) as geometric, mock.patch(
            "primitive_core.fr3_transit_planner._ompl_candidate",
            side_effect=AssertionError("geometric alternate goal was enough"),
        ):
            plan = plan_fr3_transit(
                self.model,
                self.state,
                self.adapter,
                self.start,
                first_action,
                guide_actions=(first_action,),
                seed=29,
                handoff_tool_geom_ids=tool_geoms,
                handoff_target_geom_ids=target_geoms,
            )

        self.assertEqual(geometric.call_count, 2)
        np.testing.assert_allclose(plan.waypoint_actions[-1], second_action)
        self.assertEqual(
            plan.metrics["handoff_projection_selected_goal_rank"],
            2,
        )

    def test_handoff_uses_declared_task_equivalent_attitude(self) -> None:
        desired = self.start.copy()
        desired_position, desired_orientation = self.adapter.target_pose_world(
            desired
        )
        alternative = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            alternative,
            desired_orientation,
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )
        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        tool_geoms = safety.moving_robot_geom_ids[-2:]
        target_name = self._movable_target_name()
        target_geoms = (self.model.geom(f"{target_name}_collision").id,)

        class FakeNative:
            def __init__(self, model, qpos, config):
                self._checks = 0
                self._q = np.asarray(
                    [
                        qpos[
                            model.jnt_qposadr[
                                mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_JOINT,
                                    name,
                                )
                            ]
                        ]
                        for name in config["arm_joint_names"]
                    ],
                    dtype=float,
                )

            def arm_configuration(self):
                return self._q.copy()

            def is_valid(self, _q):
                self._checks += 1
                return True

            def check_state(self, _q):
                self._checks += 1
                return True, 0.01

            def solve_pose(
                self,
                _seed,
                pose,
                _iterations,
                _position_tolerance,
                _orientation_tolerance,
            ):
                self._last_pose = np.asarray(pose, dtype=float).copy()
                solved = abs(float(np.asarray(pose[3:]) @ alternative)) > 0.999
                q = self._q.copy()
                if solved:
                    q[0] += 0.01
                return q, solved, 0.0, 0.0

            def pose(self, _q):
                return self._last_pose.copy()

            def reset_counters(self):
                self._checks = 0

            def state_checks(self):
                return self._checks

        with mock.patch(
            "primitive_core.fr3_transit_planner._load_native_type",
            return_value=FakeNative,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._guide_candidate",
            return_value=None,
        ), mock.patch(
            "primitive_core.fr3_transit_planner._ompl_geometric_candidate",
            side_effect=AssertionError("direct equivalent handoff called OMPL"),
        ):
            plan = plan_fr3_transit(
                self.model,
                self.state,
                self.adapter,
                self.start,
                desired,
                handoff_orientation_candidates=(alternative,),
                seed=23,
                handoff_tool_geom_ids=tool_geoms,
                handoff_target_geom_ids=target_geoms,
            )

        projected_position, projected_orientation = self.adapter.target_pose_world(
            plan.waypoint_actions[-1]
        )
        np.testing.assert_allclose(projected_position, desired_position)
        self.assertGreater(
            abs(float(projected_orientation @ alternative)),
            1.0 - 1e-12,
        )
        self.assertTrue(plan.metrics["handoff_projection_required"])
        self.assertFalse(plan.metrics["handoff_projection_orientation_fixed"])
        self.assertEqual(plan.metrics["handoff_orientation_candidates"], 2)
        self.assertAlmostEqual(
            plan.metrics["handoff_projection_orientation"],
            np.pi,
        )

    def test_proximity_pair_enforces_its_smaller_noncontact_margin(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="robot" pos="0 0 0">
                  <joint name="slide" type="slide" axis="1 0 0"
                         range="-0.1 0.1"/>
                  <geom name="robot_geom" type="sphere" size="0.01"/>
                </body>
                <geom name="environment_geom" type="sphere" size="0.01"
                      pos="0.021 0 0"/>
              </worldbody>
            </mujoco>
            """
        )
        state = np.zeros((1, model.nq + model.nv), dtype=float)
        robot_geom = model.geom("robot_geom").id
        environment_geom = model.geom("environment_geom").id

        unsafe = audit_fr3_trajectory(
            model,
            state,
            FR3SafetySpec(
                robot_geom_ids=(robot_geom,),
                moving_robot_geom_ids=(robot_geom,),
                allowed_proximity_pairs=(
                    AllowedProximityPair(
                        (robot_geom,),
                        (environment_geom,),
                        minimum_clearance=0.002,
                    ),
                ),
            ),
        )
        safe = audit_fr3_trajectory(
            model,
            state,
            FR3SafetySpec(
                robot_geom_ids=(robot_geom,),
                moving_robot_geom_ids=(robot_geom,),
                allowed_proximity_pairs=(
                    AllowedProximityPair(
                        (robot_geom,),
                        (environment_geom,),
                        minimum_clearance=0.0005,
                    ),
                ),
            ),
        )

        self.assertFalse(unsafe.valid)
        self.assertFalse(unsafe.declared_proximity_clearance_valid)
        self.assertLess(unsafe.minimum_declared_proximity_margin, 0.0)
        self.assertTrue(safe.valid)
        self.assertTrue(safe.declared_proximity_clearance_valid)
        self.assertGreater(safe.minimum_declared_proximity_margin, 0.0)

    def test_protected_inherited_contact_is_released_before_planning(self) -> None:
        target_name = self._movable_target_name()
        target_geom = self.model.geom(f"{target_name}_collision").id
        target_body = int(self.model.geom_bodyid[target_geom])
        protected_name = "brown_cardboard_box"
        protected_geom = self.model.geom(f"{protected_name}_collision").id
        protected_body = int(self.model.geom_bodyid[protected_geom])
        protected_joint = int(self.model.body_jntadr[protected_body])
        protected_qpos = int(self.model.jnt_qposadr[protected_joint])
        finger_position = self.data.geom("left_finger_collision_0").xpos.copy()
        self.model.geom_type[protected_geom] = mujoco.mjtGeom.mjGEOM_SPHERE
        self.model.geom_size[protected_geom, 0] = 0.006
        self.data.qpos[protected_qpos : protected_qpos + 3] = (
            finger_position + np.asarray((0.0, -0.015, 0.0))
        )
        self.data.qpos[protected_qpos + 3 : protected_qpos + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        mujoco.mj_forward(self.model, self.data)
        self.adapter = FR3OSCAdapter.from_model_data(self.model, self.data)
        self.state = np.concatenate((self.data.qpos, self.data.qvel))
        self.start = self.adapter.action_from_data(self.model, self.data)

        safety = FR3SafetySpec.from_model(self.model, self.adapter)
        hand_body = self.model.body("hand").id

        def descends_from_hand(body_id: int) -> bool:
            current = int(body_id)
            while current > 0:
                if current == hand_body:
                    return True
                current = int(self.model.body_parentid[current])
            return False

        tool_geoms = tuple(
            geom_id
            for geom_id in safety.robot_geom_ids
            if descends_from_hand(int(self.model.geom_bodyid[geom_id]))
        )
        site = self.data.site("grasp_site")
        site_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(site_quaternion, site.xmat)
        frames = {
            "tool": RoleFrame(
                site.xpos.copy(),
                site_quaternion,
                np.asarray((0.02, 0.04, 0.06)),
                tool_geoms,
            ),
            "target": RoleFrame(
                self.data.body(target_body).xpos.copy(),
                self.data.body(target_body).xquat.copy(),
                np.asarray((0.01, 0.01, 0.01)),
                (target_geom,),
                (target_body,),
            ),
            "protected": RoleFrame(
                self.data.body(protected_body).xpos.copy(),
                self.data.body(protected_body).xquat.copy(),
                np.asarray((0.01, 0.01, 0.01)),
                (protected_geom,),
                (protected_body,),
            ),
        }
        spec = TransitSpec(
            stages=(
                Withdraw(
                    ScaleExpr("target", "y", 1.0),
                    AxisRef("target", "y"),
                    0.30,
                ),
                Settle(0.10),
            ),
            tool_role="tool",
            target_role="target",
            protected_roles=("protected",),
            # The synthetic sphere is intentionally lightweight; this test is
            # about protected-only release dispatch, not fixture anchoring.
            protected_drift_limit=ScaleExpr("protected", "x", 100.0),
            inherited_release_window=InheritedReleaseWindow(0.40),
            label="inherited_release_test",
        )
        binding = SceneBinding(
            program_kind="primitive",
            program_name="test",
            instance_name="test",
            scene_module="tasks.invention_real2sim.push_scene.scene",
            goal_text="Release, then transit.",
            roles={},
        )
        result = execute_transit(
            BoundExecutionContext(
                self.model,
                self.data,
                binding,
                self.state,
                initial_action=self.start,
                initial_gripper_operation=GripperOperation.MOVE,
            ),
            self.adapter,
            frames,
            spec,
            transit_seed=42,
            transit_index=0,
        )

        self.assertTrue(result.result.metrics["transit_valid"])
        self.assertGreater(result.result.metrics["inherited_contact_steps"], 0)
        release = result.result.metrics["release"]
        self.assertTrue(release["release_contact_separation_valid"])
        self.assertTrue(release["release_clearance_acquisition_valid"])
        self.assertGreaterEqual(release["release_inherited_pair_count"], 1)
        self.assertGreaterEqual(release["release_inherited_contact_pair_count"], 1)
        self.assertGreaterEqual(release["release_effective_contact_timeout"], 1.50)
        self.assertEqual(release["release_new_contact_steps"], 0)
        self.assertEqual(release["release_recontact_steps"], 0)
        self.assertEqual(release["release_penetration_worsening_steps"], 0)
        self.assertTrue(release["release_terminal_clear"])
        self.assertGreaterEqual(release["release_terminal_clearance"], 0.010)
        self.assertTrue(release["release_geometry_search"])
        self.assertAlmostEqual(release["planning_clearance"], 0.013)
        self.assertEqual(release["release_required_goal_clearance"], 0.030)
        self.assertGreaterEqual(release["release_goal_clearance"], 0.030)
        self.assertTrue(release["release_initial_controller_reanchored"])
        self.assertTrue(release["release_stationary_valid"])
        self.assertTrue(release["release_aperture_fixed"])
        self.assertAlmostEqual(
            release["commanded_aperture_total_variation"],
            0.0,
            places=12,
        )
        self.assertEqual(release["release_path_tracking_reserve"], 0.003)
        self.assertEqual(release["release_goal_tracking_reserve"], 0.020)
        self.assertGreater(release["release_path_profile_samples"], 0)
        self.assertLessEqual(release["release_shortlist_size"], 4)
        self.assertGreater(release["release_samples_evaluated"], 0)
        self.assertEqual(
            result.result.metrics["planning"]["planning_dynamics"],
            "kinematics_only",
        )


if __name__ == "__main__":
    unittest.main()
