from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from instantiation_core import SceneBinding
from primitive_core.policy import (
    BoundExecutionContext,
    GripperIntent,
    GripperOperation,
    NominalPlan,
)
from primitive_core.transit import (
    Approach,
    AxisRef,
    ConfigureGripper,
    InheritedReleaseWindow,
    Reorient,
    RoleFrame,
    ScaleExpr,
    Settle,
    TransitSpec,
    Withdraw,
    _equivalent_handoff_orientations,
    audit_transit,
    realize_transit_waypoints,
)


class _PoseAdapter:
    action_dim = 7

    def target_pose_world(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(action[:3], dtype=float), np.asarray([1.0, 0.0, 0.0, 0.0])

    def action_from_world_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        gripper: float,
    ) -> np.ndarray:
        del quaternion
        return np.asarray([*position, 0.0, 0.0, 0.0, gripper], dtype=float)


def _yaw_quaternion(angle: float) -> np.ndarray:
    return np.asarray([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])


class MotionPolicyTests(unittest.TestCase):
    def test_nominal_plan_normalizes_scalar_sigma_and_rejects_bad_shape(self) -> None:
        plan = NominalPlan(np.zeros((4, 7)), np.asarray(0.2), True)
        self.assertEqual(plan.sigma.shape, (4, 7))
        self.assertTrue(plan.fixed_first_action)
        with self.assertRaisesRegex(ValueError, "match knot shape"):
            NominalPlan(np.zeros((4, 7)), np.zeros((3, 7)))

    def test_nominal_plan_requires_explicit_ordered_timing_when_provided(self) -> None:
        plan = NominalPlan(
            np.zeros((3, 7)),
            np.asarray(0.1),
            knot_fractions=np.asarray((0.0, 0.25, 1.0)),
        )
        np.testing.assert_array_equal(plan.knot_fractions, (0.0, 0.25, 1.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            NominalPlan(
                np.zeros((3, 7)),
                np.asarray(0.1),
                knot_fractions=np.asarray((0.0, 0.0, 1.0)),
            )

    def test_hold_rejects_changing_nominal_gripper_profile(self) -> None:
        knots = np.zeros((3, 7), dtype=float)
        knots[-1, 6] = 0.02
        with self.assertRaisesRegex(ValueError, "constant nominal gripper"):
            NominalPlan(
                knots,
                np.asarray(0.1),
                gripper_intent=GripperIntent.HOLD,
                gripper_operations=np.full(3, GripperOperation.MOVE),
            )

        plan = NominalPlan(
            np.zeros((3, 7), dtype=float),
            np.asarray(0.1),
            gripper_intent=GripperIntent.HOLD,
            gripper_operations=np.full(3, GripperOperation.MOVE),
        )
        self.assertIs(plan.gripper_intent, GripperIntent.HOLD)
        np.testing.assert_array_equal(plan.gripper_operations, np.zeros(3))

    def test_grasp_operation_requires_non_increasing_aperture(self) -> None:
        knots = np.zeros((3, 7), dtype=float)
        knots[:, 6] = (0.04, 0.02, 0.03)
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            NominalPlan(
                knots,
                np.asarray(0.1),
                gripper_intent=GripperIntent.ACTUATE,
                gripper_operations=np.full(3, GripperOperation.GRASP),
            )

    def test_configure_gripper_resolves_range_before_approach(self) -> None:
        adapter = _PoseAdapter()
        adapter.gripper_aperture = lambda fraction: 0.01 + 0.03 * fraction
        frames = {
            "target": RoleFrame(
                position=np.asarray((0.2, 0.0, 0.0)),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                half_extents=np.asarray((0.02, 0.02, 0.02)),
            ),
            "tool": RoleFrame(
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                half_extents=np.asarray((0.01, 0.01, 0.01)),
            ),
        }
        spec = TransitSpec(
            stages=(
                ConfigureGripper(0.25, 0.2),
                Approach(
                    "target",
                    ScaleExpr("target", "x", 0.5),
                    AxisRef("target", "x"),
                    0.3,
                ),
                Settle(0.1),
            ),
            tool_role="tool",
            target_role="target",
        )
        waypoints, _durations, _settle = realize_transit_waypoints(
            adapter,
            np.zeros(7),
            frames,
            spec,
        )
        self.assertAlmostEqual(waypoints[1, 6], 0.0175)
        self.assertAlmostEqual(waypoints[2, 6], 0.0175)

        with self.assertRaisesRegex(ValueError, "before the first Approach"):
            TransitSpec(
                stages=(
                    spec.stages[1],
                    ConfigureGripper(0.25, 0.2),
                    Settle(0.1),
                ),
                tool_role="tool",
                target_role="target",
            )

    def test_transit_requires_a_final_convergence_settle(self) -> None:
        with self.assertRaisesRegex(ValueError, "end with exactly one Settle"):
            TransitSpec(
                stages=(
                    Withdraw(
                        ScaleExpr("target", "x", 0.5),
                        AxisRef("target", "x"),
                        1.0,
                    ),
                ),
                tool_role="tool",
                target_role="target",
            )

    def test_reorient_declares_task_equivalent_handoff_attitudes(self) -> None:
        target_orientation = _yaw_quaternion(math.pi / 2.0)
        frames = {
            "target": RoleFrame(
                position=np.zeros(3),
                orientation=target_orientation,
                half_extents=np.ones(3),
            ),
            "tool": RoleFrame(
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                half_extents=np.ones(3),
            ),
        }
        spec = TransitSpec(
            stages=(
                Reorient(
                    "target",
                    (1.0, 0.0, 0.0, 0.0),
                    0.5,
                    equivalent_quaternion_offsets=((0.0, 0.0, 0.0, 1.0),),
                ),
                Settle(0.1),
            ),
            tool_role="tool",
            target_role="target",
        )

        candidates = _equivalent_handoff_orientations(frames, spec)
        expected = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            expected,
            target_orientation,
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )

        self.assertEqual(len(candidates), 1)
        self.assertGreater(abs(float(candidates[0] @ expected)), 1.0 - 1e-12)
        with self.assertRaisesRegex(ValueError, "cannot be zero"):
            Reorient(
                "target",
                (1.0, 0.0, 0.0, 0.0),
                0.5,
                equivalent_quaternion_offsets=((0.0, 0.0, 0.0, 0.0),),
            )

    def test_scale_expression_and_axis_are_resize_rotation_equivariant(self) -> None:
        adapter = _PoseAdapter()
        spec = TransitSpec(
            stages=(
                Withdraw(
                    distance=ScaleExpr("target", "x", 0.5),
                    direction=AxisRef("target", "x"),
                    duration=1.0,
                ),
                Settle(0.1),
            ),
            tool_role="tool",
            target_role="target",
        )
        base_frames = {
            "target": RoleFrame(
                position=np.zeros(3),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.asarray([0.2, 0.1, 0.1]),
            ),
            "tool": RoleFrame(
                position=np.zeros(3),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.asarray([0.01, 0.01, 0.01]),
            ),
        }
        rotated_frames = {
            **base_frames,
            "target": RoleFrame(
                position=np.zeros(3),
                orientation=_yaw_quaternion(math.pi / 2.0),
                half_extents=np.asarray([0.4, 0.2, 0.2]),
            ),
        }
        base, _, _ = realize_transit_waypoints(
            adapter,
            np.zeros(7),
            base_frames,
            spec,
        )
        rotated, _, _ = realize_transit_waypoints(
            adapter,
            np.zeros(7),
            rotated_frames,
            spec,
        )
        np.testing.assert_allclose(base[-1, :3], [0.2, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotated[-1, :3], [0.0, 0.4, 0.0], atol=1e-12)

    def test_inherited_target_contact_is_bounded_and_cannot_reappear(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.01" gravity="0 0 0"/>
              <worldbody>
                <body name="tool"><joint name="tool_x" type="slide" axis="1 0 0"/>
                  <geom name="tool_geom" type="sphere" size="0.1"/>
                </body>
                <body name="target" pos="0.15 0 0">
                  <geom name="target_geom" type="sphere" size="0.1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state = np.concatenate((data.qpos, data.qvel))
        binding = SceneBinding(
            program_kind="primitive",
            program_name="push",
            instance_name="contact",
            scene_module="tasks.example.scene",
            goal_text="test",
            roles={},
        )
        context = BoundExecutionContext(model, data, binding, state)
        frames = {
            "tool": RoleFrame(
                data.geom_xpos[model.geom("tool_geom").id],
                [1, 0, 0, 0],
                [0.1, 0.1, 0.1],
                geom_ids=(model.geom("tool_geom").id,),
            ),
            "target": RoleFrame(
                data.geom_xpos[model.geom("target_geom").id],
                [1, 0, 0, 0],
                [0.1, 0.1, 0.1],
                geom_ids=(model.geom("target_geom").id,),
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
            inherited_release_window=InheritedReleaseWindow(0.02),
        )
        released = state.copy()
        released[0] = 1.0
        states = np.stack((state, released, state))
        audit = audit_transit(context, frames, spec, states)
        self.assertEqual(audit.inherited_contact_steps, 1)
        self.assertEqual(audit.forbidden_contact_steps, 1)
        self.assertFalse(audit.valid)


if __name__ == "__main__":
    unittest.main()
