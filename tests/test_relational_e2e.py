from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from instantiation_core import (
    BindingContract,
    EntityKind,
    ParameterRequirement,
    ParameterType,
    RoleRequirement,
    parse_binding_manifest,
    validate_scene_binding,
)
from primitive_core import NominalPlan, PrimitiveCompileSpec, PrimitiveCompilerConfig
from primitive_core.compile import compile_primitive
from primitive_core.transit import AxisRef, ScaleExpr, Settle, TransitSpec, Withdraw
from strategies.program import (
    ConnectingConstraint,
    InterPhaseTransit,
    PrimitivePhase,
    PhaseConstraintContext,
    StrategyProgram,
    primitive_binding_for_phase,
    validate_strategy_program,
)


PRIMITIVE_CONTRACT = BindingContract(
    program_kind="primitive",
    program_name="move_fixture",
    roles=(
        RoleRequirement("controlled_joint", EntityKind.JOINT, "Effect joint."),
        RoleRequirement("controlled_geom", EntityKind.GEOM, "Effect geometry."),
        RoleRequirement(
            "controlled_actuator",
            EntityKind.ACTUATOR,
            "Effect actuator.",
        ),
    ),
    parameters=(
        ParameterRequirement(
            "displacement",
            ParameterType.FLOAT,
            "Requested relative displacement.",
            minimum=0.05,
            maximum=0.25,
        ),
    ),
)

STRATEGY_CONTRACT = BindingContract(
    program_kind="strategy",
    program_name="strategy_fixture",
    roles=(
        RoleRequirement("first_joint", EntityKind.JOINT, "First effect joint."),
        RoleRequirement("first_geom", EntityKind.GEOM, "First effect geometry."),
        RoleRequirement("first_actuator", EntityKind.ACTUATOR, "First actuator."),
        RoleRequirement("second_joint", EntityKind.JOINT, "Second effect joint."),
        RoleRequirement("second_geom", EntityKind.GEOM, "Second effect geometry."),
        RoleRequirement("second_actuator", EntityKind.ACTUATOR, "Second actuator."),
        RoleRequirement("first_goal", EntityKind.BODY, "First relational goal."),
        RoleRequirement("second_goal", EntityKind.BODY, "Second relational goal."),
    ),
)


def _goal_displacement(
    geom_role: str,
    joint_role: str,
    goal_role: str,
):
    def resolve(context: PhaseConstraintContext) -> float:
        data = mujoco.MjData(context.model)
        data.qpos[:] = context.state[: context.model.nq]
        data.qvel[:] = context.state[context.model.nq :]
        mujoco.mj_forward(context.model, data)
        geom_name = context.strategy_binding.one(geom_role)
        joint_name = context.strategy_binding.one(joint_role)
        goal_name = context.strategy_binding.one(goal_role)
        delta = data.body(goal_name).xpos - data.geom(geom_name).xpos
        axis = data.xaxis[context.model.joint(joint_name).id]
        return float(np.dot(axis, delta))

    return resolve


PHASE_PROGRAM = StrategyProgram(
    phases=(
        PrimitivePhase(
            "first",
            "move_fixture",
            {
                "controlled_joint": "first_joint",
                "controlled_geom": "first_geom",
                "controlled_actuator": "first_actuator",
            },
            (
                ConnectingConstraint(
                    "displacement",
                    "Move the first target to its relational goal.",
                    ("first_geom", "first_joint", "first_goal"),
                    "first_joint_axis",
                    _goal_displacement("first_geom", "first_joint", "first_goal"),
                ),
            ),
        ),
        PrimitivePhase(
            "second",
            "move_fixture",
            {
                "controlled_joint": "second_joint",
                "controlled_geom": "second_geom",
                "controlled_actuator": "second_actuator",
            },
            (
                ConnectingConstraint(
                    "displacement",
                    "Move the second target to its relational goal.",
                    ("second_geom", "second_joint", "second_goal"),
                    "second_joint_axis",
                    _goal_displacement(
                        "second_geom",
                        "second_joint",
                        "second_goal",
                    ),
                ),
            ),
        ),
    ),
    transits=(
        InterPhaseTransit(
            "first",
            "second",
            TransitSpec(
                stages=(
                    Withdraw(
                        ScaleExpr("second_geom", "x", 0.5),
                        AxisRef("second_geom", "x"),
                        0.1,
                    ),
                    Settle(0.1),
                ),
                tool_role="first_geom",
                target_role="second_geom",
            ),
        ),
    ),
)


def _model(index: int, *, yaw: float, size: float) -> mujoco.MjModel:
    half = yaw / 2.0
    quaternion = f"{math.cos(half)} 0 0 {math.sin(half)}"
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    first_x = -0.4 + 0.1 * index
    second_y = 0.4 + 0.1 * index
    first_goal = (first_x + 0.12 * cosine, 0.12 * sine)
    second_goal = (-0.10 * sine, second_y + 0.10 * cosine)
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco model="variant_{index}">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="first_body_{index}" pos="{first_x} 0 0" quat="{quaternion}">
              <joint name="first_joint_{index}" type="slide" axis="1 0 0" damping="10"/>
              <geom name="first_geom_{index}" type="box" size="{size} {size} {size}"/>
            </body>
            <body name="second_body_{index}" pos="0 {second_y} 0" quat="{quaternion}">
              <joint name="second_joint_{index}" type="slide" axis="0 1 0" damping="10"/>
              <geom name="second_geom_{index}" type="box" size="{size} {size} {size}"/>
            </body>
            <body name="first_goal_{index}" pos="{first_goal[0]} {first_goal[1]} 0"/>
            <body name="second_goal_{index}" pos="{second_goal[0]} {second_goal[1]} 0"/>
          </worldbody>
          <actuator>
            <position name="first_actuator_{index}" joint="first_joint_{index}" kp="80" ctrlrange="-1 1"/>
            <position name="second_actuator_{index}" joint="second_joint_{index}" kp="80" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """
    )


def _strategy_manifest(index: int) -> dict[str, object]:
    return {
        "schema_version": "rapid.binding/v1",
        "program_kind": "strategy",
        "program_name": "strategy_fixture",
        "instance_name": f"variant_{index}",
        "scene_module": f"test_scenes.strategy_fixture.variant_{index}.scene",
        "goal_text": "Move both relational targets.",
        "roles": {
            "first_joint": f"first_joint_{index}",
            "first_geom": f"first_geom_{index}",
            "first_actuator": f"first_actuator_{index}",
            "second_joint": f"second_joint_{index}",
            "second_geom": f"second_geom_{index}",
            "second_actuator": f"second_actuator_{index}",
            "first_goal": f"first_goal_{index}",
            "second_goal": f"second_goal_{index}",
        },
        "inputs": {},
    }


def _compile_phase(
    model: mujoco.MjModel,
    state: np.ndarray,
    binding: object,
) -> tuple[np.ndarray, bool]:
    joint_id = model.joint(binding.one("controlled_joint")).id
    qpos_address = int(model.jnt_qposadr[joint_id])
    actuator_id = model.actuator(binding.one("controlled_actuator")).id
    displacement = float(binding.parameters["displacement"])
    desired = float(state[qpos_address] + displacement)
    nominal_action = np.zeros(model.nu, dtype=float)
    for candidate in range(model.nu):
        controlled_joint = int(model.actuator_trnid[candidate, 0])
        address = int(model.jnt_qposadr[controlled_joint])
        nominal_action[candidate] = state[address]
    nominal_action[actuator_id] = desired
    plan = NominalPlan(
        knots=np.tile(nominal_action, (2, 1)),
        sigma=np.full((2, model.nu), 0.005),
        fixed_first_action=True,
    )

    def cost(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        del controls
        return np.square(states[:, -1, qpos_address] - desired)

    def predicate(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        del controls
        return np.abs(states[:, -1, qpos_address] - desired) < 0.03

    result, _ = compile_primitive(
        PrimitiveCompileSpec.from_nominal_plan(
            nominal_plan=plan,
            model=model,
            initial_state=state,
            cost_fn=cost,
            effect_predicate=predicate,
        ),
        PrimitiveCompilerConfig(
            num_rollouts=4,
            num_nodes=2,
            num_elites=1,
            max_iters=1,
            sigma_min=0.001,
            sigma_max=0.02,
            horizon=1.5,
            optimize_initial_pose=False,
        ),
        verbose=False,
    )
    return result.best_states[-1].copy(), bool(result.metrics["success"])


class RelationalEndToEndTests(unittest.TestCase):
    def test_renamed_translated_rotated_resized_variants_need_manifests_only(self) -> None:
        self.assertEqual(
            validate_strategy_program(
                PHASE_PROGRAM,
                STRATEGY_CONTRACT,
                {"move_fixture": PRIMITIVE_CONTRACT},
            ),
            (),
        )
        variants = ((0, 0.0, 0.04), (1, math.pi / 3.0, 0.06), (2, -math.pi / 2.0, 0.08))
        for index, yaw, size in variants:
            with self.subTest(index=index):
                model = _model(index, yaw=yaw, size=size)
                strategy_binding = parse_binding_manifest(
                    _strategy_manifest(index),
                    contract=STRATEGY_CONTRACT,
                )
                self.assertTrue(
                    validate_scene_binding(
                        model,
                        STRATEGY_CONTRACT,
                        strategy_binding,
                    ).binding_valid
                )
                state = np.zeros(model.nq + model.nv, dtype=float)
                successes: list[bool] = []
                for phase_index, phase in enumerate(PHASE_PROGRAM.phases):
                    primitive_binding = primitive_binding_for_phase(
                        phase,
                        strategy_binding,
                        PRIMITIVE_CONTRACT,
                        context=PhaseConstraintContext(
                            model=model,
                            strategy_binding=strategy_binding,
                            phase_name=phase.name,
                            phase_index=phase_index,
                            state=state,
                        ),
                    )
                    self.assertTrue(
                        validate_scene_binding(
                            model,
                            PRIMITIVE_CONTRACT,
                            primitive_binding,
                        ).binding_valid
                    )
                    state, success = _compile_phase(model, state, primitive_binding)
                    successes.append(success)
                self.assertEqual(successes, [True, True])


if __name__ == "__main__":
    unittest.main()
