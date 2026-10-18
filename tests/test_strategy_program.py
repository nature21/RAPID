from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from instantiation_core import (
    BindingContract,
    EntityKind,
    ParameterRequirement,
    ParameterType,
    RoleRequirement,
    SceneBinding,
)
from primitive_core.common import (
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    CompileResult,
    OptimizationBudget,
    ToolTrajectory,
)
from primitive_core.fr3_transit import FR3TransitViolation
from primitive_core.policy import GripperOperation
from primitive_core.transit import ConfigureGripper, Settle, TransitSpec
from strategies.program import (
    ConnectingConstraint,
    InterPhaseTransit,
    PrimitivePhase,
    PhaseConstraintContext,
    StrategyProgram,
    primitive_binding_for_phase,
    validate_strategy_program,
)
from strategies.runtime import (
    _retain_partial_strategy_failure,
    concat_phase_results,
    execute_sequential_strategy,
)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="object">
              <joint name="object_x" type="slide" axis="1 0 0"/>
              <geom name="object_geom" type="box" size="0.02 0.02 0.02"/>
            </body>
            <body name="fixture" pos="0.3 0 0">
              <geom name="fixture_geom" type="box" size="0.01 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _context(
    model: mujoco.MjModel,
    binding: SceneBinding,
    phase: str,
    *,
    index: int = 0,
    position: float = 0.0,
) -> PhaseConstraintContext:
    state = np.zeros(model.nq + model.nv, dtype=float)
    state[0] = position
    return PhaseConstraintContext(
        model=model,
        strategy_binding=binding,
        phase_name=phase,
        phase_index=index,
        state=state,
    )


def _result(
    initial_state: np.ndarray,
    final_position: float,
    control: float,
) -> CompileResult:
    final_state = np.asarray(initial_state, dtype=float).copy()
    final_state[0] = final_position
    controls = np.asarray(((control,), (control,)), dtype=float)
    return CompileResult(
        trajectory=ToolTrajectory(
            times=np.asarray((0.0, 0.01)),
            positions=controls.copy(),
            controls=controls,
        ),
        metrics={"optimized_success": True, "success": True, "effect_success": True,
                 "mechanism_valid": True, "stability_required": False},
        best_states=np.vstack((initial_state, final_state)),
        cost_history=np.asarray((0.0,)),
        controller_state=np.asarray((final_position,)),
    )


def _two_phase_fixture() -> tuple[
    BindingContract,
    BindingContract,
    StrategyProgram,
    SceneBinding,
]:
    strategy_contract = BindingContract(
        program_kind="strategy",
        program_name="two_phases",
        roles=(
            RoleRequirement("tool", EntityKind.BODY, "Tool."),
            RoleRequirement("target", EntityKind.BODY, "Target."),
        ),
    )
    primitive_contract = BindingContract(
        program_kind="primitive",
        program_name="effect",
        roles=(),
    )
    program = StrategyProgram(
        phases=(
            PrimitivePhase("first", "effect", {}),
            PrimitivePhase("second", "effect", {}),
        ),
        transits=(
            InterPhaseTransit(
                "first",
                "second",
                TransitSpec(
                    stages=(ConfigureGripper(0.5, 0.1), Settle(0.1)),
                    tool_role="tool",
                    target_role="target",
                    label="test",
                ),
            ),
        ),
    )
    binding = SceneBinding(
        program_kind="strategy",
        program_name="two_phases",
        instance_name="test",
        scene_module="tasks.example.scene",
        goal_text="Run two effects.",
        roles={"tool": "fixture", "target": "object"},
    )
    return strategy_contract, primitive_contract, program, binding


class StrategyProgramTests(unittest.TestCase):
    def test_bind_primitive_preserves_live_state_and_control_handoff(self) -> None:
        model = _model()
        _, contract, program, binding = _two_phase_fixture()
        context = PhaseConstraintContext(
            model=model,
            strategy_binding=binding,
            phase_name="second",
            phase_index=1,
            state=np.asarray((0.17, -0.23)),
            controller_state=np.asarray((0.4, 0.5)),
            initial_action=np.asarray((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02)),
            initial_gripper_operation=GripperOperation.GRASP,
        )
        primitive_binding = primitive_binding_for_phase(
            program.phases[1], binding, contract, context=context,
        )

        bound = context.bind_primitive(primitive_binding)

        self.assertIs(bound.model, model)
        self.assertIs(bound.binding, primitive_binding)
        np.testing.assert_array_equal(bound.initial_state, context.state)
        np.testing.assert_array_equal(bound.data.qpos, (0.17,))
        np.testing.assert_array_equal(bound.data.qvel, (-0.23,))
        np.testing.assert_allclose(bound.data.body("object").xpos, (0.17, 0.0, 0.0))
        np.testing.assert_array_equal(bound.controller_state, context.controller_state)
        np.testing.assert_array_equal(bound.initial_action, context.initial_action)
        self.assertEqual(bound.initial_gripper_operation, GripperOperation.GRASP)

        bound.data.qpos[:] = 0.8
        bound.initial_state[:] = 0.0
        bound.controller_state[:] = 0.0
        bound.initial_action[:] = 0.0
        np.testing.assert_array_equal(context.state, (0.17, -0.23))
        np.testing.assert_array_equal(context.controller_state, (0.4, 0.5))
        self.assertEqual(context.initial_action[-1], 0.02)

    def test_bind_primitive_preserves_absent_initial_control_state(self) -> None:
        model = _model()
        _, contract, program, binding = _two_phase_fixture()
        context = _context(model, binding, "first", position=0.12)
        primitive_binding = primitive_binding_for_phase(
            program.phases[0], binding, contract, context=context,
        )

        bound = context.bind_primitive(primitive_binding)

        np.testing.assert_allclose(bound.data.body("object").xpos, (0.12, 0.0, 0.0))
        self.assertIsNone(bound.controller_state)
        self.assertIsNone(bound.initial_action)
        self.assertEqual(bound.initial_gripper_operation, GripperOperation.MOVE)

    def test_attempt_policy_continues_from_finite_rejected_transit(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        strategy_contract, primitive_contract, program, binding = (
            _two_phase_fixture()
        )
        phase_starts: dict[str, float] = {}

        def run_phase(
            phase: PrimitivePhase,
            primitive_binding: SceneBinding,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            bound = context.bind_primitive(primitive_binding)
            phase_starts[phase.name] = float(bound.data.qpos[0])
            if phase.name == "second":
                np.testing.assert_array_equal(bound.controller_state, (0.2,))
                np.testing.assert_array_equal(bound.initial_action, (0.2,))
            return _result(
                bound.initial_state,
                0.1 if phase.name == "first" else 0.3,
                0.1,
            )

        def run_transit(
            _transit: InterPhaseTransit,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            raise FR3TransitViolation(
                "terminal clearance was below the conservative threshold",
                result=_result(context.state, 0.2, 0.2),
            )

        with patch.dict(os.environ, {"RAPID_EXECUTION_POLICY": "attempt"}):
            result = execute_sequential_strategy(
                program=program,
                strategy_contract=strategy_contract,
                primitive_contracts={"effect": primitive_contract},
                strategy_binding=binding,
                model=model,
                initial_state=initial,
                run_phase=run_phase,
                run_transit=run_transit,
                dt=0.01,
            )

        self.assertEqual(phase_starts, {"first": 0.0, "second": 0.2})
        self.assertAlmostEqual(float(result.best_states[-1, 0]), 0.3)
        self.assertEqual(result.metrics["execution_policy"], "attempt")
        self.assertTrue(result.metrics["strict_would_abort"])
        self.assertEqual(len(result.metrics["attempt_continuations"]), 1)
        continuation = result.metrics["attempt_continuations"][0]
        self.assertEqual(continuation["failed_segment"], "first__to__second")
        self.assertEqual(continuation["failure_type"], "FR3TransitViolation")

    def test_strict_policy_still_stops_on_rejected_transit(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        strategy_contract, primitive_contract, program, binding = (
            _two_phase_fixture()
        )
        phases_run: list[str] = []

        def run_phase(
            phase: PrimitivePhase,
            _binding: SceneBinding,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            phases_run.append(phase.name)
            return _result(context.state, 0.1, 0.1)

        def run_transit(
            _transit: InterPhaseTransit,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            raise FR3TransitViolation(
                "rejected transit",
                result=_result(context.state, 0.2, 0.2),
            )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FR3TransitViolation):
                execute_sequential_strategy(
                    program=program,
                    strategy_contract=strategy_contract,
                    primitive_contracts={"effect": primitive_contract},
                    strategy_binding=binding,
                    model=model,
                    initial_state=initial,
                    run_phase=run_phase,
                    run_transit=run_transit,
                    dt=0.01,
                )

        self.assertEqual(phases_run, ["first"])

    def test_attempt_policy_rejects_nonfinite_retained_transit(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        strategy_contract, primitive_contract, program, binding = (
            _two_phase_fixture()
        )

        def run_phase(
            _phase: PrimitivePhase,
            _binding: SceneBinding,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            return _result(context.state, 0.1, 0.1)

        def run_transit(
            _transit: InterPhaseTransit,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            retained = _result(context.state, 0.2, 0.2)
            retained.best_states[-1, 0] = np.nan
            raise FR3TransitViolation("nonfinite transit", result=retained)

        with patch.dict(os.environ, {"RAPID_EXECUTION_POLICY": "attempt"}):
            with self.assertRaises(FR3TransitViolation):
                execute_sequential_strategy(
                    program=program,
                    strategy_contract=strategy_contract,
                    primitive_contracts={"effect": primitive_contract},
                    strategy_binding=binding,
                    model=model,
                    initial_state=initial,
                    run_phase=run_phase,
                    run_transit=run_transit,
                    dt=0.01,
                )

    def test_failed_primitive_effect_does_not_stop_later_phases(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        strategy_contract = BindingContract(
            program_kind="strategy",
            program_name="two_phases",
            roles=(
                RoleRequirement("tool", EntityKind.BODY, "Tool."),
                RoleRequirement("target", EntityKind.BODY, "Target."),
            ),
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="effect",
            roles=(),
        )
        program = StrategyProgram(
            phases=(
                PrimitivePhase("first", "effect", {}),
                PrimitivePhase("second", "effect", {}),
            ),
            transits=(
                InterPhaseTransit(
                    "first",
                    "second",
                    TransitSpec(
                        stages=(ConfigureGripper(0.5, 0.1), Settle(0.1)),
                        tool_role="tool",
                        target_role="target",
                        label="unused",
                    ),
                ),
            ),
        )
        binding = SceneBinding(
            program_kind="strategy",
            program_name="two_phases",
            instance_name="test",
            scene_module="tasks.example.scene",
            goal_text="Run two effects.",
            roles={"tool": "fixture", "target": "object"},
        )
        phase_starts: dict[str, float] = {}
        transit_start: float | None = None

        def run_phase(
            phase: PrimitivePhase,
            _binding: SceneBinding,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            phase_starts[phase.name] = float(context.state[0])
            final_position = 0.1 if phase.name == "first" else 0.3
            result = _result(context.state, final_position, final_position)
            result.metrics["optimized_success"] = phase.name == "second"
            result.metrics["effect_success"] = phase.name == "second"
            result.metrics["success"] = phase.name == "second"
            return result

        def run_transit(
            _transit: InterPhaseTransit,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            nonlocal transit_start
            transit_start = float(context.state[0])
            return _result(context.state, 0.2, 0.2)

        result = execute_sequential_strategy(
            program=program,
            strategy_contract=strategy_contract,
            primitive_contracts={"effect": primitive_contract},
            strategy_binding=binding,
            model=model,
            initial_state=initial,
            run_phase=run_phase,
            run_transit=run_transit,
            dt=0.01,
        )

        self.assertEqual(phase_starts, {"first": 0.0, "second": 0.2})
        self.assertEqual(transit_start, 0.1)
        self.assertEqual(float(result.best_states[-1, 0]), 0.3)
        self.assertEqual(
            result.metrics["primitive_phase_effect_success"],
            {"first": False, "second": True},
        )
        self.assertFalse(result.metrics["all_primitive_phase_effects_success"])
        self.assertFalse(
            result.metrics["phase_metrics"]["first"]["optimized_success"]
        )

    def test_primitive_phase_requires_effect_metric(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        strategy_contract = BindingContract(
            program_kind="strategy",
            program_name="one_phase",
            roles=(),
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="effect",
            roles=(),
        )
        program = StrategyProgram(
            phases=(PrimitivePhase("only", "effect", {}),),
        )
        binding = SceneBinding(
            program_kind="strategy",
            program_name="one_phase",
            instance_name="test",
            scene_module="tasks.example.scene",
            goal_text="Run one effect.",
            roles={},
        )

        def run_phase(*_args) -> CompileResult:
            result = _result(initial, 0.1, 0.1)
            del result.metrics["optimized_success"]
            return result

        with self.assertRaisesRegex(ValueError, "did not report optimized_success"):
            execute_sequential_strategy(
                program=program,
                strategy_contract=strategy_contract,
                primitive_contracts={"effect": primitive_contract},
                strategy_binding=binding,
                model=model,
                initial_state=initial,
                run_phase=run_phase,
                dt=0.01,
            )

    def test_primitive_phase_accepts_typed_optimization_budget(self) -> None:
        phase = PrimitivePhase(
            name="push",
            primitive="push",
            role_map={},
            optimization_budget=OptimizationBudget(64, 10),
        )

        self.assertEqual(phase.optimization_budget, OptimizationBudget(64, 10))
        with self.assertRaisesRegex(TypeError, "OptimizationBudget"):
            PrimitivePhase(  # type: ignore[arg-type]
                name="push",
                primitive="push",
                role_map={},
                optimization_budget={"num_rollouts": 64, "max_iters": 10},
            )
        with self.assertRaisesRegex(ValueError, "GLOBAL_MAX"):
            PrimitivePhase(
                name="push",
                primitive="push",
                role_map={},
                optimization_budget=OptimizationBudget(
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts + 1,
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
                ),
            )

    def test_phase_budget_cannot_be_smaller_than_primitive_reference(self) -> None:
        program = StrategyProgram(
            phases=(
                PrimitivePhase(
                    name="push",
                    primitive="push",
                    role_map={},
                    optimization_budget=OptimizationBudget(32, 6),
                ),
            )
        )
        findings = validate_strategy_program(
            program,
            BindingContract(
                program_kind="strategy",
                program_name="strategy",
                roles=(),
            ),
            {
                "push": BindingContract(
                    program_kind="primitive",
                    program_name="push",
                    roles=(),
                )
            },
            {"push": OptimizationBudget(64, 10)},
        )

        self.assertTrue(any("cannot be smaller" in item for item in findings))

        missing_reference = validate_strategy_program(
            program,
            BindingContract(
                program_kind="strategy",
                program_name="strategy",
                roles=(),
            ),
            {
                "push": BindingContract(
                    program_kind="primitive",
                    program_name="push",
                    roles=(),
                )
            },
            {},
        )
        self.assertTrue(
            any("has no reference budget" in item for item in missing_reference)
        )

        excessive_reference = validate_strategy_program(
            StrategyProgram(
                phases=(PrimitivePhase("push", "push", {}),),
            ),
            BindingContract(
                program_kind="strategy",
                program_name="strategy",
                roles=(),
            ),
            {
                "push": BindingContract(
                    program_kind="primitive",
                    program_name="push",
                    roles=(),
                )
            },
            {
                "push": OptimizationBudget(
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts + 1,
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
                )
            },
        )
        self.assertTrue(
            any("exceeds GLOBAL_MAX" in item for item in excessive_reference)
        )

    def test_failure_retention_includes_history_and_rejected_segment(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        completed = _result(initial, 0.1, 0.1)
        rejected = _result(completed.best_states[-1], 0.2, 0.1)
        failure = RuntimeError("rejected transit")
        failure.result = rejected

        _retain_partial_strategy_failure(
            failure,
            (completed,),
            ("push",),
            failed_name="push__to__flip",
            dt=0.01,
            resolved_inputs={"push": {"displacement": 0.1}},
        )

        partial = failure.result
        self.assertIsInstance(partial, CompileResult)
        self.assertEqual(partial.best_states.shape[0], 4)
        self.assertAlmostEqual(partial.best_states[0, 0], 0.0)
        self.assertAlmostEqual(partial.best_states[-1, 0], 0.2)
        self.assertTrue(partial.metrics["partial_strategy"])
        self.assertEqual(partial.metrics["completed_segments"], ["push"])
        self.assertEqual(
            partial.metrics["retained_segments"],
            ["push", "push__to__flip__rejected"],
        )

    @patch.dict(os.environ, {"RAPID_EXECUTION_POLICY": "attempt"})
    def test_later_failure_retains_attempt_continuation_audit(self) -> None:
        model = _model()
        initial = np.zeros(model.nq + model.nv, dtype=float)
        completed = _result(initial, 0.1, 0.1)
        continuation = {
            "accepted": True,
            "failed_segment": "push__to__flip",
            "failure_type": "FR3TransitViolation",
        }
        completed.metrics["attempt_continuation"] = continuation
        failure = RuntimeError("downstream phase failed")

        _retain_partial_strategy_failure(
            failure,
            (completed,),
            ("push__to__flip",),
            failed_name="grasp",
            dt=0.01,
            resolved_inputs={},
        )

        partial = failure.result
        self.assertEqual(partial.metrics["execution_policy"], "attempt")
        self.assertTrue(partial.metrics["strict_would_abort"])
        self.assertEqual(partial.metrics["attempt_continuations"], [continuation])

    def test_phase_concatenation_reports_aperture_boundary_delta(self) -> None:
        def phase(aperture: float) -> CompileResult:
            controls = np.zeros((2, 7), dtype=float)
            controls[:, 6] = aperture
            return CompileResult(
                trajectory=ToolTrajectory(
                    times=np.asarray((0.0, 0.01)),
                    positions=controls.copy(),
                    controls=controls,
                    gripper_operations=np.zeros(2, dtype=np.uint8),
                ),
                metrics={},
                best_states=np.zeros((2, 4), dtype=float),
                cost_history=np.empty((0,), dtype=float),
            )

        result = concat_phase_results((phase(0.03), phase(0.03)), 0.01)

        self.assertEqual(len(result.metrics["phase_boundary_aperture_deltas"]), 1)
        self.assertAlmostEqual(
            result.metrics["phase_boundary_aperture_deltas"][0],
            0.0,
        )
        np.testing.assert_array_equal(
            result.trajectory.gripper_operations,
            np.zeros(4, dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "terminal gripper command"):
            concat_phase_results((phase(0.03), phase(0.02)), 0.01)

    def test_strategy_contracts_are_role_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "role-only"):
            BindingContract(
                program_kind="strategy",
                program_name="strategy_1",
                roles=(RoleRequirement("object", EntityKind.BODY, "Target."),),
                parameters=(
                    ParameterRequirement(
                        "distance",
                        ParameterType.FLOAT,
                        "Requested displacement.",
                    ),
                ),
            )

    def test_connecting_constraints_cover_primitive_parameters(self) -> None:
        strategy_contract = BindingContract(
            program_kind="strategy",
            program_name="push_to_fixture",
            roles=(
                RoleRequirement("object", EntityKind.BODY, "Target."),
                RoleRequirement("fixture", EntityKind.BODY, "Goal fixture."),
            ),
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=(RoleRequirement("target", EntityKind.BODY, "Target."),),
            parameters=(
                ParameterRequirement(
                    "displacement",
                    ParameterType.FLOAT,
                    "Requested displacement.",
                    minimum=-0.5,
                    maximum=0.5,
                ),
            ),
        )
        valid = StrategyProgram(
            phases=(
                PrimitivePhase(
                    "push_phase",
                    "push",
                    {"target": "object"},
                    (
                        ConnectingConstraint(
                            "displacement",
                            "Move from the target to the fixture goal.",
                            ("object", "fixture"),
                            "support_x",
                            lambda context: 0.2,
                        ),
                    ),
                ),
            )
        )
        missing = StrategyProgram(
            phases=(
                PrimitivePhase(
                    "push_phase",
                    "push",
                    {"target": "object"},
                ),
            )
        )

        self.assertEqual(
            validate_strategy_program(
                valid,
                strategy_contract,
                {"push": primitive_contract},
            ),
            (),
        )
        findings = validate_strategy_program(
            missing,
            strategy_contract,
            {"push": primitive_contract},
        )
        self.assertTrue(any("connecting constraints" in item for item in findings))

    def test_connecting_constraints_cover_optional_primitive_parameters(self) -> None:
        strategy_contract = BindingContract(
            program_kind="strategy",
            program_name="push_to_fixture",
            roles=(RoleRequirement("object", EntityKind.BODY, "Target."),),
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=(RoleRequirement("target", EntityKind.BODY, "Target."),),
            parameters=(
                ParameterRequirement(
                    "direction",
                    ParameterType.VECTOR,
                    "Requested push direction.",
                    required=False,
                    vector_length=2,
                ),
            ),
        )
        program = StrategyProgram(
            phases=(
                PrimitivePhase(
                    "push_phase",
                    "push",
                    {"target": "object"},
                ),
            ),
        )

        findings = validate_strategy_program(
            program,
            strategy_contract,
            {"push": primitive_contract},
        )

        self.assertTrue(
            any("primitive inputs ['direction']" in item for item in findings)
        )

    def test_phase_binding_resolves_live_constraint_values(self) -> None:
        model = _model()
        strategy_binding = SceneBinding(
            program_kind="strategy",
            program_name="push_to_fixture",
            instance_name="renamed",
            scene_module="tasks.example.scene",
            goal_text="Push the target to the fixture.",
            roles={"object": "object", "fixture": "fixture"},
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=(RoleRequirement("target", EntityKind.BODY, "Target."),),
            parameters=(
                ParameterRequirement(
                    "displacement",
                    ParameterType.FLOAT,
                    "Requested displacement.",
                    minimum=-0.5,
                    maximum=0.5,
                ),
            ),
        )
        phase = PrimitivePhase(
            "push_phase",
            "push",
            {"target": "object"},
            (
                ConnectingConstraint(
                    "displacement",
                    "Remaining distance to the fixture.",
                    ("object", "fixture"),
                    "support_x",
                    lambda context: 0.3 - float(context.state[0]),
                ),
            ),
        )

        binding = primitive_binding_for_phase(
            phase,
            strategy_binding,
            primitive_contract,
            context=_context(
                model,
                strategy_binding,
                phase.name,
                position=0.1,
            ),
        )

        self.assertEqual(binding.one("target"), "object")
        self.assertAlmostEqual(binding.parameters["displacement"], 0.2)

    def test_sequential_executor_resolves_after_previous_phase(self) -> None:
        model = _model()
        strategy_contract = BindingContract(
            program_kind="strategy",
            program_name="two_pushes",
            roles=(
                RoleRequirement("object", EntityKind.BODY, "Target."),
                RoleRequirement("fixture", EntityKind.BODY, "Goal fixture."),
            ),
        )
        primitive_contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=(RoleRequirement("target", EntityKind.BODY, "Target."),),
            parameters=(
                ParameterRequirement(
                    "displacement",
                    ParameterType.FLOAT,
                    "Requested displacement.",
                    minimum=-0.5,
                    maximum=0.5,
                ),
            ),
        )

        def first_goal(context: PhaseConstraintContext) -> float:
            return 0.1 - float(context.state[0])

        def final_goal(context: PhaseConstraintContext) -> float:
            return 0.3 - float(context.state[0])

        program = StrategyProgram(
            phases=(
                PrimitivePhase(
                    "coarse_push",
                    "push",
                    {"target": "object"},
                    (
                        ConnectingConstraint(
                            "displacement",
                            "Reach the intermediate relation.",
                            ("object", "fixture"),
                            "support_x",
                            first_goal,
                        ),
                    ),
                ),
                PrimitivePhase(
                    "final_push",
                    "push",
                    {"target": "object"},
                    (
                        ConnectingConstraint(
                            "displacement",
                            "Reach the fixture relation.",
                            ("object", "fixture"),
                            "support_x",
                            final_goal,
                        ),
                    ),
                ),
            ),
            transits=(
                InterPhaseTransit(
                    "coarse_push",
                    "final_push",
                    TransitSpec(
                        stages=(ConfigureGripper(0.5, 0.1), Settle(0.1)),
                        tool_role="tool",
                        target_role="target",
                        label="advisory_transit",
                    ),
                ),
            ),
        )
        strategy_binding = SceneBinding(
            program_kind="strategy",
            program_name="two_pushes",
            instance_name="example_0",
            scene_module="tasks.example.scene",
            goal_text="Push in two locally optimized phases.",
            roles={"object": "object", "fixture": "fixture"},
        )
        calls: list[tuple[str, float, float]] = []

        def run_phase(
            phase: PrimitivePhase,
            binding: SceneBinding,
            context: PhaseConstraintContext,
        ) -> CompileResult:
            displacement = float(binding.parameters["displacement"])
            initial_position = float(context.state[0])
            calls.append((phase.name, initial_position, displacement))
            return _result(
                context.state,
                initial_position + displacement,
                displacement,
            )

        def run_transit(_transit, context: PhaseConstraintContext) -> CompileResult:
            result = _result(context.state, 0.15, 0.1)
            result.metrics.update(
                {
                    "transit_valid": True,
                    "transit_convergence_valid": False,
                    "whole_arm_audit_valid": False,
                    "audit_advisories": [
                        "transit_convergence_or_joint_margin",
                        "whole_arm_clearance_or_contact",
                    ],
                    "audit_advisory_triggered": True,
                }
            )
            return result

        result = execute_sequential_strategy(
            program=program,
            strategy_contract=strategy_contract,
            primitive_contracts={"push": primitive_contract},
            strategy_binding=strategy_binding,
            model=model,
            initial_state=np.zeros(model.nq + model.nv, dtype=float),
            run_phase=run_phase,
            run_transit=run_transit,
            dt=0.01,
        )

        self.assertEqual([item[0] for item in calls], ["coarse_push", "final_push"])
        self.assertAlmostEqual(calls[0][1], 0.0)
        self.assertAlmostEqual(calls[0][2], 0.1)
        self.assertAlmostEqual(calls[1][1], 0.15)
        self.assertAlmostEqual(calls[1][2], 0.15)
        self.assertAlmostEqual(result.best_states[-1, 0], 0.3)
        self.assertEqual(result.metrics["local_primitive_optimization_count"], 2)
        self.assertAlmostEqual(
            result.metrics["resolved_primitive_inputs"]["final_push"][
                "displacement"
            ],
            0.15,
        )
        transit_metrics = result.metrics["phase_metrics"][
            "coarse_push__to__final_push"
        ]
        self.assertTrue(transit_metrics["audit_advisory_triggered"])
        self.assertFalse(transit_metrics["transit_convergence_valid"])


if __name__ == "__main__":
    unittest.main()
