from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from instantiation_core import (
    BindingContract,
    BindingReport,
    EntityKind,
    ParameterRequirement,
    ParameterType,
    RelationKind,
    RelationalInvariant,
    RoleRequirement,
    SceneBinding,
    inventory_model,
    to_json_value,
    validate_scene_binding,
)
from instantiation_core.introspect import find_slide, freejoint_qpos_dof
from instantiation_core.entrypoint import (
    instantiation_results_dir,
)
from instantiation_core.runner import (
    FR3_GOAL_SCHEMA,
    FR3_GOAL_SCHEMA_VERSION,
    InstantiationSpec,
    _attempt_budget,
    _failure_signature,
    _program_design_fingerprint,
    _program_execution_fingerprint,
    _search_budget_tier,
    apply_optimizer_env_overrides,
    cartesian_goals_path_for_metrics,
    make_parser,
    run_instantiation,
    write_cartesian_goal_trajectory,
    write_state_trajectory,
)
from primitive_core.common import CompileResult, ToolTrajectory
from primitive_core.fr3_osc import FR3OSCAdapter
from primitive_core.fr3_safety import FR3SafetyViolation
from scene_io.playback import PlaybackBundle


_XML = """
<mujoco model="renamed_scene">
  <worldbody>
    <geom name="renamed_support" type="plane" size="1 1 0.01"/>
    <body name="renamed_target" pos="0 0 0.05">
      <freejoint name="renamed_target_joint"/>
      <geom name="renamed_target_shape" type="box" size="0.1 0.05 0.05"/>
      <site name="renamed_target_site"/>
    </body>
    <body name="renamed_tool">
      <joint name="renamed_tool_x" type="slide" axis="1 0 0"/>
      <geom name="renamed_tool_shape" type="sphere" size="0.02"/>
      <site name="renamed_contact_site"/>
    </body>
  </worldbody>
  <actuator>
    <position name="renamed_tool_actuator" joint="renamed_tool_x"/>
  </actuator>
</mujoco>
"""


def _contract() -> BindingContract:
    return BindingContract(
        program_kind="primitive",
        program_name="push",
        roles=(
            RoleRequirement(
                role="target_body",
                kind=EntityKind.BODY,
                description="Manipulated free body.",
                required_body_joint_types=("free",),
            ),
            RoleRequirement(
                role="target_geom",
                kind=EntityKind.GEOM,
                description="Manipulated box collision geom.",
                allowed_types=("box",),
                same_body_as="target_body",
            ),
            RoleRequirement(
                role="tool_site",
                kind=EntityKind.SITE,
                description="Tool contact feature.",
            ),
            RoleRequirement(
                role="support_geom",
                kind=EntityKind.GEOM,
                description="Support surface.",
                allowed_types=("plane", "box"),
            ),
            RoleRequirement(
                role="translation_actuator",
                kind=EntityKind.ACTUATOR,
                description="Tool translation control.",
                allowed_types=("joint",),
            ),
        ),
        invariants=("target starts supported",),
        relations=(
            RelationalInvariant(
                relation=RelationKind.SAME_BODY,
                left_role="target_body",
                right_role="target_geom",
                description="The target geom must belong to the target body.",
            ),
        ),
    )


def _binding(**overrides: str) -> SceneBinding:
    roles = {
        "target_body": "renamed_target",
        "target_geom": "renamed_target_shape",
        "tool_site": "renamed_contact_site",
        "support_geom": "renamed_support",
        "translation_actuator": "renamed_tool_actuator",
    }
    roles.update(overrides)
    return SceneBinding(
        program_kind="primitive",
        program_name="push",
        instance_name="renamed",
        scene_module="tasks.example.renamed.scene",
        goal_text="Push the target across the support.",
        roles=roles,
    )


class InstantiationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_XML)

    def test_state_trajectory_export_is_numeric_and_pickle_free(self) -> None:
        timestep = float(self.model.opt.timestep)
        state_dim = self.model.nq + self.model.nv
        states = np.zeros((2, state_dim), dtype=float)
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, timestep)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": True},
            best_states=states,
            cost_history=np.asarray((1.0,)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.npz"
            write_state_trajectory(path, result)
            with np.load(path, allow_pickle=False) as retained:
                self.assertEqual(str(retained["schema"]), "rapid.state_trajectory")
                self.assertEqual(int(retained["schema_version"]), 1)
                np.testing.assert_array_equal(retained["states"], states)

    def test_instantiation_results_live_under_active_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            with mock.patch.dict(
                os.environ,
                {"RAPID_WORKSPACE": str(workspace)},
            ):
                result = instantiation_results_dir(
                    workspace / "src/instantiations/primitives/push/example/binding.json",
                    program_kind="primitive",
                    program_name="push",
                    instance_name="example",
                )

        self.assertEqual(
            result,
            workspace / "results_instantiation/primitives/push/example",
        )

    def test_cartesian_goals_are_commanded_poses_at_real_player_rate(self) -> None:
        adapter = FR3OSCAdapter(
            frame_position_world=(0.0, 0.0, 0.0),
            frame_quaternion_world=(1.0, 0.0, 0.0, 0.0),
            nullspace_configuration=(0.0,) * 7,
            gripper_range=(0.0, 0.04),
        )
        controls = np.zeros((5, 7), dtype=float)
        controls[:, 0] = np.linspace(0.0, 0.4, 5)
        controls[:, 6] = np.linspace(0.04, 0.0, 5)
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.linspace(0.0, 0.04, 5),
                positions=controls.copy(),
                controls=controls,
                gripper_operations=np.ones(5, dtype=np.uint8),
            ),
            metrics={},
            best_states=np.empty((5, 0)),
            cost_history=np.empty((0,)),
        )

        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics_retry_1.json"
            path = cartesian_goals_path_for_metrics(metrics)
            samples = write_cartesian_goal_trajectory(path, result, adapter)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "cartesian_goals_retry_1.json")
        self.assertEqual(samples, 3)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["schema"], FR3_GOAL_SCHEMA)
        self.assertEqual(payload["version"], FR3_GOAL_SCHEMA_VERSION)
        self.assertEqual(payload["sample_rate_hz"], 50.0)
        self.assertEqual(payload["time_semantics"], "zero_order_hold")
        self.assertEqual(payload["pose_frame"], "fr3_link0")
        self.assertEqual(payload["controlled_frame"], "fr3_hand_tcp")
        self.assertEqual(payload["quaternion_order"], "xyzw")
        self.assertEqual(
            payload["expected_F_T_EE"],
            {
                "position": [0.0, 0.0, 0.1034],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        )
        self.assertEqual(payload["gripper"]["width_semantic"], "total")
        self.assertEqual(payload["gripper"]["move_speed_m_s"], 0.04)
        self.assertEqual(payload["gripper"]["grasp_force_n"], 40.0)
        goals = payload["goals"]
        self.assertEqual(
            [entry["position_m"][0] for entry in goals],
            [0.0, 0.2, 0.4],
        )
        self.assertEqual(
            [entry["orientation_xyzw"] for entry in goals],
            [[0.0, 0.0, 0.0, 1.0]] * 3,
        )
        self.assertEqual(
            [entry["gripper_width_m"] for entry in goals],
            [0.08, 0.04, 0.0],
        )
        self.assertEqual(
            [entry["gripper_operation"] for entry in goals],
            ["grasp"] * 3,
        )

    def test_cartesian_goals_are_transformed_into_rotated_robot_base(self) -> None:
        half_sqrt = np.sqrt(0.5)
        adapter = FR3OSCAdapter(
            frame_position_world=(1.0, 2.0, 3.0),
            frame_quaternion_world=(half_sqrt, 0.0, 0.0, half_sqrt),
            nullspace_configuration=(0.1,) * 7,
            gripper_range=(0.0, 0.04),
        )
        controls = np.asarray([[0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 0.02]])
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray([0.0]),
                positions=controls.copy(),
                controls=controls,
                gripper_operations=np.zeros(1, dtype=np.uint8),
            ),
            metrics={},
            best_states=np.empty((1, 0)),
            cost_history=np.empty((0,)),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goals.json"
            write_cartesian_goal_trajectory(path, result, adapter)
            payload = json.loads(path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(
            payload["goals"][0]["position_m"], controls[0, :3]
        )
        np.testing.assert_allclose(
            payload["goals"][0]["orientation_xyzw"],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-12,
        )
        self.assertEqual(payload["goals"][0]["gripper_width_m"], 0.04)
        self.assertEqual(
            payload["source_scene_base_pose_world"]["position"],
            [1.0, 2.0, 3.0],
        )

    def test_cartesian_export_does_not_append_an_off_cadence_terminal_goal(
        self,
    ) -> None:
        adapter = FR3OSCAdapter(
            frame_position_world=(0.0, 0.0, 0.0),
            frame_quaternion_world=(1.0, 0.0, 0.0, 0.0),
            nullspace_configuration=(0.0,) * 7,
            gripper_range=(0.0, 0.04),
        )
        controls = np.zeros((6, 7), dtype=float)
        controls[:, 0] = np.arange(6) * 0.005
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.arange(6) * 0.005,
                positions=controls.copy(),
                controls=controls,
                gripper_operations=np.zeros(6, dtype=np.uint8),
            ),
            metrics={},
            best_states=np.empty((6, 0)),
            cost_history=np.empty((0,)),
        )

        with tempfile.TemporaryDirectory() as tmp:
            samples = write_cartesian_goal_trajectory(
                Path(tmp) / "goals.json",
                result,
                adapter,
            )

        self.assertEqual(samples, 2)

    def test_structurally_valid_renamed_scene_binds(self) -> None:
        report = validate_scene_binding(self.model, _contract(), _binding())

        self.assertTrue(report.binding_valid)
        self.assertIsNone(report.program_success)
        self.assertFalse(report.with_program_success(False).program_success)
        self.assertIsInstance(_binding().parameters, type(report.metadata))
        qpos, dof = freejoint_qpos_dof(self.model, "renamed_target")
        self.assertEqual((qpos, dof), (0, 0))
        self.assertEqual(find_slide(self.model, "renamed_tool", "x"), 7)

    def test_missing_name_and_wrong_geom_type_are_actionable(self) -> None:
        missing = validate_scene_binding(
            self.model,
            _contract(),
            _binding(tool_site="does_not_exist"),
        )
        wrong_type = validate_scene_binding(
            self.model,
            _contract(),
            _binding(target_geom="renamed_tool_shape"),
        )

        self.assertFalse(missing.binding_valid)
        self.assertIn("missing_entity", {issue.code for issue in missing.issues})
        self.assertFalse(wrong_type.binding_valid)
        self.assertIn(
            "wrong_entity_type",
            {issue.code for issue in wrong_type.issues},
        )
        self.assertIn(
            "wrong_parent_body",
            {issue.code for issue in wrong_type.issues},
        )
        self.assertIn(
            "relational_invariant_failed",
            {issue.code for issue in wrong_type.issues},
        )

    def test_declared_parameters_enforce_presence_type_and_range(self) -> None:
        contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=_contract().roles,
            parameters=(
                ParameterRequirement(
                    name="distance",
                    value_type=ParameterType.FLOAT,
                    description="Requested displacement.",
                    minimum=0.01,
                    maximum=0.5,
                ),
            ),
        )
        base = _binding()

        missing = validate_scene_binding(self.model, contract, base)
        wrong = validate_scene_binding(
            self.model,
            contract,
            SceneBinding(
                program_kind=base.program_kind,
                program_name=base.program_name,
                instance_name=base.instance_name,
                scene_module=base.scene_module,
                goal_text=base.goal_text,
                roles=base.roles,
                parameters={"distance": "far"},
            ),
        )
        out_of_range = validate_scene_binding(
            self.model,
            contract,
            SceneBinding(
                program_kind=base.program_kind,
                program_name=base.program_name,
                instance_name=base.instance_name,
                scene_module=base.scene_module,
                goal_text=base.goal_text,
                roles=base.roles,
                parameters={"distance": 0.8},
            ),
        )

        self.assertIn("missing_parameter", {issue.code for issue in missing.issues})
        self.assertIn("wrong_parameter_type", {issue.code for issue in wrong.issues})
        self.assertIn(
            "parameter_out_of_range",
            {issue.code for issue in out_of_range.issues},
        )

    def test_missing_role_and_nested_parameters_are_checked_and_immutable(
        self,
    ) -> None:
        roles = dict(_binding().roles)
        roles.pop("translation_actuator")
        binding = SceneBinding(
            program_kind="primitive",
            program_name="push",
            instance_name="renamed",
            scene_module="tasks.example.renamed.scene",
            goal_text="Push the target.",
            roles=roles,
            parameters={"goal": {"axis": [1.0, 0.0, 0.0]}},
        )

        report = validate_scene_binding(self.model, _contract(), binding)

        self.assertFalse(report.binding_valid)
        self.assertIn("role_count", {issue.code for issue in report.issues})
        self.assertEqual(binding.parameters["goal"]["axis"], (1.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            binding.parameters["goal"]["new"] = 1

    def test_json_normalization_thaws_nested_runtime_values(self) -> None:
        report = BindingReport(
            program_kind="primitive",
            program_name="push",
            instance_name="renamed",
            scene_module="tasks.example.renamed.scene",
            binding_valid=True,
            metadata={
                "generalization": {
                    "flags": frozenset({"beta", "alpha"}),
                    "enabled": np.bool_(True),
                    "weights": np.asarray((1.0, 2.0)),
                    "artifact": Path("results/example.json"),
                    "kind": ParameterType.FLOAT,
                }
            },
        )

        payload = report.to_dict()
        rendered = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            payload["metadata"]["generalization"]["flags"],
            ["alpha", "beta"],
        )
        self.assertIs(
            payload["metadata"]["generalization"]["enabled"],
            True,
        )
        self.assertEqual(
            payload["metadata"]["generalization"]["weights"],
            [1.0, 2.0],
        )
        self.assertIn('"kind": "float"', rendered)

    def test_json_normalization_rejects_unknown_values_with_path(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported JSON value at \$\.metrics\.bad: object",
        ):
            to_json_value({"metrics": {"bad": object()}})

    def test_inventory_contains_names_hierarchy_types_and_controls(self) -> None:
        inventory = inventory_model(self.model)

        self.assertEqual(inventory["summary"]["model"], "renamed_scene")
        self.assertEqual(
            {body["name"] for body in inventory["bodies"]},
            {"world", "renamed_target", "renamed_tool"},
        )
        target_geom = next(
            geom
            for geom in inventory["geoms"]
            if geom["name"] == "renamed_target_shape"
        )
        self.assertEqual(target_geom["type"], "mjGEOM_BOX")
        self.assertEqual(target_geom["body"], "renamed_target")
        self.assertEqual(
            inventory["actuators"][0]["name"],
            "renamed_tool_actuator",
        )
        self.assertEqual(
            inventory["actuators"][0]["transmission_targets"][0]["name"],
            "renamed_tool_x",
        )

    def test_contract_rejects_relation_to_unknown_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown role"):
            BindingContract(
                program_kind="primitive",
                program_name="push",
                roles=(),
                relations=(
                    RelationalInvariant(
                        relation=RelationKind.SAME_BODY,
                        left_role="missing_a",
                        right_role="missing_b",
                        description="Invalid test relation.",
                    ),
                ),
            )

    def test_optimizer_env_caps_cover_multiphase_strategy_arguments(self) -> None:
        args = argparse.Namespace(
            num_rollouts=72,
            max_parallel_rollouts=6,
            reorient_iters=25,
            push_iters=35,
            reorient_horizon=3.0,
            push_horizon=3.2,
            seed=42,
        )
        env = {
            "RAPID_OPTIMIZER_NUM_ROLLOUTS": "8",
            "RAPID_MAX_PARALLEL_ROLLOUTS": "2",
            "RAPID_OPTIMIZER_MAX_ITERS": "1",
            "RAPID_OPTIMIZER_HORIZON": "0.25",
        }

        with mock.patch.dict(os.environ, env, clear=False):
            returned = apply_optimizer_env_overrides(args)

        self.assertIs(returned, args)
        self.assertEqual(args.num_rollouts, 8)
        self.assertEqual(args.max_parallel_rollouts, 2)
        self.assertEqual(args.reorient_iters, 1)
        self.assertEqual(args.push_iters, 1)
        self.assertEqual(args.reorient_horizon, 0.25)
        self.assertEqual(args.push_horizon, 0.25)
        self.assertEqual(args.seed, 42)

    def test_optimizer_env_caps_never_enlarge_a_smoke_run(self) -> None:
        args = argparse.Namespace(
            num_rollouts=4,
            max_parallel_rollouts=2,
            max_iters=1,
            horizon=0.2,
        )
        env = {
            "RAPID_OPTIMIZER_NUM_ROLLOUTS": "32",
            "RAPID_MAX_PARALLEL_ROLLOUTS": "16",
            "RAPID_OPTIMIZER_MAX_ITERS": "8",
            "RAPID_OPTIMIZER_HORIZON": "1.0",
        }

        with mock.patch.dict(os.environ, env, clear=False):
            apply_optimizer_env_overrides(args)

        self.assertEqual(args.num_rollouts, 4)
        self.assertEqual(args.max_parallel_rollouts, 2)
        self.assertEqual(args.max_iters, 1)
        self.assertEqual(args.horizon, 0.2)

    def test_retained_safety_failure_writes_metrics_and_recording_before_raise(
        self,
    ) -> None:
        timestep = float(self.model.opt.timestep)
        state_dim = self.model.nq + self.model.nv
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, timestep)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"whole_arm_audit_valid": False},
            best_states=np.zeros((2, state_dim)),
            cost_history=np.asarray((1.0,)),
        )

        def fail_compile(_config: object) -> CompileResult:
            raise FR3SafetyViolation("unsafe retained trajectory", result=result)

        playback = PlaybackBundle(
            browser=mock.Mock(),
            mujoco=mock.Mock(),
            record=mock.Mock(),
        )
        spec = InstantiationSpec(
            label="unsafe test",
            compile_fn=fail_compile,
            playback=playback,
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            recording_path = Path(tmp) / "recording.mp4"
            args = make_parser("test").parse_args(
                [
                    "--metrics-json",
                    str(metrics_path),
                    "--record",
                    str(recording_path),
                ]
            )

            with self.assertRaises(FR3SafetyViolation):
                run_instantiation(spec, args)

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertFalse(payload["metrics"]["success"])
        self.assertFalse(payload["metrics"]["whole_arm_audit_valid"])
        self.assertEqual(
            payload["metrics"]["failure_signature"],
            "effect:whole_arm:clearance",
        )
        playback.record.assert_called_once()

    def test_nested_binding_metadata_writes_metrics_and_recording(self) -> None:
        timestep = float(self.model.opt.timestep)
        state_dim = self.model.nq + self.model.nv
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, timestep)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": True, "score": np.float64(1.5)},
            best_states=np.zeros((2, state_dim)),
            cost_history=np.asarray((1.0,)),
        )
        playback = PlaybackBundle(
            browser=mock.Mock(),
            mujoco=mock.Mock(),
            record=mock.Mock(),
        )
        report = BindingReport(
            program_kind="primitive",
            program_name="push",
            instance_name="renamed",
            scene_module="tasks.example.renamed.scene",
            binding_valid=True,
            metadata={
                "generalization": {
                    "specification": {
                        "goal": (np.float64(0.2), np.bool_(True))
                    }
                }
            },
        )
        spec = InstantiationSpec(
            label="nested metadata test",
            compile_fn=lambda _config: result,
            playback=playback,
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
            binding_report=report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            recording_path = Path(tmp) / "recording.mp4"
            args = make_parser("test").parse_args(
                [
                    "--metrics-json",
                    str(metrics_path),
                    "--record",
                    str(recording_path),
                ]
            )

            run_instantiation(spec, args)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["binding"]["metadata"]["generalization"][
                "specification"
            ]["goal"],
            [0.2, True],
        )
        self.assertEqual(payload["metrics"]["score"], 1.5)
        playback.record.assert_called_once()

    def test_postprocessing_failure_retains_completed_execution(self) -> None:
        from primitive_core.result_retention import retain_execution_result

        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)), positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": True},
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        original_error = AttributeError("optional reporting metadata unavailable")
        def fail_after_execution(_config):
            retain_execution_result(result)
            raise original_error
        spec = InstantiationSpec(
            label="postprocessing test", compile_fn=fail_after_execution,
            playback=PlaybackBundle(browser=mock.Mock(), mujoco=mock.Mock(), record=mock.Mock()),
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, pose: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.json"
            states = Path(tmp) / "states.npz"
            args = make_parser("test").parse_args(["--metrics-json", str(metrics), "--state-trajectory-npz", str(states)])
            with self.assertRaises(AttributeError) as caught:
                run_instantiation(spec, args)
            self.assertIs(caught.exception, original_error)
            self.assertTrue(states.is_file())
            payload = json.loads(metrics.read_text())
            self.assertFalse(payload["metrics"]["success"])
            self.assertEqual(payload["metrics"]["execution_error"]["type"], "AttributeError")

    def test_scene_reconstruction_failure_preserves_physical_evidence(self) -> None:
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)), positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": True},
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        spec = InstantiationSpec(
            label="recording test", compile_fn=lambda config: result,
            playback=PlaybackBundle(browser=mock.Mock(), mujoco=mock.Mock(), record=mock.Mock()),
            build_scene_data=mock.Mock(side_effect=RuntimeError("recording scene unavailable")),
            apply_initial_pose=lambda data, pose: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.json"
            states = Path(tmp) / "states.npz"
            args = make_parser("test").parse_args(["--metrics-json", str(metrics), "--state-trajectory-npz", str(states), "--record", str(Path(tmp) / "video.mp4")])
            with self.assertRaisesRegex(RuntimeError, "recording scene unavailable"):
                run_instantiation(spec, args)
            self.assertTrue(metrics.is_file())
            self.assertTrue(states.is_file())
            spec.playback.record.assert_not_called()

    def test_metrics_serialization_failure_does_not_suppress_recording(
        self,
    ) -> None:
        timestep = float(self.model.opt.timestep)
        state_dim = self.model.nq + self.model.nv
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, timestep)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": False, "unsupported": object()},
            best_states=np.zeros((2, state_dim)),
            cost_history=np.asarray((1.0,)),
        )
        playback = PlaybackBundle(
            browser=mock.Mock(),
            mujoco=mock.Mock(),
            record=mock.Mock(),
        )
        spec = InstantiationSpec(
            label="serialization failure test",
            compile_fn=lambda _config: result,
            playback=playback,
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            args = make_parser("test").parse_args(
                [
                    "--metrics-json",
                    str(Path(tmp) / "metrics.json"),
                    "--record",
                    str(Path(tmp) / "recording.mp4"),
                ]
            )
            with self.assertRaisesRegex(TypeError, r"\$\.metrics\.unsupported"):
                run_instantiation(spec, args)

        playback.record.assert_called_once()

    def test_successful_predicate_is_not_overwritten_by_audit_advisory(
        self,
    ) -> None:
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={"success": True, "whole_arm_audit_valid": False},
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        spec = InstantiationSpec(
            label="advisory audit test",
            compile_fn=lambda _config: result,
            playback=PlaybackBundle(
                browser=mock.Mock(),
                mujoco=mock.Mock(),
                record=mock.Mock(),
            ),
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )

        finished = run_instantiation(spec, make_parser("test").parse_args([]))

        self.assertTrue(finished.metrics["success"])
        self.assertEqual(finished.metrics["failure_signature"], "success")

    def test_failure_signature_separates_mechanism_from_effect(self) -> None:
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={
                "success": False,
                "effect_success": True,
                "mechanism_valid": False,
            },
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        spec = InstantiationSpec(
            label="mechanism test",
            compile_fn=lambda _config: result,
            playback=PlaybackBundle(
                browser=mock.Mock(),
                mujoco=mock.Mock(),
                record=mock.Mock(),
            ),
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )

        finished = run_instantiation(spec, make_parser("test").parse_args([]))

        self.assertEqual(
            finished.metrics["failure_signature"],
            "mechanism:required_relations",
        )

    def test_failure_signature_prioritizes_the_failed_transit_segment(self) -> None:
        metrics = {
            "success": False,
            "failed_segment": "flip_to_press",
            "failure_type": "FR3TransitViolation",
            "failure_message": (
                "failed inherited-contact separation after 3 controller rollouts"
            ),
            "phase_metrics": {
                "flip": {
                    "success": False,
                    "effect_success": False,
                }
            },
        }

        self.assertEqual(
            _failure_signature(metrics),
            "segment:flip_to_press:transit:release",
        )

    def test_strategy_design_fingerprint_tracks_only_behavioral_dependencies(
        self,
    ) -> None:
        report = BindingReport(
            program_kind="strategy",
            program_name="arrange",
            instance_name="example_0",
            scene_module="tasks.example.scene",
            binding_valid=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            strategy = source / "strategies/arrange"
            push = source / "primitives/push"
            unrelated = source / "primitives/grasp"
            runtime = source / "primitive_core"
            for directory in (strategy, push, unrelated, runtime):
                directory.mkdir(parents=True)
            (strategy / "program.py").write_text(
                "PHASE_PROGRAM = StrategyProgram(phases=("
                "PrimitivePhase(name='move', primitive='push', role_map={}),))\n",
                encoding="utf-8",
            )
            (strategy / "binding_contract.py").write_text(
                "CONTRACT_VERSION = 1\n",
                encoding="utf-8",
            )
            (push / "nominal.py").write_text("GAIN = 1.0\n", encoding="utf-8")
            (push / "binding_contract.py").write_text(
                "CONTRACT_VERSION = 1\n",
                encoding="utf-8",
            )
            (unrelated / "nominal.py").write_text(
                "GAIN = 1.0\n",
                encoding="utf-8",
            )
            (runtime / "transit.py").write_text(
                "RUNTIME_VERSION = 1\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"RAPID_ACTIVE_SRC_DIR": str(source)},
                clear=False,
            ):
                baseline, dependencies = _program_design_fingerprint(report)
                (unrelated / "nominal.py").write_text(
                    "GAIN = 2.0\n",
                    encoding="utf-8",
                )
                unrelated_change, _ = _program_design_fingerprint(report)
                (push / "nominal.py").write_text(
                    "# formatting and comments do not change design\nGAIN=1.0\n",
                    encoding="utf-8",
                )
                comment_change, _ = _program_design_fingerprint(report)
                (push / "nominal.py").write_text(
                    "GAIN = 2.0\n",
                    encoding="utf-8",
                )
                primitive_change, _ = _program_design_fingerprint(report)
                (runtime / "transit.py").write_text(
                    "RUNTIME_VERSION = 2\n",
                    encoding="utf-8",
                )
                runtime_change, _ = _program_design_fingerprint(report)

        self.assertEqual(dependencies, ("push",))
        self.assertEqual(unrelated_change, baseline)
        self.assertEqual(comment_change, baseline)
        self.assertNotEqual(primitive_change, baseline)
        self.assertNotEqual(runtime_change, primitive_change)

    def test_execution_fingerprint_tracks_binding_and_runtime_arguments(
        self,
    ) -> None:
        report = BindingReport(
            program_kind="strategy",
            program_name="arrange",
            instance_name="example_0",
            scene_module="tasks.example.scene",
            binding_valid=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            binding_path = (
                source
                / "instantiations/strategies/arrange/example_0/binding.json"
            )
            binding_path.parent.mkdir(parents=True)
            binding_path.write_text(
                json.dumps({"roles": {"target": "box"}, "parameters": {}}),
                encoding="utf-8",
            )
            first_args = make_parser("test").parse_args(["--seed", "7"])
            second_args = make_parser("test").parse_args(["--seed", "8"])
            budget = {"effective_num_rollouts": 32, "effective_max_iters": 6}
            with mock.patch.dict(
                os.environ,
                {"RAPID_ACTIVE_SRC_DIR": str(source)},
                clear=False,
            ):
                baseline = _program_execution_fingerprint(
                    first_args,
                    report,
                    budget,
                )
                seed_change = _program_execution_fingerprint(
                    second_args,
                    report,
                    budget,
                )
                binding_path.write_text(
                    json.dumps(
                        {"roles": {"target": "other_box"}, "parameters": {}}
                    ),
                    encoding="utf-8",
                )
                binding_change = _program_execution_fingerprint(
                    first_args,
                    report,
                    budget,
                )

        self.assertIsNotNone(baseline)
        self.assertNotEqual(seed_change, baseline)
        self.assertNotEqual(binding_change, baseline)

    def test_invention_attempt_log_tracks_tiers_and_rejects_same_tier_repeats(
        self,
    ) -> None:
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={
                "success": False,
                "optimized_success": False,
                "optimized_cost": 1.0,
            },
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        spec = InstantiationSpec(
            label="stagnant test",
            compile_fn=lambda _config: result,
            playback=PlaybackBundle(
                browser=mock.Mock(),
                mujoco=mock.Mock(),
                record=mock.Mock(),
            ),
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )
        diagnostic_args = make_parser("test").parse_args(
            ["--num-rollouts", "1", "--max-iters", "1"]
        )
        meaningful_args = make_parser("test").parse_args(
            ["--num-rollouts", "32", "--max-iters", "6"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            attempt_log = Path(tmp) / "attempts.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "RAPID_INVENTION_ATTEMPT_LOG": str(attempt_log),
                    "RAPID_OPTIMIZER_NUM_ROLLOUTS": "96",
                    "RAPID_OPTIMIZER_MAX_ITERS": "16",
                },
                clear=False,
            ):
                run_instantiation(spec, diagnostic_args)
                for _ in range(3):
                    run_instantiation(spec, meaningful_args)
                with mock.patch(
                    "instantiation_core.runner._program_design_fingerprint",
                    return_value=("changed-source", ()),
                ):
                    run_instantiation(spec, meaningful_args)
            records = [
                json.loads(line)
                for line in attempt_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 5)
        self.assertFalse(records[0]["stagnation_eligible"])
        self.assertEqual(records[0]["budget_tier"], "diagnostic")
        self.assertEqual(records[0]["search_decision"], "diagnostic_only")
        self.assertEqual(records[0]["consecutive_count"], 0)
        self.assertTrue(records[-2]["stagnation_eligible"])
        self.assertEqual(records[-2]["budget_tier"], "minimum")
        self.assertEqual(records[-2]["consecutive_count"], 3)
        self.assertEqual(records[-2]["same_tier_count"], 3)
        self.assertTrue(records[-2]["repeat_same_tier_not_useful"])
        self.assertTrue(records[-2]["budget_escalation_available"])
        self.assertEqual(
            records[-2]["search_decision"],
            "repeat_same_tier_not_useful",
        )
        self.assertEqual(records[-1]["consecutive_count"], 1)
        self.assertFalse(result.metrics["stagnation_detected"])

    def test_invention_attempt_log_marks_expanded_failure_as_exhausted(
        self,
    ) -> None:
        result = CompileResult(
            trajectory=ToolTrajectory(
                times=np.asarray((0.0, 0.01)),
                positions=np.zeros((2, 1)),
                controls=np.zeros((2, 1)),
            ),
            metrics={
                "success": False,
                "optimized_success": False,
                "optimized_cost": 1.0,
            },
            best_states=np.zeros((2, self.model.nq + self.model.nv)),
            cost_history=np.asarray((1.0,)),
        )
        spec = InstantiationSpec(
            label="expanded search test",
            compile_fn=lambda _config: result,
            playback=PlaybackBundle(
                browser=mock.Mock(),
                mujoco=mock.Mock(),
                record=mock.Mock(),
            ),
            build_scene_data=lambda: (self.model, mujoco.MjData(self.model)),
            apply_initial_pose=lambda data, initial_pose: None,
        )
        args = make_parser("test").parse_args(
            ["--num-rollouts", "96", "--max-iters", "16"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            attempt_log = Path(tmp) / "attempts.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "RAPID_INVENTION_ATTEMPT_LOG": str(attempt_log),
                    "RAPID_OPTIMIZER_NUM_ROLLOUTS": "96",
                    "RAPID_OPTIMIZER_MAX_ITERS": "16",
                },
                clear=False,
            ):
                run_instantiation(spec, args)
            record = json.loads(attempt_log.read_text(encoding="utf-8"))

        self.assertEqual(record["budget_tier"], "expanded")
        self.assertEqual(record["search_decision"], "expanded_search_exhausted")
        self.assertTrue(record["expanded_search_exhausted"])
        self.assertFalse(record["budget_escalation_available"])
        self.assertTrue(result.metrics["stagnation_detected"])

    def test_attempt_tier_uses_the_failing_phase_budget(self) -> None:
        args = make_parser("test").parse_args([])
        metrics = {
            "phase_metrics": {
                "flip": {
                    "success": False,
                    "num_rollouts": 32,
                    "requested_max_iters": 6,
                },
                "push": {
                    "success": True,
                    "num_rollouts": 96,
                    "requested_max_iters": 16,
                },
            }
        }

        budget = _attempt_budget(
            args,
            metrics,
            failure_signature="phase:flip:effect:predicate",
        )

        self.assertEqual(budget["evaluated_phase"], "flip")
        self.assertEqual(budget["effective_num_rollouts"], 32)
        self.assertEqual(budget["effective_max_iters"], 6)
        self.assertEqual(_search_budget_tier(budget), "minimum")


if __name__ == "__main__":
    unittest.main()
