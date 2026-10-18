from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent import instantiate_common, sandbox_validation, validation_worker
from agent.instantiate_common import build_parser
from agent.instantiation_context import (
    CANONICAL_MAIN,
    build_instantiation_prompt,
    prepare_scene_evidence,
    validate_package_name,
)
from agent import instantiation_validation


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class InstantiationHelperTests(unittest.TestCase):
    def test_helpers_require_goal_and_kind_specific_program(self) -> None:
        parser = build_parser("strategy")
        args = parser.parse_args(
            [
                "--strategy",
                "strategy_6",
                "--env-module",
                "tasks.example.scene",
                "--name",
                "renamed_scene",
                "--goal",
                "Store the target.",
            ]
        )
        self.assertEqual(args.program, "strategy_6")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--strategy",
                    "strategy_6",
                    "--env-module",
                    "tasks.example.scene",
                    "--name",
                    "renamed_scene",
                ]
            )

    def test_package_names_are_restricted(self) -> None:
        self.assertEqual(
            validate_package_name("scene_2", label="instance"),
            "scene_2",
        )
        with self.assertRaises(ValueError):
            validate_package_name("../scene", label="instance")

    def test_prompt_contains_goal_evidence_output_and_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            inventory = workspace / "agent_work/run/inventory.json"
            images = [
                workspace / f"agent_work/run/view_{index}.png"
                for index in range(3)
            ]
            with (
                mock.patch(
                    "agent.instantiation_context.WORKSPACE",
                    workspace,
                ),
                mock.patch(
                    "agent.instantiation_context.STRATEGIES_DIR",
                    workspace / "src/strategies",
                ),
                mock.patch(
                    "agent.instantiation_context.INSTANTIATIONS_DIR",
                    workspace / "src/instantiations",
                ),
            ):
                prompt = build_instantiation_prompt(
                    kind="strategy",
                    program="strategy_6",
                    instance="renamed",
                    scene_module="tasks.example.renamed.scene",
                    goal="Store the renamed target.",
                    inventory=inventory,
                    images=images,
                    run_dir=workspace / "agent_work/run",
                )

        self.assertIn("Store the renamed target.", prompt)
        self.assertIn("inventory.json", prompt)
        self.assertIn("initial scene views", prompt.lower())
        self.assertIn(
            "instantiations.strategies.strategy_6.renamed.main",
            prompt,
        )
        self.assertIn("Do not edit the selected learned program", prompt)
        self.assertIn("later invention run", prompt)
        self.assertIn(
            "do not calibrate or retain a scene-specific optimization",
            prompt,
        )
        self.assertIn(
            "regardless of whether",
            prompt,
        )
        self.assertIn("strategy contract and manifest are role-only", prompt.lower())
        self.assertIn("binding inputs to an empty mapping", prompt.lower())
        self.assertIn(
            "do not create a runnable partial package",
            prompt.lower(),
        )
        self.assertIn("agent_work/run/smoke/metrics.json", prompt)
        self.assertIn("agent_work/run/smoke/trajectory.mp4", prompt)
        self.assertIn("Never write smoke artifacts under `results/`", prompt)
        self.assertIn("Keep both files unchanged", prompt)
        self.assertIn("cartesian_goals.json", prompt)
        self.assertIn(
            "results_instantiation/strategies/strategy_6/renamed/",
            prompt,
        )
        self.assertNotIn(
            "--metrics-json results/strategy_6_renamed.json",
            prompt,
        )

    def test_scene_evidence_renders_three_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            module = SimpleNamespace(
                DEFAULT_CAMERA={"camera_azimuth": 90.0},
            )

            def fake_render(
                _scene: str,
                output: Path,
                *,
                azimuth: float,
            ) -> Path:
                _touch(output, str(azimuth))
                return output

            with (
                mock.patch(
                    "agent.instantiation_context.write_scene_inventory",
                    side_effect=lambda _scene, output: (
                        _touch(output, "{}") or output
                    ),
                ),
                mock.patch(
                    "agent.instantiation_context.load_scene_module",
                    return_value=module,
                ),
                mock.patch(
                    "agent.instantiation_context.render_scene",
                    side_effect=fake_render,
                ) as render,
            ):
                inventory, images = prepare_scene_evidence(
                    "tasks.example.scene",
                    run_dir,
                )

        self.assertEqual(inventory.name, "inventory.json")
        self.assertEqual(len(images), 3)
        self.assertEqual(
            [call.kwargs["azimuth"] for call in render.call_args_list],
            [30.0, 90.0, 150.0],
        )

    def test_helper_attaches_all_three_previews_to_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "primitives/push"
            _touch(source / "binding_contract.py")
            run_root = root / "agent_work"
            inventory = root / "inventory.json"
            _touch(inventory, "{}")
            destination = root / "existing_instantiation"
            destination.mkdir()
            images = [root / f"view_{index}.png" for index in range(3)]
            for image in images:
                _touch(image)
            argv = [
                "--primitive",
                "push",
                "--env-module",
                "tasks.example.scene",
                "--name",
                "renamed",
                "--goal",
                "Push the target.",
                "--no-stage-guidance",
            ]
            with (
                mock.patch.object(
                    instantiate_common,
                    "AGENT_WORK_DIR",
                    run_root,
                ),
                mock.patch.object(
                    instantiate_common,
                    "program_dir",
                    return_value=source,
                ),
                mock.patch.object(
                    instantiate_common,
                    "instantiation_dir",
                    return_value=destination,
                ),
                mock.patch.object(
                    instantiate_common,
                    "report_program_contract_validation",
                ),
                mock.patch.object(
                    instantiate_common,
                    "normalize_env_modules",
                    return_value=["tasks.example.scene"],
                ),
                mock.patch.object(
                    instantiate_common,
                    "validate_env_modules_importable",
                    return_value=[],
                ),
                mock.patch.object(
                    instantiate_common,
                    "sandboxed_instantiation_findings",
                    return_value=[],
                ),
                mock.patch.object(
                    instantiate_common,
                    "prepare_scene_evidence",
                    return_value=(inventory, images),
                ),
                mock.patch.object(
                    instantiate_common,
                    "build_instantiation_prompt",
                    return_value="bind",
                ) as prompt_builder,
                mock.patch.object(
                    instantiate_common,
                    "learned_source_digests",
                    return_value={"primitives": "a", "strategies": "b"},
                ),
                mock.patch.object(
                    instantiate_common,
                    "report_instantiation_validation",
                ) as validation,
                mock.patch.object(
                    instantiate_common,
                    "run_codex_exec",
                ) as codex,
                mock.patch.object(
                    instantiate_common,
                    "stage_canonical_instantiation",
                ) as stage,
            ):
                instantiate_common.run("primitive", argv)

        config = codex.call_args.args[1]
        self.assertEqual(list(config.local_images), images)
        smoke_run_dir = prompt_builder.call_args.kwargs["run_dir"]
        self.assertEqual(smoke_run_dir.parent, run_root)
        self.assertTrue(smoke_run_dir.name.startswith("instantiate_primitive_exec_"))
        self.assertFalse(validation.call_args.kwargs["fatal"])
        stage.assert_not_called()

    def test_instantiation_main_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            main_path = Path(tmp) / "main.py"
            _touch(main_path, CANONICAL_MAIN)
            self.assertEqual(
                instantiation_validation.validate_instantiation_main(main_path),
                [],
            )

            _touch(main_path, CANONICAL_MAIN + "\nCHANGED = True\n")
            findings = instantiation_validation.validate_instantiation_main(
                main_path
            )

        self.assertTrue(any("noncanonical" in item for item in findings))

    def test_best_smoke_result_is_retained_in_instantiation_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "agent_work/run"
            smoke = run_dir / "smoke"
            _touch(
                smoke / "metrics.json",
                json.dumps({"metrics": {"success": False}}),
            )
            _touch(smoke / "trajectory.mp4", "failed")
            _touch(
                smoke / "metrics_retry_1.json",
                json.dumps({"metrics": {"success": True}}),
            )
            _touch(smoke / "trajectory_retry_1.mp4", "successful")
            _touch(smoke / "cartesian_goals_retry_1.json", "[]")
            with mock.patch.object(instantiate_common, "WORKSPACE", root):
                retained = instantiate_common.preserve_instantiation_results(
                    run_dir,
                    kind="strategy",
                    program="strategy_6",
                    instance="renamed",
                )

            assert retained is not None
            payload = json.loads(
                (retained / "metrics.json").read_text(encoding="utf-8")
            )
            video = (retained / "trajectory.mp4").read_text(encoding="utf-8")
            goals = (retained / "cartesian_goals.json").read_text(encoding="utf-8")

        self.assertTrue(payload["metrics"]["success"])
        self.assertEqual(video, "successful")
        self.assertEqual(goals, "[]")

    def test_learned_source_digest_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primitives = root / "primitives"
            strategies = root / "strategies"
            _touch(primitives / "push/main.py", "before")
            _touch(strategies / "strategy_1/main.py", "same")
            with (
                mock.patch.object(
                    instantiation_validation,
                    "PRIMITIVES_DIR",
                    primitives,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "STRATEGIES_DIR",
                    strategies,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "SRC_DIR",
                    root,
                ),
            ):
                before = instantiation_validation.learned_source_digests()
                _touch(primitives / "push/main.py", "after")
                changed = instantiation_validation.learned_source_changes(before)

        self.assertEqual(changed, ["primitives"])

    def test_program_contract_is_import_checked_before_binding(self) -> None:
        from instantiation_core import BindingContract
        from primitive_core import OptimizationBudget, OptimizationPolicy

        contract = BindingContract(
            program_kind="primitive",
            program_name="push",
            roles=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "push"
            _touch(program / "binding_contract.py")

            def imported(name: str) -> SimpleNamespace:
                if name.endswith(".binding_contract"):
                    return SimpleNamespace(BINDING_CONTRACT=contract)
                symbols = {
                    ".features": {"resolve_geometry": lambda: None},
                    ".nominal": {
                        "make_nominal_plan": lambda: None,
                        "OPTIMIZATION_POLICY": OptimizationPolicy(
                            num_nodes=2,
                            elite_fraction=0.25,
                            horizon=1.0,
                        ),
                        "REFERENCE_OPTIMIZATION_BUDGET": OptimizationBudget(
                            num_rollouts=4,
                            max_iters=1,
                        ),
                    },
                    ".transit": {"default_transit_spec": lambda: None},
                    ".cost": {
                        "effect_predicate": lambda: None,
                        "effect_cost": lambda: None,
                        "EFFECT_TOLERANCES": {"error": 0.1},
                    },
                    ".main": {
                        "compile_primitive": lambda binding, args: None,
                        "make_parser": lambda: None,
                    },
                }
                for suffix, values in symbols.items():
                    if name.endswith(suffix):
                        return SimpleNamespace(**values)
                raise AssertionError(name)

            with (
                mock.patch.object(
                    instantiation_validation,
                    "program_dir",
                    return_value=program,
                ),
                mock.patch.object(
                    instantiation_validation.importlib,
                    "import_module",
                    side_effect=imported,
                ),
            ):
                findings = instantiation_validation.validate_program_contract(
                    "primitive",
                    "push",
                )

        self.assertEqual(findings, [])

    def test_invalid_manifest_fails_before_scene_loading(self) -> None:
        from agent.instantiation_context import CANONICAL_INIT, CANONICAL_MAIN

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "push/renamed"
            _touch(package.parent / "__init__.py")
            _touch(package / "__init__.py", CANONICAL_INIT)
            _touch(package / "main.py", CANONICAL_MAIN)
            _touch(
                package / "binding.json",
                json.dumps(
                    {
                        "schema_version": "rapid.binding/v1",
                        "program_kind": "primitive",
                        "program_name": "push",
                        "instance_name": "renamed",
                        "scene_module": "tasks.example.scene",
                        "goal_text": "Push.",
                        "roles": {},
                        "inputs": {},
                        "waypoints": [[0.0, 1.0]],
                    }
                ),
            )
            with (
                mock.patch.object(
                    instantiation_validation,
                    "instantiation_dir",
                    return_value=package,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
                mock.patch.object(
                    instantiation_validation,
                    "program_dir",
                    return_value=Path(tmp) / "primitives/push",
                ),
                mock.patch.object(
                    instantiation_validation.importlib,
                    "import_module",
                    return_value=SimpleNamespace(
                        BINDING_CONTRACT=__import__(
                            "instantiation_core", fromlist=["BindingContract"]
                        ).BindingContract("primitive", "push", ()),
                    ),
                ),
                mock.patch.object(
                    instantiation_validation,
                    "load_scene_module",
                ) as load_scene,
            ):
                findings = instantiation_validation.validate_instantiation_layout(
                    "primitive",
                    "push",
                    "renamed",
                )

        self.assertTrue(any("unknown keys" in finding for finding in findings))
        load_scene.assert_not_called()

    def test_program_contract_rejects_learned_motion_audit_limit_tuning(self) -> None:
        from instantiation_core import BindingContract

        contract = BindingContract("primitive", "push", ())
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "push"
            _touch(program / "binding_contract.py")
            _touch(
                program / "compile.py",
                "LIMITS = FR3SafetyLimits(nominal_linear_velocity=999.0)\n"
                "SPEC = FR3SafetySpec((), (), limits=LIMITS)\n",
            )

            with (
                mock.patch.object(
                    instantiation_validation,
                    "program_dir",
                    return_value=program,
                ),
                mock.patch.object(
                    instantiation_validation.importlib,
                    "import_module",
                    return_value=SimpleNamespace(BINDING_CONTRACT=contract),
                ),
            ):
                findings = instantiation_validation.validate_program_contract(
                    "primitive",
                    "push",
                )

        self.assertTrue(any("motion/audit limits" in item for item in findings))

    def test_attempt_policy_demotes_only_strategy_transit_contract_finding(
        self,
    ) -> None:
        transit = (
            instantiation_validation.STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX
            + "found ['rollout_fr3_transit'] in strategy.py"
        )
        other = "strategy binding contract has an invalid role"

        blocking, advisories = (
            instantiation_validation.partition_program_contract_findings(
                "strategy",
                [transit, other],
                execution_policy="attempt",
            )
        )

        self.assertEqual(blocking, [other])
        self.assertEqual(advisories, [transit])

    def test_strict_policy_keeps_strategy_transit_contract_finding_fatal(
        self,
    ) -> None:
        transit = (
            instantiation_validation.STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX
            + "found ['rollout_fr3_transit'] in strategy.py"
        )

        blocking, advisories = (
            instantiation_validation.partition_program_contract_findings(
                "strategy",
                [transit],
                execution_policy="strict",
            )
        )

        self.assertEqual(blocking, [transit])
        self.assertEqual(advisories, [])

    def test_destination_collision_stops_before_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "primitives/push"
            destination = root / "instantiations/primitives/push/example_0"
            _touch(source / "binding_contract.py")
            destination.mkdir(parents=True)
            argv = [
                "--primitive",
                "push",
                "--env-module",
                "tasks.example.scene",
                "--name",
                "example_0",
                "--goal",
                "Push the target.",
            ]
            with (
                mock.patch.object(
                    instantiate_common,
                    "program_dir",
                    return_value=source,
                ),
                mock.patch.object(
                    instantiate_common,
                    "instantiation_dir",
                    return_value=destination,
                ),
                mock.patch.object(
                    instantiate_common,
                    "report_program_contract_validation",
                ),
                mock.patch.object(
                    instantiate_common,
                    "normalize_env_modules",
                    return_value=["tasks.example.scene"],
                ),
                mock.patch.object(
                    instantiate_common,
                    "validate_env_modules_importable",
                    return_value=[],
                ),
                mock.patch.object(
                    instantiate_common,
                    "sandboxed_instantiation_findings",
                    return_value=["existing binding is invalid"],
                ),
                mock.patch.object(
                    instantiate_common,
                    "run_codex_exec",
                ) as codex,
            ):
                with self.assertRaises(SystemExit) as caught:
                    instantiate_common.run("primitive", argv)

        self.assertEqual(caught.exception.code, 2)
        codex.assert_not_called()

    def test_host_validation_delegates_generated_program_to_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (
                mock.patch.object(
                    instantiation_validation,
                    "validate_instantiation_layout",
                    side_effect=AssertionError("must not run in host process"),
                ),
                mock.patch.object(
                    instantiation_validation,
                    "record_learned_source_digests",
                    return_value=[],
                ),
                mock.patch.object(
                    instantiation_validation,
                    "run_sandboxed_validation",
                    return_value=[],
                ) as sandboxed,
            ):
                instantiation_validation.report_instantiation_validation(
                    "primitive",
                    "push",
                    "renamed",
                    learned_before={"primitives": "a", "strategies": "b"},
                    run_dir=run_dir,
                    scene_module="tasks.example.scene",
                    goal="Push the target.",
                )

        sandboxed.assert_called_once_with(
            {
                "check": "instantiation_layout",
                "kind": "primitive",
                "program": "push",
                "instance": "renamed",
                "expected_scene_module": "tasks.example.scene",
                "expected_goal": "Push the target.",
            }
        )

    def test_instantiation_report_can_return_advisory_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(
                    instantiation_validation,
                    "run_sandboxed_validation",
                    return_value=["binding evidence disagreement"],
                ),
                mock.patch.object(
                    instantiation_validation,
                    "record_learned_source_digests",
                    return_value=["primitives"],
                ),
            ):
                findings = (
                    instantiation_validation.report_instantiation_validation(
                        "primitive",
                        "push",
                        "renamed",
                        learned_before={"primitives": "a"},
                        run_dir=Path(tmp),
                        scene_module="tasks.example.scene",
                        goal="Push the target.",
                        fatal=False,
                    )
                )

        self.assertEqual(
            findings,
            [
                "binding evidence disagreement",
                "instantiation modified learned/runtime source tree(s): primitives",
            ],
        )

    def test_sandbox_validation_command_uses_safe_workspace_imports(self) -> None:
        workspace = Path("/isolated/workspace")
        repo = Path("/repo")
        request = {
            "check": "instantiation_layout",
            "kind": "strategy",
            "program": "strategy_1",
            "instance": "example_0",
        }
        with (
            mock.patch.object(sandbox_validation, "WORKSPACE", workspace),
            mock.patch.object(sandbox_validation, "REPO_ROOT", repo),
        ):
            command = sandbox_validation._sandbox_validation_command(
                request,
                codex="/tools/codex",
            )

        self.assertEqual(
            command[:8],
            [
                "/tools/codex",
                "sandbox",
                "-C",
                str(workspace),
                "-P",
                sandbox_validation.PERMISSION_PROFILE,
                "--",
                str(repo / "scripts/rapid-python"),
            ],
        )
        self.assertEqual(command[8:11], ["-P", "-m", "agent.validation_worker"])
        self.assertEqual(json.loads(command[11]), request)

    def test_sandbox_validation_fails_closed_without_launcher_profile(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sandbox_validation.subprocess, "run") as run,
        ):
            findings = sandbox_validation.run_sandboxed_validation(
                {"check": "program_contract"}
            )

        self.assertTrue(any("refusing" in finding for finding in findings))
        run.assert_not_called()

    def test_sandbox_validation_rehomes_cache_directories(self) -> None:
        workspace = Path("/isolated/workspace")
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                sandbox_validation.RESULT_PREFIX
                + json.dumps({"findings": []})
                + "\n"
            ),
            stderr="",
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "RAPID_CODEX_PERMISSION_PROFILE": "1",
                    "TMPDIR": "/tmp",
                    "XDG_CACHE_HOME": "/tmp",
                    "MPLCONFIGDIR": "/tmp",
                },
                clear=True,
            ),
            mock.patch.object(sandbox_validation, "WORKSPACE", workspace),
            mock.patch.object(
                sandbox_validation,
                "_sandbox_validation_command",
                return_value=["codex", "sandbox"],
            ),
            mock.patch.object(
                sandbox_validation.subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            findings = sandbox_validation.run_sandboxed_validation(
                {"check": "program_contract"}
            )

        self.assertEqual(findings, [])
        child_env = run.call_args.kwargs["env"]
        cache_dir = workspace / "agent_work/tmp/rapid_cache"
        self.assertEqual(child_env["TMPDIR"], str(workspace / "agent_work/tmp"))
        self.assertEqual(child_env["XDG_CACHE_HOME"], str(cache_dir))
        self.assertEqual(child_env["MPLCONFIGDIR"], str(cache_dir / "matplotlib"))

    def test_validation_worker_dispatches_instantiation_check(self) -> None:
        with mock.patch.object(
            instantiation_validation,
            "validate_instantiation_layout",
            return_value=["finding"],
        ) as validate:
            findings = validation_worker.dispatch_validation(
                {
                    "check": "instantiation_layout",
                    "kind": "strategy",
                    "program": "strategy_1",
                    "instance": "example_0",
                    "expected_scene_module": "tasks.example.scene",
                    "expected_goal": "Store the target.",
                }
            )

        self.assertEqual(findings, ["finding"])
        validate.assert_called_once_with(
            "strategy",
            "strategy_1",
            "example_0",
            expected_scene_module="tasks.example.scene",
            expected_goal="Store the target.",
            execution_policy="strict",
        )

    def test_validation_worker_forwards_attempt_execution_policy(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"RAPID_EXECUTION_POLICY": "attempt"},
            ),
            mock.patch.object(
                instantiation_validation,
                "validate_instantiation_layout",
                return_value=[],
            ) as validate,
        ):
            validation_worker.dispatch_validation(
                {
                    "check": "instantiation_layout",
                    "kind": "strategy",
                    "program": "strategy_1",
                    "instance": "example_0",
                }
            )

        self.assertEqual(validate.call_args.kwargs["execution_policy"], "attempt")

    def test_validation_worker_dispatches_generalization_check(self) -> None:
        with mock.patch.object(
            instantiation_validation,
            "validate_generated_generalization_layout",
            return_value=["finding"],
        ) as validate:
            findings = validation_worker.dispatch_validation(
                {
                    "check": "generalization_layout",
                    "kind": "primitive",
                    "program": "push",
                    "num_variants": 3,
                    "require_minimum": False,
                }
            )

        self.assertEqual(findings, ["finding"])
        validate.assert_called_once_with(
            "primitive",
            "push",
            3,
            require_minimum=False,
        )

    def test_generalization_report_delegates_all_examples_to_sandbox(self) -> None:
        with mock.patch.object(
            instantiation_validation,
            "run_sandboxed_validation",
            return_value=[],
        ) as sandboxed:
            instantiation_validation.report_generated_generalization_layout(
                "primitive",
                "push",
                3,
            )

        sandboxed.assert_called_once_with(
            {
                "check": "generalization_layout",
                "kind": "primitive",
                "program": "push",
                "num_variants": 3,
            }
        )

    def test_generalization_report_can_return_advisory_findings(self) -> None:
        with mock.patch.object(
            instantiation_validation,
            "run_sandboxed_validation",
            return_value=["trajectory metric disagreement"],
        ):
            findings = (
                instantiation_validation.report_generated_generalization_layout(
                    "primitive",
                    "push",
                    3,
                    fatal=False,
                )
            )

        self.assertEqual(findings, ["trajectory metric disagreement"])

    def test_legacy_callback_adapter_fails_before_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "strategy_1/renamed"
            _touch(package.parent / "__init__.py")
            _touch(package / "__init__.py")
            _touch(package / "binding.json", "{}")
            _touch(package / "binding.py", "SUCCESS_PREDICATE = lambda: True\n")
            _touch(package / "mapping.py", "WAYPOINTS = [[0, 1]]\n")
            _touch(package / "main.py", "def main():\n    pass\n")
            for name in (".agents", ".codex", ".git"):
                (package / name).mkdir()

            with (
                mock.patch.object(
                    instantiation_validation,
                    "instantiation_dir",
                    return_value=package,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = (
                    instantiation_validation.validate_instantiation_layout(
                        "strategy",
                        "strategy_1",
                        "renamed",
                    )
                )

        self.assertTrue(
            any("manifest-only" in finding for finding in findings)
        )
        self.assertFalse(
            any(
                metadata in finding
                for finding in findings
                for metadata in (".agents", ".codex", ".git")
            )
        )

    def test_digest_record_contains_before_after_and_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primitives = root / "primitives"
            strategies = root / "strategies"
            record = root / "record.json"
            _touch(primitives / "push/main.py", "before")
            _touch(strategies / "strategy_1/main.py", "same")
            with (
                mock.patch.object(
                    instantiation_validation,
                    "PRIMITIVES_DIR",
                    primitives,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "STRATEGIES_DIR",
                    strategies,
                ),
                mock.patch.object(
                    instantiation_validation,
                    "SRC_DIR",
                    root,
                ),
            ):
                before = instantiation_validation.learned_source_digests()
                _touch(primitives / "push/main.py", "after")
                changed = (
                    instantiation_validation.record_learned_source_digests(
                        record,
                        before,
                    )
                )

            payload = json.loads(record.read_text(encoding="utf-8"))

        self.assertEqual(changed, ["primitives"])
        self.assertEqual(payload["changed"], ["primitives"])
        self.assertEqual(
            set(payload["before"]),
            {
                "instantiation_core",
                "primitive_core",
                "primitives",
                "strategies",
            },
        )

if __name__ == "__main__":
    unittest.main()
