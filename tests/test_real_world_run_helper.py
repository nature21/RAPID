from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from suites.staged import real_world_run


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(root: Path) -> tuple[Path, Path]:
    repo = root / "repo"
    eval_run = repo / "suite_runs/fr3_osc_demo_1_001_eval"
    workspace = eval_run / "workspace"
    strategy = workspace / "src/strategies/strategy_1"
    for filename in (
        "__init__.py",
        "binding_contract.py",
        "main.py",
        "predicate.py",
        "program.py",
    ):
        _touch(strategy / filename)
    _touch(repo / "tasks/invention_real2sim/scene_3/scene.py")
    _touch(
        workspace
        / "src/instantiations/strategies/strategy_1/example_0/binding.json",
        json.dumps(
            {
                "goal_text": "Store the target in the corridor.",
                "program_name": "strategy_1",
            }
        ),
    )
    _touch(
        eval_run / "benchmark_eval.json",
        json.dumps({"strategy": "strategy_1"}),
    )
    return repo, eval_run


class RealWorldRunHelperTests(unittest.TestCase):
    def test_resolves_short_inputs_and_frozen_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, eval_run = _make_repo(Path(tmp))

            target = real_world_run.resolve_target(
                repo,
                scene_id="scene_3",
                eval_suite="fr3_osc_demo_1_001_eval",
                goal=None,
            )

        self.assertEqual(target.eval_run, eval_run)
        self.assertEqual(target.scene_module, "tasks.invention_real2sim.scene_3.scene")
        self.assertEqual(target.strategy, "strategy_1")
        self.assertEqual(target.goal, "Store the target in the corridor.")
        self.assertEqual(target.run_dir, eval_run / "real_world_runs/scene_3")

    def test_requires_an_eval_suite_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, eval_run = _make_repo(Path(tmp))
            (eval_run / "benchmark_eval.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "eval-suite manifest"):
                real_world_run.resolve_target(
                    repo,
                    scene_id="scene_3",
                    eval_suite="fr3_osc_demo_1_001_eval",
                    goal=None,
                )

    def test_runs_attempt_in_eval_workspace_and_archives_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, eval_run = _make_repo(Path(tmp))
            args = real_world_run.build_parser().parse_args(
                ["scene_3", "fr3_osc_demo_1_001_eval", "--seed", "7"]
            )
            observed: dict[str, object] = {}

            def fake_strategy_test(namespace: object, *, repo_root: Path) -> int:
                observed["namespace"] = namespace
                observed["execution_policy"] = os.environ.get(
                    "RAPID_EXECUTION_POLICY"
                )
                observed["gpu_render"] = os.environ.get("RAPID_GPU_RENDER")
                observed["record_width"] = os.environ.get("RAPID_RECORD_WIDTH")
                workspace = eval_run / "workspace"
                result = (
                    workspace
                    / "results_instantiation/strategies/strategy_1/scene_3"
                )
                _touch(
                    result / "metrics.json",
                    json.dumps(
                        {"binding_valid": True, "program_success": True}
                    ),
                )
                _touch(result / "cartesian_goals.json", '{"schema":"goals"}')
                _touch(result / "trajectory.mp4", "video")
                _touch(
                    workspace
                    / "src/instantiations/strategies/strategy_1/scene_3/binding.json",
                    '{"schema_version":"rapid.binding/v1"}',
                )
                _touch(
                    workspace / "task_inputs/tasks/invention_real2sim/scene_3/scene.py"
                )
                return 0

            with (
                mock.patch.object(
                    real_world_run,
                    "_sync_instantiation_runtime",
                ),
                mock.patch.object(
                    real_world_run,
                    "run_strategy_test",
                    side_effect=fake_strategy_test,
                ),
            ):
                returncode = real_world_run.run_real_world(args, root=repo)

            output = eval_run / "real_world_runs/scene_3"
            record = json.loads(
                (output / "real_world_run.json").read_text(encoding="utf-8")
            )

            self.assertEqual(returncode, 0)
            self.assertEqual(observed["execution_policy"], "attempt")
            self.assertEqual(observed["gpu_render"], "required")
            self.assertEqual(observed["record_width"], "480")
            namespace = observed["namespace"]
            self.assertEqual(namespace.seed, 7)
            self.assertEqual(namespace.suite_run, str(eval_run))
            self.assertTrue(namespace.sandbox_execution)
            self.assertEqual(namespace.optimization_budget_mode, "maximum")
            self.assertEqual(
                namespace.description,
                "Store the target in the corridor.",
            )
            self.assertTrue((output / "trajectory.mp4").is_file())
            self.assertTrue((output / "cartesian_goals.json").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "instantiation/binding.json").is_file())
            self.assertFalse(
                (
                    eval_run
                    / "workspace/src/instantiations/strategies/strategy_1/scene_3"
                ).exists()
            )
            self.assertFalse(
                (
                    eval_run
                    / "workspace/results_instantiation/strategies/strategy_1/scene_3"
                ).exists()
            )
            self.assertFalse(
                (
                    eval_run
                    / "workspace/task_inputs/tasks/invention_real2sim/scene_3"
                ).exists()
            )
            self.assertTrue(record["artifact_ready"])
            self.assertEqual(record["execution_policy"], "attempt")
            self.assertEqual(record["program_success"], True)

    def test_refuses_to_overwrite_a_prior_scene_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, eval_run = _make_repo(Path(tmp))
            (eval_run / "real_world_runs/scene_3").mkdir(parents=True)
            args = real_world_run.build_parser().parse_args(
                ["scene_3", "fr3_osc_demo_1_001_eval"]
            )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                real_world_run.run_real_world(args, root=repo)


if __name__ == "__main__":
    unittest.main()
