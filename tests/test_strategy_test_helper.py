from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from suites.staged import strategy_test


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_target(root: Path, *, contract: bool = True) -> tuple[Path, Path]:
    repo = root / "repo"
    workspace = repo / "suite_runs/deploy_001/workspace"
    strategy = workspace / "src/strategies/strategy_1"
    _touch(strategy / "__init__.py")
    _touch(strategy / "main.py")
    _touch(strategy / "program.py")
    _touch(strategy / "predicate.py")
    if contract:
        _touch(strategy / "binding_contract.py")
    _touch(repo / "tasks/example/new_scene/scene.py")
    _touch(repo / "scripts/rapid-codex-run")
    _touch(repo / "scripts/rapid-python")
    return repo, workspace


class StrategyTestHelperTests(unittest.TestCase):
    def test_resolves_short_run_task_path_and_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = _make_target(Path(tmp))

            target = strategy_test.resolve_target(
                repo,
                suite_run="deploy_001",
                test_case="tasks/example/new_scene",
                strategy=None,
                instance=None,
            )

        self.assertEqual(target.workspace, workspace)
        self.assertEqual(target.strategy, "strategy_1")
        self.assertEqual(target.scene_module, "tasks.example.new_scene.scene")
        self.assertEqual(target.instance, "new_scene")
        self.assertEqual(
            target.module,
            "instantiations.strategies.strategy_1.new_scene.main",
        )

    def test_rejects_legacy_strategy_without_binding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = _make_target(Path(tmp), contract=False)

            with self.assertRaisesRegex(
                ValueError,
                "not declarative-binding compatible",
            ):
                strategy_test.resolve_strategy(
                    workspace / "src",
                    "strategy_1",
                )

    def test_rejects_result_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            with self.assertRaisesRegex(ValueError, "must stay under"):
                strategy_test._workspace_output(
                    workspace,
                    Path(tmp).parent / "outside.json",
                    "unused.json",
                )

    def test_runs_binding_then_strategy_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = _make_target(Path(tmp))
            args = strategy_test.build_parser().parse_args(
                [
                    "deploy_001",
                    "tasks.example.new_scene",
                    "Store the target in the new receptacle.",
                    "--seed",
                    "7",
                ]
            )
            calls: list[tuple[list[str], Path, dict[str, str]]] = []

            def fake_run(
                command: list[str],
                *,
                cwd: Path,
                env: dict[str, str],
            ) -> int:
                calls.append((list(command), cwd, env))
                if "rapid-python" in Path(command[0]).name:
                    metrics = Path(command[command.index("--metrics-json") + 1])
                    metrics.parent.mkdir(parents=True, exist_ok=True)
                    metrics.write_text(
                        json.dumps(
                            {
                                "binding_valid": True,
                                "program_success": True,
                            }
                        ),
                        encoding="utf-8",
                    )
                return 0

            with mock.patch.object(
                strategy_test,
                "_run",
                side_effect=fake_run,
            ):
                returncode = strategy_test.run_strategy_test(
                    args,
                    repo_root=repo,
                )

            metrics = (
                workspace
                / "results_instantiation/strategies/strategy_1/new_scene/metrics.json"
            )
            recording = (
                workspace
                / "results_instantiation/strategies/strategy_1/new_scene/trajectory.mp4"
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(len(calls), 2)
        bind_command, bind_cwd, _ = calls[0]
        self.assertEqual(bind_cwd, repo)
        self.assertIn("agent/deploy_strategy_exec.py", bind_command)
        self.assertIn("--bind-only", bind_command)
        self.assertIn("tasks.example.new_scene.scene", bind_command)
        self.assertIn("Store the target in the new receptacle.", bind_command)
        self.assertNotIn("--smoke-timeout", bind_command)
        test_command, test_cwd, test_env = calls[1]
        self.assertEqual(test_cwd, workspace)
        self.assertIn(
            "instantiations.strategies.strategy_1.new_scene.main",
            test_command,
        )
        self.assertIn(str(metrics), test_command)
        self.assertIn(str(recording), test_command)
        self.assertEqual(test_env["RAPID_WORKSPACE"], str(workspace))
        self.assertEqual(test_env["RAPID_ACTIVE_SRC_DIR"], str(workspace / "src"))

    def test_program_failure_controls_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, _workspace = _make_target(Path(tmp))

            def fake_run(
                command: list[str],
                *,
                cwd: Path,
                env: dict[str, str],
            ) -> int:
                del cwd, env
                if "rapid-python" in Path(command[0]).name:
                    metrics = Path(command[command.index("--metrics-json") + 1])
                    metrics.parent.mkdir(parents=True, exist_ok=True)
                    metrics.write_text(
                        json.dumps(
                            {
                                "binding_valid": True,
                                "program_success": False,
                            }
                        ),
                        encoding="utf-8",
                    )
                return 0

            parser = strategy_test.build_parser()
            strict = parser.parse_args(
                [
                    "deploy_001",
                    "tasks.example.new_scene",
                    "Store the target.",
                    "--no-record",
                ]
            )
            permissive = parser.parse_args(
                [
                    "deploy_001",
                    "tasks.example.new_scene",
                    "Store the target.",
                    "--no-record",
                    "--allow-program-failure",
                ]
            )
            with mock.patch.object(
                strategy_test,
                "_run",
                side_effect=fake_run,
            ):
                strict_code = strategy_test.run_strategy_test(
                    strict,
                    repo_root=repo,
                )
                permissive_code = strategy_test.run_strategy_test(
                    permissive,
                    repo_root=repo,
                )

        self.assertEqual(strict_code, 1)
        self.assertEqual(permissive_code, 0)


if __name__ == "__main__":
    unittest.main()
