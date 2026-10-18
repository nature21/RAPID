from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent import invention_session
from agent.codex_exec import CodexExecConfig
from agent.primitive_context import build_primitive_run_prompt
from agent.strategy_context import build_strategy_run_prompt
from primitive_core.common import GLOBAL_MAX_OPTIMIZATION_BUDGET


class PrimitiveContextTests(unittest.TestCase):
    def _primitive_prompt(self) -> str:
        workspace = Path("/workspace")
        with mock.patch("agent.invention_context.WORKSPACE", workspace):
            return build_primitive_run_prompt(
                demo_frames=[workspace / "agent_work/demo/frame.png"],
                demo_sources=[workspace / "agent_work/demo/source.mp4"],
                primitive_name="push",
                num_variants=3,
                variant_seed=19,
                max_iters=10,
                env_modules=["tasks.example.scene"],
                language_description="Push the target.",
            )

    def test_primitive_prompt_contains_run_facts_without_repeating_skill(self) -> None:
        prompt = self._primitive_prompt()

        self.assertIn("Deterministic generalization seed: 19", prompt)
        self.assertIn("Primitive name: `push`", prompt)
        self.assertIn("Per-attempt optimizer iteration ceiling: 10", prompt)
        self.assertIn("tasks.example.scene", prompt)
        self.assertIn("Push the target.", prompt)
        self.assertIn("Infer the goal `G`", prompt)
        self.assertIn("qualitative semantic guidance", prompt)
        self.assertIn("intrinsic effect-goal parameter interface", prompt)
        self.assertIn("VARIANT_SPEC.goal_parameters", prompt)
        self.assertIn("before the first", prompt)
        self.assertIn("simulator run", prompt)
        self.assertIn(
            "Search difficulty or budget exhaustion does not justify",
            prompt,
        )
        self.assertIn("lowering a threshold or deleting a conjunct", prompt)
        self.assertIn("counterexample protocol", prompt)
        self.assertIn("`example_0` through `example_3`", prompt)
        self.assertIn("same Codex run", prompt)
        self.assertIn("1x1 diagnostic runs never", prompt)
        self.assertNotIn("## Required invention criteria", prompt)
        self.assertNotIn("GripperIntent", prompt)
        self.assertNotIn("yaw_coupled_bodies", prompt)

    def test_strategy_prompt_contains_run_facts_without_repeating_skill(self) -> None:
        workspace = Path("/workspace")
        with mock.patch("agent.invention_context.WORKSPACE", workspace):
            prompt = build_strategy_run_prompt(
                demo_frames=[workspace / "agent_work/demo/frame.png"],
                demo_sources=[workspace / "agent_work/demo/source.mp4"],
                strategy_name="strategy_2",
                available_primitives=["push", "press_move"],
                existing_strategies=[],
                max_iters=10,
                num_variants=2,
                variant_seed=19,
                library_mode="open",
                env_modules=["tasks.example.scene"],
                language_description="Push toward the wall, then grasp.",
            )

        self.assertIn("Strategy name: `strategy_2`", prompt)
        self.assertIn("Generate 2 geometry/pose variants", prompt)
        self.assertIn("Per-attempt optimizer iteration ceiling: 10", prompt)
        self.assertIn("existing learned primitives are editable", prompt)
        self.assertIn("continue the strategy run without waiting for approval", prompt)
        self.assertIn("`push`", prompt)
        self.assertIn("Push toward the wall, then grasp.", prompt)
        self.assertIn("`example_0` through `example_2`", prompt)
        self.assertIn("role-only strategy contract", prompt)
        self.assertIn("execute_sequential_strategy", prompt)
        self.assertIn(
            "Search difficulty or budget exhaustion does not justify",
            prompt,
        )
        self.assertIn("lowering a threshold or deleting a conjunct", prompt)
        self.assertIn("counterexample protocol", prompt)
        self.assertNotIn("## Required invention criteria", prompt)
        self.assertNotIn("GripperIntent", prompt)

    def test_invention_session_uses_one_codex_phase(self) -> None:
        with TemporaryDirectory() as tmp:
            session = invention_session.InventionSession(
                kind="primitive",
                run_dir=Path(tmp),
                config=CodexExecConfig("model", "high", 400),
                iteration_ceiling=16,
            )
            with (
                mock.patch.object(
                    invention_session._ArtifactGuard,
                    "capture",
                    return_value=mock.Mock(),
                ) as capture,
                mock.patch.object(
                    invention_session,
                    "run_codex_exec",
                    return_value="thread",
                ) as run,
            ):
                findings = session.run("prompt", [Path("frame.png")])

        self.assertEqual(findings, capture.return_value.check.return_value)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], "prompt")
        self.assertEqual(run.call_args.args[1].max_turns, 400)
        self.assertEqual(
            run.call_args.args[1].environment["RAPID_OPTIMIZER_NUM_ROLLOUTS"],
            str(GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts),
        )
        self.assertEqual(
            run.call_args.args[1].environment["RAPID_OPTIMIZER_MAX_ITERS"],
            str(GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters),
        )
        self.assertEqual(
            run.call_args.args[1].output_last_message.name,
            "codex_final.md",
        )
        self.assertNotIn("resume_thread_id", run.call_args.kwargs)
        capture.assert_called_once()
        capture.return_value.check.assert_called_once_with("primitive", fatal=False)

    def test_invention_session_rejects_iteration_budget_above_global_max(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "GLOBAL_MAX"):
                invention_session.InventionSession(
                    kind="primitive",
                    run_dir=Path(tmp),
                    config=CodexExecConfig("model", "high", 400),
                    iteration_ceiling=(
                        GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters + 1
                    ),
                )

    def test_artifact_guard_detects_only_captured_path_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "protected.py"
            protected.write_text("before", encoding="utf-8")
            guard = invention_session._ArtifactGuard.capture(
                "test guard", {"protected": protected}
            )
            (root / "new.py").write_text("allowed", encoding="utf-8")
            guard.check("primitive")
            protected.write_text("after", encoding="utf-8")
            with self.assertRaises(SystemExit):
                guard.check("primitive")
            guard.check("primitive", fatal=False)

    def test_invention_session_preserves_stricter_external_search_caps(self) -> None:
        with TemporaryDirectory() as tmp:
            session = invention_session.InventionSession(
                kind="primitive",
                run_dir=Path(tmp),
                config=CodexExecConfig("model", "high", 400),
                iteration_ceiling=16,
            )
            with mock.patch.dict(
                invention_session.os.environ,
                {
                    "RAPID_OPTIMIZER_NUM_ROLLOUTS": "48",
                    "RAPID_OPTIMIZER_MAX_ITERS": "8",
                },
                clear=False,
            ):
                config = session._config([])

        self.assertEqual(config.environment["RAPID_OPTIMIZER_NUM_ROLLOUTS"], "48")
        self.assertEqual(config.environment["RAPID_OPTIMIZER_MAX_ITERS"], "8")


if __name__ == "__main__":
    unittest.main()
