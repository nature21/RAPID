from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.invention_status import build_invention_status


class InventionStatusTests(unittest.TestCase):
    def _write_metrics(
        self,
        workspace: Path,
        program: str,
        index: int,
        *,
        success: bool,
        metrics: dict[str, object] | None = None,
    ) -> None:
        path = workspace / "results" / f"{program}_example_{index}_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        retained_metrics: dict[str, object] = {
            "success": success,
            "effect_success": success,
            "mechanism_valid": True,
            "stability_required": False,
            "num_rollouts": 32,
            "requested_max_iters": 6,
            "cost_components": {"total": 1.0},
            "progress_diagnostics": {"final": 1.0},
            "mechanism_diagnostics": {"required_relations_valid": True},
            **(metrics or {}),
        }
        path.write_text(
            json.dumps(
                {
                    "program_success": success,
                    "metrics": retained_metrics,
                }
            ),
            encoding="utf-8",
        )

    def test_behavior_success_and_audit_advisory_are_independent(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_metrics(workspace, "push", 0, success=True)
            self._write_metrics(
                workspace,
                "push",
                1,
                success=True,
                metrics={
                    "whole_arm_audit_valid": False,
                    "audit_advisories": ["whole_arm_clearance_or_contact"],
                },
            )
            status = build_invention_status(
                kind="primitive",
                programs=("push",),
                num_variants=1,
                workflow_completed=True,
                workspace=workspace,
            )

        self.assertTrue(status["workflow_completed"])
        self.assertTrue(status["behavior_success"])
        self.assertEqual(status["behavior_status"], "passed")
        self.assertEqual(status["effect_status"], "passed")
        self.assertEqual(status["mechanism_status"], "passed")
        self.assertEqual(status["audit_status"], "advisory")

    def test_behavior_failure_does_not_make_a_clean_audit_fail(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_metrics(workspace, "push", 0, success=False)
            status = build_invention_status(
                kind="primitive",
                programs=("push",),
                num_variants=0,
                workflow_completed=True,
                workspace=workspace,
            )

        self.assertFalse(status["behavior_success"])
        self.assertEqual(status["behavior_status"], "failed")
        self.assertEqual(status["effect_status"], "failed")
        self.assertEqual(status["mechanism_status"], "passed")
        self.assertEqual(status["audit_status"], "clean")

    def test_effect_and_mechanism_status_are_independent(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_metrics(
                workspace,
                "primitive",
                0,
                success=False,
                metrics={
                    "effect_success": True,
                    "mechanism_valid": False,
                    "mechanism_diagnostics": {
                        "required_relations_valid": False
                    },
                },
            )
            status = build_invention_status(
                kind="primitive",
                programs=("primitive",),
                num_variants=0,
                workflow_completed=True,
                workspace=workspace,
            )

        self.assertEqual(status["behavior_status"], "failed")
        self.assertEqual(status["effect_status"], "passed")
        self.assertEqual(status["mechanism_status"], "failed")

    def test_diagnostic_only_primitive_evidence_is_incomplete(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_metrics(
                workspace,
                "primitive",
                0,
                success=True,
                metrics={"num_rollouts": 1, "requested_max_iters": 1},
            )
            status = build_invention_status(
                kind="primitive",
                programs=("primitive",),
                num_variants=0,
                workflow_completed=True,
                workspace=workspace,
            )

        example = status["examples"]["primitive/example_0"]
        self.assertFalse(example["retained_evidence_valid"])
        self.assertIn(
            "retained primitive evidence is diagnostic-only",
            example["evidence_findings"],
        )
        self.assertEqual(status["behavior_status"], "incomplete")
        self.assertEqual(status["effect_status"], "incomplete")
        self.assertEqual(status["mechanism_status"], "incomplete")

    def test_missing_metrics_are_incomplete_and_unknown(self) -> None:
        with TemporaryDirectory() as tmp:
            status = build_invention_status(
                kind="strategy",
                programs=("strategy_1",),
                num_variants=1,
                workflow_completed=True,
                workspace=Path(tmp),
            )

        self.assertEqual(status["behavior_status"], "incomplete")
        self.assertEqual(status["audit_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
