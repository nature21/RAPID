from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmark_private.specs import BENCHMARK_VERSION
from suites.staged.benchmark_eval import (
    BenchmarkSceneSet,
    _archive_stale_case_results,
    _eval_run_dir,
    _write_aggregate_summary,
    _write_eval_result_summary,
    build_parser,
    recoverable_case_indices,
    resolve_scene_set,
    write_gallery,
)
from suites.staged.benchmark_runner import build_parser as runner_parser
from suites.staged.benchmark_runner import _attach_execution_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkEvalTests(unittest.TestCase):
    def test_eval_manifest_publishes_task_and_binding_success_rates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            eval_run = Path(raw)
            scene_set = BenchmarkSceneSet(
                root=eval_run / "strategy_8",
                strategy_id=8,
                cases={1: eval_run / "case_001", 2: eval_run / "case_002"},
            )
            manifest = {
                "schema_version": 1,
                "source_suite": "source",
                "eval_suite": str(eval_run),
            }
            records = [
                {
                    "case_index": 1,
                    "bucket": "diagnostic",
                    "diagnostic_dimension": "pose",
                    "stress_axis": None,
                    "benchmark_success": True,
                    "program_success": True,
                    "binding_valid": True,
                    "target_bound": True,
                    "authoritative_scene_role_coverage": 1.0,
                },
                {
                    "case_index": 2,
                    "bucket": "mixed",
                    "diagnostic_dimension": None,
                    "stress_axis": None,
                    "benchmark_success": False,
                    "program_success": False,
                    "binding_valid": False,
                    "target_bound": False,
                    "authoritative_scene_role_coverage": 0.5,
                },
            ]

            payload = _write_eval_result_summary(
                eval_run=eval_run,
                eval_manifest=manifest,
                scene_set=scene_set,
                records=records,
            )
            persisted = json.loads(
                (eval_run / "benchmark_eval.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload, persisted)
        self.assertTrue(persisted["evaluation_completed"])
        self.assertEqual(persisted["task_successes"], 1)
        self.assertEqual(persisted["task_success_rate"], 0.5)
        self.assertEqual(persisted["binding_successes"], 1)
        self.assertEqual(persisted["binding_success_rate"], 0.5)

    def test_resume_archives_results_for_replaced_scene_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scene = root / "strategy_3/case_001_scene"
            scene.mkdir(parents=True)
            (scene / "scene.py").write_text("value = 1\n", encoding="utf-8")
            output = root / "output"
            run_dir = output / "case_001"
            run_dir.mkdir(parents=True)
            (run_dir / "benchmark_result.json").write_text(
                json.dumps({"case_sha256": "old-scene-digest"}),
                encoding="utf-8",
            )
            scene_set = BenchmarkSceneSet(
                root=scene.parent,
                strategy_id=3,
                cases={1: scene},
            )

            archived = _archive_stale_case_results(output, scene_set)

            self.assertEqual(len(archived), 1)
            self.assertFalse(run_dir.exists())
            self.assertTrue(
                (
                    output
                    / "_superseded"
                    / "case_001__old-scene-di"
                    / "benchmark_result.json"
                ).is_file()
            )

    def test_preflight_failure_keeps_valid_published_binding(self) -> None:
        record = _attach_execution_diagnostics(
            {
                "binding_written": True,
                "binding_valid": False,
                "program_success": False,
            },
            {},
        )
        self.assertTrue(record["binding_valid"])
        self.assertTrue(record["execution_started"])
        self.assertFalse(record["execution_completed"])
        self.assertEqual(record["failure_stage"], "strategy_preflight")

    def test_attempt_continuation_is_exposed_in_execution_diagnostics(self) -> None:
        continuation = {
            "failed_segment": "push__to__grasp",
            "failure_type": "FR3TransitViolation",
        }
        record = _attach_execution_diagnostics(
            {"binding_written": True, "binding_valid": True},
            {
                "metrics": {
                    "attempt_continuations": [continuation],
                    "strict_would_abort": True,
                }
            },
        )
        self.assertTrue(record["execution_completed"])
        self.assertTrue(record["strict_would_abort"])
        self.assertEqual(record["attempt_continuations"], [continuation])

    def test_binding_coverage_accepts_collision_geom_roles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            binding_path = Path(raw) / "binding.json"
            binding_path.write_text(
                json.dumps(
                    {
                        "roles": {
                            "target": "target_box",
                            "fixture": "wall_collision",
                            "corridor_left_collision": "left_box_collision",
                            "corridor_right_collision": "right_box_collision",
                        }
                    }
                ),
                encoding="utf-8",
            )
            record = _attach_execution_diagnostics(
                {
                    "binding_written": True,
                    "binding_valid": True,
                    "binding_path": str(binding_path),
                    "expected_scene_roles": [
                        "target_box",
                        "wall",
                        "left_box",
                        "right_box",
                    ],
                },
                {},
            )

        self.assertEqual(record["authoritative_scene_role_coverage"], 1.0)
        self.assertTrue(record["target_bound"])

    def test_downstream_aperture_error_is_attributed_to_failed_flip(self) -> None:
        record = _attach_execution_diagnostics(
            {
                "binding_written": True,
                "binding_valid": True,
                "program_success": False,
            },
            {
                "metrics": {
                    "partial_strategy": True,
                    "completed_segments": [
                        "extract_from_corridor",
                        "push_to_fixture",
                        "flip_at_fixture",
                        "flip_at_fixture__to__grasp_and_lift",
                    ],
                    "failed_segment": "grasp_and_lift",
                    "failure_type": "ValueError",
                    "failure_message": (
                        "target narrow half-width exceeds the physical gripper aperture"
                    ),
                    "phase_metrics": {
                        "extract_from_corridor": {"success": True},
                        "push_to_fixture": {"success": True},
                        "flip_at_fixture": {
                            "success": False,
                            "effect_success": False,
                        },
                    },
                }
            },
        )
        self.assertEqual(record["failure_stage"], "primitive_effect")
        self.assertEqual(record["failure_segment"], "flip_at_fixture")
        self.assertEqual(record["downstream_failure_segment"], "grasp_and_lift")

    def test_release_failure_is_attributed_to_transit(self) -> None:
        record = _attach_execution_diagnostics(
            {"binding_written": True, "binding_valid": True},
            {
                "metrics": {
                    "partial_strategy": True,
                    "failed_segment": "push_to_fixture__to__flip_at_fixture",
                    "failure_type": "FR3TransitViolation",
                    "failure_message": "failed to certify a safe terminal release",
                }
            },
        )
        self.assertEqual(record["failure_stage"], "transit")
        self.assertEqual(
            record["failure_segment"],
            "push_to_fixture__to__flip_at_fixture",
        )

    def test_strategy_scene_alias_resolves_complete_generated_set(self) -> None:
        scene_set = resolve_scene_set(REPO_ROOT, "strategy_2")
        self.assertEqual(scene_set.strategy_id, 2)
        self.assertEqual(
            scene_set.root,
            REPO_ROOT / "tasks/sim_benchmark/strategy_2",
        )
        self.assertEqual(set(scene_set.cases), set(range(1, 51)))

    def test_default_eval_suite_uses_attempt_destination(self) -> None:
        source = REPO_ROOT / "suite_runs/fr3_osc_demo_2_001"
        destination = _eval_run_dir(REPO_ROOT, source, None)
        self.assertEqual(
            destination,
            REPO_ROOT / "suite_runs/fr3_osc_demo_2_001_eval_attempt",
        )

    def test_strict_eval_suite_uses_independent_destination(self) -> None:
        source = REPO_ROOT / "suite_runs/fr3_osc_demo_2_001"
        destination = _eval_run_dir(
            REPO_ROOT,
            source,
            None,
            execution_policy="strict",
        )
        self.assertEqual(
            destination,
            REPO_ROOT / "suite_runs/fr3_osc_demo_2_001_eval",
        )

    def test_recoverable_selection_requires_failed_typed_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prior = root / "suite_runs/source_eval"
            output = prior / "benchmark_runs/strategy_8/agent"
            scene_set = BenchmarkSceneSet(
                root=root / "tasks/strategy_8",
                strategy_id=8,
                cases={
                    1: root / "tasks/strategy_8/case_001_scene",
                    2: root / "tasks/strategy_8/case_002_scene",
                    3: root / "tasks/strategy_8/case_003_scene",
                },
            )
            prior.mkdir(parents=True)
            (prior / "benchmark_eval.json").write_text(
                json.dumps(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "strategy_id": 8,
                        "strategy": "strategy_8",
                        "execution_policy": "strict",
                    }
                ),
                encoding="utf-8",
            )
            for case_index, success, failure_type, retained in (
                (1, False, "FR3TransitViolation", ["push__rejected"]),
                (2, True, "FR3TransitViolation", ["push__rejected"]),
                (3, False, "ValueError", ["grasp__rejected"]),
            ):
                case_root = output / f"case_{case_index:03d}"
                (case_root / "artifacts").mkdir(parents=True)
                (case_root / "benchmark_result.json").write_text(
                    json.dumps(
                        {
                            "case_index": case_index,
                            "benchmark_success": success,
                        }
                    ),
                    encoding="utf-8",
                )
                (case_root / "artifacts/metrics.json").write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "failure_type": failure_type,
                                "retained_segments": retained,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            selected = recoverable_case_indices(
                root=root,
                prior_eval_suite="source_eval",
                scene_set=scene_set,
                strategy="strategy_8",
                binding_mode="agent",
            )

        self.assertEqual(selected, (1,))

    def test_public_cli_defaults_to_attempt_with_gpu_and_agent_binding(self) -> None:
        args = build_parser().parse_args(
            ["fr3_osc_demo_2_001", "strategy_2"]
        )
        self.assertEqual(args.binding_mode, "agent")
        self.assertEqual(args.gpu_render, "required")
        self.assertEqual(args.jobs, 1)
        self.assertEqual(args.execution_policy, "attempt")

    def test_public_cli_accepts_targeted_attempt_policy(self) -> None:
        args = build_parser().parse_args(
            [
                "fr3_osc_demo_2_001",
                "strategy_2",
                "--execution-policy",
                "attempt",
                "--retry-recoverable-from",
                "fr3_osc_demo_2_001_eval",
            ]
        )
        self.assertEqual(args.execution_policy, "attempt")
        self.assertEqual(
            args.retry_recoverable_from,
            "fr3_osc_demo_2_001_eval",
        )

    def test_public_cli_accepts_parallel_resource_limits(self) -> None:
        args = build_parser().parse_args(
            [
                "fr3_osc_demo_2_001",
                "strategy_2",
                "--jobs",
                "2",
                "--rollout-workers-per-job",
                "24",
            ]
        )
        self.assertEqual(args.jobs, 2)
        self.assertEqual(args.rollout_workers_per_job, 24)

    def test_public_and_internal_clis_accept_candidate_seal_roots(self) -> None:
        public = build_parser().parse_args(
            [
                "fr3_osc_demo_6_002",
                "tasks/sim_benchmark_candidates/osc_workspace_v1/strategy_6",
                "--seal-root",
                "benchmark_private/candidates/osc_workspace_v1",
            ]
        )
        internal = runner_parser().parse_args(
            [
                "fr3_osc_demo_6_002_eval_attempt_osc_workspace_v1",
                "--strategy-id",
                "8",
                "--strategy",
                "strategy_6",
                "--seal-root",
                "benchmark_private/candidates/osc_workspace_v1",
            ]
        )
        self.assertEqual(public.seal_root, internal.seal_root)

    def test_internal_runner_accepts_direct_eval_workspace(self) -> None:
        args = runner_parser().parse_args(
            [
                "fr3_osc_demo_2_001_eval",
                "--strategy-id",
                "2",
                "--strategy",
                "strategy_2",
                "--workspace-mode",
                "direct",
                "--no-summary",
            ]
        )
        self.assertEqual(args.workspace_mode, "direct")
        self.assertTrue(args.no_summary)
        self.assertEqual(args.execution_policy, "attempt")

    def test_gallery_has_all_slots_and_reads_incremental_summary(self) -> None:
        scene_set = resolve_scene_set(REPO_ROOT, "strategy_2")
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_index": 1,
                                "bucket": "diagnostic",
                                "binding_valid": True,
                                "program_success": True,
                                "trajectory_path": "retained/trajectory.mp4",
                            }
                        ],
                        "summary": {"successes": 1, "total": 1},
                    }
                ),
                encoding="utf-8",
            )
            gallery = write_gallery(
                scene_set=scene_set,
                output_root=output,
                eval_manifest={
                    "source_suite": str(REPO_ROOT / "suite_runs/source"),
                    "eval_suite": str(REPO_ROOT / "suite_runs/source_eval"),
                },
                binding_mode="agent",
            )
            document = gallery.read_text(encoding="utf-8")
            self.assertEqual(document.count("<article class="), 50)
            self.assertIn('article class="success"', document)
            self.assertIn("case_001/artifacts/trajectory.mp4", document)
            self.assertIn("complete: 1/50", document)

    def test_targeted_attempt_gallery_only_reserves_selected_slots(self) -> None:
        scene_set = resolve_scene_set(REPO_ROOT, "strategy_2")
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_index": 1,
                                "benchmark_success": True,
                                "program_success": True,
                            }
                        ],
                        "summary": {"successes": 1, "total": 1},
                    }
                ),
                encoding="utf-8",
            )
            gallery = write_gallery(
                scene_set=scene_set,
                output_root=output,
                eval_manifest={
                    "source_suite": "source",
                    "eval_suite": "source_eval_attempt",
                    "selected_cases": [1, 3],
                },
                binding_mode="agent",
            )
            document = gallery.read_text(encoding="utf-8")

        self.assertEqual(document.count("<article class="), 2)
        self.assertIn("complete: 1/2", document)
        self.assertIn("reserves 2 selected result slots", document)

    def test_aggregate_refresh_rescores_retained_storage_results(self) -> None:
        scene_set = resolve_scene_set(REPO_ROOT, "strategy_1")
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            case_root = output / "case_001"
            case_root.mkdir()
            result_path = case_root / "benchmark_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "case_index": 1,
                        "strategy_id": 1,
                        "bucket": "diagnostic",
                        "diagnostic_dimension": "pose",
                        "stress_axis": None,
                        "binding_valid": True,
                        "target_bound": True,
                        "authoritative_scene_role_coverage": 1.0,
                        "program_success": False,
                        "metrics_path": str(case_root / "missing_metrics.json"),
                        "benchmark_success": False,
                        "benchmark_evaluation": {
                            "evaluator": (
                                "strategy_1_terminal_storage_containment_v1"
                            ),
                            "source": "state_trajectory",
                            "success": False,
                            "measurements": {
                                "target_depth_min_m": -0.08,
                                "target_depth_max_m": 0.12,
                                "target_lateral_min_m": -0.04,
                                "target_lateral_max_m": 0.04,
                            },
                            "thresholds": {
                                "available_depth_m": 0.30,
                                "lateral_half_width_m": 0.05,
                                "containment_tolerance_m": 0.005,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            records = _write_aggregate_summary(
                output_root=output,
                scene_set=scene_set,
                strategy="strategy_1",
                binding_mode="agent",
                checkpoint_sha256="checkpoint",
                jobs=4,
                rollout_workers_per_job=12,
            )
            self.assertTrue(records[0]["benchmark_success"])
            refreshed = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertTrue(refreshed["benchmark_success"])
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["summary"]["successes"], 1)
            self.assertEqual(summary["execution_policy"], "attempt")


if __name__ == "__main__":
    unittest.main()
