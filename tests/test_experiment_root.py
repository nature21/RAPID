from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from devtools.agent_session_viewer import find_session_files
from suites.staged.benchmark_eval import _eval_run_dir, _resolve_eval_run_reference
from suites.staged.codex_permissions import path_exposes_blocked_checkpoint
from suites.staged.experiment_paths import experiment_root
from suites.staged.strategy_test import resolve_suite_workspace


REPO = Path(__file__).resolve().parents[1]


class ExperimentRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.custom = self.repo / "suite_runs" / "custom runs"
        self.run = "fr3_osc_demo_1_001"
        self.env_patch = mock.patch.dict(os.environ)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        os.environ.pop("RAPID_EXPERIMENT_ROOT", None)

    def test_default_relative_absolute_and_explicit_path_resolution(self) -> None:
        default = self.repo / "suite_runs"
        for root in (default, self.custom):
            (root / self.run / "workspace/src").mkdir(parents=True)
        (self.custom / "prior").mkdir()
        self.assertEqual(experiment_root(self.repo), default)
        self.assertEqual(resolve_suite_workspace(self.repo, self.run)[0], default / self.run)
        for value in ("custom runs", "suite_runs/custom runs", str(self.custom)):
            with self.subTest(value=value), mock.patch.dict(os.environ, RAPID_EXPERIMENT_ROOT=value):
                self.assertEqual(experiment_root(self.repo), self.custom)
                source, _ = resolve_suite_workspace(self.repo, self.run)
                self.assertEqual(source, self.custom / self.run)
                self.assertEqual(_eval_run_dir(self.repo, source, None), self.custom / f"{self.run}_eval_attempt")
                self.assertEqual(_eval_run_dir(self.repo, source, None, execution_policy="strict"), self.custom / f"{self.run}_eval")
                self.assertEqual(_eval_run_dir(self.repo, source, "chosen"), self.custom / "chosen")
                self.assertEqual(_eval_run_dir(self.repo, source, "elsewhere/chosen"), self.repo / "elsewhere/chosen")
                self.assertEqual(_resolve_eval_run_reference(self.repo, "prior"), self.custom / "prior")
                self.assertEqual(resolve_suite_workspace(self.repo, f"suite_runs/{self.run}")[0], default / self.run)
                self.assertEqual(experiment_root(self.repo, "override"), self.repo / "suite_runs" / "override")

    def test_session_discovery_and_checkpoint_isolation_use_selected_root(self) -> None:
        session = self.custom / self.run / "codex_home/sessions/rollout-test.jsonl"
        session.parent.mkdir(parents=True)
        session.write_text("{}\n")
        workspace = self.custom / self.run / "workspace"
        workspace.mkdir()
        with mock.patch.dict(os.environ, RAPID_EXPERIMENT_ROOT=str(self.custom)):
            self.assertEqual(find_session_files(None, root=self.repo), [session])
            self.assertEqual(find_session_files(self.run, root=self.repo), [session])
            self.assertTrue(path_exposes_blocked_checkpoint(self.custom, repo_root=self.repo, workspace=workspace))
            self.assertFalse(path_exposes_blocked_checkpoint(workspace, repo_root=self.repo, workspace=workspace))


class ExperimentShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.custom = self.repo / "suite_runs" / "custom runs"
        self.run = "fr3_osc_demo_1_001"
        for relative in (
            "launch_evals.sh", "launch_experiments.sh", "run_eval.sh",
            "run_primitive_construction.sh", "run_strategy_composition.sh",
            "scripts/lib/experiment-root.sh", "scripts/lib/slurm-arrays.sh",
            "scripts/rapid-construct-primitive", "scripts/rapid-workspace-env",
            "scripts/rapid-experiment-local",
        ):
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, destination)
        self.bin = self.repo / "bin"
        self.bin.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("RAPID_", "SLURM_", "SBATCH_"))}
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}", MOCK_SUBMISSIONS=str(self.repo / "submissions"))
        self.stub("bin/squeue", 'printf "%s\\n" "${MOCK_QUEUE:-}"')
        self.stub("bin/sinfo", "printf '%s\\n' crane0 crane1 crane2 crane5")
        self.stub("bin/scontrol", '''
[[ "$1 $2" == "show hostnames" ]] || exit 9
if [[ "$3" == 'crane[0-2]' ]]; then printf '%s\\n' crane0 crane1 crane2
else echo "$3" | tr ',' '\\n'; fi
''')
        self.stub("bin/sbatch", '''
count_file="${MOCK_SUBMISSIONS}.count"
count=100
if [[ -f "$count_file" ]]; then read -r count < "$count_file"; fi
count=$((count + 1))
printf '%s\\n' "$count" > "$count_file"
printf '%s\\n' "JOB=$count" "$RAPID_EXPERIMENT_ROOT" "$@" >> "$MOCK_SUBMISSIONS"
echo "$count;testcluster"
''')
        self.stub("scripts/rapid-suite", 'echo "$RAPID_EXPERIMENT_ROOT/runtime_001/workspace"')
        for root in (self.repo / "suite_runs", self.custom):
            for i in range(1, 9):
                (root / f"fr3_osc_demo_{i}_001/workspace/src").mkdir(parents=True)
        for i in range(1, 9):
            scene = self.repo / f"tasks/sim_benchmark/strategy_{i}/case_001_scene"
            scene.mkdir(parents=True)
            (scene / "benchmark_case.json").write_text("{}")

    def stub(self, relative: str, body: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_script(self, script: str, *args: str, success=True):
        result = subprocess.run(["bash", str(self.repo / script), *args], cwd=self.repo,
                                env=self.env, text=True, capture_output=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def submissions(self):
        return (self.repo / "submissions").read_text()

    def test_local_pipeline_uses_existing_stages_without_scheduler(self):
        self.env.update(SLURM_ARRAY_TASK_ID="3", SLURM_JOB_ID="999")
        for script in ("run_primitive_construction.sh", "run_strategy_composition.sh", "scripts/rapid-benchmark"):
            self.stub(script, '''
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_JOB_ID:-}" ]]
printf '%s|%s|%s|%s\\n' "$0" "$*" "$RAPID_EXPERIMENT_MODEL" "$RAPID_MAX_PARALLEL_ROLLOUTS" >> "$MOCK_SUBMISSIONS"
''')
        self.stub("scripts/rapid-python", 'printf "%s\\n" "$*" >> "$MOCK_SUBMISSIONS"')
        self.run_script("scripts/rapid-experiment-local", "--experiment-root", "custom runs")
        calls = self.submissions()
        self.assertNotIn("JOB=", calls)
        self.assertEqual(calls.count("run_primitive_construction.sh|"), 1)
        self.assertEqual(calls.count("run_strategy_composition.sh|"), 1)
        self.assertEqual(calls.count("scripts/rapid-benchmark|"), 8)
        self.assertEqual(calls.count("--jobs 4 --rollout-workers-per-job 8"), 8)
        self.assertEqual(calls.count("--execution-policy attempt --resume"), 8)
        self.assertEqual(calls.count("|gpt-6-astra|32"), 10)

    def test_local_dry_run_and_eval_subset(self):
        self.env["RAPID_EVAL_STRATEGIES"] = "2,8"
        result = self.run_script("scripts/rapid-experiment-local", "--stage", "eval", "--dry-run",
                                 "--experiment-root", "new run", "--rollout-workers", "8", "--eval-workers", "2")
        self.assertFalse((self.repo / "suite_runs/new run").exists())
        self.assertFalse((self.repo / "submissions").exists())
        self.assertEqual(result.stdout.count("./scripts/rapid-benchmark"), 2)
        self.assertIn("--jobs 2 --rollout-workers-per-job 4", result.stdout)
        self.assertNotIn("./run_primitive_construction.sh", result.stdout)

    def test_local_rejects_invalid_configuration_before_writing(self):
        for args in (("--stage", "bad"), ("--batch", "1"),
                     ("--rollout-workers", "0"), ("--eval-workers", "33"),
                     ("--rollout-workers", "7"), ("--model",)):
            with self.subTest(args=args):
                self.run_script("scripts/rapid-experiment-local", *args, success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_local_stops_after_construction_failure(self):
        self.stub("run_primitive_construction.sh", 'exit 9')
        self.stub("run_strategy_composition.sh", 'echo unexpected >> "$MOCK_SUBMISSIONS"')
        self.run_script("scripts/rapid-experiment-local", success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def prepare_workers(self):
        self.env.update(RAPID_EXPERIMENT_ROOT=str(self.custom), SLURM_CPUS_PER_TASK="32",
                        SLURM_SUBMIT_DIR=str(self.repo), SLURM_JOB_ID="456",
                        SLURM_ARRAY_JOB_ID="450", SLURM_ARRAY_TASK_ID="1", BATCH_ID="001")
        self.stub("scripts/rapid-suite", '''
echo "$1" >> "$RAPID_EXPERIMENT_ROOT/library_calls"
if [[ "$1" == merge-workspaces ]]; then
  [[ "${MOCK_MERGE_FAIL:-0}" == 0 ]] || exit 6
  [[ ! -d "${3}/workspace" ]] || exit 8
  mkdir -p "${3}/workspace"
fi
''')
        self.stub("scripts/rapid-python", 'echo "goal: test; description: test"')
        capture = '''
while [[ "$1" != --name ]]; do shift; done
printf '%s|%s\\n' "$2" "$RAPID_MAX_PARALLEL_ROLLOUTS" >> "$MOCK_SUBMISSIONS"
[[ "${MOCK_AGENT_FAIL:-0}" == 0 ]] || exit 5
'''
        self.stub("scripts/rapid-construct-primitive", capture)
        self.stub("scripts/rapid-codex-run", capture)
        self.stub("bin/srun", 'echo "Unexpected cross-node dispatch" >&2; exit 99')

    def test_three_arrays_have_resources_dependencies_and_priority(self):
        shutil.rmtree(self.custom)
        self.run_script("launch_experiments.sh", "--experiment-root", "custom runs", "001")
        submitted = self.submissions()
        self.assertEqual(submitted.count("JOB="), 3)
        for flag in ("--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=64G", "--gres=gpu:1", "--kill-on-invalid-dep=yes", "--exclude=crane5"):
            self.assertEqual(submitted.splitlines().count(flag), 3, flag)
        self.assertIn("--array=1,2,3,4,5,6\n", submitted)
        self.assertEqual(submitted.count("--array=1,2,3,4,5,6,7,8\n"), 2)
        self.assertIn("--dependency=afterok:101\n", submitted)
        self.assertIn("--dependency=aftercorr:102\n", submitted)
        self.assertEqual(submitted.count("--nice=0\n"), 2)
        self.assertEqual(submitted.count("--nice=10000\n"), 1)
        self.assertNotIn("--nodelist", submitted)
        self.assertFalse(self.custom.exists())

    def test_pipeline_waits_for_previous_eval_array_before_construction(self):
        self.run_script("launch_experiments.sh", "--after", "999", "001")
        jobs = self.submissions().split("JOB=")[1:]
        self.assertEqual(len(jobs), 3)
        self.assertIn("--dependency=afterok:999\n", jobs[0])
        self.assertIn("--dependency=afterok:101\n", jobs[1])
        self.assertIn("--dependency=aftercorr:102\n", jobs[2])
        self.assertNotIn("afterok:999", jobs[1] + jobs[2])

    def test_invalid_pipeline_dependency_prevents_submission(self):
        for args in (("--after",), ("--after", "0"), ("--after", "bad"),
                     ("--after", "123", "--after", "456"),
                     ("--after-start",), ("--after-start", "bad"),
                     ("--after", "123", "--after-start", "456")):
            self.run_script("launch_experiments.sh", *args, success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_pipeline_can_start_after_upstream_array_is_allocated(self):
        self.run_script("launch_experiments.sh", "--experiment-root", "custom runs", "--after-start", "999", "001")
        jobs = self.submissions().split("JOB=")[1:]
        self.assertEqual(len(jobs), 3)
        self.assertIn(str(self.custom), jobs[0])
        self.assertIn("--dependency=after:999\n", jobs[0])
        self.assertIn("--dependency=afterok:101\n", jobs[1])
        self.assertIn("--dependency=aftercorr:102\n", jobs[2])

    def test_subset_keeps_paper_ids_for_matching_eval(self):
        self.env.update(RAPID_CONSTRUCTION_TASKS="topple push", RAPID_COMPOSITION_RUNS="2,7")
        self.run_script("launch_experiments.sh", "001")
        submitted = self.submissions()
        self.assertIn("--array=6,2\n", submitted)
        self.assertEqual(submitted.count("--array=2,7\n"), 2)

    def test_unknown_duplicate_and_unmatched_ids_fail_before_submission(self):
        for variable, value in (("RAPID_CONSTRUCTION_TASKS", "flip bogus"), ("RAPID_CONSTRUCTION_TASKS", "flip flip"),
                                ("RAPID_COMPOSITION_RUNS", "9"), ("RAPID_COMPOSITION_RUNS", "1,1")):
            with self.subTest(variable=variable, value=value):
                self.env[variable] = value
                self.run_script("launch_experiments.sh", "001", success=False)
                self.env.pop(variable)
                self.assertFalse((self.repo / "submissions").exists())
        self.env.update(RAPID_COMPOSITION_RUNS="2", RAPID_EVAL_STRATEGIES="1")
        self.run_script("launch_experiments.sh", "001", success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_nodes_select_pool_without_requiring_all_nodes(self):
        self.env.update(RAPID_STAGE_NODES="unusednode", RAPID_EVAL_NODES="crane2")
        self.run_script("launch_experiments.sh", "--nodes", "crane0,crane1", "001")
        submitted = self.submissions()
        self.assertEqual(submitted.count("--exclude=crane2,crane5\n"), 2)
        self.assertIn("--exclude=crane0,crane1,crane5\n", submitted)
        self.assertNotIn("unusednode", submitted)

    def test_invalid_or_missing_nodes_rejected_before_submission(self):
        for args in (("--nodes",), ("--nodes", "unknown")):
            self.run_script("launch_experiments.sh", *args, success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_snapshot_failure_prevents_submission(self):
        self.stub("scripts/rapid-suite", "exit 9")
        result = self.run_script("launch_experiments.sh", "001", success=False)
        self.assertEqual(result.returncode, 9)
        self.assertFalse((self.repo / "submissions").exists())

    def test_failed_composition_submission_prevents_eval_submission(self):
        self.stub("bin/sbatch", '''
printf '%s\\n' "$@" >> "$MOCK_SUBMISSIONS"
[[ "$*" != *run_strategy_composition.sh* ]] || exit 1
echo 123
''')
        self.run_script("launch_experiments.sh", "001", success=False)
        self.assertNotIn("run_eval.sh", self.submissions())

    def test_no_eval_submits_only_two_arrays(self):
        self.run_script("launch_experiments.sh", "--no-eval", "001")
        self.assertEqual(self.submissions().count("JOB="), 2)
        self.assertNotIn("run_eval.sh", self.submissions())

    def test_primitive_array_maps_every_id_once_with_32_workers(self):
        self.prepare_workers()
        self.env["RAPID_CONSTRUCTION_TASKS"] = "topple"  # must not remap fixed array IDs
        for index in range(1, 7):
            self.env["SLURM_ARRAY_TASK_ID"] = str(index)
            self.run_script("run_primitive_construction.sh")
        self.assertEqual(self.submissions().splitlines(), [f"{name}|32" for name in ("flip", "push", "grasp", "press_move", "pivot", "topple")])

    def test_strategy_arrays_share_one_merged_library_under_concurrency(self):
        self.prepare_workers()
        processes = []
        for index in (2, 7):
            env = dict(self.env, SLURM_ARRAY_TASK_ID=str(index))
            processes.append(subprocess.Popen(["bash", str(self.repo / "run_strategy_composition.sh")],
                             cwd=self.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for process in processes:
            out, err = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, out + err)
        calls = (self.custom / "library_calls").read_text().splitlines()
        self.assertEqual(calls, ["merge-workspaces", "create-workspace", "create-workspace"])
        self.assertCountEqual(self.submissions().splitlines(), ["strategy_2|32", "strategy_7|32"])
        self.env.update(SLURM_ARRAY_JOB_ID="460", SLURM_ARRAY_TASK_ID="8")
        self.run_script("run_strategy_composition.sh")
        self.assertEqual((self.custom / "library_calls").read_text().count("merge-workspaces"), 2)

    def test_merge_failure_does_not_publish_marker_or_launch_agent(self):
        self.prepare_workers()
        self.env["MOCK_MERGE_FAIL"] = "1"
        self.assertEqual(self.run_script("run_strategy_composition.sh", success=False).returncode, 6)
        marker = self.custom / "library_001_array_450/.merge_complete"
        self.assertFalse(marker.exists())
        self.assertFalse((self.repo / "submissions").exists())
        self.env.pop("MOCK_MERGE_FAIL")
        self.run_script("run_strategy_composition.sh")
        self.assertTrue(marker.exists())
        self.assertIn("strategy_1|32", self.submissions())

    def test_agent_failure_propagates_to_array_element(self):
        self.prepare_workers()
        self.env["MOCK_AGENT_FAIL"] = "1"
        for script in ("run_primitive_construction.sh", "run_strategy_composition.sh"):
            self.run_script(script, success=False)

    def test_bad_array_ids_fail_without_launching(self):
        self.prepare_workers()
        for script, task_id in (("run_primitive_construction.sh", "7"), ("run_strategy_composition.sh", "9"), ("run_eval.sh", "0")):
            self.env["SLURM_ARRAY_TASK_ID"] = task_id
            self.run_script(script, success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_eval_array_uses_matching_id_and_existing_retry_manifest(self):
        self.prepare_workers()
        self.env.update(SLURM_ARRAY_TASK_ID="7", RAPID_EVAL_RETRY_RECOVERABLE_DEFAULT="auto")
        self.stub("scripts/rapid-benchmark", 'printf "%s\\n" "$RAPID_EXPERIMENT_ROOT" "$@"')
        manifest = self.custom / "fr3_osc_demo_7_001_eval_attempt/benchmark_eval.json"
        manifest.parent.mkdir()
        manifest.write_text('{\n "recoverable_selection_source": "/prior/eval7"\n}\n')
        result = self.run_script("run_eval.sh")
        self.assertIn("fr3_osc_demo_7_001\nstrategy_7\n--jobs\n4", result.stdout)
        self.assertIn("--retry-recoverable-from\n/prior/eval7", result.stdout)
        self.assertIn("--rollout-workers-per-job\n8", result.stdout)
        self.assertIn(str(self.custom), result.stdout)

    def test_eval_dry_run_and_legacy_whole_job_dependency(self):
        shutil.rmtree(self.custom)
        for flag, dependency in (("--after-array", "aftercorr"), ("--after", "afterok")):
            result = self.run_script("launch_evals.sh", "--experiment-root", "custom runs", flag, "456", "--dry-run", "001")
            self.assertIn(f"--dependency={dependency}:456", result.stdout)
        self.assertFalse((self.repo / "submissions").exists())

    def test_eval_deduplicates_array_tasks_and_skips_completed_selected_root(self):
        self.env["RAPID_EVAL_STRATEGIES"] = "1,2,3"
        digest = hashlib.sha256(str(self.custom).encode()).hexdigest()[:12]
        self.env["MOCK_QUEUE"] = f"999_2|rapid-eval-{digest}-001-attempt"
        complete = self.custom / f"{self.run}_eval_attempt/benchmark_runs/strategy_1/agent/case_001"
        complete.mkdir(parents=True)
        (complete / "benchmark_result.json").write_text("{}")
        result = self.run_script("launch_evals.sh", "--experiment-root", "custom runs", "--dry-run", "001")
        self.assertIn("--array=3 ", result.stdout)
        self.assertNotIn("--array=1", result.stdout)
        self.env["RAPID_EVAL_STRATEGIES"] = "1,2"
        result = self.run_script("launch_evals.sh", "--experiment-root", "custom runs", "001")
        self.assertIn("queued/planned evals=0", result.stdout)
        self.assertFalse((self.repo / "submissions").exists())

    def test_auto_retry_selection_and_no_work_skip(self):
        self.env.update(RAPID_EVAL_STRATEGIES="1", RAPID_EVAL_RETRY_RECOVERABLE_FROM="auto")
        prior = self.custom / f"{self.run}_eval"
        prior.mkdir()
        (prior / "benchmark_eval.json").write_text("{}")
        case = prior / "benchmark_runs/strategy_1/agent/case_001"
        (case / "artifacts").mkdir(parents=True)
        result = self.run_script("launch_evals.sh", "--experiment-root", "custom runs", "001")
        self.assertIn("queued/planned evals=0", result.stdout)
        (case / "benchmark_result.json").write_text('{"benchmark_success": false}')
        (case / "artifacts/metrics.json").write_text('{"failure_type": "FR3TransitViolation", "segment": "move__rejected"}')
        result = self.run_script("launch_evals.sh", "--experiment-root", "custom runs", "--dry-run", "001")
        self.assertIn("--array=1 ", result.stdout)

    def test_invalid_dependency_rejected(self):
        for args in (("--after-array",), ("--after-array", "0"), ("--after", "afterok:123"), ("--after", "1", "--after-array", "2")):
            self.run_script("launch_evals.sh", *args, success=False)
        self.assertFalse((self.repo / "submissions").exists())

    def test_spooled_script_uses_submission_repository(self):
        self.prepare_workers()
        spool = self.repo / "spool"
        spool.mkdir()
        for script in ("run_primitive_construction.sh", "run_strategy_composition.sh"):
            shutil.copy2(self.repo / script, spool / script)
            self.run_script(f"spool/{script}")
        self.assertIn("flip|32", self.submissions())
        self.assertIn("strategy_1|32", self.submissions())

    def test_root_validation_and_direct_eval_compatibility(self):
        self.stub("scripts/rapid-benchmark", 'printf "%s\\n" "$RAPID_EXPERIMENT_ROOT" "$@"')
        result = self.run_script("run_eval.sh", "--experiment-root", "custom runs", self.run, "strategy_1", "4")
        self.assertIn(str(self.custom), result.stdout)
        for args in (("--experiment-root",), ("--experiment-root=",)):
            self.run_script("launch_evals.sh", *args, success=False)


if __name__ == "__main__":
    unittest.main()
