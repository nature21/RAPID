"""Create an eval-suite copy and run a learned strategy on benchmark scenes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, IO, Sequence

from suites.staged.experiment_paths import (
    add_experiment_root_argument,
    configure_experiment_root,
    experiment_root,
)

from benchmark_private.generator import tree_sha256
from benchmark_private.specs import BENCHMARK_VERSION, CASES_PER_STRATEGY
from suites.staged.benchmark_runner import (
    _attach_benchmark_evaluation,
    _case_indices,
    _frozen_checkpoint_digest,
    _resolve_seal_root,
    _summary,
)
from suites.staged.gpu_render_broker import GPU_RENDER_MODES
from suites.staged.strategy_test import resolve_strategy, resolve_suite_workspace
from suites.staged.workspaces import create_isolated_workspace, repo_root_from_env


EVAL_MANIFEST_NAME = "benchmark_eval.json"
EXECUTION_POLICIES = ("strict", "attempt")
EXECUTION_POLICY_ENV = "RAPID_EXECUTION_POLICY"
ATTEMPT_EXECUTION_POLICY = "attempt"
DEFAULT_EXECUTION_POLICY = ATTEMPT_EXECUTION_POLICY
_STRATEGY_NAME = re.compile(r"^strategy_([1-9][0-9]*)$")
_CASE_NAME = re.compile(r"^case_([0-9]{3})_scene$")


@dataclass(frozen=True)
class BenchmarkSceneSet:
    """One complete, generated strategy benchmark scene set."""

    root: Path
    strategy_id: int
    cases: dict[int, Path]

    @property
    def name(self) -> str:
        return self.root.name


@dataclass
class RunningCase:
    """One case subprocess occupying a reusable parallel worker."""

    case_index: int
    worker_index: int
    worker_run: Path
    process: subprocess.Popen[str]
    log_path: Path
    log_file: IO[str]


def _resolve_repo_path(root: Path, value: str) -> Path:
    text = value.strip().removesuffix("/")
    if text.startswith("tasks."):
        return (root / text.replace(".", "/")).resolve()
    if _STRATEGY_NAME.fullmatch(text):
        return (root / "tasks" / "sim_benchmark" / text).resolve()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def resolve_scene_set(root: Path, value: str) -> BenchmarkSceneSet:
    """Resolve and validate a full ``case_NNN_scene`` strategy directory."""

    scene_root = _resolve_repo_path(root, value)
    try:
        scene_root.relative_to(root / "tasks")
    except ValueError as exc:
        raise ValueError(
            f"strategy scenes must live under {root / 'tasks'}: {scene_root}"
        ) from exc
    if not scene_root.is_dir():
        raise FileNotFoundError(
            f"strategy scene directory does not exist: {scene_root}"
        )
    match = _STRATEGY_NAME.fullmatch(scene_root.name)
    if match is None:
        raise ValueError(
            "strategy scene directory must be named strategy_N; got "
            f"{scene_root.name!r}"
        )
    strategy_id = int(match.group(1))
    cases: dict[int, Path] = {}
    for child in scene_root.iterdir():
        case_match = _CASE_NAME.fullmatch(child.name)
        if case_match is None or not child.is_dir():
            continue
        case_index = int(case_match.group(1))
        if not 1 <= case_index <= CASES_PER_STRATEGY:
            continue
        missing = [
            name
            for name in ("scene.py", "goal.txt", "benchmark_case.json")
            if not (child / name).is_file()
        ]
        if missing:
            raise ValueError(f"{child} is missing {', '.join(missing)}")
        public = json.loads(
            (child / "benchmark_case.json").read_text(encoding="utf-8")
        )
        expected_id = f"strategy_{strategy_id}/case_{case_index:03d}"
        if public.get("benchmark_version") != BENCHMARK_VERSION:
            raise ValueError(f"benchmark version mismatch in {child}")
        if public.get("case_id") != expected_id or public.get("scored") is not True:
            raise ValueError(f"invalid benchmark metadata in {child}")
        cases[case_index] = child
    expected = set(range(1, CASES_PER_STRATEGY + 1))
    if set(cases) != expected:
        missing = sorted(expected - set(cases))
        raise ValueError(
            f"{scene_root} must contain all {CASES_PER_STRATEGY} scenes; "
            f"missing cases: {missing}"
        )
    return BenchmarkSceneSet(
        root=scene_root,
        strategy_id=strategy_id,
        cases=cases,
    )


def _eval_run_dir(
    root: Path,
    source_run: Path,
    requested: str | None,
    *,
    execution_policy: str = DEFAULT_EXECUTION_POLICY,
) -> Path:
    if requested is None:
        suffix = "eval_attempt" if execution_policy == "attempt" else "eval"
        return (experiment_root(root) / f"{source_run.name}_{suffix}").resolve()
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = (
            experiment_root(root) / candidate
            if len(candidate.parts) == 1
            else root / candidate
        )
    candidate = candidate.resolve()
    if candidate.name == "workspace":
        candidate = candidate.parent
    return candidate


def _eval_runtime_sha256(root: Path) -> str:
    """Digest the canonical shared runtime that implements eval continuation."""

    digest = hashlib.sha256()
    for relative in (
        Path("strategies/runtime.py"),
        Path("instantiation_core/runner.py"),
    ):
        path = root / "workspace/src" / relative
        if not path.is_file():
            raise FileNotFoundError(f"canonical eval runtime is missing: {path}")
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_eval_run_reference(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (
            experiment_root(root) / candidate
            if len(candidate.parts) == 1
            else root / candidate
        )
    candidate = candidate.resolve()
    if candidate.name == "workspace":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"prior eval suite does not exist: {candidate}")
    return candidate


def recoverable_case_indices(
    *,
    root: Path,
    prior_eval_suite: str,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    binding_mode: str,
    source_run: Path | None = None,
) -> tuple[int, ...]:
    """Select failed strict cases with a retained rejected transit trajectory."""

    prior_eval = _resolve_eval_run_reference(root, prior_eval_suite)
    manifest_path = prior_eval / EVAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"prior eval suite has no {EVAL_MANIFEST_NAME}: {prior_eval}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "benchmark_version": BENCHMARK_VERSION,
        "strategy_id": scene_set.strategy_id,
        "strategy": strategy,
    }
    if source_run is not None:
        expected["source_suite"] = str(source_run)
        expected["checkpoint_sha256"] = _frozen_checkpoint_digest(
            source_run / "workspace" / "src",
            strategy,
        )
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if manifest.get("execution_policy", "strict") != "strict":
        mismatches.append("execution_policy")
    if mismatches:
        raise ValueError(
            "prior eval suite does not match recoverable selection ("
            + ", ".join(mismatches)
            + f"): {prior_eval}"
        )
    output_root = (
        prior_eval / "benchmark_runs" / scene_set.name / binding_mode
    )
    selected: list[int] = []
    for result_path in sorted(output_root.glob("case_*/benchmark_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        case_index = result.get("case_index")
        if (
            not isinstance(case_index, int)
            or result.get("benchmark_success") is not False
            or case_index not in scene_set.cases
        ):
            continue
        metrics_path = result_path.parent / "artifacts" / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        core = metrics.get("metrics")
        if not isinstance(core, dict):
            continue
        retained = core.get("retained_segments")
        if (
            isinstance(retained, list)
            and any(str(segment).endswith("__rejected") for segment in retained)
            and core.get("failure_type") == "FR3TransitViolation"
        ):
            selected.append(case_index)
    return tuple(sorted(set(selected)))


def _eval_manifest_payload(
    *,
    source_run: Path,
    source_workspace: Path,
    eval_run: Path,
    scene_set: BenchmarkSceneSet,
    seal_root: Path,
    strategy: str,
    checkpoint_sha256: str,
    execution_policy: str,
    eval_runtime_sha256: str,
    selected_cases: Sequence[int],
    recoverable_selection_source: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_suite": str(source_run),
        "source_workspace": str(source_workspace),
        "eval_suite": str(eval_run),
        "scene_root": str(scene_set.root),
        "seal_root": str(seal_root),
        "benchmark_version": BENCHMARK_VERSION,
        "strategy_id": scene_set.strategy_id,
        "strategy": strategy,
        "checkpoint_sha256": checkpoint_sha256,
        "execution_policy": execution_policy,
        "eval_runtime_sha256": eval_runtime_sha256,
        "selected_cases": list(selected_cases),
        "recoverable_selection_source": recoverable_selection_source,
    }


def prepare_eval_suite(
    *,
    root: Path,
    source_suite: str,
    eval_suite: str | None,
    scene_set: BenchmarkSceneSet,
    seal_root: Path,
    strategy: str,
    resume: bool,
    execution_policy: str,
    selected_cases: Sequence[int],
    recoverable_selection_source: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Create or validate the disposable suite copy used for evaluation."""

    source_run, source_workspace = resolve_suite_workspace(root, source_suite)
    resolved_strategy = resolve_strategy(source_workspace / "src", strategy)
    if resolved_strategy != strategy:
        raise ValueError(
            f"resolved strategy {resolved_strategy!r} does not match {strategy!r}"
        )
    eval_run = _eval_run_dir(
        root,
        source_run,
        eval_suite,
        execution_policy=execution_policy,
    )
    runtime_sha256 = _eval_runtime_sha256(root)
    if eval_run == source_run:
        raise ValueError("eval suite must be different from the source suite")
    manifest_path = eval_run / EVAL_MANIFEST_NAME
    if eval_run.exists():
        if not resume:
            raise FileExistsError(
                f"eval suite already exists: {eval_run}; pass --resume to continue"
            )
        _, eval_workspace = resolve_suite_workspace(root, str(eval_run))
        if not manifest_path.is_file():
            raise ValueError(
                f"existing eval suite has no {EVAL_MANIFEST_NAME}: {eval_run}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "source_workspace": str(source_workspace),
            "scene_root": str(scene_set.root),
            "seal_root": str(seal_root),
            "benchmark_version": BENCHMARK_VERSION,
            "strategy_id": scene_set.strategy_id,
            "strategy": strategy,
            "execution_policy": execution_policy,
            "eval_runtime_sha256": runtime_sha256,
            "selected_cases": list(selected_cases),
            "recoverable_selection_source": recoverable_selection_source,
        }
        mismatches: list[str] = []
        for key, value in expected.items():
            actual = manifest.get(key)
            if key == "seal_root" and actual is None:
                actual = str(_resolve_seal_root(root, None))
            if execution_policy == "strict":
                if key == "execution_policy" and actual is None:
                    actual = "strict"
                elif key == "selected_cases" and actual is None:
                    actual = sorted(scene_set.cases)
                elif key == "recoverable_selection_source" and actual is None:
                    actual = None
                elif key == "eval_runtime_sha256" and actual is None:
                    # Version-1 strict eval manifests predate runtime auditing.
                    actual = runtime_sha256
            if actual != value:
                mismatches.append(key)
        eval_checkpoint = _frozen_checkpoint_digest(
            eval_workspace / "src", strategy
        )
        if eval_checkpoint != manifest.get("checkpoint_sha256"):
            mismatches.append("checkpoint_sha256")
        if mismatches:
            raise ValueError(
                "eval suite does not match this request: " + ", ".join(mismatches)
            )
        return eval_run, eval_workspace, manifest

    source_checkpoint = _frozen_checkpoint_digest(
        source_workspace / "src", strategy
    )
    created = create_isolated_workspace(
        eval_run,
        repo_root=root,
        from_workspace=source_workspace,
    )
    copied_checkpoint = _frozen_checkpoint_digest(created.src, strategy)
    if copied_checkpoint != source_checkpoint:
        raise RuntimeError("eval-suite copy changed the frozen strategy checkpoint")
    manifest = _eval_manifest_payload(
        source_run=source_run,
        source_workspace=source_workspace,
        eval_run=eval_run,
        scene_set=scene_set,
        seal_root=seal_root,
        strategy=strategy,
        checkpoint_sha256=source_checkpoint,
        execution_policy=execution_policy,
        eval_runtime_sha256=runtime_sha256,
        selected_cases=selected_cases,
        recoverable_selection_source=recoverable_selection_source,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return eval_run, created.workspace, manifest


def _read_summary(output_root: Path) -> dict[str, Any]:
    path = output_root / "summary.json"
    if not path.is_file():
        return {"cases": [], "summary": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"cases": [], "summary": {}}
    return payload if isinstance(payload, dict) else {"cases": [], "summary": {}}


def _archive_stale_case_results(
    output_root: Path,
    scene_set: BenchmarkSceneSet,
) -> list[Path]:
    """Move results for replaced scene hashes aside before a resumed run."""

    archived: list[Path] = []
    superseded_root = output_root / "_superseded"
    for case_index, case_dir in sorted(scene_set.cases.items()):
        run_dir = output_root / f"case_{case_index:03d}"
        result_path = run_dir / "benchmark_result.json"
        if not result_path.is_file():
            continue
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        current_sha256 = tree_sha256(case_dir)
        if record.get("case_sha256") == current_sha256:
            continue
        superseded_root.mkdir(parents=True, exist_ok=True)
        old_digest = str(record.get("case_sha256") or "unknown")[:12]
        destination = superseded_root / f"case_{case_index:03d}__{old_digest}"
        if destination.exists():
            destination = superseded_root / (
                f"case_{case_index:03d}__{old_digest}__{os.getpid()}"
            )
        run_dir.rename(destination)
        archived.append(destination)
    return archived


def write_gallery(
    *,
    scene_set: BenchmarkSceneSet,
    output_root: Path,
    eval_manifest: dict[str, Any],
    binding_mode: str,
) -> Path:
    """Write a fixed-slot video gallery compatible with incremental runs."""

    output_root.mkdir(parents=True, exist_ok=True)
    payload = _read_summary(output_root)
    retained_records = _collect_records(output_root)
    records_collected_from_cases = bool(retained_records)
    if not retained_records:
        retained_records = [
            item for item in payload.get("cases", []) if isinstance(item, dict)
        ]
    records = {
        int(item["case_index"]): item
        for item in retained_records
        if isinstance(item, dict) and "case_index" in item
    }
    selected_raw = eval_manifest.get("selected_cases")
    selected_cases = (
        {int(index) for index in selected_raw}
        if isinstance(selected_raw, list)
        else set(scene_set.cases)
    )
    cards: list[str] = []
    for case_index, case_dir in sorted(scene_set.cases.items()):
        if case_index not in selected_cases:
            continue
        record = records.get(case_index, {})
        completed = bool(record)
        benchmark_success = record.get("benchmark_success")
        success = (
            benchmark_success
            if isinstance(benchmark_success, bool)
            else record.get("program_success")
        )
        card_class = (
            "success" if success is True else "failure" if completed else "pending"
        )
        goal = (case_dir / "goal.txt").read_text(encoding="utf-8").strip()
        case_name = f"case_{case_index:03d}"
        trajectory_link = f"{case_name}/artifacts/trajectory.mp4"
        metrics_link = f"{case_name}/artifacts/metrics.json"
        states_link = f"{case_name}/artifacts/state_trajectory.npz"
        binding_link = f"{case_name}/artifacts/instantiation/binding.json"
        benchmark_evaluation = record.get("benchmark_evaluation")
        evaluation_source = (
            benchmark_evaluation.get("source", "pending")
            if isinstance(benchmark_evaluation, dict)
            else "pending"
        )
        states_detail = (
            f'<a href="{states_link}">state_trajectory.npz</a>'
            if record.get("state_trajectory_path")
            else "not retained"
        )
        metrics_detail = (
            f'<a href="{metrics_link}">metrics.json</a>'
            if record.get("metrics_available") is True
            else "not produced"
        )
        binding_detail = (
            f'<a href="{binding_link}">binding.json</a>'
            if record.get("binding_written")
            else "not produced"
        )
        trajectory_detail = (
            '<video controls autoplay muted loop playsinline preload="metadata" '
            'data-pending="true" '
            f'data-trajectory-src="{trajectory_link}"></video>'
            '<p class="availability">Loading trajectory...</p>'
            if record.get("trajectory_path")
            else (
                '<div class="no-trajectory">No trajectory retained.</div>'
                '<p class="availability">The run stopped before a video was produced.</p>'
                if completed
                else (
                    '<div class="no-trajectory">Pending evaluation.</div>'
                    '<p class="availability">No worker result yet.</p>'
                )
            )
        )
        execution_status = (
            "complete"
            if record.get("execution_completed") is True
            else "stopped early"
            if record.get("execution_started") is True
            else "pending"
        )
        failure_stage = record.get("failure_stage") or "none"
        failure_segment = record.get("failure_segment") or "none"
        failure_message = record.get("failure_message") or "none"
        cards.append(
            f'<article class="{card_class}">'
            f"<h2>Case {case_index:03d}</h2>"
            f"{trajectory_detail}"
            f"<p>{html.escape(goal)}</p>"
            "<dl>"
            "<dt>Bucket</dt><dd>"
            f"{html.escape(str(record.get('bucket', 'pending')))}</dd>"
            f"<dt>Binding valid</dt><dd>{record.get('binding_valid', 'pending')}</dd>"
            "<dt>Benchmark success</dt><dd>"
            f"{record.get('benchmark_success', 'pending')}</dd>"
            "<dt>Benchmark evidence</dt><dd>"
            f"{html.escape(str(evaluation_source))}</dd>"
            "<dt>Strategy predicate</dt><dd>"
            f"{record.get('program_success', 'pending')}</dd>"
            f"<dt>Execution</dt><dd>{html.escape(execution_status)}</dd>"
            f"<dt>Failure stage</dt><dd>{html.escape(str(failure_stage))}</dd>"
            f"<dt>Failure segment</dt><dd>{html.escape(str(failure_segment))}</dd>"
            f"<dt>Failure reason</dt><dd>{html.escape(str(failure_message))}</dd>"
            f"<dt>Metrics</dt><dd>{metrics_detail}</dd>"
            f"<dt>States</dt><dd>{states_detail}</dd>"
            f"<dt>Binding</dt><dd>{binding_detail}</dd>"
            "</dl></article>"
        )
    aggregate = (
        _summary(list(records.values()))
        if records_collected_from_cases
        else payload.get("summary", {})
    )
    successes = aggregate.get("successes", 0)
    total = aggregate.get("total", 0)
    source_name = Path(str(eval_manifest["source_suite"])).name
    eval_name = Path(str(eval_manifest["eval_suite"])).name
    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(scene_set.name)} generalization evaluation</title>
<style>
:root{{color-scheme:dark}}body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
.meta{{color:#bbb}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:18px}}
article{{background:#222;padding:14px;border:1px solid #444;border-radius:9px}}
article.success{{border-color:#39734a}}article.failure{{border-color:#8e4545}}
video{{width:100%;height:auto;background:#080808}}.no-trajectory{{display:grid;place-items:center;aspect-ratio:16/9;background:#181818;color:#999}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:5px 12px}}
dt{{color:#bbb}}dd{{margin:0;overflow-wrap:anywhere}}a{{color:#8cc8ff}}.availability{{color:#bbb}}
</style></head><body>
<h1>{html.escape(scene_set.name)} generalization evaluation</h1>
<p class="meta">Source suite: {html.escape(source_name)} · Eval suite: {html.escape(eval_name)} ·
binding mode: {html.escape(binding_mode)} · complete: {total}/{len(selected_cases)} · success: {successes}/{total}</p>
<p class="meta">This page reserves {len(selected_cases)} selected result slots. Retained videos are retried every 30 seconds while loading.</p>
<div class="grid">{''.join(cards)}</div>
<script>
function retryPendingVideos() {{
  document.querySelectorAll('video[data-pending="true"]').forEach((video) => {{
    const status = video.nextElementSibling;
    video.onloadedmetadata = () => {{ video.dataset.pending = 'false'; status.textContent = 'Trajectory available.'; }};
    video.onerror = () => {{ status.textContent = 'Waiting for trajectory...'; }};
    video.src = video.dataset.trajectorySrc + '?retry=' + Date.now();
    video.load();
  }});
}}
retryPendingVideos(); setInterval(retryPendingVideos, 30000);
</script></body></html>
"""
    gallery = output_root / "trajectory_gallery.html"
    gallery.write_text(document, encoding="utf-8")
    return gallery


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _clean_case_state(
    *,
    root: Path,
    workspace: Path,
    output_root: Path,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    case_index: int,
) -> None:
    """Remove only an incomplete case's state before retrying it."""

    case_name = f"case_{case_index:03d}"
    run_dir = output_root / case_name
    if (run_dir / "benchmark_result.json").is_file():
        return
    _remove_path(run_dir)
    instance = f"benchmark_{case_index:03d}"
    _remove_path(
        workspace
        / "src"
        / "instantiations"
        / "strategies"
        / strategy
        / instance
    )
    _remove_path(
        workspace
        / "results_instantiation"
        / "strategies"
        / strategy
        / instance
    )
    task_copy = workspace / "task_inputs" / scene_set.cases[case_index].relative_to(
        root
    )
    _remove_path(task_copy)


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux fallback
        return os.cpu_count() or 1


def _parallel_rollout_workers(args: argparse.Namespace) -> int:
    if args.rollout_workers_per_job is not None:
        if args.rollout_workers_per_job < 1:
            raise ValueError("--rollout-workers-per-job must be positive")
        return args.rollout_workers_per_job
    return max(1, min(48, _available_cpu_count() // args.jobs))


def _prepare_parallel_workers(
    *,
    root: Path,
    eval_run: Path,
    eval_workspace: Path,
    output_root: Path,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    checkpoint_sha256: str,
    pending: Sequence[int],
    jobs: int,
) -> list[tuple[int, Path, Path]]:
    """Create independent reusable suite workspaces for parallel cases."""

    for case_index in pending:
        _clean_case_state(
            root=root,
            workspace=eval_workspace,
            output_root=output_root,
            scene_set=scene_set,
            strategy=strategy,
            case_index=case_index,
        )
    workers: list[tuple[int, Path, Path]] = []
    workers_root = (
        eval_run / "benchmark_workers" / scene_set.name / output_root.name
    )
    for worker_index in range(1, jobs + 1):
        worker_run = workers_root / f"worker_{worker_index:02d}"
        try:
            _, workspace = resolve_suite_workspace(root, str(worker_run))
        except ValueError:
            if worker_run.exists():
                shutil.rmtree(worker_run)
            created = create_isolated_workspace(
                worker_run,
                repo_root=root,
                from_workspace=eval_workspace,
            )
            workspace = created.workspace
        if _frozen_checkpoint_digest(workspace / "src", strategy) != checkpoint_sha256:
            raise RuntimeError(
                f"parallel worker {worker_index} changed the frozen checkpoint"
            )
        for case_index in pending:
            _clean_case_state(
                root=root,
                workspace=workspace,
                output_root=output_root,
                scene_set=scene_set,
                strategy=strategy,
                case_index=case_index,
            )
        workers.append((worker_index, worker_run, workspace))
    return workers


def _runner_command(
    *,
    suite_run: Path,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    output_root: Path,
    args: argparse.Namespace,
    case_index: int | None = None,
    no_summary: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "suites.staged.benchmark_runner",
        str(suite_run),
        "--strategy-id",
        str(scene_set.strategy_id),
        "--strategy",
        strategy,
        "--scene-root",
        str(scene_set.root),
        "--seal-root",
        str(args.seal_root),
        "--binding-mode",
        args.binding_mode,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--execution-policy",
        args.execution_policy,
        "--output",
        str(output_root),
        "--workspace-mode",
        "direct",
    ]
    if case_index is None:
        for selection in args.case or []:
            command.extend(("--case", selection))
    else:
        command.extend(("--case", str(case_index)))
    if args.resume and case_index is None:
        command.append("--resume")
    if args.model:
        command.extend(("--model", args.model))
    if args.thinking_effort:
        command.extend(("--thinking-effort", args.thinking_effort))
    if args.max_turns is not None:
        command.extend(("--max-turns", str(args.max_turns)))
    if args.no_record:
        command.append("--no-record")
    if no_summary:
        command.append("--no-summary")
    if not args.strict_program_success:
        command.append("--allow-program-failure")
    return command


def _collect_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("case_*/benchmark_result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not isinstance(
            record.get("case_index"), int
        ):
            continue
        metrics_path = Path(str(record.get("metrics_path", "")))
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        strategy_id = record.get("strategy_id")
        if isinstance(strategy_id, int):
            refreshed = _attach_benchmark_evaluation(
                record,
                strategy_id,
                metrics,
            )
            if refreshed != record:
                temporary = path.with_name(
                    f".{path.name}.{os.getpid()}.tmp"
                )
                temporary.write_text(
                    json.dumps(refreshed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(path)
            record = refreshed
        records.append(record)
    return sorted(records, key=lambda item: int(item["case_index"]))


def _write_aggregate_summary(
    *,
    output_root: Path,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    binding_mode: str,
    checkpoint_sha256: str,
    jobs: int,
    rollout_workers_per_job: int,
    execution_policy: str = DEFAULT_EXECUTION_POLICY,
) -> list[dict[str, Any]]:
    records = _collect_records(output_root)
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "strategy_id": scene_set.strategy_id,
        "strategy": strategy,
        "scene_root": str(scene_set.root),
        "binding_mode": binding_mode,
        "checkpoint_sha256": checkpoint_sha256,
        "execution_policy": execution_policy,
        "optimization_budget": {
            "mode": "maximum",
            "early_stopping": "primitive_policy",
        },
        "parallelism": {
            "jobs": jobs,
            "rollout_workers_per_job": rollout_workers_per_job,
        },
        "cases": records,
        "summary": _summary(records),
    }
    temporary = output_root / f".summary.json.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root / "summary.json")
    return records


def _write_eval_result_summary(
    *,
    eval_run: Path,
    eval_manifest: dict[str, Any],
    scene_set: BenchmarkSceneSet,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Publish the final task and binding rates in ``benchmark_eval.json``."""

    summary = _summary(list(records))
    evaluated_cases = int(summary["total"])
    selected_cases = eval_manifest.get("selected_cases")
    if not isinstance(selected_cases, list):
        selected_cases = sorted(scene_set.cases)
    expected_cases = len(selected_cases)
    completed = (
        {int(item["case_index"]) for item in records}
        == {int(index) for index in selected_cases}
    )
    timestamp = datetime.now(UTC).isoformat()
    payload = dict(eval_manifest)
    payload.update(
        {
            "results_updated_at": timestamp,
            "evaluation_completed": completed,
            "evaluated_cases": evaluated_cases,
            "expected_cases": expected_cases,
            "task_successes": int(summary["successes"]),
            "task_success_rate": summary["success_rate"],
            "forced_continuation_cases": int(
                summary["forced_continuation_cases"]
            ),
            "successes_requiring_forced_continuation": int(
                summary["successes_requiring_forced_continuation"]
            ),
            "binding_successes": sum(
                item.get("binding_valid") is True for item in records
            ),
            "binding_success_rate": summary["binding_valid_rate"],
        }
    )
    if completed:
        payload["completed_at"] = timestamp
    else:
        payload.pop("completed_at", None)
    path = eval_run / EVAL_MANIFEST_NAME
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def _stop_running_cases(running: dict[int, RunningCase]) -> None:
    for item in running.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    for item in running.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        finally:
            item.log_file.close()


def _run_parallel(
    *,
    root: Path,
    eval_run: Path,
    eval_workspace: Path,
    output_root: Path,
    scene_set: BenchmarkSceneSet,
    strategy: str,
    eval_manifest: dict[str, Any],
    args: argparse.Namespace,
    environment: dict[str, str],
) -> int:
    selected = _case_indices(args.case)
    completed = {
        int(item["case_index"]) for item in _collect_records(output_root)
    }
    pending = [index for index in selected if index not in completed]
    if not args.resume and completed:
        raise FileExistsError(
            f"benchmark output already contains cases {sorted(completed)}; "
            "pass --resume"
        )
    jobs = min(args.jobs, len(pending)) if pending else args.jobs
    rollout_workers = _parallel_rollout_workers(args)
    environment = environment.copy()
    environment["RAPID_MAX_PARALLEL_ROLLOUTS"] = str(rollout_workers)
    records = _write_aggregate_summary(
        output_root=output_root,
        scene_set=scene_set,
        strategy=strategy,
        binding_mode=args.binding_mode,
        checkpoint_sha256=str(eval_manifest["checkpoint_sha256"]),
        jobs=jobs,
        rollout_workers_per_job=rollout_workers,
        execution_policy=args.execution_policy,
    )
    write_gallery(
        scene_set=scene_set,
        output_root=output_root,
        eval_manifest=eval_manifest,
        binding_mode=args.binding_mode,
    )
    if not pending:
        return _records_returncode(
            records,
            selected=selected,
            strict=args.strict_program_success,
        )

    workers = _prepare_parallel_workers(
        root=root,
        eval_run=eval_run,
        eval_workspace=eval_workspace,
        output_root=output_root,
        scene_set=scene_set,
        strategy=strategy,
        checkpoint_sha256=str(eval_manifest["checkpoint_sha256"]),
        pending=pending,
        jobs=jobs,
    )
    logs_root = output_root / "_worker_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    available = list(workers)
    queue = list(pending)
    running: dict[int, RunningCase] = {}
    print(
        f"[rapid-benchmark] parallel jobs: {jobs}; "
        f"rollout workers/job: {rollout_workers}; pending cases: {len(pending)}",
        flush=True,
    )
    try:
        while queue or running:
            while queue and available:
                worker_index, worker_run, _ = available.pop(0)
                case_index = queue.pop(0)
                log_path = logs_root / f"worker_{worker_index:02d}.log"
                log_file = log_path.open("a", encoding="utf-8", buffering=1)
                log_file.write(f"\n===== case_{case_index:03d} =====\n")
                command = _runner_command(
                    suite_run=worker_run,
                    scene_set=scene_set,
                    strategy=strategy,
                    output_root=output_root,
                    args=args,
                    case_index=case_index,
                    no_summary=True,
                )
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                running[worker_index] = RunningCase(
                    case_index=case_index,
                    worker_index=worker_index,
                    worker_run=worker_run,
                    process=process,
                    log_path=log_path,
                    log_file=log_file,
                )
                print(
                    f"[rapid-benchmark] worker {worker_index:02d} started "
                    f"case {case_index:03d}",
                    flush=True,
                )
            finished = [
                worker_index
                for worker_index, item in running.items()
                if item.process.poll() is not None
            ]
            if not finished:
                time.sleep(1.0)
                continue
            for worker_index in finished:
                item = running.pop(worker_index)
                returncode = int(item.process.returncode or 0)
                item.log_file.close()
                result_path = (
                    output_root
                    / f"case_{item.case_index:03d}"
                    / "benchmark_result.json"
                )
                result_status = "recorded" if result_path.is_file() else "missing"
                print(
                    f"[rapid-benchmark] worker {worker_index:02d} finished "
                    f"case {item.case_index:03d}: exit={returncode}, "
                    f"result={result_status}, log={item.log_path}",
                    flush=True,
                )
                available.append(
                    (
                        worker_index,
                        item.worker_run,
                        item.worker_run / "workspace",
                    )
                )
                records = _write_aggregate_summary(
                    output_root=output_root,
                    scene_set=scene_set,
                    strategy=strategy,
                    binding_mode=args.binding_mode,
                    checkpoint_sha256=str(eval_manifest["checkpoint_sha256"]),
                    jobs=jobs,
                    rollout_workers_per_job=rollout_workers,
                    execution_policy=args.execution_policy,
                )
                write_gallery(
                    scene_set=scene_set,
                    output_root=output_root,
                    eval_manifest=eval_manifest,
                    binding_mode=args.binding_mode,
                )
    except KeyboardInterrupt:
        print("[rapid-benchmark] stopping parallel workers", file=sys.stderr)
        _stop_running_cases(running)
        raise
    records = _collect_records(output_root)
    return _records_returncode(
        records,
        selected=selected,
        strict=args.strict_program_success,
    )


def _records_returncode(
    records: Sequence[dict[str, Any]],
    *,
    selected: Sequence[int],
    strict: bool,
) -> int:
    by_index = {int(item["case_index"]): item for item in records}
    if any(index not in by_index for index in selected):
        return 1
    if any(by_index[index].get("binding_valid") is not True for index in selected):
        return 1
    if strict and any(
        (
            by_index[index].get("benchmark_success")
            if isinstance(by_index[index].get("benchmark_success"), bool)
            else by_index[index].get("program_success")
        )
        is not True
        for index in selected
    ):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-benchmark",
        description=(
            "Copy a learned suite run into a disposable eval suite, execute its "
            "strategy on one complete benchmark scene set, and build a video gallery."
        ),
    )
    parser.add_argument(
        "source_suite", help="Suite-run ID, run path, or workspace path"
    )
    parser.add_argument(
        "strategy_scenes",
        help=(
            "Strategy scene directory, task module, or strategy_N alias; for "
            "example tasks/sim_benchmark/strategy_2"
        ),
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help=(
            "Destination suite ID/path; defaults to SOURCE_SUITE_eval for "
            "strict runs or SOURCE_SUITE_eval_attempt for attempt runs"
        ),
    )
    parser.add_argument(
        "--seal-root",
        type=Path,
        default=None,
        help=(
            "Private seal/manifests root matching the selected scene set; "
            "defaults to the production sealed benchmark."
        ),
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Learned strategy package; defaults to the strategy scene directory name",
    )
    parser.add_argument(
        "--binding-mode",
        choices=("agent", "authoritative"),
        default="agent",
    )
    parser.add_argument(
        "--gpu-render",
        choices=GPU_RENDER_MODES,
        default="required",
        help="GPU rendering policy (default: required)",
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of cases to evaluate concurrently in independent worker "
            "workspaces (default: 1)"
        ),
    )
    parser.add_argument(
        "--rollout-workers-per-job",
        type=int,
        default=None,
        help=(
            "CPU rollout threads per parallel case; defaults to the available "
            "CPU count divided by --jobs, capped at 48"
        ),
    )
    parser.add_argument("--optimizer", choices=("cem", "mppi"), default="cem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution-policy",
        choices=EXECUTION_POLICIES,
        default=DEFAULT_EXECUTION_POLICY,
        help=(
            "Continue from finite rejected transit trajectories in benchmark "
            "attempt mode (default), or strictly stop on execution rejection."
        ),
    )
    parser.add_argument(
        "--retry-recoverable-from",
        help=(
            "Select only failed cases whose prior strict eval retained a rejected "
            "transit trajectory. Accepts an eval-suite ID or path and requires "
            "--execution-policy attempt."
        ),
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--thinking-effort",
        choices=("low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--record-width", type=int, default=480)
    parser.add_argument("--record-height", type=int, default=270)
    parser.add_argument("--record-fps", type=int, default=10)
    parser.add_argument("--record-playback-speed", type=float, default=2.0)
    parser.add_argument(
        "--gallery-only",
        action="store_true",
        help=(
            "Rescore retained cases and refresh an existing eval suite's "
            "summary and gallery without rerunning simulation"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the eval-suite copy and gallery without running cases",
    )
    parser.add_argument(
        "--strict-program-success",
        action="store_true",
        help=(
            "Return nonzero when any case fails its benchmark-owned success "
            "function"
        ),
    )
    add_experiment_root_argument(parser)
    return parser


def _positive_recording_values(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "RAPID_RECORD_WIDTH": args.record_width,
        "RAPID_RECORD_HEIGHT": args.record_height,
        "RAPID_RECORD_FPS": args.record_fps,
        "RAPID_RECORD_PLAYBACK_SPEED": args.record_playback_speed,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return {name: str(value) for name, value in values.items()}


def run_evaluation(args: argparse.Namespace, *, root: Path) -> int:
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if (
        args.retry_recoverable_from
        and args.execution_policy != ATTEMPT_EXECUTION_POLICY
    ):
        raise ValueError(
            "--retry-recoverable-from requires --execution-policy attempt"
        )
    if args.retry_recoverable_from and args.case:
        raise ValueError("--retry-recoverable-from cannot be combined with --case")
    scene_set = resolve_scene_set(root, args.strategy_scenes)
    args.seal_root = _resolve_seal_root(root, args.seal_root)
    strategy = args.strategy or scene_set.name
    source_run, _ = resolve_suite_workspace(root, args.source_suite)
    recoverable_selection_source = None
    if args.retry_recoverable_from:
        recoverable_selection_source = str(
            _resolve_eval_run_reference(root, args.retry_recoverable_from)
        )
        recoverable = recoverable_case_indices(
            root=root,
            prior_eval_suite=args.retry_recoverable_from,
            scene_set=scene_set,
            strategy=strategy,
            binding_mode=args.binding_mode,
            source_run=source_run,
        )
        if not recoverable:
            print(
                "[rapid-benchmark] no recoverable rejected-trajectory cases "
                f"for {strategy}",
                flush=True,
            )
            return 0
        args.case = [",".join(str(index) for index in recoverable)]
        print(
            "[rapid-benchmark] recoverable case selection: "
            + ",".join(f"{index:03d}" for index in recoverable),
            flush=True,
        )
    selected_cases = _case_indices(args.case)
    eval_run = _eval_run_dir(
        root,
        source_run,
        args.eval_suite,
        execution_policy=args.execution_policy,
    )
    if args.gallery_only:
        if not eval_run.is_dir():
            raise FileNotFoundError(f"eval suite does not exist: {eval_run}")
        eval_run, eval_workspace, manifest = prepare_eval_suite(
            root=root,
            source_suite=args.source_suite,
            eval_suite=args.eval_suite,
            scene_set=scene_set,
            seal_root=args.seal_root,
            strategy=strategy,
            resume=True,
            execution_policy=args.execution_policy,
            selected_cases=selected_cases,
            recoverable_selection_source=recoverable_selection_source,
        )
    else:
        eval_run, eval_workspace, manifest = prepare_eval_suite(
            root=root,
            source_suite=args.source_suite,
            eval_suite=args.eval_suite,
            scene_set=scene_set,
            seal_root=args.seal_root,
            strategy=strategy,
            resume=args.resume,
            execution_policy=args.execution_policy,
            selected_cases=selected_cases,
            recoverable_selection_source=recoverable_selection_source,
        )
    output_root = (
        eval_run / "benchmark_runs" / scene_set.name / args.binding_mode
    )
    if args.resume and not args.gallery_only:
        archived = _archive_stale_case_results(output_root, scene_set)
        if archived:
            print(
                "[rapid-benchmark] archived "
                f"{len(archived)} stale case result(s) under "
                f"{output_root / '_superseded'}",
                flush=True,
            )
    if args.gallery_only:
        retained = _read_summary(output_root)
        parallelism = retained.get("parallelism", {})
        if not isinstance(parallelism, dict):
            parallelism = {}
        jobs = parallelism.get("jobs", args.jobs)
        if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
            jobs = args.jobs
        rollout_workers = parallelism.get(
            "rollout_workers_per_job",
            _parallel_rollout_workers(args),
        )
        if (
            not isinstance(rollout_workers, int)
            or isinstance(rollout_workers, bool)
            or rollout_workers < 1
        ):
            rollout_workers = _parallel_rollout_workers(args)
        records = _write_aggregate_summary(
            output_root=output_root,
            scene_set=scene_set,
            strategy=strategy,
            binding_mode=args.binding_mode,
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            jobs=jobs,
            rollout_workers_per_job=rollout_workers,
            execution_policy=args.execution_policy,
        )
        manifest = _write_eval_result_summary(
            eval_run=eval_run,
            eval_manifest=manifest,
            scene_set=scene_set,
            records=records,
        )
    gallery = write_gallery(
        scene_set=scene_set,
        output_root=output_root,
        eval_manifest=manifest,
        binding_mode=args.binding_mode,
    )
    print(f"[rapid-benchmark] eval suite: {eval_run}", flush=True)
    print(f"[rapid-benchmark] gallery: {gallery}", flush=True)
    if args.gallery_only or args.prepare_only:
        return 0

    environment = os.environ.copy()
    environment["RAPID_REPO_ROOT"] = str(root)
    environment["RAPID_GPU_RENDER"] = args.gpu_render
    environment[EXECUTION_POLICY_ENV] = args.execution_policy
    environment.update(_positive_recording_values(args))
    if args.jobs > 1:
        returncode = _run_parallel(
            root=root,
            eval_run=eval_run,
            eval_workspace=eval_workspace,
            output_root=output_root,
            scene_set=scene_set,
            strategy=strategy,
            eval_manifest=manifest,
            args=args,
            environment=environment,
        )
        manifest = _write_eval_result_summary(
            eval_run=eval_run,
            eval_manifest=manifest,
            scene_set=scene_set,
            records=_collect_records(output_root),
        )
        gallery = write_gallery(
            scene_set=scene_set,
            output_root=output_root,
            eval_manifest=manifest,
            binding_mode=args.binding_mode,
        )
        print(f"[rapid-benchmark] gallery refreshed: {gallery}", flush=True)
        return returncode

    command = _runner_command(
        suite_run=eval_run,
        scene_set=scene_set,
        strategy=strategy,
        output_root=output_root,
        args=args,
    )
    if args.rollout_workers_per_job is not None:
        environment["RAPID_MAX_PARALLEL_ROLLOUTS"] = str(
            _parallel_rollout_workers(args)
        )
    print(f"[rapid-benchmark] $ {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=root, env=environment, check=False)
    manifest = _write_eval_result_summary(
        eval_run=eval_run,
        eval_manifest=manifest,
        scene_set=scene_set,
        records=_collect_records(output_root),
    )
    gallery = write_gallery(
        scene_set=scene_set,
        output_root=output_root,
        eval_manifest=manifest,
        binding_mode=args.binding_mode,
    )
    print(f"[rapid-benchmark] gallery refreshed: {gallery}", flush=True)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_experiment_root(args, repo_root_from_env().resolve())
    try:
        return run_evaluation(args, root=repo_root_from_env().resolve())
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[rapid-benchmark] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
