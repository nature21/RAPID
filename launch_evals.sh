#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: ./launch_evals.sh [--experiment-root DIR] [--after-array ID | --after ID] [--dry-run] [BATCH_ID]

Submit one evaluation array, indexed by paper strategy ID (1-8).
Each element requests 32 CPU, 1 GPU, 64G and evaluates 50 cases with 4 workers.
--after-array ID: aftercorr dependency on matching strategy array elements.
--after ID: afterok dependency on an entire upstream job/array (legacy use).
Without a dependency, all selected source strategy workspaces must exist.
Slurm handles resource waits; --once is retained as a no-op compatibility flag.

Environment:
  RAPID_EVAL_NODES       Eligible nodes (default: crane0 crane1 crane2)
  RAPID_EVAL_STRATEGIES  Space/comma-separated paper IDs (default: 1-8)
  RAPID_EVAL_WORKERS     Concurrent cases per element (default: 4)
  RAPID_EVAL_EXECUTION_POLICY  attempt (default) or strict
  RAPID_EVAL_RETRY_RECOVERABLE_FROM  Prior strict suite, or auto for each task

Evaluation uses nice=10000; invention uses nice=0. This affects pending job
priority and does not preempt running jobs. Default BATCH_ID is 002.
EOF
}
die() { echo "error: $*" >&2; exit 2; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$script_dir"
source "${repo_root}/scripts/lib/experiment-root.sh"
source "${repo_root}/scripts/lib/slurm-arrays.sh"
rapid_experiment_root_args "$@"
set -- "${RAPID_EXPERIMENT_ARGS[@]}"
readonly experiment_root="$RAPID_EXPERIMENT_ROOT"
batch_id=002
batch_seen=0
dry_run=0
after_job_id=""
dependency_kind=""
while (($#)); do
  case "$1" in
    --after|--after-array)
      [[ -z "$after_job_id" ]] || die "supply only one dependency"
      (($# >= 2)) && is_positive_integer "$2" || die "$1 requires a positive numeric Slurm job ID"
      dependency_kind=afterok
      [[ "$1" != --after-array ]] || dependency_kind=aftercorr
      after_job_id="$2"; shift ;;
    --dry-run) dry_run=1 ;;
    --once) : ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) ((batch_seen == 0)) || die "only one BATCH_ID may be supplied"
       batch_id="$1"; batch_seen=1 ;;
  esac
  shift
done
[[ "$batch_id" =~ ^[0-9]{3}$ ]] || die "BATCH_ID must be exactly three digits"
workers="${RAPID_EVAL_WORKERS:-4}"
is_positive_integer "$workers" && ((workers <= 32)) || die "RAPID_EVAL_WORKERS must be between 1 and 32"
execution_policy="${RAPID_EVAL_EXECUTION_POLICY:-attempt}"
[[ "$execution_policy" == strict || "$execution_policy" == attempt ]] || die "invalid eval execution policy"
strategy_ids="$(rapid_strategy_ids "${RAPID_EVAL_STRATEGIES:-1 2 3 4 5 6 7 8}")"
read -r -a strategies <<< "${strategy_ids//,/ }"
# Validate the eligible pool without polling free resources.
rapid_slurm_task_args "${RAPID_EVAL_NODES:-crane0 crane1 crane2}"
resolve_eval_reference() {
  local value="$1"
  local candidate
  if [[ "$value" == /* ]]; then
    candidate="$value"
  elif [[ "$value" != */* ]]; then
    candidate="${experiment_root}/${value}"
  else
    candidate="${repo_root}/${value}"
  fi
  if [[ "$(basename -- "$candidate")" == "workspace" ]]; then
    candidate="$(dirname -- "$candidate")"
  fi
  printf '%s\n' "$candidate"
}

json_matches() {
  local pattern="$1"
  local path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q "$pattern" "$path"
  else
    grep -Eq "$pattern" "$path"
  fi
}

manifest_selection_source() {
  sed -nE \
    's/^[[:space:]]*"recoverable_selection_source":[[:space:]]*"([^"]*)",?[[:space:]]*$/\1/p' \
    "$1"
}

retry_reference_for_run() {
  local run_id="$1"
  local attempt_manifest selection_source sibling_manifest
  if [[ -n "${RAPID_EVAL_RETRY_RECOVERABLE_FROM+x}" ]]; then
    if [[ "$RAPID_EVAL_RETRY_RECOVERABLE_FROM" == "auto" ]]; then
      printf '%s_eval\n' "$run_id"
    else
      printf '%s\n' "$RAPID_EVAL_RETRY_RECOVERABLE_FROM"
    fi
    return
  fi
  attempt_manifest="${experiment_root}/${run_id}_eval_attempt/benchmark_eval.json"
  if [[ -f "$attempt_manifest" ]]; then
    selection_source="$(manifest_selection_source "$attempt_manifest")"
    if [[ -n "$selection_source" ]]; then
      printf '%s\n' "$selection_source"
      return
    fi
  fi
  for sibling_manifest in \
    "${experiment_root}"/fr3_osc_demo_*_"${batch_id}"_eval_attempt/benchmark_eval.json; do
    [[ -f "$sibling_manifest" ]] || continue
    if [[ -n "$(manifest_selection_source "$sibling_manifest")" ]]; then
      printf '%s_eval\n' "$run_id"
      return
    fi
  done
  printf '\n'
}

task_is_complete() {
  local strategy_index="$1"
  local run_id scene_root result_root expected completed retry_reference
  local prior_eval prior_result_root result_path metrics_path case_name
  run_id="fr3_osc_demo_${strategy_index}_${batch_id}"
  scene_root="${repo_root}/tasks/sim_benchmark/strategy_${strategy_index}"
  if [[ "$execution_policy" == "attempt" ]]; then
    result_root="${experiment_root}/${run_id}_eval_attempt/benchmark_runs/strategy_${strategy_index}/agent"
    retry_reference="$(retry_reference_for_run "$run_id")"
  else
    result_root="${experiment_root}/${run_id}_eval/benchmark_runs/strategy_${strategy_index}/agent"
    retry_reference=""
  fi

  if [[ "$execution_policy" == "attempt" && -n "$retry_reference" ]]; then
    prior_eval="$(resolve_eval_reference "$retry_reference")"
    prior_result_root="${prior_eval}/benchmark_runs/strategy_${strategy_index}/agent"
    [[ -d "$prior_result_root" ]] || return 1
    expected=0
    completed=0
    while IFS= read -r result_path; do
      metrics_path="${result_path%/benchmark_result.json}/artifacts/metrics.json"
      if [[ -f "$metrics_path" ]] && \
          json_matches '"benchmark_success"[[:space:]]*:[[:space:]]*false' "$result_path" && \
          json_matches '"failure_type"[[:space:]]*:[[:space:]]*"FR3TransitViolation"' "$metrics_path" && \
          json_matches '__rejected"' "$metrics_path"; then
        expected=$((expected + 1))
        case_name="$(basename -- "${result_path%/benchmark_result.json}")"
        if [[ -f "${result_root}/${case_name}/benchmark_result.json" ]]; then
          completed=$((completed + 1))
        fi
      fi
    done < <(
      find "$prior_result_root" -mindepth 2 -maxdepth 2 \
        -type f -name benchmark_result.json | sort
    )
    # A task with no recoverable strict failures needs no attempt job.
    ((expected > 0)) || return 0
    ((completed >= expected))
    return
  else
    expected="$(find "$scene_root" -mindepth 2 -maxdepth 2 -type f -name benchmark_case.json | wc -l)"
    [[ "$expected" =~ ^[1-9][0-9]*$ ]] || return 1
  fi
  [[ -d "$result_root" ]] || return 1
  completed="$(find "$result_root" -mindepth 2 -maxdepth 2 -type f -name benchmark_result.json | wc -l)"
  ((completed >= expected))
}


cd "$repo_root"
mkdir -p eval_logs
root_digest="$(printf '%s' "$experiment_root" | sha256sum)"
job_name="rapid-eval-${root_digest:0:12}-${batch_id}-${execution_policy}"
declare -A active=()
queue="$(squeue --array --noheader --user "${USER:-$(id -un)}" --states=PENDING,RUNNING,CONFIGURING,COMPLETING --format='%i|%j')"
while IFS='|' read -r job_id queued_name; do
  [[ "$queued_name" == "$job_name" && "$job_id" =~ _([1-8])$ ]] || continue
  active[${BASH_REMATCH[1]}]=1
done <<< "$queue"
selected=()
for strategy_index in "${strategies[@]}"; do
  run_id="fr3_osc_demo_${strategy_index}_${batch_id}"
  [[ -n "$after_job_id" || -d "$experiment_root/$run_id" ]] || die "source suite does not exist: $experiment_root/$run_id"
  [[ -d "$repo_root/tasks/sim_benchmark/strategy_${strategy_index}" ]] || die "benchmark scene set does not exist: strategy_${strategy_index}"
  if [[ "$execution_policy" == attempt ]]; then
    retry_reference="$(retry_reference_for_run "$run_id")"
    if [[ -n "$retry_reference" ]]; then
      prior_eval="$(resolve_eval_reference "$retry_reference")"
      [[ -f "$prior_eval/benchmark_eval.json" ]] || die "prior strict eval suite does not exist: $prior_eval"
    fi
  fi
  if [[ -z "${active[$strategy_index]:-}" ]] && ! task_is_complete "$strategy_index"; then
    selected+=("$strategy_index")
  fi
done
if ((${#selected[@]} == 0)); then
  echo "[launch_evals] every requested eval is complete or already submitted; queued/planned evals=0"
  exit 0
fi
# Preserve automatic per-task retry selection when resuming a targeted batch.
if [[ "$execution_policy" == attempt && -z "${RAPID_EVAL_RETRY_RECOVERABLE_FROM+x}" ]]; then
  for strategy_index in "${selected[@]}"; do
    if [[ -n "$(retry_reference_for_run "fr3_osc_demo_${strategy_index}_${batch_id}")" ]]; then
      export RAPID_EVAL_RETRY_RECOVERABLE_DEFAULT=auto
      break
    fi
  done
fi
export BATCH_ID="$batch_id" RAPID_REPO_ROOT="$repo_root"
export RAPID_EVAL_WORKERS="$workers" RAPID_EVAL_EXECUTION_POLICY="$execution_policy"
command=(sbatch --parsable "${RAPID_SLURM_TASK_ARGS[@]}" --nice=10000
  "--array=$(IFS=,; echo "${selected[*]}")" "--job-name=$job_name"
  "--output=$repo_root/eval_logs/${job_name}_%A_%a.log" --export=ALL)
if [[ -n "$after_job_id" ]]; then
  command+=("--dependency=${dependency_kind}:${after_job_id}")
fi
command+=("$repo_root/run_eval.sh" --experiment-root "$experiment_root")
echo "[launch_evals] root=$experiment_root; strategies=${selected[*]}; request=32cpu/64G/1gpu; nice=10000"
if ((dry_run)); then
  printf '[launch_evals] dry-run:'; printf ' %q' "${command[@]}"; printf '\n'
else
  job_id="$("${command[@]}")"; job_id="${job_id%%;*}"
  is_positive_integer "$job_id" || die "invalid eval array job ID: $job_id"
  echo "[launch_evals] eval array submitted: $job_id"
fi
echo "[launch_evals] dependency=${dependency_kind:-none}:${after_job_id}; queued/planned evals=${#selected[@]}; dry_run=$dry_run"
