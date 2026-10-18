#!/usr/bin/env bash
#SBATCH --job-name=run_eval
#SBATCH --output=eval_logs/eval_output_%A_%a.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nice=10000
set -euo pipefail

usage() {
  echo "usage: sbatch $0 [--experiment-root DIR] RUN_ID STRATEGY_NAME [NUM_WORKERS]" >&2
  echo "DIR defaults to RAPID_EXPERIMENT_ROOT or repository suite_runs." >&2
  echo "example: sbatch $0 fr3_osc_demo_1_001 strategy_1 4" >&2
  echo "RAPID_EVAL_EXECUTION_POLICY defaults to attempt." >&2
  echo "Attempt mode runs all cases unless RAPID_EVAL_RETRY_RECOVERABLE_FROM is set." >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm runs a private script copy; resolve helpers from the submission repo.
repo_root="${RAPID_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
repo_root="$(cd -- "$repo_root" && pwd)"
source "${repo_root}/scripts/lib/experiment-root.sh"
rapid_experiment_root_args "$@"
set -- "${RAPID_EXPERIMENT_ARGS[@]}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-8]$ && "${BATCH_ID:-}" =~ ^[0-9]{3}$ ]] || {
    echo "error: eval array requires strategy ID 1-8 and three-digit BATCH_ID" >&2; exit 2;
  }
  (($# == 0)) || { echo "error: eval array derives its run from the array ID" >&2; exit 2; }
  set -- "fr3_osc_demo_${SLURM_ARRAY_TASK_ID}_${BATCH_ID}" "strategy_${SLURM_ARRAY_TASK_ID}" "${RAPID_EVAL_WORKERS:-4}"
fi

if (( $# < 2 || $# > 3 )); then
  usage
  exit 2
fi

readonly run_id="$1"
readonly strategy_name="$2"
readonly num_workers="${3:-4}"
readonly execution_policy="${RAPID_EVAL_EXECUTION_POLICY:-attempt}"

if [[ -z "$run_id" ]]; then
  echo "error: RUN_ID must not be empty" >&2
  exit 2
fi
if [[ -z "$strategy_name" ]]; then
  echo "error: STRATEGY_NAME must not be empty" >&2
  exit 2
fi
if [[ ! "$num_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: NUM_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ "$execution_policy" != "strict" && "$execution_policy" != "attempt" ]]; then
  echo "error: RAPID_EVAL_EXECUTION_POLICY must be strict or attempt" >&2
  exit 2
fi

cd "$repo_root"

if [[ ! -x ./scripts/rapid-benchmark ]]; then
  echo "error: missing executable: ${repo_root}/scripts/rapid-benchmark" >&2
  echo "submit from the repository root or set RAPID_REPO_ROOT explicitly" >&2
  exit 1
fi

# GPU rendering is required for evaluation recordings. rapid-benchmark divides
# the CPUs in this Slurm allocation across the concurrent case workers.
export RAPID_GPU_RENDER="required"

echo "[run_eval] started: $(date -Is)"
echo "[run_eval] run_id=${run_id}; strategy=${strategy_name}; workers=${num_workers}; policy=${execution_policy}"

extra_args=(--execution-policy "$execution_policy")
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  [[ "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] && ((num_workers <= SLURM_CPUS_PER_TASK)) || {
    echo "error: eval case workers exceed the Slurm CPU allocation" >&2; exit 2;
  }
  # CPU affinity may include SMT siblings; budget against the actual request.
  rollout_workers=$((SLURM_CPUS_PER_TASK / num_workers))
  ((rollout_workers <= 48)) || rollout_workers=48
  extra_args+=(--rollout-workers-per-job "$rollout_workers")
fi
if [[ "$execution_policy" == "attempt" ]]; then
  retry_source="${RAPID_EVAL_RETRY_RECOVERABLE_FROM-}"
  if [[ -z "${RAPID_EVAL_RETRY_RECOVERABLE_FROM+x}" ]]; then
    attempt_manifest="${RAPID_EXPERIMENT_ROOT}/$(basename -- "$run_id")_eval_attempt/benchmark_eval.json"
    if [[ -f "$attempt_manifest" ]]; then
      retry_source="$(
        sed -nE \
          's/^[[:space:]]*"recoverable_selection_source":[[:space:]]*"([^"]*)",?[[:space:]]*$/\1/p' \
          "$attempt_manifest"
      )"
    fi
    retry_source="${retry_source:-${RAPID_EVAL_RETRY_RECOVERABLE_DEFAULT:-}}"
  fi
  if [[ "$retry_source" == "auto" ]]; then
    retry_source="${run_id}_eval"
  fi
  if [[ -n "$retry_source" ]]; then
    extra_args+=(--retry-recoverable-from "$retry_source")
  fi
fi

./scripts/rapid-benchmark \
  --experiment-root "$RAPID_EXPERIMENT_ROOT" \
  "$run_id" \
  "$strategy_name" \
  --jobs "$num_workers" \
  --gpu-render required \
  "${extra_args[@]}" \
  --resume

echo "[run_eval] finished: $(date -Is)"
