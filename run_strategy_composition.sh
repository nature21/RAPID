#!/usr/bin/env bash
#SBATCH --job-name=run_strategy_composition
#SBATCH --output=strategy_composition_logs/composition_output_%A_%a.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nice=0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${RAPID_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
repo_root="$(cd -- "$repo_root" && pwd)"
source "${repo_root}/scripts/lib/experiment-root.sh"
rapid_experiment_root_args "$@"
set -- "${RAPID_EXPERIMENT_ARGS[@]}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "usage: $0 [--experiment-root DIR] (batch: BATCH_ID environment variable)"
  echo "DIR defaults to RAPID_EXPERIMENT_ROOT or repository suite_runs."
  exit 0
fi
(($# == 0)) || { echo "error: unexpected argument: $1" >&2; exit 2; }
cd "$repo_root"
echo "Experiment root: ${RAPID_EXPERIMENT_ROOT}"

readonly BATCH_ID="${BATCH_ID:-001}"
if [[ ! "$BATCH_ID" =~ ^[0-9]{3}$ ]]; then
  echo "error: BATCH_ID must be exactly three digits, such as 001, 002, or 003" >&2
  exit 2
fi

# CPU MuJoCo rollout parallelism is inferred from the Slurm allocation and
# capped centrally. Set RAPID_MAX_PARALLEL_ROLLOUTS only to request a lower cap.
# Rendering uses the generic host NVIDIA/EGL broker in both Slurm and local
# launches. These GPU-requesting production runs fail early instead of silently
# falling back to Mesa software rendering; set RAPID_GPU_RENDER=off explicitly
# for a deliberate CPU-only run.
export RAPID_GPU_RENDER="${RAPID_GPU_RENDER:-required}"

format_elapsed() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

suite_started_epoch="$(date +%s)"
suite_started_at="$(date -Is)"
suite_timing_reported=0

report_suite_timing() {
  local exit_status="$?"
  local finished_epoch elapsed
  if ((suite_timing_reported == 0)); then
    finished_epoch="$(date +%s)"
    elapsed=$((finished_epoch - suite_started_epoch))
    echo \
      "[run_strategy_composition] timing: started=${suite_started_at}; finished=$(date -Is);" \
      "elapsed=$(format_elapsed "$elapsed")"
    suite_timing_reported=1
  fi
  return "$exit_status"
}

trap report_suite_timing EXIT

log_dir="${RAPID_COMPOSITION_LOG_DIR:-strategy_composition_logs}"
mkdir -p "$log_dir"
run_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
description_csv="demos/strategy/description.csv"
composition_runs=(1 2 3 4 5 6 7 8)
if [[ -n "${RAPID_COMPOSITION_RUNS:-}" ]]; then
  read -r -a composition_runs <<< "${RAPID_COMPOSITION_RUNS//,/ }"
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-8]$ ]] || { echo "error: strategy array ID must be 1-8" >&2; exit 2; }
  composition_runs=("$SLURM_ARRAY_TASK_ID")
fi

if ((${#composition_runs[@]} == 0)); then
  echo "error: RAPID_COMPOSITION_RUNS must contain at least one run number" >&2
  exit 2
fi
for index in "${composition_runs[@]}"; do
  if [[ ! "$index" =~ ^[0-9]+$ ]]; then
    echo "error: RAPID_COMPOSITION_RUNS must be a space- or comma-separated list of integers" >&2
    exit 2
  fi
done

# One strategy per array element; local direct runs are sequential.
parallel_demos=1

library_run="${RAPID_EXPERIMENT_ROOT}/library_${BATCH_ID}${SLURM_ARRAY_JOB_ID:+_array_${SLURM_ARRAY_JOB_ID}}"
# All strategy elements in an array share one merged seed library.
# A filesystem lock serializes merges; the completion marker permits reuse.
prepare_library() (
  mkdir -p "$RAPID_EXPERIMENT_ROOT"
  flock -x 9
  if [[ ! -f "${library_run}/.merge_complete" ]]; then
    ./scripts/rapid-suite merge-workspaces \
    --run-dir "$library_run" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_flip_${BATCH_ID}" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_grasp_${BATCH_ID}" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_press_move_${BATCH_ID}" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_push_${BATCH_ID}" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_pivot_${BATCH_ID}" \
    --source-workspace "${RAPID_EXPERIMENT_ROOT}/fr3_osc_primitive_topple_${BATCH_ID}" \
    || exit $?
    touch "${library_run}/.merge_complete"
  fi
) 9>"${library_run}.lock"
mkdir -p "$RAPID_EXPERIMENT_ROOT"
prepare_library


allocated_cpus="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "$allocated_cpus" ]]; then
  allocated_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
if [[ ! "$allocated_cpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: allocated CPU count must be a positive integer" >&2
  exit 2
fi
if ((parallel_demos > allocated_cpus)); then
  echo "error: parallel composition demos (${parallel_demos}) exceed allocated CPUs (${allocated_cpus})" >&2
  exit 2
fi

rollout_workers_per_demo=$((allocated_cpus / parallel_demos))
if ((rollout_workers_per_demo > 48)); then
  rollout_workers_per_demo=48
fi
if [[ -n "${RAPID_MAX_PARALLEL_ROLLOUTS:-}" ]]; then
  if [[ ! "$RAPID_MAX_PARALLEL_ROLLOUTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RAPID_MAX_PARALLEL_ROLLOUTS must be a positive integer" >&2
    exit 2
  fi
  if ((RAPID_MAX_PARALLEL_ROLLOUTS < rollout_workers_per_demo)); then
    rollout_workers_per_demo="$RAPID_MAX_PARALLEL_ROLLOUTS"
  fi
fi

run_strategy_composition_demo() {
  local index="$1"
  local description
  description="$(
    ./scripts/rapid-python - "$index" "$description_csv" <<'PY'
import csv
import sys
from pathlib import Path

demo_id = sys.argv[1]
csv_path = Path(sys.argv[2])

with csv_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, skipinitialspace=True)
    for row in reader:
        if (row.get("demo") or "").strip() == demo_id:
            goal = (row.get("goal") or "").strip()
            description = (row.get("description") or "").strip()
            if not goal:
                print(f"error: empty goal for demo {demo_id}", file=sys.stderr)
                sys.exit(2)
            if not description:
                print(f"error: empty description for demo {demo_id}", file=sys.stderr)
                sys.exit(2)
            print(f"goal: {goal}; description: {description}")
            sys.exit(0)

print(f"error: no description found for demo {demo_id} in {csv_path}", file=sys.stderr)
sys.exit(2)
PY
  )" || return $?

  local candidate_run="${RAPID_EXPERIMENT_ROOT}/fr3_osc_demo_${index}_${BATCH_ID}"
  ./scripts/rapid-suite create-workspace \
    --run-dir "$candidate_run" --from-workspace "$library_run/workspace" \
    || return $?

  ./scripts/rapid-codex-run \
    --workspace "$candidate_run/workspace" \
    -- agent/compose_strategy_exec.py demos/strategy/demo_${index}/rgb.mp4 \
      --env-module tasks.invention_real2sim.demo_${index}_scene \
      --name strategy_${index} \
      --model "${RAPID_EXPERIMENT_MODEL:-gpt-6-astra}" \
      --thinking-effort "${RAPID_EXPERIMENT_THINKING_EFFORT:-high}" \
      --description "$description" \
    || return $?
}

declare -a passed_demos=()
declare -a failed_demos=()
declare -A demo_by_pid=()
declare -A demo_started_epoch_by_pid=()
declare -A demo_elapsed_by_index=()
declare -A demo_result_by_index=()
running_demos=0

reap_composition_demo() {
  local pid index demo_status finished_epoch elapsed
  while true; do
    for pid in "${!demo_by_pid[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      index="${demo_by_pid[$pid]}"
      if wait "$pid"; then
        demo_status=0
      else
        demo_status=$?
      fi
      finished_epoch="$(date +%s)"
      elapsed=$((finished_epoch - demo_started_epoch_by_pid[$pid]))
      demo_elapsed_by_index[$index]="$elapsed"
      unset 'demo_by_pid[$pid]'
      unset 'demo_started_epoch_by_pid[$pid]'
      running_demos=$((running_demos - 1))
      if ((demo_status == 0)); then
        passed_demos+=("${index}")
        demo_result_by_index[$index]="passed"
        echo "[run_strategy_composition] demo ${index}: passed; elapsed=$(format_elapsed "$elapsed")"
      else
        failed_demos+=("${index} (exit ${demo_status})")
        demo_result_by_index[$index]="failed (exit ${demo_status})"
        echo \
          "[run_strategy_composition] demo ${index}: failed with exit ${demo_status};" \
          "elapsed=$(format_elapsed "$elapsed"); continuing" >&2
      fi
      return
    done
    sleep 1
  done
}

echo "[run_strategy_composition] started: ${suite_started_at}; node=$(hostname)"
echo \
  "[run_strategy_composition] batch ID: ${BATCH_ID};" \
  "[run_strategy_composition] parallel demos: ${parallel_demos};" \
  "rollout workers/demo: ${rollout_workers_per_demo};" \
  "allocated CPUs: ${allocated_cpus}"

for index in "${composition_runs[@]}"; do
  composition_log="${log_dir}/demo_${index}_${run_id}.log"
  demo_started_epoch="$(date +%s)"
  echo "[run_strategy_composition] demo ${index}: started=$(date -Is); log=${composition_log}"

  RAPID_MAX_PARALLEL_ROLLOUTS="$rollout_workers_per_demo" \
    run_strategy_composition_demo "$index" >"$composition_log" 2>&1 &
  demo_pid=$!
  demo_by_pid[$demo_pid]="$index"
  demo_started_epoch_by_pid[$demo_pid]="$demo_started_epoch"
  running_demos=$((running_demos + 1))

  if ((running_demos >= parallel_demos)); then
    reap_composition_demo
  fi
done

while ((running_demos > 0)); do
  reap_composition_demo
done

echo "[run_strategy_composition] completed: ${#passed_demos[@]} passed, ${#failed_demos[@]} failed"
for index in "${composition_runs[@]}"; do
  echo \
    "[run_strategy_composition] timing demo ${index}: status=${demo_result_by_index[$index]};" \
    "elapsed=$(format_elapsed "${demo_elapsed_by_index[$index]}")"
done
report_suite_timing
if ((${#failed_demos[@]})); then
  printf '[run_strategy_composition] failed: demo %s\n' "${failed_demos[@]}" >&2
  exit 1
fi
