#!/usr/bin/env bash
#SBATCH --job-name=run_primitive_construction
#SBATCH --output=primitive_construction_logs/construction_output_%A_%a.log
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
      "[run_primitive_construction] timing: started=${suite_started_at}; finished=$(date -Is);" \
      "elapsed=$(format_elapsed "$elapsed")"
    suite_timing_reported=1
  fi
  return "$exit_status"
}

trap report_suite_timing EXIT

# Keep each task's output separate. The Slurm output file serves as a short
# index pointing to these task-specific logs.
log_dir="${RAPID_CONSTRUCTION_LOG_DIR:-primitive_construction_logs}"
mkdir -p "${log_dir}"
run_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"

task_list="${RAPID_CONSTRUCTION_TASKS:-flip push grasp press_move pivot topple}"
common_description="The assigned scene uses the complete FR3 arm. Learn directly with the shared Cartesian impedance OSC; never optimize joint positions or torques. The optimizer action order remains x, y, z, roll, pitch, yaw, gripper in the FR3 base frame."
declare -A task_descriptions
task_descriptions[press_move]="The effect of this skill is to move the target object by a displacement while keeping the orientation of the target object unchanged. $common_description"
task_descriptions[flip]="The effect of this skill is to flip the target object against a fixture with a vertical face. $common_description"
task_descriptions[grasp]="The effect of this skill is to grasp the target object with the gripper. $common_description"
task_descriptions[push]="The effect of this skill is to push the target object by a displacement while keeping the orientation of the target object unchanged. $common_description"
task_descriptions[topple]="The effect of this skill is to topple the target object (the red cheez-it box inthe middle) by pressing on the top of it. $common_description"
task_descriptions[pivot]="The effect of this skill is to pivot the target object. $common_description"

source "${repo_root}/scripts/lib/slurm-arrays.sh"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-6]$ ]] || { echo "error: primitive array ID must be 1-6" >&2; exit 2; }
  task_list="${RAPID_PRIMITIVE_NAMES[$((SLURM_ARRAY_TASK_ID - 1))]}"
fi
read -r -a tasks <<<"$task_list"
if ((${#tasks[@]} == 0)); then
  echo "error: task_list must contain at least one task" >&2
  exit 2
fi

# An array element owns exactly one primitive. Local direct runs are sequential.
parallel_tasks=1

allocated_cpus="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "$allocated_cpus" ]]; then
  allocated_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
if [[ ! "$allocated_cpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: allocated CPU count must be a positive integer" >&2
  exit 2
fi
if ((parallel_tasks > allocated_cpus)); then
  echo "error: parallel construction tasks (${parallel_tasks}) exceed allocated CPUs (${allocated_cpus})" >&2
  exit 2
fi

rollout_workers_per_task=$((allocated_cpus / parallel_tasks))
if ((rollout_workers_per_task > 48)); then
  rollout_workers_per_task=48
fi
if [[ -n "${RAPID_MAX_PARALLEL_ROLLOUTS:-}" ]]; then
  if [[ ! "$RAPID_MAX_PARALLEL_ROLLOUTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RAPID_MAX_PARALLEL_ROLLOUTS must be a positive integer" >&2
    exit 2
  fi
  if ((RAPID_MAX_PARALLEL_ROLLOUTS < rollout_workers_per_task)); then
    rollout_workers_per_task="$RAPID_MAX_PARALLEL_ROLLOUTS"
  fi
fi

declare -a completed_tasks=()
declare -a failed_tasks=()
declare -A task_by_pid=()
declare -A task_started_epoch_by_pid=()
declare -A task_elapsed_by_name=()
declare -A task_result_by_name=()
declare -A task_run_name_by_name=()
declare -A task_status_file_by_name=()
declare -A task_status_signature_before_by_name=()
running_tasks=0

run_primitive_construction_task() {
  local task="$1"
  local task_log="$2"
  local task_run_name="$3"

  RAPID_MAX_PARALLEL_ROLLOUTS="$rollout_workers_per_task" \
    ./scripts/rapid-construct-primitive \
      --run "$task_run_name" \
      --demo "demos/primitive/${task}/rgb.mp4" \
      --env-module "tasks.invention_real2sim.${task}_scene" \
      --name "${task}" \
      --model "${RAPID_EXPERIMENT_MODEL:-gpt-6-astra}" \
      --thinking-effort "${RAPID_EXPERIMENT_THINKING_EFFORT:-high}" \
      --num-variants 3 \
      --description "${task_descriptions[$task]}" \
      >"${task_log}" 2>&1
}

reap_construction_task() {
  local pid task task_status finished_epoch elapsed started_epoch
  local status_file status_line status_signature_before status_signature_after
  local workflow_status behavior_status audit_status
  while true; do
    for pid in "${!task_by_pid[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      task="${task_by_pid[$pid]}"
      if wait "$pid"; then
        task_status=0
      else
        task_status=$?
      fi
      finished_epoch="$(date +%s)"
      started_epoch="${task_started_epoch_by_pid[$pid]}"
      elapsed=$((finished_epoch - started_epoch))
      task_elapsed_by_name[$task]="$elapsed"
      unset 'task_by_pid[$pid]'
      unset 'task_started_epoch_by_pid[$pid]'
      running_tasks=$((running_tasks - 1))
      status_file="${task_status_file_by_name[$task]}"
      workflow_status="unknown"
      behavior_status="unknown"
      audit_status="unknown"
      status_signature_before="${task_status_signature_before_by_name[$task]}"
      if [[ -f "${status_file}" ]]; then
        status_signature_after="$(stat -c '%y:%s' "${status_file}")"
      else
        status_signature_after=""
      fi
      if [[ -n "${status_signature_after}" && "${status_signature_after}" != "${status_signature_before}" ]]; then
        if status_line="$(
          "${repo_root}/scripts/rapid-python" \
            -m agent.invention_status "${status_file}" 2>/dev/null
        )"; then
          IFS=$'\t' read -r \
            workflow_status behavior_status audit_status <<<"${status_line}"
        fi
      fi
      if ((task_status == 0)); then
        completed_tasks+=("${task}")
        task_result_by_name[$task]="process=completed workflow=${workflow_status} behavior=${behavior_status} audit=${audit_status}"
        echo \
          "[run_primitive_construction] ${task}: process=completed;" \
          "workflow=${workflow_status}; behavior=${behavior_status};" \
          "audit=${audit_status}; elapsed=$(format_elapsed "$elapsed")"
      else
        failed_tasks+=("${task} (exit ${task_status})")
        task_result_by_name[$task]="process=failed(exit ${task_status}) workflow=${workflow_status} behavior=${behavior_status} audit=${audit_status}"
        echo \
          "[run_primitive_construction] ${task}: process=failed(exit ${task_status});" \
          "workflow=${workflow_status}; behavior=${behavior_status};" \
          "audit=${audit_status};" \
          "elapsed=$(format_elapsed "$elapsed"); continuing" >&2
      fi
      return
    done
    sleep 1
  done
}

echo "[run_primitive_construction] started: ${suite_started_at}; node=$(hostname)"
echo \
  "[run_primitive_construction] batch ID: ${BATCH_ID};" \
  "[run_primitive_construction] parallel tasks: ${parallel_tasks};" \
  "rollout workers/task: ${rollout_workers_per_task};" \
  "allocated CPUs: ${allocated_cpus}"

for task in "${tasks[@]}"; do
  task_log="${log_dir}/${task}_${run_id}.log"
  task_started_epoch="$(date +%s)"
  echo "[run_primitive_construction] ${task}: started=$(date -Is); log=${task_log}"
  task_run_name_by_name[$task]="fr3_osc_primitive_${task}_${BATCH_ID}"
  task_status_file_by_name[$task]="${RAPID_EXPERIMENT_ROOT}/${task_run_name_by_name[$task]}/workspace/agent_work/invention_status.json"
  if [[ -f "${task_status_file_by_name[$task]}" ]]; then
    task_status_signature_before_by_name[$task]="$(
      stat -c '%y:%s' "${task_status_file_by_name[$task]}"
    )"
  else
    task_status_signature_before_by_name[$task]=""
  fi
  run_primitive_construction_task "$task" "$task_log" "${task_run_name_by_name[$task]}" &
  task_pid=$!
  task_by_pid[$task_pid]="$task"
  task_started_epoch_by_pid[$task_pid]="$task_started_epoch"
  running_tasks=$((running_tasks + 1))

  if ((running_tasks >= parallel_tasks)); then
    reap_construction_task
  fi
done

while ((running_tasks > 0)); do
  reap_construction_task
done

echo "[run_primitive_construction] completed: ${#completed_tasks[@]} processes completed, ${#failed_tasks[@]} processes failed"
for task in "${tasks[@]}"; do
  echo \
    "[run_primitive_construction] timing ${task}: status=${task_result_by_name[$task]};" \
    "elapsed=$(format_elapsed "${task_elapsed_by_name[$task]}")"
done
report_suite_timing
if ((${#failed_tasks[@]})); then
  printf '[run_primitive_construction] failed: %s\n' "${failed_tasks[@]}" >&2
  exit 1
fi
