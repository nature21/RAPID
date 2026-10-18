#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
source "${repo_root}/scripts/lib/experiment-root.sh"
rapid_experiment_root_args "$@"
set -- "${RAPID_EXPERIMENT_ARGS[@]}"

with_eval=1
batch_id="001"
batch_seen=0
after_job_id=""
after_dependency="afterok"
stage_nodes="${RAPID_STAGE_NODES:-crane[0-2]}"
while (($#)); do
    case "$1" in
        --nodes)
            (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || {
                echo "error: --nodes requires a Slurm node list" >&2; exit 2;
            }
            stage_nodes="$2"
            shift ;;
        --after|--after-start)
            (($# >= 2)) && [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
                echo "error: $1 requires a positive numeric Slurm job ID" >&2; exit 2;
            }
            [[ -z "$after_job_id" ]] || { echo "error: supply only one upstream dependency" >&2; exit 2; }
            [[ "$1" != --after-start ]] || after_dependency="after"
            after_job_id="$2"
            shift ;;
        --with-eval) with_eval=1 ;;
        --no-eval) with_eval=0 ;;
        -h|--help)
            echo "usage: $0 [--experiment-root DIR] [--nodes NODELIST] [--after JOB_ID | --after-start JOB_ID] [--no-eval] [BATCH_ID]"
            echo "DIR defaults to RAPID_EXPERIMENT_ROOT or repository suite_runs."
            echo "Relative experiment names are placed under suite_runs/."
            echo "--nodes selects eligible construction/composition nodes (default crane[0-2])."
            echo "Evaluation nodes are selected separately by RAPID_EVAL_NODES."
            echo "--after delays construction until the upstream job/array succeeds."
            echo "--after-start waits for all upstream array tasks to start or be cancelled."
            echo "One task per array element: 32 CPU, 1 GPU, 64G. Eval uses aftercorr on composition."
            echo "--no-eval submits only construction and composition; --with-eval explicitly enables eval."
            exit 0 ;;
        -*) echo "error: unknown option: $1" >&2; exit 2 ;;
        *)
            ((batch_seen == 0)) || { echo "error: only one BATCH_ID may be supplied" >&2; exit 2; }
            batch_id="$1"
            batch_seen=1 ;;
    esac
    shift
done

readonly BATCH_ID="$batch_id"
if [[ ! "$BATCH_ID" =~ ^[0-9]{3}$ ]]; then
    echo "error: BATCH_ID must be exactly three digits, such as 001, 002, or 003" >&2
    exit 2
fi

cd "${repo_root}"
mkdir -p primitive_construction_logs strategy_composition_logs

# Both stages inherit the same model settings through sbatch --export=ALL.
export RAPID_EXPERIMENT_MODEL="${RAPID_EXPERIMENT_MODEL:-gpt-6-astra}"
export RAPID_EXPERIMENT_THINKING_EFFORT="${RAPID_EXPERIMENT_THINKING_EFFORT:-high}"
source "${repo_root}/scripts/lib/slurm-arrays.sh"
construction_ids="$(rapid_primitive_ids "${RAPID_CONSTRUCTION_TASKS:-flip push grasp press_move pivot topple}")"
composition_ids="$(rapid_strategy_ids "${RAPID_COMPOSITION_RUNS:-1 2 3 4 5 6 7 8}")"
export RAPID_EVAL_STRATEGIES="${RAPID_EVAL_STRATEGIES:-${RAPID_COMPOSITION_RUNS:-1 2 3 4 5 6 7 8}}"
if ((with_eval)); then
  eval_ids="$(rapid_strategy_ids "$RAPID_EVAL_STRATEGIES")"
  for id in ${eval_ids//,/ }; do
    [[ ",$composition_ids," == *",$id,"* ]] || {
      echo "error: eval strategy $id has no corresponding composition array element" >&2; exit 2;
    }
  done
  # Validate the eval node pool before any construction job is submitted.
  rapid_slurm_task_args "${RAPID_EVAL_NODES:-crane0 crane1 crane2}"
fi
rapid_slurm_task_args "$stage_nodes"
stage_sbatch_args=("${RAPID_SLURM_TASK_ARGS[@]}" --nice=0)
export RAPID_REPO_ROOT="$repo_root"

# Freeze the source used by every stage before submitting asynchronous jobs.
RAPID_RUNTIME_WORKSPACE="$(
    env -u RAPID_RUNTIME_WORKSPACE ./scripts/rapid-suite prepare-runtime \
        --run-dir "${RAPID_EXPERIMENT_ROOT}/runtime_${BATCH_ID}"
)"
export RAPID_RUNTIME_WORKSPACE

echo "Model: ${RAPID_EXPERIMENT_MODEL}; thinking effort: ${RAPID_EXPERIMENT_THINKING_EFFORT}"
echo "Experiment root: ${RAPID_EXPERIMENT_ROOT}"
echo "Launching construction with BATCH_ID=${BATCH_ID}..."

construction_dependency=()
if [[ -n "$after_job_id" ]]; then
    construction_dependency=("--dependency=${after_dependency}:${after_job_id}")
    echo "Construction waits for ${after_dependency}:${after_job_id}"
fi
CONSTRUCTION_JOB_ID=$(sbatch \
    --parsable \
    "${stage_sbatch_args[@]}" \
    "--array=${construction_ids}" \
    "${construction_dependency[@]}" \
    --export=ALL,BATCH_ID="${BATCH_ID}" \
    "${repo_root}/run_primitive_construction.sh")
CONSTRUCTION_JOB_ID="${CONSTRUCTION_JOB_ID%%;*}"
[[ "$CONSTRUCTION_JOB_ID" =~ ^[0-9]+$ ]] || { echo "error: invalid construction job ID: $CONSTRUCTION_JOB_ID" >&2; exit 1; }

echo "Construction job submitted: ${CONSTRUCTION_JOB_ID}"

echo "Launching composition dependent on construction..."

COMPOSITION_JOB_ID=$(sbatch \
    --parsable \
    "${stage_sbatch_args[@]}" \
    "--array=${composition_ids}" \
    --dependency=afterok:"${CONSTRUCTION_JOB_ID}" \
    --export=ALL,BATCH_ID="${BATCH_ID}" \
    "${repo_root}/run_strategy_composition.sh")
COMPOSITION_JOB_ID="${COMPOSITION_JOB_ID%%;*}"
[[ "$COMPOSITION_JOB_ID" =~ ^[0-9]+$ ]] || { echo "error: invalid composition job ID: $COMPOSITION_JOB_ID" >&2; exit 1; }

echo "Composition job submitted: ${COMPOSITION_JOB_ID}"
echo "Dependency: ${COMPOSITION_JOB_ID} -> afterok:${CONSTRUCTION_JOB_ID}"

if ((with_eval)); then
    # Match a restricted composition batch unless an explicit eval selection is set.
    export RAPID_EVAL_STRATEGIES="${RAPID_EVAL_STRATEGIES:-${RAPID_COMPOSITION_RUNS:-1 2 3 4 5 6 7 8}}"
    echo "Queuing evaluations dependent on composition ${COMPOSITION_JOB_ID}..."
    "${repo_root}/launch_evals.sh" \
        --experiment-root "$RAPID_EXPERIMENT_ROOT" \
        --after-array "$COMPOSITION_JOB_ID" "$BATCH_ID"
fi
