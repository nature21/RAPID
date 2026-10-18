# `scripts/` - RAPID Command Entry Points

This directory intentionally contains shell entry points only. Most scripts are
thin wrappers that set the active RAPID Python environment and call a Python
module. Keep importable implementation logic in Python packages; keep shell
scripts focused on environment setup, permission-profile bootstrap, and command
composition.

## Start Here

| Script | Use For | Notes |
|---|---|---|
| `rapid-python` | Run any RAPID Python module or script. | Always prefer this over plain `python`; it sets `RAPID_*`, `PYTHONPATH`, and the `rapid` conda environment. |
| `rapid-view` | Interactively inspect an FR3 scene. | OSC torque control by default; `--controller kinematic` enables IK dragging. Supports scene names, modules, directories, and Python files. |
| `rapid-suite` | Create, merge, and run staged suite workspaces. | Wrapper for `python -m suites.staged.cli`. |
| `rapid-construct-primitive` | Common construction-stage primitive invention run. | Creates/reuses a suite workspace, launches `rapid-codex-run`, and writes command/output logs. |
| `rapid-codex-run` | Permission-profile Codex agent launch. | Uses the candidate workspace's `.codex/config.toml`, a run-local trusted `CODEX_HOME`, and an optional host NVIDIA/EGL video broker. |
| `rapid-strategy-test` | Bind and test an existing strategy on a new scene. | Reuses a suite-run workspace, invokes the strategy binding agent, then runs the generated instantiation with metrics and video. |
| `rapid-real-world-run` | Prepare one real-world trajectory from an eval suite. | Reuses the eval workspace, binds locally, executes with the `attempt` policy, and archives video plus Cartesian goals. |
| `rapid-workspace-env` | Switch the current shell to an existing suite workspace. | Use with `eval "$(./scripts/rapid-workspace-env RUN)"`. |
| `rapid-resume-workspace` | Resume a Codex session for an existing suite workspace. | Sets run-local `CODEX_HOME` and opens `codex resume` from the candidate workspace. |
| `rapid-session-view` | Inspect a saved Codex agent session. | Prints a colored terminal transcript by default; can also generate searchable HTML and follow a live session. |
| `rapid-rgbd-demo-clip` | Trim a synchronized RGB-D demo. | Removes redundant beginning/end content from RGB, lossless depth, depth preview, and timestamps together. |

## Common Workflows

Inspect the active Python/import environment:

```bash
./scripts/rapid-python --rapid-env
```

Learn primitives directly on the complete FR3 with Cartesian OSC actions:

```bash
./scripts/rapid-construct-primitive \
  --run push_fr3_osc_001 \
  --demo demos/primitive/push/rgb.mp4 \
  --env-module tasks.invention_real2sim.push_scene \
  --name push \
  --num-variants 3 \
  --description "Use the complete FR3 arm and the shared Cartesian OSC. The optimizer action order is x, y, z, roll, pitch, yaw, gripper; never optimize joint commands."
```

The optimizer uses seven channels: equilibrium-pose and gripper commands
in the FR3 base frame.
`FR3OSCAdapter` converts them to torque-limited Cartesian impedance control at
every MuJoCo step, with dynamics compensation, nullspace posture control,
joint-limit avoidance, and torque-rate limiting. The default native C++
controller runs entirely inside the CPU MuJoCo workers. Controller-aware
learning currently requires `--rollout-backend mujoco`; MJWarp direct-control
rollouts are rejected because they do not execute the OSC feedback law. The
project has one OSC implementation: the native C++ controller used by both
optimizer rollouts and the interactive viewer.

Try the same controller interactively in the flip scene:

```bash
./scripts/rapid-view invention_real2sim/flip_scene
```

Ctrl-drag the preselected hand to move or rotate its equilibrium pose. The
numeric keypad provides precise Cartesian and rotational increments without
activating MuJoCo's native letter-key shortcuts. Numpad Enter cycles soft,
default, and stiff compliance; Numpad decimal cycles the step size. The terminal
displays tracking error, commanded force, maximum joint torque, and joint-limit
margin. The viewer prints its key map when it starts; `./scripts/rapid-view
--help` lists gain and safety-limit options. Use `--controller kinematic` for
direct IK dragging; see [visualization tools](../vistools/README.md) for both
control modes. Cartesian targets are not restricted relative to the home pose
by default; pass `--workspace-radius METERS` when a local viewer boundary is useful.

Create a suite workspace:

```bash
./scripts/rapid-suite create-workspace --run-dir suite_runs/fr3_osc_primitive_push_001
```

Run a standard construction primitive invention experiment:

```bash
./scripts/rapid-construct-primitive \
  --run fr3_osc_primitive_push_001 \
  --demo demos/primitive/push/rgb.mp4 \
  --env-module tasks.invention_real2sim.push_scene \
  --name push \
  --num-variants 2
```

The invention agent handles the baseline and all variants in one Codex phase.
The helper prints separate workflow, behavior, and audit outcomes from
`workspace/agent_work/invention_status.json`; an audit advisory therefore does
not masquerade as either a process failure or a clean behavioral result.

Launch a deploy strategy agent in a prepared workspace:

```bash
./scripts/rapid-codex-run \
  --workspace suite_runs/deploy_example_001/workspace \
  -- agent/compose_strategy_exec.py demos/strategy/demo_1/rgb.mp4 \
     --name candidate_strategy_001 \
     --library-mode closed
```

The generic launcher supports GPU video rendering in both local and Slurm
runs. `--gpu-render auto` uses the host NVIDIA/EGL broker when available,
`--gpu-render required` fails its GPU preflight before launching the agent, and
`--gpu-render off` disables it. `RAPID_GPU_RENDER` sets the same policy. The
root `run_primitive_construction.sh` and `run_strategy_composition.sh` scripts default to `required`.

For an already-running development Codex sandbox, start the same trusted
broker from a host shell (outside the Codex sandbox):

```bash
./scripts/rapid-gpu-render-broker --workspace workspace
```

Use the active isolated workspace path instead of `workspace` when validating
a suite copy. The recorder automatically discovers the live broker through
`agent_work/gpu_render_ready.json`; no GPU device is exposed inside bubblewrap.
Stop the broker with Ctrl-C when the development session is finished.

Remove redundant content from the beginning and end of a recorded RGB-D demo:

```bash
./scripts/rapid-rgbd-demo-clip \
  demos/strategy/demo_1 \
  --start 1.2 \
  --end 18.4
```

The interval is half-open: the first recorded frame at or after `--start` is
included, while the first frame at or after `--end` is excluded. Omitting
`--end` clips through the final frame. By default, the example writes
`demos/strategy/demo_1_trimmed/`, preserving the untrimmed recording.

The trimmed output contains synchronized `rgb.mp4`, lossless `depth.mkv`,
`depth_preview.mp4`, and `timestamps.csv`. Timestamp frame indices and relative
seconds restart at zero; `source_frame_index` and `source_relative_seconds`
retain the original demo coordinates. The helper also writes `color.png`,
metric `depth.npy`, `intrinsics.json`, and `panda_eob_calib.yaml`, so the start
of the cleaned demo can be reconstructed directly. Use `--output PATH` to
choose a different output directory.

Generate primitive-learning segments after annotating
`demos/strategy/segments.csv`:

```bash
./scripts/rapid-python demos/strategy/.agents/skills/segment-demo-videos/scripts/segment_demo.py \
  --root demos/strategy \
  --manifest demos/strategy/segments.csv \
  --force
```

Each `demo_N/segment_J` contains clipped RGB and rebased timestamps plus the
first color frame, first metric depth frame, a view-only depth PNG, intrinsics,
extrinsics, and provenance metadata. Segment directories intentionally omit
`depth.mkv` and `depth_preview.mp4`; the lossless depth video remains only in
the unchanged source demo.

Switch follow-up commands to a suite workspace:

```bash
eval "$(./scripts/rapid-workspace-env suite_runs/fr3_osc_primitive_push_001)"
./scripts/rapid-python -m instantiations.primitives.push.example_0.main
```

Bind a learned strategy to a new assigned scene:

```bash
./scripts/rapid-codex-run \
  --workspace suite_runs/generalization_001/workspace \
  -- agent/deploy_strategy_exec.py \
     --strategy strategy_6 \
     --env-module tasks.invention_real2sim.new_scene \
     --name new_scene \
     --goal "Reorient the box and store it inside the corridor."
```

Bind and test a strategy in one command:

```bash
./scripts/rapid-strategy-test \
  deploy_candidate_001 \
  tasks.invention_real2sim.new_scene \
  "Reorient the box and store it inside the corridor."
```

The strategy is inferred when the workspace has exactly one strategy.
Otherwise pass `--strategy NAME`. The task package name becomes the
instantiation name unless overridden with `--name`. Final metrics and video
are written under that workspace's `results_instantiation/`. By default the command returns
nonzero when `program_success` is false; use `--allow-program-failure` for a
binding-validity-only run. The selected strategy must use the declarative
interface and provide `binding_contract.py`.

Prepare a trajectory for one reconstructed real-world scene using a frozen eval
suite checkpoint:

```bash
./scripts/rapid-real-world-run scene_3 fr3_osc_demo_1_001_eval
```

This runs directly on the local machine (no Slurm), creates
`suite_runs/fr3_osc_demo_1_001_eval/real_world_runs/scene_3/`, invokes the
binding agent in the eval workspace, and executes the strategy with
`RAPID_EXECUTION_POLICY=attempt`. After archiving, scene-local binding, result,
and copied task state are removed from the shared workspace so it remains
reusable. The review video,
`cartesian_goals.json`, metrics, binding package, and `real_world_run.json` are
archived directly under the scene run. The goal defaults to the frozen
strategy's `example_0` binding; use `--goal TEXT` only when the intended task
differs. GPU rendering and recording settings match benchmark eval defaults;
run `./scripts/rapid-real-world-run --help` for local overrides.

Copy a learned suite checkpoint into a disposable eval suite, run its frozen
strategy across a complete generated benchmark scene set, and record a GPU-
rendered trajectory for every case:

```bash
./scripts/rapid-benchmark \
  fr3_osc_demo_2_001 \
  tasks/sim_benchmark/strategy_2
```

The two positional inputs are a suite-run ID/path and a `strategy_N` scene
directory/module/alias. The command above creates
`suite_runs/fr3_osc_demo_2_001_eval_attempt`, preserving the original suite,
and archives each case's binding, metrics, Cartesian goals, and trajectory
under `benchmark_runs/strategy_2/agent/` inside that eval suite. Agent binding,
`--gpu-render required`, and the benchmark-only `attempt` execution policy are
the defaults. Pass `--execution-policy strict` to stop on every transit
rejection and use the traditional `_eval` destination. Review videos default
to 480x270 at 10 fps and 2x playback speed. The evaluator uses the repository-
wide maximum optimization budget while retaining every learned primitive's
frozen early-stopping rule.

Create the eval copy and its 50-slot gallery without starting the batch:

```bash
./scripts/rapid-benchmark fr3_osc_demo_2_001 strategy_2 --prepare-only
```

The gallery stays in place and silently retries missing videos every 30
seconds. Refresh its case metrics after or during a batch with the same inputs
plus `--gallery-only`.

Use `--case 1-3` for a smoke subset and then omit `--case` with `--resume` to
finish all 50 cases. `--eval-suite NAME` selects another destination, and
`--binding-mode authoritative` isolates strategy execution from semantic-
binding accuracy. The reusable command refuses to overwrite an existing eval
suite unless `--resume` is supplied and its origin, scene set, strategy, and
frozen checkpoint all match.

Evaluate multiple cases concurrently with independent worker workspaces:

```bash
./scripts/rapid-benchmark fr3_osc_demo_2_001 strategy_2 \
  --resume --jobs 2
```

By default, the evaluator divides the process's available CPU affinity across
the jobs and caps each case at 48 rollout workers. For example, two jobs on a
48-core allocation use 24 rollout workers each. Override this with
`--rollout-workers-per-job N`. Worker output is kept under `_worker_logs/`,
while the parent atomically updates the shared `summary.json` and gallery as
cases finish.

For a full local trial without Slurm, use `scripts/rapid-experiment-local`.
It calls the same stage entrypoints sequentially and accepts `--stage eval`
to evaluate existing strategies. See [SETUP.md](../SETUP.md#reproduce-the-experiments) for
asset installation, local resource controls, and cluster examples.

Submit strategy 1-8 evaluations as a Slurm array. Slurm places each element
on an available node in the configured pool:

```bash
export RAPID_EVAL_NODES="gpu01 gpu02 gpu03"
./launch_evals.sh --dry-run 002
./launch_evals.sh 002
```

Experiment helpers accept `--experiment-root DIR`, or the environment variable
`RAPID_EXPERIMENT_ROOT`. The flag takes precedence; the default is `suite_runs`
under the repository. Relative experiment names are resolved under `suite_runs/`;
paths starting with `suite_runs/` are repository-relative, and absolute paths
are used as supplied. For example, evaluate the `001` batch in
`suite_runs/new`:

```bash
./launch_evals.sh --experiment-root new --dry-run 001
./launch_evals.sh --experiment-root new 001
```

The selected root is used for source checkpoints, eval output, completion
checks, and automatic retry selection. Different roots have separate Slurm
job identities even when batch and strategy numbers match. Eval copies stay
under the selected root, such as
`suite_runs/new/fr3_osc_demo_1_001_eval_attempt`.

The same option is supported by `launch_experiments.sh`, `run_primitive_construction.sh`,
`run_strategy_composition.sh`, `run_eval.sh`, `rapid-construct-primitive`, `rapid-benchmark`,
`rapid-strategy-test`, `rapid-real-world-run`, `rapid-resume-workspace`,
`rapid-workspace-env`, and `rapid-session-view`. Construction/deploy jobs inherit the
root from the experiment launcher. Existing explicit run/workspace paths
retain their meaning; short run IDs and short eval destination names use the
selected root.

```bash
./launch_experiments.sh --experiment-root new 002
sbatch run_eval.sh --experiment-root new fr3_osc_demo_1_001 strategy_1 4
./scripts/rapid-benchmark --experiment-root new \
  fr3_osc_demo_1_001 strategy_1 --case 1-3 --resume
```

Submit the full construction → deploy → eval pipeline without monitoring the
experiment jobs yourself:

```bash
./launch_experiments.sh --experiment-root new 002
```

The pipeline submits three arrays. Primitive IDs 1-6 map to `flip`, `push`,
`grasp`, `press_move`, `pivot`, and `topple`. Strategy and eval IDs 1-8 match
task indices.

| Array | Resource request per element | Dependency | Slurm nice |
| --- | --- | --- | --- |
| Primitive construction | 32 CPU, 1 GPU, 64G memory | None | 0 |
| Strategy composition | 32 CPU, 1 GPU, 64G memory | `afterok:<primitive_array>` | 0 |
| Evaluation | 32 CPU, 1 GPU, 64G memory | `aftercorr:<strategy_array>` | 10000 |

All primitives must finish successfully before any strategy starts. Eval task
`i` becomes eligible as soon as strategy task `i` succeeds, independently of
other strategies. Strategy invention has higher scheduling priority than eval
through the eval nice penalty. This does not preempt running evals or promise
a strict execution order under every cluster scheduling policy. Dependencies
that can never succeed are cancelled with `--kill-on-invalid-dep=yes`.

All arrays are queued immediately; the terminal can close after submission.
`--no-eval` omits the evaluation array; `--with-eval` remains supported.
`RAPID_CONSTRUCTION_TASKS` selects primitive names, and `RAPID_COMPOSITION_RUNS`
selects strategy IDs. Eval inherits the strategy selection unless overridden
by `RAPID_EVAL_STRATEGIES`, which must be a subset of the composition IDs.
Subsets retain their original array IDs. Composition still needs all six
primitive workspaces in the selected experiment root.

Submission freezes a shared runtime checkpoint at `runtime_<batch>/workspace`
under the experiment root and exports `RAPID_RUNTIME_WORKSPACE` to each stage.
Construction, strategy, and evaluation copies retain that version even if the
repository changes. Shared-source edits during invention fail publication;
workspace copying and merging verify runtime fingerprints.

Select the eligible construction/composition node pool with `--nodes` (Slurm
node-list syntax) or `RAPID_STAGE_NODES`; the default is `crane[0-2]`.
Evaluation uses `RAPID_EVAL_NODES`, defaulting to `crane0 crane1 crane2`.
Each element requests exactly one node. The launchers exclude nodes outside
the selected pool, allowing Slurm to choose placement without requiring every
listed node in each allocation. Re-submit if cluster membership changes.

Run a full experiment with Astra:

```bash
export RAPID_EVAL_NODES="gpu01 gpu02 gpu03"
./launch_experiments.sh --nodes 'gpu[01-03]' \
  --experiment-root astra_cluster 001
```

Construction and composition default to `gpt-6-astra` with high reasoning
effort. `RAPID_EXPERIMENT_MODEL` and `RAPID_EXPERIMENT_THINKING_EFFORT`
configure those agents. `RAPID_COMPOSITION_MODEL` selects the frozen-strategy
deployment binder; its default is `gpt-6-luna` with low reasoning effort.
Each trial builds six primitives, composes eight strategies, and evaluates
50 cases per strategy. Use a fresh experiment root for each trial.

Use `--after JOB_ID` to wait for an upstream job to succeed, or
`--after-start JOB_ID` to wait for its tasks to start or be cancelled.

The strategy array merges the primitives into
`library_<batch>_array_<array_job_id>` on the frozen runtime before starting
strategy invention. Assembly verifies shared runtime integrity. A shared
filesystem lock serializes library preparation; a merge completion marker allows
the other strategy elements to reuse it. Each new strategy array has its own
library.

Each primitive or strategy array element runs one agent with up to 32 rollout
workers, subject to a lower `RAPID_MAX_PARALLEL_ROLLOUTS` override. Each eval
element runs four case workers by default, with eight rollout workers per
case. The memory and GPU allocation belong to that array element; independent
primitive/strategy runs each own their allocation. Direct stage runs
outside Slurm process their selected tasks sequentially.

To attach evals to an existing strategy array, use its array job ID:

```bash
./launch_evals.sh --experiment-root new --after-array 12345 002
```

`--after-array` uses `aftercorr`; source workspaces may be created later.
`--after JOB_ID` waits for the entire upstream job/array with `afterok`.
Both modes submit one sparse eval array, skipping active or completed evals.
Without a dependency, source workspaces must already exist. `--dry-run` prints
the submission without queuing it. Slurm handles resource waits. Stage and eval logs contain `%A_%a` array IDs.

`vistools/build_suite_runs_gallery.py` also accepts `--experiment-root` (with
`--suite-runs` as an alias). See [visualization tools](../vistools/README.md)
for scene viewing, orbit videos, and gallery commands.

The launcher defaults to the benchmark-only `attempt` policy and evaluates all
50 cases for each newly invented strategy. Set
`RAPID_EVAL_EXECUTION_POLICY=strict` for a strict full evaluation. To retry only
recoverable failures from each task's prior strict eval, select automatic mode:

```bash
RAPID_EVAL_RETRY_RECOVERABLE_FROM=auto \
  ./launch_evals.sh 002
```

An existing targeted `_eval_attempt` manifest retains its original selection
source when resumed. The launcher avoids duplicate active array elements and
skips completed targets. Interrupting the launcher does not cancel submitted
arrays. Resource requests are fixed at 32 CPU, 1 GPU, and 64G for all stages.

Resume the latest Codex session for a suite workspace:

```bash
./scripts/rapid-resume-workspace fr3_osc_primitive_push_001
./scripts/rapid-resume-workspace fr3_osc_primitive_push_001 -- "<session-id>"
```

Read the newest saved session for a run directly in the terminal:

```bash
./scripts/rapid-session-view fr3_osc_primitive_push_001
./scripts/rapid-session-view fr3_osc_primitive_push_001 --watch
./scripts/rapid-session-view suite_runs/fr3_osc_primitive_push_001 --list
```

Interactive terminal output opens in `less -R` automatically; use `/` to
search, `n` for the next match, and `q` to exit. Messages, readable reasoning
summaries, paired tool calls/outputs, and turn timing are shown by default.
Developer/context records and duplicate stream events are hidden unless
`--show-system`, `--show-context`, or `--show-stream` is passed. Long entries
are capped at 12,000 characters; use `--full` when the complete tool output is
needed, or `--raw` for the underlying sanitized JSON.

HTML remains available when useful:

```bash
./scripts/rapid-session-view fr3_osc_primitive_push_001 --html
./scripts/rapid-session-view fr3_osc_primitive_push_001 --html --no-images
```

The viewer reads `codex_home/sessions/**/rollout-*.jsonl` directly. HTML is
written under the run-local `codex_home/session_reports/` by default. Both
formats can contain full prompt and tool-output text, so keep their output
local.

## Review Rules

- Do not add new shell scripts when a subcommand in an existing Python CLI is
  enough.
- New scripts should have `set -euo pipefail`, a `usage()` block when they
  parse arguments, and should call `rapid-python` rather than plain `python`.
- Keep generated outputs under the selected workspace or suite run directory.
- Keep Codex permission policy in generated workspace `.codex/config.toml`;
  launch helpers should compose commands rather than duplicating permission
  setup.
