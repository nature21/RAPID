# Staged Suite Runs

This project uses a two-stage suite model:

1. **Construction stage**: an agent starts from a workspace checkpoint and
   learns named primitives from demonstrations and assigned MuJoCo tasks.
2. **composition stage**: an agent starts from a learned or merged workspace
   checkpoint and builds a strategy that composes the primitives already
   present in that workspace.

The workspace is the checkpoint. It is the only artifact used to transfer
learned state between stages.

## Runtime Wrapper

See `scripts/README.md` for the script index and common command recipes.

Run project Python from the repo root:

```bash
./scripts/rapid-suite --help
```

The wrapper uses the `rapid` conda environment while putting the active source
tree on `PYTHONPATH`. It preserves the caller's current working directory.
Inspect the active runtime with:

```bash
./scripts/rapid-python --rapid-env
```

For isolated suite workspaces, set:

```bash
export RAPID_WORKSPACE=/abs/path/to/run/workspace
export RAPID_ACTIVE_SRC_DIR=/abs/path/to/run/workspace/src
export RAPID_SRC_DIR=/abs/path/to/run/workspace/src
```

`RAPID_ACTIVE_SRC_DIR` is the source tree imported first. `RAPID_SRC_DIR` is a
compatibility alias and should match it. Both default to `$RAPID_WORKSPACE/src`
when the workspace has the standard generated layout.

Codex agent launchers use a compact human-readable transcript by
default. To print raw command output and patch diffs while debugging,
set:

```bash
export RAPID_AGENT_LOG=verbose
```

## Create A Workspace

Start from the canonical clean checkpoint:

```bash
RUN=suite_runs/fr3_osc_primitive_push_001
./scripts/rapid-suite create-workspace --run-dir "$RUN"
```

Or continue from a previous workspace checkpoint:

```bash
RUN=suite_runs/fr3_osc_primitive_push_002
BASE=suite_runs/fr3_osc_primitive_push_001
./scripts/rapid-suite create-workspace --run-dir "$RUN" --from-workspace "$BASE"
```

Checkpoint input arguments accept either the run directory (`$RUN`) or the
nested workspace directory (`$RUN/workspace`). Prefer the run directory form
for `--from-workspace`, `--base-workspace`, and `--source-workspace`.

This creates:

```text
$RUN/workspace/
  src/          # copied learned source checkpoint
  .codex/       # generated Codex permission profile for this workspace
  test_scenes/  # copied when present
  results/      # fresh
  agent_work/   # fresh
  README.md
```

The command prints the `RAPID_WORKSPACE`, `RAPID_ACTIVE_SRC_DIR`, and
`RAPID_SRC_DIR` values to use when launching agents. When no
`--from-workspace` is provided, it audits the canonical `workspace/src`
baseline and refuses to run if learned primitives, generated instantiations,
or reference strategies are present there.

## Merge Workspaces

If several construction runs learned useful primitives separately, merge them into a
new composition-start workspace:

```bash
RUN=suite_runs/deploy_merged_001
./scripts/rapid-suite merge-workspaces \
  --run-dir "$RUN" \
  --source-workspace suite_runs/fr3_osc_primitive_push_001 \
  --source-workspace suite_runs/fr3_osc_primitive_press_move_001
```

The merged workspace includes copied primitives, primitive instantiations, and
generated validation scenes. It does not copy prior
`results/`, `agent_work/`, or generated strategies. If two source
workspaces define different implementations with the same primitive package
name, the incoming package is renamed deterministically and copied primitive
instantiation imports are rewritten. The merge provenance is
recorded in:

```text
$RUN/workspace/merge_index.yaml
```

## Smoke-Test A Candidate Workspace

The clean baseline has no concrete primitive modules. Use import checks for
the shared infrastructure before launching an agent:

```bash
./scripts/rapid-suite run-module \
  --workspace "$RUN/workspace" \
  --module tasks.invention_real2sim.push_scene.scene
```

After an agent constructs a primitive and its `example_0` instantiation, validate
the generated module with absolute metrics and recording paths.

## Construction Stage

For common construction experiments, use the helper that creates or reuses the
workspace, launches the Codex agent with the workspace-local permission
profile, and records the exact command/output logs under `agent_work/`:

```bash
./scripts/rapid-construct-primitive \
  --run fr3_osc_primitive_push_001 \
  --demo demos/primitive/push/rgb.mp4 \
  --env-module tasks.invention_real2sim.push_scene \
  --name push \
  --num-variants 2 \
  --description "Use the complete FR3 arm and the shared Cartesian OSC. The optimizer action order is x, y, z, roll, pitch, yaw, gripper; never optimize joint commands."
```

Short `--run` and `--from-workspace` names expand under `suite_runs/`. Add
`--dry-run` to inspect the workspace creation and agent launch commands without
starting an experiment.

Construction invention runs use one Codex phase for `example_0` and every requested
variant. The agent autonomously infers and commits the learned goal and
predicate before simulation, and may develop the shared program using all
examples. The launcher writes `workspace/agent_work/invention_status.json`
with independent `workflow_completed`, `behavior_status`, and `audit_status`
fields. Collision/contact/clearance checker disagreement can remain an
explicit audit advisory without being mislabeled as a failed process or a
clean audit.

For lower-level control, launch a construction agent inside a fresh candidate
workspace directly:

```bash
./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/construct_primitive_exec.py demos/primitive/push/rgb.mp4 \
     --env-module tasks.invention_real2sim.push_scene \
     --name push \
     --num-variants 2
```

The `--` separator ends wrapper options; everything after it is forwarded
to `agent/construct_primitive_exec.py`. The wrapper sets `RAPID_WORKSPACE`,
`RAPID_ACTIVE_SRC_DIR`, and `RAPID_SRC_DIR` for the launched agent process.

### GPU video rendering

`rapid-codex-run` starts a host-side NVIDIA/EGL render broker before launching
the sandboxed agent. Calls to the normal `scene_io.record.record_video` API in
suite workspaces use that broker transparently; the launcher refreshes this
shared recording helper in reused workspaces before a new agent starts. The
agent keeps its filesystem isolation while MuJoCo rendering can open the host
GPU devices.

An already-running development Codex session can use the identical host-side
path by starting `./scripts/rapid-gpu-render-broker --workspace WORKSPACE` from
a host shell. The sandboxed recorder discovers the broker readiness file
automatically. This is intentionally brokered: Codex's nested bubblewrap
namespace still does not receive direct access to `/dev/nvidia*`.

This is launcher-based rather than scheduler-specific: under Slurm it inherits
the job's GPU visibility, and when launched directly it uses the GPUs visible
to the local shell.

Control the preflight and fallback policy with either `--gpu-render MODE` on
`rapid-codex-run` or `RAPID_GPU_RENDER=MODE`:

- `auto` (the direct-launch default) uses verified NVIDIA/EGL rendering when
  available and otherwise leaves `record_video` on its local fallback.
- `required` fails before the agent starts unless NVIDIA/EGL rendering and the
  MP4 encoder both work.
- `off` disables the broker deliberately.

`run_primitive_construction.sh` and `run_strategy_composition.sh` default to `required` because those
production jobs request a GPU. The same scripts can still be run locally; set
`RAPID_GPU_RENDER=off` only for an intentional CPU/software-rendered run. Broker
startup and backend evidence are written to
`$RAPID_WORKSPACE/agent_work/gpu_render_broker.log` and
`gpu_render_ready.json` while the run is active.

The repository root `AGENTS.md` is for human developers and agents modifying
RAPID itself. Workspace creation stages a separate automatic-agent bundle into
the candidate workspace; exec invention launchers may refresh the same files
before running Codex:

```text
$RAPID_WORKSPACE/AGENTS.md
$RAPID_WORKSPACE/.agents/skills/rapid-construct-primitive/SKILL.md
$RAPID_WORKSPACE/.agents/skills/rapid-compose-strategy/SKILL.md
$RAPID_WORKSPACE/.agents/skills/rapid-instantiate-primitive/SKILL.md
$RAPID_WORKSPACE/.agents/skills/rapid-instantiate-strategy/SKILL.md
$RAPID_WORKSPACE/.agents/skills/rapid-deploy-strategy/SKILL.md
```

Codex is launched with cwd `$RAPID_WORKSPACE`, so these workspace-local files
are the instructions and reusable skills for the automatic invent agent.
Workspace creation also writes `$RAPID_WORKSPACE/.codex/config.toml`, which
defines the `rapid-experiment` permission profile, and `$RUN_DIR/codex_home/config.toml`,
which trusts the exact generated workspace. The launcher uses
`$RUN_DIR/codex_home` as `CODEX_HOME` and refreshes that bootstrap before launch,
copying host `auth.json` only when available.

The Codex permission profile is permissive for dependencies but strict about
task inputs. It grants read access to repository launch helpers, the full conda
installation, `/opt` when present, the vendored `third_party/judo` runtime
dependency, safe external `PYTHONPATH` entries, and video tools such as
`ffmpeg`/`ffprobe` when they live outside common system paths.
Demo files are granted only when they appear explicitly in the agent command.
The agent `--env-module` flag accepts
either `tasks.invention_real2sim.push_scene` or
`tasks.invention_real2sim.push_scene.scene`, and may
be repeated for multiple assigned practice scenes. The wrapper copies
matching task roots into the run workspace's `task_inputs/` directory, which
is read-only to generated Codex commands.
Use explicit `--task-bind` only when the scene imports shared helper
modules outside its own task root.
The agent can also take optional language guidance with
`--description "..."` or `--description-file path.txt`; omit these flags
when the demo and task scene are enough. If a description file is outside
`demos/`, expose it with wrapper `--ro-bind`.
The launcher keeps Codex state under the experiment run directory by default
and does not copy host `config.toml`, so personal permission settings and
installed skills are not inherited. The profile keeps `$RUN_DIR/codex_home`
unreadable except for the exact `$RUN_DIR/codex_home/tmp/arg0` helper
directory required by Codex on Linux. The profile
still blocks non-candidate RAPID source checkpoints such as canonical
`workspace/src`, sibling `suite_runs`, historical artifacts, and the global
`tasks/` tree, so the generated agent cannot inspect existing primitives or
unassigned tasks from nearby runs.

To practice one primitive against multiple assigned FR3 scenes,
repeat environment modules:

```bash
./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/construct_primitive_exec.py demos/primitive/push/rgb.mp4 \
     --env-module tasks.invention_real2sim.push_scene \
     --env-module tasks.invention_real2sim.press_move_scene \
     --name push \
     --num-variants 2
```

The first environment module is used as `example_0`; subsequent modules
are used as `example_1`, `example_2`, ... before the agent invents any
synthetic variants. The launcher copies the selected scene packages and their
direct package-level Python helpers into the isolated workspace. Use an
explicit `--task-bind` only when a scene needs files outside those roots.

## composition Stage

Create or merge a composition-start workspace, then launch a composition agent:

```bash
RUN=suite_runs/deploy_codex_001
BASE=suite_runs/fr3_osc_primitive_push_001

./scripts/rapid-suite create-workspace \
  --run-dir "$RUN" \
  --from-workspace "$BASE"

./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/compose_strategy_exec.py demos/strategy/demo_1/rgb.mp4 \
     --name candidate_strategy_001 \
     --library-mode closed
```

Default composition protocol should be **closed library**: the agent may read
and compose the existing learned primitives under `src/primitives`, but should
not edit `src/primitives` or the shared `src/primitive_core`. If a suite allows
primitive creation during
composition, mark it as `open_library` in the composition suite manifest and pass
`--library-mode open`.

Use a supplied reconstructed scene as the assigned environment module. Packaged Real2Sim
task scenes are self-contained, and the launcher auto-copies the selected
task package:

```bash
RUN=suite_runs/deploy_merged_005

./scripts/rapid-suite merge-workspaces \
  --run-dir "$RUN" \
  --source-workspace suite_runs/fr3_osc_primitive_flip_001 \
  --source-workspace suite_runs/fr3_osc_primitive_push_001 \
  --source-workspace suite_runs/fr3_osc_primitive_press_move_001

./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/compose_strategy_exec.py demos/strategy/demo_5/rgb.mp4 \
     --env-module tasks.invention_real2sim.demo_5_scene \
     --name push__flip__press_move \
     --library-mode closed \
     --description "Push the cookie box toward the wall, flip it to align with an opening between the wall and the large box, then pull it into the opening for storage."
```

## Generalization Binding

Bind an unchanged learned primitive or deploy a frozen strategy to one new scene.
The required goal text guides semantic role selection and is stored as metadata;
generated runtime code is deterministic.

```bash
./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/instantiate_primitive_exec.py \
     --primitive push \
     --env-module tasks.invention_real2sim.push_variant_scene \
     --name push_variant \
     --goal "Push the target into the marked region."

./scripts/rapid-codex-run \
  --workspace "$RUN/workspace" \
  -- agent/deploy_strategy_exec.py \
     --strategy strategy_6 \
     --env-module tasks.invention_real2sim.new_scene \
     --name new_scene \
     --goal "Reorient the box and store it inside the corridor."
```

Primitive instantiation retains its invention-time smoke workflow. Strategy
deployment gives Codex a compact contract/scene snapshot and accepts only a
strict semantic role-mapping response. The host validates and writes the
canonical role-only manifest, then executes the frozen per-primitive policies
once with no scene-specific budget calibration. Binding defaults to
`gpt-6-luna` at low reasoning effort and allows at most one
validator-directed correction. Use `--bind-only` to omit execution.
`binding_valid`, strategy-owned `program_success`, benchmark-owned
`benchmark_success`, and audit advisories are reported separately;
metrics, video, and FR3 Cartesian goals live under
`workspace/results_instantiation/`.

For the common strategy-test workflow, use the combined helper:

```bash
./scripts/rapid-strategy-test \
  deploy_candidate_001 \
  tasks.invention_real2sim.new_scene \
  "Reorient the box and store it inside the corridor."
```

It resolves `suite_runs/deploy_candidate_001/workspace`, runs the binding agent,
then executes the generated strategy instantiation from that workspace. Final
metrics and recording remain under its `results_instantiation/` directory. It returns
nonzero for `program_success=False` unless `--allow-program-failure` is passed.

### Private full-arm benchmark scenes

Full-arm benchmark cases live under
`tasks/sim_benchmark/strategy_<n>/case_<nnn>_scene/`. Each case owns its
scene XML, metadata, and object assets, while all cases use the immutable
`runtime_assets/fr3/fr3_task_v1/` robot. The launcher grants read access only
to that exact asset version and never copies it into `task_inputs/`.

The initial `case_000_scene` packages are placeholders copied from the matching
`tasks.invention_real2sim.demo_<n>_scene`. They are not included in invention runs. Bind
one explicitly during benchmark deployment, for example:

```bash
./scripts/rapid-strategy-test \
  fr3_osc_demo_1_001 \
  tasks.sim_benchmark.strategy_1.case_000_scene \
  "Complete the benchmark task using the frozen strategy." \
  --strategy strategy_1 \
  --name benchmark_000
```

Use `rapid-strategy-test` only for an individual diagnostic case. The reusable
multi-case runner freezes the learned checkpoint hash, copies the source suite
to a disposable evaluation suite, and scrubs case-local state before every
run. Parallel jobs use independent worker workspaces, so the binding agent can
see only the selected task package and cannot accumulate earlier scenes:

```bash
./scripts/rapid-benchmark \
  fr3_osc_demo_1_001 \
  tasks/sim_benchmark/strategy_1 \
  --jobs 4
```

The runner verifies every sealed case before staging it and writes incremental
per-case results, recordings, an aggregate summary, and an HTML gallery under
`benchmark_runs/`. Its default `agent` binding mode runs the image-and-language
deployment binder from one rendered camera view plus the goal text. Use
`--binding-mode authoritative` when separately measuring strategy
generalization without semantic-binding errors. Both modes execute the frozen
strategy inside the workspace permission sandbox. Every benchmark primitive
phase uses `GLOBAL_MAX_OPTIMIZATION_BUDGET`; the primitive's predicate-gated
early-stop policy remains active. Individual `rapid-strategy-test` diagnostics
use retained phase/reference budgets unless
`--optimization-budget-mode maximum` is passed.

Official benchmark summaries use benchmark-owned success functions for all eight
strategies, while retaining each frozen strategy predicate only as a
disagreement diagnostic. Strategies 2, 3, 4, 7, and 8 succeed when both gripper
jaws retain target contact over the terminal window and the target remains at
least 1 cm clear of its support (the raised platform for strategy 8, the floor
for the others). Strategies 1, 5, and 6 succeed when the target center has
crossed the storage entrance and remains within the available slot or corridor
center region. The rear and lateral center boundaries use a 5 mm geometric
tolerance; the complete collision footprint remains diagnostic rather than a
success requirement. Benchmark runs retain `state_trajectory.npz` and compute
these decisions directly from MuJoCo states plus the sealed task layout;
strategy-owned measurements are not the oracle. Safety audits remain separate
from task success.

## Suite Modes

- **Fixed checkpoint**: every composition agent receives the same learned or
  merged workspace checkpoint. This measures strategy composition.
- **End-to-end**: each agent builds its own construction workspace, then uses
  that workspace directly or through a merge for composition. This measures the
  full pipeline.

Keep these modes separate in reports.

## Metrics

For each rollout, store at least:

- success boolean
- nominal success and nominal cost
- optimized cost and optimizer improvement
- nominal-to-optimized control RMS and maximum
- per-variant and, for strategies, per-phase results
- final task error
- seed
- optimizer name
- planning/rollout wall time if available
- metrics JSON path
- MP4 path if recorded

The shared instantiation runner supports:

```bash
--metrics-json /abs/path/metrics.json
```

Strategy scripts that do not use `instantiation_core.runner` should be
updated separately to write equivalent JSON if they become suite targets.
