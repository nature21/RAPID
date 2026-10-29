# RAPID Environment Setup

This guide sets up RAPID on a new machine from a clean checkout. The supported
runtime is the conda environment named `rapid`; do not install `workspace/src`
as an editable package. RAPID's wrapper scripts set `PYTHONPATH` so the selected
workspace source tree is imported directly.

RAPID supports reproducing simulation experiments from supplied
demonstrations and reconstructed scenes. Reconstructing new scenes is covered
in [Reconstruct new scenes](#reconstruct-new-scenes); it is not needed to reproduce
the supplied simulation experiments. Real-robot interfaces remain available for
separately configured use.

The commands below target Linux. A machine with an NVIDIA GPU is recommended
for headless MuJoCo rendering, but the core suite
tooling can also run on CPU.

## 1. System prerequisites

Install these tools before creating the Python environment:

- Git with submodule support.
- Miniforge, Mambaforge, or another conda installation.
- A C/C++ build toolchain and CMake.
- Node.js 20 or newer if you will run the Codex invention agents.
- Current NVIDIA drivers if you will use CUDA or EGL rendering.

On Ubuntu or Debian, the basic system packages are:

```bash
sudo apt update
sudo apt install -y \
  build-essential git curl ffmpeg unzip \
  libegl1 libgl1 libglfw3 libosmesa6
```

Do not use the system Python or conda's `base` environment for RAPID commands.

## 2. Clone the repository

Clone with the Judo submodule:

```bash
git clone --recurse-submodules https://github.com/nature21/RAPID.git rapid
cd rapid
```

If the repository was already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

Confirm that `third_party/judo/judo/__init__.py` exists before continuing.
RAPID pins the matching source release from
[`nature21/Judo-osc`](https://github.com/nature21/Judo-osc). Use the committed
submodule revision; `git submodule update --remote` selects a different runtime.
GitHub's automatic source ZIP does not include submodule contents, so recursive
cloning is the supported installation path.

Restore the [simulation asset ZIP](#simulation-assets) into the checkout before
running tests or experiments. It supplies the non-Git benchmark scenes and
matching sealed records.

## 3. Create the `rapid` conda environment

From the repository root, create the environment with the checked-in
specification. It includes Python 3.13, GCC/G++ 13, the native build dependencies,
Python packages, and pytest. Create `rapid` only if an environment with that name
does not already exist:

```bash
conda env create -f environment.yml
```

The specification installs the Judo submodule as an editable package.
Do not use `pip install -e` for `workspace/src`. To update an existing `rapid`
environment with the required dependencies, use
`conda env update -n rapid -f environment.yml`.

When moving a development checkout or cloning an existing environment, rerun
`conda run -n rapid python -m pip install -e third_party/judo` from the new
repository root. It must reference
this checkout's `third_party/judo`, not a previous source directory. Do not
reuse a CMake build directory from another checkout or conda environment;
configure the native extensions in a fresh build directory below.

The repository wrappers automatically add both `workspace/src` and
`third_party/judo` to `PYTHONPATH`.

Tests use pytest, which also runs the existing unittest test cases. To add the
test runner to an existing `rapid` environment:

```bash
conda install -n rapid -c conda-forge -y 'pytest>=9,<10'
```

## 4. Build the required Judo C++ extensions

The FR3 experiments require these extensions for OSC rollouts and collision-aware
transition planning. Build them before running tests or experiments:

```bash
conda run -n rapid cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -B third_party/judo/mujoco_extensions/build \
  -S third_party/judo/mujoco_extensions
conda run -n rapid cmake \
  --build third_party/judo/mujoco_extensions/build \
  -j4
```

The build writes `policy_rollout_pybind*.so`, `fr3_osc_rollout_pybind*.so`, and
`fr3_transit_planner_pybind*.so` under their corresponding directories in
`third_party/judo/mujoco_extensions/`.

### Optional feature dependencies

Install only the groups needed on the new machine.

#### MuJoCo Warp GPU rollouts

Sampling-based primitive compilation uses MuJoCo's CPU rollout backend by
default. On an NVIDIA CUDA machine, install MuJoCo Warp to evaluate all
optimizer candidates as one GPU batch:

```bash
conda run -n rapid python -m pip install \
  'mujoco_warp>=3.6,<3.7' 'warp-lang==1.12.1'
conda run -n rapid python -c \
  'import warp as wp; print(wp.__version__); print("CUDA:", wp.is_cuda_available())'
```

Keep the Warp pin: it matches Judo's lockfile. Newer Warp releases are not
necessarily source-compatible with MuJoCo Warp 3.6's generated sensor kernels.
The first run for a new model compiles CUDA kernels and can be noticeably
slower; later runs reuse the Warp kernel cache.

Select it for an instantiation with either the CLI or environment:

```bash
./scripts/rapid-python -m instantiations.primitives.<primitive>.example_0.main \
  --rollout-backend mjwarp

RAPID_ROLLOUT_BACKEND=mjwarp \
  ./scripts/rapid-python -m instantiations.primitives.<primitive>.example_0.main
```

Use `--rollout-device cuda:1` or `RAPID_ROLLOUT_DEVICE=cuda:1` to select a
different GPU. `RAPID_MAX_PARALLEL_ROLLOUTS` applies only to CPU rollout;
when unset, RAPID respects process affinity, common scheduler allocations, and
cgroup quotas, caps the pool at 48 workers, and prefers an even population
chunk (the default 128 rollouts use 32 workers on a large server). An explicit
value remains an upper bound. MJWarp advances all `num_rollouts` worlds
together. Backend selection is explicit: requesting MJWarp without its package
or a CUDA device exits with an actionable error instead of silently falling
back to CPU.

## 5. Install and authenticate Codex CLI

This step is required for `scripts/rapid-codex-run`, `run_primitive_construction.sh`, and
`run_strategy_composition.sh`. It is not required merely to run existing simulations.

Install Codex CLI with npm:

```bash
npm install -g @openai/codex@0.142.5
codex --version
```

Use the tested CLI version above for reproduction. Start Codex once and follow
its sign-in flow. File-based credential storage lets RAPID copy authentication
into each isolated run:

```bash
codex -c cli_auth_credentials_store='"file"' login
```

RAPID launches Codex through `scripts/rapid-codex-run`, which prepares a
workspace-local configuration and trusted `CODEX_HOME`. If your user config sets
`cli_auth_credentials_store`, use `"file"` there too. See the official
[authentication guide](https://developers.openai.com/codex/auth) for API-key and
headless login options. Model access depends on your account; confirm access to
both the chosen invention model and the deployment binder before a full run.

## 6. Validate the installation

Run all commands from the repository root unless a command says otherwise.

Inspect the active source and Python configuration:

```bash
./scripts/rapid-python --rapid-env
```

The output should report:

- conda environment: `rapid`
- workspace: `<repo>/workspace`
- active source: `<repo>/workspace/src`
- source loading: `PYTHONPATH`

Check the main Python imports:

```bash
./scripts/rapid-python -c '
import imageio, mujoco, numpy, scipy
from ompl import base, control
import judo
from mujoco_extensions.policy_rollout import System
from mujoco_extensions.fr3_osc_rollout import NativeFR3OSCRollout
from mujoco_extensions.fr3_transit_planner import NativeFR3TransitKinematics
print("RAPID core imports: OK")
print("MuJoCo:", mujoco.__version__)
print("Judo:", judo.__file__)
'
```

Check the suite entry point:

```bash
./scripts/rapid-suite --help
```

Run the repository tests:

```bash
./scripts/rapid-python -m pytest -q
```

Before running invention agents, check the Codex sandbox without a model call:

```bash
./scripts/rapid-suite create-workspace --run-dir suite_runs/setup_check
./scripts/rapid-codex-run --workspace suite_runs/setup_check/workspace \
  --no-codex-auth --probe-permissions
```

The probe should print `rapid-permission-probe-ok`. Keep the checkout, conda
environment, and CLI installation outside `/tmp`, which the experiment sandbox
deliberately denies.

## 7. Run RAPID

Use the provided wrappers rather than plain `python`.

Create an isolated suite workspace:

```bash
./scripts/rapid-suite create-workspace \
  --run-dir suite_runs/example_001
```

Run a generated instantiation from the canonical workspace:

```bash
cd workspace
conda run -n rapid python \
  -m instantiations.primitives.<primitive>.example_0.main
```

Run a standard construction invention experiment:

```bash
cd ..
./scripts/rapid-construct-primitive \
  --run fr3_osc_primitive_push_001 \
  --demo demos/primitive/push/rgb.mp4 \
  --env-module tasks.invention_real2sim.push_scene \
  --name push \
  --num-variants 2
```

See `scripts/README.md` and `SUITE_RUNS.md` for the available launchers and the
construction/deploy workflow.

## 8. Headless rendering and troubleshooting

### EGL errors

`scripts/rapid-python` defaults to:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

EGL is the preferred mode for an NVIDIA headless machine. Confirm that the
driver is visible with `nvidia-smi`. If the machine has no working EGL device,
try CPU software rendering for a smoke test:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  ./scripts/rapid-python -c 'import mujoco; print(mujoco.__version__)'
```

For an interactive desktop viewer, use GLFW instead:

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx ./scripts/rapid-view
```

### `conda not found`

Initialize conda for the current shell or set `CONDA_EXE` explicitly:

```bash
export CONDA_EXE="$HOME/miniforge3/bin/conda"
./scripts/rapid-python --rapid-env
```

### Wrong imports or stale packages

Do not install the workspace source tree into site-packages. Inspect the active
paths with:

```bash
./scripts/rapid-python --rapid-env
```

For an isolated suite run, switch the shell to that workspace with:

```bash
eval "$(./scripts/rapid-workspace-env suite_runs/example_001)"
./scripts/rapid-python --rapid-env
```

`RAPID_ACTIVE_SRC_DIR` and `RAPID_SRC_DIR` must point to the same source tree.

### Missing Judo extension

If an error mentions `mujoco_extensions`, `policy_rollout_pybind`, or
`fr3_osc_rollout_pybind`, or `fr3_transit_planner_pybind`, complete the C++
extension build in section 4. A plain `import judo` does not prove that the
compiled extensions have been built.

### Video encoding errors

Ensure both the system `ffmpeg` executable and the Python package are present:

```bash
ffmpeg -version
conda run -n rapid python -m pip install imageio-ffmpeg
```

## Simulation assets

Download the [simulation asset ZIP](https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip)
and its [SHA-256 checksum](https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip.sha256).
The ZIP contains HOPE source models, curated visuals, all 400 sealed benchmark
scenes, and their matching evaluation records. Demonstrations, reconstructed
invention scenes, and FR3 runtime assets already tracked in Git come with the
checkout. No reconstruction service is needed to use them.

```bash
# In a download directory:
curl -fLO --retry 3 https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip
curl -fLO --retry 3 https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip.sha256
sha256sum -c rapid-assets-20261017.zip.sha256
unzip -n rapid-assets-20261017.zip -d /absolute/path/to/rapid

# In the repository root, after environment setup:
./scripts/rapid-python -m devtools.release_assets verify
./scripts/rapid-python -m benchmark_private.cli audit
./scripts/rapid-python -m pytest -q
```

Extract directly into the repository root, without another enclosing folder.
The manifest checks every asset's SHA-256; the corpus audit checks scene and
certification consistency. If verification fails, use a clean checkout and
matching ZIP rather than mixing versions. Keep this fixed corpus for experiment
comparisons. Private benchmark records are available to the host evaluator but
hidden from agents by the existing launchers.

RAPID code uses the [MIT license](LICENSE). Third-party code and assets retain
their upstream terms: Judo's license is in `third_party/judo/LICENSE`, and the
FR3 asset license is in `runtime_assets/fr3/fr3_task_v1/LICENSE`. RAPID's license
does not relicense third-party assets, including HOPE models.

## Reproduce the experiments

Local and Slurm runs use the same invention entrypoints, prompts, scene variant
workflow, search budgets, and evaluator. Each full trial invents six primitives,
composes eight strategies, and evaluates 50 cases per strategy. Invention uses
`gpt-6-astra` with `high` reasoning effort; deployment binding uses `gpt-6-luna`
with its default `low` effort. Evaluation uses seed 42, maximum optimization budgets, and the
`attempt` execution policy. Model access and Codex authentication are required;
sampling and numerical variation mean reruns need not reproduce identical
programs or success rates.

### One machine, without Slurm

```bash
# Inspect commands without starting agents or creating a run:
./scripts/rapid-experiment-local --experiment-root astra_local --dry-run

# Run a full trial:
./scripts/rapid-experiment-local --experiment-root astra_local
```

Defaults are 32 CPU workers and NVIDIA rendering, with 64 GB RAM recommended.
Invention runs one task at a time; evaluation runs four cases concurrently with
eight rollout workers each. On a smaller machine, use e.g.
`--rollout-workers 8 --eval-workers 2`; add `--gpu-render off` for software
rendering without NVIDIA. The worker budget must be at most 48 and divisible
by the evaluation worker count. These options change parallelism and rendering,
not optimization budgets. Keep the foreground shell open or use `tmux`.

Use fresh experiment names for new trials. The helper stops when a stage fails;
inspect its logs, then resume at the appropriate stage:

```bash
# Requires all six primitive workspaces:
./scripts/rapid-experiment-local --experiment-root astra_local \
  --stage compose
# Requires frozen strategies; completed eval cases are reused:
./scripts/rapid-experiment-local --experiment-root astra_local --stage eval
# Evaluate only strategy 8:
RAPID_EVAL_STRATEGIES=8 ./scripts/rapid-experiment-local \
  --experiment-root astra_local --stage eval
```

Other options include `--stage construct`, `--batch 002`,
`RAPID_CONSTRUCTION_TASKS`, and `RAPID_COMPOSITION_RUNS`. Do not reuse a populated
composition directory for a new invention trial. Successful strategy siblings
can be evaluated explicitly with `RAPID_EVAL_STRATEGIES` after a partial failure.

### Slurm cluster

Keep the existing cluster entrypoints: `launch_experiments.sh`,
`launch_evals.sh`, and the three `run_*.sh` stage scripts. Array elements request
32 CPUs, 64 GB RAM, and one GPU. Override the development node pools (`crane[0-2]`)
with your cluster's names; set `SBATCH_ACCOUNT` / `SBATCH_PARTITION` if required.
Optional notifications use `SBATCH_MAIL_USER` and `SBATCH_MAIL_TYPE`.

```bash
export RAPID_EVAL_NODES='gpu01 gpu02 gpu03'
export RAPID_EXPERIMENT_THINKING_EFFORT=high
export RAPID_COMPOSITION_MODEL=gpt-6-luna
./launch_experiments.sh \
  --nodes 'gpu[01-03]' --experiment-root astra_cluster 001
```

Composition waits for construction success and evaluation waits for its
corresponding strategy.
See [scripts/README.md](scripts/README.md) for subset runs and resource options.

### Results

Outputs live in `suite_runs/<experiment>/`. `benchmark_success` is the official
task outcome; `program_success`, behavior checks, and audit advisories are
separate signals. Per-case artifacts include metrics, available trajectories
and videos, and `execution.log`, including exceptions before metrics exist.

The local helper creates an HTML index after evaluation. For Slurm runs:

```bash
./scripts/rapid-python -m vistools.build_suite_runs_gallery \
  --experiment-root astra_cluster
```

Open `suite_runs/astra_cluster/index.html` to browse the results.

## Reconstruct new scenes

The `real2sim/` pipeline accepts an RGB image, the first frame of a video, or a
calibrated RGB-D directory. It identifies objects with a VLM, segments them with
SAM3, reconstructs their meshes with SAM 3D Objects, estimates their poses with
FoundationPose, and packages a MuJoCo task scene. RGB inputs additionally use
Depth Anything 3 for depth and camera estimation. A calibrated RGB-D input uses
its measured depth and calibration directly.

### Client dependencies and VLM configuration

Run reconstruction in the `rapid` environment. Install Open3D for mesh
simplification and ICP, and the camera package only if capturing new data:

```bash
conda install -n rapid -c conda-forge open3d
# Only for a connected RealSense camera:
conda run -n rapid python -m pip install pyrealsense2
./scripts/rapid-real2sim --help
```

Configure a vision-capable OpenAI-compatible model endpoint. These credentials
are separate from the Codex login used by invention and scene-repair agents:

```bash
export REAL2SIM_VLM_BACKEND=CUSTOM
export CUSTOM_OPENAI_BASE_URL="https://your-provider.example/v1"
export CUSTOM_OPENAI_API_KEY="..."
export RAPID_VLM_MODEL="your-vision-model"
```

For the OpenAI endpoint, set `REAL2SIM_VLM_BACKEND=OPENAI` and `OPENAI_API_KEY`
instead. Pass the model name with `--vlm-model`; if the provider does not support
reasoning effort, also pass `--vlm-reasoning-effort none`.

### Install the model services

The adapters in `server/` use upstream model installations and checkpoints.
Install each model in the environment or container recommended by its project;
these GPU dependencies are independent of RAPID's `rapid` environment.

| Service | Upstream installation and weights | Adapter | Default port |
| --- | --- | --- | --- |
| SAM3 | [SAM3](https://github.com/facebookresearch/sam3#installation) | `server.sam3_server` | 4002 |
| SAM 3D Objects | [SAM 3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) | `server.sam3d_server` | 4001 |
| FoundationPose | [FoundationPose](https://github.com/NVlabs/FoundationPose) | `server.foundation_pose_server` | 4004 |
| Depth Anything 3 | [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3#installation) | `server.depth_anything_server` | 4003 |

Follow the upstream checkpoint access/download instructions, including any
Hugging Face access requirements. Model code and weights remain subject to their
upstream licenses; this repository does not redistribute them. The SAM3, SAM 3D
and FoundationPose HTTP interfaces follow the service pattern in
[GROW²](https://github.com/dengyh16code/improvised_tooluse/tree/main/server).
FoundationPose works with its official single-pose return API; no estimator
source patch is required.

In each model's activated environment, install the lightweight adapter dependencies:

```bash
python -m pip install -r /absolute/path/to/RAPID/server/requirements.txt
```

Run each service in a separate terminal, from the RAPID repository root, using
that model's environment. Select the GPU with `CUDA_VISIBLE_DEVICES` if needed.
The examples below use localhost and match the reconstruction client's defaults:

```bash
# SAM3 environment; --checkpoint /path/to/sam3.pt can select a local checkpoint.
python -m server.sam3_server --port 4002

# SAM 3D Objects environment, with its checkpoint configuration downloaded.
python -m server.sam3d_server \
  --repo /absolute/path/to/sam-3d-objects \
  --config checkpoints/hf/pipeline.yaml --port 4001

# FoundationPose environment, with its weights/ populated as documented upstream.
python -m server.foundation_pose_server \
  --repo /absolute/path/to/FoundationPose --port 4004

# DA3 environment; omit this service when supplying calibrated RGB-D inputs.
python -m server.depth_anything_server \
  --model depth-anything/DA3NESTED-GIANT-LARGE-1.1 --port 4003
```

Each process loads its model before serving requests and processes one inference
at a time. Check readiness with, for example:

```bash
curl --fail http://127.0.0.1:4002/health
```

For services on another machine, bind with `--host 0.0.0.0` on a private network
and point the client at that machine's reachable address. Full endpoint options
are `--sam3-url`, `--sam3d-url`, `--foundation-pose-url`, and `--depth-url`.
For example:

```bash
./scripts/rapid-real2sim capture/ --name captured_scene \
  --vlm-model "$RAPID_VLM_MODEL" \
  --sam3-url http://gpu.example.org:4002/sam3_query \
  --sam3d-url http://gpu.example.org:4001/sam3d_query \
  --foundation-pose-url http://gpu.example.org:4004/fp_registration_query
```

Alternatively set `RAPID_SAM3_HOST` / `RAPID_SAM3_PORT`,
`RAPID_SAM3D_HOST` / `RAPID_SAM3D_PORT`,
`RAPID_FOUNDATION_POSE_HOST` / `RAPID_FOUNDATION_POSE_PORT`, or
`RAPID_DEPTH_ANYTHING_HOST` / `RAPID_DEPTH_ANYTHING_PORT` before starting the
client. Explicit URL flags take precedence. Services can also be forwarded to
the default local ports with SSH tunnels.

### Reconstruct and inspect a scene

A calibrated RGB-D input directory contains `color.png`, metric `depth.npy`,
`intrinsics.json`, and an Easy Handeye `*_eob_calib.yaml`. Use a new package
name so the supplied experiment scenes remain intact:

```bash
./scripts/rapid-real2sim demos/primitive/push \
  --name reconstructed_push --vlm-model "$RAPID_VLM_MODEL" --preview
```

For an RGB image or a video, start DA3 as well:

```bash
./scripts/rapid-real2sim /path/to/frame.png \
  --name reconstructed_frame --vlm-model "$RAPID_VLM_MODEL" --preview
```

Reconstruction artifacts are written to `results/real2sim/<name>/`, and packaged
scenes to `tasks/invention_real2sim/<name>/`. Packaging checks visual/collision
alignment, compiles the MuJoCo model, and settles it before saving its reset state.
The optional scene-repair agent uses the included
[`repair-real2sim-scene` skill](real2sim/.agents/skills/repair-real2sim-scene/SKILL.md).
Use `--no-agent-fallback` to disable agent repair; `--no-package` keeps only the
reconstruction bundle. To omit FoundationPose, use `--no-foundation-pose`; this
uses SAM 3D placement, with local ICP when the support correction triggers it,
and can change reconstruction quality. Use `--no-fp-fallback` to disable SAM 3D
placement as a fallback when FoundationPose fails. By default, failed objects
are recorded in `skipped_objects` and reconstruction continues with the remaining
objects. When using the Python API, set
`PipelineConfig(continue_on_object_error=False)` to stop on an object failure.

```bash
./scripts/rapid-view invention_real2sim/reconstructed_push
```

### Test the interfaces without model weights

The following tests start local HTTP servers with deterministic model substitutes,
call them through the production clients, and exercise both RGB and RGB-D inputs
through scene export, packaging, and MuJoCo simulation. They also check malformed
requests, depth/camera scaling, and FoundationPose tracking-frame conventions.
They do not establish reconstruction accuracy or validate GPU model installations.

```bash
./scripts/rapid-python -m pytest -q \
  tests/test_reconstruction_servers.py tests/test_vfm_client_config.py
./scripts/rapid-python -m pytest -q
```
