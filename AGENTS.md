# Project Instructions

This root file guides Codex sessions that are developing RAPID itself. Automatic
Codex invention runs use a separate workspace-local `AGENTS.md` staged into
`$RAPID_WORKSPACE` by the exec launchers, plus workflow skills under
`$RAPID_WORKSPACE/.agents/skills/`.

## Python Environment

Project code is loaded by putting the active source tree on `PYTHONPATH`
through the wrapper scripts. The only supported runtime is the conda
environment named `rapid`; the user's base conda environment is not a supported
runtime. The external GPU model services under `server/` use their upstream
model environments; their HTTP interface tests run in `rapid` without model
weights. Do not install model-specific CUDA stacks into the RAPID environment.

Concrete primitive implementations created by agents live under
`workspace/src/primitives/<primitive>/`. The clean baseline has no preinstalled
concrete primitives. Shared primitive contracts and compilation infrastructure
live under `workspace/src/primitive_core/`.

### Running Commands

Always run Python through the conda environment named `rapid`. From the repo
root, use `conda run`:

```bash
conda run -n rapid python ...
conda run -n rapid python -m suites.staged.cli --help
./scripts/rapid-suite --help
```

Equivalently, activate the environment first:

```bash
conda activate rapid
python ...
```

For commands that write relative output paths, prefer running from
`workspace/` so recordings and generated artifacts stay under the workspace
layout. Generated validation scenes live under `test_scenes/` in the active
workspace, not under `workspace/src`. For isolated suite workspaces, set
`RAPID_WORKSPACE` and `RAPID_ACTIVE_SRC_DIR` explicitly for the generated layout, and mirror
`RAPID_SRC_DIR` to the same source tree for compatibility.

Use the wrapper to inspect the active source/import configuration:

```bash
./scripts/rapid-python --rapid-env
```

During isolated runs, `RAPID_ACTIVE_SRC_DIR` is prepended to `PYTHONPATH` and is
the source tree Python imports from. Do not use `pip install -e` for
`workspace/src` or suite workspaces.

### Running Generated Primitive Code

```bash
cd workspace
conda run -n rapid python -m instantiations.primitives.<primitive>.example_0.main
conda run -n rapid python -m instantiations.primitives.<primitive>.example_0.main --optimizer cem
conda run -n rapid python -m instantiations.primitives.<primitive>.example_0.main --record results/<primitive>_example_0.mp4
```

Use module form. Do not execute files by path from the base conda
environment; direct-file execution can cause `ModuleNotFoundError` or
stale-install bugs.

Learned primitives and strategies are scene-agnostic and publish
`binding_contract.py`. Concrete scene names and mappings live under
`src/instantiations/`. For suite-stable code, import the named primitive package,
for example `primitives.<primitive>`.

### Installing New Dependencies

Install new dependencies into the `rapid` conda environment, not into the base
environment. Prefer updating the tracked conda environment specification when
one exists; otherwise use `conda install -n rapid ...` for conda packages, or
`conda run -n rapid python -m pip install ...` only for Python packages that are
not available through conda.
