# RAPID (Preview)

**Robot Agentic Programming from Demonstrations**

[Paper](https://arxiv.org/abs/2609.30249) · [Environment setup](SETUP.md) ·
[Experiment workflow](SUITE_RUNS.md) · [Visualization tools](vistools/README.md) ·
[Reproduce experiments](SETUP.md#reproduce-the-experiments) ·
[MIT license](LICENSE)

RAPID constructs reusable manipulation programs from demonstrations and
language instructions. This repository contains the FR3 nonprehensile
manipulation pipeline and staged construction and evaluation
infrastructure. The clean source tree contains shared infrastructure; concrete
primitives and strategies are generated in isolated experiment workspaces.

This preview includes demonstrations, reconstructed scenes, and the tools to
reproduce agent invention and simulation evaluation, locally or with Slurm.
The Real2Sim pipeline and reconstruction servers are not included; the supplied
scenes allow these experiments to run without reconstruction services. The matching
[Judo-osc](https://github.com/nature21/Judo-osc) runtime is pinned as a submodule.

## Program representation

| Term | Implementation | Meaning |
| --- | --- | --- |
| Primitive | `workspace/src/primitives/<name>/` | Reusable manipulation program with a relational interface, geometry resolver, objective and success predicate. |
| Strategy | `workspace/src/strategies/<name>/` | Ordered primitive calls with semantic role maps and connecting constraints resolved from the live phase-start state. |
| Instantiation | `workspace/src/instantiations/` | Concrete scene-role bindings; primitive invocations also supply effect arguments. |

Shared compilation and control code lives in `primitive_core`. Shared binding
contracts and execution utilities live in `instantiation_core`.

## Start here

Follow [SETUP.md](SETUP.md) to create the **`rapid` conda environment** from
[environment.yml](environment.yml) and build the required native extensions.
From the repository root:

```bash
./scripts/rapid-python --rapid-env
./scripts/rapid-suite --help
./scripts/rapid-python -m pytest -q
```

The workflow has four stages:

1. Install the supplied demonstrations, reconstructed scenes, and benchmark
   assets following [SETUP.md](SETUP.md#simulation-assets).
2. Construct primitives with `scripts/rapid-construct-primitive` or the batch
   launcher `run_primitive_construction.sh`.
3. Compose strategies with `agent/compose_strategy_exec.py` or the batch
   launcher `run_strategy_composition.sh`.
4. Bind and execute frozen programs on new scenes using the instantiation and
   deployment helpers; use `run_eval.sh` / `launch_evals.sh` for benchmark evaluation.

Use `scripts/rapid-experiment-local` for a complete trial without Slurm, or
`launch_experiments.sh` for cluster job arrays. Asset installation and both
workflows are documented in [SETUP.md](SETUP.md#reproduce-the-experiments).
The batch launchers contain configurable development-cluster resource settings.
During strategy construction,
evidence-backed reusable primitive repairs remain supported and require fresh
qualification; held-out deployment keeps the learned programs frozen.

## Tasks

| Task / demo | Sequence |
| --- | --- |
| 1 | push → push |
| 2 | push → flip → grasp |
| 3 | press_move → push → flip → grasp |
| 4 | push → flip → press_move → grasp |
| 5 | push → flip → press_move |
| 6 | pivot → push → press_move |
| 7 | topple → grasp |
| 8 | press_move → grasp |

## Assets and results

Recorded demonstrations and scene meshes/textures are inputs. Some required
assets and generated benchmark scenes are excluded from Git. Restore the
separate [simulation asset ZIP](https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip)
and [SHA-256 checksum](https://github.com/nature21/RAPID/releases/download/v0.1.0-preview.1/rapid-assets-20261017.zip.sha256),
following the [installation instructions](SETUP.md#simulation-assets).

Strategy demonstrations live in `demos/strategy/demo_N/`; primitive
demonstrations live in `demos/primitive/<name>/`. Both include tracked
first-frame metric depth in `depth.npy`.
The preview includes pre-segmented demonstration clips; the
[segmentation agent and clip-generation code](demos/strategy/.agents/skills/segment-demo-videos/)
are also provided.

Generated workspaces, experiment logs, and result galleries live under
`suite_runs/` and are excluded from Git.

## TODO

- [ ] Release the Real2Sim pipeline and reconstruction server implementations.
