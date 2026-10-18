# Simulation benchmark

This package generates, certifies, and scores the eight paper tasks. The current
corpus version is `real2sim_generalization_v3`: 50 fixed cases per task, 400 in
total. Each task has 12 diagnostic, 28 mixed, and 10 stress cases.

`private` means that the generator and evaluation metadata are hidden from
invention and deployment agents. They remain available to the host evaluation
runner. This directory contains runtime dependencies, not just cached results.

## Layout

| Path | Purpose |
| --- | --- |
| `specs.py` | Task contracts, deterministic sampling, and corpus version. |
| `strategies/strategy_1.py`–`strategy_8.py` | Task-specific scene layouts and relational constraints. |
| `generator.py`, `strategy_common.py` | Scene construction and packaging. |
| `assets.py`, `primitives.py`, `materials.py`, `physics.py` | Object geometry, material presets, and physical parameters. Here, `primitives.py` defines geometric solids, not learned manipulation primitives. |
| `certify.py` | Stability, geometry, task feasibility, and FR3 reachability checks. |
| `success.py` | Benchmark-owned success predicates and trajectory rescoring. |
| `sealed/real2sim_generalization_v3/` | Per-case private manifests, certification records, migration provenance, and the corpus seal. Generated locally and excluded from Git. |

Public scene packages live in `tasks/sim_benchmark/strategy_N/`.
Cases `001`–`050` are scored; `case_000_scene` is a reference placeholder.
The generated scenes, private manifests, and seal must correspond to each other.

## Evaluate an existing corpus

Run commands from the repository root after following [SETUP.md](../SETUP.md).
The wrappers use the `rapid` conda environment. A source checkout alone does not
include the generated corpus or every required asset. Restore and verify the
[simulation asset ZIP](../SETUP.md#simulation-assets) first.

First verify the existing corpus and its private records:

```bash
./scripts/rapid-python -m benchmark_private.cli audit
```

Evaluate a frozen learned strategy using an existing suite run:

```bash
./scripts/rapid-benchmark \
  fr3_osc_demo_1_001 \
  tasks/sim_benchmark/strategy_1 \
  --jobs 4
```

The runner checks sealed case hashes and uses `success.py` for official task
success. The learned strategy's own predicate is retained as a diagnostic.
See [SUITE_RUNS.md](../SUITE_RUNS.md) for binding modes, execution policies,
recordings, and result reporting.

## Generate or maintain the corpus

Generation requires the reference scenes under `tasks/invention_real2sim/`, the curated
HOPE assets, and the shared FR3 runtime assets. Sampling and layout constraints
are defined in `specs.py` and `strategies/`; material presets are defined in
`materials.py`.

```bash
# Generate and certify the full corpus.
./scripts/rapid-python -m benchmark_private.cli generate

# Rebuild the seal from existing certified cases and matching private manifests.
./scripts/rapid-python -m benchmark_private.cli seal

# Verify case hashes, manifest hashes, sampling fingerprints, and certification.
./scripts/rapid-python -m benchmark_private.cli audit
```

Use `generate --help` for task selection and output locations. Certification
checks scene feasibility without running the learned strategy under evaluation;
it does not establish a strategy success rate. Camera visibility is recorded as
a diagnostic, while physical and task constraints remain certification gates.

Keep generation code, private records, and preview outputs outside agent-visible
task packages. Agents receive only their assigned materialized scene.
