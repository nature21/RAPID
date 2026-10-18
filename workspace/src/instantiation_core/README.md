# Instantiation Core

`instantiation_core` contains shared typed contracts, MuJoCo introspection,
scene inventory, validation, and runner utilities used by generated
instantiations.

Learned primitives and strategies publish binding contracts. Concrete scene
names live under `instantiations/`. Primitive bindings also contain complete
effect-goal parameters; strategy bindings are role-only. Relational layout
derivation remains in learned `features.py` and connecting constraints.

Contracts declare typed semantic roles plus cardinality, MuJoCo entity types,
body/joint structure, and `RelationalInvariant` relationships. A generated
`SceneBinding` supplies concrete names plus structured primitive parameters.
Strategy `SceneBinding.parameters` is empty.
`validate_scene_binding()` resolves both against the compiled model and returns
a `BindingReport`.

Runtime reports intentionally keep these outcomes separate:

- `binding_valid`: the scene mapping and execution wiring are valid.
- `program_success`: the unchanged learned-program predicate passed.

When `RAPID_INVENTION_ATTEMPT_LOG` is set, the runner also records the effective
budget, transitive design fingerprint, and execution fingerprint. Strategy
design fingerprints include the strategy, its actual primitive dependencies,
binding contracts, transition code, and shared runtime; execution fingerprints
include the concrete binding and optimizer/runtime arguments. Nominal and 1x1 diagnostics are
not retained evidence. A failed tier is not repeated under unchanged source;
minimum and standard failures may advance when the mechanism remains plausible.
An expanded-tier or run-ceiling failure exhausts search for that source.

Generated `main.py` files may retain only parser-supported optimizer/resource
defaults proven by a successful bounded smoke. They cannot change task policy,
geometry, learned knot topology, or output paths, and explicit CLI arguments
still take precedence. Plain module executions retain metrics and video under
`$RAPID_WORKSPACE/results_instantiation/`.

For FR3 OSC instantiations, that directory also contains `cartesian_goals.json`.
It is one `rapid.fr3_cartesian_goals` version-2 document containing the 50 Hz
controller-commanded pose, total gripper width, explicit `move`/`grasp`
operation, and embedded controller/gripper parameters. Orientations use
`[qx, qy, qz, qw]`, and poses are expressed in `fr3_link0`. It is derived from
retained OSC controls, never from the achieved simulator pose.

Large numeric metrics (more than 256 values) are retained in the adjacent
`metrics.arrays.npz`; JSON records the archive name, array key, shape and dtype.
The in-memory result keeps the original arrays, and benchmark archival copies
the compressed evidence alongside the JSON. Console metrics use bounded
representations. `timings_inclusive` records compiler and transit costs;
strategy phases also record their elapsed wall time.

## Fixed qualification probes

`qualification.freeze_validation_cases` preserves declared diagnostic inputs.
`run_validation_suite` executes typed `GeometryProbe`, `SceneInterfaceProbe`
and `HandoffProbe` checks and fingerprints both learned and probe sources.
Geometry probes compare measured positions, directions, scalars and decisions
with a distinct negative control. Interface probes execute the public manifest
entrypoint at a two-rollout, one-iteration budget, with temporary artifacts.

During invention, the sequential runtime saves actual noninitial phase states
under `agent_work/handoffs/<program>/<instance>/<phase>/`. `HandoffProbe` replays
the selected boundary through the same dispatcher, transit and next primitive,
preserving controller state and explicit tangent-space configuration
perturbations. Every required example and boundary needs coverage. Missing
checkpoints are incomplete; stale checkpoints require a development rerun.
Low-budget effect success is recorded separately from precontact reachability.
Probe executions do not consume the retained behavioral search ledger.

## Generalization scenes

Primitive invention keeps `example_0` as the assigned baseline and represents
every example with `generalization.SceneVariantSpec`. The shared helper builds
a fresh assigned-scene `MjSpec`, scales target collision and visual geometry,
updates mass and inertia, changes contact friction and compatible fixture
placement, then applies the target free-joint translation and full-circle yaw
after keyframe reset. `balanced_generalization_samples` supplies reproducible
stratified samples across these dimensions and declared goal parameters.
Every primitive example records its complete invocation in
`SceneVariantSpec.goal_parameters`, exactly matching the binding inputs.
`varied_goal_parameters` identifies values that differ from `example_0`, so a
generalization set can separately cover geometry changes and goal changes.
Strategy variants keep both mappings empty because connecting constraints
resolve phase goals from their role-bound layouts at execution time.
When yaw represents a global task-layout variation, `yaw_coupled_bodies` names
fixture companions that rotate rigidly about the target's baseline origin so
the target/fixture affordance and relative orientation remain intact. A
fixture-bearing yaw variant must use this coupling unless it deliberately sets
`allow_independent_fixture_yaw=True` to test relative yaw.

Generated scene modules export `VARIANT_SPEC`, `make_spec`, `make_model`, and
`make_data`. Their bindings point back to those generated modules and attach
`generalization_metadata`, which records both a physical-scene fingerprint and
a goal-inclusive runtime fingerprint in retained metrics.

Initial-state validation uses two equal simulation windows. The first permits
the one-time gravity/contact relaxation common to imported reconstructions; the
second measures continued translation, rotation, and terminal free-joint
speed. A scene that lands and converges is valid, while continuing fall,
rotation, or other residual motion remains a pre-simulation failure. The
validation rollout is diagnostic and never replaces the generated scene's
declared reset state.
