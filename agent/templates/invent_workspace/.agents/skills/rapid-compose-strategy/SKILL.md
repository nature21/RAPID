---
name: rapid-compose-strategy
description: Invent and freeze a scene-agnostic multi-phase RAPID strategy that owns phase order, semantic primitive mappings, live phase-input composition constraints, typed transit intent, and final Psi.
---

# Invent and freeze a relational strategy

The learned strategy must be the plan. It sequences reusable learned
primitives, maps semantic roles, declares relational connecting constraints and
typed effect-free transits, propagates physical/controller state, and evaluates
the final object-centric predicate. A scene binding contains semantic roles,
not numeric phase goals or executable adapter code.

An invention is complete only when the same learned strategy succeeds on the
assigned demo scene and every requested geometry/pose variant (at least two), with
strict data-only manifests, retained per-phase metrics, complete replay, and
visual verification.

## Read first

Start with the run prompt, demo, assigned scene inventory/module, workspace
`AGENTS.md`, `src/strategies/explanation.md`, and available primitive
descriptions/contracts. Inspect the public entrypoint only for primitives that
semantically match an inferred phase. Use `rg` to locate other required APIs
and read their definitions on demand; do not preload every primitive or whole
runtime, rollout, and telemetry modules.

Task environments copied under `task_inputs/` are allowed inputs, not learned
implementations. Do not inspect hidden baselines or sibling runs.

## Non-negotiable ownership boundary

The strategy owns and freezes:

- ordered primitive phases and explicit phase subgoals;
- primitive-role to strategy-role mappings;
- connecting constraints that derive every primitive effect-goal parameter from
  live bound geometry and the propagated phase-start state;
- typed inter-phase `TransitSpec` values and their `ScaleExpr` coefficients;
- physical-state and controller-state handoff requirements;
- final object-centric predicate `Psi` and its tolerances;
- phase-boundary and whole-plan smoothness/safety checks.

Each primitive continues to own its relational geometry resolver, effect-knot
topology and nominal values, sigma, cost, effect predicate/tolerances, and
precontact geometry and transit for standalone and composed execution.
A strategy may compose these capabilities but may not reproduce their motion
heuristics in phase wrappers.

Strategy bindings own only identity, scene/goal metadata, and semantic names.
The strategy `BINDING_CONTRACT` has `parameters=()`. Bindings may not contain
phase bindings, phase compilers, geometry callbacks, derived goals, predicates,
waypoints, axes, action arrays, motion/clearance/timing coefficients, transit
execution, costs, or tolerances.

Each declared primitive parameter has a learned `ConnectingConstraint`. Resolve
it once from the bound roles and current propagated state immediately before
the phase. The resolved value exists only in the ephemeral primitive binding.
Geometry and relations intrinsic to an primitive's contact mechanism remain
inside that primitive and are resolved from its bound roles and live state.
The strategy supplies the requested effect through the declared parameter
interface.

When a validated manifest exposes a behavioral failure, repair the learned
strategy program, predicate, or transit. Existing learned primitives are
editable development inputs when retained evidence justifies the smallest
reusable contract or implementation repair. Validate and record every such
change. Never repair the rollout in an instantiation.

## Step 1 - Analyze the demonstration and scene

Inspect supplied frames and extract additional frames when phase ordering,
contact timing, or motion direction is uncertain. Treat the user's
natural-language description as qualitative task semantics. Interpret it
together with the demonstration and assigned scene, then propose the goal `G`,
phase subgoals, and final measurable `Psi` yourself, including the necessary
measurable formulation and dimension-relative coefficients. Commit them in
`strategy_description.md` before the first simulator run. Predicate calibration
follows the evidence protocol under **Final predicate**. Then record:

1. Tool/end-effector, every manipulated object, supports, fixtures,
   receptacles, protected objects, free-space corridors, and final goal.
2. Ordered contact and effect sequence, including free motion, acquisition,
   maintained contact, release, and reorientation.
3. Strategy goal `G`, phase subgoals, and every conjunct of final `Psi`.
4. Relational phase signature for each target-changing phase—effect,
   goal/reference frame, orientation policy, contact mode, and terminal
   relation—followed by the candidate learned primitive and why every component
   matches.
5. Strategy semantic roles and the primitive-role mapping needed by each phase.
6. For every primitive parameter, the strategy-owned connecting constraint that
   derives it. State each equation in semantic roles and identify the live
   phase-start state, frame, units, and geometry it reads.
7. Effect-free transitions, their relational intent, protected objects, and
   any bounded inherited-contact requirement.
8. State/controller continuity requirements at every phase boundary.
9. Per-phase and whole-plan smoothness, collision, contact, and safety signals.
10. Task-layout groups that must transform rigidly during generalization.

### Scene fidelity sketch

Write an object-centric scene sketch before constructing variants. Describe
entity roles, relative side/order, adjacency, containment, support relations,
fixture alignment, aspect ratios, contact features, clearance corridors, and
the demonstrated initial-to-final relation. Express invariants as relational
statements or inequalities wherever practical.

Do not silently mirror, rotate, simplify, or resize the scene into an easier
task. World gravity and support normals are acceptable invariants. Arbitrary
world axes and copied absolute demo poses are not.

Before declaring an primitive missing, separate its intrinsic effect from the
strategy's choice of requested subgoal. A bound entity or a preceding phase's
result may determine the desired outcome without participating in the
primitive's contact mechanism.

When an existing primitive can express that outcome through its declared
parameters, reuse it. Define a learned connecting constraint that derives the
subgoal from live semantic geometry and the current phase-start state, then
converts it into the parameter types, units, and reference frames required by
the primitive contract. Account for relevant geometry and learned tolerances;
do not copy absolute demonstration coordinates.

Use this sequence at every phase boundary:

1. Identify the semantic relation defining the requested outcome and the bound
   entities that define its origin, axes, and admissible region.
2. Derive the physical goal from those entities and the actual retained state.
3. Compute the required effect relative to the current target state.
4. Encode the effect in the primitive contract's units and reference frame.
5. Check that the retained outcome satisfies the next phase's preconditions,
   including the allowed position, orientation, and contact-state residuals.

For a translation request this is `delta_world = goal_world - current_world`,
followed by `delta_input = R_world_from_input.T @ delta_world`. Derive
`goal_world` before choosing its representation. Input axes do not define
physical goal axes. Use the contract's corresponding representation for other
effect types. Record the semantic source and frame of each measured feature.

A different source of the requested subgoal does not by itself justify a
separate primitive.

Apply the same rule at every subsequent phase boundary. Resolve the relevant
entity state from the actual retained result of preceding execution, derive
the next subgoal from the bound semantic relations, and compute the next
primitive's declared parameters in their required frames. Do not reuse values
derived from an assumed intermediate state.

If no available primitive semantically matches a target-changing phase:

- first confirm that no existing primitive plus a relational connecting
  constraint expresses the required effect;
- then implement the smallest reusable primitive addition or repair, validate
  it on its canonical examples, record the evidence and rationale, and
  continue the strategy run. Never emulate the missing effect in
  strategy-local code.

### Primitive selection and contract mismatch

Infer each relational phase signature before testing concrete bindings. Match
primitives by the complete signature, not by whichever contract is easiest to
satisfy. Do not confuse a fixture that causally changes the primitive's contact
mode or effect with a fixture that only supplies a relational phase goal. The
latter should normally be expressed by a strategy composition constraint and
projected into the available primitive's declared input frame. Different
orientation/contact policies remain non-interchangeable merely because their
final object translations look similar.

When the demonstration or supplied language explicitly names an available
primitive and its documented semantics match the inferred phase, treat that as
strong evidence for the selection. If its concrete contract then fails:

1. separate a true missing task affordance from a requirement on incidental or
   safety-only context;
2. if the effect matches but the contract requires noncausal scene occupants,
   report an **over-specific primitive contract**;
3. do not replace it with an primitive having a different relational phase
   signature; and
4. make and validate the narrowest reusable upstream contract or
   implementation repair, then continue.

A successful smoke run cannot override this semantic gate.

### Upstream primitive repairs

Preserve primitive semantics when repairing its implementation: keep the declared
physical goal, required contact mechanism, and terminal behavior. You may revise
geometry resolution, approach selection, nominal motion, optimization, and
supporting code, including adding a reusable way to perform the same operation.
Prefer targeted repairs supported by retained failure evidence and retain
existing implementation that remains valid; do not replace the primitive
wholesale merely to fit the current strategy. Do not embed strategy-specific
sequences or scene-specific assumptions, or weaken success requirements to hide
a failure. Corrections to predicate implementation must preserve its physical
meaning.

Keep repairs local to the strategy workspace. Check the modified primitive's
interface and contract, run its package tests, and rerun the affected strategy
examples. Briefly explain the defect and why the repair still implements the
same primitive. Report the scope of the testing actually performed.

## Step 2 - Write the strategy description

Create `src/strategies/<name>/strategy_description.md` before tuning. Include:

- a one-paragraph summary and final goal;
- `## Inferred goal and final success predicate`, listing every `Psi` conjunct;
- `## Demonstrated scene and contact sequence`;
- `## Scene fidelity sketch and invariants`;
- `## Plan skeleton` as an ordered phase/transit code block;
- `## Phase signature audit` comparing each demonstrated phase with the chosen
  primitive's effect, reference frame, orientation policy, contact mode, and
  terminal relation;
- `## Relational representation` naming roles, frames, faces, axes, supports,
  fixtures, receptacles, goal regions, and transforms;
- one `## Phase K - <primitive>` section containing its subgoal, semantic role
  map, connecting constraints, preconditions,
  expected effect, and handoff;
- one section for each inter-phase transit and protected-contact policy;
- `## Smoothness and safety objective` covering phase and boundary metrics;
- `## Strategy binding roles`;
- `## Connecting constraints` as
  `Phase | Primitive input | Role dependencies | Frame/units | Relation`;
- `## Known assumptions and unsupported scenes`.

The description must make it possible to audit whether `PHASE_PROGRAM`, final
`Psi`, and the implementation express the same strategy.

## Step 3 - Implement the learned package

Create exactly:

```text
src/strategies/<name>/
  __init__.py
  binding_contract.py
  program.py
  predicate.py
  main.py
  strategy_description.md
```

Learned strategy files may not import `tasks`, `test_scenes`,
`assigned_tasks`, or `instantiations`, and may not contain assigned-scene
names, XML, hard-coded qpos indices, absolute poses, or module-global scene
selection.

### Strategy binding contract

`binding_contract.py` publishes `BINDING_CONTRACT: BindingContract`.

- Declare every strategy semantic role needed by any phase, transit, or final
  predicate.
- Preserve optionality when mapping optional primitive context: use a strategy
  role with `min_count=0` and ensure the strategy remains valid when that role
  is unbound. Do not promote an primitive's optional or incidental context to a
  required strategy role.
- Declare entity kind, type, multiplicity, body relationships, and relational
  invariants precisely enough to reject incompatible scenes before simulation.
- Make body-role descriptions agree with ownership constraints. For a site or
  geom with `same_body_as`, the related body role denotes its exact owning
  body, not an ancestor or the root of an articulated assembly. Validate
  related assignments together on every invention example.
- Set `parameters=()`. The strategy binding supplies roles; connecting
  constraints supply primitive parameter values.
- Do not expose phase compiler, geometry, success-predicate, camera,
  initial-state, or any other callback.

### Frozen phase program

`program.py` publishes `PHASE_PROGRAM: StrategyProgram` composed from
`PrimitivePhase`, `ConnectingConstraint`, and `InterPhaseTransit`.

For each `PrimitivePhase`:

- choose one learned primitive by its full relational phase signature, not by
  scene name or structural bindability alone;
- give it a stable phase name;
- map every primitive role exactly once from a strategy role;
- cover every declared primitive parameter with exactly one
  `ConnectingConstraint`;
- do not pass knots, geometry, sigma, costs, predicates, or phase callbacks.

Each `ConnectingConstraint` declares its primitive input, semantic role
dependencies, output frame, relation description, and pure learned resolver.
For every resolver:

1. reconstruct `MjData` from the propagated phase-start state and call
   `mj_forward` once outside the optimizer population loop;
2. resolve positions, orientations, faces, axes, and extents through semantic
   strategy roles, never assigned-scene names or qpos constants;
3. compute the primitive input in the frame and units declared by that
   primitive's contract; and
4. let `execute_sequential_strategy` construct and validate the ephemeral
   primitive binding before local compilation.

Resolvers follow the workspace parameter-domain rule and may not introduce
additional hard cutoffs for optimizer convenience. A preferred execution
scale is not a capability limit. Use bounded decomposition only under the
phase-necessity criteria below, preserving the full requested effect.

Connecting constraints read the current propagated state, including the
physical result of previous phases. Do not freeze their resolved numeric values
in `PHASE_PROGRAM`, expose them in `binding.json`, or recompute them per
optimizer candidate.

For each phase boundary, state which geometric preconditions the next primitive
requires and which preceding effect or transit establishes them. Check those
relations on the actual retained state before the next optimization. During
invention, repair the preceding subgoal, composition, or a demonstrated
geometry-resolution defect when a requirement is unmet. Do not assume that
success of one phase establishes the next phase's full preconditions, or
weaken a valid requirement to make the composition pass.

### Phase necessity and bounded decomposition

Start with the smallest phase sequence that expresses the required mechanism
and intermediate relations. Justify each boundary by a necessary change in
contact mechanism, a required intermediate relation, or a demonstrated limit
of one execution. An arbitrary waypoint or smaller requested displacement
alone does not justify another primitive invocation.

Before splitting an invocation or adding a correction phase, diagnose frame,
geometry, parameter, objective, control, and search-support errors using the
existing evidence protocol. Residual error, a failed optimizer run, or a
reference search scale alone is not proof that the primitive cannot express
the effect. Repair an identified upstream defect or use the permitted search
ladder before treating it as a composition limitation.

For a necessary decomposition, record its capability or semantic evidence,
each segment's contribution to the full goal, and why adjacent segments
cannot be merged while preserving the mechanism and feasibility. Repeated
invocations toward the same relation require measured progress, a justified
finite count, and evidence that the retained sequence establishes that
relation across all required examples. Do not append more copies merely
because the last one fell short. Report unresolved or stalled behavior;
do not hide failures with implicit retries or change runtime continuation
semantics to enforce this design review.

Compare a proposed expansion with the simpler design using retained evidence
and targeted checks within the existing run budget. Report total primitive
optimizations, transit count, wall time, and release/handoff failures alongside
goal success; account for total search effort, not only per-phase settings.
Retain extra phases only when their necessity and benefit are supported.
Remove redundant phases and rerun affected examples. Do not impose an
arbitrary universal phase cap or merge genuinely distinct contact mechanisms.

### Inter-phase transit

By default, let each primitive's `execute(context, cfg)` own its approach and
effect. In `run_phase`, pass `context.bind_primitive(primitive_binding)` to the
named primitive's entrypoint. This preserves the live physical/controller state,
last command, and gripper operation, including for the first phase. An existing
primitive with an equivalent complete entrypoint can be called directly.

For each transition between adjacent phases, decide explicitly whether a
separate release, withdrawal, or scene-level routing transit is required before
the next primitive's approach. `InterPhaseTransit` contains a learned
`TransitSpec`, not realized waypoints. Compose intent with `Withdraw`,
`RaiseToClearance`, `Reorient`, `ConfigureGripper`, `Approach`, and `Settle`. Use
dimension-relative `ScaleExpr` values with frozen coefficients and durations.
Every transit ends with exactly one convergence-based `Settle`; shared runtime
may lengthen moving and settle horizons for live geometry, dynamics, and
controller filtering.

When a custom inter-phase approach is needed, reuse the next primitive's
precontact resolver for its contact point, task-required contact axes, and
equivalent attitudes, then call its effect execution without repeating the
approach. The next primitive owns its task-required contact axes, not an
arbitrary full quaternion independently chosen by the strategy. Resolve them
from the live phase-start state and provide retained obstacle/corridor roles
for feasibility. Preserve every unconstrained
rotational degree of freedom by selecting the safe task-equivalent attitude
requiring the least rotation from the propagated hand attitude. When material
reorientation is required, place one smooth `Reorient` after free-space
clearance and before `Approach`; declare discrete task-equivalent choices with
`Reorient.equivalent_quaternion_offsets` so the runtime can select a reachable
handoff without changing a required axis. Preserve the selected phase attitude
through contact unless the primitive mechanism intrinsically rotates the tool.
Under task-layout transforms, co-rotate every task-required axis and reapply
the shared hand-local-X/support-normal posture rule before the
minimum-rotation tie-break.

Keep release and approach diagnostics with their executed segments. The
executor's `attempt` recovery applies to `run_transit` failures with retained
trajectories; moving that work inside `run_phase` changes recovery behavior.
Choose the boundary deliberately when composing an existing primitive.

When the next phase needs a different aperture, place exactly one
`ConfigureGripper` before its first `Approach`. Configuration occurs in free
space, resolves against the physical range, and settles in both position and
velocity before approach. Keep aperture fixed afterward and never change it
while target or protected-object contact is active. If a transit begins with
authorized inherited contact, shared runtime separates it with fixed-aperture
arm motion; it neither opens the gripper nor searches over aperture. A requested
`ConfigureGripper` runs only after certified separation. Preserve each
primitive's declared `GripperIntent`: nonprehensile phases are `HOLD`; only
intrinsic grasp/release actuation is `ACTUATE` with its frozen nominal profile.

Declare protected roles and drift limits. Target contact is invalid unless the
frozen spec declares an `InheritedReleaseWindow`. That declaration authorizes
the runtime-owned, bounded release phase; the runtime reports its effective
controller-aware timeout. Release reanchors the Cartesian equilibrium to the
measured pose without changing aperture. It classifies actual or numerically
contact-equivalent pairs as inherited contact and other sub-clearance pairs as
nearby contact-free geometry. Inherited contact may only separate without
worsening; transient recontact remains diagnostic, and terminal stable
separation is required. Nearby geometry may not cross into contact. The runtime
searches a continuous local SE(3) region using native IK and collision checks,
requires contact-pair clearance progress and contact-free floors along each
kinematic path, and gives ordinary obstacles a controller-tracking reserve. It
executes at most four diverse
native-OSC candidates and uses the first measured violating pair, step, and
normal to prioritize the remaining candidates. Acceptance additionally
requires no new contact, bounded protected-role drift, stationarity, and full
free-space clearance. Declare the semantic release window and protected roles;
do not add strategy-specific retreat code or tune runtime release limits.

After release, shared runtime configures any requested next aperture in free
space and plans spatial motion in FR3 joint state with Cartesian controls.
Native kinematics/collision checks screen the path without Python
`mj_forward`. Kinematic planning itself never simulates dynamics; native OSC is
used for the release shortlist and retained phase executions. Release paths
carry a 3 mm tracking reserve above the 10 mm executed-state clearance, with a
safe entry floor for pairs already inside that reserve. Ordinary post-release
robot/environment pairs keep the runtime-owned 10 mm path clearance. For the
next primitive's initial pose, resolve its task-required axes and any declared
equivalent attitudes before projection. The runtime tests the desired attitude
and declared equivalents for IK first, then keeps the selected attitude and
aperture fixed during deterministic low-discrepancy translation search over
the full 0.100 m ball at 2 mm radial resolution. The selected handoff minimizes
translation, using attitude change as a tie-break, while remaining contact-free;
it may use the smaller 2 mm tool/target precontact margin. The translation
search is not restricted to fixed rays or a learned retreat direction.

Planning, IK/projection, inherited-contact release, nonfinite execution, and an
invalid physical/controller handoff are hard failures. Finite final-OSC
convergence, semantic-contact, whole-arm collision, and clearance findings are
structured advisories: retain and visually inspect the trajectory, and do not
stop later primitive phases solely because of those findings. If a run reports
`next_primitive_initial_knot_requires_safe_handoff_projection`, inspect its
structured `initial_knot_advisory` and adjust the next phase's first nominal
knot or the typed `Approach` standoff into nearby collision-free precontact.
Contact acquisition belongs to later knots owned by that primitive; do not
weaken or rewrite its later contact-rich trajectory. A settle timeout,
collision, or clearance finding reported only by a finite final OSC rollout is
an advisory; the corresponding kinematic planning or release failure remains
hard. When a hard failure occurs after physical motion exists, shared runtime
attaches the complete beginning-to-failure strategy trajectory for recording
and visual diagnosis. Never tune shared motion/planning/audit limits in
strategy or binding code.

### Final predicate

`predicate.py` defines learned `evaluate_final_psi(...)` and frozen final
tolerances. Evaluate the final strategy goal, not merely the final phase's
primitive predicate. Report every `Psi` conjunct separately.

Calibrate `Psi` when replayable physical evidence shows a false positive, a
false negative, sensitivity to timestep, horizon, or scale that is unrelated
to `G`, or a mismatch with a declared parameter meaning. Search difficulty or
budget exhaustion does not justify lowering a threshold or deleting a
conjunct.

- Evaluate one retained combined-trajectory counterexample with both the
  current and proposed formulations, and record every conjunct result.
- Keep `G`, phase subgoals, goal text, parameter names and meanings, and `Psi`
  consistent. If a parameter or goal meaning changes, revise them together
  before further search.
- Measure each declared final effect directly, or establish the geometric
  equivalence of its proxy. Evaluate any required direction, axis, or
  reference-frame relation on the effect itself. An unrelated change in
  another phase cannot establish that the effect satisfies that relation.
- When time-to-effect is not semantic, measure retention over a fixed physical
  dwell or settle interval after goal attainment, rather than over a fraction
  of the optimized horizon. Measure support, contact, or stationarity directly
  when it is part of `G`.
- Treat an accepted calibration as a source-fingerprint change. Rerun every
  strategy example and repeat the bounded visual checkpoint. If an primitive
  predicate changes, follow the primitive calibration protocol and rerun that
  primitive's canonical examples first.

Keep any trajectory-wide numerical work tensorized when it is evaluated over
optimizer populations. Never loop over candidates/timesteps and call
`mj_forward`, `mj_step`, or replay saved states inside CEM/MPPI. Request
explicit native telemetry when needed. Exact reconstruction is allowed only
for auditing the one retained combined trajectory.

### Strategy compiler and public entrypoint

`main.py` defines `make_parser()` and
`compile_strategy(binding, args) -> InstantiationSpec`.

Compilation must:

1. Load the scene through the binding and validate strategy invariants.
2. Validate `PHASE_PROGRAM` against the strategy and primitive contracts.
3. Invoke `execute_sequential_strategy` with the frozen program, contracts,
   initial state, primitive phase runner, and typed transit runner. The shared
   executor resolves connecting constraints after each live state handoff and
   invokes each primitive as one local optimization problem.
4. Evaluate final `Psi`, attach per-phase/final diagnostics, and return an
   `InstantiationSpec` suitable for canonical execution and recording.

An primitive phase whose retained `optimized_success` is false still contributes
its real terminal physical/controller state to the strategy. Continue through
the remaining transit and primitive phases; do not turn an unsatisfied primitive
effect into a strategy exception or use it to suppress later execution. Missing
effect metrics and genuine execution or contract errors remain hard failures.

Set the combined result's `success` exactly from final `Psi`. Report
`primitive_phase_effect_success` and `all_primitive_phase_effects_success`
independently, and do not conjoin phase effects with final `Psi`.

Do not define a strategy-instance protocol or dataclass containing callbacks.
Do not select an instance using environment variables or patched globals. Do
not write a strategy-local effect compiler that bypasses an primitive's frozen
nominal policy.

The typed transit runner must call `execute_transit` with
`transit_seed=args.transit_seed` and
`transit_index=context.phase_index - 1`. The root transit seed is runtime-owned;
`TransitSpec.label`, strategy names, phase names, primitive names, scene names,
and object names must not influence planner sampling.

Use each primitive's frozen `OptimizationPolicy` unchanged, including horizon,
`elite_fraction`, knot topology, early-stop rule, and sigma support. Start each
phase at that primitive's `REFERENCE_OPTIMIZATION_BUDGET`. When evidence warrants
more search, set `PrimitivePhase.optimization_budget` to an
`OptimizationBudget` from Step 6's deterministic ladder. A phase request may
increase either allowance only when neither allowance is below the primitive
reference, and it may never exceed `GLOBAL_MAX_OPTIMIZATION_BUDGET`. Record
retained phase requests and prefer consistent requests across examples. Leave
`max_parallel_rollouts` to shared machine-aware runtime; CPU worker counts are
not learned strategy constants.

Resolve every primitive compiler config with shared `build_config`, passing the
primitive's `OPTIMIZATION_POLICY` as `learned_policy`, its
`REFERENCE_OPTIMIZATION_BUDGET` as `reference_budget`, and the current
`PrimitivePhase.optimization_budget` as `requested_budget`. Do not pass raw
`num_rollouts`, `num_elites`, or `max_iters` programmatic overrides. Shared
runtime derives integer `num_elites` from the frozen `elite_fraction` and the
effective rollout count.

## Step 4 - Preserve physical and phase fidelity

Follow the physical execution invariants in `AGENTS.md`. Propagate actual
physical/controller state and the terminal command through every boundary, and
start each phase continuously from that measured handoff. Require every FR3
phase/transit to report `controller_rollout_backend ==
"native_cpp_fr3_osc"`; enforce hard planning/release/nonfinite handoff gates,
and propagate finite final-OSC audit findings as advisories to later phases.

## Step 5 - Create the baseline and variants

Preserve the assigned scene as `example_0`. Generate at least two deterministic
geometry/pose variants using the run seed and
`balanced_generalization_samples`. Use the same learned strategy and primitive
packages in every example.

Use each sample's `size_scale_xyz` directly in `SceneVariantSpec`: the three
factors independently scale the first bound collision geom's local XYZ axes.
The shared builder transforms visual geometry, collision geometry, mass and
inertia consistently. Derive fixture clearances and placement from the resized
geometry; do not average the factors into a uniform scale.

Create and statically validate `example_0` and every requested variant before
behavioral tuning. Develop on all examples in the same Codex phase. A failure
on any example may inform a shared strategy or allowed primitive repair, after
which rerun affected examples and disclose the evidence and change.

Across the set cover semantic renaming, translation, full-circle yaw,
independent three-axis resizing, mass/inertia, friction, relational subgoal changes, and compatible
fixture/receptacle placement only when a retained causal role exists. Each
nonbaseline example must change at least three eligible physical/layout
dimensions. The changed layouts must cause connecting constraints to produce
different feasible primitive invocations.

Preserve the complete assigned scene only in `example_0`. In every nonbaseline
example, remove bodies that the scene sketch and phase program classified as
incidental. Retain only strategy roles, their necessary physical parent/child
closure, the embodied tool and supports, and entities explicitly used by the
final goal, a phase mechanism, a connecting constraint, or final `Psi`.
Optional context may remain only in a deliberate compatible variant that tests
it; document why it remains. Never perturb an incidental body merely to count
it as a fixture, receptacle, or generalization dimension. Delete the complete
incidental body subtree only after checking joints, tendons, actuators,
sensors, equality constraints, contacts, and other named references. If a
reference is causal, retain its minimal physical closure; otherwise remove the
reference too, then compile the resulting `MjSpec` before accepting the
variant.

Apply global pose changes jointly to bound entities whose relative geometry
defines a phase mechanism, a required motion region, or final `Psi`. Preserve
those defining relations through `SceneVariantSpec.yaw_coupled_bodies`.
Use `allow_independent_fixture_yaw=True` only when changing relative pose is
the intentional test and every required affordance remains valid.

Freeze and run the fixed development qualification suite in `AGENTS.md`,
including its phase-handoff probes. Keep its cases separate from repairable
development variants and report unsupported checks as findings.

Each example needs:

```text
test_scenes/<name>/example_<k>/
  __init__.py
  scene.py
  variant.json

src/instantiations/strategies/<name>/example_<k>/
  __init__.py       # canonical
  binding.json      # data only
  main.py           # canonical
```

Create immutable instantiation scaffolds with
`agent.instantiation_context.stage_canonical_instantiation`; edit only each
manifest. Never create `binding.py`, `mapping.py`, callbacks, phase factories,
custom entrypoints, predicates, waypoints, or transit code in an
instantiation. `variant.json` must exactly equal `VARIANT_SPEC.to_dict()` and
each manifest must reference its matching scene. Strategy bindings use
`inputs: {}` and `VARIANT_SPEC.goal_parameters` is empty; primitive parameters
are recorded from the shared executor's resolved phase inputs.

Before behavioral search, read each exact runtime scene and binding, render and
inspect its initialized layout, and combine that evidence with
`validate_variant_initial_state` and contract/program/binding checks. Inspect
phase-relevant relations and robot approach/contact space rather than treating
helper success as proof of feasibility. Follow the three-outcome
feasibility procedure in `AGENTS.md`. Resolve connecting constraints from the
actual retained state before each phase. Record the checked roles, frames,
contact mode, moving bodies, measured inequalities, candidate budget and
decision under `agent_work/`. A failed nominal path is a failed candidate;
allow bounded search without weakening physical constraints. Any movable
incidental body remains an obstacle unless its motion is an explicit measured
phase effect. Co-transform goal/reference frames with retained layout groups.

## Step 6 - Run, inspect, and improve upstream

Keep `results_instantiation/` empty throughout invention. That directory is
reserved exclusively for later deploy-instantiation runs. Every invention-time
execution must pass explicit artifact paths: use `agent_work/debug/` for
disposable metrics and recordings, and `results/` only for retained evidence.
Never rely on the instantiation entry point's default output paths.

Run every example through its canonical module entrypoint:

```bash
$RAPID_REPO_ROOT/scripts/rapid-python \
  -m instantiations.strategies.<name>.example_<k>.main
```

Run package/program/manifest/variant checks before simulator work. Record their
findings, repair genuine defects, and then use bounded behavioral experiments.
Start from consistent optimizer settings and adapt them only with task-relevant
evidence, recording retained differences.

Both feasible candidates and unresolved invocations may enter the bounded
search ladder. Reserve `scene_or_goal_infeasible` for an evidenced violation of
a necessary physical condition. Incorrect frame transport is a code defect.
Candidate collisions or finite IK failure alone do not justify that label.

Treat each nominal phase trajectory as an initialization, not as a candidate
that must succeed. A nominal or 1x1 run may verify package wiring, finite
shapes, action semantics, typed transit/controller handoff, telemetry, and
deterministic hard safety checks only. Nominal success, contact, and task
progress are not prerequisites for population search. Do not infer search
support or edit learned motion coefficients or topology from a 1x1 behavioral
outcome alone; one sample cannot show whether the declared sigma reaches the
contact manifold or whether the objective ranks useful perturbations.

After wiring passes, immediately run the minimum search on every feasible
optimized phase. A second nominal-only behavioral revision before that search
is not allowed. Repeat a 1x1 first only when deterministic evidence prevented a
meaningful population evaluation: a wrong frame/sign/action order, an
unreachable typed transit or required contact outside the declared sigma
support, nonfinite execution, missing telemetry/objective coverage, or a hard
safety violation. After repairing such a blocker, use at most one 1x1 to
confirm that the blocker is gone, then run the minimum search even when nominal
contact or progress remains zero.

Use visual feedback as a bounded debugging checkpoint, not only as a final
audit and not as a default for every optimization attempt. After the first
minimum-or-higher search reaches the relevant phase for a design fingerprint,
record and inspect the single most informative unresolved example before
further source tuning, even when a phase predicate or final `Psi` is false.
Allow one diagnostic render per design fingerprint by default and use
`--no-record` for repeated runs with the same physical failure signature,
citing the existing checkpoint. Prefer the shortest replay that preserves the
failure context: use a phase-local primitive replay for an primitive-internal
contact defect when its phase-start state can be reproduced, and a full-strategy
replay for phase choice/order, a boundary, state or controller handoff,
inter-phase transit, or final-goal defects. Render
one additional representative only for a distinct failure signature, a
metrics/recording disagreement, or a critical phase absent from the existing
video. A behavior-changing revision starts a new checkpoint only after
minimum-or-higher search shows a material change in phase order, contact or
attitude, path or timing, transit/handoff, release/settling, collision, or
protected-object drift. A budget, seed, documentation, or threshold-only
change does not by itself require a new video. Put disposable artifacts under
`agent_work/debug/`. Inspect reset, each approach, first/peak/last contact,
phase order, every boundary, release/reorientation, and final state, and record
the design fingerprint, metrics path, video path, reason for the checkpoint,
and observed failure signature in the invention report. Classify
population failures before editing:

- incorrect phase selection/order, role map, or connecting constraint;
- scene-fidelity or task-layout relation violation;
- primitive feature/nominal/predicate failure;
- inter-phase transit or protected-contact failure;
- physical/controller discontinuity at a boundary;
- final predicate mismatch;
- insufficient search support or budget;
- penetration, unsafe torque/joint margin, jitter, jump, reversal, or
  backtracking.

Use this deterministic search ladder per optimized phase, capped by the run
prompt:

1. **Nominal diagnostic:** use at most one 1x1 smoke to verify wiring only; do
   not require behavioral success and do not tune from its task outcome.
2. **Minimum meaningful search:** immediately run 32 rollouts x 6 requested
   iterations on every feasible optimized phase after wiring passes. Allow
   predicate-gated early stopping only after at least 4 iterations. Early
   stopping additionally requires the selected trajectory to satisfy that
   phase's primitive effect predicate and its cost/progress to plateau; a
   plateau while the predicate is false must run to the requested iteration
   limit. This population, not the nominal sample, provides the first evidence
   about contact acquisition and sigma support.
3. **Standard search:** run 64 rollouts x 10 iterations only when the minimum
   stage produced finite, safe candidates with a physically plausible
   mechanism.
4. **Expanded search:** at most 96 rollouts x 16 iterations, at most once per
   example for an unchanged design, only when standard search improved the
   same task-relevant metrics and the mechanism remains correct.

Do not rerun an unchanged source at the same failed tier. Escalate a plausible
mechanism; use population and trajectory evidence to form and test concrete
repair hypotheses about geometry, contact, trajectory representation,
objective ranking, controller tracking, or retention. A hypothesis need not
already be proved to justify an informative experiment. Nominal failure alone
is not such evidence. Compare revisions against
retained outcomes and rerun affected examples after shared changes.

The ladder bounds optimization under one design, not the entire invention.
An expanded-search failure or a worse revision calls for diagnosis of the
remaining plausible program changes; neither is by itself a stopping or
resampling condition. When the main mechanism works but measurable goals
remain unmet, prioritize program improvement. Repeated failures across
layouts are shared capability gaps, not cases to discard. Use the scene review
and bounded-resampling guidance in `AGENTS.md` for documented case-specific
unresolved difficulties. Continue informative experiments within the overall
run budget; report partial completion if it is exhausted or a reasoned review
leaves no promising next experiment within it. Record budget tier, design
fingerprint, execution fingerprint, failure signature, cost components,
progress, and mechanism diagnostics. The design fingerprint covers the
strategy, its actual primitive dependencies and binding contracts, typed
transition/resolver code, and shared runtime revision. Scene binding, task
parameters, optimizer settings, and seeds belong to the separate execution
fingerprint; changing only those values is not a design change.

Repair learned strategy/primitive code. Avoid tuning a validated manifest in
response to ordinary search variance or a failed candidate. Preserve assigned
and fixed validation scenes. Resample development variants only under the
evidence-based review and bounded-resampling rule in `AGENTS.md`, preserving
and reporting the original unresolved case. After behavior-changing edits, rerun
every example and fixed probe and repeat the bounded visual checkpoint when
those runs show a material physical change.
When metrics and video disagree, preserve both, inspect the predicate/audit,
and report the disagreement. Apply the predicate calibration protocol when the
retained evidence establishes a false classification.

For every retained example, write exactly:

```bash
$RAPID_REPO_ROOT/scripts/rapid-python \
  -m instantiations.strategies.<name>.example_<k>.main \
  --seed <retained-seed> \
  --metrics-json results/<name>_example_<k>_metrics.json \
  --record results/<name>_example_<k>.mp4
```

Retain representative verification frames under `agent_work/verify/`. Compare
them with the demonstration for scene geometry, contact ordering, phase
effects, transition continuity, final relation, penetration, jitter, and
avoidable motion. Confirm object-centric attitude co-rotation, free-space-only
reorientation, the hand-local-X posture rule, and stable contact-phase attitude.

## Step 7 - Validation evidence

Before finishing, evaluate all of the following. Report workflow completion,
behavioral final-`Psi` success, and audit status independently. Preserve useful
trajectories and list any unmet or disputed item in the final report:

- strict role-only manifests fail invalid bindings before simulation;
- `compile_strategy` accepts `(binding, args)`, not an executable instance;
- strategy contracts and manifests contain semantic roles and no parameters;
- `PHASE_PROGRAM` validates against every primitive contract;
- `example_0` preserves the assigned scene, while every nonbaseline runtime
  scene contains only its documented causal role closure and compiles after
  incidental bodies and stale references are removed;
- every example and retained phase-start state has a recorded feasibility
  classification and evidence, distinguishing unresolved search;
- the fixed qualification report covers scene interface, representation,
  layout and handoff checks against the final learned source;
- phase order, maps, connecting constraints, transits, and final `Psi` live only in
  learned strategy code;
- every added or repeated phase has recorded necessity and progress evidence,
  with total search/transit cost and the simpler-design comparison disclosed;
- every primitive parameter is declared by a `ConnectingConstraint`, resolved
  once from live semantic geometry/current phase-start state, validated against
  the primitive contract, and absent from concrete manifests;
- `execute_sequential_strategy` performs one local optimization per primitive
  phase and propagates retained physical/controller state before resolving the
  next phase;
- an unsatisfied primitive effect is reported but does not stop later transits or
  phases, and final `Psi` alone supplies the combined result's `success`;
- every primitive phase invokes its upstream geometry and nominal policy and no
  action arrays leak through phase invocation types;
- learned sources import no assigned tasks, generated scenes, or
  instantiations;
- no callback adapter, custom predicate, waypoint, policy coefficient, or
  transit executor exists in an instantiation;
- all phase nominal plans and all typed transits are valid on every example;
- every FR3 phase declares `GripperIntent`, transit aperture configuration
  settles before approach, the optimizer leaves channel 6 exact, and no
  nonprehensile target contact contains commanded aperture motion;
- optimized final `Psi` succeeds on the demo and all variants, with any phase
  budget adaptation recorded;
- every retained persistent phase result passes the shared bounded
  terminal-stability probe, reusing an equivalent retained transition when one
  is already available;
- every chosen primitive matches the documented phase signature; no contract
  failure was bypassed through a semantically weaker substitution;
- FR3 phases/transits use the native backend with controller-state handoff;
- per-phase convergence, retained whole-arm audits, and whole-plan
  contact/collision/penetration/gripper/smoothness checks are reported
  separately from final `Psi`;
- retained metrics and nonempty videos exist for every example and visual
  review confirms phase ordering and the intended final effect.

Each retained metrics file must include nominal success/cost, optimized cost,
optimizer improvement, nominal-to-optimized control RMS/max, per-phase effect
results, transit/contact audits, controller backend/handoff, final `Psi`,
generalization metadata, and phase-boundary/whole-plan smoothness diagnostics.
Include each phase's requested/effective/minimum horizon, actual/requested
iterations, early-stop status, and structured whole-arm audit diagnostics.
Nominal-only success is diagnostic and is not required.

## Final report and quality bar

Report strategy name, `G`, full final `Psi`, scene sketch/invariants, frozen
phase program, semantic contract, role maps, connecting constraints, typed
transits, state/controller handoffs, retained phase budgets, design/execution
fingerprints, and the required provenance for every locally modified primitive.
Include package, test-scene, and instantiation trees. For every example, report
each phase and final predicate, nominal/optimized diagnostics,
transition/contact/collision and smoothness results, exact commands, and
metrics/video paths. End with known limitations and any missing primitive
capability.

All generated Python must run. Do not leave TODOs, placeholder stubs, fake
metrics, fake success, scripted target motion, hidden cleanup, or scene-specific
logic in learned code. Use type hints, short module docstrings, dataclasses
where useful, and `from __future__ import annotations`.
