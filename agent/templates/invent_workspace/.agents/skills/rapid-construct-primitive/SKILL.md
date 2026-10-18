---
name: rapid-construct-primitive
description: Invent and freeze a scene-agnostic RAPID manipulation primitive whose learned package owns relational geometry, nominal effect knots, transit intent, cost, tolerances, and predicate.
---

# Invent and freeze a relational primitive

Produce a runnable learned primitive, not a thin strategy around a smart scene
adapter. The learned package must contain every reusable motion heuristic. A
concrete instantiation binds semantic names and the primitive's declared
effect-goal parameters.

An invention is complete only when the same learned code succeeds on the
assigned demo scene and every requested geometry/pose variant (at least two), using
strict data-only manifests, retained metrics, replayable trajectories, and
visually inspected recordings.

## Read first

Start with the run prompt, demo, assigned scene inventory/module, workspace
`AGENTS.md`, `src/primitive_core/interaction_descriptors.md`, and the public binding/policy
types named by this workflow. Use `rg` to locate a required API and read its
definition before using it. Read compilation, transit, telemetry,
generalization, or recording implementation only when the current design
actually needs that API; do not preload whole runtime modules.

Existing primitives are references for public conventions only. Never copy a
legacy callback adapter, action-bearing invocation, or assigned-scene
heuristic.

Task environments copied under `task_inputs/` are allowed inputs, not prior
primitive implementations. Do not inspect hidden baselines or sibling runs.

## Non-negotiable ownership boundary

The learned primitive owns and freezes:

- semantic feature definitions and a scene-agnostic geometry resolver;
- nominal effect-knot topology and geometry-conditioned knot values;
- all dimension-relative clearances, offsets, standoffs, and timing fractions;
- per-knot sigma rules and whether the first optimized action is fixed;
- effect cost, effect predicate `Psi`, tolerances, and smoothness/safety terms;
- a default typed precontact `TransitSpec` for standalone execution.

The binding manifest may contain only program identity, scene module, goal
metadata, semantic entity names, and typed effect-goal parameters declared by
`BINDING_CONTRACT`. Declare every requested-effect variable that can vary
without changing the primitive's contact semantics. Its value may be supplied
by a direct invocation, inferred from the demonstration for a construction example,
or resolved by a strategy connecting constraint. Requested effect magnitudes
and directions belong in that interface when they can vary independently of
the mechanism. Geometry intrinsic to the contact mechanism and motion-policy
coefficients remain learned primitive logic. The following can never be
manifest inputs:

- measured dimensions or derived geometry;
- poses, axes, waypoints, knots, sigma, or actions;
- clearance/timing/policy coefficients or controller gains;
- costs, thresholds, tolerances, predicates, or callbacks;
- transit stages, transit execution, or optimizer implementation details.

Do not move a failed heuristic into a generated scene or instantiation. Repair
`features.py`, `nominal.py`, `transit.py`, `cost.py`, or the learned compiler,
then rerun the same already-validated manifests.

## Step 1 - Analyze the demonstration

Inspect the supplied keyframes. Extract more frames from the original source
when contact timing, motion direction, or the final effect is ambiguous.
Treat the user's natural-language description as qualitative task semantics.
Interpret it together with the demonstration and assigned scene. Define a
purely semantic, role-relative goal `G`; an ordered semantic mechanism `M`;
the measurable goal predicate `Psi`; and phase-wise partial progress `P(t)`.
`G` may name declared goal parameters but contains no values, equations,
thresholds, weights, waypoints, or simulator details. `M` names intended
relations, contacts, and transition events without numerical motion. `Psi` and
`P(t)` supply the learned numerical formulation. Commit all four before the
first simulator run. Revise `G` or `M` only when physical evidence contradicts
the inferred semantics.
Before implementation, write down:

1. Tool/end-effector, target, supports, fixtures, protected objects, and goal
   entities.
2. Semantic goal `G`, stated without its measurement or implementation.
3. Mechanism `M` as ordered phases, required versus preferred mode relations,
   and transition events.
4. Every conjunct of `Psi`, which numerically measures achievement of `G`, and
   separate measurable diagnostics for the required relations in `M`.
5. A vector-valued `P(t)` whose phase-specific components measure causal
   progress toward `G` through `M`, including regression and late retention.
6. Object-centric frames, faces, normals, tangent axes, contact edges, support
   planes, fixture relations, and goal regions used by `G` and `Psi`.
7. Required semantic roles, their MuJoCo kinds, multiplicity, parent-body
   relationships, and relational invariants.
8. Intrinsic effect-goal parameters and their types, units, frames, and
   semantic domains. Infer the demonstration invocation's value for each
   parameter. Keep policy coefficients and measured scene values out of the
   parameter interface.
9. Effect-knot phases and their intended topology, without copying absolute
   demo coordinates.
10. Required precontact transit stages, protected objects, inherited-contact
   assumptions, and controller-state handoff.
11. Smoothness and safety signals that must be optimized and reported.

Map the manipulation onto the primitives in `primitive.md` and choose an
`effect_type`. If no primitive name was supplied, choose a short snake_case
name, record the choice, and continue creating the primitive.

Define relational geometry explicitly. Derive mechanism-dependent axes and
contact features from live bound-role geometry. Requested effect directions
must use their declared semantic reference frames; clearances should scale
from live object or tool dimensions. Gravity and a semantic support normal
are valid invariants. Do not introduce arbitrary world axes or copy absolute
coordinates from the demonstration.

### Role necessity audit

Do not turn the assigned scene inventory into the primitive schema. Before
writing `BINDING_CONTRACT`, classify every candidate role as:

- **essential**: removing it makes `G`, the relational frame, nominal effect,
  `Psi`, or an intended contact mode undefined;
- **optional context**: learned motion or auditing can use zero or more such
  entities, but the primitive effect remains meaningful without them; or
- **incidental**: it merely happens to be present in the demonstration and is
  not consumed by learned code.

For every proposed required role, write the counterfactual: “without this
role, the primitive cannot ___.” If that sentence cannot name a causal use,
the role is not required. Protection from accidental contact alone does not
make a particular scene object part of the task schema; shared whole-scene
contact auditing remains active. When named protected context is useful,
declare a plural optional role with `min_count=0` and tolerate an empty
binding. Do not require a free joint, exact count, geom type, or body type just
because the demonstration contains one.

The contract must describe the smallest scene that still has the task
affordance. Richer scenes may bind optional context, but they must not redefine
the primitive.

## Step 2 - Write the primitive description

Create `src/primitives/<name>/primitive_description.md` before tuning. Include:

- a one-paragraph semantic summary;
- `## Semantic goal G`;
- `## Mechanism M`, separating required and preferred phase relations;
- `## Success predicate Psi`, listing every measurable goal conjunct;
- `## Progress representation P(t)`, with component, phase, frame/sign,
  normalization, and retained diagnostic;
- `## Primitive composition` with the effect-knot phase topology;
- `## Mode invariants (construction-time)`;
- `## Effect schema`, including `effect_type` and relational frame;
- `## Relational representation` naming all roles, faces, axes, supports,
  fixtures, and transforms;
- `## Role necessity audit` as
  `Role | Essential/optional/incidental | Causal use | Cardinality/type reason`;
- `## Geometry features and detectors`, mapping each feature to its upstream
  resolver;
- `## Invocation parameters` as `Name | Type/domain | Frame/units | Meaning`;
- `## Nominal policy`, describing knot phases, scaling logic, and sigma intent
  without scene-specific numbers;
- `## Default transit intent`;
- `## Smoothness and safety objective`;
- `## Known assumptions and unsupported geometries`.

Keep the description concise enough to audit, but concrete enough that another
agent could tell whether `features.py`, `nominal.py`, and `Psi` implement the
same primitive.

## Step 3 - Implement the learned package

Create exactly one flat package:

```text
src/primitives/<name>/
  __init__.py
  binding_contract.py
  types.py
  features.py
  nominal.py
  transit.py
  cost.py
  compile.py
  main.py
  primitive_description.md
```

Do not create version subpackages. Learned files may not import `tasks`,
`test_scenes`, `assigned_tasks`, or `instantiations`, and may not contain
assigned-scene entity names, XML, hard-coded qpos indices, or absolute poses.

### Binding contract

`binding_contract.py` publishes `BINDING_CONTRACT: BindingContract`.

- Declare each semantic entity with `RoleRequirement`, including its
  `EntityKind`, type constraints, multiplicity, and body relationship.
- For a site or geom with `same_body_as`, describe and bind the related body
  role as its exact owning body, not an ancestor or assembly root. Validate
  the connected role assignments together.
- Declare only causally essential roles with `min_count >= 1`. Optional
  context uses `min_count=0`, normally `max_count=None`, and every consumer
  must handle zero values without calling `binding.one(...)`.
- Do not encode incidental scene composition as a relational invariant. A
  distractor's mobility, joint type, shape, or existence is not an primitive
  precondition unless the learned effect explicitly depends on it.
- Declare machine-checkable relations with `RelationalInvariant` where the
  shared relation vocabulary is sufficient.
- Declare intrinsic effect-goal variables with typed `ParameterRequirement`
  values. Require finite values and declare vector lengths, frames, units, or
  choices whenever meaningful. Add a hard numeric bound only when it follows
  from the semantic domain, hardware, safety, or numerical validity; a
  reference search scale is not such a bound. Infer each construction example's
  parameter values from its requested or demonstrated effect. Derive axes or
  contact features that are fixed by the primitive's bound-role semantics
  inside learned geometry code.
- Put geometric compatibility statements in contract invariants and implement
  allowed invariant checks without generating motion.
- Reject missing, unknown, mistyped, nonfinite, and semantically invalid inputs
  before simulation.

### Relational features

`features.py` defines `resolve_geometry(context)` from a
`BoundExecutionContext`. Resolve live model/data features using bound semantic
roles, not concrete names embedded in learned code. Useful features include:

- body/geom/site poses and role-relative frames;
- collision extents and support radii along relational axes;
- support planes, vertical faces, contact faces, and fixture edges;
- target/tool approach directions and reachable goal regions;
- start-state relations and primitive parameter values from `context.binding`.

Return semantic typed features or a documented mapping. Validate degeneracy,
nonfinite geometry, missing affordances, and impossible clearances explicitly.
Do not silently fall back to demo coordinates.

Apply the workspace geometry rules to every feature. Keep body, geom, and
site frames distinct, and document the semantic source of each axis and
origin. Before rejecting a request, verify that the measured extent or signed
distance uses the feature's physical frame and relevant collision geometry.
If the mechanism permits multiple orientations or contact features, resolve
the admissible candidates before declaring incompatibility; a failed initial
choice is not proof that all choices fail. Preserve genuine physical limits
and report the measured quantity and bound for an infeasible request.

### Frozen nominal policy

`nominal.py` defines
`make_nominal_plan(context, features) -> NominalPlan`.
It also publishes one frozen `OPTIMIZATION_POLICY: OptimizationPolicy` and one
`REFERENCE_OPTIMIZATION_BUDGET: OptimizationBudget`. `main.py` passes both to
the corresponding `InstantiationSpec` fields. The policy owns knot topology,
horizon, `elite_fraction`, minimum iterations, and predicate-gated early-stop
behavior. The reference budget records the rollout population and iteration
allowance that established the primitive during invention.

- Build every effect knot here from relational features and declared task
  inputs. Try to make the nominal trajectory object-centric.
- Resolve a stable object-centric end-effector attitude from live bound-role
  geometry. The visual human demonstration does not provide an FR3 attitude.
  State the task-required contact/tool axes, task-equivalent rotational
  freedom, and whether the mechanism requires the attitude to change during
  contact. Constrain only the required tool axes, apply the hand-local-X /
  support-normal posture rule in `AGENTS.md`, and choose from the remaining
  safe, reachable, task-feasible orientation family the member requiring the
  least rotation from the current embodied or propagated hand attitude. Never
  introduce a fixed cant angle without task geometry, a shared physical
  threshold, or measured reachability/collision evidence.
- Keep a stable semantic knot topology: acquisition, loading, effect, release,
  or retreat stages as appropriate to the primitive.
- Make each knot and coefficient schedule traceable to a phase in `M` and use
  one shared dimension-relative policy across all examples.
- Express offsets and clearances relative to live dimensions. Freeze the
  dimensionless coefficients in learned code.
- Define per-node/per-channel sigma intentionally. Contact-critical channels
  can be narrow while task-relevant refinement channels remain searchable.
  Rotational sigma refines the selected object-centric attitude; it must not
  introduce independent per-knot orientation wiggle or be zeroed merely to
  preserve an inherited world-frame attitude.
- Declare `gripper_intent=GripperIntent.HOLD` for nonprehensile effects and
  keep every nominal gripper knot identical. Supply one
  `GripperOperation.MOVE` for every knot. Declare
  `gripper_intent=GripperIntent.ACTUATE` only when deterministic open/close
  actuation is intrinsic to grasp/release; freeze the exact aperture profile
  and one explicit `GripperOperation.MOVE` or `GripperOperation.GRASP` per
  knot. `GRASP` may only accompany a non-increasing aperture. The shared
  compiler zero-order-holds the operation schedule and fixes channel 6 to the
  nominal profile, so never infer operations from aperture direction or rely
  on gripper sigma or optimizer exploration.
- For FR3 policies set `fixed_first_action=True` and make the first knot
  continuous with the preceding commanded target; the optimizer may not create
  an instantaneous target jump at the start of a real-arm trajectory. When
  `context.initial_action` is present, inherit its channel-6 command exactly.
- Return finite arrays with matching knot/sigma shapes. Never accept knot or
  sigma arrays from an invocation, manifest, or instantiation.
- Return explicit strictly increasing `knot_fractions` spanning `[0, 1]`.

A strong nominal trajectory is desirable and nominal/optimized success are
both reported. Optimized predicate failure is an important diagnostic, but it
does not erase evidence from a physically correct retained trajectory. Do not
make sigma zero merely to turn optimization into playback; the deterministic
gripper channel is the required exception and remains fixed by shared runtime.

### Default transit

`transit.py` defines `default_transit_spec(...) -> TransitSpec`. Compose intent
from `Withdraw`, `RaiseToClearance`, `Reorient`, `ConfigureGripper`, `Approach`,
and `Settle`.
Use one smooth `Reorient` after free-space clearance and before `Approach`
whenever the resolved object-centric contact attitude differs materially from
the current embodied or propagated hand attitude. When the task constrains
only an axis or contact normal, choose the full quaternion closest to that
physical attitude from the safe, reachable, task-equivalent family; do not
arbitrarily complete the frame with task tangents. Retained obstacle/corridor
geometry may select among those equivalent attitudes without changing the
contact mechanism. Apply the hand-local-X posture rule before the
minimum-rotation tie-break. When discrete task-equivalent attitudes remain,
declare them with `Reorient.equivalent_quaternion_offsets`; the runtime checks
their IK and clearance without changing any task-required axis. Treat the
180-degree wrist-roll alternative only as a symmetry of the already-resolved
task axes.
Use `ScaleExpr` so offsets scale with resolved role dimensions; coefficients
and stage durations remain frozen in learned code.
Every transit ends with exactly one `Settle` that declares a minimum hold, a
requested observation limit, and a consecutive stable duration. Stage and
settle durations are learned requests; shared hardware-parity runtime may
lengthen them for live geometry, dynamics, and controller filtering.

For a nonprehensile primitive, choose any learned compact aperture only through
one `ConfigureGripper(aperture_fraction, duration)` in effect-free transit,
before the first `Approach`. It resolves against the physical range and must
settle before spatial approach; the aperture then remains fixed through the
effect. Grasp/release actuation belongs to the deterministic `ACTUATE` nominal
effect profile, not transit-time optimizer exploration. Never change aperture
while target or protected-object contact is active. If a transit begins with
authorized inherited contact, shared runtime separates it with fixed-aperture
arm motion; it neither opens the gripper nor searches over aperture. A requested
`ConfigureGripper` runs only after the robot has reached certified free space.

The hard invariant is commanded aperture constancy and a fixed optimizer
channel. Physical finger position can deflect under contact compliance even
with a constant command. Treat actual aperture range, velocity, and tracking
error as diagnostics; do not classify micrometer-scale passive deflection as
intentional gripper actuation.

Declare `protected_roles` and a dimension-relative drift limit. Target contact
during transit is invalid unless the learned spec declares an
`InheritedReleaseWindow`. That declaration authorizes the runtime-owned,
bounded release phase; the runtime reports its effective controller-aware
timeout. Release reanchors the Cartesian equilibrium to the measured pose
without changing aperture. It classifies actual or numerically
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
do not add primitive-specific retreat code or tune runtime release limits.

After release, shared runtime configures any requested next aperture in free
space and plans spatial motion in FR3 joint state with Cartesian controls. The
planner uses native kinematics/collision checks rather than Python
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
stop later invention phases solely because of those findings. If metrics report
`next_primitive_initial_knot_requires_safe_handoff_projection`, treat the
structured `initial_knot_advisory` as invention feedback: make the first
nominal knot a nearby collision-free precontact knot (or increase the typed
`Approach` standoff), and begin loading/contact acquisition in a later knot.
Do not weaken or rewrite the later contact-rich trajectory. Settle timeout,
collision, or clearance reported only by a finite final OSC rollout remains an
advisory; the corresponding kinematic planning or release failure remains
hard. When a hard failure occurs after physical motion exists, shared runtime
attaches the complete beginning-to-failure trajectory for recording and visual
diagnosis. Shared motion/planning/audit limits are not learned or tuned.

### Cost and predicate

`cost.py` publishes frozen `EFFECT_TOLERANCES` and defines both
`effect_cost(...)` and `effect_predicate(...)`.

Calibrate `Psi` when replayable physical evidence shows a false positive, a
false negative, sensitivity to timestep, horizon, or scale that is unrelated
to `G`, or a mismatch with a declared parameter meaning. Search difficulty or
budget exhaustion does not justify lowering a threshold or deleting a
conjunct.

- Evaluate one retained counterexample with both the current and proposed
  formulations, and record every conjunct result.
- Keep `G`, the binding contract, goal text, parameter names and meanings, and
  `Psi` consistent. If a parameter or goal meaning changes, revise them
  together before further search.
- Measure each declared effect directly, or establish the geometric
  equivalence of its proxy. Evaluate any required direction, axis, or
  reference-frame relation on the effect itself. An unrelated change before
  or after the effect cannot establish that the effect satisfies that relation.
- When time-to-effect is not semantic, measure retention over a fixed physical
  dwell or settle interval after goal attainment, rather than over a fraction
  of the optimized horizon. Measure support, contact, or stationarity directly
  when it is part of `G`.
- Treat an accepted calibration as a source-fingerprint change. Rerun the
  minimum tier on every example and repeat the bounded visual checkpoint before
  further tuning.

- Keep `Psi` object-centric and evaluate it against the invocation's requested
  parameter values. Measure required relations in `M` separately; invention
  acceptance requires `effect_success=Psi`, `mechanism_valid`, and any required
  terminal stability evidence. Publish the outcome fields and finalize through
  the shared helper specified in `AGENTS.md` after all retained probes.
- Implement `P(t)` as a vector of tensorized, phase-specific continuous
  features. Each component measures causal progress in its declared frame;
  reward intermediate progress, penalize regression, and require the achieved
  effect to persist through a late window.
- Build the scalar objective from smooth goal, mechanism, progress,
  nominal-reference, and regularization terms as applicable. A binary contact
  signal cannot be the sole acquisition signal, and a terminal predicate
  cannot be the sole progress signal.
- Preserve sign when direction, face, or rotation sense is semantic. Normalize
  geometric errors by a live dimension or learned tolerance and audit nominal
  term magnitudes so no term dominates accidentally.
- Before search, record a coverage table mapping every `Psi` conjunct and
  required relation in `M` to its smooth cost signal and retained check.
- Retained metrics include `cost_components`, `progress_diagnostics`, and
  `mechanism_diagnostics` mappings, separating goal achievement, mechanism
  validity, and safety/audit validity.
- Keep objective weights and tolerances frozen in learned code. Add
  control-delta, acceleration, oscillation, reversal, penetration, and
  unnecessary-motion terms only where they express a declared risk.
- Keep population costs tensorized over candidates/timesteps.
- Never loop over population elements and call `mj_forward`, `mj_step`, or
  replay saved state to recover derived features.
- Request opt-in native telemetry through `PrimitiveCompileSpec` if batched
  state/action/sensor/controller tensors are insufficient.
- Exact MuJoCo reconstruction is allowed only for auditing the one retained
  best trajectory after optimization.

### Compilation and public entrypoint

`compile.py` publishes `execute(context: BoundExecutionContext, cfg) -> CompileResult`
for both standalone and strategy calls. Its precontact transit starts from the
supplied state and approaches this primitive's own effect start. It then
resolves the measured post-transit geometry, calls `make_nominal_plan`,
builds `PrimitiveCompileSpec.from_nominal_plan`, and invokes the shared compiler.
Execution ends after this primitive's effect and declared terminal verification.
Preparation for a subsequent primitive belongs to the strategy's inter-phase
transition and that subsequent primitive's approach.
Return the combined transit/effect trajectory, diagnostics, and terminal
controller state. Preserve the supplied state, initial action, gripper operation,
and controller state throughout; scene initialization belongs to the standalone
wrapper. The selected task-equivalent approach attitude and effect attitude
come from the same primitive geometry logic.

Its precontact call to `execute_transit` must
pass `transit_seed=cfg.transit_seed` and `transit_index=0`; the learned
`TransitSpec.label` is diagnostic only and never controls planner sampling.
`main.py` defines `make_parser()` and
`compile_primitive(binding, args) -> InstantiationSpec`; its execution callback
creates the initial `BoundExecutionContext` and calls `execute`.

When composition needs a custom approach, expose the primitive's precontact
resolver and effect execution separately, sharing their implementation with
`execute`. The caller can then reuse the same contact geometry and execute the
approach once before the effect.

`Invocation`, `LocalInit`, and other adapter-facing types must not contain
nominal/sigma arrays, waypoint/action arrays, clearances, tolerances,
predicates, or transit executors. Internal compile-only objects may receive
arrays only after the learned nominal policy has constructed them.

Do not write a custom CEM/MPPI implementation. The agent chooses horizon,
`elite_fraction`, nodes, and sigma support. Rollout and iteration allowances
must follow the deterministic search ladder in Step 6 within the supplied run
ceiling. Store the learned behavior in `OPTIMIZATION_POLICY` and the retained
allowance in `REFERENCE_OPTIMIZATION_BUDGET`. Prefer one final policy and
reference budget across the baseline and variants, and disclose any retained exception. Leave
`max_parallel_rollouts` at the shared machine-aware default; worker concurrency
is a runtime resource, not policy.

## Step 4 - Preserve physical fidelity

Follow the physical execution invariants in `AGENTS.md`. In particular, start
an embodied FR3 from its measured reset state, hand controller state through
typed transit, and begin the effect at measured precontact. Use
`osc.action_from_world_pose(...)` for resolved poses and require
`controller_rollout_backend == "native_cpp_fr3_osc"`. Do not perform an
unreported reset or post-correction.

## Step 5 - Create the baseline and variants

Freeze and run the fixed development qualification suite described in
`AGENTS.md`. Keep its cases separate from the variants below and retain its
report alongside behavioral evidence.

Preserve the assigned scene as `example_0`. Generate at least two deterministic
geometry/pose variants from fresh `MjSpec` values using
`balanced_generalization_samples` and the run seed.

Use each sample's `size_scale_xyz` directly in `SceneVariantSpec`: the three
factors independently scale the first bound collision geom's local XYZ axes.
The shared builder transforms visual geometry, collision geometry, mass and
inertia consistently. Derive fixture clearances and placement from the resized
geometry; do not average the factors into a uniform scale.

Create and statically validate `example_0` and every requested variant before
behavioral tuning. Develop on all examples in the same Codex phase. A failure
on any example may inform a shared learned-source repair, after which rerun the
affected examples and disclose the evidence and change.

Across the generated set cover renaming, translation, full-circle yaw,
independent three-axis resizing, target mass/inertia, contact friction,
primitive-goal parameter changes, and compatible fixture placement only when a retained essential or
optional fixture role exists. Each nonbaseline example must change at least
three eligible dimensions, including at least two physical dimensions.
For an primitive with parameters, include both a geometry variant with the
baseline goal and a variant with a different feasible goal.

Use the required variants as a contract-minimality test. At least one
compatible example must omit each optional contextual binding or bind it to an
empty collection while preserving the task affordance. The learned feature,
nominal, transit, cost, and predicate code must still compile and run. Do not
add dummy objects merely to satisfy a contract. Preserve the complete assigned
scene only in `example_0`. In every nonbaseline example, remove bodies that the
role-necessity audit classified as incidental. Retain only required roles,
their necessary physical parent/child closure, the embodied tool and support,
and entities explicitly used by `G`, `M`, or `Psi`. Optional context may remain
only in a deliberate compatible variant that tests it; document why it remains.
Never rotate, translate, resize, or relabel an incidental body merely to count
it as a fixture/generalization dimension. Delete the complete incidental body
subtree only after checking joints, tendons, actuators, sensors, equality
constraints, contacts, and other named references. If a reference is causal,
retain its minimal physical closure; otherwise remove the reference too, then
compile the resulting `MjSpec` before accepting the variant.

Before applying pose samples, identify the groups of bound entities whose
relative geometry defines the mechanism or `Psi`. Apply each sampled rigid
transform jointly to those entities so their defining relations are preserved.
Use `SceneVariantSpec.yaw_coupled_bodies`. Set
`allow_independent_fixture_yaw=True` only when independent relative yaw is the
intentional test and the required affordance remains valid.

Each example needs:

```text
test_scenes/<name>/example_<k>/
  __init__.py
  scene.py
  variant.json

src/instantiations/primitives/<name>/example_<k>/
  __init__.py       # canonical
  binding.json      # data only
  main.py           # canonical
```

Create immutable instantiation scaffolds with
`agent.instantiation_context.stage_canonical_instantiation`; edit only
`binding.json`. Never create `binding.py`, `mapping.py`, helper modules, custom
entrypoints, or executable scene adapters. `variant.json` must exactly equal
`VARIANT_SPEC.to_dict()`, and each manifest must reference its matching scene.
Every example is a complete primitive invocation: `VARIANT_SPEC.goal_parameters`
must exactly equal `binding.json`'s `inputs` and cover the primitive contract.
`example_0` records the demonstration invocation. Each variant records all
parameter values and lists values that differ from `example_0` in
`varied_goal_parameters`. Parameter-free primitives use empty mappings.

Read each exact runtime scene and binding, render and inspect its initialized
layout,
and combine that evidence with `validate_variant_initial_state` and binding
validation. Inspect task relations and robot approach/contact space rather than
treating helper success as proof of feasibility. Follow the three-outcome
feasibility procedure in `AGENTS.md`. From `G`, `M`, and `Psi`, derive the requested terminal
state or admissible region and candidate interaction paths. Record roles,
frames, contact mode, moving bodies, measured inequalities, candidate budget
and decision under `agent_work/`. A failed nominal path is a failed candidate;
allow bounded search without weakening physical constraints. A movable
incidental body remains an obstacle unless its motion is an explicit measured
relation in `M`. Co-transform goal/reference frames with each retained layout
group, or transport frame-relative goal values explicitly.

## Step 6 - Run, inspect, and improve upstream

Keep `results_instantiation/` empty throughout invention. That directory is
reserved exclusively for later deploy-instantiation runs. Every invention-time
execution must pass explicit artifact paths: use `agent_work/debug/` for
disposable metrics and recordings, and `results/` only for retained evidence.
Never rely on the instantiation entry point's default output paths.

Run every example through its canonical module entrypoint:

```bash
$RAPID_REPO_ROOT/scripts/rapid-python \
  -m instantiations.primitives.<name>.example_<k>.main
```

First run static/package, manifest, contract, and variant validation without
consuming simulator calls. Record their findings, repair genuine defects, and
then run bounded behavioral experiments. Start from the same optimizer policy
on every example; adapt it only with task-relevant evidence and disclose
retained differences.

Both feasible candidates and unresolved invocations may enter the bounded
search ladder. Reserve `scene_or_goal_infeasible` for an evidenced violation of
a necessary physical condition. Incorrect frame transport is a code defect.
Candidate collisions or finite IK failure alone do not justify that label.

Treat the nominal trajectory as an initialization, not as a candidate that must
succeed. A nominal or 1x1 run may verify package wiring, finite shapes, action
semantics, typed transit/controller handoff, telemetry, and deterministic hard
safety checks only. Nominal success, contact, and task progress are not
prerequisites for population search. Do not infer search support or edit learned
motion coefficients or topology from a 1x1 behavioral outcome alone; one sample
cannot show whether the declared sigma reaches the contact manifold or whether
the objective ranks useful perturbations.

After wiring passes, immediately run the minimum search on every feasible
example. A second nominal-only behavioral revision before that search is not
allowed. Repeat a 1x1 first only when deterministic evidence prevented a
meaningful population evaluation: a wrong frame/sign/action order, an
unreachable typed transit or required contact outside the declared sigma
support, nonfinite execution, missing telemetry/objective coverage, or a hard
safety violation. After repairing such a blocker, use at most one 1x1 to
confirm that the blocker is gone, then run the minimum search even when nominal
contact or progress remains zero.

Use visual feedback as a bounded debugging checkpoint, not only as a final
audit and not as a default for every optimization attempt. After the first
minimum-or-higher search produces task-relevant interaction for a design
fingerprint, record and inspect the single most informative unresolved example
before further source tuning, even when `Psi` is false. Allow one diagnostic
render per design fingerprint by default. Use `--no-record` for repeated runs
with the same physical failure signature and cite the existing checkpoint.
Render one additional representative only when another example has a distinct
failure signature, metrics disagree with the recording, or the existing video
does not expose the critical phase. A behavior-changing revision starts a new
checkpoint only after minimum-or-higher search shows a material change in
contact body/face/order, task-required attitude, path or knot timing, transit or
handoff, release or settling, collision, or protected-object drift. A budget,
seed, documentation, or threshold-only change does not by itself require a new
video. Put disposable debugging artifacts under `agent_work/debug/`. Inspect at
least reset, precontact, first contact, peak interaction, release, and final
state, and record the design fingerprint, metrics path, video path, reason for
the checkpoint, and observed failure signature in the invention report.
Classify population failures before editing:

- wrong semantic role or broken scene relation;
- geometry resolver/frame/sign error;
- invalid nominal topology or clearance scaling;
- transit contact, protected-object drift, or controller-handoff failure;
- cost/predicate mismatch or missing telemetry;
- insufficient search budget or sigma support;
- penetration, torque/joint-margin, jitter, reversal, or backtracking issue.

Use this deterministic search ladder, capped by the run prompt:

1. **Nominal diagnostic:** use at most one 1x1 smoke to verify wiring only; do
   not require behavioral success and do not tune from its task outcome.
2. **Minimum search:** immediately run 32 rollouts x 6 requested iterations on
   every feasible example after wiring passes. Allow predicate-gated early
   stopping only after 4 iterations. Early stopping additionally requires the
   selected trajectory to satisfy the primitive effect predicate and its
   cost/progress to plateau; a plateau while the predicate is false must run to
   the requested iteration limit. This population, not the nominal sample,
   provides the first evidence about contact acquisition and sigma support.
3. **Standard search:** run 64 x 10 on unresolved examples whose minimum-search
   candidates are finite and safe and show a physically plausible mechanism.
4. **Expanded search:** run at most 96 x 16 once for one unresolved
   representative example under an unchanged design fingerprint. If it
   succeeds, freeze that reference budget and verify it on every example.

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
primitive package, binding contract, and shared runtime revision. Scene binding,
task parameters, optimizer settings, and seeds belong to the separate
execution fingerprint; changing only those values is not a design change.

Do not tune a validated manifest in response to ordinary search variance or a
failed candidate. Preserve assigned and fixed validation scenes. Resample a
development variant under the evidence-based review and bounded-resampling
rule in `AGENTS.md`; preserve and report the original unresolved case.
Improve upstream logic for shared program defects and rerun all examples and
fixed probes after behavior-changing edits. Satisfy the bounded visual
checkpoint above before further tuning when those runs show
a material physical change or a metrics/recording disagreement.

For the retained run of every example, write exactly:

```bash
$RAPID_REPO_ROOT/scripts/rapid-python \
  -m instantiations.primitives.<name>.example_<k>.main \
  --seed <retained-seed> \
  --metrics-json results/<name>_example_<k>_metrics.json \
  --record results/<name>_example_<k>.mp4
```

A nominal or 1x1 diagnostic is never retained as final evidence. Successful
evidence uses the frozen `OPTIMIZATION_POLICY`; unresolved evidence uses the
highest justified attempted tier.

Retain representative verification frames under `agent_work/verify/`. Check
them against the demonstration for contact mode, direction, target effect,
clearance, final relation, penetration, jitter, and unnecessary motion. Verify
that task-layout yaw co-rotates every task-required axis, the world-space hand
local-X axis avoids alignment with the support normal, reorientation occurs in
free space, and contact phases contain no unexplained cant, avoidable wrist
roll, or orientation oscillation.

## Step 7 - Validation evidence

Before finishing, evaluate all of the following. Report workflow completion,
behavioral `Psi` success, and audit status independently. Preserve useful
trajectories and list any unmet or disputed item in the final report:

- strict manifests pass JSON schema, typed parameter, entity, and relation
  validation before simulation;
- every example records a complete primitive goal, variant metadata matches its
  binding inputs, and the set covers geometry-only and goal changes;
- every required role has a documented causal use, optional roles tolerate an
  empty binding, and incidental demonstration entities are absent from the
  contract;
- `example_0` preserves the assigned scene, while every nonbaseline runtime
  scene contains only its documented causal role closure and compiles after
  incidental bodies and stale references are removed;
- every example has a recorded feasibility classification and evidence for its
  exact runtime scene and invocation, distinguishing unresolved search;
- the fixed qualification report covers scene interface, representation and
  layout checks against the final learned source;
- learned code contains no assigned-task, generated-scene, or instantiation
  imports or names;
- compilation actually invokes `resolve_geometry`, `make_nominal_plan`, the
  learned cost/predicate, and shared typed transit runtime;
- nominal plans contain finite correctly shaped knots/sigma and transits are
  valid on all examples;
- every FR3 nominal plan declares the correct `GripperIntent`; nonprehensile
  `HOLD` commands are constant and use only `MOVE`, `ACTUATE` aperture and
  operation profiles are frozen, `GRASP` intervals are non-increasing, gripper
  optimizer exploration is zero, and any transit configuration completes and
  settles before approach; inherited-contact release preserves the aperture,
  and any requested aperture configuration happens only after certified
  separation;
- no action, waypoint, policy coefficient, custom predicate, callback, or
  transit executor leaks into an instantiation;
- transformed/resized scenes preserve relational behavior with the same
  learned coefficients;
- optimized `Psi` succeeds on the demo and every required variant, with any
  budget adaptation recorded;
- every retained persistent result passes the shared bounded terminal-stability
  probe, with the postcondition true before release and retained after settling;
- required relations in `M` are reported separately and every retained run
  includes finite cost-component and partial-progress diagnostics;
- FR3 runs report the native controller backend and valid controller-state
  handoff;
- convergence settling, whole-arm clearance/contact audits,
  collision/penetration checks, gripper tracking/contact-motion audits, and
  smoothness checks are reported separately from `Psi`;
- retained metrics and nonempty videos exist for every example and visual
  review confirms the intended physical effect.

Each retained metrics file must include nominal success/cost, optimized cost,
optimizer improvement, nominal-to-optimized control RMS/max, optimized effect
success, transit/audit validity, controller backend/handoff, generalization
metadata, requested/effective/minimum horizon, actual/requested iterations,
early-stop status, whole-arm audit results, and effect, mechanism, progress,
cost-component, and smoothness diagnostics. Nominal-only success is diagnostic
and is not required.

## Final report and quality bar

Report the primitive name, semantic `G`, mechanism `M`, full `Psi`, progress
representation `P(t)`, primitive and knot topology, relational feature
resolver, frozen nominal/transit heuristic,
invocation parameters, package/test/instantiation file trees, retained
optimizer settings, and exact validation commands. For every example, report all
predicate conjuncts, effect/mechanism/progress/cost-component diagnostics,
transit/contact/collision and smoothness results, plus metrics/video paths. End
with limitations and unsupported geometry.

All generated Python must run. Do not leave TODOs, placeholder stubs, fake
metrics, fake success, scripted target motion, or hidden postprocessing. Use
type hints, short module docstrings, dataclasses where useful, and
`from __future__ import annotations`.
