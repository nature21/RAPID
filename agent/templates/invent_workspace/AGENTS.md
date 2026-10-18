# RAPID Invent Workspace

This file defines rules shared by every automatic RAPID invention workflow.
Primitive- and strategy-specific implementation steps live only in their
respective skills; the run prompt contains only task facts and phase limits.

## Workspace and runtime

- Treat this isolated workspace as the complete editable world. Source is in
  `src/`, generated scenes in `test_scenes/`, and artifacts in `results/` or
  `agent_work/`.
- `task_inputs/` contains immutable assigned inputs. Never inspect hidden
  baselines, sibling suite runs, or canonical source outside this workspace.
- Run Python only through `$RAPID_REPO_ROOT/scripts/rapid-python`. The launcher
  already sets `RAPID_WORKSPACE`, `RAPID_ACTIVE_SRC_DIR`, and `RAPID_SRC_DIR`.
  Never install packages or create replacement dependencies.

## Source ownership and integrity

- Learned primitives live in `src/primitives/<name>/`; learned strategies live
  in `src/strategies/<name>/`. They publish `binding_contract.py` and contain
  no assigned-scene names or imports.
- Concrete instantiations contain strict data-only `binding.json` plus the
  fixed launcher-generated `__init__.py` and `main.py` shell. Never add adapter
  code, callbacks, predicates, waypoints, actions, task-policy constants,
  output paths, or executable binding helpers.
- Shared infrastructure is read-only during invention: do not modify
  `primitive_core`, `instantiation_core`, `scene_io`, or the shared files directly
  under `strategies/` and `primitives/`. Do not monkey-patch these modules.
  Implement behavior in learned packages through their public contracts.
  Shared-source changes make the artifact unpublishable and stop downstream
  stages. Record changes to pre-existing learned programs separately.
- A native FR3 `NominalPlan` may declare `constant_rotation_from_knot=k` to
  share one searchable attitude from noninitial knot `k` through the final
  knot. Those nominal attitudes must agree. The compiler preserves the initial
  command and timing limits; the common attitude remains optimizable. The
  default `None` allows independent attitudes. Use this public parameter when
  appropriate; do not implement a private compiler extension.
- Learned primitives own geometry resolution, task policy, costs, predicates,
  tolerances, and typed transit intent. Their contracts declare intrinsic
  effect-goal parameters such as displacement, direction, or angle. A concrete
  primitive invocation supplies every declared value.
- Primitive parameter values may come from a direct invocation, inference from
  the demonstration for a construction example, or a strategy connecting constraint.
  Scene measurements and motion-policy coefficients remain in learned code.
- Parameter contracts express semantic and physical validity, not optimizer
  convenience. Do not promote an arbitrary absolute cutoff or a reference
  search scale into a hard parameter bound. Keep only limits justified by the
  declared semantics, hardware, safety, or numerical validity, and derive
  execution feasibility from the live state. Use bounded decomposition only
  when a required semantic transition or demonstrated execution-capability
  limit justifies it. A preferred search scale or residual error from one
  optimization does not establish such a limit. Preserve the full requested
  effect rather than rejecting, silently clipping, or masking it with retries.
- Strategy contracts and bindings are role-only. Learned connecting constraints
  resolve every primitive parameter from bound roles and the current propagated
  phase state immediately before local primitive optimization.

## Relational geometry and coordinate representations

- For each derived axis, origin, extent, and signed distance, identify the
  semantic relation that defines it, the bound entities it depends on, and
  its reference frame. Derive the physical goal in that relation's frame
  before expressing it in a primitive's input frame. An input frame does not
  determine the goal geometry.
- Body, collision-geometry, and site frames are distinct. Use the live frame
  of the entity defining the measured feature and account for local offsets
  and rotations. A body axis or origin is not a physical feature unless the
  contract establishes that meaning; asset coordinate conventions alone do
  not establish it. Include all relevant bound collision geometry in extents.
- Equivalent coordinate representations of the same physical state must give
  the same physical goals, feasibility decisions, and predicates. Under an
  allowed rigid layout transform, derived positions and directions must
  transform with the layout while intrinsic lengths and clearances agree.
  Transport declared inputs and reference frames consistently, and preserve
  gravity, embodiment, and other physical constraints of the comparison.
- A degenerate projection or tied candidate does not by itself prove a
  missing affordance. Resolve remaining candidates from the declared semantic
  relations and live geometry. Break ties only among task-equivalent feasible
  choices using a documented relational rule; report unresolved ambiguity
  when the required effect distinguishes them. Never invent an arbitrary
  world axis or relax a physical constraint to force a choice.
- Before rejecting geometry, record the defining roles, frames, measured
  quantities, and violated inequality. Distinguish a representation or
  selection error from a physical incompatibility. Repair the former in
  learned code; preserve genuine hardware and geometric feasibility limits.
- Use `primitive_core.geometry.geom_distance_within` for thresholded distance
  queries; its `None` means outside the query range. A distance-query ceiling
  is not a measured contact. Use `contact_pairs` for actual contact evidence,
  and verify the contact location, normal and moving entities before reusing
  a contact mode. Build acquisition from measured geometry; an inherited
  equilibrium command alone does not establish a physically acquired contact.
- Raise `PhysicalPreconditionError(condition, measurements=...)` only for a
  measured necessary physical condition. Use
  `CandidateSearchExhausted(message, candidates=...)` for exhausted candidate
  selection. Ordinary `ValueError`, shape, import and attribute errors are
  implementation defects, not physical rejection. Keep sampled orientation
  preferences separate from hardware limits. Check both signs and all
  task-equivalent physical features before excluding an invocation.

## Fixed development qualification

Before behavioral tuning, freeze a small set of validation inputs with
`instantiation_core.qualification.freeze_validation_cases` at
`agent_work/qualification/<program>/cases.json`. Use the requested seed and
declare transforms, perturbations, expected relations and tolerances explicitly.
Keep these cases separate from repairable development variants. Preserve every
case and failure; repair learned code rather than selecting easier inputs.
Do not inspect benchmark scenes, results or evaluator internals.

Run all frozen cases through `run_validation_suite`, writing `report.json` in
the same directory. Use its typed probes; callbacks returning a list of findings
are not accepted as evidence. Required categories are:

- `scene_interface`: `SceneInterfaceProbe(manifest)` compiles and executes the
  public entrypoint with a small diagnostic budget. Inspect learned source with
  `agent.instantiation_validation.scene_interface_findings` as well.
- `representation` and `layout`: use
  `GeometryProbe(reference, observed, negative_control)`. Each function receives
  the frozen case and returns physical observations from the actual learned
  resolver. The host compares observations using the case's `rotation`,
  `translation`, and `atol`, and requires the incorrect negative control to be
  distinguishable. Representation changes preserve physical geometry; layout
  transforms preserve semantic relations, gravity and embodiment.
- `handoff` (strategies): use `HandoffProbe(manifest, phase)` for every
  noninitial primitive phase on each required example. Normal invention rollouts
  automatically save immutable physical/controller boundary states under
  `agent_work/handoffs/<program>/<instance>/<phase>/`. The probe replays the
  actual transit and next primitive from that boundary with two rollouts and
  one iteration. Cases declare `configuration_perturbations`: a list of finite
  `model.nv` tangent vectors integrated into the retained configuration. Include
  zero and small perturbations within the declared preceding effect tolerance.
  It measures precontact reachability separately from low-budget effect success.
  Do not substitute initial-scene states or resolver-only checks for this probe.
  Missing or stale checkpoints require a development rerun, not a passing report.

Observations use named `positions`, `directions`, `scalars` and `decisions`;
decode primitive-frame outputs into world coordinates before comparison.
Measurement functions must call the actual learned resolvers. Negative controls
must expose a wrong frame or an unjustified rejection.
Unsupported probes are findings, never passing checks. Use the learned package
directory as `source_paths`; strategies also include the complete
`src/primitives` directory. Rerun the suite after source changes: qualification
reports are checked against source and case fingerprints and reported separately
from behavior success. These probes supplement physical development rollouts;
they do not establish held-out benchmark performance.

## Retained outcomes and execution cost

Every learned primitive publishes boolean `effect_success`, `mechanism_valid`
and `stability_required` in its final metrics. When its terminal semantics
require persistence, also publish boolean `stability_valid` from the bounded
terminal probe. Call `primitive_core.evidence.finalize_primitive_result` after
all retained diagnostics and probes. The shared caller also derives `success`
from these fields; `optimized_success` describes only the optimizer's effect
predicate. Missing evidence is incomplete, never successful. Do not require a
release when holding contact is part of the declared terminal semantics.
Strategy metrics keep final `primitive_phase_success` separate from measured
`primitive_phase_effect_success`; the strategy's own final goal predicate is
evaluated independently on the retained trajectory.

Use `primitive_core.profiling.execution_timer` around geometry, nominal-policy
construction and retained audits. The host records inclusive compiler, transit
and phase wall times. Keep candidate counts and rejection reasons in metrics;
numeric arrays larger than 256 values are stored in an adjacent compressed
NPZ by the artifact writer. Retain evidence without expanding trajectories into
JSON lists. Diagnose total search effort before changing budgets or adding
phases. Avoid terminal joint-limit states by considering measured arm margins
when selecting task-equivalent attitudes; do not change shared joint limits.

## Physical execution invariants

- Target changes must result from MuJoCo simulation. Never teleport, reset,
  interpolate, or post-correct a target, and never replace embodied robot/tool
  collision geometry with a proxy.
- Complete FR3 scenes use the shared native C++ OSC with the CPU MuJoCo rollout
  backend for effect and transit. Optimizer actions remain the seven Cartesian
  OSC channels; never add Python controller callbacks, IK, torque/joint action
  optimization, or controller-aware MJWarp.
- Visual human demonstrations specify contact mode, manipulation direction,
  object relations, and phase structure; they do not supply an FR3
  end-effector attitude. Resolve each parallel-jaw FR3 contact attitude from
  live target, support, fixture, manipulation direction, and retained
  obstacle/corridor geometry. Declare only the task-required contact/tool axes
  and the remaining task-equivalent rotational freedom. Use the current
  embodied hand attitude, or the propagated attitude from the preceding phase,
  only as the minimum-rotation tie-break among safe, reachable, converged,
  task-feasible attitudes.
- In the bundled FR3 hand frame, local X is the short-thickness axis of the
  hand collision geometry, local Y is the jaw-closing axis, and local Z is the
  remaining hand axis. Unless the contact mechanism explicitly requires an
  exception, reject attitudes whose world-space hand local-X axis is nearly
  parallel or antiparallel to the live tabletop support normal; this posture is
  kinematically restrictive and can destabilize planning and Cartesian
  control. Never substitute an axis constructed from task approach and jaw
  directions for hand local X. Use world vertical only when no semantic
  support frame exists, and report any required exception.
- Do not complete an underconstrained attitude with arbitrary task tangents or
  an unexplained fixed cant angle. Numeric attitude offsets must follow from
  task geometry, a shared physical threshold, or measured reachability and
  collision evidence, and must validate across every primitive example.
- Select one stable phase-level attitude and perform any material reorientation
  smoothly in free space before approach. Hold that attitude through contact
  unless tool rotation is intrinsic to the manipulation mechanism; never add
  arbitrary per-knot rotation, oscillation, or backtracking.
- The two wrist-roll attitudes separated by 180 degrees are local symmetry
  candidates only after the object-centric task axes are resolved. Hold every
  non-roll task quantity fixed when comparing them. Never rotate solely for a
  marginally larger aggregate minimum joint-margin scalar; inspect per-joint
  margins and do not improve another joint by driving the wrist closer to its
  limit. If candidates satisfy the same hard requirements and posture rule,
  retain the one requiring less rotation from the current embodied or
  propagated hand attitude.
- Effect-free approaches and inter-phase moves are typed relational transits,
  physically simulated by shared runtime, and end in convergence-based
  `Settle`. Retained whole-arm clearance, contact, drift, and controller-state
  handoff audits must be reported and reconciled with the actual trajectory;
  an audit disagreement is advisory to the invention workflow rather than a
  reason for the launcher to discard an otherwise useful trajectory.
- Every FR3 plan declares `GripperIntent` and one explicit
  `GripperOperation.MOVE` or `GripperOperation.GRASP` per nominal knot. The
  optimizer never explores channel 6 or the operation schedule. `HOLD` uses a
  constant aperture and only `MOVE`. `ACTUATE` freezes both its aperture
  profile and operation schedule; `GRASP` is permitted only on non-increasing
  aperture intervals. Aperture changes for nonprehensile primitives occur
  through `ConfigureGripper` in free-space transit before `Approach`.
- Leave rollout concurrency to shared machine-aware runtime. Do not hardcode
  worker or CPU counts in learned code.

## Validation and evaluation

- Run package, contract, manifest, and generated-scene checks before simulator
  work. Binding validity and behavioral predicate success are separate.
- Aim to solve every required development example. A completed workflow, a
  failed revision, or an exhausted optimizer configuration is not completion
  of invention. Use failures to choose the next informative experiment while
  the overall run budget permits. When that budget is exhausted, or a reasoned
  review leaves no promising next experiment within it, report partial
  completion, the remaining failures, and the evidence behind stopping. Keep
  workflow completion, behavioral success, and audit status separate.
- Preserve `example_0`, but make every generated nonbaseline variant contain
  only the causal role closure: required bound entities, their necessary
  parent/child physical structure, the embodied tool, supports, and entities
  explicitly used by the semantic goal, declared mechanism or phase program,
  or success predicate. Remove incidental scene bodies instead of translating
  or yaw-coupling them into new obstacles. Keep optional context only in a
  deliberate variant that tests it, and document its causal purpose.
- Check invocation feasibility against the exact runtime model and state.
  Distinguish invalid inputs or a violated necessary physical condition,
  a feasible candidate, and an unresolved search. Reject only with evidence
  of the first. Failure of one nominal path, sampled attitude set, or finite
  IK search does not prove task infeasibility. For unresolved cases, use a
  bounded search over task-equivalent candidates or the learned optimizer;
  report exhausted search as unresolved. Preserve shared collision, hardware,
  and contact constraints throughout. Model the actual contact mode and moving
  bodies in each witness; a static open-tool proxy cannot certify all phases.
  Initial review combines reading the actual test scene and binding, rendering
  and inspecting the initialized scene, and relevant physical checks such as
  `validate_variant_initial_state`. Check target/fixture/support relations,
  intended contact faces, approach space, and the robot's placement. Choose
  task-relevant views and measurements; passing a fixed helper is not a
  feasibility certificate. Revisit this review when execution stalls.
  If the main mechanism works but a measurable goal remains unmet, prioritize
  program improvement: diagnose contact, trajectory representation, objective
  ranking, controller tracking, or retention as appropriate. A short lift or
  excessive translation during an otherwise real rotation is evidence of an
  incomplete solution, not by itself a reason to replace the scene. The same
  failure across different layouts points to shared program weakness; repair
  that weakness or report it unresolved rather than resampling those cases.
  Resampling is for demonstrated physical incompatibility or a difficulty
  specific to a development scene that remains unresolved after substantial
  targeted attempts. For an unresolved case, inspect the failing trajectory
  and progress, investigate plausible geometry/contact/controller or objective
  defects, and try materially different task-equivalent approaches or
  evidence-supported shared repairs. Explain what distinguishes this case
  from a shared capability deficit; difficulty or iteration count alone is
  insufficient.
  Use meaningful population search when execution reaches the effect; escalate
  its budget when the mechanism and progress make further search promising.
  Repeating an unchanged failure or a nominal-only run is not sufficient.
  If transit prevents effect search, use measured alternative contact poses
  and paths rather than spending optimizer calls on an unreachable start.
  Review the remaining plausible program changes before replacement; a failed
  expanded search or one unsuccessful repair only evaluates that approach.
  Explain why further attempts on this scene are unlikely to be informative
  within the run budget. Keep this classification `unresolved`, not physically
  infeasible.
  Preserve the original scene, binding, source fingerprint, attempts and
  failure evidence before replacement. Sample a replacement deterministically
  with comparable task relations and variation coverage, before testing its
  behavior; do not lower goals or select among samples by success. Keep
  replacements bounded and report all replaced cases separately from retained
  successes. A replacement's success does not resolve the original case or
  establish that the program supports its difficult layout.
  Fixed validation cases and assigned scenes may not be replaced to avoid
  failure.
  Audit every learned geometry/preflight exception before freezing: identify
  the necessary condition it establishes or the exhausted search budget it
  reports. Reject individual failed candidates inside their search. An
  unsuccessful retained rollout must remain available for evaluation; do not
  turn its unsatisfied effect predicate into a strategy exception.
- Learned execution depends on the scene's `make_model()` and `make_data()`
  interface and live bound entities. Development metadata such as
  `VARIANT_SPEC` is optional and belongs to the shared artifact writer; do not
  read it in learned code or exception handlers. Return the executed result
  before optional reporting. Preserve the original exception and any retained
  result when execution fails.
- When a retained result is declared to persist after commanded interaction
  ends, verify it with one bounded terminal-stability probe. The postcondition
  must hold before the probe; use the shared transit runtime to release and
  settle, then confirm that it remains true. The probe may not create the
  postcondition. Reuse an equivalent retained transition when available, run
  the probe only for retained candidates, and use the runtime's existing
  convergence tolerances rather than exact-zero or stricter custom thresholds.
  Allow small settling transients, cap the observation window, and report
  release feasibility separately from result stability. When release conflicts
  with the declared terminal semantics, use the corresponding bounded
  terminal-state probe instead.
- Preserve the declared relational frame of every goal or derived subgoal under
  layout transforms. Numeric equality preserves a semantic goal only when its
  reference frame co-transforms. Treat every incidental or unbound object that
  intentionally remains as a real obstacle, even when it is dynamically
  movable, unless the declared mechanism or a measured phase effect explicitly
  requires moving it.
- Preserve task-defining relative geometry when generating variants; transform
  a target and its defining fixture as one layout group unless independent
  relative pose is the intentional valid test dimension.
- Validate behavior with retained metrics and video. Metrics never override a
  visibly incorrect contact sequence or physical effect, and a validator
  disagreement does not by itself override a visibly correct, useful
  trajectory. Preserve and explain both sources of evidence. This advisory
  exception does not apply to a deterministic pre-search proof that the
  requested invocation is infeasible.
- Invention uses one Codex phase for `example_0` and all requested variants.
  Treat the user's natural-language description as qualitative semantic
  guidance. Before simulation, commit the semantic goal, phase/contact
  structure, success predicate, and progress evidence required by the active
  invention skill. Every example then serves as development evidence. Source
  changes, repeated runs, and example reuse are provenance facts that must be
  disclosed.
- Report workflow completion, behavioral predicate success, and audit status
  independently. Collision/contact/clearance disagreement may remain an
  advisory when physical telemetry and video support the trajectory, but it
  must never be hidden or mislabeled as a clean audit.
