# Primitive Core

`primitive_core/` contains shared infrastructure used by invented primitives.

Concrete primitive implementations must not live directly in this package.
Construction agents create canonical implementations under `primitives/<primitive>/`.

Shared source is fixed for each experiment batch. Invention may configure public
contracts but must not edit this package or monkey-patch its implementation.

`geometry.geom_distance_within` returns a measured distance within the requested
threshold, or `None`; MuJoCo's distance-query cutoff is not contact evidence.
`geometry.contact_pairs` reports actual intersecting contact pairs. Learned
code must still verify the contact mode required by its declared mechanism.
Use `PhysicalPreconditionError` with finite measurements for a necessary
physical-condition rejection, and `CandidateSearchExhausted` with the examined
candidate count for an unresolved search. Ordinary `ValueError` indicates a
contract or implementation defect.

Primitives publish `effect_success`, `mechanism_valid`, and `stability_required`.
When terminal semantics require persistence, `stability_valid` records its
bounded physical probe. `evidence.finalize_primitive_result` derives final
`success` from these fields and physical transit validity; missing evidence
cannot pass. `optimized_success` remains the optimizer's effect predicate.
`profiling.execution_timer` measures named stages, including failed searches;
reported times are inclusive and must not be summed across nested stages.

For native FR3 Cartesian control, `NominalPlan.constant_rotation_from_knot=k`
couples the roll/pitch/yaw channels from noninitial knot `k` to the final knot.
The nominal suffix attitudes must match. Each optimizer candidate searches one
common attitude; the first command remains fixed. Timing projection bounds the
entry rotation, and subsequent rotations remain constant. Positions and gripper
semantics retain their normal behavior. `None` (the default) leaves attitudes
independent. `PrimitiveCompileSpec.from_nominal_plan` carries this declaration
into compilation.

Shared infrastructure remains in this package:

- `primitive_core.common`
- `primitive_core.compile`
- `primitive_core.fr3_osc`
- `primitive_core.fr3_safety`
- `primitive_core.fr3_transit`
- `primitive_core.native_fr3_osc`
- `primitive_core.policy`
- `primitive_core.rollout_control`
- `primitive_core.transit`
- `primitive_core.interaction_descriptors.md`

## FR3 Cartesian-action learning

`FR3OSCAdapter` uses the Cartesian optimizer action order:

```text
x, y, z, roll, pitch, yaw, gripper
```

The pose is expressed in the fixed FR3 base frame. During every MuJoCo
physics step, the native C++ FR3 rollout backend converts that seven-channel
equilibrium pose into bounded FR3 joint torques using Cartesian impedance,
dynamics compensation, and nullspace posture control. The controller stays
inside each MuJoCo worker thread, so rollouts do not cross the Python callback
boundary at every step. The optimizer never receives joint positions or
torques as actions. A separate per-knot `GripperOperation` schedule is
zero-order-held into each rollout sample. It is fixed by the nominal policy,
never optimized, and never inferred from aperture direction.

An embodied learned primitive resolves bound relational features and constructs
its frozen nominal policy before calling the internal shared compiler:

```python
osc = FR3OSCAdapter.from_model_data(model, data)
context = BoundExecutionContext(model, data, binding, initial_state)
features = resolve_geometry(context)
nominal = make_nominal_plan(context, features)

# Published by learned nominal.py and passed to InstantiationSpec.
OPTIMIZATION_POLICY = OptimizationPolicy(
    num_nodes=6,
    elite_fraction=0.125,
    horizon=2.5,
    min_iters=4,
    early_stop_patience=2,
)
REFERENCE_OPTIMIZATION_BUDGET = OptimizationBudget(
    num_rollouts=64,
    max_iters=10,
)

spec = PrimitiveCompileSpec.from_nominal_plan(
    nominal_plan=nominal,
    model=model,
    initial_state=context.initial_state,
    cost_fn=cost_fn,
    action_ctrlrange=osc.action_ctrlrange,
    rollout_control=osc,
    # Used only for retained whole-arm contact/clearance auditing.
    rollout_safety=FR3SafetySpec.from_model(model, osc),
)
```

Standalone execution uses the reference budget. A learned strategy may attach
a larger `OptimizationBudget` to an `PrimitivePhase`, bounded by
`GLOBAL_MAX_OPTIMIZATION_BUDGET` (96 rollouts and 16 iterations). Benchmark
execution selects that global maximum for every primitive phase. The runtime
derives `num_elites` from `elite_fraction` after resolving the effective
rollout population, so changing compute does not silently change the learned
elite ratio. CLI and launcher limits may only cap the selected budget for
diagnostic or resource-constrained runs.

Early stopping is predicate-gated: it is eligible only after `min_iters`, after
the selected trajectory satisfies the primitive effect predicate, and after the
cost/progress has plateaued for `early_stop_patience`. A plateau while the
effect predicate is false never ends the search early.

`NominalPlan.knot_fractions` freezes the semantic timing allocation. For FR3
effects, runtime realizes those knots with minimum-jerk interpolation and
expands a requested horizon when the nominal translation, rotation, or gripper
travel requires more time under the versioned dynamics profile. Runtime never
shortens the learned horizon. Every sampled knot sequence is then projected
into the same interval-by-interval motion envelope before rollout and optimizer
update, so the retained trajectory cannot exceed that profile. FR3 plans must
fix a command-continuous first action. Population candidates are ranked by the
learned tensorized cost without a native binary safety penalty. Early stopping
is allowed only after effect-predicate success and a learned cost plateau.

For an effect-free reset-to-precontact approach, learned code declares a typed
`TransitSpec` with dimension-relative `ScaleExpr` values and calls shared
`execute_transit(...)` with the runtime-owned `cfg.transit_seed` and standalone
`transit_index=0`. A transit label is diagnostic and never seeds the planner.
Start the optimized interaction from
`transit.terminal_state`, pass
`transit.controller_state` through
`PrimitiveCompileSpec.initial_controller_state`, and clamp the first optimized
knot to `transit.terminal_action`. This preserves the torque-rate limiter state
without repeatedly sampling or simulating the free-space approach in CEM/MPPI.
Use `prepend_fr3_transit(...)` after evaluating the effect result so replay and
recording still show the complete physical motion. The effect compile reports
`controller_state_handoff=True` when the state was supplied.

The transit begins at the measured reset action; neither phase may teleport or
directly optimize the arm's initial joint configuration. Set
`optimize_initial_pose=False` and leave `initial_pose_slots` empty. Concrete
shared runtime audits protected-object drift, forbidden contacts, and the
strict bounded inherited-release rule. An inherited contact is first separated
with fixed aperture. Actual and solver-scale contact-equivalent pairs may only
separate without worsening; transient recontact is diagnostic and terminal
stable separation is required. Nearby pairs retain no contact
exemption and may not cross into penetration. Release planning samples a
continuous local SE(3) region with native kinematics/collision checks, requires
contact-pair clearance progress and contact-free floors along each path, and
uses a 13 mm planning tube around the 10 mm executed-state requirement. It
physically certifies at most four diverse native-OSC candidates and uses the
first measured violating pair and normal to prioritize any remaining
candidates. Joint-space planning then uses the same native checks while
retained controls remain Cartesian. The free path keeps the 10 mm global
obstacle buffer; only the final tool/next-target relation may use the smaller,
still contact-free 2 mm handoff margin. If the requested next initial pose is
in contact, runtime first checks its declared task-equivalent attitudes for IK,
then projects each reachable attitude to a nearby sampled safe pose using a
deterministic low-discrepancy translation search over the full 0.100 m ball at
2 mm radial resolution. The selected attitude and aperture remain fixed during
translation; no fixed search rays or learned retreat direction are used.
Runtime records both poses and their displacement. The following contact-rich
primitive starts from that physical state and owns all later contact behavior.
Instantiations provide no transit code.
Every `TransitSpec` ends with one `Settle` stage. Its duration is a minimum,
its maximum is the requested observation cap, and its stable duration is the
consecutive window in which pose/orientation error and Cartesian speed must
remain below tolerance. Shared hardware-parity runtime may expand moving and
settle horizons for geometry, dynamics, and controller filtering. A projection
also emits
`next_primitive_initial_knot_requires_safe_handoff_projection` and a structured
`initial_knot_advisory`; invention should move the nominal boundary into free
precontact space, not edit later contact-rich knots.

Retained validation is advisory. `transit_valid` means the simulated state and
controller trajectory are finite and can be handed to the next phase;
`transit_convergence_valid`, `transit_audit_valid`, and
`whole_arm_audit_valid` separately preserve stricter convergence, semantic,
clearance, and contact findings. Those findings do not abort composition or
overwrite an effect predicate. A non-finite execution remains a hard failure.
IK/projection, collision-free planning, and inherited-contact release failures
also remain hard because they do not produce a certified physical handoff.

Controller-aware cost functions may accept a third positional argument. It
contains the observations named by `FR3_OSC_OBSERVATION_NAMES`: actual
end-effector pose and twist, commanded task wrench, applied arm torques, and
minimum joint-limit margin. Phase handoff preserves the seven previous torques
needed by the torque-rate limiter.

Population-level features that MuJoCo does not expose through ordinary sensors
can be requested from the native rollout without replaying every saved state in
Python. Contact telemetry is opt-in and does not add a cost by itself:

```python
from primitive_core.rollout_control import (
    ContactPairTelemetry,
    RolloutTelemetrySpec,
)

telemetry = RolloutTelemetrySpec(
    contact_pairs=(
        ContactPairTelemetry(
            "forbidden_contact_count",
            tuple(robot_geom_ids),
            tuple(forbidden_geom_ids),
        ),
    ),
    include_maximum_penetration=True,
)
spec = PrimitiveCompileSpec(
    # ...normal fields...
    rollout_control=osc,
    rollout_telemetry=telemetry,
)

# Inside the batched cost function:
contact_features = telemetry.view(sensors, model.nsensordata)
forbidden_count = np.sum(contact_features[..., 0], axis=1)
maximum_penetration = np.max(contact_features[..., 1], axis=1)
```

The telemetry channels are inserted after `model.sensordata` and before the
OSC observations, which remain the final `len(FR3_OSC_OBSERVATION_NAMES)`
channels. When telemetry is not
requested, the native rollout follows its original output and compute path.
Do not call MuJoCo from a candidate-by-timestep Python cost loop; extend native
telemetry if another derived population feature is required. A single retained
trajectory may still be replayed for independent exact validation.

Controller-aware rollouts use the CPU MuJoCo backend and the native C++ OSC.
MJWarp does not yet execute this state-feedback controller and is rejected
explicitly. There is no separate Python callback implementation.

`FR3SafetySpec` resolves the complete robot collision geometry and optional
phase-specific allowed contact pairs for one retained whole-arm audit. The
native worker does not latch or raise explicit force, acceleration, joint, or
contact safety faults. It continues to apply the impedance controller's task
wrench, torque, joint-limit, and torque-rate saturation. After optimization,
the selected trajectory is replayed and audited for whole-arm contact and
minimum environment clearance. Those findings are retained in the phase and
top-level metrics as advisories, while the task effect predicate remains the
source of success or failure for a finite simulated trajectory. Malformed or
non-finite executions still stop the run.

These rollout backends are simulation implementations, not real-time robot
drivers. Their seven-channel contract, gains, saturations, nullspace target,
and torque-rate limit are kept explicit so the same law can be ported to a
1 kHz FR3 torque-control process. Hardware deployment still requires a
libfranka integration plus robot-side collision, communication, and
emergency-stop safety checks.
