# Interaction descriptors

This document defines the small set of symbolic interaction descriptors that every primitive is built out of. Interaction descriptors are intentionally coarse,
contact-level abstractions: they tell the compiler *what kind of motion
or contact force to expect*, not the numerical trajectory. Each
primitive's ``primitive_description.md`` lists the motion and contact
description plus construction-time constraints (invariants that hold for
every invocation of the primitive).

Interaction descriptors come in two families:

1. **Motion descriptors** – describe how a body (tool, target, ...)
   moves through space.
2. **Contact-force descriptors** – describe a contact force that one body
   applies to another.

The interaction descriptors are never instantiated at runtime; they are purely
documentation / specification for the primitive's intended composition.

Every learned primitive specifies four distinct objects:

- `G` is the purely semantic, role-relative goal. It may name declared goal
  parameters but contains no values, equations, thresholds, weights, or motion.
- `M` is the ordered semantic mechanism: phases, required versus preferred
  relations, contacts, and transition events. It contains no numerical
  trajectory.
- `Psi` is the measurable object-centric predicate for achievement of `G`.
- `P(t)` is a vector of continuous, phase-specific, causally relevant progress
  features used to shape optimization. It is not a success predicate.

Effect success (`Psi`), required-mechanism validity (`M`), and safety/audit
validity are measured and reported separately. An invented primitive is accepted
only when its effect and required mechanism both succeed. Its scalar objective
may combine smooth goal, mechanism, progress, nominal-reference, and motion
regularization terms, but does not redefine `G`.

---

## Motion descriptors

### `Translate(body)`

The body undergoes translational motion. Used for tools that sweep
along a straight path or for targets whose centre-of-mass is intended
to move along a roughly straight line.

### `Rotate(body)`

The body rotates about an axis. Used for tools that trace a circular
arc or for targets that spin / tip.

### `Static(body)`

The body is intended to remain stationary, such as a support surface,
fixture, wall, or blocker. Useful to make the inertness of a scene
object explicit.

### `MoveInFreeSpace(body)`

The body moves through free space to a target pose, without making
contact with anything. Typically the approach phase of a primitive,
before the first contact.

For an embodied FR3, a relationally determined collision-free approach is
normally simulated once as a deterministic native-OSC transit and kept outside
the target-effect optimizer. Optimize it separately only when route selection
is itself constrained or ambiguous.

An `Approach` authorizes only contact-free proximity between its declared tool
and target. It does not authorize contact. Other robot/environment pairs retain
the shared 5 mm clearance requirement; actual contact needs a separate
`AllowedContactPair`.

### `MoveAlongSurface(body)`

The body moves while maintaining sliding contact with a named support
surface. Used for tools that must stay in a support plane while applying
contact.

---

## Contact-force descriptors

### `SideForce(tool, target)`

The tool applies a lateral (horizontal, in-plane) contact force on the
target.

### `UpForce(tool, target)`

The tool applies an upward contact force on the target from below.

### `DownForce(tool, target)`

The tool applies a downward contact force on the target from above.

---

## How interaction descriptors are used in a primitive description

A primitive description (``primitive_description.md`` inside
``primitives/<primitive>/``) lists:

1. The semantic goal `G`, mechanism `M`, success predicate `Psi`, and progress
   representation `P(t)` defined above.
2. The motion and contact description (an ordered collection of interaction
   descriptors).
3. The construction-time mode invariants – contact pairs that must remain
   engaged, whether the support contact is maintained, whether the
   tool must remain in the support plane, etc.
4. The effect schema – the macro-level outcome the primitive is supposed
   to realise (``translation``, ``target_flip``, ``in_plane_rotation``,
   ``vertical_transport``, ``target_toppling``, ``friction_translation``).

For example, a lateral translation primitive might be described as:

    body              = (Translate(tool), SideForce(tool, target))
    mode invariants   = tool_feature in contact with target_feature,
                        tool remains in the support plane
    effect schema     = translation (in a support/fixture-relative task frame)

The clean baseline has no concrete primitive folders. Construction agents create
them under ``primitives/<primitive>/``. The ``primitive_core/`` package contains
shared infrastructure only.

---

## Optimizer back-ends

Every primitive's ``compile_*`` entry point forwards to the shared driver
in ``compile``, which supports two sampling optimizers
from ``judo.optimizers`` selected via ``PrimitiveCompilerConfig.optimizer``:

| value | optimizer | knot-update rule | notes |
|-------|-----------|------------------|-------|
| ``"cem"`` (default) | ``CrossEntropyMethod`` | refit Gaussian to top-k elite knots | per-node sigma, clipped to ``[sigma_min, sigma_max]`` |
| ``"mppi"`` | ``MPPI`` | softmax weighting of all rollouts | scalar sigma (mean of ``sigma_init``), temperature ``mppi_temperature`` |

Both back-ends share the same spline parameterisation, initial-pose
optimisation hook, parallel rollout backend, and reward function, so a
primitive can be compiled with either optimiser without touching any
primitive-specific code. All demos expose ``--optimizer {cem,mppi}`` and
``--mppi-temperature`` command-line flags.

For FR3 plans, channel 6 is deterministic policy rather than search space.
Every `NominalPlan` declares `GripperIntent.HOLD` (constant aperture for a
nonprehensile effect) or `GripperIntent.ACTUATE` (a fixed intrinsic
grasp/release profile), plus one `GripperOperation.MOVE` or
`GripperOperation.GRASP` per nominal knot. The operation is zero-order-held to
trajectory samples and is not inferred from aperture direction. `HOLD` uses
only `MOVE`; `GRASP` may only accompany a non-increasing aperture. CEM and MPPI
copy the nominal per-knot gripper command into every candidate and never refit
that channel or the operation schedule. A nonprehensile aperture change belongs
in effect-free transit as `ConfigureGripper` before `Approach`; shared settling
checks physical aperture position and velocity before handoff.
