---
name: rapid-instantiate-primitive
description: Bind an existing learned RAPID primitive to a concrete MuJoCo scene using a strict binding manifest. Use for invention examples and primitive generalization.
---

# Bind a learned primitive with data only

This is compatibility analysis, not motion invention. The learned primitive
already owns geometry resolution, nominal effect knots, sigma, clearances,
timing, tolerances, cost, effect predicate, and default precontact transit.

## Read first

1. The run prompt and attached scene views.
2. The deterministic scene inventory and assigned scene XML/module.
3. `src/primitives/<primitive>/binding_contract.py` and the primitive description.
4. `src/instantiation_core/contracts.py` and `manifest.py`.

The selected primitive, shared runtime, assigned tasks, generated test scenes,
and the entrypoint's dispatch logic are read-only. Preserve any
learned-program defect as evidence for a later invention run.

## Instantiation output

The launcher has created:

```text
src/instantiations/primitives/<primitive>/<instance>/
  __init__.py     # canonical, read-only
  binding.json    # editable semantic names and declared inputs
  main.py         # canonical, read-only
```

Do not create `binding.py`, `mapping.py`, helpers, callbacks, waypoint arrays,
or any other file in the package. The manifest schema is
`rapid.binding/v1`. Preserve the launcher-provided identity, scene module, and
goal text. Fill only:

- semantic role names required by `BINDING_CONTRACT`; and
- every contract-declared effect-goal parameter from the assigned primitive
  invocation. Construction examples use `VARIANT_SPEC.goal_parameters`; direct
  invocations use their supplied displacement, direction, angle, or other
  primitive goal.

Inputs that encode measured geometry, waypoints, poses, axes, motion stages,
clearances, timing fractions, sigma, gains, costs, thresholds, tolerances, or
success predicates are forbidden even when they would make the smoke run pass.
Unknown, missing, wrongly typed, and out-of-range inputs fail before simulation.
An invocation without every required primitive goal parameter is incomplete.

## Procedure

1. Match every semantic role to named scene entities using the inventory, XML,
   and rendered views. Treat names as evidence, not semantics.
2. Check kind, type, parent-body, multiplicity, and relational invariants.
3. Enter every declared primitive goal parameter from the assigned invocation.
   For construction examples, require exact equality between manifest `inputs` and
   `VARIANT_SPEC.goal_parameters`. Keep policy constants and measured scene
   geometry out of the manifest.
4. Run manifest/layout validation. Correct role-selection mistakes or
   incorrectly copied primitive parameters now; validation must pass before any
   simulation.
5. Start with one explicitly bounded smoke execution using one optimizer
   iteration, a small rollout count, and a horizon cap. Shared minimum timing,
   convergence/gripper settling, deterministic gripper intent, and retained
   whole-arm/contact/aperture audits remain active. Put all artifacts under the
   run-provided `agent_work/.../smoke/` directory. This checks wiring; it need
   not satisfy the effect predicate.
6. If the smoke fails, inspect its retained trajectory and structured failure
   signature. Correct genuine binding mistakes in the manifest. Do not correct
   motion, geometry, predicate, transit, or audit behavior in learned/shared
   source, and never compensate with instantiation adapter logic.
7. Rerun the bounded smoke only after correcting a genuine binding mistake.
   Do not calibrate or retain a scene-specific optimization budget.
8. Once binding and execution wiring are valid, run the plain user-facing
   module command exactly once without smoke caps to evaluate the learned policy
   and reference budget. It automatically writes `metrics.json` and
   `trajectory.mp4` under
   `results_instantiation/primitives/<primitive>/<instance>/`. FR3 OSC programs
   also write `cartesian_goals.json`: the 50 Hz commanded EE goals for
   `play_cartesian_goals.py`, not achieved simulator poses.

Binding validity is required. Predicate and audit results are separate
behavioral evidence and are advisory to the workflow: preserve useful
trajectories even when a check disagrees, investigate the disagreement, and do
not falsify success. A structurally compatible scene that executes poorly is a
reason to recommend a later invention repair, never permission for this agent
to edit the learned primitive or compensate in the manifest.

## Completion report

Report the manifest path, bound roles, primitive parameters, structural
validation result, every smoke command, and observed metrics, including gripper
intent and aperture diagnostics for FR3. List every retry and unresolved
validator disagreement. Explicitly confirm that learned/shared
source was not modified and that package changes were limited to `binding.json`.
