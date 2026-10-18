---
name: rapid-instantiate-strategy
description: Create and evaluate strict data-only strategy bindings for invention examples and generated variants. Do not use for held-out deployment.
---

# Instantiate a strategy during invention

This workflow determines whether a scene satisfies an already learned strategy
contract. It does not author phases, motion, transits, or success logic.

## Read first

1. The run prompt, scene inventory, XML/module, and attached views.
2. `src/strategies/<strategy>/binding_contract.py`, `program.py`, predicate,
   description, and imported primitive binding contracts.
3. `src/instantiation_core/contracts.py` and `manifest.py`.

The learned strategy owns phase order, primitive-role maps, connecting
constraints, typed inter-phase `TransitSpec` values, controller-state/aperture
handoff, and final `Psi`. Primitives own their feature resolution, nominal
effect knots, sigma, effect predicates, costs, tolerances, and standalone
precontact transit.

## Instantiation output

The launcher has created:

```text
src/instantiations/strategies/<strategy>/<instance>/
  __init__.py     # canonical, read-only
  binding.json    # editable semantic names; inputs remains empty
  main.py         # canonical, read-only
```

Do not create `binding.py`, `mapping.py`, phase factories, callbacks, custom
predicates, waypoint arrays, transit executors, or any other instantiation
file. The selected strategy, imported primitives, shared runtime, assigned tasks,
generated test scenes, and the entrypoint's dispatch logic are read-only.
Preserve any learned-program defect as evidence for a later invention run.

The `rapid.binding/v1` manifest contains launcher-provided identity and goal
metadata plus semantic role names. Set `inputs` to `{}`. Connecting constraints
derive primitive parameters from live geometry and propagated phase state.

## Procedure

1. Resolve each strategy role to concrete names using structural and visual
   evidence. Verify type, multiplicity, parent body, and relations.
2. Ensure that each frozen phase role map can satisfy its primitive contract.
   Do not create phase-specific bindings in the instantiation.
3. Keep manifest `inputs` empty and verify that every declared primitive
   parameter is covered by a learned connecting constraint.
4. Run manifest/layout validation and correct role-selection errors now.
   Validation must pass before simulation.
5. Start with one explicitly bounded smoke execution using one optimizer
   iteration per phase, small rollout counts, and horizon caps. Shared minimum
   timing, convergence/gripper settling, deterministic gripper intent, and
   retained whole-arm/contact/aperture audits remain active. Keep every smoke
   artifact under the run-provided `agent_work/.../smoke/` directory. This
   checks wiring; it need not satisfy phase or final predicates.
6. If the smoke fails, inspect its retained trajectory and structured failure
   signature. Correct genuine binding mistakes in the manifest. Do not correct
   phase, primitive, motion, geometry, predicate, transit, or audit behavior in
   learned/shared source, and never compensate with instantiation adapter logic.
7. Rerun the bounded smoke only after correcting a genuine binding mistake.
   Do not calibrate or retain scene-specific phase budgets.
8. Once binding and execution wiring are valid, run the plain user-facing
   module command exactly once without smoke caps to evaluate the strategy's
   retained phase budgets. It automatically writes `metrics.json` and
   `trajectory.mp4` under
   `results_instantiation/strategies/<strategy>/<instance>/`. FR3 OSC programs
   also write `cartesian_goals.json`: the 50 Hz commanded EE goals for
   `play_cartesian_goals.py`, not achieved simulator poses.

Binding validity is required. Phase/final predicates and audit results are
separate behavioral evidence and are advisory to the workflow: preserve useful
trajectories even when a check disagrees, investigate the disagreement, and do
not falsify success. A compatible binding that executes poorly is a reason to
recommend a later invention repair, never permission for this agent to edit the
learned strategy/primitive or add adapter logic.

## Completion report

Report the manifest path, semantic roles, compatibility result, every smoke
command, and observed resolved phase inputs and per-phase/final metrics, including
phase gripper intent and aperture-boundary diagnostics for FR3. List every retry
and unresolved validator disagreement. Explicitly confirm
that learned/shared source was not modified and that package changes were
limited to `binding.json`.
