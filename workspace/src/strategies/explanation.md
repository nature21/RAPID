# Strategy Composition

A composition-time strategy is a scene-agnostic sequential controller that composes
primitives from the learned library. It publishes a `BINDING_CONTRACT` and
receives one explicit data-only binding:

```python
compile_strategy(binding, args)
```

Assigned-scene names live in the instance's strict `binding.json`; its `inputs`
mapping is empty. Layout derivation and phase bindings remain in learned code.

Each primitive is treated as a parametric trajectory optimizer:

```text
execute(BoundExecutionContext, cfg)
  -> primitive-owned precontact transit
  -> learned geometry resolver + learned NominalPlan + local optimization
  -> CompileResult
```

The strategy owns phase ordering, semantic primitive-role mappings, late-bound
connecting constraints, typed inter-phase transit intent, and the final success
predicate. Each primitive owns its parameter interface, relational geometry
resolver, nominal effect knots and sigma, effect predicate/tolerances/cost, and
precontact geometry and transit for standalone and composed execution.

The standalone `compile_primitive(binding, args)` wrapper creates the initial
context and calls the same `execute` entrypoint used by composition. A strategy
phase runner can call it with `context.bind_primitive(primitive_binding)`, which
restores live `MjData` and carries the controller state, last command, and
gripper operation. The primitive continues from that state without resetting
the scene. An existing equivalent complete entrypoint can also be used.

A composition constraint converts a relational phase goal into the input frame
of a reusable primitive. It is evaluated immediately before that phase from the
bound semantic roles and propagated physical state. For example, a divider may
define a desired target contact position; the strategy subtracts the target's
current position and projects the resulting displacement onto the push
primitive's declared target/support axes. This remains a generic push—the
divider selects the subgoal but does not change the primitive's intrinsic
effect. Resolved composition values never live in `binding.json`.

A primitive `ParameterRequirement` represents an intrinsic effect goal such as
displacement, direction, or angle. Construction examples infer demonstrated values;
direct primitive invocations supply values; strategies derive values from live
relations. Contact allowances, timing relationships, and motion-policy
coefficients remain frozen learned logic.

For a multi-phase strategy:

```text
state_0
  -> compile local primitive 1 with x0=state_0
  -> phase_1.best_states[-1]
  -> optional release, withdrawal, or scene-level routing transit
  -> compile local primitive 2 with x0=transit_or_phase_1_final_state
  -> ...
  -> final success predicate
```

composition strategies use `execute_sequential_strategy`, which runs each
target-changing phase through its named learned primitive as a local optimization
problem. Every required primitive parameter has a `ConnectingConstraint` whose
resolver is evaluated exactly once from the live phase-start state. The shared
executor constructs and validates the ephemeral primitive binding. After each
phase, the next constraint sees `phase_result.best_states[-1]`, the inherited
controller state, and `phase_result.trajectory.controls[-1]` as the continuous
initial action.

A primitive effect predicate guides its local search and remains required
diagnostic evidence, but a false predicate does not stop strategy execution.
The executor propagates the retained live state through the remaining transits
and phases and reports both per-phase and aggregate primitive-effect status.
Only an execution or contract error prevents the next phase from running.

The strategy's final `Psi` defines program success. Primitive-effect results are
reported independently and must not be conjoined with `Psi`.

Non-effect transitions use frozen typed `TransitSpec` stages. Shared runtime
realizes dimension-relative waypoints, simulates them from the current state,
audits protected objects/contact, and hands physical/controller state to the
next phase. By default, the next primitive owns its approach. A custom
inter-phase approach reuses that primitive's precontact resolver and then calls
its effect execution, so approach geometry has one owner and approach executes
once. `ConfigureGripper` establishes and settles the next phase's
physical aperture before `Approach`; the command remains fixed afterward and
passive contact-compliance deflection is reported separately through retained
aperture diagnostics. Phase concatenation retains aperture-boundary
diagnostics. Scene instantiations cannot add wrappers, costs, callbacks,
waypoints, or predicates. A mismatch must be corrected in learned code during
invention and revalidated against unchanged manifests.

The executor's `attempt` policy can recover retained `run_transit` failures;
exceptions inside `run_phase` follow the phase failure path. Preserve this
distinction when choosing where an existing primitive's approach executes.

Strategy code should import learned primitives by name, for example
`primitives.some_primitive`.

The clean `rapid` baseline intentionally contains no reference
strategies. composition-stage agents should build strategies from the learned
primitives present in the active workspace checkpoint.
