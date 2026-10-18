---
name: rapid-deploy-strategy
description: Bind a frozen learned RAPID strategy to a held-out MuJoCo scene for fast deployment. Do not use for strategy invention, invention variants, optimizer calibration, or learned-source repair.
---

# Deploy a frozen strategy

Decide whether the selected strategy is compatible with the assigned scene and,
when it is, return its semantic role mapping. The launcher owns manifest
construction, validation, execution, and artifacts.

## Supplied evidence

Use only the run prompt's:

- natural-language goal;
- compact strategy contract and semantic summary;
- frozen phase/primitive role maps and connecting-constraint descriptions;
- compact scene inventory; and
- attached initial scene view, when present.

Do not inspect workspace source, validator implementation, costs, rollout code,
or prior runs. Use visual evidence only to distinguish structurally plausible
entities whose names and relationships remain ambiguous.

The goal text is compatibility metadata. A strategy binding has no numeric
primitive inputs. Learned connecting constraints resolve every primitive
parameter from the bound geometry and propagated phase-start state immediately
before that phase's local optimization.

## Frozen boundary

The strategy, primitives, scene, and shared runtime are read-only. Do not edit or
create files, run commands or simulation, inspect trajectories, tune optimizer
settings, choose seeds or horizons, or install per-scene runtime defaults. Do
not invent missing roles or executable adapter logic.

## Binding decision

The supplied `structural_role_candidates` have been filtered by entity kind,
type, required joints and singular structural relations. Choose semantic
assignments within these domains and verify their joint consistency; a
candidate list alone does not establish semantic compatibility. The host may
resolve a body role from the unique direct owner of your selected site, geom
or joint. It preserves ambiguous selections for correction and validates the
complete mapping before writing a binding.

1. Verify that the goal is compatible with the strategy's learned final effect.
2. Resolve every required role using its kind, multiplicity, body relationship,
   declared relations, semantic description, and scene evidence.
3. Bind optional roles only when matching entities are present.
4. Verify that the frozen phase role maps remain satisfiable.
5. Return incompatibility when a required role is absent or when supplied
   evidence cannot distinguish multiple candidates.

Resolve related roles jointly rather than committing to each name in
isolation. Filter candidates by entity kind and declared type, multiplicity,
and joint requirements, then apply ownership and relational constraints
across the mapping. A role key is a semantic variable; its value must be an
entity name from the corresponding inventory, not another role key.

For a site or geom constrained by `same_body_as`, its inventory `body` must
equal the scene entity selected for the referenced body role. An ancestor or
assembly root is not interchangeable with the exact owning body. Use that
ownership evidence to constrain both assignments before relying on names or
the image. Check every connected relation again after changing either role.

A rejected mapping is not proof that the scene is incompatible. When given
validator findings, reconsider all roles participating in each failed
relation, including a referenced body role even if the finding names the
attached entity first. Keep unrelated assignments fixed. Return incompatibility
only when the supplied evidence leaves no valid mapping or cannot resolve
semantically distinct candidates; do not repeat a mapping that still violates
the reported structural constraint.

## Response

Return exactly one JSON object without Markdown or surrounding commentary.

For a compatible scene:

```json
{
  "schema_version": "rapid.deploy-binding-response/v1",
  "compatible": true,
  "roles": {
    "strategy_role": "scene_entity"
  },
  "diagnostics": []
}
```

Use arrays for multi-valued roles. Omit unbound optional roles unless the
contract requires an explicit empty array.

For an incompatible or unresolved scene:

```json
{
  "schema_version": "rapid.deploy-binding-response/v1",
  "compatible": false,
  "roles": {},
  "diagnostics": [
    "Concise description of the missing, invalid, or ambiguous role."
  ]
}
```

Do not include identity fields, goal text, `inputs`, optimizer settings,
geometry values, code, or output paths. If the launcher provides validator
findings, reconsider the connected implicated roles and return one corrected
JSON response. If they cannot be resolved from the supplied evidence, return an
incompatible response.
