# Primitives

`primitives/` contains concrete primitive implementations invented by
construction-stage agents.

Layout:

```text
primitives/
  <primitive>/
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

The clean `rapid` baseline intentionally contains no concrete primitives.
The first construction run creates `primitives/<primitive>/`.

Every primitive publishes `BINDING_CONTRACT`, resolves relational geometry in
`features.py`, constructs frozen effect `NominalPlan` values in `nominal.py`,
and declares its default typed precontact transit in `transit.py`. Its cost,
effect predicate, tolerances, and deterministic compiler are learned code.

Suite instantiations contain only canonical entrypoints and strict
`binding.json` data. Assigned-scene names and declared primitive effect-goal
parameters belong in the manifest; state layouts, geometry derivation, motion, clearances,
tolerances, and predicates do not.

Use a distinct primitive name and package for an incompatible alternative.
Do not create version subpackages.
