# Instantiations

Generated scene bindings use program-first package paths:

```text
instantiations/primitives/<primitive>/<instance>/
instantiations/strategies/<strategy>/<instance>/
```

Each runnable package contains a strict versioned `binding.json` and canonical
launcher-generated `__init__.py`/`main.py`. The manifest contains only program
identity, scene and goal metadata, and semantic role names. Primitive manifests
also contain declared effect-goal parameters; strategy manifests keep `inputs`
empty. A manifest cannot contain executable mapping code, motion, geometry,
policy constants, or predicates. Runtime execution never invokes an agent.
