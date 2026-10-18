# Scene IO

Shared replay, recording, and playback helpers for task and generated
test-scene modules.

Canonical task environments live under `tasks/`. Agent-generated local
test scenes live in each active workspace under `test_scenes/`.

Generated scene layout:

```text
test_scenes/<primitive>/<example>/
  __init__.py
  scene.py
  replay.py
```

Runnable data-only binding packages belong under
`instantiations/primitives/<primitive>/<instance>/`, not next to the scene.
