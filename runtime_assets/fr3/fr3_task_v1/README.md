# FR3 task runtime asset, version 1

This immutable asset is the shared full-arm FR3 used by packaged Real2Sim
tasks. Behavioral changes require a new versioned directory rather than edits
in place.

Relative to the vendored Judo FR3 source, this task version:

- sets all six collision geoms on each finger to MuJoCo friction
  `1.5 0.02 0.002` (sliding, torsional, and rolling);
- removes the Judo sensor section whose sensors refer to scene-specific
  `object` and `table` bodies; and
- rewrites mesh references so the model is self-contained in this directory.

All other dynamics, joint, actuator, and controller parameters are retained
from the current packaged FR3. See `LICENSE` for the upstream asset license.
