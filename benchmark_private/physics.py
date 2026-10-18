"""Private physical-parameter mappings and calibration checks.

MuJoCo's compliant contacts do not expose a rigid-body Newton restitution
coefficient directly. The benchmark therefore treats ``restitution`` as a
normalized elasticity control and maps it monotonically to the contact
damping ratio. The drop calibration below makes that convention measurable
and regression-testable without exposing it in a benchmark scene package.
"""

from __future__ import annotations

import numpy as np


ELASTICITY_GAIN = 0.7
MIN_DAMPING_RATIO = 0.2


def elasticity_solref(
    time_constant: float,
    restitution: float,
) -> tuple[float, float]:
    """Map the sealed elasticity control to MuJoCo's positive solref form."""

    if not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be in [0, 1]")
    damping_ratio = float(
        np.clip(1.0 - ELASTICITY_GAIN * restitution, MIN_DAMPING_RATIO, 1.0)
    )
    return float(time_constant), damping_ratio


def calibrated_drop_rebound(restitution: float) -> float:
    """Measure the first rebound height for the benchmark contact mapping.

    The plane contact matches the support contact settings in the source
    real2sim scenes. This is a calibration probe, not a feasibility oracle.
    """

    import mujoco

    time_constant, damping_ratio = elasticity_solref(0.01, restitution)
    xml = f"""
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <geom type="plane" size="2 2 0.1" solref="0.01 0.979"/>
        <body pos="0 0 0.6">
          <freejoint/>
          <geom type="sphere" size="0.04" density="500"
                solref="{time_constant:.9g} {damping_ratio:.9g}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    touched = False
    rebound_height = 0.0
    for _ in range(1500):
        mujoco.mj_step(model, data)
        touched = touched or bool(data.ncon)
        if touched:
            rebound_height = max(rebound_height, float(data.xpos[1, 2]))
    return rebound_height


__all__ = ["calibrated_drop_rebound", "elasticity_solref"]
