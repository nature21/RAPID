"""Measured collision geometry and explicit invocation outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import mujoco


class PhysicalPreconditionError(ValueError):
    """A measured necessary condition excludes the requested invocation."""

    def __init__(self, condition: str, *, measurements: Mapping[str, float]):
        if not condition or not measurements or not all(
            math.isfinite(float(value)) for value in measurements.values()
        ):
            raise ValueError("physical rejection requires a condition and finite measurements")
        self.condition = condition
        self.measurements = dict(measurements)
        super().__init__(f"{condition}: {self.measurements}")


class CandidateSearchExhausted(RuntimeError):
    """A finite candidate search did not establish a usable invocation."""

    def __init__(self, message: str, *, candidates: int):
        if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
            raise ValueError("exhausted search requires a positive candidate count")
        self.candidates = candidates
        super().__init__(f"{message} (candidates={candidates}; feasibility unresolved)")


def geom_distance_within(
    model: mujoco.MjModel, data: mujoco.MjData,
    first: int, second: int, threshold: float,
) -> float | None:
    """Return a measured distance <= threshold, or None outside that range.

    MuJoCo returns distmax when no distance below distmax was found. Query
    beyond the decision threshold so that sentinel cannot imply proximity.
    Proximity alone does not certify a contact mode or a usable acquisition.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("distance threshold must be finite and nonnegative")
    for geom in (first, second):
        if not 0 <= geom < model.ngeom:
            raise ValueError("invalid collision geometry id")
    limit = threshold + max(1e-6, abs(threshold) * 1e-6)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, limit, None))
    return distance if distance <= threshold else None


def contact_pairs(
    data: mujoco.MjData, first: Sequence[int], second: Sequence[int],
) -> tuple[tuple[int, int, float], ...]:
    """Actual contact pairs; callers must also verify their semantic contact mode."""
    a, b = set(first), set(second)
    return tuple(
        (int(c.geom1), int(c.geom2), float(c.dist))
        for c in data.contact
        if c.dist <= 0 and ((c.geom1 in a and c.geom2 in b) or (c.geom2 in a and c.geom1 in b))
    )
