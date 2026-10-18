"""Optimizer-facing contracts for native state-feedback rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np


MAXIMUM_PENETRATION_OBSERVATION = "maximum_penetration"


@dataclass(frozen=True)
class ContactPairTelemetry:
    """Count contacts between two symmetric groups of MuJoCo geom ids."""

    name: str
    geom_ids_a: tuple[int, ...]
    geom_ids_b: tuple[int, ...]

    def __post_init__(self) -> None:
        geom_ids_a = tuple(self.geom_ids_a)
        geom_ids_b = tuple(self.geom_ids_b)
        if not self.name:
            raise ValueError("contact telemetry name cannot be empty")
        if not geom_ids_a or not geom_ids_b:
            raise ValueError("contact telemetry geom groups cannot be empty")
        if any(
            not isinstance(value, Integral) or value < 0 for value in geom_ids_a
        ):
            raise ValueError("contact telemetry geom_ids_a must be nonnegative ints")
        if any(
            not isinstance(value, Integral) or value < 0 for value in geom_ids_b
        ):
            raise ValueError("contact telemetry geom_ids_b must be nonnegative ints")
        object.__setattr__(self, "geom_ids_a", tuple(map(int, geom_ids_a)))
        object.__setattr__(self, "geom_ids_b", tuple(map(int, geom_ids_b)))


@dataclass(frozen=True)
class RolloutTelemetrySpec:
    """Opt-in native features appended ahead of controller observations.

    Each contact-pair channel contains the number of matching MuJoCo contact
    records at one returned post-step state. ``maximum_penetration`` is the
    largest positive penetration depth across all contacts at that state.
    """

    contact_pairs: tuple[ContactPairTelemetry, ...] = ()
    include_maximum_penetration: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "contact_pairs", tuple(self.contact_pairs))
        names = tuple(pair.name for pair in self.contact_pairs)
        if len(set(names)) != len(names):
            raise ValueError("rollout telemetry names must be unique")
        if (
            self.include_maximum_penetration
            and MAXIMUM_PENETRATION_OBSERVATION in names
        ):
            raise ValueError(
                f"{MAXIMUM_PENETRATION_OBSERVATION!r} is a reserved telemetry name"
            )
        if not self.contact_pairs and not self.include_maximum_penetration:
            raise ValueError(
                "rollout telemetry spec must request at least one feature"
            )

    @property
    def observation_names(self) -> tuple[str, ...]:
        names = tuple(pair.name for pair in self.contact_pairs)
        if self.include_maximum_penetration:
            names += (MAXIMUM_PENETRATION_OBSERVATION,)
        return names

    def observation_slice(self, model_sensor_dim: int) -> slice:
        if model_sensor_dim < 0:
            raise ValueError("model_sensor_dim cannot be negative")
        return slice(
            model_sensor_dim,
            model_sensor_dim + len(self.observation_names),
        )

    def view(self, sensors: np.ndarray, model_sensor_dim: int) -> np.ndarray:
        """Return this spec's channels from a native rollout sensor tensor."""

        telemetry_slice = self.observation_slice(model_sensor_dim)
        if sensors.shape[-1] < telemetry_slice.stop:
            raise ValueError("sensor tensor does not contain requested telemetry")
        return sensors[..., telemetry_slice]


@runtime_checkable
class RolloutControlAdapter(Protocol):
    """Describe compact optimizer actions consumed by a native controller."""

    action_dim: int
    action_ctrlrange: np.ndarray
    name: str
    observation_names: tuple[str, ...]


__all__ = [
    "ContactPairTelemetry",
    "MAXIMUM_PENETRATION_OBSERVATION",
    "RolloutControlAdapter",
    "RolloutTelemetrySpec",
]
