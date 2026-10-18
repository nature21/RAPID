"""Upstream-owned geometry and nominal-policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping, Protocol, runtime_checkable

import mujoco
import numpy as np

from instantiation_core.contracts import SceneBinding


class GripperIntent(str, Enum):
    """Whether gripper actuation is intrinsic to one learned effect."""

    HOLD = "hold"
    ACTUATE = "actuate"


class GripperOperation(IntEnum):
    """Explicit execution mode for one gripper command interval."""

    MOVE = 0
    GRASP = 1


@dataclass(frozen=True)
class BoundExecutionContext:
    """Read-only scene context shared by learned geometry and motion code."""

    model: mujoco.MjModel
    data: mujoco.MjData
    binding: SceneBinding
    initial_state: np.ndarray
    controller_state: np.ndarray | None = None
    initial_action: np.ndarray | None = None
    initial_gripper_operation: GripperOperation = GripperOperation.MOVE

    def __post_init__(self) -> None:
        state = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if state.shape != (self.model.nq + self.model.nv,):
            raise ValueError(
                "initial_state must contain model.nq + model.nv values"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be finite")
        object.__setattr__(self, "initial_state", state.copy())
        if self.controller_state is not None:
            controller = np.asarray(self.controller_state, dtype=float).reshape(-1)
            if not np.all(np.isfinite(controller)):
                raise ValueError("controller_state must be finite")
            object.__setattr__(self, "controller_state", controller.copy())
        if self.initial_action is not None:
            action = np.asarray(self.initial_action, dtype=float).reshape(-1)
            if action.size < 1 or not np.all(np.isfinite(action)):
                raise ValueError("initial_action must be a finite action vector")
            object.__setattr__(self, "initial_action", action.copy())
        try:
            operation = GripperOperation(self.initial_gripper_operation)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown initial gripper operation") from exc
        object.__setattr__(self, "initial_gripper_operation", operation)


@dataclass(frozen=True)
class NominalPlan:
    """Frozen, geometry-conditioned optimizer initialization."""

    knots: np.ndarray
    sigma: np.ndarray
    fixed_first_action: bool = False
    knot_fractions: np.ndarray | None = None
    gripper_intent: GripperIntent | None = None
    gripper_operations: np.ndarray | None = None
    constant_rotation_from_knot: int | None = None

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if knots.ndim != 2 or knots.shape[0] < 2 or knots.shape[1] < 1:
            raise ValueError(
                "nominal knots must be a 2-D array with at least two knots"
            )
        if sigma.ndim == 0:
            sigma = np.full_like(knots, float(sigma))
        if sigma.shape != knots.shape:
            raise ValueError("nominal sigma must be scalar or match knot shape")
        if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(sigma)):
            raise ValueError("nominal knots and sigma must be finite")
        if np.any(sigma < 0.0):
            raise ValueError("nominal sigma cannot be negative")
        if not isinstance(self.fixed_first_action, (bool, np.bool_)):
            raise TypeError("fixed_first_action must be boolean")
        gripper_intent = self.gripper_intent
        if gripper_intent is not None and not isinstance(
            gripper_intent, GripperIntent
        ):
            try:
                gripper_intent = GripperIntent(gripper_intent)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gripper_intent must be GripperIntent.HOLD or "
                    "GripperIntent.ACTUATE"
                ) from exc
        if (
            gripper_intent is GripperIntent.HOLD
            and knots.shape[1] >= 7
            and not np.allclose(knots[:, 6], knots[0, 6], rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "GripperIntent.HOLD requires a constant nominal gripper profile"
            )
        if self.gripper_operations is None:
            operations = None
        else:
            raw_operations = np.asarray(self.gripper_operations).reshape(-1)
            if raw_operations.shape != (knots.shape[0],):
                raise ValueError(
                    "gripper_operations must contain one operation per knot"
                )
            try:
                operations = np.asarray(
                    [GripperOperation(value) for value in raw_operations],
                    dtype=np.uint8,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gripper_operations entries must be GripperOperation.MOVE "
                    "or GripperOperation.GRASP"
                ) from exc
        if gripper_intent is not None and operations is None:
            raise ValueError(
                "a gripper operation is required for every nominal knot"
            )
        if operations is not None and gripper_intent is None:
            raise ValueError(
                "gripper_operations require an explicit gripper_intent"
            )
        if (
            gripper_intent is GripperIntent.HOLD
            and operations is not None
            and np.any(operations != int(GripperOperation.MOVE))
        ):
            raise ValueError("GripperIntent.HOLD permits only MOVE operations")
        if operations is not None and knots.shape[1] < 7:
            raise ValueError(
                "gripper_operations require a seven-channel nominal action"
            )
        if operations is not None:
            grasp_intervals = operations[:-1] == int(GripperOperation.GRASP)
            if np.any(np.diff(knots[:, 6])[grasp_intervals] > 1e-12):
                raise ValueError(
                    "GRASP may only accompany a non-increasing aperture"
                )
        if self.knot_fractions is None:
            fractions = np.linspace(0.0, 1.0, knots.shape[0], dtype=float)
        else:
            fractions = np.asarray(self.knot_fractions, dtype=float).reshape(-1)
        if fractions.shape != (knots.shape[0],):
            raise ValueError("knot_fractions must contain one value per knot")
        if not np.all(np.isfinite(fractions)):
            raise ValueError("knot_fractions must be finite")
        if not np.isclose(fractions[0], 0.0) or not np.isclose(
            fractions[-1], 1.0
        ):
            raise ValueError("knot_fractions must start at 0 and end at 1")
        if np.any(np.diff(fractions) <= 0.0):
            raise ValueError("knot_fractions must be strictly increasing")
        validate_constant_rotation(knots, self.constant_rotation_from_knot)
        object.__setattr__(self, "knots", np.ascontiguousarray(knots))
        object.__setattr__(self, "sigma", np.ascontiguousarray(sigma))
        object.__setattr__(self, "fixed_first_action", bool(self.fixed_first_action))
        object.__setattr__(self, "gripper_intent", gripper_intent)
        object.__setattr__(
            self,
            "gripper_operations",
            None if operations is None else np.ascontiguousarray(operations),
        )
        object.__setattr__(
            self,
            "knot_fractions",
            np.ascontiguousarray(fractions),
        )


def validate_constant_rotation(knots: np.ndarray, start: int | None) -> None:
    """Validate an optional shared Cartesian attitude over a knot suffix."""
    if start is None:
        return
    if (
        isinstance(start, (bool, np.bool_))
        or not isinstance(start, (int, np.integer))
        or knots.ndim != 2
        or knots.shape[1] != 7
        or not 1 <= start < len(knots)
    ):
        raise ValueError(
            "constant_rotation_from_knot requires seven channels and a "
            "noninitial knot index"
        )
    if not np.allclose(knots[start:, 3:6], knots[start, 3:6], rtol=0, atol=1e-12):
        raise ValueError("shared rotation requires equal nominal suffix attitudes")


@runtime_checkable
class GeometryResolver(Protocol):
    """Scene-agnostic learned feature resolver."""

    def __call__(self, context: BoundExecutionContext) -> Mapping[str, Any]: ...


@runtime_checkable
class NominalPolicy(Protocol):
    """Learned policy from relational features to effect knots."""

    def __call__(
        self,
        context: BoundExecutionContext,
        features: Mapping[str, Any],
    ) -> NominalPlan: ...


__all__ = [
    "BoundExecutionContext",
    "GeometryResolver",
    "GripperIntent",
    "GripperOperation",
    "NominalPlan",
    "NominalPolicy",
]
