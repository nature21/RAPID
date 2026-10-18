"""Shared data contracts for concrete primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np


AUTO_MAX_PARALLEL_ROLLOUTS = 48
_SCHEDULER_CPU_ENV = (
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "NSLOTS",
)


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting common job limits."""

    candidates: list[int] = []
    detected = os.cpu_count()
    if detected is not None and detected > 0:
        candidates.append(detected)

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = 0
        if affinity_count > 0:
            candidates.append(affinity_count)

    for name in _SCHEDULER_CPU_ENV:
        raw = os.environ.get(name, "").strip()
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            candidates.append(value)

    # Linux cgroup v2 quotas may be narrower than both os.cpu_count() and the
    # affinity mask (notably in containers and some batch-system setups).
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(
            encoding="utf-8"
        ).split()[:2]
        if quota != "max":
            quota_count = max(1, int(quota) // int(period))
            candidates.append(quota_count)
    except (OSError, ValueError, ZeroDivisionError):
        pass

    return max(1, min(candidates)) if candidates else 1


def resolve_parallel_rollouts(
    num_rollouts: int,
    max_parallel_rollouts: int = 0,
    *,
    available_cpus: int | None = None,
) -> int:
    """Resolve a safe CPU worker count for one optimizer population.

    An explicit positive cap is honored. Zero selects an affinity, scheduler,
    and cgroup-aware automatic limit capped at 48 workers. In automatic mode,
    a nearby divisor of ``num_rollouts`` is preferred so the common population
    does not pay for padded final chunks (128 rollouts therefore use 32 workers
    on a large server, while 192 rollouts use 48).
    """

    if num_rollouts < 1:
        raise ValueError("num_rollouts must be positive")
    if max_parallel_rollouts < 0:
        raise ValueError("max_parallel_rollouts cannot be negative")
    if max_parallel_rollouts:
        return min(max_parallel_rollouts, num_rollouts)

    cpu_count = available_cpu_count() if available_cpus is None else available_cpus
    if cpu_count < 1:
        raise ValueError("available_cpus must be positive")
    limit = min(num_rollouts, cpu_count, AUTO_MAX_PARALLEL_ROLLOUTS)
    lower = max(1, (limit + 1) // 2)
    for candidate in range(limit, lower - 1, -1):
        if num_rollouts % candidate == 0:
            return candidate
    return limit


@dataclass
class ToolTrajectory:
    """Optimized tool trajectory produced by the compile driver.

    Attributes:
        times: Simulation timestamps, shape ``(T,)``.
        positions: Interpolated optimizer actions, shape ``(T, action_dim)``.
            For direct-control scenes ``action_dim == model.nu``. Controller-
            aware scenes may instead use a compact action such as the FR3's
            seven Cartesian gripper channels.
        controls: Optimizer actions applied during the rollout, shape
            ``(T, action_dim)``. A low-level rollout controller may translate
            them into a different number of physical actuator commands.
        gripper_operations: Optional explicit per-sample gripper operation
            stream. FR3 trajectories contain one value per sample.
    """

    times: np.ndarray
    positions: np.ndarray
    controls: np.ndarray
    gripper_operations: np.ndarray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        positions = np.asarray(self.positions, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        if positions.ndim != 2 or controls.ndim != 2:
            raise ValueError("trajectory positions and controls must be 2-D")
        if positions.shape[0] != times.size or controls.shape[0] != times.size:
            raise ValueError("trajectory arrays must have the same sample count")
        operations = self.gripper_operations
        if operations is not None:
            operations = np.asarray(operations, dtype=np.uint8).reshape(-1)
            if operations.shape != times.shape:
                raise ValueError(
                    "gripper_operations must contain one value per sample"
                )
            if np.any(operations > 1):
                raise ValueError("unknown gripper operation")
            operations = np.ascontiguousarray(operations)
        self.times = np.ascontiguousarray(times)
        self.positions = np.ascontiguousarray(positions)
        self.controls = np.ascontiguousarray(controls)
        self.gripper_operations = operations


@dataclass
class CompileResult:
    """Full result of a compile call.

    Attributes:
        trajectory: Best control trajectory.
        metrics: Primitive-specific scalar metrics (e.g. ``success``,
            ``best_cost``, final error).
        best_states: Rolled-out state trajectory for the best candidate,
            shape ``(T, nq + nv)``.
        cost_history: Best cost per optimiser iteration, shape
            ``(max_iters,)``.
        initial_pose: Optimiser-chosen initial pose of the tool(s), keyed by
            names chosen by the learned primitive. Its frozen learned
            ``InstantiationSpec.apply_initial_pose`` implementation copies
            these values back into ``data.qpos`` for playback.
        controller_state: Optional low-level state required to continue a
            controller-aware rollout without a phase-boundary reset. For the
            native FR3 OSC this contains the seven previous commanded torques
            used by its torque-rate limiter.
    """

    trajectory: ToolTrajectory
    metrics: dict
    best_states: np.ndarray
    cost_history: np.ndarray
    initial_pose: dict[str, Any] = field(default_factory=dict)
    controller_state: np.ndarray | None = None


@dataclass(frozen=True)
class OptimizationBudget:
    """Compute allowance for one primitive optimization."""

    num_rollouts: int
    max_iters: int

    def __post_init__(self) -> None:
        for name in ("num_rollouts", "max_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def contains(self, other: "OptimizationBudget") -> bool:
        """Return whether both dimensions of ``other`` fit this budget."""

        if not isinstance(other, OptimizationBudget):
            raise TypeError("other must be an OptimizationBudget")
        return (
            other.num_rollouts <= self.num_rollouts
            and other.max_iters <= self.max_iters
        )


GLOBAL_MAX_OPTIMIZATION_BUDGET = OptimizationBudget(
    num_rollouts=96,
    max_iters=16,
)
MINIMUM_PREDICATE_GATED_EARLY_STOP_ITERS = 4


@dataclass(frozen=True)
class OptimizationPolicy:
    """Frozen primitive-owned search behavior and effect-timing policy.

    The learned primitive chooses these algorithmic values once during
    invention. Compute allowance is represented separately by
    :class:`OptimizationBudget`.
    """

    num_nodes: int
    elite_fraction: float
    horizon: float
    min_iters: int = 1
    early_stop_patience: int = 0
    early_stop_cost_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        for name in ("num_nodes", "min_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.early_stop_patience, bool) or not isinstance(
            self.early_stop_patience,
            Integral,
        ):
            raise TypeError("early_stop_patience must be an integer")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience cannot be negative")
        if (
            self.early_stop_patience > 0
            and self.min_iters < MINIMUM_PREDICATE_GATED_EARLY_STOP_ITERS
        ):
            raise ValueError(
                "predicate-gated early stopping requires min_iters >= "
                f"{MINIMUM_PREDICATE_GATED_EARLY_STOP_ITERS}"
            )
        for name in (
            "elite_fraction",
            "horizon",
            "early_stop_cost_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
        if (
            not np.isfinite(self.elite_fraction)
            or self.elite_fraction <= 0.0
            or self.elite_fraction > 1.0
        ):
            raise ValueError("elite_fraction must be finite and in (0, 1]")
        if not np.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError("horizon must be positive and finite")
        if (
            not np.isfinite(self.early_stop_cost_tolerance)
            or self.early_stop_cost_tolerance < 0.0
        ):
            raise ValueError(
                "early_stop_cost_tolerance must be finite and nonnegative"
            )

    def num_elites(self, num_rollouts: int) -> int:
        """Resolve the learned elite ratio for a concrete population."""

        if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, Integral):
            raise TypeError("num_rollouts must be an integer")
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be positive")
        return max(1, min(num_rollouts, int(self.elite_fraction * num_rollouts + 0.5)))

    def compiler_overrides(self, *, num_rollouts: int) -> dict[str, int | float]:
        """Return fields consumed by :class:`PrimitiveCompilerConfig`."""

        return {
            "num_nodes": self.num_nodes,
            "num_elites": self.num_elites(num_rollouts),
            "horizon": self.horizon,
            "min_iters": self.min_iters,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_cost_tolerance": self.early_stop_cost_tolerance,
        }


@dataclass
class PrimitiveCompilerConfig:
    """Default hyper-parameters for the shared compile driver.

    Rollout population and iteration count may not exceed
    ``GLOBAL_MAX_OPTIMIZATION_BUDGET``.

    Two optimiser back-ends are supported:

    * ``optimizer="cem"`` - cross-entropy method with per-node sigma,
      elite refit, sigma clipped to ``[sigma_min, sigma_max]``.
    * ``optimizer="mppi"`` - model-predictive path integral with a
      fixed scalar sigma (mean of the primitive's per-node ``sigma_init``)
      and softmax weighting at temperature ``mppi_temperature``.

    ``rollout_backend="mujoco"`` uses MuJoCo's multithreaded CPU rollout.
    ``rollout_backend="mjwarp"`` uses Judo's MuJoCo Warp backend and batches
    all rollout worlds on ``rollout_device``. The latter requires the optional
    ``mujoco_warp`` package and an NVIDIA CUDA device.

    Controller-aware FR3 rollouts execute the low-level OSC inside the Judo C++
    extension so MuJoCo worker threads never re-enter Python.

    ``max_parallel_rollouts`` caps the number of CPU MuJoCo rollout threads
    created at once. Zero selects a scheduler/affinity-aware automatic value,
    capped at 48 and chosen to avoid padded population chunks when practical.
    It is ignored by the MJWarp backend, where all candidates are evaluated as
    one GPU batch.
    """

    num_rollouts: int = GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts
    max_parallel_rollouts: int = 0
    num_nodes: int = 8
    num_elites: int = 12
    max_iters: int = GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters
    sigma_init: float = 0.08
    sigma_min: float = 0.01
    sigma_max: float = 0.3
    horizon: float = 2.0
    min_iters: int = 1
    early_stop_patience: int = 0
    early_stop_cost_tolerance: float = 1e-4
    transit_seed: int = 42
    optimize_initial_pose: bool = True
    optimizer: str = "cem"
    mppi_temperature: float = 0.05
    rollout_backend: str = "mujoco"
    rollout_device: str = "cuda:0"
