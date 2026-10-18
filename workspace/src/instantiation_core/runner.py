"""Shared deterministic runner for primitive scene instantiations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from primitive_core.common import (
    CompileResult,
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    OptimizationBudget,
    OptimizationPolicy,
    PrimitiveCompilerConfig,
)
from primitive_core.fr3_osc import FR3OSCAdapter
from primitive_core.result_retention import capture_execution_results
from scene_io.playback import PlaybackBundle

from .contracts import BindingReport
from .json_values import to_json_value, write_json_file


FR3_GOAL_SCHEMA = "rapid.fr3_cartesian_goals"
FR3_GOAL_SCHEMA_VERSION = 2
FR3_STANDARD_HAND_F_T_EE = {
    "position": [0.0, 0.0, 0.1034],
    "orientation": [0.0, 0.0, 0.0, 1.0],
}


CompileCallable = Callable[[Any], CompileResult]
SceneBuilder = Callable[[], tuple[mujoco.MjModel, mujoco.MjData]]
InitialPoseApplier = Callable[[mujoco.MjData, dict[str, Any]], None]


@dataclass
class InstantiationSpec:
    """Everything needed to execute one deterministic primitive instantiation."""

    label: str
    compile_fn: CompileCallable
    playback: PlaybackBundle
    build_scene_data: SceneBuilder
    apply_initial_pose: InitialPoseApplier
    config_cls: type = PrimitiveCompilerConfig
    optimization_policy: OptimizationPolicy | None = None
    reference_optimization_budget: OptimizationBudget | None = None
    banner_lines: Sequence[str] = field(default_factory=tuple)
    binding_report: BindingReport | None = None


def make_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser with standard instantiation execution flags."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--replay", choices=["browser", "mujoco"], default=None)
    parser.add_argument("--record", type=str, default=None, metavar="PATH")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable the canonical default trajectory recording.",
    )
    parser.add_argument(
        "--cartesian-goals-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write commanded FR3 end-effector goals for play_cartesian_goals.py. "
            "Defaults next to --metrics-json."
        ),
    )
    parser.add_argument("--optimizer", choices=["cem", "mppi"], default="cem")
    parser.add_argument("--mppi-temperature", type=float, default=0.05)
    parser.add_argument(
        "--rollout-backend",
        choices=["mujoco", "mjwarp"],
        default="mujoco",
        help="Simulation backend for optimizer rollouts (default: mujoco).",
    )
    parser.add_argument(
        "--rollout-device",
        default="cuda:0",
        help="Warp device used by --rollout-backend=mjwarp (default: cuda:0).",
    )
    parser.add_argument(
        "--num-rollouts",
        type=_positive_int,
        default=None,
        help="Cap the selected rollout count for smoke/resource limits.",
    )
    parser.add_argument(
        "--max-parallel-rollouts",
        type=_positive_int,
        default=None,
        help=(
            "CPU rollout worker cap. Omit for scheduler/affinity-aware auto "
            "sizing (maximum 48)."
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=_positive_int,
        default=None,
        help="Diagnostic setting; learned policies keep their frozen knot count.",
    )
    parser.add_argument(
        "--max-iters",
        type=_positive_int,
        default=None,
        help="Cap the selected optimizer iterations.",
    )
    parser.add_argument(
        "--horizon",
        type=_positive_float,
        default=None,
        help="Upper-bound learned horizon; FR3 minimum safe timing still applies.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transit-seed",
        type=_nonnegative_int,
        default=42,
        help="Runtime-owned seed for transit planning (default: 42).",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write binding status, metrics, and initial pose as JSON.",
    )
    parser.add_argument(
        "--state-trajectory-npz",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Retain the achieved nq+nv state trajectory for benchmark-owned "
            "physical evaluation."
        ),
    )
    parser.add_argument(
        "--validate-binding",
        action="store_true",
        help="Validate and print the binding before running the program.",
    )
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def cartesian_goals_path_for_metrics(metrics_path: str | Path) -> Path:
    """Return the paired real-arm Cartesian-goal path for retained metrics."""

    metrics = Path(metrics_path)
    stem = metrics.stem
    if "metrics" in stem:
        stem = stem.replace("metrics", "cartesian_goals", 1)
    else:
        stem = f"{stem}_cartesian_goals"
    return metrics.with_name(f"{stem}.json")


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two unit quaternions along the shortest rotation."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = float(left @ right)
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = (1.0 - fraction) * left + fraction * right
        return value / np.linalg.norm(value)
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return (
        np.sin((1.0 - fraction) * angle) / denominator * left
        + np.sin(fraction * angle) / denominator * right
    )


def write_cartesian_goal_trajectory(
    path: str | Path,
    result: CompileResult,
    adapter: FR3OSCAdapter,
    *,
    publish_rate_hz: float = 50.0,
) -> int:
    """Write one strict version-2 FR3 Cartesian-goal document."""

    controls = np.asarray(result.trajectory.controls, dtype=float)
    times = np.asarray(result.trajectory.times, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != adapter.action_dim:
        raise ValueError(
            "FR3 Cartesian-goal export requires controls with shape "
            f"(T, {adapter.action_dim}); got {controls.shape}"
        )
    if times.shape != (controls.shape[0],):
        raise ValueError("trajectory times must provide one value per control")
    if controls.shape[0] == 0 or not np.all(np.isfinite(controls)):
        raise ValueError("trajectory controls must be non-empty and finite")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory times must be finite and strictly increasing")
    if not np.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
        raise ValueError("publish_rate_hz must be finite and positive")
    operations = result.trajectory.gripper_operations
    if operations is None:
        raise ValueError("FR3 Cartesian-goal export requires explicit operations")
    operations = np.asarray(operations, dtype=np.uint8).reshape(-1)
    if operations.shape != times.shape or np.any(operations > 1):
        raise ValueError("invalid FR3 gripper operation stream")
    grasp_intervals = operations[:-1] == 1
    if np.any(np.diff(controls[:, 6])[grasp_intervals] > 1e-12):
        raise ValueError("GRASP may only accompany a non-increasing aperture")

    positions = np.empty((controls.shape[0], 3), dtype=float)
    quaternions_wxyz = np.empty((controls.shape[0], 4), dtype=float)
    for index, action in enumerate(controls):
        position, quaternion = adapter.target_pose_world(action)
        positions[index] = position
        quaternions_wxyz[index] = quaternion
    total_width_grippers = np.clip(2.0 * controls[:, 6], 0.0, 0.08)

    base_position_world = np.asarray(adapter.frame_position_world, dtype=float)
    base_rotation_world = adapter.frame_rotation_world
    inverse_base_quaternion = np.asarray(adapter.frame_quaternion_world) * np.asarray(
        (1.0, -1.0, -1.0, -1.0)
    )
    positions_base = np.empty_like(positions)
    quaternions_base_wxyz = np.empty_like(quaternions_wxyz)
    for index in range(positions.shape[0]):
        positions_base[index] = base_rotation_world.T @ (
            positions[index] - base_position_world
        )
        mujoco.mju_mulQuat(
            quaternions_base_wxyz[index],
            inverse_base_quaternion,
            quaternions_wxyz[index],
        )
        quaternions_base_wxyz[index] /= np.linalg.norm(
            quaternions_base_wxyz[index]
        )

    interval = 1.0 / float(publish_rate_hz)
    sample_times = np.arange(times[0], times[-1] + 1e-12, interval)
    if sample_times.size == 0:
        sample_times = times[:1].copy()

    goals: list[dict[str, list[float] | float | str]] = []
    for sample_time in sample_times:
        right = int(np.searchsorted(times, sample_time, side="right"))
        if right == 0:
            left = right = 0
            fraction = 0.0
        elif right >= times.shape[0]:
            left = right = times.shape[0] - 1
            fraction = 0.0
        else:
            left = right - 1
            fraction = float(
                (sample_time - times[left]) / (times[right] - times[left])
            )
        position = (
            (1.0 - fraction) * positions_base[left]
            + fraction * positions_base[right]
        )
        quaternion_wxyz = _slerp(
            quaternions_base_wxyz[left],
            quaternions_base_wxyz[right],
            fraction,
        )
        operation_index = max(0, min(left, operations.size - 1))
        goals.append(
            {
                "position_m": [float(value) for value in position],
                "orientation_xyzw": [
                    float(quaternion_wxyz[1]),
                    float(quaternion_wxyz[2]),
                    float(quaternion_wxyz[3]),
                    float(quaternion_wxyz[0]),
                ],
                "gripper_width_m": float(
                    (1.0 - fraction) * total_width_grippers[left]
                    + fraction * total_width_grippers[right]
                ),
                "gripper_operation": (
                    "move" if int(operations[operation_index]) == 0 else "grasp"
                ),
            }
        )

    config = adapter.config
    payload = {
        "schema": FR3_GOAL_SCHEMA,
        "version": FR3_GOAL_SCHEMA_VERSION,
        "sample_rate_hz": float(publish_rate_hz),
        "time_semantics": "zero_order_hold",
        "pose_frame": "fr3_link0",
        "controlled_frame": "fr3_hand_tcp",
        "quaternion_order": "xyzw",
        "expected_F_T_EE": FR3_STANDARD_HAND_F_T_EE,
        "initial_joint_configuration": [
            float(value) for value in adapter.nullspace_configuration
        ],
        "source_scene_base_pose_world": {
            "position": [float(value) for value in adapter.frame_position_world],
            "orientation_wxyz": [
                float(value) for value in adapter.frame_quaternion_world
            ],
        },
        "controller": {
            "profile": config.profile,
            "translational_stiffness": list(config.translational_stiffness),
            "rotational_stiffness": list(config.rotational_stiffness),
            "translational_damping": list(config.translational_damping),
            "rotational_damping": list(config.rotational_damping),
            "nullspace_stiffness": config.nullspace_stiffness,
            "nullspace_damping": config.nullspace_damping,
            "equilibrium_filter_alpha": config.equilibrium_filter_alpha,
            "controller_rate_hz": config.controller_rate_hz,
            "torque_rate_limit_nm_per_s": config.maximum_torque_rate,
            "libfranka_torque_lowpass_hz": config.torque_lowpass_cutoff_hz,
            "libfranka_rate_limiting": config.libfranka_rate_limiting,
        },
        "gripper": {
            "width_semantic": "total",
            "unit": "m",
            "range_m": [0.0, 0.08],
            "move_speed_m_s": 2.0 * config.gripper_move_speed,
            "grasp_force_n": config.gripper_grasp_force,
            "maximum_force_n": config.gripper_maximum_force,
            "grasp_epsilon_inner_m": 0.005,
            "grasp_epsilon_outer_policy": "remaining_physical_opening",
        },
        "goals": goals,
    }

    output = Path(path)
    write_json_file(output, payload, sort_keys=False)
    print(
        f"Cartesian goals saved to {output} "
        f"({len(goals)} samples at {publish_rate_hz:g} Hz in fr3_link0)"
    )
    return len(goals)


def _env_override(env_name: str, caster: Callable[[str], Any]) -> Any | None:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return None
    try:
        return caster(raw)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        raise ValueError(f"{env_name}={raw!r} is invalid: {exc}") from exc


def _config_field_names(config_cls: type) -> set[str]:
    if not is_dataclass(config_cls):
        return set()
    return {item.name for item in fields(config_cls)}


_OPTIMIZER_ENV_SPECS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "num_rollouts": ("RAPID_OPTIMIZER_NUM_ROLLOUTS", _positive_int),
    "max_parallel_rollouts": ("RAPID_MAX_PARALLEL_ROLLOUTS", _positive_int),
    "num_nodes": ("RAPID_OPTIMIZER_NUM_NODES", _positive_int),
    "max_iters": ("RAPID_OPTIMIZER_MAX_ITERS", _positive_int),
    "horizon": ("RAPID_OPTIMIZER_HORIZON", _positive_float),
}

_OPTIMIZATION_BUDGET_MODE_ENV = "RAPID_OPTIMIZATION_BUDGET_MODE"


def _optimizer_argument_names(
    args: argparse.Namespace,
    resource: str,
) -> list[str]:
    """Return generic and phase-specific argument names for one resource."""

    names = vars(args)
    if resource == "max_iters":
        return [
            name
            for name in names
            if name == resource or name.endswith("_iters")
        ]
    return [
        name
        for name in names
        if name == resource or name.endswith(f"_{resource}")
    ]


def apply_optimizer_env_overrides(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Apply launcher resource caps to generic and multiphase CLI arguments.

    Environment values are launcher-enforced upper bounds, not tuning
    defaults. They fill an unset value and reduce a larger explicit/default
    value, but never enlarge a deliberately small smoke run. Multiphase
    strategy options such as ``reorient_iters`` and ``push_horizon`` receive
    the same corresponding cap.
    """

    for resource, (env_name, caster) in _OPTIMIZER_ENV_SPECS.items():
        value = _env_override(env_name, caster)
        if value is None:
            continue
        for argument_name in _optimizer_argument_names(args, resource):
            current = getattr(args, argument_name)
            setattr(
                args,
                argument_name,
                value if current is None else min(current, value),
            )
    return args


def build_config(
    config_cls: type,
    args: argparse.Namespace,
    *,
    learned_policy: OptimizationPolicy | None = None,
    reference_budget: OptimizationBudget | None = None,
    requested_budget: OptimizationBudget | None = None,
    **overrides: Any,
) -> Any:
    """Resolve learned behavior and compute allowance into a compiler config."""

    rollout_backend = (
        os.environ.get("RAPID_ROLLOUT_BACKEND") or args.rollout_backend
    )
    if rollout_backend not in {"mujoco", "mjwarp"}:
        raise ValueError(
            f"RAPID_ROLLOUT_BACKEND={rollout_backend!r} is invalid; "
            "expected 'mujoco' or 'mjwarp'"
        )
    rollout_device = os.environ.get("RAPID_ROLLOUT_DEVICE") or args.rollout_device
    kwargs: dict[str, Any] = {
        "optimizer": args.optimizer,
        "mppi_temperature": args.mppi_temperature,
        "rollout_backend": rollout_backend,
        "rollout_device": rollout_device,
    }
    field_names = _config_field_names(config_cls)
    if not field_names or "transit_seed" in field_names:
        kwargs["transit_seed"] = args.transit_seed
    if learned_policy is None:
        if reference_budget is not None or requested_budget is not None:
            raise ValueError("optimization budgets require a learned policy")
    else:
        if reference_budget is None:
            raise ValueError("a learned policy requires a reference budget")
        if not isinstance(reference_budget, OptimizationBudget):
            raise TypeError("reference_budget must be an OptimizationBudget")
        if not GLOBAL_MAX_OPTIMIZATION_BUDGET.contains(reference_budget):
            raise ValueError(
                "reference budget exceeds GLOBAL_MAX_OPTIMIZATION_BUDGET"
            )
        if learned_policy.min_iters > reference_budget.max_iters:
            raise ValueError("policy min_iters exceeds its reference budget")
        if requested_budget is not None:
            if not isinstance(requested_budget, OptimizationBudget):
                raise TypeError("requested_budget must be an OptimizationBudget")
            if not requested_budget.contains(reference_budget):
                raise ValueError(
                    "a strategy phase budget cannot be smaller than the "
                    "primitive reference budget"
                )
            if not GLOBAL_MAX_OPTIMIZATION_BUDGET.contains(requested_budget):
                raise ValueError(
                    "strategy phase budget exceeds "
                    "GLOBAL_MAX_OPTIMIZATION_BUDGET"
                )
        mode = os.environ.get(_OPTIMIZATION_BUDGET_MODE_ENV, "reference")
        if mode not in {"reference", "maximum"}:
            raise ValueError(
                f"{_OPTIMIZATION_BUDGET_MODE_ENV}={mode!r} is invalid; "
                "expected 'reference' or 'maximum'"
            )
        selected_budget = (
            GLOBAL_MAX_OPTIMIZATION_BUDGET
            if mode == "maximum"
            else requested_budget or reference_budget
        )
        num_rollouts = selected_budget.num_rollouts
        max_iters = selected_budget.max_iters
        requested_rollouts = getattr(args, "num_rollouts", None)
        requested_iters = getattr(args, "max_iters", None)
        if requested_rollouts is not None:
            num_rollouts = min(num_rollouts, requested_rollouts)
        if requested_iters is not None:
            max_iters = min(max_iters, requested_iters)
        rollout_cap = _env_override(
            _OPTIMIZER_ENV_SPECS["num_rollouts"][0],
            _OPTIMIZER_ENV_SPECS["num_rollouts"][1],
        )
        iteration_cap = _env_override(
            _OPTIMIZER_ENV_SPECS["max_iters"][0],
            _OPTIMIZER_ENV_SPECS["max_iters"][1],
        )
        if rollout_cap is not None:
            num_rollouts = min(num_rollouts, rollout_cap)
        if iteration_cap is not None:
            max_iters = min(max_iters, iteration_cap)
        forbidden = {"num_rollouts", "num_elites", "max_iters"} & overrides.keys()
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(
                f"learned optimization budgets must use requested_budget, not {names}"
            )
        kwargs.update(
            learned_policy.compiler_overrides(num_rollouts=num_rollouts)
        )
        kwargs["num_rollouts"] = num_rollouts
        kwargs["max_iters"] = max_iters
        kwargs["min_iters"] = min(learned_policy.min_iters, max_iters)
    if learned_policy is not None:
        learned_nodes = learned_policy.num_nodes
        requested_nodes = getattr(args, "num_nodes", None)
        environment_nodes = _env_override(
            _OPTIMIZER_ENV_SPECS["num_nodes"][0],
            _OPTIMIZER_ENV_SPECS["num_nodes"][1],
        )
        override_nodes = overrides.get("num_nodes")
        for source, value in (
            ("CLI", requested_nodes),
            ("environment", environment_nodes),
            ("programmatic", override_nodes),
        ):
            if value is not None and value != learned_nodes:
                raise ValueError(
                    f"{source} num_nodes={value} cannot change frozen learned "
                    f"knot topology ({learned_nodes})"
                )
    for field_name, (env_name, caster) in _OPTIMIZER_ENV_SPECS.items():
        if field_names and field_name not in field_names:
            continue
        requested = getattr(args, field_name, None)
        learned = kwargs.get(field_name)
        if learned_policy is not None and field_name == "num_nodes":
            requested = None
        value = (
            requested
            if learned is None
            else learned
            if requested is None
            else min(learned, requested)
        )
        cap = _env_override(env_name, caster)
        if learned_policy is not None and field_name == "num_nodes":
            cap = None
        if cap is not None:
            value = cap if value is None else min(value, cap)
        if value is not None:
            kwargs[field_name] = value
    policy_values = (
        learned_policy.compiler_overrides(
            num_rollouts=int(kwargs["num_rollouts"]),
        )
        if learned_policy is not None
        else {}
    )
    frozen_overrides = policy_values.keys() & overrides.keys()
    if frozen_overrides:
        names = ", ".join(sorted(frozen_overrides))
        raise ValueError(
            f"learned optimization policy fields cannot be overridden: {names}"
        )
    for name, value in overrides.items():
        if name in _OPTIMIZER_ENV_SPECS and name in kwargs:
            kwargs[name] = min(kwargs[name], value)
        else:
            kwargs[name] = value
    # Programmatic phase overrides are subject to the same launcher caps as
    # parser values. This prevents generated multiphase code from accidentally
    # bypassing a bounded smoke or suite run.
    for field_name, (env_name, caster) in _OPTIMIZER_ENV_SPECS.items():
        if field_name not in kwargs:
            continue
        cap = _env_override(env_name, caster)
        if learned_policy is not None and field_name == "num_nodes":
            cap = None
        if cap is not None:
            current = kwargs[field_name]
            kwargs[field_name] = cap if current is None else min(current, cap)
    if "num_elites" in kwargs and "num_rollouts" in kwargs:
        kwargs["num_elites"] = min(kwargs["num_elites"], kwargs["num_rollouts"])
    if "min_iters" in kwargs and "max_iters" in kwargs:
        kwargs["min_iters"] = min(kwargs["min_iters"], kwargs["max_iters"])
    effective_rollouts = kwargs.get(
        "num_rollouts",
        GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts,
    )
    effective_iters = kwargs.get(
        "max_iters",
        GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
    )
    if (
        effective_rollouts > GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts
        or effective_iters > GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters
    ):
        raise ValueError("optimizer compute exceeds GLOBAL_MAX_OPTIMIZATION_BUDGET")
    return config_cls(**kwargs)


def _banner(title: str, lines: Sequence[str]) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    for line in lines:
        print(f"  {line}")
    print(f"{bar}\n")


def _print_binding(report: BindingReport) -> None:
    print("--- Binding ---")
    print(f"  binding_valid: {report.binding_valid}")
    print(f"  program_success: {report.program_success}")
    for issue in report.issues:
        suffix = f" [{issue.role}]" if issue.role else ""
        print(f"  {issue.code}{suffix}: {issue.message}")
    print()


def _print_execution_diagnostics(label: str, metrics: Mapping[str, Any]) -> None:
    """Expose retained phase outcomes and transit errors without changing metrics."""

    def flag(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value)).lower()
        return "unavailable"

    def quantity(value: Any, scale: float, unit: str) -> str:
        if (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        ):
            return f"{float(value) * scale:.2f} {unit}"
        return "unavailable"

    def transit_summary(name: str, transit: Mapping[str, Any]) -> None:
        projection = transit.get("initial_knot_advisory")
        if isinstance(projection, Mapping):
            print(
                f"  {name} / handoff projection: "
                f"translation={quantity(projection.get('translation'), 1000, 'mm')}; "
                f"rotation={quantity(projection.get('orientation'), 180 / np.pi, 'deg')}"
            )
        phases = transit.get("transit_phases")
        if isinstance(phases, (list, tuple)) and any(
            isinstance(p, Mapping) for p in phases
        ):
            for index, phase in enumerate(phases):
                if isinstance(phase, Mapping):
                    transit_summary(f"{name} / {phase.get('phase', index)}", phase)
            return
        position_error = quantity(transit.get("terminal_position_error"), 1000, "mm")
        orientation_error = quantity(
            transit.get("terminal_orientation_error"), 180 / np.pi, "deg"
        )
        print(
            f"  {name}: "
            f"convergence={flag(transit.get('transit_convergence_valid'))}; "
            f"terminal_position_error={position_error}; "
            f"terminal_orientation_error={orientation_error}"
        )

    def phase_summary(name: str, phase: Mapping[str, Any]) -> None:
        transit = phase.get("transit")
        if isinstance(transit, Mapping):
            transit_summary(f"{name} / transit", transit)
        elif any(
            key in phase for key in (
                "transit_phases", "transit_convergence_valid",
                "terminal_position_error", "initial_knot_advisory",
            )
        ):
            transit_summary(f"{name} / transit", phase)
        if any(key in phase for key in ("effect_success", "mechanism_valid")):
            print(
                f"  {name} / effect: "
                f"effect_success={flag(phase.get('effect_success'))}; "
                f"mechanism_valid={flag(phase.get('mechanism_valid'))}; "
                f"stability_valid={flag(phase.get('stability_valid'))}"
            )
        if phase.get("failure_type"):
            print(
                f"  {name} / failure: {phase['failure_type']}: "
                f"{phase.get('failure_message') or 'unavailable'}"
            )
        phases = phase.get("phase_metrics")
        if isinstance(phases, Mapping):
            for child_name, child in phases.items():
                if isinstance(child, Mapping):
                    phase_summary(f"{name} / {child_name}", child)

    print("\n--- Execution diagnostics ---")
    phase_summary(label, metrics)


def _print_metrics(label: str, result: CompileResult) -> None:
    from reprlib import repr as compact_repr

    print(f"\n--- Metrics for {label!r} ---")
    for key, value in result.metrics.items():
        print(f"  {key}: {compact_repr(value)}")
    _print_execution_diagnostics(label, result.metrics)
    if result.initial_pose:
        print("  initial_pose:")
        for key, value in result.initial_pose.items():
            print(f"    {key}: {value}")
    print()


def _write_metrics_json(
    path: str,
    label: str,
    result: CompileResult,
    report: BindingReport | None,
) -> None:
    output = Path(path)
    binding_valid = report.binding_valid if report is not None else None
    program_success = report.program_success if report is not None else None
    payload = {
        "label": label,
        "binding_valid": binding_valid,
        "program_success": program_success,
        "binding": report.to_dict() if report is not None else None,
        "metrics": result.metrics,
        "initial_pose": result.initial_pose,
    }
    from instantiation_core.metric_artifacts import externalize_metric_arrays
    write_json_file(output, externalize_metric_arrays(payload, output))
    print(f"Metrics saved to {output}")


def write_state_trajectory(
    path: str | Path,
    result: CompileResult,
) -> None:
    """Atomically retain one finite achieved state trajectory as NPZ."""

    states = np.asarray(result.best_states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("state trajectory must be a nonempty 2-D array")
    if not np.all(np.isfinite(states)):
        raise ValueError("state trajectory contains non-finite values")
    times = np.asarray(result.trajectory.times, dtype=float).reshape(-1)
    if times.shape != (states.shape[0],) or not np.all(np.isfinite(times)):
        raise ValueError("state trajectory times do not match retained states")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray("rapid.state_trajectory"),
                schema_version=np.asarray(1, dtype=np.int64),
                states=np.ascontiguousarray(states),
                times=np.ascontiguousarray(times),
            )
        temporary.replace(output)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
    print(f"State trajectory saved to {output}")


def _failure_signature(metrics: Mapping[str, Any]) -> str:
    """Return a stable, coarse signature for invention-loop triage."""

    if bool(metrics.get("success", False)):
        return "success"
    failed_segment = metrics.get("failed_segment")
    if isinstance(failed_segment, str) and failed_segment:
        failure_type = str(metrics.get("failure_type", ""))
        failure_message = str(metrics.get("failure_message", "")).lower()
        if (
            "transit" in failure_type.lower()
            or "planning" in failure_type.lower()
        ):
            if (
                "release" in failure_message
                or "inherited-contact" in failure_message
            ):
                category = "transit:release"
            elif "handoff" in failure_message or "projection" in failure_message:
                category = "transit:handoff"
            elif "planning" in failure_type.lower():
                category = "transit:planning"
            else:
                category = "transit:execution"
        else:
            category = failure_type or "program:unspecified_failure"
        return f"segment:{failed_segment}:{category}"
    phase_metrics = metrics.get("phase_metrics")
    if isinstance(phase_metrics, Mapping):
        for phase_name, phase_values in phase_metrics.items():
            if not isinstance(phase_values, Mapping):
                continue
            phase_signature = _failure_signature(phase_values)
            if phase_signature not in {"success", "program:unspecified_failure"}:
                return f"phase:{phase_name}:{phase_signature}"
            if phase_values.get("success") is False:
                return f"phase:{phase_name}:program:unspecified_failure"
    if metrics.get("retained_timing_valid") is False:
        return "effect:retained_timing"
    if metrics.get("whole_arm_audit_valid") is False:
        if int(metrics.get("self_collision_steps", 0) or 0) > 0:
            return "effect:whole_arm:self_collision"
        pair = metrics.get("dominant_unapproved_contact_pair")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return "effect:whole_arm:unapproved_contact:" + "|".join(
                sorted(map(str, pair))
            )
        if int(metrics.get("unapproved_contact_steps", 0) or 0) > 0:
            return "effect:whole_arm:unapproved_contact"
        return "effect:whole_arm:clearance"
    if metrics.get("retained_gripper_contact_audit_valid") is False:
        return "effect:gripper_command_during_contact"
    for key in (
        "transit_valid",
        "retained_transit_valid",
        "controller_state_handoff_valid",
    ):
        if metrics.get(key) is False:
            return f"transit:{key}"
    if metrics.get("mechanism_valid") is False:
        return "mechanism:required_relations"
    if metrics.get("effect_success") is False:
        return "effect:predicate"
    if metrics.get("optimized_success") is False:
        return "effect:predicate"
    if bool(metrics.get("success", False)):
        return "success"
    return "program:unspecified_failure"


_MIN_MEANINGFUL_SEARCH_ROLLOUTS = 32
_MIN_MEANINGFUL_SEARCH_ITERS = 6
_STANDARD_SEARCH_ROLLOUTS = 64
_STANDARD_SEARCH_ITERS = 10
_EXPANDED_SEARCH_ROLLOUTS = GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts
_EXPANDED_SEARCH_ITERS = GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters


def _nested_numeric_values(value: Any, key: str) -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                values.extend(_nested_numeric_values(child, key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.extend(_nested_numeric_values(child, key))
    return values


def _attempt_budget(
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    *,
    failure_signature: str,
) -> dict[str, Any]:
    evaluated_metrics = metrics
    evaluated_phase: str | None = None
    if failure_signature.startswith("phase:"):
        _, evaluated_phase, _ = failure_signature.split(":", 2)
        phase_metrics = metrics.get("phase_metrics")
        if isinstance(phase_metrics, Mapping):
            phase_values = phase_metrics.get(evaluated_phase)
            if isinstance(phase_values, Mapping):
                evaluated_metrics = phase_values
    rollouts = _nested_numeric_values(evaluated_metrics, "num_rollouts")
    requested_iters = _nested_numeric_values(
        evaluated_metrics,
        "requested_max_iters",
    )
    if not rollouts:
        value = getattr(args, "num_rollouts", None)
        if isinstance(value, int):
            rollouts.append(float(value))
    if not requested_iters:
        values = [
            value
            for name, value in vars(args).items()
            if (name == "max_iters" or name.endswith("_iters"))
            and isinstance(value, int)
        ]
        requested_iters.extend(map(float, values))
    budget = {
        key: getattr(args, key)
        for key in (
            "num_rollouts",
            "max_iters",
            "horizon",
            "seed",
            "transit_seed",
        )
        if hasattr(args, key)
    }
    if evaluated_phase is not None:
        budget["evaluated_phase"] = evaluated_phase
    if rollouts:
        budget["effective_num_rollouts"] = int(max(rollouts))
    if requested_iters:
        budget["effective_max_iters"] = int(max(requested_iters))
    return budget


def _positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _search_budget_tier(budget: Mapping[str, Any]) -> str:
    rollouts = int(budget.get("effective_num_rollouts", 0) or 0)
    iterations = int(budget.get("effective_max_iters", 0) or 0)
    rollout_ceiling = min(
        _EXPANDED_SEARCH_ROLLOUTS,
        _positive_environment_int(
            "RAPID_OPTIMIZER_NUM_ROLLOUTS",
            _EXPANDED_SEARCH_ROLLOUTS,
        ),
    )
    iteration_ceiling = min(
        _EXPANDED_SEARCH_ITERS,
        _positive_environment_int(
            "RAPID_OPTIMIZER_MAX_ITERS",
            _EXPANDED_SEARCH_ITERS,
        ),
    )
    at_ceiling = rollouts >= rollout_ceiling and iterations >= iteration_ceiling
    below_minimum = (
        rollouts < _MIN_MEANINGFUL_SEARCH_ROLLOUTS
        or iterations < _MIN_MEANINGFUL_SEARCH_ITERS
    )
    if below_minimum:
        return "ceiling" if at_ceiling else "diagnostic"
    if at_ceiling:
        if (
            rollout_ceiling < _EXPANDED_SEARCH_ROLLOUTS
            or iteration_ceiling < _EXPANDED_SEARCH_ITERS
        ):
            return "ceiling"
        return "expanded"
    if (
        rollouts >= _STANDARD_SEARCH_ROLLOUTS
        and iterations >= _STANDARD_SEARCH_ITERS
    ):
        return "standard"
    return "minimum"


_DESIGN_FILE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}


def _semantic_source_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".py":
        try:
            tree = ast.parse(raw, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            return raw
        return ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
    if path.suffix == ".json":
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return raw
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return raw


def _strategy_primitive_dependencies(
    source_root: Path,
    program_name: str,
) -> tuple[str, ...]:
    module = sys.modules.get(f"strategies.{program_name}.program")
    program_root = (source_root / "strategies" / program_name).resolve()
    module_file = getattr(module, "__file__", None)
    loaded_from_program = bool(
        module_file
        and Path(module_file).resolve().is_relative_to(program_root)
    )
    phase_program = (
        getattr(module, "PHASE_PROGRAM", None)
        if loaded_from_program
        else None
    )
    phases = getattr(phase_program, "phases", ())
    names = {
        str(phase.primitive)
        for phase in phases
        if isinstance(getattr(phase, "primitive", None), str)
        and phase.primitive
    }
    if names:
        return tuple(sorted(names))

    program_path = source_root / "strategies" / program_name / "program.py"
    if not program_path.is_file():
        return ()
    try:
        tree = ast.parse(program_path.read_bytes(), filename=str(program_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if function_name != "PrimitivePhase":
            continue
        value = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "primitive"
            ),
            node.args[1] if len(node.args) > 1 else None,
        )
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.add(value.value)
    return tuple(sorted(names))


def _program_design_fingerprint(
    report: BindingReport | None,
) -> tuple[str | None, tuple[str, ...]]:
    if report is None:
        return None, ()
    source_root = Path(
        os.environ.get(
            "RAPID_ACTIVE_SRC_DIR",
            os.environ.get("RAPID_SRC_DIR", Path(__file__).resolve().parents[1]),
        )
    )
    plural = "primitives" if report.program_kind == "primitive" else "strategies"
    program_root = source_root / plural / report.program_name
    if not program_root.is_dir():
        return None, ()
    dependencies = (
        _strategy_primitive_dependencies(source_root, report.program_name)
        if report.program_kind == "strategy"
        else ()
    )
    components: list[Path] = [program_root]
    components.extend(source_root / "primitives" / name for name in dependencies)
    components.extend(
        path
        for path in (
            source_root / "primitive_core",
            source_root / "instantiation_core",
            source_root / "strategies" / "program.py",
            source_root / "strategies" / "runtime.py",
        )
        if path.exists()
    )
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for component in components:
        paths = (component,) if component.is_file() else component.rglob("*")
        for path in sorted(paths):
            resolved = path.resolve()
            if (
                resolved in seen
                or not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in _DESIGN_FILE_SUFFIXES
            ):
                continue
            seen.add(resolved)
            try:
                label = path.relative_to(source_root).as_posix()
            except ValueError:
                label = path.as_posix()
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_semantic_source_bytes(path))
            digest.update(b"\0")
    return digest.hexdigest(), dependencies


def _program_execution_fingerprint(
    args: argparse.Namespace,
    report: BindingReport | None,
    budget: Mapping[str, Any],
) -> str | None:
    if report is None:
        return None
    source_root = Path(
        os.environ.get(
            "RAPID_ACTIVE_SRC_DIR",
            os.environ.get("RAPID_SRC_DIR", Path(__file__).resolve().parents[1]),
        )
    )
    plural = "primitives" if report.program_kind == "primitive" else "strategies"
    manifest_path = (
        source_root
        / "instantiations"
        / plural
        / report.program_name
        / report.instance_name
        / "binding.json"
    )
    manifest: Any = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = manifest_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            manifest = None
    excluded_arguments = {
        "cartesian_goals_json",
        "metrics_json",
        "no_record",
        "record",
        "replay",
        "state_trajectory_npz",
        "validate_binding",
    }
    arguments = {
        name: value
        for name, value in vars(args).items()
        if name not in excluded_arguments
    }
    payload = {
        "program_kind": report.program_kind,
        "program_name": report.program_name,
        "instance_name": report.instance_name,
        "scene_module": report.scene_module,
        "binding": manifest,
        "generalization": report.metadata.get("generalization"),
        "arguments": arguments,
        "budget": dict(budget),
    }
    encoded = json.dumps(
        to_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_invention_attempt(
    args: argparse.Namespace,
    result: CompileResult,
    *,
    label: str,
    report: BindingReport | None,
) -> None:
    """Append one compact attempt record when an invention launcher requests it."""

    raw_path = os.environ.get("RAPID_INVENTION_ATTEMPT_LOG", "").strip()
    if not raw_path:
        return
    output = Path(raw_path)
    signature = str(result.metrics["failure_signature"])
    program_kind = report.program_kind if report is not None else None
    program_name = report.program_name if report is not None else None
    instance_name = report.instance_name if report is not None else None
    design_fingerprint, primitive_dependencies = _program_design_fingerprint(report)
    budget = _attempt_budget(
        args,
        result.metrics,
        failure_signature=signature,
    )
    execution_fingerprint = _program_execution_fingerprint(args, report, budget)
    budget_tier = _search_budget_tier(budget)
    stagnation_eligible = bool(
        signature != "success" and budget_tier != "diagnostic"
    )
    repeated = 1 if stagnation_eligible else 0
    same_tier = 1 if stagnation_eligible else 0
    if output.is_file() and stagnation_eligible:
        try:
            previous_records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            previous_records = []
        for record in reversed(previous_records):
            previous_program = (
                record.get("program_kind"),
                record.get("program_name"),
                record.get("instance_name"),
            )
            current_program = (program_kind, program_name, instance_name)
            if previous_program != current_program:
                continue
            if record.get("design_fingerprint") != design_fingerprint:
                break
            if record.get("failure_signature") == "success":
                break
            if not record.get("stagnation_eligible", False):
                continue
            if record.get("failure_signature") != signature:
                break
            repeated += 1
            same_tier += int(record.get("budget_tier") == budget_tier)
    search_exhausted = bool(
        signature != "success" and budget_tier in {"expanded", "ceiling"}
    )
    escalation_available = bool(
        signature != "success" and budget_tier in {"minimum", "standard"}
    )
    repeat_same_tier = bool(stagnation_eligible and same_tier >= 2)
    if signature == "success":
        search_decision = "success"
    elif budget_tier == "diagnostic":
        search_decision = "diagnostic_only"
    elif search_exhausted:
        search_decision = "expanded_search_exhausted"
    elif repeat_same_tier:
        search_decision = "repeat_same_tier_not_useful"
    else:
        search_decision = "budget_escalation_available"
    result.metrics["consecutive_failure_signature_count"] = repeated
    result.metrics["same_tier_failure_count"] = same_tier
    result.metrics["invention_search_budget_tier"] = budget_tier
    result.metrics["budget_escalation_available"] = escalation_available
    result.metrics["repeat_same_tier_not_useful"] = repeat_same_tier
    result.metrics["expanded_search_exhausted"] = search_exhausted
    result.metrics["invention_search_decision"] = search_decision
    result.metrics["stagnation_detected"] = search_exhausted
    result.metrics["design_fingerprint"] = design_fingerprint
    result.metrics["execution_fingerprint"] = execution_fingerprint
    result.metrics["primitive_dependencies"] = list(primitive_dependencies)
    generalization = (
        report.metadata.get("generalization")
        if report is not None
        else None
    )
    record = {
        "label": label,
        "program_kind": program_kind,
        "program_name": program_name,
        "instance_name": instance_name,
        "variant_index": (
            generalization.get("variant_index")
            if isinstance(generalization, Mapping)
            else None
        ),
        "failure_signature": signature,
        "success": signature == "success",
        "design_fingerprint": design_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "primitive_dependencies": list(primitive_dependencies),
        "stagnation_eligible": stagnation_eligible,
        "consecutive_count": repeated,
        "same_tier_count": same_tier,
        "budget_tier": budget_tier,
        "search_decision": search_decision,
        "budget_escalation_available": escalation_available,
        "repeat_same_tier_not_useful": repeat_same_tier,
        "expanded_search_exhausted": search_exhausted,
        "budget": budget,
        "optimized_cost": result.metrics.get("optimized_cost"),
        "optimized_success": result.metrics.get("optimized_success"),
        "cost_components": result.metrics.get("cost_components"),
        "progress_diagnostics": result.metrics.get("progress_diagnostics"),
        "mechanism_diagnostics": result.metrics.get("mechanism_diagnostics"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(to_json_value(record), sort_keys=True) + "\n")
    if search_exhausted:
        print(
            "[invention-search] expanded search is exhausted for failure "
            f"signature {signature!r} under this design and configuration. "
            "Review the trajectory and remaining program improvements; this "
            "does not establish scene infeasibility or complete invention. "
            "Prioritize repair when the mechanism works but the goal is unmet "
            "or the failure recurs across scenes. Continue informative "
            "experiments within the overall run budget; use AGENTS.md for "
            "partial completion and case-specific resampling decisions."
        )
    elif repeat_same_tier:
        print(
            f"[invention-search] {budget_tier} search repeated for failure "
            f"signature {signature!r}. Do not rerun this unchanged tier; "
            "advance the budget when the mechanism remains plausible, otherwise "
            "revisit the scene and trajectory diagnosis before another attempt."
        )


def _report_instantiation_result(
    spec: InstantiationSpec,
    args: argparse.Namespace,
    result: CompileResult,
    *,
    force_failure: bool = False,
) -> CompileResult:
    """Report and retain one completed or audit-rejected trajectory."""

    if spec.binding_report is not None and spec.binding_report.program_kind == "primitive":
        from primitive_core.evidence import finalize_primitive_result
        finalize_primitive_result(result)
    if force_failure:
        result.metrics["success"] = False
    program_success = bool(result.metrics.get("success", False))
    report = (
        spec.binding_report.with_program_success(program_success)
        if spec.binding_report is not None
        else None
    )
    result.metrics["failure_signature"] = _failure_signature(result.metrics)
    retention_errors: list[Exception] = []
    try:
        _record_invention_attempt(args, result, label=spec.label, report=report)
    except Exception as exc:
        retention_errors.append(exc)
    if report is not None:
        result.metrics["binding_valid"] = report.binding_valid
        result.metrics["program_success"] = program_success
        generalization = report.metadata.get("generalization")
        if generalization is not None:
            result.metrics["generalization"] = generalization
        _print_binding(report)
    _print_metrics(spec.label, result)

    # Physical evidence does not depend on metadata, scene reconstruction or video.
    if getattr(args, "state_trajectory_npz", None):
        try:
            write_state_trajectory(args.state_trajectory_npz, result)
        except Exception as exc:
            retention_errors.append(exc)

    cartesian_goals_path = getattr(args, "cartesian_goals_json", None)
    needs_scene_data = bool(args.record or args.replay or cartesian_goals_path)
    model = data = None
    if needs_scene_data:
        try:
            model, data = spec.build_scene_data()
            spec.apply_initial_pose(data, result.initial_pose)
            mujoco.mj_forward(model, data)
        except Exception as exc:
            model = data = None
            retention_errors.append(exc)
    if cartesian_goals_path and model is not None and data is not None:
        try:
            adapter = FR3OSCAdapter.from_model_data(model, data)
            samples = write_cartesian_goal_trajectory(
                cartesian_goals_path,
                result,
                adapter,
            )
        except ValueError as exc:
            result.metrics["cartesian_goal_trajectory_available"] = False
            result.metrics["cartesian_goal_trajectory_reason"] = str(exc)
            print(
                "Cartesian goals were not exported: this is not a compatible "
                f"FR3 OSC trajectory ({exc})"
            )
        except Exception as exc:
            retention_errors.append(exc)
        else:
            result.metrics["cartesian_goal_trajectory_available"] = True
            result.metrics["cartesian_goal_trajectory_samples"] = samples
            result.metrics["cartesian_goal_trajectory_rate_hz"] = 50.0
            result.metrics["cartesian_goal_trajectory_frame"] = "fr3_link0"
            result.metrics["cartesian_goal_trajectory_schema"] = FR3_GOAL_SCHEMA

    if getattr(args, "metrics_json", None):
        try:
            _write_metrics_json(args.metrics_json, spec.label, result, report)
        except Exception as exc:
            retention_errors.append(exc)

    if (args.record or args.replay) and model is not None and data is not None:
        try:
            if args.record:
                spec.playback.record(result, model, data, args.record)
            elif args.replay == "browser":
                spec.playback.browser(result, model, data)
            elif args.replay == "mujoco":
                spec.playback.mujoco(result, model, data)
        except Exception as playback_error:
            retention_errors.append(playback_error)
    if retention_errors:
        first, *additional = retention_errors
        for error in additional:
            first.add_note(f"artifact retention also failed: {error}")
        raise first
    return result


def run_instantiation(
    spec: InstantiationSpec,
    args: argparse.Namespace,
) -> CompileResult:
    """Validate, execute, report, and optionally record one instantiation."""

    apply_optimizer_env_overrides(args)
    np.random.seed(args.seed)
    _banner(spec.label, spec.banner_lines)
    if spec.binding_report is not None:
        if getattr(args, "validate_binding", False):
            _print_binding(spec.binding_report)
        spec.binding_report.require_valid()

    config = build_config(
        spec.config_cls,
        args,
        learned_policy=spec.optimization_policy,
        reference_budget=spec.reference_optimization_budget,
    )
    from primitive_core.profiling import capture_timings
    with capture_execution_results() as completed_results, capture_timings() as timings:
        try:
            result = spec.compile_fn(config)
        except Exception as exc:
            retained_result = getattr(exc, "result", None)
            if not isinstance(retained_result, CompileResult):
                retained_result = completed_results[-1] if completed_results else None
            if isinstance(retained_result, CompileResult):
                retained_result.metrics["timings_inclusive"] = timings
                retained_result.metrics["execution_error"] = {
                    "type": type(exc).__name__, "message": str(exc),
                }
                try:
                    _report_instantiation_result(
                        spec, args, retained_result, force_failure=True,
                    )
                except Exception as retention_error:
                    exc.add_note(f"retaining the trajectory also failed: {retention_error}")
            raise
    result.metrics["timings_inclusive"] = timings
    return _report_instantiation_result(spec, args, result)


__all__ = [
    "InstantiationSpec",
    "apply_optimizer_env_overrides",
    "build_config",
    "cartesian_goals_path_for_metrics",
    "make_parser",
    "run_instantiation",
    "write_cartesian_goal_trajectory",
    "write_state_trajectory",
]
