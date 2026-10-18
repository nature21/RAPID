"""Benchmark-owned success predicates for sealed real-to-sim evaluations.

The learned strategy predicate is an optimization and debugging contract.  It
is deliberately not the benchmark oracle.  New benchmark runs are scored from
the retained MuJoCo state trajectory and the private task layout.  A small
metrics adapter is retained so runs created before state retention was added
can still be rescored where the required physical measurements exist.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


PICKUP_STRATEGIES = frozenset({2, 3, 4, 7, 8})
STORAGE_LAYOUT_KEYS = {
    1: "corridor",
    5: "upright_slot",
    6: "planar_slot",
}

MINIMUM_JAW_CONTACT_FRACTION = 0.60
MINIMUM_SUPPORT_CLEARANCE_M = 0.01
TERMINAL_GRASP_WINDOW_SECONDS = 0.30
STORAGE_CONTAINMENT_TOLERANCE_M = 0.005
_NUMERIC_EPSILON_M = 1e-9

# Kept as aliases because these names were published in the first strategy-2
# benchmark result schema.
STRATEGY_2_EVALUATOR = "strategy_2_terminal_bilateral_grasp_clearance_v1"
STRATEGY_2_MINIMUM_JAW_CONTACT_FRACTION = MINIMUM_JAW_CONTACT_FRACTION
STRATEGY_2_MINIMUM_SUPPORT_CLEARANCE_M = MINIMUM_SUPPORT_CLEARANCE_M


def _storage_evaluator(strategy_id: int) -> str:
    return f"strategy_{strategy_id}_terminal_storage_center_region_v3"


def _finite_number(values: Mapping[str, Any], name: str) -> float | None:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name)
    return value if isinstance(value, Mapping) else {}


def _final_psi(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    retained = metrics.get("metrics", metrics)
    if not isinstance(retained, Mapping):
        return {}
    return _mapping(retained, "final_psi")


def _evaluation(
    *,
    evaluator: str,
    success: bool,
    available: bool,
    conjuncts: Mapping[str, bool],
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    source: str,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluator": evaluator,
        "source": source,
        "success": bool(success and available),
        "measurements_available": bool(available),
        "conjuncts": {name: bool(value) for name, value in conjuncts.items()},
        "measurements": dict(measurements),
        "thresholds": dict(thresholds),
    }
    if error is not None:
        result["error"] = error
    return result


def _collision_geoms_below(model: mujoco.MjModel, body_id: int) -> set[int]:
    def descends_from(candidate: int) -> bool:
        current = int(candidate)
        while current > 0:
            if current == body_id:
                return True
            current = int(model.body_parentid[current])
        return current == body_id

    return {
        geom_id
        for geom_id in range(model.ngeom)
        if descends_from(int(model.geom_bodyid[geom_id]))
        and bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    }


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"scene has no body named {name!r}")
    return int(body_id)


def _projected_radius(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    direction: np.ndarray,
) -> float:
    local = data.geom_xmat[geom_id].reshape(3, 3).T @ direction
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.abs(local) @ size)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(np.linalg.norm(local * size))
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        axial = abs(float(local[2]))
        radial = math.sqrt(max(0.0, 1.0 - axial * axial))
        radius = radial * float(size[0]) + axial * float(size[1])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius = float(size[0]) + axial * float(size[1])
        return radius
    return float(model.geom_rbound[geom_id])


def _load_states(path: str | Path, model: mujoco.MjModel) -> np.ndarray:
    state_path = Path(path)
    if not state_path.is_file():
        raise ValueError(f"state trajectory was not written: {state_path}")
    with np.load(state_path, allow_pickle=False) as retained:
        if "states" not in retained:
            raise ValueError("state trajectory has no 'states' array")
        states = np.asarray(retained["states"], dtype=float)
    expected_width = model.nq + model.nv
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != expected_width:
        raise ValueError(
            "state trajectory must have shape (T, nq + nv); "
            f"got {states.shape}, expected width {expected_width}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("state trajectory contains non-finite values")
    return states


def _load_model(scene_module: str) -> mujoco.MjModel:
    module = importlib.import_module(scene_module)
    make_model = getattr(module, "make_model", None)
    if not callable(make_model):
        raise ValueError(f"scene module {scene_module!r} has no make_model()")
    model = make_model()
    if not isinstance(model, mujoco.MjModel):
        raise TypeError(f"{scene_module}.make_model() did not return MjModel")
    return model


def _unit_xy(layout: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(layout.get(name), dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"layout field {name!r} must be a finite XY vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError(f"layout field {name!r} must be nonzero")
    return value / norm


def _target_projection_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geoms: set[int],
    direction: np.ndarray,
    origin: np.ndarray,
) -> tuple[float, float]:
    lower = math.inf
    upper = -math.inf
    for geom_id in target_geoms:
        center = float((data.geom_xpos[geom_id] - origin) @ direction)
        radius = _projected_radius(model, data, geom_id, direction)
        lower = min(lower, center - radius)
        upper = max(upper, center + radius)
    return lower, upper


def _storage_evaluation_from_bounds(
    strategy_id: int,
    *,
    source: str,
    depth_min: float,
    depth_max: float,
    lateral_min: float,
    lateral_max: float,
    available_depth: float,
    half_width: float,
    tolerance: float,
) -> dict[str, Any]:
    """Score storage after the target center, rather than its rear edge, enters."""

    center_depth = 0.5 * (depth_min + depth_max)
    center_lateral = 0.5 * (lateral_min + lateral_max)
    conjuncts = {
        "center_past_entrance": center_depth >= -_NUMERIC_EPSILON_M,
        "center_before_back_boundary": (
            center_depth <= available_depth + tolerance
        ),
        "center_laterally_contained": (
            center_lateral >= -half_width - tolerance
            and center_lateral <= half_width + tolerance
        ),
    }
    return _evaluation(
        evaluator=_storage_evaluator(strategy_id),
        source=source,
        success=all(conjuncts.values()),
        available=True,
        conjuncts=conjuncts,
        measurements={
            "target_center_depth_m": center_depth,
            "target_center_lateral_m": center_lateral,
            "target_depth_min_m": depth_min,
            "target_depth_max_m": depth_max,
            "target_lateral_min_m": lateral_min,
            "target_lateral_max_m": lateral_max,
        },
        thresholds={
            "minimum_target_center_depth_m": 0.0,
            "maximum_target_center_depth_m": available_depth,
            "target_center_lateral_half_width_m": half_width,
            # Retain v1/v2 names so archived decisions remain self-describing
            # and can be rescored again without loading the simulation.
            "available_depth_m": available_depth,
            "lateral_half_width_m": half_width,
            "containment_tolerance_m": tolerance,
        },
    )


def _trajectory_storage_success(
    strategy_id: int,
    model: mujoco.MjModel,
    states: np.ndarray,
    private_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    layout_key = STORAGE_LAYOUT_KEYS[strategy_id]
    context_layout = _mapping(private_manifest, "context_layout")
    layout = _mapping(context_layout, layout_key)
    roles = _mapping(private_manifest, "authoritative_roles")
    target_name = roles.get("target")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("private manifest has no authoritative target body")

    entrance = np.asarray(layout.get("entrance_xy"), dtype=float)
    if entrance.shape != (2,) or not np.all(np.isfinite(entrance)):
        raise ValueError("storage layout has no finite entrance_xy")
    axis_xy = _unit_xy(layout, "axis_xy")
    lateral_xy = _unit_xy(layout, "lateral_xy")
    clear_width = _finite_number(layout, "clear_width")
    available_depth = _finite_number(layout, "available_depth")
    if strategy_id == 1:
        available_depth = _finite_number(layout, "available_store_depth")
    if clear_width is None or clear_width <= 0.0:
        raise ValueError("storage layout has no positive clear_width")
    if available_depth is None or available_depth <= 0.0:
        raise ValueError("storage layout has no positive available depth")

    target_geoms = _collision_geoms_below(model, _body_id(model, target_name))
    if not target_geoms:
        raise ValueError("authoritative target has no collision geometry")
    data = mujoco.MjData(model)
    data.qpos[:] = states[-1, : model.nq]
    data.qvel[:] = states[-1, model.nq :]
    mujoco.mj_forward(model, data)
    origin = np.asarray((entrance[0], entrance[1], 0.0), dtype=float)
    axis = np.asarray((axis_xy[0], axis_xy[1], 0.0), dtype=float)
    lateral = np.asarray((lateral_xy[0], lateral_xy[1], 0.0), dtype=float)
    depth_min, depth_max = _target_projection_bounds(
        model, data, target_geoms, axis, origin
    )
    lateral_min, lateral_max = _target_projection_bounds(
        model, data, target_geoms, lateral, origin
    )
    half_width = 0.5 * clear_width
    tolerance = STORAGE_CONTAINMENT_TOLERANCE_M
    return _storage_evaluation_from_bounds(
        strategy_id,
        source="state_trajectory",
        depth_min=depth_min,
        depth_max=depth_max,
        lateral_min=lateral_min,
        lateral_max=lateral_max,
        available_depth=available_depth,
        half_width=half_width,
        tolerance=tolerance,
    )


def _support_top_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    private_manifest: Mapping[str, Any],
) -> float:
    roles = _mapping(private_manifest, "authoritative_roles")
    support_name = roles.get("support")
    if support_name is None:
        world = _mapping(_mapping(private_manifest, "context_layout"), "world_frame")
        floor_z = _finite_number(world, "floor_z")
        if floor_z is None:
            raise ValueError("private manifest has no benchmark floor height")
        return floor_z
    if not isinstance(support_name, str) or not support_name:
        raise ValueError("authoritative support must be a body name or null")
    support_geoms = _collision_geoms_below(model, _body_id(model, support_name))
    if not support_geoms:
        raise ValueError("authoritative support has no collision geometry")
    up = np.asarray((0.0, 0.0, 1.0), dtype=float)
    return max(
        float(data.geom_xpos[geom_id, 2])
        + _projected_radius(model, data, geom_id, up)
        for geom_id in support_geoms
    )


def _trajectory_pickup_success(
    strategy_id: int,
    model: mujoco.MjModel,
    states: np.ndarray,
    private_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    roles = _mapping(private_manifest, "authoritative_roles")
    target_name = roles.get("target")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("private manifest has no authoritative target body")
    target_geoms = _collision_geoms_below(model, _body_id(model, target_name))
    jaw_a_geoms = _collision_geoms_below(model, _body_id(model, "left_finger"))
    jaw_b_geoms = _collision_geoms_below(model, _body_id(model, "right_finger"))
    if not target_geoms or not jaw_a_geoms or not jaw_b_geoms:
        raise ValueError("target and both FR3 fingers need collision geometry")

    late_count = min(
        states.shape[0],
        max(
            2,
            int(math.ceil(TERMINAL_GRASP_WINDOW_SECONDS / model.opt.timestep)),
        ),
    )
    late_states = states[-late_count:]
    jaw_a_contact = np.zeros(late_count, dtype=bool)
    jaw_b_contact = np.zeros(late_count, dtype=bool)
    clearances = np.empty(late_count, dtype=float)
    data = mujoco.MjData(model)
    up = np.asarray((0.0, 0.0, 1.0), dtype=float)
    for index, state in enumerate(late_states):
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq :]
        mujoco.mj_forward(model, data)
        target_bottom = min(
            float(data.geom_xpos[geom_id, 2])
            - _projected_radius(model, data, geom_id, up)
            for geom_id in target_geoms
        )
        clearances[index] = target_bottom - _support_top_z(
            model, data, private_manifest
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            jaw_a_contact[index] |= bool(pair & jaw_a_geoms and pair & target_geoms)
            jaw_b_contact[index] |= bool(pair & jaw_b_geoms and pair & target_geoms)

    jaw_a_fraction = float(np.mean(jaw_a_contact))
    jaw_b_fraction = float(np.mean(jaw_b_contact))
    minimum_clearance = float(np.min(clearances))
    conjuncts = {
        "terminal_bilateral_grasp": (
            jaw_a_fraction >= MINIMUM_JAW_CONTACT_FRACTION
            and jaw_b_fraction >= MINIMUM_JAW_CONTACT_FRACTION
        ),
        "terminal_support_clearance": (
            minimum_clearance
            >= MINIMUM_SUPPORT_CLEARANCE_M - _NUMERIC_EPSILON_M
        ),
    }
    evaluator = (
        STRATEGY_2_EVALUATOR
        if strategy_id == 2
        else f"strategy_{strategy_id}_terminal_bilateral_grasp_clearance_v1"
    )
    return _evaluation(
        evaluator=evaluator,
        source="state_trajectory",
        success=all(conjuncts.values()),
        available=True,
        conjuncts=conjuncts,
        measurements={
            "jaw_a_late_contact_fraction": jaw_a_fraction,
            "jaw_b_late_contact_fraction": jaw_b_fraction,
            "minimum_late_support_clearance_m": minimum_clearance,
            "terminal_window_steps": late_count,
        },
        thresholds={
            "minimum_jaw_contact_fraction": MINIMUM_JAW_CONTACT_FRACTION,
            "minimum_support_clearance_m": MINIMUM_SUPPORT_CLEARANCE_M,
            "terminal_window_seconds": TERMINAL_GRASP_WINDOW_SECONDS,
        },
    )


def _legacy_pickup_success(
    strategy_id: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    final = _final_psi(metrics)
    if not final:
        return None
    sections = [final]
    for name in ("measurements", "diagnostics", "values"):
        sections.append(_mapping(final, name))

    def first_number(*names: str) -> float | None:
        for section in sections:
            for name in names:
                value = _finite_number(section, name)
                if value is not None:
                    return value
        return None

    jaw_a = first_number("jaw_a_late_contact_fraction")
    jaw_b = first_number("jaw_b_late_contact_fraction")
    clearance_names = (
        "minimum_late_support_clearance",
        "minimum_late_target_bottom_clearance_m",
        "minimum_late_lift_progress",
        "minimum_late_lift_height",
    )
    clearance = first_number(*clearance_names)
    available = all(value is not None for value in (jaw_a, jaw_b, clearance))
    if not available:
        return None
    assert jaw_a is not None and jaw_b is not None and clearance is not None
    conjuncts = {
        "terminal_bilateral_grasp": (
            jaw_a >= MINIMUM_JAW_CONTACT_FRACTION
            and jaw_b >= MINIMUM_JAW_CONTACT_FRACTION
        ),
        "terminal_support_clearance": (
            clearance >= MINIMUM_SUPPORT_CLEARANCE_M - _NUMERIC_EPSILON_M
        ),
    }
    evaluator = (
        STRATEGY_2_EVALUATOR
        if strategy_id == 2
        else f"strategy_{strategy_id}_terminal_bilateral_grasp_clearance_v1"
    )
    return _evaluation(
        evaluator=evaluator,
        source="legacy_strategy_measurements",
        success=all(conjuncts.values()),
        available=True,
        conjuncts=conjuncts,
        measurements={
            "jaw_a_late_contact_fraction": jaw_a,
            "jaw_b_late_contact_fraction": jaw_b,
            "minimum_late_support_clearance_m": clearance,
        },
        thresholds={
            "minimum_jaw_contact_fraction": MINIMUM_JAW_CONTACT_FRACTION,
            "minimum_support_clearance_m": MINIMUM_SUPPORT_CLEARANCE_M,
        },
    )


def _legacy_storage_success(
    strategy_id: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    final = _final_psi(metrics)
    if not final:
        return None
    tolerance = STORAGE_CONTAINMENT_TOLERANCE_M
    if strategy_id == 1:
        conjuncts = _mapping(final, "conjuncts")
        lateral = _mapping(conjuncts, "lateral_containment")
        depth = _mapping(conjuncts, "insertion_depth")
        lateral_value = _finite_number(lateral, "value")
        lateral_threshold = _finite_number(lateral, "threshold")
        depth_value = _finite_number(depth, "value")
        entrance = _finite_number(_mapping(final, "corridor_frame"), "entrance")
        available = all(
            value is not None
            for value in (
                lateral_value,
                lateral_threshold,
                depth_value,
                entrance,
            )
        )
        if not available:
            return None
        assert lateral_value is not None and lateral_threshold is not None
        assert depth_value is not None and entrance is not None
        center_depth = depth_value - entrance
        checks = {
            "center_laterally_contained": lateral_value <= lateral_threshold,
            "center_past_entrance": center_depth >= -_NUMERIC_EPSILON_M,
        }
        measurements = {
            "lateral_containment_violation_m": lateral_value,
            "target_center_depth_m": center_depth,
        }
        thresholds = {
            "maximum_lateral_violation_m": lateral_threshold,
            "minimum_target_center_depth_m": 0.0,
        }
    elif strategy_id == 5:
        values = _mapping(final, "values")
        depth = _finite_number(values, "median_insertion_depth_m")
        radius = _finite_number(values, "median_target_longitudinal_radius_m")
        wall_margin = _finite_number(values, "minimum_wall_containment_margin_m")
        boundary_margin = _finite_number(
            values, "minimum_boundary_containment_margin_m"
        )
        available = all(
            value is not None
            for value in (depth, wall_margin, boundary_margin)
        )
        if not available:
            return None
        assert depth is not None
        assert wall_margin is not None and boundary_margin is not None
        checks = {
            "center_past_entrance": depth >= -_NUMERIC_EPSILON_M,
            "center_laterally_contained": (
                wall_margin >= -tolerance and boundary_margin >= -tolerance
            ),
        }
        measurements = {
            "target_center_depth_m": depth,
            "target_longitudinal_radius_m": radius,
            "wall_containment_margin_m": wall_margin,
            "boundary_containment_margin_m": boundary_margin,
        }
        thresholds = {
            "minimum_target_center_depth_m": 0.0,
            "containment_tolerance_m": tolerance,
        }
    else:
        return None
    return _evaluation(
        evaluator=_storage_evaluator(strategy_id),
        source="legacy_strategy_measurements",
        success=all(checks.values()),
        available=True,
        conjuncts=checks,
        measurements=measurements,
        thresholds=thresholds,
    )


def rescore_retained_benchmark_evaluation(
    strategy_id: int,
    evaluation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Migrate a retained state-based storage decision without rerunning physics."""

    if strategy_id not in STORAGE_LAYOUT_KEYS:
        return dict(evaluation)
    if evaluation.get("source") != "state_trajectory":
        return None
    measurements = _mapping(evaluation, "measurements")
    thresholds = _mapping(evaluation, "thresholds")
    depth_min = _finite_number(measurements, "target_depth_min_m")
    depth_max = _finite_number(measurements, "target_depth_max_m")
    lateral_min = _finite_number(measurements, "target_lateral_min_m")
    lateral_max = _finite_number(measurements, "target_lateral_max_m")
    available_depth = _finite_number(thresholds, "available_depth_m")
    half_width = _finite_number(thresholds, "lateral_half_width_m")
    tolerance = _finite_number(thresholds, "containment_tolerance_m")
    if any(
        value is None
        for value in (
            depth_min,
            depth_max,
            lateral_min,
            lateral_max,
            available_depth,
            half_width,
            tolerance,
        )
    ):
        return None
    assert depth_min is not None and depth_max is not None
    assert lateral_min is not None and lateral_max is not None
    assert available_depth is not None and half_width is not None
    assert tolerance is not None
    return _storage_evaluation_from_bounds(
        strategy_id,
        source="state_trajectory",
        depth_min=depth_min,
        depth_max=depth_max,
        lateral_min=lateral_min,
        lateral_max=lateral_max,
        available_depth=available_depth,
        half_width=half_width,
        tolerance=tolerance,
    )


def evaluate_benchmark_success(
    strategy_id: int,
    metrics: Mapping[str, Any],
    *,
    state_trajectory_path: str | Path | None = None,
    private_manifest: Mapping[str, Any] | None = None,
    scene_module: str | None = None,
    model: mujoco.MjModel | None = None,
) -> dict[str, Any] | None:
    """Return the benchmark-owned task decision for one strategy.

    When ``state_trajectory_path`` is supplied, failure to read or evaluate it
    is itself a failed benchmark decision.  Omitting the path selects the
    legacy measurement adapter; old artifacts without enough physical evidence
    fail closed.
    """

    if strategy_id not in PICKUP_STRATEGIES and strategy_id not in STORAGE_LAYOUT_KEYS:
        return None
    if state_trajectory_path is not None:
        evaluator = (
            STRATEGY_2_EVALUATOR
            if strategy_id == 2
            else (
                f"strategy_{strategy_id}_terminal_bilateral_grasp_clearance_v1"
                if strategy_id in PICKUP_STRATEGIES
                else _storage_evaluator(strategy_id)
            )
        )
        try:
            if private_manifest is None:
                raise ValueError("private manifest is required for trajectory scoring")
            if model is None:
                if not scene_module:
                    raise ValueError("scene module is required for trajectory scoring")
                model = _load_model(scene_module)
            states = _load_states(state_trajectory_path, model)
            if strategy_id in PICKUP_STRATEGIES:
                return _trajectory_pickup_success(
                    strategy_id, model, states, private_manifest
                )
            return _trajectory_storage_success(
                strategy_id, model, states, private_manifest
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            return _evaluation(
                evaluator=evaluator,
                source="state_trajectory",
                success=False,
                available=False,
                conjuncts={},
                measurements={},
                thresholds={},
                error=str(exc),
            )

    if strategy_id in PICKUP_STRATEGIES:
        legacy = _legacy_pickup_success(strategy_id, metrics)
        if legacy is not None:
            return legacy
        return _evaluation(
            evaluator=(
                STRATEGY_2_EVALUATOR
                if strategy_id == 2
                else f"strategy_{strategy_id}_terminal_bilateral_grasp_clearance_v1"
            ),
            source="legacy_strategy_measurements",
            success=False,
            available=False,
            conjuncts={
                "terminal_bilateral_grasp": False,
                "terminal_support_clearance": False,
            },
            measurements={
                "jaw_a_late_contact_fraction": None,
                "jaw_b_late_contact_fraction": None,
                "minimum_late_support_clearance_m": None,
            },
            thresholds={
                "minimum_jaw_contact_fraction": MINIMUM_JAW_CONTACT_FRACTION,
                "minimum_support_clearance_m": MINIMUM_SUPPORT_CLEARANCE_M,
            },
        )
    legacy = _legacy_storage_success(strategy_id, metrics)
    if legacy is not None:
        return legacy
    return _evaluation(
        evaluator=_storage_evaluator(strategy_id),
        source="legacy_strategy_measurements",
        success=False,
        available=False,
        conjuncts={
            "center_past_entrance": False,
            "center_before_back_boundary": False,
            "center_laterally_contained": False,
        },
        measurements={},
        thresholds={"containment_tolerance_m": STORAGE_CONTAINMENT_TOLERANCE_M},
    )


__all__ = [
    "MINIMUM_JAW_CONTACT_FRACTION",
    "MINIMUM_SUPPORT_CLEARANCE_M",
    "PICKUP_STRATEGIES",
    "STORAGE_CONTAINMENT_TOLERANCE_M",
    "STORAGE_LAYOUT_KEYS",
    "STRATEGY_2_EVALUATOR",
    "STRATEGY_2_MINIMUM_JAW_CONTACT_FRACTION",
    "STRATEGY_2_MINIMUM_SUPPORT_CLEARANCE_M",
    "TERMINAL_GRASP_WINDOW_SECONDS",
    "evaluate_benchmark_success",
    "rescore_retained_benchmark_evaluation",
]
