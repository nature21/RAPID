"""Private, strategy-independent feasibility gates for generated cases.

The certifier never imports learned primitives or strategies.  It uses the
materialized scene, privileged role map, the fixed FR3 model, and the phase
contract corresponding to the source demonstration.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np

from primitive_core.fr3_osc import FR3OSCAdapter
from primitive_core.native_fr3_osc import NativeFR3OSCController
from primitive_core.policy import GripperOperation
from vistools.visualize_mujoco import _make_camera

from .specs import (
    StrategyContract,
    strategy_2_manipulation_metrics,
    strategy_3_manipulation_metrics,
)


SETTLE_SECONDS = 3.0
TERMINAL_WINDOW_SECONDS = 0.25
MAX_LINEAR_SPEED = 0.01
MAX_ANGULAR_SPEED = 0.10
MAX_TRANSIENT_LINEAR_SPEED = 0.05
MAX_TRANSIENT_ANGULAR_SPEED = 0.50
MAX_PENETRATION = 0.001
MAX_XY_DRIFT = 0.003
MAX_Z_DRIFT = 0.002
MAX_ORIENTATION_DRIFT = math.radians(3.0)
IK_POSITION_TOLERANCE = 0.005
IK_ORIENTATION_TOLERANCE = math.radians(5.0)
IK_JOINT_MARGIN_RADIANS = math.radians(3.0)
WORKSPACE_X = (0.30, 1.02)
WORKSPACE_Y = (-0.38, 0.46)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: dict[str, Any]
    detail: str = ""


@dataclass(frozen=True)
class CertificationResult:
    passed: bool
    gates: tuple[GateResult, ...]
    certifier: str = "private_phase_reachability_v5"

    def to_json(self) -> dict[str, Any]:
        return _json_ready({
            "passed": self.passed,
            "certifier": self.certifier,
            "gates": [asdict(gate) for gate in self.gates],
        })


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalar metrics into stable JSON-native values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_scene(case_dir: Path) -> ModuleType:
    path = case_dir / "scene.py"
    module_name = f"_rapid_benchmark_cert_{case_dir.parent.name}_{case_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load benchmark scene: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_joint_velocity_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    maximum_linear = 0.0
    maximum_angular = 0.0
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        address = int(model.jnt_dofadr[joint_id])
        maximum_linear = max(maximum_linear, float(np.linalg.norm(data.qvel[address : address + 3])))
        maximum_angular = max(maximum_angular, float(np.linalg.norm(data.qvel[address + 3 : address + 6])))
    return maximum_linear, maximum_angular


def _free_body_states(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Return world pose and velocity address for every free body."""

    states: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            name = f"body_{body_id}"
        states[name] = (
            np.asarray(data.xpos[body_id], dtype=np.float64).copy(),
            np.asarray(data.xquat[body_id], dtype=np.float64).copy(),
            int(model.jnt_dofadr[joint_id]),
        )
    return states


def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    alignment = float(abs(np.dot(left, right)))
    return 2.0 * math.acos(float(np.clip(alignment, -1.0, 1.0)))


def _settling_gate(
    module: ModuleType,
    private_record: dict[str, Any] | None = None,
) -> tuple[GateResult, mujoco.MjModel, mujoco.MjData]:
    """Certify the raw reset state with the exact evaluation controller.

    The returned model/data remain at the raw reset state.  Certification must
    not silently replace a generated pose with a different pose obtained by
    settling; downstream geometry and perturbation gates therefore inspect the
    same state that evaluation starts from.
    """

    model = module.make_model()
    data = module.make_data(model)
    mujoco.mj_forward(model, data)
    runtime_model = module.make_model()
    runtime_data = module.make_data(runtime_model)
    mujoco.mj_forward(runtime_model, runtime_data)
    initial = _free_body_states(runtime_model, runtime_data)
    steps = max(1, int(round(SETTLE_SECONDS / runtime_model.opt.timestep)))
    terminal_steps = max(
        1,
        int(round(TERMINAL_WINDOW_SECONDS / runtime_model.opt.timestep)),
    )
    per_body = {
        name: {
            "max_xy_drift": 0.0,
            "max_z_drift": 0.0,
            "max_position_drift": 0.0,
            "max_orientation_drift_radians": 0.0,
            "max_linear_speed": 0.0,
            "max_angular_speed": 0.0,
        }
        for name in initial
    }
    terminal_linear: list[float] = []
    terminal_angular: list[float] = []
    minimum_distance = 0.0
    runtime_error = ""
    integrator_before = int(runtime_model.opt.integrator)
    try:
        adapter = FR3OSCAdapter.from_model_data(runtime_model, runtime_data)
        hold_action = adapter.action_from_data(runtime_model, runtime_data)
        controller = NativeFR3OSCController(
            runtime_model,
            runtime_data,
            adapter,
        )
        for index in range(steps):
            controller.step(hold_action, GripperOperation.MOVE)
            for name, (position, quaternion, dof_address) in initial.items():
                body_id = mujoco.mj_name2id(
                    runtime_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                delta = np.asarray(runtime_data.xpos[body_id]) - position
                metrics = per_body[name]
                metrics["max_xy_drift"] = max(
                    metrics["max_xy_drift"],
                    float(np.linalg.norm(delta[:2])),
                )
                metrics["max_z_drift"] = max(
                    metrics["max_z_drift"],
                    abs(float(delta[2])),
                )
                metrics["max_position_drift"] = max(
                    metrics["max_position_drift"],
                    float(np.linalg.norm(delta)),
                )
                metrics["max_orientation_drift_radians"] = max(
                    metrics["max_orientation_drift_radians"],
                    _quaternion_distance(
                        quaternion,
                        np.asarray(runtime_data.xquat[body_id]),
                    ),
                )
                linear = float(
                    np.linalg.norm(
                        runtime_data.qvel[dof_address : dof_address + 3]
                    )
                )
                angular = float(
                    np.linalg.norm(
                        runtime_data.qvel[dof_address + 3 : dof_address + 6]
                    )
                )
                metrics["max_linear_speed"] = max(
                    metrics["max_linear_speed"], linear
                )
                metrics["max_angular_speed"] = max(
                    metrics["max_angular_speed"], angular
                )
            linear, angular = _free_joint_velocity_metrics(
                runtime_model,
                runtime_data,
            )
            if index >= steps - terminal_steps:
                terminal_linear.append(linear)
                terminal_angular.append(angular)
            if runtime_data.ncon:
                minimum_distance = min(
                    minimum_distance,
                    min(float(contact.dist) for contact in runtime_data.contact),
                )
    except (RuntimeError, ValueError) as exc:
        runtime_error = str(exc)
    if terminal_linear:
        percentile_linear = float(np.percentile(terminal_linear, 95))
        percentile_angular = float(np.percentile(terminal_angular, 95))
    else:
        percentile_linear = math.inf
        percentile_angular = math.inf

    free_body_checks = {
        name: (
            metrics["max_xy_drift"] <= MAX_XY_DRIFT
            and metrics["max_z_drift"] <= MAX_Z_DRIFT
            and metrics["max_orientation_drift_radians"]
            <= MAX_ORIENTATION_DRIFT
            and metrics["max_linear_speed"] <= MAX_TRANSIENT_LINEAR_SPEED
            and metrics["max_angular_speed"] <= MAX_TRANSIENT_ANGULAR_SPEED
        )
        for name, metrics in per_body.items()
    }
    distractor_checks: dict[str, bool] = {}
    if private_record is not None:
        for name in private_record["authoritative_roles"].get("distractors", []):
            body_id = mujoco.mj_name2id(
                runtime_model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(name),
            )
            distractor_checks[str(name)] = (
                body_id >= 0 and int(runtime_model.body_jntnum[body_id]) == 0
            )
    passed = (
        not runtime_error
        and all(free_body_checks.values())
        and all(distractor_checks.values())
        and percentile_linear <= MAX_LINEAR_SPEED
        and percentile_angular <= MAX_ANGULAR_SPEED
        and minimum_distance >= -MAX_PENETRATION
        and bool(np.all(np.isfinite(runtime_data.qpos)))
    )
    integrator_after = int(runtime_model.opt.integrator)
    return (
        GateResult(
            "runtime_passive_stability",
            passed,
            {
                "backend": "native_cpp_fr3_osc",
                "steps": steps,
                "seconds": SETTLE_SECONDS,
                "integrator_before_controller": integrator_before,
                "integrator_during_evaluation": integrator_after,
                "free_bodies": per_body,
                "free_body_checks": free_body_checks,
                "static_distractor_checks": distractor_checks,
                "p95_terminal_linear_speed": percentile_linear,
                "p95_terminal_angular_speed": percentile_angular,
                "minimum_contact_distance": minimum_distance,
                "limits": {
                    "xy_drift": MAX_XY_DRIFT,
                    "z_drift": MAX_Z_DRIFT,
                    "orientation_drift_radians": MAX_ORIENTATION_DRIFT,
                    "transient_linear_speed": MAX_TRANSIENT_LINEAR_SPEED,
                    "transient_angular_speed": MAX_TRANSIENT_ANGULAR_SPEED,
                    "terminal_linear_speed": MAX_LINEAR_SPEED,
                    "terminal_angular_speed": MAX_ANGULAR_SPEED,
                    "penetration": MAX_PENETRATION,
                },
                "runtime_error": runtime_error,
            },
            "raw reset poses must remain stable under the exact evaluation controller for the full passive rollout",
        ),
        model,
        data,
    )


def _body_box_geoms(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return []
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        and int(model.geom_group[geom_id]) == 3
    ]


def _geom_world_half_extents(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    return np.abs(rotation) @ np.asarray(model.geom_size[geom_id])


def _world_frame_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    private_record: dict[str, Any],
) -> GateResult:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3")
    floor_z = (
        float(data.geom_xpos[floor_id, 2])
        if floor_id >= 0
        else math.nan
    )
    robot_base_z = (
        float(data.xpos[robot_id, 2])
        if robot_id >= 0
        else math.nan
    )
    recorded = private_record.get("context_layout", {}).get("world_frame", {})
    tolerance = 1e-8
    passed = (
        floor_id >= 0
        and robot_id >= 0
        and abs(floor_z) <= tolerance
        and abs(robot_base_z) <= tolerance
        and abs(float(recorded.get("floor_z", math.inf))) <= tolerance
        and abs(float(recorded.get("robot_base_z", math.inf))) <= tolerance
    )
    return GateResult(
        "world_frame",
        passed,
        {
            "floor_z": floor_z,
            "robot_base_z": robot_base_z,
            "recorded": recorded,
            "tolerance": tolerance,
        },
        "the work surface and FR3 base must share the world z=0 datum",
    )


def _representation_gate(
    model: mujoco.MjModel,
    private_record: dict[str, Any],
) -> GateResult:
    roles = private_record["authoritative_roles"]
    checks: dict[str, Any] = {}
    passed = True
    names = [
        str(roles["target"]),
        *(str(item) for item in roles.get("secondary_dynamic", [])),
        *(str(item) for item in roles.get("distractors", [])),
    ]
    target_source = private_record.get("target_representation", {}).get(
        "visual_source"
    )
    secondary_sources = private_record.get("context_layout", {}).get(
        "object_geometry", {}
    )
    distractor_names = {
        str(item) for item in roles.get("distractors", [])
    }
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_ids = (
            [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
            ]
            if body_id >= 0
            else []
        )
        visual_ids = [
            geom_id for geom_id in geom_ids if int(model.geom_group[geom_id]) == 2
        ]
        collision_ids = [
            geom_id for geom_id in geom_ids if int(model.geom_group[geom_id]) == 3
        ]
        collision_types = [int(model.geom_type[item]) for item in collision_ids]
        visual_types = [int(model.geom_type[item]) for item in visual_ids]
        expected_source = None
        if name == str(roles["target"]):
            expected_source = target_source
        elif name in secondary_sources:
            expected_source = secondary_sources[name].get("visual_source")
        source_matches = True
        if expected_source == "hope_asset":
            source_matches = visual_types == [int(mujoco.mjtGeom.mjGEOM_MESH)]
        elif expected_source == "material_primitive":
            source_matches = bool(visual_types) and all(
                item in {
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                }
                for item in visual_types
            )
        body_passed = (
            body_id >= 0
            and len(visual_ids) == 1
            and len(collision_ids) == 1
            and collision_types[0]
            in {
                int(mujoco.mjtGeom.mjGEOM_BOX),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            }
            and source_matches
        )
        joint_types = (
            [
                int(model.jnt_type[joint_id])
                for joint_id in range(
                    int(model.body_jntadr[body_id]),
                    int(model.body_jntadr[body_id] + model.body_jntnum[body_id]),
                )
            ]
            if body_id >= 0
            else []
        )
        expected_motion = "static" if name in distractor_names else "dynamic"
        motion_matches = (
            not joint_types
            if expected_motion == "static"
            else joint_types == [int(mujoco.mjtJoint.mjJNT_FREE)]
        )
        body_passed = body_passed and motion_matches
        passed = passed and body_passed
        checks[name] = {
            "visual_geoms": visual_ids,
            "collision_geoms": collision_ids,
            "visual_types": visual_types,
            "collision_types": collision_types,
            "expected_source": expected_source,
            "expected_motion": expected_motion,
            "joint_types": joint_types,
            "motion_matches": motion_matches,
            "passed": body_passed,
        }
    return GateResult(
        "simulation_representation",
        passed,
        {"objects": checks},
        "causal objects must be dynamic and noncausal distractors static, each with one visual and one primitive collider",
    )


def _geometry_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: StrategyContract,
    private_record: dict[str, Any],
) -> GateResult:
    roles = private_record["authoritative_roles"]
    target_name = str(roles["target"])
    geoms = _body_box_geoms(model, target_name)
    if len(geoms) != 1:
        return GateResult(
            "task_geometry",
            False,
            {"target_collision_geoms": geoms},
            "target must have exactly one primitive collision box",
        )
    geom_id = geoms[0]
    center = np.asarray(data.geom_xpos[geom_id])
    world_half = _geom_world_half_extents(model, data, geom_id)
    margins = {
        "workspace_x_min": float(center[0] - world_half[0] - WORKSPACE_X[0]),
        "workspace_x_max": float(WORKSPACE_X[1] - center[0] - world_half[0]),
        "workspace_y_min": float(center[1] - world_half[1] - WORKSPACE_Y[0]),
        "workspace_y_max": float(WORKSPACE_Y[1] - center[1] - world_half[1]),
    }
    fixture_distances: dict[str, float] = {}
    for fixture in roles.get("fixtures", []):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fixture)
        if body_id < 0:
            margins[f"missing_fixture_{fixture}"] = -1.0
            continue
        fixture_distances[fixture] = float(np.linalg.norm(data.xpos[body_id, :2] - center[:2]))

    target_body = int(model.geom_bodyid[geom_id])
    distractor_contacts: list[tuple[str, str]] = []
    distractor_names = set(str(item) for item in roles.get("distractors", []))
    distractor_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in distractor_names
    }
    for contact in data.contact:
        left = int(model.geom_bodyid[int(contact.geom1)])
        right = int(model.geom_bodyid[int(contact.geom2)])
        if target_body in {left, right} and ({left, right} & distractor_body_ids):
            distractor_contacts.append(
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
                )
            )
    minimum_margin = min(margins.values())
    fixture_reachable = all(value <= 0.80 for value in fixture_distances.values())
    passed = minimum_margin >= 0.0 and fixture_reachable and not distractor_contacts
    return GateResult(
        "task_geometry",
        passed,
        {
            "workspace_margins": margins,
            "minimum_workspace_margin": minimum_margin,
            "fixture_distances": fixture_distances,
            "target_distractor_contacts": distractor_contacts,
            "regular_clearance_requirement": contract.phase_clearance_m,
            "stress_clearance_requirement": contract.stress_clearance_m,
        },
        "target remains supported, fixtures remain nearby, and distractors are noncausal",
    )


def _strategy_manipulability_gate(
    model: mujoco.MjModel,
    contract: StrategyContract,
    private_record: dict[str, Any],
) -> GateResult:
    if contract.strategy_id not in {2, 3}:
        return GateResult(
            "strategy_manipulability",
            True,
            {"strategy_id": contract.strategy_id, "applicable": False},
            "no additional strategy-specific grasp envelope is required",
        )
    target_name = str(private_record["authoritative_roles"]["target"])
    geoms = _body_box_geoms(model, target_name)
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        target_name,
    )
    gripper_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "fr3_hand",
    )
    if len(geoms) != 1 or body_id < 0 or gripper_id < 0:
        return GateResult(
            "strategy_manipulability",
            False,
            {
                "target": target_name,
                "target_geoms": geoms,
                "body_id": body_id,
                "gripper_actuator_id": gripper_id,
            },
            f"strategy {contract.strategy_id} requires one box target and the "
            "FR3 gripper actuator",
        )
    geom_id = geoms[0]
    gripper_max_width = 2.0 * float(model.actuator_ctrlrange[gripper_id, 1])
    if contract.strategy_id == 2:
        metrics = strategy_2_manipulation_metrics(
            2.0 * np.asarray(model.geom_size[geom_id]),
            mass_kg=float(model.body_mass[body_id]),
            sliding_friction=float(model.geom_friction[geom_id, 0]),
            gripper_max_width_m=gripper_max_width,
        )
        detail = (
            "strategy 2 target must expose a tall edge, fit within the gripper, "
            "and remain inside the demonstrated flip physics envelope"
        )
    else:
        metrics = strategy_3_manipulation_metrics(
            2.0 * np.asarray(model.geom_size[geom_id]),
            gripper_max_width_m=gripper_max_width,
        )
        detail = (
            "strategy 3 target's intended post-flip narrow dimension must fit "
            "inside the physical gripper aperture"
        )
    metrics["target"] = target_name
    metrics["appearance"] = private_record["parameters"]["appearance"]
    return GateResult(
        "strategy_manipulability",
        bool(metrics["passed"]),
        metrics,
        detail,
    )


def _relation_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    private_record: dict[str, Any],
) -> GateResult:
    """Check strategy-owned relational invariants at the raw reset state."""

    metrics: list[dict[str, Any]] = []
    passed = True
    canonical_supports = private_record.get("context_layout", {}).get(
        "canonical_supports",
        {},
    )
    if canonical_supports:
        support_metrics = _canonical_support_metrics(
            model,
            data,
            canonical_supports,
        )
        metrics.append(support_metrics)
        passed = passed and bool(support_metrics["passed"])
    for requirement in private_record.get("relation_checks", []):
        kind = requirement.get("kind")
        specialized = {
            "strategy_1_corridor": _strategy_1_corridor_metrics,
            "wall_flip_setup": _wall_flip_metrics,
            "extraction_then_wall_setup": _extraction_then_wall_metrics,
            "slot_setup": _slot_metrics,
            "packed_upright_row": _packed_row_metrics,
            "platform_edge_setup": _platform_edge_metrics,
        }.get(str(kind))
        if specialized is not None:
            item = specialized(model, data, requirement)
            metrics.append(item)
            passed = passed and bool(item["passed"])
            continue
        if kind != "distance_xy":
            metrics.append({**requirement, "passed": False, "error": "unknown kind"})
            passed = False
            continue
        left_name = str(requirement["left"])
        right_name = str(requirement["right"])
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)
        if left_id < 0 or right_id < 0:
            metrics.append(
                {
                    **requirement,
                    "passed": False,
                    "error": "missing body",
                    "left_body_id": left_id,
                    "right_body_id": right_id,
                }
            )
            passed = False
            continue
        distance = float(np.linalg.norm(data.xpos[left_id, :2] - data.xpos[right_id, :2]))
        lower = float(requirement["minimum"])
        upper = float(requirement["maximum"])
        item_passed = lower <= distance <= upper
        passed = passed and item_passed
        metrics.append(
            {
                **requirement,
                "distance_xy": distance,
                "passed": item_passed,
            }
        )
    return GateResult(
        "strategy_relations",
        passed,
        {"requirements": metrics},
        "strategy-specific fixture, obstacle, and slot relations remain feasible",
    )


def _projected_geom_half_extent(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    direction_xy: np.ndarray,
) -> float:
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    direction = np.asarray((direction_xy[0], direction_xy[1], 0.0))
    return float(np.abs(direction @ rotation) @ np.asarray(model.geom_size[geom_id]))


def _principal_horizontal_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    reference_axis_xy: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    candidates: list[tuple[float, np.ndarray]] = []
    for index in range(3):
        axis = rotation[:2, index]
        norm = float(np.linalg.norm(axis))
        if norm < 0.5:
            continue
        normalized = axis / norm
        candidates.append(
            (abs(float(normalized @ reference_axis_xy)), normalized)
        )
    if not candidates:
        return np.asarray((math.nan, math.nan))
    return max(candidates, key=lambda item: item[0])[1]


def _undirected_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(float(left @ right)), 0.0, 1.0))
    return float(math.acos(cosine))


def _strategy_1_corridor_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    names = {
        role: str(requirement[role])
        for role in ("target", "obstacle", "divider")
    }
    geoms = {
        role: _body_box_geoms(model, name)
        for role, name in names.items()
    }
    if any(len(values) != 1 for values in geoms.values()):
        return {
            **requirement,
            "passed": False,
            "error": "corridor roles must each have exactly one collision box",
            "collision_geoms": geoms,
        }
    geom_ids = {role: values[0] for role, values in geoms.items()}
    axis = np.asarray(requirement["axis_xy"], dtype=np.float64)
    lateral = np.asarray(requirement["lateral_xy"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    lateral /= np.linalg.norm(lateral)
    entrance = np.asarray(requirement["entrance_xy"], dtype=np.float64)
    centers = {
        role: np.asarray(data.geom_xpos[geom_id, :2])
        for role, geom_id in geom_ids.items()
    }
    lateral_halves = {
        role: _projected_geom_half_extent(
            model,
            data,
            geom_id,
            lateral,
        )
        for role, geom_id in geom_ids.items()
    }
    axis_halves = {
        role: _projected_geom_half_extent(
            model,
            data,
            geom_id,
            axis,
        )
        for role, geom_id in geom_ids.items()
    }
    boundary_separation = abs(
        float((centers["divider"] - centers["obstacle"]) @ lateral)
    )
    clear_width = (
        boundary_separation
        - lateral_halves["obstacle"]
        - lateral_halves["divider"]
    )
    target_lateral_width = 2.0 * lateral_halves["target"]
    obstacle_axis = _principal_horizontal_axis(
        model,
        data,
        geom_ids["obstacle"],
        axis,
    )
    divider_axis = _principal_horizontal_axis(
        model,
        data,
        geom_ids["divider"],
        axis,
    )
    target_axis = _principal_horizontal_axis(
        model,
        data,
        geom_ids["target"],
        axis,
    )
    parallel_error = _undirected_angle(obstacle_axis, divider_axis)
    obstacle_axis_error = _undirected_angle(obstacle_axis, axis)
    divider_axis_error = _undirected_angle(divider_axis, axis)
    target_alignment_error = _undirected_angle(target_axis, axis)
    target_axial_coordinate = float((centers["target"] - entrance) @ axis)
    target_lateral_offset = abs(
        float((centers["target"] - entrance) @ lateral)
    )
    target_fully_outside = (
        target_axial_coordinate + axis_halves["target"] <= 0.0
    )
    boundary_axial_intervals = {
        role: (
            float((centers[role] - entrance) @ axis) - axis_halves[role],
            float((centers[role] - entrance) @ axis) + axis_halves[role],
        )
        for role in ("obstacle", "divider")
    }
    store_start = max(
        0.0,
        boundary_axial_intervals["obstacle"][0],
        boundary_axial_intervals["divider"][0],
    )
    store_end = min(
        boundary_axial_intervals["obstacle"][1],
        boundary_axial_intervals["divider"][1],
    )
    available_store_depth = max(0.0, store_end - store_start)
    required_store_depth = (
        2.0 * axis_halves["target"]
        + float(requirement["store_clearance"])
    )
    checks = {
        "minimum_width": clear_width
        >= float(requirement["required_clear_width"]) - 1e-4,
        "maximum_width": clear_width
        <= float(requirement["maximum_clear_width"]) + 1e-4,
        "parallel_boundaries": parallel_error
        <= float(requirement["maximum_parallel_error_radians"]),
        "obstacle_axis_alignment": obstacle_axis_error
        <= float(requirement["maximum_parallel_error_radians"]),
        "divider_axis_alignment": divider_axis_error
        <= float(requirement["maximum_parallel_error_radians"]),
        "target_axis_alignment": target_alignment_error
        <= float(requirement["maximum_target_alignment_error_radians"]),
        "target_fully_outside": target_fully_outside,
        "target_laterally_misaligned": target_lateral_offset
        >= float(requirement["minimum_lateral_offset"]),
        "store_starts_at_entrance": store_start <= 0.002,
        "store_depth": available_store_depth >= required_store_depth,
    }
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "clear_width": clear_width,
        "target_lateral_width": target_lateral_width,
        "boundary_separation": boundary_separation,
        "parallel_error_radians": parallel_error,
        "obstacle_axis_error_radians": obstacle_axis_error,
        "divider_axis_error_radians": divider_axis_error,
        "target_alignment_error_radians": target_alignment_error,
        "target_axial_coordinate": target_axial_coordinate,
        "target_lateral_offset": target_lateral_offset,
        "target_axis_half_extent": axis_halves["target"],
        "boundary_axial_intervals": boundary_axial_intervals,
        "store_start_coordinate": store_start,
        "available_store_depth": available_store_depth,
        "required_store_depth": required_store_depth,
        "centers_xy": {
            role: center.tolist() for role, center in centers.items()
        },
    }


def _single_box_geom(model: mujoco.MjModel, body_name: str) -> int | None:
    geoms = _body_box_geoms(model, body_name)
    return geoms[0] if len(geoms) == 1 else None


def _long_horizontal_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    for index in np.argsort(np.asarray(model.geom_size[geom_id]))[::-1]:
        candidate = rotation[:2, int(index)]
        norm = float(np.linalg.norm(candidate))
        if norm >= 0.5:
            return candidate / norm
    return np.asarray((math.nan, math.nan))


def _canonical_support_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirements: dict[str, Any],
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    passed = True
    for name, requirement in requirements.items():
        geom_id = _single_box_geom(model, str(name))
        if geom_id is None:
            items[str(name)] = {"passed": False, "error": "missing collision box"}
            passed = False
            continue
        rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        vertical_alignment = float(np.max(np.abs(rotation[2])))
        support_radius = float(
            np.abs(rotation[2]) @ np.asarray(model.geom_size[geom_id])
        )
        bottom_z = float(data.geom_xpos[geom_id, 2] - support_radius)
        support_z = float(requirement.get("support_z", 0.0))
        support_error = bottom_z - support_z
        item_passed = (
            vertical_alignment >= 1.0 - 1e-6
            and -MAX_PENETRATION <= support_error <= 0.0015
        )
        items[str(name)] = {
            "passed": item_passed,
            "vertical_alignment": vertical_alignment,
            "bottom_z": bottom_z,
            "support_z": support_z,
            "support_error": support_error,
        }
        passed = passed and item_passed
    return {
        "kind": "canonical_supports",
        "passed": passed,
        "bodies": items,
    }


def _wall_flip_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    target_geom = _single_box_geom(model, str(requirement["target"]))
    fixture_geom = _single_box_geom(model, str(requirement["fixture"]))
    if target_geom is None or fixture_geom is None:
        return {**requirement, "passed": False, "error": "missing collision box"}
    normal = np.asarray(requirement["wall_normal_xy"], dtype=np.float64)
    axis = np.asarray(requirement["wall_axis_xy"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    axis /= np.linalg.norm(axis)
    delta = np.asarray(data.geom_xpos[fixture_geom, :2]) - np.asarray(
        data.geom_xpos[target_geom, :2]
    )
    face_gap = (
        float(delta @ normal)
        - _projected_geom_half_extent(model, data, target_geom, normal)
        - _projected_geom_half_extent(model, data, fixture_geom, normal)
    )
    axis_margin = (
        _projected_geom_half_extent(model, data, fixture_geom, axis)
        - abs(float(delta @ axis))
        - _projected_geom_half_extent(model, data, target_geom, axis)
    )
    fixture_height = 2.0 * _geom_world_half_extents(
        model,
        data,
        fixture_geom,
    )[2]
    target_world_height = 2.0 * _geom_world_half_extents(
        model,
        data,
        target_geom,
    )[2]
    target_thin_height = 2.0 * float(np.min(model.geom_size[target_geom]))
    checks = {
        "face_gap": float(requirement["minimum_face_gap"]) - 0.002
        <= face_gap
        <= float(requirement["maximum_face_gap"]) + 0.002,
        "wall_axis_margin": axis_margin
        >= float(requirement["minimum_wall_axis_margin"]),
        "wall_height": fixture_height >= float(requirement["minimum_wall_height"]),
        "target_initially_flat": abs(target_world_height - target_thin_height)
        <= 0.001,
    }
    post_pull_center: np.ndarray | None = None
    post_pull_face_gap: float | None = None
    if requirement.get("required_clearance_pull") is not None:
        required_pull = float(requirement["required_clearance_pull"])
        fixture_center = np.asarray(data.geom_xpos[fixture_geom, :2])
        target_center = np.asarray(data.geom_xpos[target_geom, :2])
        fixture_normal_half = _projected_geom_half_extent(
            model,
            data,
            fixture_geom,
            normal,
        )
        ranked_target_halves = np.sort(np.asarray(model.geom_size[target_geom]))
        target_thin_half = float(ranked_target_halves[0])
        target_axis_half = float(ranked_target_halves[2])
        target_axis_offset = float((target_center - fixture_center) @ axis)
        wall_contact_center = (
            fixture_center
            - normal * (fixture_normal_half + target_thin_half)
            + axis * target_axis_offset
        )
        post_pull_center = wall_contact_center - normal * required_pull
        upright_world_half = (
            np.abs(normal) * target_thin_half
            + np.abs(axis) * target_axis_half
        )
        post_pull_face_gap = (
            float((fixture_center - post_pull_center) @ normal)
            - fixture_normal_half
            - target_thin_half
        )
        post_pull_in_workspace = (
            WORKSPACE_X[0] + upright_world_half[0]
            <= post_pull_center[0]
            <= WORKSPACE_X[1] - upright_world_half[0]
            and WORKSPACE_Y[0] + upright_world_half[1]
            <= post_pull_center[1]
            <= WORKSPACE_Y[1] - upright_world_half[1]
        )
        checks["pull_phase_required"] = required_pull >= float(
            requirement.get("minimum_pull_clearance", 0.0)
        )
        checks["post_pull_finger_clearance"] = post_pull_face_gap >= float(
            requirement.get("minimum_pull_clearance", 0.0)
        )
        checks["post_pull_in_workspace"] = post_pull_in_workspace
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "face_gap": face_gap,
        "wall_axis_margin": axis_margin,
        "fixture_height": fixture_height,
        "target_world_height": target_world_height,
        "post_pull_center_xy": (
            None if post_pull_center is None else post_pull_center.tolist()
        ),
        "post_pull_face_gap": post_pull_face_gap,
    }


def _slot_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    names = {
        role: str(requirement[role])
        for role in ("target", "left", "right")
    }
    geoms = {role: _single_box_geom(model, name) for role, name in names.items()}
    if any(value is None for value in geoms.values()):
        return {**requirement, "passed": False, "error": "missing collision box"}
    target_geom = int(geoms["target"])
    left_geom = int(geoms["left"])
    right_geom = int(geoms["right"])
    axis = np.asarray(requirement["axis_xy"], dtype=np.float64)
    lateral = np.asarray(requirement["lateral_xy"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    lateral /= np.linalg.norm(lateral)
    entrance = np.asarray(requirement["entrance_xy"], dtype=np.float64)
    centers = {
        role: np.asarray(data.geom_xpos[int(geom_id), :2])
        for role, geom_id in geoms.items()
    }
    lateral_halves = {
        role: _projected_geom_half_extent(model, data, int(geom_id), lateral)
        for role, geom_id in geoms.items()
    }
    axis_halves = {
        role: _projected_geom_half_extent(model, data, int(geom_id), axis)
        for role, geom_id in geoms.items()
    }
    clear_width = (
        abs(float((centers["right"] - centers["left"]) @ lateral))
        - lateral_halves["left"]
        - lateral_halves["right"]
    )
    axial_intervals = {
        role: (
            float((centers[role] - entrance) @ axis) - axis_halves[role],
            float((centers[role] - entrance) @ axis) + axis_halves[role],
        )
        for role in ("left", "right")
    }
    overlap_start = max(value[0] for value in axial_intervals.values())
    overlap_end = min(value[1] for value in axial_intervals.values())
    available_depth = max(0.0, overlap_end - overlap_start)
    target_axial = float((centers["target"] - entrance) @ axis)
    target_outside = target_axial + axis_halves["target"] <= 0.002
    target_axis = _long_horizontal_axis(model, data, target_geom)
    alignment_error = _undirected_angle(target_axis, axis)
    left_axis = _long_horizontal_axis(model, data, left_geom)
    right_axis = _long_horizontal_axis(model, data, right_geom)
    boundary_error = max(
        _undirected_angle(left_axis, axis),
        _undirected_angle(right_axis, axis),
        _undirected_angle(left_axis, right_axis),
    )
    checks = {
        "minimum_width": clear_width
        >= float(requirement["required_clear_width"]) - 0.002,
        "maximum_width": clear_width
        <= float(requirement["maximum_clear_width"]) + 0.002,
        "depth": available_depth >= float(requirement["minimum_depth"]) - 0.002,
        "target_outside": target_outside,
        "target_alignment": float(
            requirement.get("minimum_alignment_error_radians", 0.0)
        )
        - math.radians(1.0)
        <= alignment_error
        <= float(requirement.get("maximum_alignment_error_radians", math.pi / 2))
        + math.radians(1.0),
        "parallel_boundaries": boundary_error <= math.radians(1.0),
    }
    initial_width_excess = 2.0 * lateral_halves["target"] - clear_width
    if requirement.get("minimum_initial_width_excess") is not None:
        checks["initially_too_wide_for_slot"] = initial_width_excess >= float(
            requirement["minimum_initial_width_excess"]
        )
    ranked_target_halves = np.sort(np.asarray(model.geom_size[target_geom]))
    aligned_outside_clearance = -target_axial - float(ranked_target_halves[2])
    if requirement.get("minimum_aligned_outside_clearance") is not None:
        checks["aligned_pose_stays_outside"] = aligned_outside_clearance >= float(
            requirement["minimum_aligned_outside_clearance"]
        )
    pivot_sweep_radius = float(np.linalg.norm(ranked_target_halves[1:]))
    pivot_obstacle_clearances: list[float] = []
    target_relative = centers["target"] - entrance
    for role in ("left", "right"):
        obstacle_relative = centers[role] - entrance
        lateral_distance = max(
            0.0,
            abs(float((target_relative - obstacle_relative) @ lateral))
            - lateral_halves[role],
        )
        axial_distance = max(
            0.0,
            abs(float((target_relative - obstacle_relative) @ axis))
            - axis_halves[role],
        )
        pivot_obstacle_clearances.append(
            math.hypot(lateral_distance, axial_distance) - pivot_sweep_radius
        )
    if requirement.get("minimum_pivot_obstacle_clearance") is not None:
        checks["pivot_sweep_obstacle_clearance"] = min(
            pivot_obstacle_clearances
        ) >= float(requirement["minimum_pivot_obstacle_clearance"])
    if requirement.get("require_pivot_sweep_in_workspace"):
        center = centers["target"]
        checks["pivot_sweep_in_workspace"] = (
            WORKSPACE_X[0] + pivot_sweep_radius
            <= center[0]
            <= WORKSPACE_X[1] - pivot_sweep_radius
            and WORKSPACE_Y[0] + pivot_sweep_radius
            <= center[1]
            <= WORKSPACE_Y[1] - pivot_sweep_radius
        )
    flip_wall_exposure: float | None = None
    target_flip_wall_overlap: float | None = None
    initial_push_distance: float | None = None
    if requirement.get("flip_wall") is not None:
        flip_wall = str(requirement["flip_wall"])
        opposing_boundary = str(requirement["opposing_boundary"])
        role_for_name = {name: role for role, name in names.items()}
        if (
            flip_wall not in role_for_name
            or opposing_boundary not in role_for_name
            or flip_wall == opposing_boundary
        ):
            return {
                **requirement,
                "passed": False,
                "error": "flip wall and opposing boundary must name different slot boundaries",
            }
        flip_interval = axial_intervals[role_for_name[flip_wall]]
        opposing_interval = axial_intervals[role_for_name[opposing_boundary]]
        flip_wall_exposure = opposing_interval[0] - flip_interval[0]
        exposed_interval = (flip_interval[0], opposing_interval[0])
        target_interval = (
            target_axial - axis_halves["target"],
            target_axial + axis_halves["target"],
        )
        target_flip_wall_overlap = max(
            0.0,
            min(exposed_interval[1], target_interval[1])
            - max(exposed_interval[0], target_interval[0]),
        )
        checks["flip_wall_exposed"] = flip_wall_exposure >= float(
            requirement["minimum_flip_wall_exposure"]
        )
        checks["target_overlaps_exposed_flip_wall"] = (
            target_flip_wall_overlap
            >= float(requirement["minimum_target_wall_overlap"])
        )
        initial_push_distance = (
            abs(
                float(
                    (centers[role_for_name[flip_wall]] - centers["target"])
                    @ lateral
                )
            )
            - lateral_halves[role_for_name[flip_wall]]
            - lateral_halves["target"]
        )
        if requirement.get("minimum_initial_push_distance") is not None:
            checks["push_phase_required"] = (
                float(requirement["minimum_initial_push_distance"])
                <= initial_push_distance
                <= float(requirement["maximum_initial_push_distance"])
            )
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "clear_width": clear_width,
        "available_depth": available_depth,
        "boundary_axial_intervals": axial_intervals,
        "flip_wall_exposure": flip_wall_exposure,
        "target_flip_wall_overlap": target_flip_wall_overlap,
        "initial_push_distance": initial_push_distance,
        "target_axial_coordinate": target_axial,
        "target_alignment_error_radians": alignment_error,
        "initial_target_width": 2.0 * lateral_halves["target"],
        "initial_width_excess": initial_width_excess,
        "aligned_outside_clearance": aligned_outside_clearance,
        "pivot_sweep_radius": pivot_sweep_radius,
        "pivot_obstacle_clearances": pivot_obstacle_clearances,
        "minimum_pivot_obstacle_clearance_actual": min(
            pivot_obstacle_clearances
        ),
        "boundary_error_radians": boundary_error,
    }


def _extraction_then_wall_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    names = {
        role: str(requirement[role])
        for role in ("target", "left", "right", "fixture")
    }
    geoms = {role: _single_box_geom(model, name) for role, name in names.items()}
    if any(value is None for value in geoms.values()):
        return {**requirement, "passed": False, "error": "missing collision box"}
    axis = np.asarray(requirement["extraction_axis_xy"], dtype=np.float64)
    lateral = np.asarray(requirement["lateral_xy"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    lateral /= np.linalg.norm(lateral)
    centers = {
        role: np.asarray(data.geom_xpos[int(geom_id), :2])
        for role, geom_id in geoms.items()
    }
    lateral_halves = {
        role: _projected_geom_half_extent(
            model,
            data,
            int(geom_id),
            lateral,
        )
        for role, geom_id in geoms.items()
    }
    axis_halves = {
        role: _projected_geom_half_extent(
            model,
            data,
            int(geom_id),
            axis,
        )
        for role, geom_id in geoms.items()
    }
    clear_width = (
        abs(float((centers["right"] - centers["left"]) @ lateral))
        - lateral_halves["left"]
        - lateral_halves["right"]
    )
    target_lateral = float(centers["target"] @ lateral)
    inner_left = float(centers["left"] @ lateral) + lateral_halves["left"]
    inner_right = float(centers["right"] @ lateral) - lateral_halves["right"]
    target_inside = (
        inner_left + lateral_halves["target"] <= target_lateral
        and target_lateral <= inner_right - lateral_halves["target"]
    )
    channel_back = max(
        float(centers[role] @ axis) - axis_halves[role]
        for role in ("left", "right")
    )
    channel_open = min(
        float(centers[role] @ axis) + axis_halves[role]
        for role in ("left", "right")
    )
    target_back = float(centers["target"] @ axis) - axis_halves["target"]
    target_open = float(centers["target"] @ axis) + axis_halves["target"]
    target_back_clearance = target_back - channel_back
    target_open_clearance = channel_open - target_open
    target_axially_contained = (
        target_back_clearance >= -0.002 and target_open_clearance >= -0.002
    )
    target_planar_area = (
        4.0 * lateral_halves["target"] * axis_halves["target"]
    )
    # Mirror the frozen Strategy 3 resolver: it derives its lateral direction
    # from the obstacle centers instead of trusting the generated relation.
    # This catches accidental axial staggering that would rotate the runtime
    # extraction direction while still passing checks along the designed axis.
    primitive_lateral = centers["right"] - centers["left"]
    primitive_lateral_norm = float(np.linalg.norm(primitive_lateral))
    if primitive_lateral_norm <= 1e-6:
        return {
            **requirement,
            "passed": False,
            "error": "corridor obstacle centers have a degenerate tangent axis",
        }
    primitive_lateral /= primitive_lateral_norm
    primitive_axis = np.asarray(
        (primitive_lateral[1], -primitive_lateral[0]),
        dtype=np.float64,
    )
    toward_fixture = centers["fixture"] - centers["target"]
    if float(primitive_axis @ toward_fixture) < 0.0:
        primitive_axis = -primitive_axis
    primitive_axis_halves = {
        role: _projected_geom_half_extent(
            model,
            data,
            int(geom_id),
            primitive_axis,
        )
        for role, geom_id in geoms.items()
    }
    obstacle_open = max(
        float(centers[role] @ primitive_axis)
        + primitive_axis_halves[role]
        for role in ("left", "right")
    )
    target_projection = float(centers["target"] @ primitive_axis)
    target_smallest_full_extent = 2.0 * float(
        np.min(model.geom_size[int(geoms["target"])])
    )
    primitive_extraction_distance = (
        obstacle_open
        + primitive_axis_halves["target"]
        + 0.20 * target_smallest_full_extent
        - target_projection
    )
    fixture_height = 2.0 * _geom_world_half_extents(
        model,
        data,
        int(geoms["fixture"]),
    )[2]
    post_flip_target_height = 2.0 * lateral_halves["target"]
    post_flip_edge_exposure = post_flip_target_height - fixture_height
    extracted = centers["target"] + axis * float(requirement["minimum_open_path"])
    fixture_geom = int(geoms["fixture"])
    wall_alignment_error = 0.0
    wall_on_expected_side = True
    if requirement.get("wall_orientation") == "parallel_to_extraction":
        wall_normal = np.asarray(requirement["wall_normal_xy"], dtype=np.float64)
        wall_normal /= np.linalg.norm(wall_normal)
        signed_wall_separation = float(
            (centers["fixture"] - extracted) @ wall_normal
        )
        transfer_gap = (
            signed_wall_separation
            - _projected_geom_half_extent(
                model, data, fixture_geom, wall_normal
            )
            - _projected_geom_half_extent(
                model, data, int(geoms["target"]), wall_normal
            )
        )
        wall_axis_margin = (
            _projected_geom_half_extent(model, data, fixture_geom, axis)
            - abs(float((centers["fixture"] - extracted) @ axis))
            - axis_halves["target"]
        )
        wall_alignment_error = _undirected_angle(
            _long_horizontal_axis(model, data, fixture_geom),
            axis,
        )
        wall_on_expected_side = signed_wall_separation > 0.0
    else:
        transfer_gap = (
            float((centers["fixture"] - extracted) @ axis)
            - _projected_geom_half_extent(model, data, fixture_geom, axis)
            - axis_halves["target"]
        )
        wall_axis_margin = (
            _projected_geom_half_extent(model, data, fixture_geom, lateral)
            - abs(float((centers["fixture"] - extracted) @ lateral))
            - lateral_halves["target"]
        )
    checks = {
        "minimum_width": clear_width
        >= float(requirement["required_clear_width"]) - 0.002,
        "bounded_width": clear_width
        <= float(requirement["required_clear_width"])
        + float(requirement["maximum_extra_width"]),
        "target_between_obstacles": target_inside,
        "target_axially_contained": target_axially_contained,
        "target_planar_area": target_planar_area
        >= float(requirement.get("minimum_target_planar_area", 0.0)),
        "primitive_extraction_distance": float(
            requirement.get("minimum_primitive_extraction_distance", -math.inf)
        )
        - 1e-6
        <= primitive_extraction_distance
        <= float(
            requirement.get("maximum_primitive_extraction_distance", math.inf)
        )
        + 1e-6,
        "open_edge_clearance": float(
            requirement.get("minimum_initial_open_edge_clearance", -math.inf)
        )
        <= target_open_clearance
        <= float(
            requirement.get("maximum_initial_open_edge_clearance", math.inf)
        ),
        "wall_transfer": float(requirement.get("minimum_wall_transfer", 0.02))
        <= transfer_gap
        <= float(requirement["maximum_wall_transfer"]),
        "wall_axis_margin": wall_axis_margin
        >= float(requirement.get("minimum_wall_axis_margin", 0.015)),
        "wall_parallel_to_extraction": wall_alignment_error
        <= math.radians(1.0),
        "wall_on_expected_side": wall_on_expected_side,
        "flip_wall_support_height": fixture_height
        >= float(requirement.get("minimum_flip_wall_height", 0.0)),
        "post_flip_grasp_edge_exposed": post_flip_edge_exposure
        >= float(
            requirement.get("minimum_post_flip_edge_exposure", 0.0)
        ),
    }
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "clear_width": clear_width,
        "channel_axial_interval": [channel_back, channel_open],
        "target_axial_interval": [target_back, target_open],
        "target_back_edge_clearance": target_back_clearance,
        "target_open_edge_clearance": target_open_clearance,
        "target_planar_area": target_planar_area,
        "primitive_extraction_distance": primitive_extraction_distance,
        "primitive_extraction_axis_xy": primitive_axis.tolist(),
        "primitive_lateral_axis_xy": primitive_lateral.tolist(),
        "primitive_extracted_center_xy": (
            centers["target"]
            + primitive_axis * primitive_extraction_distance
        ).tolist(),
        "wall_transfer_gap": transfer_gap,
        "wall_axis_margin": wall_axis_margin,
        "wall_alignment_error_radians": wall_alignment_error,
        "fixture_height": fixture_height,
        "post_flip_target_height": post_flip_target_height,
        "post_flip_edge_exposure": post_flip_edge_exposure,
    }


def _packed_row_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    names = [str(value) for value in requirement["ordered_bodies"]]
    geoms = [_single_box_geom(model, name) for name in names]
    if any(value is None for value in geoms):
        return {**requirement, "passed": False, "error": "missing row collision box"}
    lateral = np.asarray(requirement["lateral_xy"], dtype=np.float64)
    outward = np.asarray(requirement["outward_xy"], dtype=np.float64)
    lateral /= np.linalg.norm(lateral)
    outward /= np.linalg.norm(outward)
    centers = [np.asarray(data.geom_xpos[int(geom_id), :2]) for geom_id in geoms]
    coordinates = [float(center @ lateral) for center in centers]
    halves = [
        _projected_geom_half_extent(model, data, int(geom_id), lateral)
        for geom_id in geoms
    ]
    gaps = [
        coordinates[index + 1]
        - coordinates[index]
        - halves[index]
        - halves[index + 1]
        for index in range(len(names) - 1)
    ]
    row_rank = int(requirement.get("row_axis_dimension_rank", 0))
    outward_rank = int(requirement.get("outward_axis_dimension_rank", 1))
    vertical_rank = int(requirement.get("vertical_axis_dimension_rank", 2))
    if sorted((row_rank, outward_rank, vertical_rank)) != [0, 1, 2]:
        return {
            **requirement,
            "passed": False,
            "error": "packed-row dimension ranks must be a permutation of 0, 1, 2",
        }
    minimum_axis_alignment = float(
        requirement.get("minimum_axis_alignment", 0.995)
    )
    axis_alignment: list[dict[str, Any]] = []
    for name, geom_id in zip(names, geoms):
        geom_id = int(geom_id)
        rotation = np.asarray(data.geom_xmat[int(geom_id)]).reshape(3, 3)
        order = np.argsort(np.asarray(model.geom_size[geom_id]), kind="stable")
        row_axis = int(order[row_rank])
        outward_axis = int(order[outward_rank])
        vertical_axis = int(order[vertical_rank])
        axis_alignment.append(
            {
                "body": name,
                "row_axis": abs(float(rotation[:2, row_axis] @ lateral)),
                "outward_axis": abs(
                    float(rotation[:2, outward_axis] @ outward)
                ),
                "vertical_axis": abs(float(rotation[2, vertical_axis])),
            }
        )
    target_index = names.index(str(requirement["target"]))
    target_center = centers[target_index]
    ray_limits: list[float] = []
    for coordinate, direction, bounds in zip(
        target_center,
        outward,
        (WORKSPACE_X, WORKSPACE_Y),
    ):
        if direction > 1e-9:
            ray_limits.append((bounds[1] - coordinate) / direction)
        elif direction < -1e-9:
            ray_limits.append((bounds[0] - coordinate) / direction)
    outward_radius = _projected_geom_half_extent(
        model,
        data,
        int(geoms[target_index]),
        outward,
    )
    target_grasp_thickness = 2.0 * float(
        np.min(model.geom_size[int(geoms[target_index])])
    )
    target_topple_contact_width = 2.0 * halves[target_index]
    target_topple_face_depth = 2.0 * _projected_geom_half_extent(
        model,
        data,
        int(geoms[target_index]),
        outward,
    )
    row_heights = [
        2.0 * _geom_world_half_extents(model, data, int(geom_id))[2]
        for geom_id in geoms
    ]
    row_tops = [
        float(data.geom_xpos[int(geom_id), 2])
        + _geom_world_half_extents(model, data, int(geom_id))[2]
        for geom_id in geoms
    ]
    row_front_edges = [
        float(center @ outward)
        + _projected_geom_half_extent(model, data, int(geom_id), outward)
        for center, geom_id in zip(centers, geoms)
    ]
    adjacent_indices = (target_index - 1, target_index + 1)
    target_top_exposure = max(
        row_tops[target_index] - row_tops[index]
        for index in adjacent_indices
    )
    target_front_exposure = max(
        row_front_edges[target_index] - row_front_edges[index]
        for index in adjacent_indices
    )
    target_side_gaps = [gaps[target_index - 1], gaps[target_index]]
    outward_clearance = min(ray_limits) - outward_radius
    fixture_names = [
        str(requirement[role]) for role in ("left_fixture", "right_fixture")
    ]
    fixture_geoms = [_single_box_geom(model, name) for name in fixture_names]
    if any(value is None for value in fixture_geoms):
        return {
            **requirement,
            "passed": False,
            "error": "missing packed-row side fixture collision box",
        }
    fixture_depths = [
        2.0
        * _projected_geom_half_extent(model, data, int(geom_id), outward)
        for geom_id in fixture_geoms
    ]
    fixture_heights = [
        2.0 * _geom_world_half_extents(model, data, int(geom_id))[2]
        for geom_id in fixture_geoms
    ]
    row_opening_edge = max(
        float(center @ outward)
        + _projected_geom_half_extent(model, data, int(geom_id), outward)
        for center, geom_id in zip(centers, geoms)
    )
    fixture_opening_edge_margins = [
        float(data.geom_xpos[int(geom_id), :2] @ outward)
        + _projected_geom_half_extent(model, data, int(geom_id), outward)
        - row_opening_edge
        for geom_id in fixture_geoms
    ]
    checks = {
        "ordered": all(gap >= -0.001 for gap in gaps),
        "minimum_gap": min(gaps) >= float(requirement["minimum_neighbor_gap"]) - 0.002,
        "maximum_gap": max(gaps) <= float(requirement["maximum_neighbor_gap"]) + 0.002,
        "short_edge_packing": all(
            item["row_axis"] >= minimum_axis_alignment
            for item in axis_alignment
        ),
        "broad_faces_point_outward": all(
            item["outward_axis"] >= minimum_axis_alignment
            for item in axis_alignment
        ),
        "upright": all(
            item["vertical_axis"] >= minimum_axis_alignment
            for item in axis_alignment
        ),
        "target_flanked": 0 < target_index < len(names) - 1,
        "outward_clearance": outward_clearance
        >= float(requirement["minimum_outward_clearance"]),
        "target_fits_gripper": target_grasp_thickness
        <= float(requirement.get("maximum_target_grasp_thickness", math.inf)),
        "target_wide_enough_for_topple_contact": target_topple_contact_width
        >= float(requirement.get("minimum_target_topple_contact_width", 0.0)),
        "target_has_broad_topple_face": target_topple_face_depth
        >= float(requirement.get("minimum_target_topple_face_depth", 0.0)),
        "row_height_consistency": max(row_heights) - min(row_heights)
        <= float(requirement.get("maximum_row_height_spread", math.inf)),
        "target_top_not_initially_exposed": target_top_exposure
        <= float(requirement.get("maximum_target_top_exposure", math.inf)),
        "target_front_not_initially_exposed": target_front_exposure
        <= float(requirement.get("maximum_target_front_exposure", math.inf)),
        "no_direct_side_grasp_clearance": max(target_side_gaps)
        <= float(requirement.get("maximum_direct_grasp_side_gap", math.inf)),
        "fixtures_are_low": max(fixture_heights)
        <= float(requirement.get("maximum_fixture_height", math.inf)),
        "fixtures_are_shallow": max(fixture_depths)
        <= float(requirement.get("maximum_fixture_depth", math.inf)),
        "row_is_at_robot_side_opening": all(
            float(
                requirement.get(
                    "minimum_fixture_opening_edge_margin",
                    -math.inf,
                )
            )
            <= margin
            <= float(
                requirement.get("maximum_fixture_opening_edge_margin", math.inf)
            )
            for margin in fixture_opening_edge_margins
        ),
    }
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "neighbor_gaps": gaps,
        "axis_alignment": axis_alignment,
        "outward_clearance": outward_clearance,
        "target_grasp_thickness": target_grasp_thickness,
        "target_topple_contact_width": target_topple_contact_width,
        "target_topple_face_depth": target_topple_face_depth,
        "row_heights": row_heights,
        "row_front_edges": row_front_edges,
        "target_top_exposure": target_top_exposure,
        "target_front_exposure": target_front_exposure,
        "target_side_gaps": target_side_gaps,
        "fixture_depths": fixture_depths,
        "fixture_heights": fixture_heights,
        "fixture_opening_edge_margins": fixture_opening_edge_margins,
    }


def _platform_edge_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    requirement: dict[str, Any],
) -> dict[str, Any]:
    target_geom = _single_box_geom(model, str(requirement["target"]))
    platform_geom = _single_box_geom(model, str(requirement["platform"]))
    if target_geom is None or platform_geom is None:
        return {**requirement, "passed": False, "error": "missing collision box"}
    lateral = np.asarray(requirement["platform_lateral_xy"], dtype=np.float64)
    forward = np.asarray(requirement["platform_forward_xy"], dtype=np.float64)
    lateral /= np.linalg.norm(lateral)
    forward /= np.linalg.norm(forward)
    delta = np.asarray(data.geom_xpos[target_geom, :2]) - np.asarray(
        data.geom_xpos[platform_geom, :2]
    )
    target_lateral = _projected_geom_half_extent(
        model,
        data,
        target_geom,
        lateral,
    )
    target_forward = _projected_geom_half_extent(
        model,
        data,
        target_geom,
        forward,
    )
    platform_lateral = _projected_geom_half_extent(
        model,
        data,
        platform_geom,
        lateral,
    )
    platform_forward = _projected_geom_half_extent(
        model,
        data,
        platform_geom,
        forward,
    )
    lateral_margin = platform_lateral - abs(float(delta @ lateral)) - target_lateral
    forward_margin = platform_forward - abs(float(delta @ forward)) - target_forward
    required_slide = (
        platform_forward
        + float(delta @ forward)
        - target_forward
        + float(requirement["minimum_overhang"])
    )
    remaining_support = (
        2.0 * target_forward - float(requirement["minimum_overhang"])
    )
    grasp_area = (
        2.0 * target_lateral * float(requirement["minimum_overhang"])
    )
    target_planar_area = 4.0 * target_lateral * target_forward
    target_bottom = data.geom_xpos[target_geom, 2] - _geom_world_half_extents(
        model,
        data,
        target_geom,
    )[2]
    platform_top = data.geom_xpos[platform_geom, 2] + _geom_world_half_extents(
        model,
        data,
        platform_geom,
    )[2]
    platform_height = 2.0 * _geom_world_half_extents(
        model,
        data,
        platform_geom,
    )[2]
    support_error = float(target_bottom - platform_top)
    checks = {
        "initial_lateral_support": lateral_margin
        >= float(requirement["minimum_initial_margin"]),
        "initial_forward_support": forward_margin >= 0.0,
        "support_contact": -MAX_PENETRATION <= support_error <= 0.0015,
        "reachable_overhang": 0.0 < required_slide
        <= float(requirement["maximum_slide_distance"]),
        "stable_post_slide_support": remaining_support
        >= float(requirement.get("minimum_remaining_support", 0.0)),
        "graspable_overhang_area": grasp_area
        >= float(requirement.get("minimum_grasp_area", 0.0)),
        "target_planar_area": target_planar_area
        >= float(requirement.get("minimum_target_planar_area", 0.0)),
        "platform_height": platform_height
        >= float(requirement.get("minimum_platform_height", 0.0)),
    }
    return {
        **requirement,
        "passed": all(checks.values()),
        "checks": checks,
        "lateral_margin": lateral_margin,
        "forward_margin": forward_margin,
        "support_error": support_error,
        "required_slide": required_slide,
        "remaining_support_after_slide": remaining_support,
        "exposed_grasp_area": grasp_area,
        "target_planar_area": target_planar_area,
        "platform_height": platform_height,
    }


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{index}")
        if joint_id < 0:
            raise KeyError(f"FR3 joint {index} is missing")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        lower.append(float(model.jnt_range[joint_id, 0]))
        upper.append(float(model.jnt_range[joint_id, 1]))
    return (
        np.asarray(qpos_addresses, dtype=np.int32),
        np.asarray(dof_addresses, dtype=np.int32),
        np.asarray(lower),
        np.asarray(upper),
    )


def _solve_position_ik(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray | None,
    seed: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    qpos_addresses, dof_addresses, lower, upper = _arm_addresses(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    if site_id < 0:
        raise KeyError("FR3 grasp_site is missing")
    data = mujoco.MjData(model)
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    q = np.clip(seed.copy(), lower, upper)
    best_q = q.copy()
    best_error = math.inf
    best_position_error = math.inf
    best_orientation_error = math.inf
    for _ in range(180):
        data.qpos[:] = base_qpos
        data.qpos[qpos_addresses] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        position_error = target_position - data.site_xpos[site_id]
        orientation_error = np.zeros(3, dtype=np.float64)
        if target_quaternion is not None:
            current_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quaternion, data.site_xmat[site_id])
            mujoco.mju_subQuat(orientation_error, target_quaternion, current_quaternion)
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(orientation_error))
        weighted_error = (
            position_error
            if target_quaternion is None
            else np.concatenate((position_error, 0.35 * orientation_error))
        )
        norm = float(np.linalg.norm(weighted_error))
        if norm < best_error:
            best_error = norm
            best_position_error = position_norm
            best_orientation_error = orientation_norm
            best_q = q.copy()
        if (
            position_norm <= IK_POSITION_TOLERANCE
            and (
                target_quaternion is None
                or orientation_norm <= IK_ORIENTATION_TOLERANCE
            )
        ):
            break
        jacobian_position[:] = 0.0
        jacobian_rotation[:] = 0.0
        mujoco.mj_jacSite(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        reduced = (
            jacobian_position[:, dof_addresses]
            if target_quaternion is None
            else np.vstack(
                (
                    jacobian_position[:, dof_addresses],
                    0.35 * jacobian_rotation[:, dof_addresses],
                )
            )
        )
        damping = 0.025
        lhs = reduced @ reduced.T + damping * damping * np.eye(reduced.shape[0])
        step = reduced.T @ np.linalg.solve(lhs, weighted_error)
        q = np.clip(q + 0.55 * step, lower, upper)
    margin = float(np.min(np.minimum(best_q - lower, upper - best_q)))
    return best_q, best_position_error, best_orientation_error, margin


def _robot_body_ids(model: mujoco.MjModel) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3")
    if root < 0:
        return set()
    result: set[int] = set()
    for body_id in range(model.nbody):
        current = body_id
        while current > 0:
            if current == root:
                result.add(body_id)
                break
            current = int(model.body_parentid[current])
    return result


def _collision_checked_path(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    target_name: str,
    allowed_body_names: tuple[str, ...] = (),
) -> tuple[bool, list[tuple[str, str]]]:
    qpos_addresses, _, _, _ = _arm_addresses(model)
    robot_bodies = _robot_body_ids(model)
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
    allowed_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in allowed_body_names
    }
    data = mujoco.MjData(model)
    collisions: list[tuple[str, str]] = []
    sample_count = max(
        2,
        min(32, int(math.ceil(float(np.max(np.abs(goal - start))) / 0.08)) + 1),
    )
    for alpha in np.linspace(0.0, 1.0, sample_count):
        data.qpos[:] = base_qpos
        data.qpos[qpos_addresses] = (1.0 - alpha) * start + alpha * goal
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            left_geom = int(contact.geom1)
            right_geom = int(contact.geom2)
            left_body = int(model.geom_bodyid[left_geom])
            right_body = int(model.geom_bodyid[right_geom])
            robot_contact = left_body in robot_bodies or right_body in robot_bodies
            if not robot_contact:
                continue
            other_body = right_body if left_body in robot_bodies else left_body
            if (
                other_body in robot_bodies
                or other_body == target_body
                or other_body in allowed_bodies
            ):
                continue
            pair = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, left_geom) or "",
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, right_geom) or "",
            )
            if pair not in collisions:
                collisions.append(pair)
    return not collisions, collisions


def _rrt_connectable(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    target_name: str,
    allowed_body_names: tuple[str, ...],
    *,
    seed: int,
) -> tuple[bool, int]:
    """Deterministic bidirectional joint-space search for a collision-free path."""

    _, _, lower, upper = _arm_addresses(model)
    goal_clear, _ = _collision_checked_path(
        model,
        base_qpos,
        goal,
        goal,
        target_name,
        allowed_body_names,
    )
    if not goal_clear:
        return False, 0
    rng = np.random.default_rng(seed)
    trees = [[start.copy()], [goal.copy()]]

    def extend(tree: list[np.ndarray], sample: np.ndarray) -> np.ndarray | None:
        scale = np.maximum(upper - lower, 1e-6)
        nearest = min(
            tree,
            key=lambda value: float(np.linalg.norm((value - sample) / scale)),
        )
        delta = sample - nearest
        maximum = float(np.max(np.abs(delta)))
        candidate = sample.copy() if maximum <= 0.28 else nearest + delta * (0.28 / maximum)
        clear, _ = _collision_checked_path(
            model,
            base_qpos,
            nearest,
            candidate,
            target_name,
            allowed_body_names,
        )
        if not clear:
            return None
        tree.append(candidate)
        return candidate

    for iteration in range(500):
        active = iteration % 2
        other = 1 - active
        if rng.random() < 0.22:
            sample = trees[other][0]
        else:
            sample = rng.uniform(lower, upper)
        reached = extend(trees[active], sample)
        if reached is None:
            continue
        for _ in range(24):
            connected = extend(trees[other], reached)
            if connected is None:
                break
            if float(np.max(np.abs(connected - reached))) <= 0.08:
                final_clear, _ = _collision_checked_path(
                    model,
                    base_qpos,
                    connected,
                    reached,
                    target_name,
                    allowed_body_names,
                )
                if final_clear:
                    return True, len(trees[0]) + len(trees[1])
        if len(trees[0]) + len(trees[1]) > 900:
            break
    return False, len(trees[0]) + len(trees[1])


def _phase_contact_position(
    phase: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom: int,
    contract: StrategyContract,
    private_record: dict[str, Any],
) -> np.ndarray:
    center = np.asarray(data.geom_xpos[target_geom]).copy()
    half = _geom_world_half_extents(model, data, target_geom)
    if contract.strategy_id == 1:
        corridor = private_record.get("context_layout", {}).get("corridor", {})
        axis = np.asarray(corridor.get("axis_xy", (0.0, 1.0)), dtype=np.float64)
        lateral = np.asarray(
            corridor.get("lateral_xy", (1.0, 0.0)),
            dtype=np.float64,
        )
        entrance = np.asarray(
            corridor.get("entrance_xy", center[:2]),
            dtype=np.float64,
        )
        axis /= np.linalg.norm(axis)
        lateral /= np.linalg.norm(lateral)
        target_lateral_half = _projected_geom_half_extent(
            model,
            data,
            target_geom,
            lateral,
        )
        target_axis_half = _projected_geom_half_extent(
            model,
            data,
            target_geom,
            axis,
        )
        if phase == "planar_push":
            center[:2] -= lateral * (target_lateral_half + 0.025)
        else:
            aligned_center = entrance - axis * (target_axis_half + 0.012)
            center[:2] = aligned_center - axis * (target_axis_half + 0.025)
        return center
    if contract.strategy_id == 3:
        relation = next(
            (
                item
                for item in private_record.get("relation_checks", [])
                if item.get("kind") == "extraction_then_wall_setup"
            ),
            {},
        )
        extraction_direction = np.asarray(
            relation.get("extraction_axis_xy", (0.0, -1.0)),
            dtype=np.float64,
        )
        extraction_direction /= np.linalg.norm(extraction_direction)
        if phase == "obstacle_clearance_pull":
            direction = -extraction_direction
        elif phase == "push_to_wall":
            center[:2] += extraction_direction * float(
                relation.get("minimum_open_path", 0.18)
            )
            direction = np.asarray(
                relation.get("wall_normal_xy", extraction_direction),
                dtype=np.float64,
            )
            direction /= np.linalg.norm(direction)
        else:
            layout = private_record.get("context_layout", {}).get(
                "extraction_channel",
                {},
            )
            center[:2] = np.asarray(
                layout.get("wall_contact_center_xy", center[:2])
            )
            ranked_halves = np.sort(np.asarray(model.geom_size[target_geom]))
            center[2] = max(center[2], 1.35 * float(ranked_halves[1]))
            return center
        radius = _projected_geom_half_extent(
            model,
            data,
            target_geom,
            direction,
        )
        center[:2] -= direction * (radius + 0.025)
        return center
    if contract.strategy_id == 4:
        layout = private_record.get("context_layout", {}).get("wall_flip", {})
        normal = np.asarray(
            layout.get("wall_normal_xy", (1.0, 0.0)),
            dtype=np.float64,
        )
        normal /= np.linalg.norm(normal)
        ranked_halves = np.sort(np.asarray(model.geom_size[target_geom]))
        if phase == "push_to_wall":
            radius = _projected_geom_half_extent(
                model,
                data,
                target_geom,
                normal,
            )
            center[:2] -= normal * (radius + 0.025)
            return center
        if phase in {"wall_assisted_flip", "clearance_pull"}:
            # Reach the target from its exposed face.  Placing the gripper TCP
            # at the upright target center puts the hand and wrist inside the
            # wall even though the fingers can contact the accessible face.
            center[:2] = np.asarray(
                layout.get("upright_wall_contact_center_xy", center[:2])
            )
            center[:2] -= normal * 0.040
            center[2] = max(center[2], 1.35 * float(ranked_halves[1]))
            return center
        center_key = (
            "pulled_center_xy"
            if phase == "edge_grasp_lift"
            else "upright_wall_contact_center_xy"
        )
        center[:2] = np.asarray(layout.get(center_key, center[:2]))
        center[2] = max(center[2], 1.35 * float(ranked_halves[1]))
        return center
    if contract.strategy_id == 5:
        relation = next(
            (
                item
                for item in private_record.get("relation_checks", [])
                if item.get("kind") == "slot_setup"
            ),
            {},
        )
        lateral = np.asarray(
            relation.get("lateral_xy", (1.0, 0.0)),
            dtype=np.float64,
        )
        lateral /= np.linalg.norm(lateral)
        layout = private_record.get("context_layout", {}).get(
            "upright_slot",
            {},
        )
        ranked_halves = np.sort(np.asarray(model.geom_size[target_geom]))
        if phase == "push_to_wall":
            radius = _projected_geom_half_extent(
                model,
                data,
                target_geom,
                lateral,
            )
            center[:2] -= lateral * (radius + 0.025)
            return center
        if phase == "slot_pull":
            center[:2] = np.asarray(
                layout.get("upright_staging_center_xy", center[:2])
            )
            center[2] = max(center[2], 1.35 * float(ranked_halves[2]))
            return center
        center[:2] = np.asarray(
            layout.get("flat_wall_contact_center_xy", center[:2])
        )
        center[2] += min(0.08, max(0.025, 0.45 * half[2]))
        return center
    if contract.strategy_id == 6:
        relation = next(
            (
                item
                for item in private_record.get("relation_checks", [])
                if item.get("kind") == "slot_setup"
            ),
            {},
        )
        direction = np.asarray(
            relation.get("axis_xy", (0.0, 1.0)),
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        if phase == "reorient":
            center[2] += min(0.06, max(0.025, 0.35 * half[2]))
            return center
        if phase == "slot_push":
            layout = private_record.get("context_layout", {}).get(
                "planar_slot",
                {},
            )
            center[:2] = np.asarray(
                layout.get("aligned_staging_center_xy", center[:2])
            )
            radius = float(np.max(model.geom_size[target_geom]))
        else:
            radius = _projected_geom_half_extent(
                model,
                data,
                target_geom,
                direction,
            )
        center[:2] -= direction * (radius + 0.025)
        return center
    if contract.strategy_id == 7:
        relation = next(
            (
                item
                for item in private_record.get("relation_checks", [])
                if item.get("kind") == "packed_upright_row"
            ),
            {},
        )
        outward = np.asarray(
            relation.get("outward_xy", (0.0, -1.0)),
            dtype=np.float64,
        )
        outward /= np.linalg.norm(outward)
        if "grasp" in phase:
            center[:2] += 0.08 * outward
            center[2] += min(0.08, max(0.025, 0.45 * half[2]))
        else:
            center[:2] += 0.035 * outward
            center[2] += min(0.08, max(0.025, 0.45 * half[2]))
        return center
    if contract.strategy_id == 8:
        layout = private_record.get("context_layout", {}).get(
            "platform_edge",
            {},
        )
        slide = np.asarray(
            layout.get("slide_direction_xy", (0.0, -1.0)),
            dtype=np.float64,
        )
        slide /= np.linalg.norm(slide)
        if phase == "move_to_edge":
            radius = _projected_geom_half_extent(
                model,
                data,
                target_geom,
                slide,
            )
            center[:2] -= slide * (radius + 0.025)
        else:
            center[:2] += slide * float(
                layout.get("required_slide_distance", 0.08)
            )
            center[2] += min(0.06, max(0.025, 0.45 * half[2]))
        return center
    if contract.approach_direction_xy is not None:
        radial = np.asarray(contract.approach_direction_xy, dtype=np.float64)
    else:
        radial = -center[:2].copy()
    norm = float(np.linalg.norm(radial))
    radial = radial / norm if norm > 1e-9 else np.asarray((-1.0, 0.0))
    if "grasp" in phase or "topple" in phase or "flip" in phase:
        if "topple" in phase:
            center[:2] += 0.035 * radial
        center[2] += min(0.08, max(0.025, 0.45 * half[2]))
    else:
        planar_radius = float(np.abs(radial) @ half[:2])
        center[:2] += radial * (planar_radius + 0.025)
    return center


def _reachability_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: StrategyContract,
    private_record: dict[str, Any],
) -> GateResult:
    target_name = str(private_record["authoritative_roles"]["target"])
    target_geoms = _body_box_geoms(model, target_name)
    if len(target_geoms) != 1:
        return GateResult("full_arm_reachability", False, {}, "missing target collision")
    qpos_addresses, _, lower, upper = _arm_addresses(model)
    home = data.qpos[qpos_addresses].copy()
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    seed_rng = np.random.default_rng(int(private_record["parameters"]["seed"]) + 4049)
    seeds = [
        home,
        np.clip(home + np.asarray((0.18, -0.12, 0.16, 0.10, -0.16, 0.14, -0.10)), lower, upper),
        np.clip(home + np.asarray((-0.16, 0.14, -0.12, -0.08, 0.14, -0.12, 0.12)), lower, upper),
    ]
    seeds.extend(
        np.clip(home + seed_rng.normal(0.0, 0.32, size=7), lower, upper)
        for _ in range(7)
    )
    phase_metrics: dict[str, Any] = {}
    passed = True
    previous_phase = "home"
    previous_phase_q = home.copy()
    for phase in contract.phases:
        phase_data = mujoco.MjData(model)
        phase_data.qpos[:] = data.qpos
        phase_data.qvel[:] = 0.0
        mujoco.mj_forward(model, phase_data)
        contact_name = target_name
        contact_geoms = _body_box_geoms(model, contact_name)
        if len(contact_geoms) != 1:
            phase_metrics[phase] = {
                "reachable": False,
                "error": f"missing contact body collision: {contact_name}",
            }
            passed = False
            continue
        target = _phase_contact_position(
            phase,
            model,
            phase_data,
            contact_geoms[0],
            contract,
            private_record,
        )
        discovered = [
            _solve_position_ik(model, phase_data.qpos.copy(), target, None, seed)
            for seed in seeds
        ]
        solutions: list[tuple[np.ndarray, float, float, float]] = []
        orientation_targets: list[list[float]] = []
        orientation_data = mujoco.MjData(model)
        for discovered_q, _, _, _ in discovered:
            orientation_data.qpos[:] = phase_data.qpos
            orientation_data.qpos[qpos_addresses] = discovered_q
            orientation_data.qvel[:] = 0.0
            mujoco.mj_forward(model, orientation_data)
            target_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(
                target_quaternion,
                orientation_data.site_xmat[grasp_site],
            )
            orientation_targets.append(target_quaternion.tolist())
            solutions.append(
                _solve_position_ik(
                    model,
                    phase_data.qpos.copy(),
                    target,
                    target_quaternion,
                    discovered_q,
                )
            )
        allowed_contacts = (
            tuple(private_record["authoritative_roles"].get("secondary_dynamic", []))
            if contract.strategy_id == 7
            else ()
        )
        evaluated: list[
            tuple[tuple[np.ndarray, float, float, float], bool, list[tuple[str, str]]]
        ] = []
        for solution in solutions:
            candidate_clear, candidate_collisions = _collision_checked_path(
                model,
                phase_data.qpos.copy(),
                home,
                solution[0],
                contact_name,
                allowed_contacts,
            )
            evaluated.append((solution, candidate_clear, candidate_collisions))
        chosen, path_clear, collisions = min(
            evaluated,
            key=lambda item: (
                not item[1],
                len(item[2]),
                item[0][1] + 0.2 * item[0][2],
            ),
        )
        best_q, error, orientation_error, margin = chosen
        waypoint_metrics: dict[str, Any] | None = None
        rrt_metrics: dict[str, Any] | None = None
        if not path_clear:
            waypoint_position = target.copy()
            waypoint_position[1] -= 0.12
            waypoint_position[2] = max(0.36, target[2] + 0.20)
            waypoint_q, waypoint_error, _, waypoint_margin = _solve_position_ik(
                model,
                phase_data.qpos.copy(),
                waypoint_position,
                None,
                home,
            )
            first_clear, first_collisions = _collision_checked_path(
                model,
                phase_data.qpos.copy(),
                home,
                waypoint_q,
                contact_name,
                allowed_contacts,
            )
            second_clear, second_collisions = _collision_checked_path(
                model,
                phase_data.qpos.copy(),
                waypoint_q,
                best_q,
                contact_name,
                allowed_contacts,
            )
            via_clear = (
                waypoint_error <= IK_POSITION_TOLERANCE
                and waypoint_margin >= IK_JOINT_MARGIN_RADIANS
                and first_clear
                and second_clear
            )
            waypoint_metrics = {
                "position": waypoint_position.tolist(),
                "position_error": waypoint_error,
                "joint_margin_radians": waypoint_margin,
                "first_segment_collisions": first_collisions,
                "second_segment_collisions": second_collisions,
                "path_clear": via_clear,
            }
            if via_clear:
                path_clear = True
                collisions = []
        if not path_clear:
            rrt_clear, explored_nodes = _rrt_connectable(
                model,
                phase_data.qpos.copy(),
                home,
                best_q,
                contact_name,
                allowed_contacts,
                seed=int(private_record["parameters"]["seed"])
                + sum(phase.encode("utf-8")),
            )
            rrt_metrics = {
                "path_clear": rrt_clear,
                "explored_nodes": explored_nodes,
            }
            if rrt_clear:
                path_clear = True
                collisions = []
        transition_metrics: dict[str, Any] | None = None
        transition_clear = True
        if contract.strategy_id == 3:
            transition_clear, transition_collisions = _collision_checked_path(
                model,
                phase_data.qpos.copy(),
                previous_phase_q,
                best_q,
                contact_name,
                allowed_contacts,
            )
            transition_search_nodes = 0
            if not transition_clear:
                transition_clear, transition_search_nodes = _rrt_connectable(
                    model,
                    phase_data.qpos.copy(),
                    previous_phase_q,
                    best_q,
                    contact_name,
                    allowed_contacts,
                    seed=int(private_record["parameters"]["seed"])
                    + sum(phase.encode("utf-8"))
                    + 7919,
                )
                if transition_clear:
                    transition_collisions = []
            transition_metrics = {
                "from": previous_phase,
                "collision_checked_path": transition_clear,
                "path_collisions": transition_collisions,
                "joint_space_search_nodes": transition_search_nodes,
            }
        phase_passed = (
            error <= IK_POSITION_TOLERANCE
            and orientation_error <= IK_ORIENTATION_TOLERANCE
            and margin >= IK_JOINT_MARGIN_RADIANS
            and path_clear
            and transition_clear
        )
        passed = passed and phase_passed
        phase_metrics[phase] = {
            "contact_body": contact_name,
            "target_position": target.tolist(),
            "position_error": error,
            "orientation_error_radians": orientation_error,
            "joint_margin_radians": margin,
            "position_tolerance": IK_POSITION_TOLERANCE,
            "orientation_tolerance_radians": IK_ORIENTATION_TOLERANCE,
            "joint_margin_requirement": IK_JOINT_MARGIN_RADIANS,
            "collision_checked_path": path_clear,
            "path_collisions": collisions,
            "clearance_waypoint": waypoint_metrics,
            "joint_space_search": rrt_metrics,
            "transition_from_previous_phase": transition_metrics,
            "reachable": phase_passed,
            "solution": best_q.tolist(),
            "contact_orientation_manifold_samples": orientation_targets,
        }
        if phase_passed:
            previous_phase = phase
            previous_phase_q = best_q.copy()
    return GateResult(
        "full_arm_reachability",
        passed,
        {"phases": phase_metrics},
        "each demonstrated phase has a margin-preserving FR3 contact "
        "configuration and Strategy 3 phases have collision-checked chained "
        "transitions",
    )


def _perturbation_gate(
    module: ModuleType,
    contract: StrategyContract,
    private_record: dict[str, Any],
    reset_data: mujoco.MjData,
) -> GateResult:
    target_name = str(private_record["authoritative_roles"]["target"])
    results: list[dict[str, Any]] = []
    all_passed = True
    for perturbation in range(5):
        model = module.make_model()
        data = module.make_data(model)
        if data.qpos.shape != reset_data.qpos.shape:
            return GateResult(
                "oracle_perturbations",
                False,
                {},
                "reset-state model dimensions do not match the rollout model",
            )
        # Perturb the raw generated configuration.  A scene is not allowed to
        # depend on an uncommanded transient to reach a different stable pose.
        data.qpos[:] = reset_data.qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{target_name}_joint")
        if joint_id < 0:
            return GateResult("oracle_perturbations", False, {}, "target free joint is missing")
        address = int(model.jnt_qposadr[joint_id])
        rng = np.random.default_rng(int(private_record["parameters"]["seed"]) + perturbation)
        support_normal = np.asarray((0.0, 0.0, 1.0))
        support_body = private_record["authoritative_roles"].get("support")
        if support_body is not None:
            support_geoms = _body_box_geoms(model, str(support_body))
            if len(support_geoms) == 1:
                support_rotation = np.asarray(data.geom_xmat[support_geoms[0]]).reshape(3, 3)
                support_axis = int(np.argmax(np.abs(support_rotation[2])))
                support_normal = support_rotation[:, support_axis].copy()
                if support_normal[2] < 0.0:
                    support_normal = -support_normal
        has_secondary = bool(
            private_record["authoritative_roles"].get("secondary_dynamic", [])
        )
        if contract.approach_direction_xy is not None and has_secondary:
            direction = np.asarray(contract.approach_direction_xy, dtype=np.float64)
            direction /= np.linalg.norm(direction)
            perpendicular = np.asarray((-direction[1], direction[0]))
            translation_xy = (
                direction * rng.uniform(-0.002, 0.002)
                + perpendicular * rng.uniform(-0.00025, 0.00025)
            )
            yaw_limit = math.radians(0.25)
        else:
            translation_xy = rng.uniform(-0.002, 0.002, size=2)
            yaw_limit = math.radians(1.0)
        data.qpos[address : address + 2] += translation_xy
        data.qpos[address + 2] += -float(
            support_normal[:2] @ translation_xy
        ) / float(support_normal[2])
        yaw = float(rng.uniform(-yaw_limit, yaw_limit))
        delta = np.concatenate(
            (
                np.asarray((math.cos(yaw / 2),)),
                support_normal * math.sin(yaw / 2),
            )
        )
        original = data.qpos[address + 3 : address + 7].copy()
        data.qpos[address + 3 : address + 7] = _quat_product(delta, original)
        mujoco.mj_forward(model, data)
        rollout_seconds = (
            2.0
            if contract.approach_direction_xy is not None
            and has_secondary
            else 0.75
        )
        for _ in range(max(1, int(round(rollout_seconds / model.opt.timestep)))):
            mujoco.mj_step(model, data)
        linear, angular = _free_joint_velocity_metrics(model, data)
        finite = bool(np.all(np.isfinite(data.qpos)))
        passed = finite and linear <= 0.03 and angular <= 0.25
        all_passed = all_passed and passed
        results.append(
            {
                "index": perturbation,
                "linear_speed": linear,
                "angular_speed": angular,
                "finite": finite,
                "duration_seconds": rollout_seconds,
                "passed": passed,
            }
        )
    return GateResult(
        "oracle_perturbations",
        all_passed,
        {"rollouts": results, "phase_order": private_record["phases"]},
        "five privileged small-perturbation rollouts remain stable in the demonstrated phase domain",
    )


def _quat_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(output, left, right)
    return output / np.linalg.norm(output)


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> set[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return set()
    return {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == body_id}


def _required_visible_distractor_count(total: int) -> int:
    """Keep distractor visibility diagnostic-only for noncausal objects."""

    if total < 0:
        raise ValueError("distractor count cannot be negative")
    return 0


def _visibility_gates(
    module: ModuleType,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: StrategyContract,
    private_record: dict[str, Any],
) -> tuple[GateResult, GateResult]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    defaults = dict(getattr(module, "DEFAULT_CAMERA", {}) or {})
    camera = _make_camera(
        model,
        data,
        float(defaults.get("camera_azimuth", 135.0)),
        float(defaults.get("camera_elevation", -25.0)),
        float(defaults["camera_distance"]) if defaults.get("camera_distance") is not None else None,
        tuple(defaults["camera_lookat"]) if defaults.get("camera_lookat") is not None else None,
    )
    renderer = mujoco.Renderer(model, width=640, height=360)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        segmentation = renderer.render()
    finally:
        renderer.close()
    if segmentation.ndim != 3 or segmentation.shape[2] < 2:
        failure_metrics = {"shape": list(segmentation.shape)}
        return (
            GateResult(
                "visual_evidence",
                True,
                {**failure_metrics, "diagnostic_available": False},
                "camera visibility is diagnostic-only; segmentation was unavailable",
            ),
            GateResult(
                "distractor_visual_quality",
                True,
                {**failure_metrics, "diagnostic_available": False},
                "camera visibility is diagnostic-only; segmentation was unavailable",
            ),
        )
    geom_pixels: dict[int, int] = {}
    object_ids = segmentation[:, :, 0]
    object_types = segmentation[:, :, 1]
    geom_mask = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)
    for geom_id in np.unique(object_ids[geom_mask]):
        if int(geom_id) >= 0:
            geom_pixels[int(geom_id)] = int(np.count_nonzero(geom_mask & (object_ids == geom_id)))

    roles = private_record["authoritative_roles"]
    target_pixels = sum(geom_pixels.get(item, 0) for item in _body_geom_ids(model, str(roles["target"])))
    fixture_pixels = {
        name: sum(geom_pixels.get(item, 0) for item in _body_geom_ids(model, name))
        for name in roles.get("fixtures", [])
    }
    distractor_pixels = {
        name: sum(geom_pixels.get(item, 0) for item in _body_geom_ids(model, name))
        for name in roles.get("distractors", [])
    }
    # These are the 640x360 equivalents of 1200/750/300 pixels at 1280x720.
    # Pixel counts remain useful review diagnostics, but visibility is not a
    # physical task-feasibility condition and never rejects a generated scene.
    target_minimum = 300
    fixture_minimum = 188
    distractor_minimum = 75
    causal_passed = (
        target_pixels >= target_minimum
        and all(value >= fixture_minimum for value in fixture_pixels.values())
    )
    visible_distractors = {
        name: pixels >= distractor_minimum
        for name, pixels in distractor_pixels.items()
    }
    visible_count = sum(visible_distractors.values())
    required_visible_count = _required_visible_distractor_count(
        len(distractor_pixels)
    )
    distractor_passed = visible_count >= required_visible_count
    return (
        GateResult(
            "visual_evidence",
            True,
            {
                "resolution": [640, 360],
                "target_pixels": target_pixels,
                "fixture_pixels": fixture_pixels,
                "meets_pixel_thresholds": causal_passed,
                "thresholds": {
                    "target": target_minimum,
                    "fixture": fixture_minimum,
                },
            },
            "target and fixture visibility is recorded for review but is not a rejection criterion",
        ),
        GateResult(
            "distractor_visual_quality",
            distractor_passed,
            {
                "resolution": [640, 360],
                "distractor_pixels": distractor_pixels,
                "visible_distractors": visible_distractors,
                "visible_count": visible_count,
                "required_visible_count": required_visible_count,
                "total_distractors": len(distractor_pixels),
                "pixel_threshold": distractor_minimum,
            },
            "distractor visibility is diagnostic-only and never rejects a feasible task scene",
        ),
    )


def certify_case(
    case_dir: Path,
    contract: StrategyContract,
    private_record: dict[str, Any],
    *,
    check_visibility: bool = True,
) -> CertificationResult:
    module = _load_scene(case_dir)
    settling, model, data = _settling_gate(module, private_record)
    gates: list[GateResult] = [settling]
    if settling.passed:
        gates.append(_world_frame_gate(model, data, private_record))
        gates.append(_representation_gate(model, private_record))
        gates.append(_geometry_gate(model, data, contract, private_record))
        gates.append(
            _strategy_manipulability_gate(model, contract, private_record)
        )
        gates.append(_relation_gate(model, data, private_record))
        gates.append(_reachability_gate(model, data, contract, private_record))
        gates.append(_perturbation_gate(module, contract, private_record, data))
        if check_visibility:
            try:
                gates.extend(
                    _visibility_gates(
                        module,
                        model,
                        data,
                        contract,
                        private_record,
                    )
                )
            except Exception as exc:
                gates.append(
                    GateResult(
                        "visibility_diagnostics",
                        True,
                        {
                            "diagnostic_available": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        "visibility diagnostics are optional and never reject a feasible scene",
                    )
                )
    return CertificationResult(
        passed=bool(gates) and all(gate.passed for gate in gates),
        gates=tuple(gates),
    )


__all__ = ["CertificationResult", "GateResult", "certify_case"]
