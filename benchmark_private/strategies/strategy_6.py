"""Strategy 6: pivot a flat box, align it with a slot, and insert."""

from __future__ import annotations

import math

import numpy as np

from ..specs import StrategyContract
from ..strategy_common import (
    ContextResult,
    SceneEditor,
    planar_axes,
    run_strategy_cli,
    semantic_box_rotation,
)


# Keep the complete slot-and-pivot layout centered in the FR3's reachable
# table region, including the positive-yaw stress poses.
REFERENCE_ENTRANCE = np.asarray((0.575, 0.18), dtype=np.float64)
ROBOTWARD_LAYOUT_TRANSLATION = np.asarray((-0.10, 0.0), dtype=np.float64)
SIDE_CLEARANCE = 0.011
PIVOT_OBSTACLE_CLEARANCE = 0.150
DISTRACTOR_MANIPULATION_CLEARANCE = 0.160
PIVOT_SWEEP_BUFFER = 0.030
PIVOT_WORKSPACE_X = (0.30, 1.02)
PIVOT_WORKSPACE_Y = (-0.38, 0.46)
PIVOT_WORKSPACE_MARGIN = 0.004
TARGET_WIDTH_SCALE = 0.50
MAX_THICKNESS_TO_WIDTH_RATIO = 0.90


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    left, left_label = editor.reskin_static(
        "brown_cardboard_box",
        editor.parameters.secondary_appearances[0],
        canonical_box=True,
    )
    right, right_label = editor.reskin_static(
        "white_product_box",
        editor.parameters.secondary_appearances[1],
        canonical_box=True,
    )
    task_yaw = editor.parameters.yaw_radians + math.radians(14.0 * c[2])
    lateral, slot_axis = planar_axes(task_yaw)
    entrance = (
        REFERENCE_ENTRANCE
        + np.asarray(editor.parameters.translation_xy)
        + ROBOTWARD_LAYOUT_TRANSLATION
        + 0.018 * c[0] * lateral
        + 0.018 * c[1] * slot_axis
    )
    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_half = target_half.copy()
    target_thickness_axis, target_width_axis, _ = np.argsort(target_half)
    target_half[target_width_axis] *= TARGET_WIDTH_SCALE
    target_half[target_thickness_axis] = min(
        target_half[target_thickness_axis],
        MAX_THICKNESS_TO_WIDTH_RATIO * target_half[target_width_axis],
    )
    _, target_lateral_half, target_axis_half = np.sort(target_half)
    pivot_sweep_radius = math.hypot(target_axis_half, target_lateral_half)
    pivot_obstacle_clearance = (
        PIVOT_OBSTACLE_CLEARANCE + 0.015 * (0.5 * (c[6] + 1.0))
    )
    pivot_center_distance = pivot_sweep_radius + pivot_obstacle_clearance
    target_lateral_offset = 0.045 * c[7]
    provisional_target_center = (
        entrance
        - slot_axis * pivot_center_distance
        + lateral * target_lateral_offset
    )
    pivot_center_lower = np.asarray(
        (
            PIVOT_WORKSPACE_X[0]
            + pivot_sweep_radius
            + PIVOT_WORKSPACE_MARGIN,
            PIVOT_WORKSPACE_Y[0]
            + pivot_sweep_radius
            + PIVOT_WORKSPACE_MARGIN,
        )
    )
    pivot_center_upper = np.asarray(
        (
            PIVOT_WORKSPACE_X[1]
            - pivot_sweep_radius
            - PIVOT_WORKSPACE_MARGIN,
            PIVOT_WORKSPACE_Y[1]
            - pivot_sweep_radius
            - PIVOT_WORKSPACE_MARGIN,
        )
    )
    workspace_correction = (
        np.clip(
            provisional_target_center,
            pivot_center_lower,
            pivot_center_upper,
        )
        - provisional_target_center
    )
    entrance = entrance + workspace_correction
    clear_width = 2.0 * (
        target_lateral_half
        + SIDE_CLEARANCE
        + 0.005 * (0.5 * (c[3] + 1.0))
    )

    boundary_records: list[tuple[str, np.ndarray, float, float]] = []
    for name in (left, right):
        _, _, half_extents = editor.collision_world_geometry(name)
        rotation, lateral_half, axis_half, _ = semantic_box_rotation(
            half_extents,
            lateral=lateral,
            forward=slot_axis,
            lateral_rank=1,
            vertical_rank=0,
        )
        boundary_records.append((name, rotation, lateral_half, axis_half))
    left_center = (
        entrance
        - lateral * (0.5 * clear_width + boundary_records[0][2])
        + slot_axis * boundary_records[0][3]
    )
    right_center = (
        entrance
        + lateral * (0.5 * clear_width + boundary_records[1][2])
        + slot_axis * boundary_records[1][3]
    )
    for record, center in zip(boundary_records, (left_center, right_center)):
        editor.place_box_collision_on_floor(
            record[0],
            center_xy=(float(center[0]), float(center[1])),
            world_rotation=record[1],
        )

    # The demo uses a near-quarter-turn pivot: the long target initially lies
    # across the opening.  A small 18--30 degree error only asks for a minor
    # correction and removes the demonstrated reorientation phase.
    relative_angle = math.radians(
        (72.0 + 16.0 * (0.5 * (c[4] + 1.0)))
        * (-1.0 if c[5] < 0.0 else 1.0)
    )
    target_axis_projection = (
        target_axis_half * abs(math.cos(relative_angle))
        + target_lateral_half * abs(math.sin(relative_angle))
    )
    aligned_outside_clearance = pivot_center_distance - target_axis_half
    target_center = (
        entrance
        # Keep the complete rotation disk away from both boundary rectangles.
        # Clearance based only on the current oriented footprint lets a long
        # corner sweep into a boundary midway through the pivot.
        - slot_axis * pivot_center_distance
        + lateral * target_lateral_offset
    )
    target_yaw = task_yaw + relative_angle
    available_depth = 2.0 * min(item[3] for item in boundary_records)
    required_depth = 2.0 * target_axis_half + 0.020
    if available_depth < required_depth:
        raise ValueError(
            "strategy-8 slot is too shallow for the aligned target: "
            f"available={available_depth:.4f}, required={required_depth:.4f}"
        )
    target_lateral_projection = (
        target_axis_half * abs(math.sin(relative_angle))
        + target_lateral_half * abs(math.cos(relative_angle))
    )
    aligned_staging_center = entrance - slot_axis * (
        target_axis_half + 0.012
    )
    stored_center = entrance + slot_axis * (target_axis_half + 0.010)
    editor.layout_record["planar_slot"] = {
        "entrance_xy": entrance.tolist(),
        "axis_xy": slot_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "clear_width": clear_width,
        "required_target_width": 2.0 * float(target_lateral_half),
        "target_width_scale": TARGET_WIDTH_SCALE,
        "maximum_thickness_to_width_ratio": MAX_THICKNESS_TO_WIDTH_RATIO,
        "available_depth": available_depth,
        "required_depth": required_depth,
        "initial_alignment_error_radians": relative_angle,
        "initial_target_axis_half_extent": target_axis_projection,
        "initial_target_lateral_width": 2.0 * target_lateral_projection,
        "initial_width_excess": 2.0 * target_lateral_projection - clear_width,
        "aligned_outside_clearance": aligned_outside_clearance,
        "pivot_sweep_radius": pivot_sweep_radius,
        "pivot_obstacle_clearance": pivot_obstacle_clearance,
        "pivot_center_distance": pivot_center_distance,
        "workspace_correction_xy": workspace_correction.tolist(),
        "target_center_xy": target_center.tolist(),
        "aligned_staging_center_xy": aligned_staging_center.tolist(),
        "stored_center_xy": stored_center.tolist(),
        "left_center_xy": left_center.tolist(),
        "right_center_xy": right_center.tolist(),
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        aligned_staging_center.tolist(),
        stored_center.tolist(),
    ]
    editor.layout_record["manipulation_exclusion_regions"] = [
        {
            "center_xy": target_center.tolist(),
            "half_extents_xy": [
                pivot_sweep_radius + PIVOT_SWEEP_BUFFER,
                pivot_sweep_radius + PIVOT_SWEEP_BUFFER,
            ],
        }
    ]
    editor.layout_record["distractor_manipulation_clearance"] = (
        DISTRACTOR_MANIPULATION_CLEARANCE
    )
    relation = {
        "kind": "slot_setup",
        "target": contract.target_body,
        "left": left,
        "right": right,
        "entrance_xy": entrance.tolist(),
        "axis_xy": slot_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "required_clear_width": clear_width - 1e-4,
        "maximum_clear_width": clear_width + 0.025,
        "minimum_depth": required_depth,
        "target_alignment_mode": "misaligned_flat",
        "minimum_alignment_error_radians": math.radians(70.0),
        "maximum_alignment_error_radians": math.radians(90.0),
        "minimum_initial_width_excess": 0.012,
        "minimum_aligned_outside_clearance": aligned_outside_clearance - 0.003,
        "minimum_pivot_obstacle_clearance": (
            pivot_obstacle_clearance - 0.003
        ),
        "require_pivot_sweep_in_workspace": True,
    }
    return ContextResult(
        (left, right),
        (),
        None,
        dict(editor.entity_renames),
        {"secondary_0": left_label, "secondary_1": right_label},
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
        target_half_extents_override=tuple(float(value) for value in target_half),
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(6))
