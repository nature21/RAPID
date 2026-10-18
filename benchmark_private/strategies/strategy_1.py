"""Strategy 1: align with and store inside a two-boundary corridor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..specs import StrategyContract
from ..strategy_common import (
    ContextResult,
    SceneEditor,
    planar_axes,
    run_strategy_cli,
    semantic_box_rotation,
)


# Center the complete align-and-store trajectory in the FR3 workspace.  The
# contact pose for the recontact push sits well behind the corridor entrance,
# so anchoring only the visible slot near the workspace center can leave that
# required pose outside the arm's reliable reach envelope.
REFERENCE_ENTRANCE_XY = np.asarray((0.660, 0.120), dtype=np.float64)
CORRIDOR_YAW_LIMIT_DEGREES = 28.0
TARGET_RELATIVE_YAW_LIMIT_DEGREES = 4.0
SIDE_CLEARANCE_METERS = 0.010
MAX_EXTRA_WIDTH_METERS = 0.045
MAX_CERTIFIED_EXTRA_WIDTH_METERS = 0.060
STORE_CLEARANCE_METERS = 0.020
MAX_PARALLEL_ERROR_DEGREES = 1.0
DIVIDER_APPROACH_EXTENSION_METERS = 0.105
DISTRACTOR_MANIPULATION_CLEARANCE_METERS = 0.140
LAYOUT_WORKSPACE_X = (0.30, 1.02)
LAYOUT_WORKSPACE_Y = (-0.38, 0.46)
LAYOUT_WORKSPACE_MARGIN = 0.004


def _corridor_check(
    *,
    target: str,
    obstacle: str,
    divider: str,
    entrance: np.ndarray,
    axis: np.ndarray,
    lateral: np.ndarray,
    required_clear_width: float,
    maximum_clear_width: float,
    minimum_lateral_offset: float,
) -> dict[str, Any]:
    return {
        "kind": "strategy_1_corridor",
        "target": target,
        "obstacle": obstacle,
        "divider": divider,
        "entrance_xy": entrance.tolist(),
        "axis_xy": axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "required_clear_width": required_clear_width,
        "maximum_clear_width": maximum_clear_width,
        "maximum_parallel_error_radians": math.radians(
            MAX_PARALLEL_ERROR_DEGREES
        ),
        "maximum_target_alignment_error_radians": math.radians(
            TARGET_RELATIVE_YAW_LIMIT_DEGREES + 0.5
        ),
        "minimum_lateral_offset": minimum_lateral_offset,
        "store_clearance": STORE_CLEARANCE_METERS,
    }


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    fixture, fixture_label = editor.apply_fixture_material(
        "white_divider",
        canonical_box_visual=True,
    )
    obstacle, obstacle_label = editor.reskin_dynamic(
        "brown_cardboard_box",
        editor.parameters.secondary_appearances[0],
        canonical_box=True,
    )

    target_half_extents = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_vertical_half, target_lateral_base, target_axis_base = np.sort(
        target_half_extents
    )
    corridor_yaw = math.radians(CORRIDOR_YAW_LIMIT_DEGREES * c[2])
    relative_target_yaw = math.radians(
        TARGET_RELATIVE_YAW_LIMIT_DEGREES * c[3]
    )
    target_lateral_half = float(
        target_lateral_base * abs(math.cos(relative_target_yaw))
        + target_axis_base * abs(math.sin(relative_target_yaw))
    )
    target_axis_half = float(
        target_axis_base * abs(math.cos(relative_target_yaw))
        + target_lateral_base * abs(math.sin(relative_target_yaw))
    )

    width_variation = 0.5 * (c[4] + 1.0) * (
        MAX_EXTRA_WIDTH_METERS - 2.0 * SIDE_CLEARANCE_METERS
    )
    required_clear_width = 2.0 * (
        target_lateral_half + SIDE_CLEARANCE_METERS
    )
    corridor_clear_width = required_clear_width + width_variation
    maximum_clear_width = (
        2.0 * target_lateral_half + MAX_CERTIFIED_EXTRA_WIDTH_METERS
    )

    lateral, axis = planar_axes(corridor_yaw)
    entrance = (
        REFERENCE_ENTRANCE_XY
        + 0.030 * c[0] * lateral
        + 0.025 * c[1] * axis
    )

    _, _, obstacle_half_extents = editor.collision_world_geometry(obstacle)
    obstacle_world_rotation, obstacle_lateral_half, obstacle_axis_half, _ = (
        semantic_box_rotation(
            obstacle_half_extents,
            lateral=lateral,
            forward=axis,
            lateral_rank=1,
            vertical_rank=0,
        )
    )
    required_depth = 2.0 * target_axis_half + STORE_CLEARANCE_METERS
    available_depth = 2.0 * obstacle_axis_half
    if available_depth < required_depth:
        raise ValueError(
            "strategy-1 obstacle is too short for the target: "
            f"available={available_depth:.4f}, required={required_depth:.4f}"
        )

    # The divider is not merely a flip wall: together with the obstacle it
    # must bound the complete storage depth.  Anchor its near end behind the
    # entrance (as in the demo) and extend the far end at least as deep as the
    # opposite obstacle.
    _, _, divider_half_extents = editor.collision_world_geometry(fixture)
    divider_axis_index = int(
        np.argsort(np.asarray(divider_half_extents), kind="stable")[2]
    )
    divider_approach_extension = (
        DIVIDER_APPROACH_EXTENSION_METERS
        + 0.015 * (0.5 * (c[5] + 1.0))
    )
    required_divider_length = divider_approach_extension + available_depth
    divider_half_extents = np.asarray(divider_half_extents).copy()
    divider_half_extents[divider_axis_index] = max(
        float(divider_half_extents[divider_axis_index]),
        0.5 * required_divider_length,
    )
    editor.resize_static_canonical_box(fixture, divider_half_extents)
    (
        divider_world_rotation,
        divider_lateral_half,
        divider_axis_half,
        _,
    ) = semantic_box_rotation(
        divider_half_extents,
        lateral=lateral,
        forward=axis,
        lateral_rank=0,
        vertical_rank=1,
    )

    obstacle_center = (
        entrance
        - lateral * (0.5 * corridor_clear_width + obstacle_lateral_half)
        + axis * obstacle_axis_half
    )
    divider_center = (
        entrance
        + lateral * (0.5 * corridor_clear_width + divider_lateral_half)
        + axis * (divider_axis_half - divider_approach_extension)
    )
    editor.place_box_collision_on_floor(
        obstacle,
        center_xy=(float(obstacle_center[0]), float(obstacle_center[1])),
        world_rotation=obstacle_world_rotation,
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(divider_center[0]), float(divider_center[1])),
        world_rotation=divider_world_rotation,
    )

    approach_clearance = 0.035 + 0.015 * (0.5 * (c[6] + 1.0))
    initial_lateral_offset = -(
        0.5 * corridor_clear_width + 0.85 * target_lateral_half
    )
    target_center = (
        entrance
        - axis * (target_axis_half + approach_clearance)
        + lateral * initial_lateral_offset
    )
    target_yaw = corridor_yaw + relative_target_yaw

    target_world_half = (
        np.abs(lateral) * target_lateral_half
        + np.abs(axis) * target_axis_half
    )
    target_center_lower = np.asarray(
        (
            LAYOUT_WORKSPACE_X[0] + target_world_half[0] + LAYOUT_WORKSPACE_MARGIN,
            LAYOUT_WORKSPACE_Y[0] + target_world_half[1] + LAYOUT_WORKSPACE_MARGIN,
        )
    )
    target_center_upper = np.asarray(
        (
            LAYOUT_WORKSPACE_X[1] - target_world_half[0] - LAYOUT_WORKSPACE_MARGIN,
            LAYOUT_WORKSPACE_Y[1] - target_world_half[1] - LAYOUT_WORKSPACE_MARGIN,
        )
    )
    workspace_correction = (
        np.clip(target_center, target_center_lower, target_center_upper)
        - target_center
    )
    if np.any(workspace_correction):
        entrance = entrance + workspace_correction
        obstacle_center = obstacle_center + workspace_correction
        divider_center = divider_center + workspace_correction
        target_center = target_center + workspace_correction
        editor.place_box_collision_on_floor(
            obstacle,
            center_xy=(float(obstacle_center[0]), float(obstacle_center[1])),
            world_rotation=obstacle_world_rotation,
        )
        editor.place_box_collision_on_floor(
            fixture,
            center_xy=(float(divider_center[0]), float(divider_center[1])),
            world_rotation=divider_world_rotation,
        )

    store_center = entrance + axis * (
        target_axis_half + STORE_CLEARANCE_METERS
    )
    aligned_center = entrance - axis * (target_axis_half + 0.012)
    editor.layout_record["corridor"] = {
        "entrance_xy": entrance.tolist(),
        "axis_xy": axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "corridor_yaw_radians": corridor_yaw,
        "obstacle_yaw_radians": corridor_yaw,
        "divider_yaw_radians": corridor_yaw,
        "target_relative_yaw_radians": relative_target_yaw,
        "clear_width": corridor_clear_width,
        "required_clear_width": required_clear_width,
        "maximum_clear_width": maximum_clear_width,
        "target_lateral_half_extent": target_lateral_half,
        "target_axis_half_extent": target_axis_half,
        "target_vertical_half_extent": float(target_vertical_half),
        "target_initial_center_xy": target_center.tolist(),
        "workspace_correction_xy": workspace_correction.tolist(),
        "target_initial_lateral_offset": initial_lateral_offset,
        "store_center_xy": store_center.tolist(),
        "available_store_depth": available_depth,
        "required_store_depth": required_depth,
        "divider_approach_extension": divider_approach_extension,
        "divider_axis_half_extent": divider_axis_half,
        "obstacle_center_xy": obstacle_center.tolist(),
        "divider_center_xy": divider_center.tolist(),
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        aligned_center.tolist(),
        store_center.tolist(),
    ]
    editor.layout_record["distractor_manipulation_clearance"] = (
        DISTRACTOR_MANIPULATION_CLEARANCE_METERS
    )

    return ContextResult(
        (fixture,),
        (obstacle,),
        None,
        dict(editor.entity_renames),
        {"fixture_0": fixture_label, "secondary_0": obstacle_label},
        (
            _corridor_check(
                target=contract.target_body,
                obstacle=obstacle,
                divider=fixture,
                entrance=entrance,
                axis=axis,
                lateral=lateral,
                required_clear_width=required_clear_width,
                maximum_clear_width=maximum_clear_width,
                minimum_lateral_offset=0.5 * target_lateral_half,
            ),
        ),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(1))
