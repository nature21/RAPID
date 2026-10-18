"""Strategy 5: flip a flat box upright and pull it into a narrow slot."""

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


REFERENCE_ENTRANCE = np.asarray((0.69, 0.075), dtype=np.float64)
SIDE_CLEARANCE = 0.009
FLIP_WALL_LEAD = 0.005
SLOT_DEPTH_MARGIN = 0.020


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
    task_yaw = editor.parameters.yaw_radians + math.radians(10.0 * c[2])
    lateral, slot_axis = planar_axes(task_yaw)
    entrance = (
        REFERENCE_ENTRANCE
        + np.asarray(editor.parameters.translation_xy)
        + 0.018 * c[0] * lateral
        + 0.018 * c[1] * slot_axis
    )

    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_thin_half, target_middle_half, target_long_half = np.sort(target_half)
    clear_width = 2.0 * (
        target_thin_half
        + SIDE_CLEARANCE
        + 0.004 * (0.5 * (c[3] + 1.0))
    )
    approach_gap = 0.050 + 0.025 * (0.5 * (c[5] + 1.0))
    required_depth = 2.0 * target_middle_half + 0.020
    # Match the demonstrated scene: the divider reaches from the slot past
    # the complete flat target footprint, leaving a standalone wall segment
    # that can support the wall-assisted flip.  The opposite boundary still
    # starts at the slot entrance.
    flip_wall_exposure = (
        approach_gap + 2.0 * target_middle_half + FLIP_WALL_LEAD
    )

    _, _, obstacle_half = editor.collision_world_geometry(obstacle)
    obstacle_rotation, obstacle_lateral_half, obstacle_axis_half, _ = (
        semantic_box_rotation(
            obstacle_half,
            lateral=lateral,
            forward=slot_axis,
            lateral_rank=1,
            vertical_rank=0,
        )
    )
    _, _, fixture_half = editor.collision_world_geometry(fixture)
    fixture_axis_index = int(
        np.argsort(np.asarray(fixture_half), kind="stable")[2]
    )
    required_fixture_length = (
        flip_wall_exposure + required_depth + SLOT_DEPTH_MARGIN
    )
    fixture_half = np.asarray(fixture_half).copy()
    fixture_half[fixture_axis_index] = max(
        float(fixture_half[fixture_axis_index]),
        0.5 * required_fixture_length,
    )
    editor.resize_static_canonical_box(fixture, fixture_half)
    fixture_rotation, fixture_lateral_half, fixture_axis_half, _ = (
        semantic_box_rotation(
            fixture_half,
            lateral=lateral,
            forward=slot_axis,
            lateral_rank=0,
            vertical_rank=1,
        )
    )
    obstacle_center = (
        entrance
        - lateral * (0.5 * clear_width + obstacle_lateral_half)
        + slot_axis * obstacle_axis_half
    )
    fixture_center = (
        entrance
        + lateral * (0.5 * clear_width + fixture_lateral_half)
        + slot_axis * (fixture_axis_half - flip_wall_exposure)
    )
    editor.place_box_collision_on_floor(
        obstacle,
        center_xy=(float(obstacle_center[0]), float(obstacle_center[1])),
        world_rotation=obstacle_rotation,
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(fixture_center[0]), float(fixture_center[1])),
        world_rotation=fixture_rotation,
    )

    initial_push_distance = 0.055 + 0.025 * (0.5 * (c[6] + 1.0))
    initial_lateral_offset = (
        0.5 * clear_width - target_long_half - initial_push_distance
    )
    target_center = (
        entrance
        - slot_axis * (target_middle_half + approach_gap)
        + lateral * initial_lateral_offset
    )
    target_yaw = task_yaw - math.pi / 2.0
    available_depth = min(
        2.0 * obstacle_axis_half,
        2.0 * fixture_axis_half - flip_wall_exposure,
    )
    flat_wall_contact_center = target_center + lateral * initial_push_distance
    upright_staging_center = entrance - slot_axis * (
        target_middle_half + 0.012
    )
    stored_center = entrance + slot_axis * (
        target_middle_half + 0.010
    )
    editor.layout_record["upright_slot"] = {
        "entrance_xy": entrance.tolist(),
        "axis_xy": slot_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "clear_width": clear_width,
        "required_upright_width": 2.0 * float(target_thin_half),
        "available_depth": available_depth,
        "required_depth": required_depth,
        "flip_wall": fixture,
        "flip_wall_exposure": flip_wall_exposure,
        "minimum_target_wall_overlap": 2.0 * target_middle_half,
        "initial_push_distance": initial_push_distance,
        "target_center_xy": target_center.tolist(),
        "flat_wall_contact_center_xy": flat_wall_contact_center.tolist(),
        "upright_staging_center_xy": upright_staging_center.tolist(),
        "stored_center_xy": stored_center.tolist(),
        "target_initial_lateral_offset": initial_lateral_offset,
        "obstacle_center_xy": obstacle_center.tolist(),
        "fixture_center_xy": fixture_center.tolist(),
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        flat_wall_contact_center.tolist(),
        upright_staging_center.tolist(),
        stored_center.tolist(),
    ]
    relation = {
        "kind": "slot_setup",
        "target": contract.target_body,
        "left": obstacle,
        "right": fixture,
        "entrance_xy": entrance.tolist(),
        "axis_xy": slot_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "required_clear_width": clear_width - 1e-4,
        "maximum_clear_width": clear_width + 0.025,
        "minimum_depth": required_depth,
        "flip_wall": fixture,
        "opposing_boundary": obstacle,
        "minimum_flip_wall_exposure": flip_wall_exposure - 0.004,
        "minimum_target_wall_overlap": 2.0 * target_middle_half - 0.006,
        "minimum_initial_push_distance": initial_push_distance - 0.004,
        "maximum_initial_push_distance": initial_push_distance + 0.004,
        "target_alignment_mode": "crosswise_flat",
        "minimum_alignment_error_radians": math.radians(85.0),
        "maximum_alignment_error_radians": math.radians(95.0),
        "minimum_initial_width_excess": (
            2.0 * target_long_half - clear_width - 0.004
        ),
    }
    return ContextResult(
        (fixture,),
        (obstacle,),
        None,
        dict(editor.entity_renames),
        {"fixture_0": fixture_label, "secondary_0": obstacle_label},
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(5))
