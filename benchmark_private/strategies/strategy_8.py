"""Strategy 8: slide a flat object to a platform edge, then lift."""

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


REFERENCE_PLATFORM_CENTER = np.asarray((0.665, 0.05), dtype=np.float64)
MIN_INITIAL_EDGE_MARGIN = 0.035
MIN_OPPOSITE_EDGE_MARGIN = 0.025
MIN_LATERAL_MARGIN = 0.020
MIN_GRASP_OVERHANG = 0.035
MIN_REMAINING_SUPPORT = 0.050
MIN_GRASP_AREA = 0.006
MIN_PLATFORM_HEIGHT = 0.180
HORIZONTAL_GRASP_APPROACH = 0.220
DISTRACTOR_MANIPULATION_CLEARANCE = 0.140


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    fixture, fixture_label = editor.apply_fixture_material(
        "white_drawer",
        canonical_box_visual=True,
    )
    task_yaw = editor.parameters.yaw_radians + math.radians(8.0 * c[2])
    platform_lateral, platform_forward = planar_axes(task_yaw)
    platform_center = (
        REFERENCE_PLATFORM_CENTER
        + np.asarray(editor.parameters.translation_xy)
        + 0.016 * c[0] * platform_lateral
        + 0.016 * c[1] * platform_forward
    )
    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_vertical_half, target_forward_half, target_lateral_half = np.sort(
        target_half
    )
    initial_edge_margin = MIN_INITIAL_EDGE_MARGIN + 0.015 * (
        0.5 * (c[3] + 1.0)
    )
    lateral_offset = 0.020 * c[4]

    _, _, fixture_half = editor.collision_world_geometry(fixture)
    fixture_order = np.argsort(np.asarray(fixture_half), kind="stable")
    platform_vertical_index = int(fixture_order[0])
    platform_forward_index = int(fixture_order[1])
    platform_lateral_index = int(fixture_order[2])
    fixture_half = np.asarray(fixture_half).copy()
    fixture_half[platform_vertical_index] = max(
        float(fixture_half[platform_vertical_index]),
        0.5 * MIN_PLATFORM_HEIGHT,
    )
    fixture_half[platform_forward_index] = max(
        float(fixture_half[platform_forward_index]),
        float(
            target_forward_half
            + 0.5 * (initial_edge_margin + MIN_OPPOSITE_EDGE_MARGIN)
        ),
    )
    fixture_half[platform_lateral_index] = max(
        float(fixture_half[platform_lateral_index]),
        float(target_lateral_half + abs(lateral_offset) + MIN_LATERAL_MARGIN),
    )
    editor.resize_static_canonical_box(fixture, fixture_half)
    (
        fixture_rotation,
        platform_lateral_half,
        platform_forward_half,
        platform_vertical_half,
    ) = semantic_box_rotation(
        fixture_half,
        lateral=platform_lateral,
        forward=platform_forward,
        lateral_rank=2,
        vertical_rank=0,
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(platform_center[0]), float(platform_center[1])),
        world_rotation=fixture_rotation,
    )

    initial_forward_offset = (
        target_forward_half - platform_forward_half + initial_edge_margin
    )
    target_center = (
        platform_center
        + platform_forward * initial_forward_offset
        + platform_lateral * lateral_offset
    )
    slide_direction = -platform_forward
    desired_overhang = max(
        MIN_GRASP_OVERHANG,
        0.35 * float(2.0 * target_forward_half),
    )
    required_slide = (
        platform_forward_half
        + desired_overhang
        + initial_forward_offset
        - target_forward_half
    )
    lateral_margin = (
        platform_lateral_half
        - abs(lateral_offset)
        - target_lateral_half
    )
    remaining_support = 2.0 * target_forward_half - desired_overhang
    grasp_area = 2.0 * target_lateral_half * desired_overhang

    editor.layout_record["platform_edge"] = {
        "platform_center_xy": platform_center.tolist(),
        "platform_lateral_xy": platform_lateral.tolist(),
        "platform_forward_xy": platform_forward.tolist(),
        "slide_direction_xy": slide_direction.tolist(),
        "platform_lateral_half_extent": platform_lateral_half,
        "platform_forward_half_extent": platform_forward_half,
        "platform_height": 2.0 * platform_vertical_half,
        "target_center_xy": target_center.tolist(),
        "target_vertical_half_extent": float(target_vertical_half),
        "target_lateral_half_extent": float(target_lateral_half),
        "target_forward_half_extent": float(target_forward_half),
        "target_planar_area": float(
            4.0 * target_lateral_half * target_forward_half
        ),
        "initial_lateral_margin": lateral_margin,
        "initial_edge_margin": initial_edge_margin,
        "desired_overhang": desired_overhang,
        "remaining_support_after_slide": remaining_support,
        "exposed_grasp_area": grasp_area,
        "required_slide_distance": required_slide,
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        (target_center + slide_direction * required_slide).tolist(),
    ]
    target_world_half = (
        np.abs(platform_lateral) * target_lateral_half
        + np.abs(platform_forward) * target_forward_half
    )
    platform_world_half = (
        np.abs(platform_lateral) * platform_lateral_half
        + np.abs(platform_forward) * platform_forward_half
    )
    post_slide_center = target_center + slide_direction * required_slide
    grasp_approach_end = (
        post_slide_center + slide_direction * HORIZONTAL_GRASP_APPROACH
    )
    editor.layout_record["manipulation_exclusion_regions"] = [
        {
            "center_xy": platform_center.tolist(),
            "half_extents_xy": platform_world_half.tolist(),
        },
        {
            "center_xy": (
                0.5 * (post_slide_center + grasp_approach_end)
            ).tolist(),
            "half_extents_xy": (
                target_world_half
                + 0.5 * np.abs(grasp_approach_end - post_slide_center)
            ).tolist(),
        },
    ]
    editor.layout_record["distractor_manipulation_clearance"] = (
        DISTRACTOR_MANIPULATION_CLEARANCE
    )
    relation = {
        "kind": "platform_edge_setup",
        "target": contract.target_body,
        "platform": fixture,
        "platform_lateral_xy": platform_lateral.tolist(),
        "platform_forward_xy": platform_forward.tolist(),
        "slide_direction_xy": slide_direction.tolist(),
        "minimum_initial_margin": 0.018,
        "minimum_target_planar_area": 0.032,
        "minimum_overhang": desired_overhang,
        "minimum_remaining_support": MIN_REMAINING_SUPPORT,
        "minimum_grasp_area": MIN_GRASP_AREA,
        "minimum_platform_height": MIN_PLATFORM_HEIGHT - 0.002,
        "maximum_slide_distance": 0.22,
    }
    return ContextResult(
        (fixture,),
        (),
        fixture,
        dict(editor.entity_renames),
        {"fixture_0": fixture_label},
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        task_yaw,
        True,
        2,
        0,
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(8))
