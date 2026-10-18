"""Strategy 3: extract from a channel, then flip at a separate wall."""

from __future__ import annotations

import math

import numpy as np

from ..specs import (
    STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M,
    STRATEGY_3_GRIPPER_MAX_WIDTH_M,
    STRATEGY_3_MAX_EXTRACTION_DISTANCE_M,
    STRATEGY_3_MIN_EXTRACTION_DISTANCE_M,
    StrategyContract,
    strategy_3_manipulation_metrics,
)
from ..strategy_common import (
    ContextResult,
    SceneEditor,
    planar_axes,
    run_strategy_cli,
    semantic_box_rotation,
)


REFERENCE_ROW_CENTER = np.asarray((0.63, 0.25), dtype=np.float64)
SIDE_CLEARANCE = 0.010
MINIMUM_POST_FLIP_EDGE_EXPOSURE = 0.050
DESIGN_POST_FLIP_EDGE_EXPOSURE = 0.055
MINIMUM_FLIP_WALL_HEIGHT = 0.075


def _primitive_extraction_distance(
    target_half_extents: np.ndarray,
    *,
    obstacle_opening_mismatch: float,
    target_open_clearance: float,
) -> float:
    """Mirror the frozen Strategy 3 displacement resolver exactly."""

    ranked = np.sort(np.asarray(target_half_extents, dtype=np.float64))
    return float(
        obstacle_opening_mismatch
        + 2.0 * ranked[2]
        + target_open_clearance
        + 0.4 * ranked[0]
    )


def _fit_target_to_primitive_envelope(
    target_half_extents: np.ndarray,
    *,
    obstacle_opening_mismatch: float,
    target_open_clearance: float,
) -> tuple[np.ndarray, float, float]:
    """Uniformly shrink only oversized targets into the learned domain."""

    original = np.asarray(target_half_extents, dtype=np.float64)
    ranked = np.sort(original)
    fixed_distance = obstacle_opening_mismatch + target_open_clearance
    target_distance = 2.0 * ranked[2] + 0.4 * ranked[0]
    if target_distance <= 0.0:
        raise ValueError("strategy-3 target extraction geometry is degenerate")
    scale = min(
        1.0,
        (STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M - fixed_distance)
        / target_distance,
    )
    if scale <= 0.0:
        raise ValueError("strategy-3 corridor leaves no feasible extraction domain")
    fitted = original * scale
    distance = _primitive_extraction_distance(
        fitted,
        obstacle_opening_mismatch=obstacle_opening_mismatch,
        target_open_clearance=target_open_clearance,
    )
    if not (
        STRATEGY_3_MIN_EXTRACTION_DISTANCE_M
        <= distance
        <= STRATEGY_3_MAX_EXTRACTION_DISTANCE_M
    ):
        raise ValueError(
            "strategy-3 target cannot satisfy the learned extraction domain: "
            f"{distance:.6f} m"
        )
    return fitted, float(scale), distance


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    fixture, fixture_label = editor.apply_fixture_material(
        "large_white_panel",
        canonical_box_visual=True,
    )
    left, left_label = editor.reskin_dynamic(
        "purple_and_white_carton",
        editor.parameters.secondary_appearances[0],
        canonical_box=True,
    )
    right, right_label = editor.reskin_dynamic(
        "brown_cardboard_box",
        editor.parameters.secondary_appearances[1],
        canonical_box=True,
    )

    task_yaw = editor.parameters.yaw_radians + math.radians(8.0 * c[2])
    lateral, rearward = planar_axes(task_yaw)
    extraction_axis = -rearward
    row_center = (
        REFERENCE_ROW_CENTER
        + np.asarray(editor.parameters.translation_xy)
        + 0.015 * c[0] * lateral
        + 0.015 * c[1] * extraction_axis
    )
    raw_target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )

    obstacle_records: list[tuple[str, np.ndarray, float, float]] = []
    for name in (left, right):
        _, _, half_extents = editor.collision_world_geometry(name)
        rotation, lateral_half, forward_half, _ = semantic_box_rotation(
            half_extents,
            lateral=lateral,
            forward=extraction_axis,
            lateral_rank=1,
            vertical_rank=0,
        )
        obstacle_records.append((name, rotation, lateral_half, forward_half))

    # The frozen resolver infers the corridor lateral axis directly from the
    # two obstacle centers.  Keep their axial coordinates equal so that this
    # inferred axis is exactly the designed ``lateral`` axis.  A stagger here
    # silently rotates the runtime extraction direction and can push the
    # resolved press_move distance beyond its 0.30 m domain.
    corridor_axial_offset = 0.012 * c[4]
    left_axial_offset = corridor_axial_offset
    right_axial_offset = corridor_axial_offset
    channel_back = max(
        left_axial_offset - obstacle_records[0][3],
        right_axial_offset - obstacle_records[1][3],
    )
    channel_open = min(
        left_axial_offset + obstacle_records[0][3],
        right_axial_offset + obstacle_records[1][3],
    )
    # The target is completely inside the shared part of the corridor, whose
    # opening is the earlier of the two obstacle ends.  The separate flip wall
    # must instead start beyond the later end; otherwise unequal obstacle
    # lengths make the wall overlap the longer box at reset.
    channel_outer_open = max(
        left_axial_offset + obstacle_records[0][3],
        right_axial_offset + obstacle_records[1][3],
    )
    target_open_clearance = 0.012 + 0.008 * (0.5 * (c[7] + 1.0))
    target_half, target_geometry_scale, primitive_extraction_distance = (
        _fit_target_to_primitive_envelope(
            raw_target_half,
            obstacle_opening_mismatch=channel_outer_open - channel_open,
            target_open_clearance=target_open_clearance,
        )
    )
    _, target_lateral_half, target_extraction_half = np.sort(target_half)
    manipulation = strategy_3_manipulation_metrics(
        2.0 * target_half,
        gripper_max_width_m=STRATEGY_3_GRIPPER_MAX_WIDTH_M,
    )
    if not manipulation["passed"]:
        raise ValueError(
            "strategy-3 target is not post-flip graspable: "
            "post_flip_grasp_thickness"
        )
    channel_clear_width = 2.0 * target_lateral_half + 2.0 * (
        SIDE_CLEARANCE + 0.004 * (0.5 * (c[3] + 1.0))
    )
    left_center = (
        row_center
        - lateral
        * (0.5 * channel_clear_width + obstacle_records[0][2])
        + extraction_axis * left_axial_offset
    )
    right_center = (
        row_center
        + lateral
        * (0.5 * channel_clear_width + obstacle_records[1][2])
        + extraction_axis * right_axial_offset
    )
    for record, center in zip(obstacle_records, (left_center, right_center)):
        editor.place_box_collision_on_floor(
            record[0],
            center_xy=(float(center[0]), float(center[1])),
            world_rotation=record[1],
        )

    target_offset = (
        channel_open - target_extraction_half - target_open_clearance
    )
    target_back_clearance = (
        target_offset - target_extraction_half - channel_back
    )
    if target_back_clearance < 0.004:
        raise ValueError(
            "strategy-3 target does not fit completely inside the sampled corridor"
        )
    target_center = row_center + extraction_axis * target_offset

    _, _, fixture_half = editor.collision_world_geometry(fixture)
    fixture_half = np.asarray(fixture_half, dtype=np.float64).copy()
    fixture_vertical_index = int(
        np.argsort(fixture_half, kind="stable")[0]
    )
    maximum_fixture_half_height = (
        target_lateral_half - 0.5 * DESIGN_POST_FLIP_EDGE_EXPOSURE
    )
    if maximum_fixture_half_height < 0.5 * MINIMUM_FLIP_WALL_HEIGHT:
        raise ValueError(
            "strategy-3 target is too short to expose a post-flip grasp edge"
        )
    fixture_half[fixture_vertical_index] = min(
        fixture_half[fixture_vertical_index],
        maximum_fixture_half_height,
    )
    editor.resize_static_canonical_box(fixture, fixture_half)
    fixture_rotation, fixture_normal_half, fixture_axis_half, fixture_height = (
        semantic_box_rotation(
            fixture_half,
            lateral=lateral,
            forward=extraction_axis,
            # Match the demo: the large panel lies on its broad face, so its
            # shortest dimension is vertical.  Standing the middle dimension
            # upright hides the target's top edge after the flip.
            lateral_rank=1,
            vertical_rank=0,
        )
    )
    transfer_gap = 0.065 + 0.020 * (0.5 * (c[5] + 1.0))
    wall_channel_clearance = 0.012 + 0.008 * (0.5 * (c[6] + 1.0))
    wall_axis_margin = 0.020
    extraction_distance = (
        2.0 * target_extraction_half
        + target_open_clearance
        + (channel_outer_open - channel_open)
        + wall_channel_clearance
        + wall_axis_margin
    )
    fixture_center = (
        row_center
        + extraction_axis
        * (channel_outer_open + wall_channel_clearance + fixture_axis_half)
        + lateral
        * (target_lateral_half + transfer_gap + fixture_normal_half)
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(fixture_center[0]), float(fixture_center[1])),
        world_rotation=fixture_rotation,
    )

    target_yaw = task_yaw + math.pi
    editor.layout_record["extraction_channel"] = {
        "channel_center_xy": row_center.tolist(),
        "extraction_axis_xy": extraction_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "clear_width": channel_clear_width,
        "target_lateral_width": 2.0 * float(target_lateral_half),
        "target_extraction_half_extent": float(target_extraction_half),
        "target_initial_center_xy": target_center.tolist(),
        "left_center_xy": left_center.tolist(),
        "right_center_xy": right_center.tolist(),
        "fixture_center_xy": fixture_center.tolist(),
        "fixture_axis_half_extent": fixture_axis_half,
        "fixture_height": 2.0 * fixture_height,
        "post_flip_target_height": 2.0 * float(target_lateral_half),
        "post_flip_edge_exposure": (
            2.0 * float(target_lateral_half) - 2.0 * fixture_height
        ),
        "channel_back_coordinate": channel_back,
        "channel_open_coordinate": channel_open,
        "channel_outer_open_coordinate": channel_outer_open,
        "channel_overlap_depth": channel_open - channel_back,
        "target_open_edge_clearance": target_open_clearance,
        "target_back_edge_clearance": target_back_clearance,
        "target_extraction_offset": target_offset,
        "target_planar_area": float(
            4.0 * target_lateral_half * target_extraction_half
        ),
        "raw_target_half_extents": raw_target_half.tolist(),
        "target_geometry_scale": target_geometry_scale,
        "primitive_extraction_distance": primitive_extraction_distance,
        "primitive_extraction_domain": [
            STRATEGY_3_MIN_EXTRACTION_DISTANCE_M,
            STRATEGY_3_MAX_EXTRACTION_DISTANCE_M,
        ],
        "manipulation_feasibility": manipulation,
        "minimum_extraction_distance": extraction_distance,
        "wall_channel_clearance": wall_channel_clearance,
        "wall_transfer_gap": transfer_gap,
    }
    # A fixed world-frame camera makes translation and yaw variation visible
    # in the gallery instead of tracking every sampled target pose.
    editor.layout_record["fixed_default_camera"] = True
    extracted_center = target_center + extraction_axis * extraction_distance
    wall_contact_center = extracted_center + lateral * transfer_gap
    editor.layout_record["extraction_channel"]["extracted_center_xy"] = (
        extracted_center.tolist()
    )
    editor.layout_record["extraction_channel"]["wall_contact_center_xy"] = (
        wall_contact_center.tolist()
    )
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        extracted_center.tolist(),
        wall_contact_center.tolist(),
    ]
    relation = {
        "kind": "extraction_then_wall_setup",
        "target": contract.target_body,
        "left": left,
        "right": right,
        "fixture": fixture,
        "extraction_axis_xy": extraction_axis.tolist(),
        "lateral_xy": lateral.tolist(),
        "required_clear_width": channel_clear_width - 1e-4,
        "maximum_extra_width": 0.040,
        "minimum_open_path": extraction_distance,
        "minimum_primitive_extraction_distance": (
            STRATEGY_3_MIN_EXTRACTION_DISTANCE_M
        ),
        "maximum_primitive_extraction_distance": (
            STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M
        ),
        "primitive_extraction_runtime_maximum": (
            STRATEGY_3_MAX_EXTRACTION_DISTANCE_M
        ),
        "minimum_target_planar_area": 0.032,
        "minimum_initial_open_edge_clearance": target_open_clearance - 0.004,
        "maximum_initial_open_edge_clearance": target_open_clearance + 0.004,
        "wall_orientation": "parallel_to_extraction",
        "wall_normal_xy": lateral.tolist(),
        "minimum_wall_transfer": transfer_gap - 0.004,
        "maximum_wall_transfer": transfer_gap + 0.004,
        "minimum_wall_axis_margin": wall_axis_margin - 0.004,
        "minimum_flip_wall_height": MINIMUM_FLIP_WALL_HEIGHT,
        "minimum_post_flip_edge_exposure": (
            MINIMUM_POST_FLIP_EDGE_EXPOSURE
        ),
    }
    return ContextResult(
        (fixture,),
        (left, right),
        None,
        dict(editor.entity_renames),
        {
            "fixture_0": fixture_label,
            "secondary_0": left_label,
            "secondary_1": right_label,
        },
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
        target_half_extents_override=tuple(float(value) for value in target_half),
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(3))
