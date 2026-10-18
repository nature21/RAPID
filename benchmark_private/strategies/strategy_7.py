"""Strategy 7: topple one tightly packed upright row member outward."""

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


REFERENCE_ROW_CENTER = np.asarray((0.64, 0.14), dtype=np.float64)
FIXTURE_THICKNESS_METERS = 0.060
FIXTURE_HEIGHT_METERS = 0.115
FIXTURE_REAR_MARGIN_METERS = 0.050
FIXTURE_OPENING_EDGE_MARGIN_METERS = 0.012
MIN_TARGET_ROW_WIDTH_METERS = 0.070
MAX_TARGET_GRASP_THICKNESS_METERS = 0.0755
MIN_TARGET_OUTWARD_DEPTH_METERS = 0.180
MIN_ROW_HEIGHT_METERS = 0.230
MIN_NEIGHBOR_ROW_WIDTH_METERS = 0.070
MAX_NEIGHBOR_ROW_WIDTH_METERS = 0.100
MAX_INITIAL_EDGE_EXPOSURE_METERS = 0.002
MAX_DIRECT_GRASP_SIDE_GAP_METERS = 0.005


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    left_fixture, _ = editor.apply_fixture_material(
        "left_white_divider",
        canonical_box_visual=True,
    )
    right_fixture, _ = editor.apply_fixture_material(
        "right_white_divider",
        palette_offset=3,
        canonical_box_visual=True,
    )
    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_order = np.argsort(np.asarray(target_half), kind="stable")
    target_half = np.asarray(target_half).copy()
    target_half[int(target_order[0])] = max(
        float(target_half[int(target_order[0])]),
        0.5 * MIN_TARGET_ROW_WIDTH_METERS,
    )
    target_half[int(target_order[1])] = max(
        float(target_half[int(target_order[1])]),
        0.5 * MIN_TARGET_OUTWARD_DEPTH_METERS,
    )
    target_half[int(target_order[2])] = max(
        float(target_half[int(target_order[2])]),
        0.5 * MIN_ROW_HEIGHT_METERS,
    )
    target_sorted_half = np.sort(target_half)
    source_row = (
        "left_dark_food_box",
        "purple_food_box",
        "tall_red_food_box",
        "brown_food_box",
    )
    secondaries: list[str] = []
    labels: list[str] = []
    for source, appearance in zip(
        source_row,
        editor.parameters.secondary_appearances,
    ):
        secondary_half = editor.appearance_half_extents(
            appearance,
            (1.0, 1.0, 1.0),
            broad_primitive_scale=False,
        )
        secondary_order = np.argsort(
            np.asarray(secondary_half),
            kind="stable",
        )
        secondary_half = np.asarray(secondary_half).copy()
        secondary_half[int(secondary_order[0])] = np.clip(
            float(secondary_half[int(secondary_order[0])]),
            0.5 * MIN_NEIGHBOR_ROW_WIDTH_METERS,
            0.5 * MAX_NEIGHBOR_ROW_WIDTH_METERS,
        )
        # A flush top and robot-side front edge remove the initially exposed
        # target edge. Toppling is what creates the first graspable edge.
        secondary_half[int(secondary_order[1])] = target_sorted_half[1]
        secondary_half[int(secondary_order[2])] = target_sorted_half[2]
        renamed, label = editor.reskin_dynamic(
            source,
            appearance,
            canonical_box=True,
            half_extents_override=secondary_half,
        )
        secondaries.append(renamed)
        labels.append(label)

    task_yaw = editor.parameters.yaw_radians + math.radians(3.0 * c[2])
    lateral, rearward = planar_axes(task_yaw)
    outward = -rearward
    row_center = (
        REFERENCE_ROW_CENTER
        + 0.008 * c[0] * lateral
        + 0.008 * c[1] * outward
    )
    ordered_names = [secondaries[0], secondaries[1], contract.target_body]
    ordered_names.extend((secondaries[2], secondaries[3]))
    ordered_half_extents: list[np.ndarray] = []
    ordered_rotations: list[np.ndarray] = []
    lateral_halves: list[float] = []
    forward_halves: list[float] = []
    vertical_halves: list[float] = []
    for name in ordered_names:
        half_extents = (
            target_half
            if name == contract.target_body
            else editor.collision_world_geometry(name)[2]
        )
        rotation, lateral_half, forward_half, vertical_half = semantic_box_rotation(
            half_extents,
            lateral=lateral,
            forward=rearward,
            # Pack the row like books on a shelf: each box contributes its
            # shortest (thickness) dimension along the row, while its broad
            # face points toward the topple direction.
            lateral_rank=0,
            vertical_rank=2,
        )
        ordered_half_extents.append(half_extents)
        ordered_rotations.append(rotation)
        lateral_halves.append(lateral_half)
        forward_halves.append(forward_half)
        vertical_halves.append(vertical_half)

    gap = 0.0025 + 0.0015 * (0.5 * (c[3] + 1.0))
    total_width = 2.0 * sum(lateral_halves) + gap * (
        len(ordered_names) - 1
    )
    cursor = -0.5 * total_width
    centers: list[np.ndarray] = []
    for lateral_half in lateral_halves:
        coordinate = cursor + lateral_half
        centers.append(row_center + lateral * coordinate)
        cursor += 2.0 * lateral_half + gap

    for name, center, rotation in zip(
        ordered_names,
        centers,
        ordered_rotations,
    ):
        if name == contract.target_body:
            continue
        editor.place_box_collision_on_floor(
            name,
            center_xy=(float(center[0]), float(center[1])),
            world_rotation=rotation,
        )

    # These fixtures are low, shallow bookends, not full-height corridor
    # walls.  Their robot-side edges sit just beyond the row's front faces so
    # the topple and subsequent grasp do not have to pass between long walls.
    maximum_row_forward_half = max(forward_halves)
    fixture_forward_half = (
        maximum_row_forward_half + 0.5 * FIXTURE_REAR_MARGIN_METERS
    )
    fixture_records: list[tuple[str, np.ndarray, float, float, float]] = []
    for name in (left_fixture, right_fixture):
        _, _, half_extents = editor.collision_world_geometry(name)
        order = np.argsort(np.asarray(half_extents), kind="stable")
        half_extents = np.asarray(half_extents).copy()
        half_extents[int(order[0])] = 0.5 * FIXTURE_THICKNESS_METERS
        half_extents[int(order[1])] = 0.5 * FIXTURE_HEIGHT_METERS
        half_extents[int(order[2])] = fixture_forward_half
        editor.resize_static_canonical_box(name, half_extents)
        rotation, thickness, forward_half, vertical_half = semantic_box_rotation(
            half_extents,
            lateral=lateral,
            forward=rearward,
            lateral_rank=0,
            vertical_rank=1,
        )
        fixture_records.append(
            (name, rotation, thickness, forward_half, vertical_half)
        )
    side_clearance = 0.008 + 0.004 * (0.5 * (c[4] + 1.0))
    fixture_forward_shift = (
        maximum_row_forward_half
        + FIXTURE_OPENING_EDGE_MARGIN_METERS
        - fixture_forward_half
    )
    left_wall_center = row_center - lateral * (
        0.5 * total_width + side_clearance + fixture_records[0][2]
    ) + outward * fixture_forward_shift
    right_wall_center = row_center + lateral * (
        0.5 * total_width + side_clearance + fixture_records[1][2]
    ) + outward * fixture_forward_shift
    for record, center in zip(
        fixture_records,
        (left_wall_center, right_wall_center),
    ):
        editor.place_box_collision_on_floor(
            record[0],
            center_xy=(float(center[0]), float(center[1])),
            world_rotation=record[1],
        )

    target_center = centers[2]
    target_yaw = task_yaw
    required_outward_clearance = 2.0 * vertical_halves[2] + 0.030
    editor.layout_record["packed_row"] = {
        "ordered_bodies": ordered_names,
        "centers_xy": [center.tolist() for center in centers],
        "lateral_xy": lateral.tolist(),
        "outward_xy": outward.tolist(),
        "row_axis_dimension_rank": 0,
        "outward_axis_dimension_rank": 1,
        "vertical_axis_dimension_rank": 2,
        "inter_object_gap": gap,
        "side_clearance": side_clearance,
        "total_width": total_width,
        "target_index": 2,
        "target_forward_half_extent": forward_halves[2],
        "target_vertical_half_extent": vertical_halves[2],
        "target_row_width": 2.0 * lateral_halves[2],
        "target_outward_depth": 2.0 * forward_halves[2],
        "uniform_row_height": 2.0 * vertical_halves[2],
        "required_outward_clearance": required_outward_clearance,
        "fixture_height": FIXTURE_HEIGHT_METERS,
        "fixture_depth": 2.0 * fixture_forward_half,
        "fixture_opening_edge_margin": FIXTURE_OPENING_EDGE_MARGIN_METERS,
        "left_wall_center_xy": left_wall_center.tolist(),
        "right_wall_center_xy": right_wall_center.tolist(),
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        (
            target_center + outward * required_outward_clearance
        ).tolist(),
    ]
    relation = {
        "kind": "packed_upright_row",
        "target": contract.target_body,
        "ordered_bodies": ordered_names,
        "left_fixture": left_fixture,
        "right_fixture": right_fixture,
        "lateral_xy": lateral.tolist(),
        "outward_xy": outward.tolist(),
        "row_axis_dimension_rank": 0,
        "outward_axis_dimension_rank": 1,
        "vertical_axis_dimension_rank": 2,
        "minimum_axis_alignment": 0.995,
        "minimum_neighbor_gap": gap - 5e-4,
        "maximum_neighbor_gap": gap + 0.003,
        "minimum_outward_clearance": required_outward_clearance,
        "minimum_target_topple_contact_width": (
            MIN_TARGET_ROW_WIDTH_METERS - 0.001
        ),
        "maximum_target_grasp_thickness": (
            MAX_TARGET_GRASP_THICKNESS_METERS
        ),
        "minimum_target_topple_face_depth": (
            MIN_TARGET_OUTWARD_DEPTH_METERS - 0.002
        ),
        "maximum_row_height_spread": MAX_INITIAL_EDGE_EXPOSURE_METERS,
        "maximum_target_top_exposure": MAX_INITIAL_EDGE_EXPOSURE_METERS,
        "maximum_target_front_exposure": MAX_INITIAL_EDGE_EXPOSURE_METERS,
        "maximum_direct_grasp_side_gap": MAX_DIRECT_GRASP_SIDE_GAP_METERS,
        "maximum_fixture_height": FIXTURE_HEIGHT_METERS + 0.001,
        "maximum_fixture_depth": 2.0 * fixture_forward_half + 0.001,
        "minimum_fixture_opening_edge_margin": (
            FIXTURE_OPENING_EDGE_MARGIN_METERS - 0.002
        ),
        "maximum_fixture_opening_edge_margin": (
            FIXTURE_OPENING_EDGE_MARGIN_METERS + 0.002
        ),
    }
    return ContextResult(
        (left_fixture, right_fixture),
        tuple(secondaries),
        None,
        dict(editor.entity_renames),
        {f"secondary_{index}": label for index, label in enumerate(labels)},
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
        0,
        2,
        target_half_extents_override=tuple(
            float(value) for value in target_half
        ),
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(7))
