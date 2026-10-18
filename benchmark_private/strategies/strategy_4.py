"""Strategy 4: wall-assisted flip, clearance pull, and edge grasp."""

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


REFERENCE_WALL_CENTER = np.asarray((0.75, 0.19), dtype=np.float64)
TARGET_FLIP_HEIGHT_SCALE = 0.65


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    fixture, fixture_label = editor.apply_fixture_material(
        "white_divider",
        canonical_box_visual=True,
    )
    task_yaw = editor.parameters.yaw_radians + math.radians(12.0 * c[2])
    wall_normal, wall_axis = planar_axes(task_yaw)
    wall_center = (
        REFERENCE_WALL_CENTER
        + np.asarray(editor.parameters.translation_xy)
        + np.asarray((0.018 * c[0], 0.018 * c[1]))
    )
    _, _, fixture_half = editor.collision_world_geometry(fixture)
    fixture_rotation, fixture_normal_half, fixture_axis_half, fixture_height = (
        semantic_box_rotation(
            fixture_half,
            lateral=wall_normal,
            forward=wall_axis,
            lateral_rank=0,
            vertical_rank=1,
        )
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(wall_center[0]), float(wall_center[1])),
        world_rotation=fixture_rotation,
    )

    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_half = target_half.copy()
    target_middle_axis = int(np.argsort(target_half)[1])
    target_half[target_middle_axis] *= TARGET_FLIP_HEIGHT_SCALE
    target_vertical, target_normal_half, target_axis_half = np.sort(target_half)
    face_gap = 0.115 + 0.045 * (0.5 * (c[3] + 1.0))
    target_center = (
        wall_center
        - wall_normal * (fixture_normal_half + target_normal_half + face_gap)
        + wall_axis * (0.030 * c[4])
    )
    target_yaw = task_yaw + math.radians(3.0 * c[5])
    pull_clearance = 0.055 + 0.025 * (0.5 * (c[6] + 1.0))
    target_axis_offset = 0.030 * c[4]
    flat_wall_contact_center = (
        wall_center
        - wall_normal * (fixture_normal_half + target_normal_half)
        + wall_axis * target_axis_offset
    )
    upright_wall_contact_center = (
        wall_center
        - wall_normal * (fixture_normal_half + target_vertical)
        + wall_axis * target_axis_offset
    )
    pulled_center = upright_wall_contact_center - wall_normal * pull_clearance

    editor.layout_record["wall_flip"] = {
        "wall_center_xy": wall_center.tolist(),
        "wall_normal_xy": wall_normal.tolist(),
        "wall_axis_xy": wall_axis.tolist(),
        "wall_axis_half_extent": fixture_axis_half,
        "wall_height": 2.0 * fixture_height,
        "target_center_xy": target_center.tolist(),
        "target_initial_vertical_half_extent": float(target_vertical),
        "target_flip_height_scale": TARGET_FLIP_HEIGHT_SCALE,
        # The target flips about its long wall-parallel axis, so its middle
        # dimension—not its long dimension—becomes vertical.
        "target_upright_height": 2.0 * float(target_normal_half),
        "initial_face_gap": face_gap,
        "required_clearance_pull": pull_clearance,
        "flat_wall_contact_center_xy": flat_wall_contact_center.tolist(),
        "upright_wall_contact_center_xy": upright_wall_contact_center.tolist(),
        "pulled_center_xy": pulled_center.tolist(),
    }
    editor.layout_record["manipulation_waypoints_xy"] = [
        target_center.tolist(),
        flat_wall_contact_center.tolist(),
        upright_wall_contact_center.tolist(),
        pulled_center.tolist(),
    ]
    relation = {
        "kind": "wall_flip_setup",
        "target": contract.target_body,
        "fixture": fixture,
        "wall_normal_xy": wall_normal.tolist(),
        "wall_axis_xy": wall_axis.tolist(),
        "minimum_face_gap": 0.105,
        "maximum_face_gap": 0.17,
        "minimum_wall_axis_margin": 0.012,
        "minimum_wall_height": max(
            0.14,
            1.5 * float(target_normal_half),
        ),
        "required_clearance_pull": pull_clearance,
        "minimum_pull_clearance": 0.050,
    }
    return ContextResult(
        (fixture,),
        (),
        None,
        dict(editor.entity_renames),
        {"fixture_0": fixture_label},
        (relation,),
        tuple(editor.object_updates),
        dict(editor.layout_record),
        (float(target_center[0]), float(target_center[1])),
        target_yaw,
        True,
        target_half_extents_override=tuple(float(value) for value in target_half),
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(4))
