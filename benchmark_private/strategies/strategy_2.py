"""Strategy 2: push a flat box to a wall, flip it upright, and lift."""

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


REFERENCE_TARGET_CENTER = np.asarray((0.60, 0.02), dtype=np.float64)
ROBOTWARD_LAYOUT_TRANSLATION = np.asarray((-0.10, 0.0), dtype=np.float64)
TARGET_WORKSPACE_X = (0.30, 1.02)
TARGET_WORKSPACE_Y = (-0.38, 0.46)
TARGET_WORKSPACE_MARGIN = 0.004


def apply(editor: SceneEditor, contract: StrategyContract) -> ContextResult:
    c = editor.values
    fixture, fixture_label = editor.apply_fixture_material(
        "white_drawer",
        canonical_box_visual=True,
    )
    # The sampled yaw is the complete task-frame rotation.  Keeping it in one
    # explicit parameter makes the advertised +/-90 degree envelope exact
    # instead of silently adding another context-dependent rotation.
    task_yaw = editor.parameters.yaw_radians
    wall_normal, wall_axis = planar_axes(task_yaw)
    base_global_offset = np.asarray(editor.parameters.translation_xy) + np.asarray(
        (0.018 * c[0], 0.018 * c[1])
    )
    _, _, fixture_half = editor.collision_world_geometry(fixture)
    fixture_rotation, fixture_normal_half, fixture_axis_half, fixture_height = (
        semantic_box_rotation(
            fixture_half,
            lateral=wall_normal,
            forward=wall_axis,
            lateral_rank=1,
            vertical_rank=0,
        )
    )
    target_half = editor.appearance_half_extents(
        editor.parameters.appearance,
        editor.parameters.size_scale_xyz,
        broad_primitive_scale=True,
    )
    target_vertical, target_normal_half, target_axis_half = np.sort(target_half)
    face_gap = 0.13 + 0.045 * (0.5 * (c[3] + 1.0))
    target_yaw = task_yaw + math.radians(3.0 * c[5])
    target_lateral, target_axis = planar_axes(target_yaw)
    target_world_half = (
        np.abs(target_lateral) * target_normal_half
        + np.abs(target_axis) * target_axis_half
    )
    requested_translation = ROBOTWARD_LAYOUT_TRANSLATION
    base_target_center = (
        REFERENCE_TARGET_CENTER
        + base_global_offset
        + wall_axis * (0.035 * c[4])
    )
    provisional_target_center = base_target_center + requested_translation
    target_center_lower = np.asarray(
        (
            TARGET_WORKSPACE_X[0] + target_world_half[0] + TARGET_WORKSPACE_MARGIN,
            TARGET_WORKSPACE_Y[0] + target_world_half[1] + TARGET_WORKSPACE_MARGIN,
        )
    )
    target_center_upper = np.asarray(
        (
            TARGET_WORKSPACE_X[1] - target_world_half[0] - TARGET_WORKSPACE_MARGIN,
            TARGET_WORKSPACE_Y[1] - target_world_half[1] - TARGET_WORKSPACE_MARGIN,
        )
    )
    workspace_correction = (
        np.clip(
            provisional_target_center,
            target_center_lower,
            target_center_upper,
        )
        - provisional_target_center
    )
    global_offset = base_global_offset + requested_translation + workspace_correction
    anchor = REFERENCE_TARGET_CENTER + global_offset
    target_center = anchor + wall_axis * (0.035 * c[4])
    wall_center = anchor + wall_normal * (
        fixture_normal_half + target_normal_half + face_gap
    )
    editor.place_box_collision_on_floor(
        fixture,
        center_xy=(float(wall_center[0]), float(wall_center[1])),
        world_rotation=fixture_rotation,
    )
    editor.layout_record["wall_flip"] = {
        "wall_center_xy": wall_center.tolist(),
        "wall_normal_xy": wall_normal.tolist(),
        "wall_axis_xy": wall_axis.tolist(),
        "wall_axis_half_extent": fixture_axis_half,
        "wall_height": 2.0 * fixture_height,
        "target_center_xy": target_center.tolist(),
        "requested_task_translation_xy": requested_translation.tolist(),
        "workspace_correction_xy": workspace_correction.tolist(),
        "target_initial_vertical_half_extent": float(target_vertical),
        "target_upright_height": 2.0 * float(target_axis_half),
        "initial_face_gap": face_gap,
    }
    relation = {
        "kind": "wall_flip_setup",
        "target": contract.target_body,
        "fixture": fixture,
        "wall_normal_xy": wall_normal.tolist(),
        "wall_axis_xy": wall_axis.tolist(),
        "minimum_face_gap": 0.12,
        "maximum_face_gap": 0.19,
        "minimum_wall_axis_margin": 0.015,
        "minimum_wall_height": 0.075,
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
    )


if __name__ == "__main__":
    raise SystemExit(run_strategy_cli(2))
