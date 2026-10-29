"""Progress logging helpers for the real2sim pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim.schema import PipelineConfig, ReconstructedObject


def log(config: PipelineConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def export_mode_label(config: PipelineConfig) -> str:
    if not config.dynamic_objects:
        return "static bodies, 3D oriented-box collisions"
    return "dynamic free bodies, 3D oriented-box collisions"


def log_static_selections(
    config: PipelineConfig,
    static_selections: list[dict[str, Any]],
) -> None:
    if not config.verbose:
        return
    static_objects = [
        selection
        for selection in static_selections
        if bool(selection.get("is_static", False))
    ]
    if not static_selections:
        print("[real2sim] static object selection: none", flush=True)
        return
    if not static_objects:
        print("[real2sim] static object selection: all objects dynamic", flush=True)
        return
    print("[real2sim] static object selection:", flush=True)
    for selection in static_objects:
        print(
            "  "
            f"{selection['object_name']} ({selection['confidence']}) - "
            f"{selection.get('reason', '')}",
            flush=True,
        )


def log_physical_parameters(
    config: PipelineConfig,
    physical_parameters: dict[str, Any],
) -> None:
    if not config.verbose:
        return
    support = physical_parameters.get("support_surface")
    objects = physical_parameters.get("objects", [])
    if not support and not objects:
        print("[real2sim] physical parameter estimates: defaults", flush=True)
        return
    print("[real2sim] physical parameter estimates:", flush=True)
    if isinstance(support, dict):
        print(
            "  support "
            f"{support.get('object_name', '')}: "
            f"material={support.get('material', '')}, "
            f"friction={physics_friction_summary(support)}, "
            f"restitution={float(support.get('restitution', 0.0)):.3g} "
            f"({support.get('confidence', '')})",
            flush=True,
        )
    for item in objects if isinstance(objects, list) else []:
        if not isinstance(item, dict):
            continue
        mass = item.get("mass_kg")
        mass_text = "fixed" if mass is None else f"{float(mass):.3g}kg"
        print(
            "  "
            f"{item.get('object_name', '')}: mass={mass_text}, "
            f"material={item.get('material', '')}, "
            f"friction={physics_friction_summary(item)}, "
            f"restitution={float(item.get('restitution', 0.0)):.3g} "
            f"({item.get('confidence', '')})",
            flush=True,
        )


def physics_friction_summary(item: dict[str, Any]) -> str:
    values = item.get("friction")
    if values is None:
        values = [
            item.get("sliding_friction", 0.0),
            item.get("torsional_friction", 0.0),
            item.get("rolling_friction", 0.0),
        ]
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=0.0)
    return "[" + ", ".join(f"{float(v):.3g}" for v in arr[:3]) + "]"


def log_world_frame(config: PipelineConfig, metadata: dict[str, Any]) -> None:
    if not config.verbose:
        return
    plane = metadata.get("plane")
    source = metadata.get("source", metadata.get("mode"))
    if isinstance(plane, dict):
        normal = np.asarray(plane.get("normal_camera_opencv"), dtype=float)
        floor_z = metadata.get("support_floor_z")
        floor_text = "" if floor_z is None else f", floor_z={float(floor_z):.4g}"
        plane_world = metadata.get("support_plane_world")
        tilt_text = ""
        if isinstance(plane_world, dict) and plane_world.get(
            "tilt_degrees_from_world_z"
        ) is not None:
            tilt_text = (
                ", plane_tilt="
                f"{float(plane_world['tilt_degrees_from_world_z']):.2f}deg"
            )
        print(
            "[real2sim] world frame: "
            f"source={source}, normal_camera_opencv="
            f"{[round(float(v), 4) for v in normal.reshape(-1)]}, "
            f"inlier_ratio={float(plane.get('inlier_ratio', 0.0)):.3f}, "
            f"residual_p95={float(plane.get('residual_p95', 0.0)):.4g}"
            f"{floor_text}{tilt_text}",
            flush=True,
        )
    else:
        print(f"[real2sim] world frame: source={source}", flush=True)


def log_depth_camera(
    config: PipelineConfig,
    depth: np.ndarray,
    K: np.ndarray,
    c2w_opencv: np.ndarray,
    intrinsics_source: str,
    extrinsics_source: str,
) -> None:
    if not config.verbose:
        return
    finite = np.isfinite(depth)
    if np.any(finite):
        stats = (
            f"min={float(np.nanmin(depth)):.4g}, "
            f"max={float(np.nanmax(depth)):.4g}, "
            f"mean={float(np.nanmean(depth)):.4g}"
        )
    else:
        stats = "all invalid"
    print(f"[real2sim] depth: shape={depth.shape}, {stats}", flush=True)
    print(
        "[real2sim] intrinsics "
        f"source={intrinsics_source}, fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, "
        f"cx={K[0,2]:.3f}, cy={K[1,2]:.3f}",
        flush=True,
    )
    print(
        "[real2sim] extrinsics "
        f"source={extrinsics_source}, camera_pos={c2w_opencv[:3, 3].tolist()}",
        flush=True,
    )


def log_object_result(config: PipelineConfig, obj: ReconstructedObject) -> None:
    if not config.verbose:
        return
    grounding = ""
    if obj.grounding_query and obj.grounding_query_source != "object_name":
        grounding = f", grounding={obj.grounding_query!r}"
    print(
        "[real2sim] object done: "
        f"{obj.safe_name}, mask_area={obj.mask_area_px}, "
        f"sam3_score={obj.sam3_score:.3f}, pose_source={obj.pose_source}, "
        f"body_pos={[round(v, 4) for v in obj.body_pos]}{grounding}",
        flush=True,
    )
