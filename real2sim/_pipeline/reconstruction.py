"""Object reconstruction scheduling for the real2sim pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any
import traceback

import numpy as np
from PIL import Image

from real2sim.camera import T_OPENGL_TO_OPENCV, T_P3D_TO_OPENCV
from real2sim._pipeline.progress import log, log_object_result
from real2sim._shared.naming import safe_identifier, same_object_name
from real2sim._stages.reconstruction import (
    place_reconstructed_geometry,
    reconstruct_grounded_object_geometry,
    segment_object_grounding,
)
from real2sim.schema import PipelineConfig, ReconstructedObject


def reconstruct_scene_objects(
    *,
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    K: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
    pointmap_p3d: np.ndarray,
    meshes_dir: Path,
    masks_dir: Path,
    world_frame_metadata: dict[str, Any],
    config: PipelineConfig,
) -> tuple[list[ReconstructedObject], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_names: set[str] = set()
    support_object_name = selected_support_object_name(world_frame_metadata)
    sam3d_urls = sam3d_urls_for_reconstruction(config)

    for index, obj in enumerate(detected_objects):
        name = str(obj.get("name", "")).strip()
        if not name:
            skipped.append({"name": name, "reason": "empty object name"})
            continue
        if (
            not config.reconstruct_support_surface
            and support_object_name is not None
            and same_object_name(name, support_object_name)
        ):
            skipped.append(
                {
                    "name": name,
                    "reason": (
                        "selected support surface; represented by fitted MuJoCo "
                        "floor"
                    ),
                }
            )
            log(
                config,
                f"[real2sim] skipped support surface object {name!r}; "
                "using fitted floor instead",
            )
            continue

        safe_name = safe_identifier(name, used_names)
        sam3d_url = sam3d_urls[len(tasks) % len(sam3d_urls)]
        tasks.append(
            {
                "index": index,
                "object_info": obj,
                "name": name,
                "safe_name": safe_name,
                "sam3d_url": sam3d_url,
            }
        )

    if not tasks:
        return [], skipped

    workers = min(max(1, int(config.reconstruction_workers)), len(tasks))
    if workers > 1:
        log(
            config,
            "[real2sim] reconstructing "
            f"{len(tasks)} objects in parallel with {workers} workers "
            f"across {len(sam3d_urls)} SAM3D endpoint(s)",
        )

    placement_kwargs = {
        "K": K,
        "color": color,
        "depth": depth,
        "T_world_from_camera_opencv": T_world_from_camera_opencv,
        "processed_camera_mesh_to_world": (
            T_world_from_camera_opencv @ T_OPENGL_TO_OPENCV
        ),
        "raw_camera_mesh_to_world": T_world_from_camera_opencv @ T_P3D_TO_OPENCV,
        "meshes_dir": meshes_dir,
    }

    grounding_kwargs = {
        "image": image,
        "detected_objects": detected_objects,
    }
    outcomes = _run_tasks(
        tasks,
        workers=workers,
        fn=_ground_one_object_task,
        config=config,
        shared_kwargs=grounding_kwargs,
    )
    skipped.extend(
        outcome["skipped"]
        for outcome in outcomes
        if outcome.get("skipped") is not None
    )
    grounded_tasks = [
        outcome for outcome in outcomes if outcome.get("grounding") is not None
    ]
    grounded_tasks, conflict_skipped = _resolve_grounding_conflicts(
        grounded_tasks,
        image=image,
        detected_objects=detected_objects,
        config=config,
    )
    skipped.extend(conflict_skipped)

    geometry_kwargs = {
        "image": image,
        "pointmap_p3d": pointmap_p3d,
        "masks_dir": masks_dir,
    }
    geometry_outcomes = _run_tasks(
        grounded_tasks,
        workers=min(workers, max(1, len(grounded_tasks))),
        fn=_reconstruct_grounded_object_task,
        config=config,
        shared_kwargs=geometry_kwargs,
    )
    skipped.extend(
        outcome["skipped"]
        for outcome in geometry_outcomes
        if outcome.get("skipped") is not None
    )
    geometries = [
        outcome
        for outcome in geometry_outcomes
        if outcome.get("geometry") is not None
    ]
    reconstructed: list[ReconstructedObject] = []
    if config.use_foundation_pose and geometries:
        log(
            config,
            "[real2sim] running Foundation Pose placement sequentially "
            "after SAM3D reconstruction",
        )
    for outcome in geometries:
        try:
            reconstructed_obj = place_reconstructed_geometry(
                geometry=outcome["geometry"],
                config=config,
                log_fn=lambda message: log(config, message),
                **placement_kwargs,
            )
        except Exception as exc:
            error_info = {
                "name": str(outcome.get("name", "")),
                "safe_name": str(outcome.get("safe_name", "")),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            if config.continue_on_object_error:
                log(
                    config,
                    f"[real2sim] skipped {error_info['name']!r}: "
                    f"{error_info['reason']}",
                )
                error_info["traceback"] = traceback.format_exc()
                skipped.append(error_info)
                continue
            raise RuntimeError(
                f"failed to place reconstructed object {error_info['name']!r}"
            ) from exc
        log_object_result(config, reconstructed_obj)
        reconstructed.append(reconstructed_obj)
    _resolve_stacked_supports(reconstructed, config=config)
    return reconstructed, skipped


def _resolve_stacked_supports(
    objects: list[ReconstructedObject],
    *,
    config: PipelineConfig,
) -> None:
    """Lift movable objects along with raised fixed supports beneath them.

    Each reconstructed mesh is first clamped independently to the foundational
    support plane.  That is correct for a raised platform but can leave an item
    that originally rested on it behind.  Use the registered pre-lift heights
    and XY overlap to identify those contacts, then translate the movable item
    onto the platform's corrected top face.
    """

    if str(config.world_frame_mode).replace("-", "_") != "support_surface":
        return

    supports = [
        obj for obj in objects if obj.is_static and _is_support_like_object(obj)
    ]
    if not supports:
        return

    clearance = max(0.0, float(config.object_support_clearance))
    for obj in objects:
        if obj.is_static:
            continue
        bounds = np.asarray(obj.world_bounds, dtype=float)
        original_min_z = (
            float(obj.world_min_z_before_lift)
            if obj.world_min_z_before_lift is not None
            else float(bounds[0, 2] - obj.support_lift_meters)
        )
        candidates: list[tuple[float, ReconstructedObject]] = []
        for support in supports:
            support_bounds = np.asarray(support.world_bounds, dtype=float)
            overlap = _xy_footprint_overlap_fraction(bounds, support_bounds)
            if overlap < 0.5:
                continue
            original_support_top = float(
                support_bounds[1, 2] - support.support_lift_meters
            )
            if abs(original_min_z - original_support_top) > 0.05:
                continue
            candidates.append((float(support_bounds[1, 2]), support))
        if not candidates:
            continue

        support_top, support = max(candidates, key=lambda item: item[0])
        stack_lift = max(0.0, support_top + clearance - float(bounds[0, 2]))
        if stack_lift <= 0.0:
            continue
        _translate_reconstructed_object(obj, stack_lift)
        obj.support_object_name = support.name
        obj.support_stack_lift_meters += stack_lift
        log(
            config,
            f"[real2sim] stacked object {obj.safe_name} on {support.safe_name}: "
            f"dz={stack_lift:.6g}",
        )


def _is_support_like_object(obj: ReconstructedObject) -> bool:
    text = " ".join(
        (obj.name, obj.description, obj.static_reason)
    ).strip().lower()
    return any(
        term in text
        for term in (
            "platform",
            "table",
            "counter",
            "shelf",
            "tray",
            "mat",
            "pedestal",
            "support surface",
            "supporting surface",
        )
    )


def _xy_footprint_overlap_fraction(
    object_bounds: np.ndarray,
    support_bounds: np.ndarray,
) -> float:
    overlap_min = np.maximum(object_bounds[0, :2], support_bounds[0, :2])
    overlap_max = np.minimum(object_bounds[1, :2], support_bounds[1, :2])
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
    object_size = np.maximum(object_bounds[1, :2] - object_bounds[0, :2], 1e-9)
    return float(np.prod(overlap_size) / np.prod(object_size))


def _translate_reconstructed_object(obj: ReconstructedObject, dz: float) -> None:
    obj.body_pos[2] += dz
    obj.world_bounds[0][2] += dz
    obj.world_bounds[1][2] += dz
    if obj.T_mesh_world is not None:
        obj.T_mesh_world[2][3] += dz
    obj.support_lift_meters += dz
    if obj.world_min_z_after_lift is not None:
        obj.world_min_z_after_lift += dz


def sam3d_urls_for_reconstruction(config: PipelineConfig) -> tuple[str, ...]:
    urls = tuple(
        str(url).strip()
        for url in config.sam3d_server_urls
        if str(url).strip()
    )
    if urls:
        return urls
    return (str(config.sam3d_server_url),)


def selected_support_object_name(world_frame_metadata: dict[str, Any]) -> str | None:
    selection = world_frame_metadata.get("support_selection")
    if not isinstance(selection, dict):
        return None
    name = str(selection.get("object_name", "")).strip()
    return name or None


def _run_tasks(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    fn: Any,
    config: PipelineConfig,
    shared_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if workers <= 1:
        outcomes = [
            fn(task, config=config, **shared_kwargs)
            for task in tasks
        ]
    else:
        outcomes = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    fn,
                    task,
                    config=config,
                    **shared_kwargs,
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
    outcomes.sort(key=lambda item: int(item["index"]))
    return outcomes


def _ground_one_object_task(
    task: dict[str, Any],
    *,
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any]:
    name = str(task["name"])
    safe_name = str(task["safe_name"])
    log(
        config,
        f"[real2sim] grounding object: {name!r} as {safe_name!r}",
    )

    try:
        grounding = segment_object_grounding(
            image=image,
            object_info=task["object_info"],
            detected_objects=detected_objects,
            config=config,
            log_fn=lambda message: log(config, message),
        )
    except Exception as exc:
        error_info = {
            "name": name,
            "safe_name": safe_name,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        if config.continue_on_object_error:
            log(config, f"[real2sim] skipped {name!r}: {error_info['reason']}")
            error_info["traceback"] = traceback.format_exc()
            return {
                "index": task["index"],
                "name": name,
                "safe_name": safe_name,
                "grounding": None,
                "skipped": error_info,
            }
        raise RuntimeError(f"failed to ground {name!r}") from exc

    if grounding is None:
        grounding = task["object_info"].get("_grounding")
        reason = "no usable SAM3 mask"
        if isinstance(grounding, dict):
            reason = str(grounding.get("rejection_reason") or reason)
        skipped_info = _grounding_skip(task, reason, grounding)
        log(config, f"[real2sim] skipped {name!r}: {reason}")
        return {
            "index": task["index"],
            "name": name,
            "safe_name": safe_name,
            "grounding": None,
            "skipped": skipped_info,
        }

    return {**task, "grounding": grounding, "skipped": None}


def _reconstruct_grounded_object_task(
    task: dict[str, Any],
    *,
    image: Image.Image,
    pointmap_p3d: np.ndarray,
    masks_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    name = str(task["name"])
    safe_name = str(task["safe_name"])
    task_config = replace(config, sam3d_server_url=str(task["sam3d_url"]))
    log(
        config,
        f"[real2sim] reconstructing object: {name!r} as {safe_name!r} "
        f"(sam3d={task_config.sam3d_server_url})",
    )
    try:
        geometry = reconstruct_grounded_object_geometry(
            image=image,
            object_info=task["object_info"],
            safe_name=safe_name,
            pointmap_p3d=pointmap_p3d,
            masks_dir=masks_dir,
            grounding=task["grounding"],
            config=task_config,
        )
    except Exception as exc:
        error_info = _grounding_skip(
            task,
            f"{type(exc).__name__}: {exc}",
            task.get("grounding"),
        )
        if config.continue_on_object_error:
            log(config, f"[real2sim] skipped {name!r}: {error_info['reason']}")
            error_info["traceback"] = traceback.format_exc()
            return {**task, "geometry": None, "skipped": error_info}
        raise RuntimeError(f"failed to reconstruct {name!r}") from exc

    grounding = geometry["grounding"]
    log(
        config,
        f"[real2sim] SAM3D done: {safe_name}, "
        f"mask_area={geometry['mask_area_px']}, "
        f"sam3_score={float(grounding['score']):.3f}",
    )
    return {
        **task,
        "geometry": geometry,
        "skipped": None,
    }


def _resolve_grounding_conflicts(
    tasks: list[dict[str, Any]],
    *,
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retry and deterministically remove duplicate or superset object masks."""

    active = list(tasks)
    initial_conflicts = _find_grounding_conflicts(active, config=config)
    retry_indices = sorted(
        {index for first, second, _ in initial_conflicts for index in (first, second)}
    )
    for index in retry_indices:
        task = active[index]
        old_grounding = task["grounding"]
        retry_records = [
            {
                **metrics,
                "with": active[other]["name"],
                "action": "box_conditioned_alias_retry",
            }
            for first, second, metrics in initial_conflicts
            for other in (
                second if first == index else first if second == index else -1,
            )
            if other >= 0
        ]
        old_conflicts = list(old_grounding.get("conflicts", [])) + retry_records
        log(
            config,
            f"[real2sim] retrying conflicting mask for {task['name']!r} "
            "with instance-conditioned aliases",
        )
        try:
            retry = segment_object_grounding(
                image=image,
                object_info=task["object_info"],
                detected_objects=detected_objects,
                config=config,
                force_aliases=True,
                log_fn=lambda message: log(config, message),
            )
        except Exception as exc:
            retry = None
            old_grounding.setdefault("attempts", []).append(
                {
                    "query": "",
                    "query_source": "conflict_retry",
                    "success": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if retry is not None:
            retry["attempts"] = (
                list(old_grounding.get("attempts", []))
                + [{"query": "", "query_source": "conflict_retry", "success": True}]
                + list(retry.get("attempts", []))
            )
            retry["conflicts"] = old_conflicts
            task["grounding"] = retry
        else:
            old_grounding["conflicts"] = old_conflicts

    skipped: list[dict[str, Any]] = []
    while True:
        conflicts = _find_grounding_conflicts(active, config=config)
        if not conflicts:
            break
        first, second, metrics = conflicts[0]
        loser_index, winner_index, resolution = _conflict_resolution(
            active,
            first,
            second,
            metrics,
            config=config,
        )
        loser = active[loser_index]
        winner = active[winner_index]
        record = {
            **metrics,
            "with": winner["name"],
            "winner": winner["name"],
            "loser": loser["name"],
            "action": resolution,
        }
        winner["grounding"].setdefault("conflicts", []).append(record)
        loser["grounding"].setdefault("conflicts", []).append(record)
        reason = (
            f"unresolved grounding conflict with {winner['name']!r}: "
            f"IoU={metrics['iou']:.6f}, "
            f"containment={metrics['containment']:.6f}; {resolution}"
        )
        skipped.append(_grounding_skip(loser, reason, loser["grounding"]))
        log(config, f"[real2sim] skipped {loser['name']!r}: {reason}")
        active.pop(loser_index)

    active.sort(key=lambda item: int(item["index"]))
    return active, skipped


def _find_grounding_conflicts(
    tasks: list[dict[str, Any]],
    *,
    config: PipelineConfig,
) -> list[tuple[int, int, dict[str, Any]]]:
    conflicts = []
    for first in range(len(tasks)):
        for second in range(first + 1, len(tasks)):
            metrics = _mask_overlap_metrics(
                tasks[first]["grounding"]["mask"],
                tasks[second]["grounding"]["mask"],
            )
            if (
                metrics["iou"] >= float(config.duplicate_mask_iou_threshold)
                or metrics["containment"]
                >= float(config.duplicate_mask_containment_threshold)
            ):
                conflicts.append((first, second, metrics))
    return conflicts


def _mask_overlap_metrics(first_mask: np.ndarray, second_mask: np.ndarray) -> dict[str, Any]:
    first = np.asarray(first_mask, dtype=bool)
    second = np.asarray(second_mask, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"mask shape mismatch: {first.shape} != {second.shape}")
    first_area = int(first.sum())
    second_area = int(second.sum())
    intersection = int(np.logical_and(first, second).sum())
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    return {
        "iou": 0.0 if union <= 0 else intersection / union,
        "containment": 0.0 if smaller <= 0 else intersection / smaller,
        "intersection_px": intersection,
        "first_area_px": first_area,
        "second_area_px": second_area,
    }


def _conflict_resolution(
    tasks: list[dict[str, Any]],
    first: int,
    second: int,
    metrics: dict[str, Any],
    *,
    config: PipelineConfig,
) -> tuple[int, int, str]:
    first_area = int(metrics["first_area_px"])
    second_area = int(metrics["second_area_px"])
    containment_only = (
        float(metrics["containment"])
        >= float(config.duplicate_mask_containment_threshold)
        and float(metrics["iou"]) < float(config.duplicate_mask_iou_threshold)
    )
    if containment_only and first_area != second_area:
        if _is_static_boundary_task(tasks[first]) and _is_static_boundary_task(
            tasks[second]
        ):
            winner = first if first_area > second_area else second
            loser = second if winner == first else first
            return (
                loser,
                winner,
                "kept more complete mask for connected static boundary",
            )
        loser = first if first_area > second_area else second
        winner = second if loser == first else first
        return loser, winner, "rejected larger superset mask"

    first_rank = _grounding_task_rank(tasks[first])
    second_rank = _grounding_task_rank(tasks[second])
    if first_rank >= second_rank:
        return second, first, "rejected lower-ranked near-duplicate mask"
    return first, second, "rejected lower-ranked near-duplicate mask"


def _is_static_boundary_task(task: dict[str, Any]) -> bool:
    info = task.get("object_info") or {}
    if not bool(info.get("is_static", False)):
        return False
    text = " ".join(
        str(info.get(key, ""))
        for key in ("name", "description", "static_reason")
    ).lower()
    return any(
        term in text
        for term in ("divider", "panel", "barrier", "partition", "boundary")
    )


def _grounding_task_rank(task: dict[str, Any]) -> tuple[float, int, float, int]:
    confidence = {"high": 2, "medium": 1, "low": 0}.get(
        str(task["object_info"].get("confidence", "")).strip().lower(),
        -1,
    )
    grounding = task["grounding"]
    return (
        float(grounding.get("bbox_iou", 0.0)),
        confidence,
        float(grounding.get("score", 0.0)),
        -int(task["index"]),
    )


def _grounding_skip(
    task: dict[str, Any],
    reason: str,
    grounding: Any,
) -> dict[str, Any]:
    info = {
        "name": str(task.get("name", "")),
        "safe_name": str(task.get("safe_name", "")),
        "reason": str(reason),
    }
    if isinstance(grounding, dict):
        info["grounding_attempts"] = grounding.get("attempts", [])
        info["grounding_aliases"] = grounding.get("aliases", [])
        info["grounding_alias_raw_response"] = grounding.get(
            "alias_raw_response",
            "",
        )
        info["grounding_conflicts"] = grounding.get("conflicts", [])
        border = grounding.get("border_truncation")
        if isinstance(border, dict):
            info["border_truncation"] = border
    return info
