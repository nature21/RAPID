"""Object segmentation, geometry reconstruction, and placement stages."""

from __future__ import annotations

from dataclasses import replace
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from real2sim.mesh_utils import (
    export_color_cluster_visual_meshes,
    process_sam3d_mesh,
    simplify_mesh,
    to_trimesh,
)
from real2sim.schema import PipelineConfig, ReconstructedObject
from real2sim._stages.helpers import (
    canonical_text,
    list_or_none,
    optional_float,
    stage_log,
)
from real2sim.vfm_clients.foundation_pose import fp_registration_request
from real2sim.vfm_clients.sam3 import sam3_request
from real2sim.vfm_clients.sam3d import sam3d_request
from real2sim._stages.vlm import suggest_grounding_aliases


_FOUNDATION_POSE_LOCK = threading.Lock()


def reconstruct_object_geometry(
    *,
    image: Image.Image,
    object_info: dict[str, Any],
    detected_objects: list[dict[str, Any]] | None = None,
    safe_name: str,
    pointmap_p3d: np.ndarray,
    masks_dir: Path,
    config: PipelineConfig,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Run SAM grounding and SAM3D reconstruction without pose registration."""

    grounding = segment_object_grounding(
        image=image,
        object_info=object_info,
        detected_objects=detected_objects,
        config=config,
        log_fn=log_fn,
    )
    if grounding is None:
        return None

    return reconstruct_grounded_object_geometry(
        image=image,
        object_info=object_info,
        safe_name=safe_name,
        pointmap_p3d=pointmap_p3d,
        masks_dir=masks_dir,
        grounding=grounding,
        config=config,
    )


def segment_object_grounding(
    *,
    image: Image.Image,
    object_info: dict[str, Any],
    detected_objects: list[dict[str, Any]] | None,
    config: PipelineConfig,
    force_aliases: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Ground and validate one object mask without invoking SAM3D."""

    grounding = _segment_object_with_grounding_fallback(
        image=image,
        object_info=object_info,
        detected_objects=detected_objects,
        config=config,
        force_aliases=force_aliases,
        log_fn=log_fn,
    )
    if grounding["mask"] is None:
        object_info["_grounding"] = grounding
        return None

    mask = grounding["mask"]
    border_rejection = _border_truncation_rejection(
        mask=mask,
        object_info=object_info,
        config=config,
    )
    if border_rejection is not None:
        skipped_grounding = dict(grounding)
        skipped_grounding["mask"] = None
        skipped_grounding["rejection_reason"] = border_rejection["reason"]
        skipped_grounding["border_truncation"] = border_rejection
        object_info["_grounding"] = skipped_grounding
        stage_log(
            log_fn,
            f"[real2sim] rejected border-truncated object {safe_name}: "
            f"{border_rejection['reason']}",
        )
        return None
    return grounding


def reconstruct_grounded_object_geometry(
    *,
    image: Image.Image,
    object_info: dict[str, Any],
    safe_name: str,
    pointmap_p3d: np.ndarray,
    masks_dir: Path,
    grounding: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Run SAM3D for a scene-level accepted object grounding."""

    name = str(object_info["name"])
    mask = np.asarray(grounding["mask"], dtype=np.uint8)
    mask_area = int(grounding["mask_area_px"])
    Image.fromarray(mask * 255).save(masks_dir / f"{safe_name}.png")

    glb_raw, glb_processed, scale, translation, quat = sam3d_request(
        image=image,
        mask=mask,
        point_map=pointmap_p3d,
        server_url=config.sam3d_server_url,
    )

    mesh_raw_processed = process_sam3d_mesh(glb_raw, scale, config.target_faces)
    mesh_camera, mesh_frame = _select_sam3d_mesh(glb_raw, glb_processed, scale)
    if config.foundation_pose_target_faces == config.target_faces:
        mesh_for_fp = mesh_raw_processed.copy()
    else:
        mesh_for_fp = process_sam3d_mesh(
            glb_raw,
            scale,
            config.foundation_pose_target_faces,
        )

    return {
        "name": name,
        "object_info": object_info,
        "safe_name": safe_name,
        "grounding": grounding,
        "mask": mask,
        "mask_area_px": mask_area,
        "mesh_for_fp": mesh_for_fp,
        "mesh_for_fp_export": mesh_raw_processed.copy(),
        "mesh_fallback": simplify_mesh(mesh_camera, config.target_faces),
        "mesh_frame": mesh_frame,
        "scale": scale,
        "translation": translation,
        "quat": quat,
    }


def _border_truncation_rejection(
    *,
    mask: np.ndarray,
    object_info: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any] | None:
    """Describe an unreliable partial-object mask, or return ``None``."""

    if not config.reject_border_truncated_objects:
        return None
    confidence = str(object_info.get("confidence", "")).strip().lower()
    if confidence == "high":
        return None

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    height, width = mask_bool.shape
    margin_fraction = max(0.0, float(config.border_truncation_margin_fraction))
    margin_x = max(1, int(np.ceil(width * margin_fraction)))
    margin_y = max(1, int(np.ceil(height * margin_fraction)))
    touched_edges = []
    if mask_bool[:, :margin_x].any():
        touched_edges.append("left")
    if mask_bool[:, width - margin_x :].any():
        touched_edges.append("right")
    if mask_bool[:margin_y, :].any():
        touched_edges.append("top")
    if mask_bool[height - margin_y :, :].any():
        touched_edges.append("bottom")
    if not touched_edges:
        return None

    edges = ", ".join(touched_edges)
    return {
        "reason": (
            f"{confidence or 'non-high'}-confidence mask touches image border "
            f"({edges}); full object geometry is not observable"
        ),
        "confidence": confidence,
        "touched_edges": touched_edges,
        "margin_px": [margin_x, margin_y],
        "mask_area_px": int(mask_bool.sum()),
        "image_size": [width, height],
    }


def reconstruct_object(
    *,
    image: Image.Image,
    object_info: dict[str, Any],
    detected_objects: list[dict[str, Any]] | None = None,
    safe_name: str,
    K: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
    pointmap_p3d: np.ndarray,
    processed_camera_mesh_to_world: np.ndarray,
    raw_camera_mesh_to_world: np.ndarray,
    meshes_dir: Path,
    masks_dir: Path,
    config: PipelineConfig,
    log_fn: Callable[[str], None] | None = None,
) -> ReconstructedObject | None:
    geometry = reconstruct_object_geometry(
        image=image,
        object_info=object_info,
        detected_objects=detected_objects,
        safe_name=safe_name,
        pointmap_p3d=pointmap_p3d,
        masks_dir=masks_dir,
        config=config,
        log_fn=log_fn,
    )
    if geometry is None:
        return None

    return place_reconstructed_geometry(
        geometry=geometry,
        K=K,
        color=color,
        depth=depth,
        T_world_from_camera_opencv=T_world_from_camera_opencv,
        processed_camera_mesh_to_world=processed_camera_mesh_to_world,
        raw_camera_mesh_to_world=raw_camera_mesh_to_world,
        meshes_dir=meshes_dir,
        config=config,
        log_fn=log_fn,
    )


def place_reconstructed_geometry(
    *,
    geometry: dict[str, Any],
    K: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
    processed_camera_mesh_to_world: np.ndarray,
    raw_camera_mesh_to_world: np.ndarray,
    meshes_dir: Path,
    config: PipelineConfig,
    log_fn: Callable[[str], None] | None = None,
) -> ReconstructedObject:
    name = str(geometry["name"])
    object_info = geometry["object_info"]
    safe_name = str(geometry["safe_name"])
    grounding = geometry["grounding"]
    mask = geometry["mask"]
    mask_area = int(geometry["mask_area_px"])
    mesh_for_fp = geometry["mesh_for_fp"]
    mesh_for_fp_export = geometry["mesh_for_fp_export"]
    mesh_camera = geometry["mesh_fallback"]
    mesh_frame = str(geometry["mesh_frame"])
    scale = geometry["scale"]
    translation = geometry["translation"]
    quat = geometry["quat"]

    (
        mesh_world,
        pose_source,
        T_mesh_world,
        pose_last,
        pose_opencv,
        fp_input,
        fp_error,
    ) = _place_mesh(
        mesh_for_fp=mesh_for_fp,
        mesh_for_fp_export=mesh_for_fp_export,
        mesh_fallback=mesh_camera,
        mesh_frame=mesh_frame,
        K=K,
        color=color,
        depth=depth,
        mask=mask,
        T_world_from_camera_opencv=T_world_from_camera_opencv,
        processed_camera_mesh_to_world=processed_camera_mesh_to_world,
        raw_camera_mesh_to_world=raw_camera_mesh_to_world,
        config=config,
        log_fn=log_fn,
    )

    if mesh_world.vertices.size == 0:
        raise ValueError("SAM3D returned an empty mesh")
    if not np.isfinite(mesh_world.vertices).all():
        raise ValueError("SAM3D returned a mesh with non-finite vertices")

    min_z_before_lift = float(np.min(np.asarray(mesh_world.vertices)[:, 2]))
    support_lift = _support_clearance_lift(
        min_z=min_z_before_lift,
        config=config,
    )
    max_support_lift = max(0.0, float(config.max_support_lift_meters))
    support_correction = _support_penetration_correction(
        support_lift=support_lift,
        config=config,
    )
    icp_registration: dict[str, Any] = {}
    icp_trigger = max(
        0.0,
        float(config.icp_fallback_support_correction_meters),
    )
    force_native_fallback = False
    if (
        pose_source == "foundation_pose_camera_frame"
        and support_correction > icp_trigger
    ):
        stage_log(
            log_fn,
            f"[real2sim] {safe_name}: Foundation Pose support correction "
            f"{support_correction:.6g} m exceeds ICP trigger "
            f"{icp_trigger:.6g} m; refining against metric depth",
        )
        mesh_world, T_mesh_world, icp_registration = _refine_pose_with_icp(
            mesh_world=mesh_world,
            T_mesh_world=T_mesh_world,
            depth=depth,
            K=K,
            mask=mask,
            T_world_from_camera_opencv=T_world_from_camera_opencv,
            config=config,
        )
        if bool(icp_registration.get("accepted", False)):
            pose_source = f"{pose_source}_metric_depth_icp"
            min_z_before_lift = float(
                np.min(np.asarray(mesh_world.vertices)[:, 2])
            )
            support_lift = _support_clearance_lift(
                min_z=min_z_before_lift,
                config=config,
            )
            support_correction = _support_penetration_correction(
                support_lift=support_lift,
                config=config,
            )
            stage_log(
                log_fn,
                f"[real2sim] {safe_name}: ICP accepted "
                f"(fitness={icp_registration['fitness']:.3f}, "
                f"rmse={icp_registration['inlier_rmse_meters']:.6g} m, "
                f"support_correction={support_correction:.6g} m)",
            )
        else:
            force_native_fallback = True
            stage_log(
                log_fn,
                f"[real2sim] {safe_name}: ICP rejected "
                f"({icp_registration.get('reason', 'unknown reason')}); "
                "trying native metric-pointmap SAM3D placement",
            )
    if (
        (support_correction > max_support_lift or force_native_fallback)
        and pose_source.startswith("foundation_pose_camera_frame")
    ):
        if force_native_fallback and support_correction <= max_support_lift:
            rejection = (
                "Foundation Pose metric-depth ICP was rejected: "
                f"{icp_registration.get('reason', 'unknown reason')}"
            )
        else:
            rejection = (
                "Foundation Pose placement requires excessive support-plane "
                f"correction: {support_correction:.6g} m > "
                f"{max_support_lift:.6g} m"
            )
        stage_log(
            log_fn,
            f"[real2sim] {safe_name}: {rejection}; retrying calibrated "
            "metric-pointmap SAM3D placement",
        )
        fp_pose_last = pose_last
        fp_pose_opencv = pose_opencv
        fp_metadata = fp_input
        (
            mesh_world,
            pose_source,
            T_mesh_world,
            _,
            _,
            _,
            fallback_error,
        ) = _place_mesh(
            mesh_for_fp=mesh_for_fp,
            mesh_for_fp_export=mesh_for_fp_export,
            mesh_fallback=mesh_camera,
            mesh_frame=mesh_frame,
            K=K,
            color=color,
            depth=depth,
            mask=mask,
            T_world_from_camera_opencv=T_world_from_camera_opencv,
            processed_camera_mesh_to_world=processed_camera_mesh_to_world,
            raw_camera_mesh_to_world=raw_camera_mesh_to_world,
            config=replace(config, use_foundation_pose=False),
            log_fn=log_fn,
        )
        pose_source = f"{pose_source}_support_lift_fallback"
        pose_last = fp_pose_last
        pose_opencv = fp_pose_opencv
        fp_input = fp_metadata
        fp_error = "; ".join(
            item for item in (fp_error, rejection, fallback_error) if item
        )
        min_z_before_lift = float(np.min(np.asarray(mesh_world.vertices)[:, 2]))
        support_lift = _support_clearance_lift(
            min_z=min_z_before_lift,
            config=config,
        )
        support_correction = _support_penetration_correction(
            support_lift=support_lift,
            config=config,
        )

    if support_correction > max_support_lift:
        raise ValueError(
            "object pose requires excessive support-plane correction: "
            f"{support_correction:.6g} m "
            f"> {max_support_lift:.6g} m"
        )
    if support_lift > 0.0:
        mesh_world.apply_translation([0.0, 0.0, support_lift])
        T_lift = np.eye(4, dtype=float)
        T_lift[2, 3] = support_lift
        T_mesh_world = T_lift @ np.asarray(T_mesh_world, dtype=float)
        stage_log(
            log_fn,
            f"[real2sim] lifted object {safe_name}: "
            f"min_z={min_z_before_lift:.6g} -> "
            f"{min_z_before_lift + support_lift:.6g} "
            f"(dz={support_lift:.6g})",
        )
    min_z_after_lift = float(np.min(np.asarray(mesh_world.vertices)[:, 2]))

    world_bounds = np.asarray(mesh_world.bounds, dtype=float)
    body_pos = world_bounds.mean(axis=0)
    local_mesh = mesh_world.copy()
    local_mesh.vertices = local_mesh.vertices - body_pos
    local_bounds = np.asarray(local_mesh.bounds, dtype=float)
    extents = np.maximum(local_bounds[1] - local_bounds[0], 1e-4)
    box_size = np.maximum(extents / 2.0, 5e-5)

    obb_source_mesh_path = meshes_dir / f"{safe_name}.obj"
    local_mesh.export(obb_source_mesh_path)
    visual_mesh_files = export_color_cluster_visual_meshes(
        local_mesh,
        meshes_dir,
        safe_name,
        max_clusters=config.visual_color_clusters,
    )
    if not visual_mesh_files:
        visual_mesh_files = [
            {
                "file": f"meshes/{obb_source_mesh_path.name}",
                "rgba": [1.0, 1.0, 1.0, 1.0],
            }
        ]

    bbox = grounding["bbox"]
    return ReconstructedObject(
        name=name,
        safe_name=safe_name,
        description=str(object_info.get("description", "")),
        confidence=str(object_info.get("confidence", "")),
        sam3_score=float(grounding["score"]),
        sam3_bbox=list_or_none(bbox),
        mask_area_px=mask_area,
        mesh_file=str(visual_mesh_files[0]["file"]),
        obb_source_mesh_file=f"meshes/{obb_source_mesh_path.name}",
        visual_mesh_files=visual_mesh_files,
        body_pos=body_pos.tolist(),
        local_bounds=local_bounds.tolist(),
        world_bounds=world_bounds.tolist(),
        box_size=box_size.tolist(),
        sam3d_scale=np.asarray(scale, dtype=float).reshape(-1).tolist(),
        sam3d_translation=np.asarray(translation, dtype=float).reshape(-1).tolist(),
        sam3d_quat=np.asarray(quat, dtype=float).reshape(-1).tolist(),
        mesh_frame=mesh_frame,
        pose_source=pose_source,
        grounding_query=str(grounding["query"]),
        grounding_query_source=str(grounding["query_source"]),
        grounding_attempts=grounding["attempts"],
        grounding_aliases=grounding["aliases"],
        grounding_alias_raw_response=str(grounding.get("alias_raw_response", "")),
        grounding_conflicts=list(grounding.get("conflicts", [])),
        is_static=bool(object_info.get("is_static", False)),
        static_reason=str(object_info.get("static_reason", "")),
        static_confidence=str(object_info.get("static_confidence", "")),
        physical_material=str(object_info.get("physical_material", "")),
        physical_mass_kg=optional_float(object_info.get("physical_mass_kg")),
        physical_sliding_friction=optional_float(
            object_info.get("physical_sliding_friction")
        ),
        physical_torsional_friction=optional_float(
            object_info.get("physical_torsional_friction")
        ),
        physical_rolling_friction=optional_float(
            object_info.get("physical_rolling_friction")
        ),
        physical_restitution=optional_float(object_info.get("physical_restitution")),
        physical_confidence=str(object_info.get("physical_confidence", "")),
        physical_reason=str(object_info.get("physical_reason", "")),
        T_mesh_world=T_mesh_world.tolist(),
        support_lift_meters=float(support_lift),
        support_penetration_correction_meters=float(support_correction),
        world_min_z_before_lift=float(min_z_before_lift),
        world_min_z_after_lift=float(min_z_after_lift),
        foundation_pose_opencv=None if pose_opencv is None else pose_opencv.tolist(),
        foundation_pose_last=None if pose_last is None else pose_last.tolist(),
        foundation_pose_input=fp_input,
        foundation_pose_error=fp_error,
        icp_registration=icp_registration,
    )


def _segment_object_with_grounding_fallback(
    *,
    image: Image.Image,
    object_info: dict[str, Any],
    detected_objects: list[dict[str, Any]] | None,
    config: PipelineConfig,
    force_aliases: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    name = str(object_info["name"]).strip()
    description = str(object_info.get("description", "")).strip()
    attempts: list[dict[str, Any]] = []
    aliases: list[str] = []
    alias_raw_response = ""
    target_bbox = _expanded_object_bbox(
        object_info,
        image_size=image.size,
        margin_fraction=float(config.grounding_bbox_margin_fraction),
    )
    candidates: list[dict[str, Any]] = []

    queries = [(name, "object_name")]
    if description and canonical_text(description) != canonical_text(name):
        queries.append((f"{name}, {description}", "object_description"))
    seen: set[str] = set()
    for query, source in queries:
        canonical = canonical_text(query)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result = _try_sam3_grounding_query(
            image=image,
            query=query,
            query_source=source,
            request_bbox=target_bbox,
            target_bbox=target_bbox,
            config=config,
            attempts=attempts,
        )
        if result is not None:
            candidates.append(result)

    if bool(config.use_grounding_alias_fallback) and (not candidates or force_aliases):
        stage_log(
            log_fn,
            f"[real2sim] requesting VLM grounding aliases for {name!r}"
            + (" after a mask conflict..." if force_aliases else "..."),
        )
        try:
            aliases, alias_raw_response = suggest_grounding_aliases(
                image,
                object_info,
                model=config.vlm_model,
                reasoning_effort=config.vlm_reasoning_effort,
                max_aliases=int(config.max_grounding_aliases),
                detected_objects=detected_objects,
            )
            stage_log(
                log_fn,
                f"[real2sim] VLM grounding aliases for {name!r}: {aliases}",
            )
        except Exception as exc:
            stage_log(
                log_fn,
                f"[real2sim] VLM grounding aliases failed for {name!r}: "
                f"{type(exc).__name__}: {exc}",
            )
            attempts.append(
                {
                    "query": "",
                    "query_source": "grounding_alias_vlm",
                    "success": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            aliases = []

        for alias in aliases:
            canonical = canonical_text(alias)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            stage_log(
                log_fn,
                f"[real2sim] retrying SAM3 for {name!r} with alias {alias!r}",
            )
            result = _try_sam3_grounding_query(
                image=image,
                query=alias,
                query_source="vlm_alias",
                request_bbox=target_bbox,
                target_bbox=target_bbox,
                config=config,
                attempts=attempts,
            )
            if result is not None:
                candidates.append(result)
                stage_log(
                    log_fn,
                    f"[real2sim] grounded {name!r} using alias {alias!r}",
                )

    if candidates:
        result = max(candidates, key=_grounding_candidate_rank)
        result["aliases"] = aliases
        result["alias_raw_response"] = alias_raw_response
        result["conflicts"] = []
        return result

    return {
        "mask": None,
        "query": name,
        "query_source": "none",
        "score": 0.0,
        "bbox": None,
        "mask_area_px": 0,
        "attempts": attempts,
        "aliases": aliases,
        "alias_raw_response": alias_raw_response,
        "target_bbox": list_or_none(target_bbox),
        "bbox_iou": 0.0,
        "conflicts": [],
    }


def _try_sam3_grounding_query(
    *,
    image: Image.Image,
    query: str,
    query_source: str,
    request_bbox: np.ndarray | None,
    target_bbox: np.ndarray | None,
    config: PipelineConfig,
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    query = str(query).strip()
    if not query:
        return None

    request_image = image
    crop_origin = np.asarray([0.0, 0.0], dtype=float)
    crop_box: tuple[int, int, int, int] | None = None
    if request_bbox is not None:
        x1 = max(0, int(np.floor(request_bbox[0])))
        y1 = max(0, int(np.floor(request_bbox[1])))
        x2 = min(image.size[0], int(np.ceil(request_bbox[2])))
        y2 = min(image.size[1], int(np.ceil(request_bbox[3])))
        if x2 > x1 and y2 > y1:
            crop_box = (x1, y1, x2, y2)
            crop_origin = np.asarray([x1, y1], dtype=float)
            request_image = image.crop(crop_box)

    try:
        scores, bboxes, masks = sam3_request(
            image=request_image,
            text=query,
            bbox=None,
            confidence_threshold=config.sam3_confidence_threshold,
            server_url=config.sam3_server_url,
        )
    except Exception as exc:
        attempts.append(
            {
                "query": query,
                "query_source": query_source,
                "success": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "request_bbox": list_or_none(request_bbox),
            }
        )
        return None

    if not scores:
        attempts.append(
            {
                "query": query,
                "query_source": query_source,
                "success": False,
                "reason": "no masks",
                "request_bbox": list_or_none(request_bbox),
            }
        )
        return None

    candidates: list[dict[str, Any]] = []
    rejected_areas: list[int] = []
    rejected_fill_ratios: list[float] = []
    for index, raw_mask in enumerate(masks):
        cropped_mask = (np.asarray(raw_mask) > 0).astype(np.uint8)
        if crop_box is None:
            mask = cropped_mask
        else:
            expected_shape = (crop_box[3] - crop_box[1], crop_box[2] - crop_box[0])
            if cropped_mask.shape != expected_shape:
                continue
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            mask[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]] = cropped_mask
        mask_area = int(mask.sum())
        min_area = max(
            int(config.min_mask_area_px),
            int(mask.size * float(config.min_mask_area_fraction)),
        )
        if mask_area < min_area:
            rejected_areas.append(mask_area)
            continue
        bbox = _coerce_bbox(bboxes[index] if index < len(bboxes) else None)
        if bbox is not None and crop_box is not None:
            bbox = bbox + np.asarray(
                [crop_origin[0], crop_origin[1], crop_origin[0], crop_origin[1]],
                dtype=float,
            )
        if bbox is None:
            bbox = _mask_bbox(mask)
        mask_bbox = _mask_bbox(mask)
        mask_bbox_area = _bbox_area(mask_bbox)
        mask_bbox_fill_ratio = (
            0.0 if mask_bbox_area <= 0.0 else mask_area / mask_bbox_area
        )
        if mask_bbox_fill_ratio < float(
            config.min_grounding_mask_bbox_fill_ratio
        ):
            rejected_fill_ratios.append(mask_bbox_fill_ratio)
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        bbox_iou = _bbox_iou(bbox, target_bbox)
        candidates.append(
            {
                "mask": mask,
                "query": query,
                "query_source": query_source,
                "score": score,
                "bbox": bbox,
                "mask_area_px": mask_area,
                "target_bbox": target_bbox,
                "bbox_iou": bbox_iou,
                "mask_bbox_fill_ratio": mask_bbox_fill_ratio,
            }
        )

    if not candidates:
        attempts.append(
            {
                "query": query,
                "query_source": query_source,
                "success": False,
                "reason": "all masks failed area or bbox-fill validation",
                "request_bbox": list_or_none(request_bbox),
                "candidate_mask_areas_px": rejected_areas,
                "candidate_mask_bbox_fill_ratios": rejected_fill_ratios,
            }
        )
        return None

    result = max(candidates, key=_grounding_candidate_rank)
    attempts.append(
        {
            "query": query,
            "query_source": query_source,
            "success": True,
            "score": result["score"],
            "bbox": list_or_none(result["bbox"]),
            "request_bbox": list_or_none(request_bbox),
            "target_bbox": list_or_none(target_bbox),
            "bbox_iou": result["bbox_iou"],
            "mask_bbox_fill_ratio": result["mask_bbox_fill_ratio"],
            "mask_area_px": result["mask_area_px"],
            "candidate_count": len(candidates),
        }
    )
    result["attempts"] = attempts
    return result


def _expanded_object_bbox(
    object_info: dict[str, Any],
    *,
    image_size: tuple[int, int],
    margin_fraction: float,
) -> np.ndarray | None:
    raw = object_info.get("bbox_2d")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        normalized = np.asarray(raw, dtype=float).reshape(4)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(normalized).all():
        return None
    width, height = image_size
    scale = np.asarray([width, height, width, height], dtype=float) / 1000.0
    bbox = np.clip(normalized, 0.0, 1000.0) * scale
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    margin = max(0.0, float(margin_fraction))
    dx = (bbox[2] - bbox[0]) * margin
    dy = (bbox[3] - bbox[1]) * margin
    bbox += np.asarray([-dx, -dy, dx, dy], dtype=float)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(width))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(height))
    return bbox


def _coerce_bbox(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        bbox = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if bbox.size != 4 or not np.isfinite(bbox).all():
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=float,
    )


def _bbox_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    overlap_min = np.maximum(first[:2], second[:2])
    overlap_max = np.minimum(first[2:], second[2:])
    overlap = np.maximum(overlap_max - overlap_min, 0.0)
    intersection = float(np.prod(overlap))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _bbox_area(bbox: np.ndarray | None) -> float:
    if bbox is None:
        return 0.0
    return float(np.prod(np.maximum(bbox[2:] - bbox[:2], 0.0)))


def _grounding_candidate_rank(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(result.get("bbox_iou", 0.0)),
        float(result.get("mask_bbox_fill_ratio", 0.0)),
        float(result.get("score", 0.0)),
    )


def _uses_support_surface_world(config: PipelineConfig) -> bool:
    return str(config.world_frame_mode).replace("-", "_") == "support_surface"


def _support_clearance_lift(
    *,
    min_z: float,
    config: PipelineConfig,
) -> float:
    """Return the upward correction needed to clear the world support plane."""

    if _uses_support_surface_world(config):
        floor_z = 0.0
    elif config.support_floor_z is not None:
        floor_z = float(config.support_floor_z)
    else:
        return 0.0
    clearance = max(0.0, float(config.object_support_clearance))
    return max(0.0, floor_z + clearance - float(min_z))


def _support_penetration_correction(
    *,
    support_lift: float,
    config: PipelineConfig,
) -> float:
    """Return the pose correction excluding intentional support clearance."""

    clearance = max(0.0, float(config.object_support_clearance))
    return max(0.0, float(support_lift) - clearance)


def _refine_pose_with_icp(
    *,
    mesh_world: Any,
    T_mesh_world: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
    config: PipelineConfig,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Rigidly refine a candidate mesh against its calibrated masked depth cloud.

    The RGB-D observation only contains camera-visible object surfaces.  Using the
    complete reconstructed mesh as the ICP source lets a visible face snap to an
    unrelated hidden face, especially for box-like objects.  Project the candidate
    through the calibrated camera and retain its front-most samples before ICP so
    both point clouds represent the same partial view.
    """

    try:
        import open3d as o3d

        target_points = _masked_depth_points_world(
            depth=depth,
            K=K,
            mask=mask,
            T_world_from_camera_opencv=T_world_from_camera_opencv,
        )
        mesh_points = _camera_visible_mesh_points(
            mesh_world=mesh_world,
            K=K,
            mask=mask,
            T_world_from_camera_opencv=T_world_from_camera_opencv,
        )
        if len(target_points) < 64 or len(mesh_points) < 64:
            raise ValueError(
                f"insufficient ICP points: depth={len(target_points)}, "
                f"mesh={len(mesh_points)}"
            )

        observed = o3d.geometry.PointCloud()
        observed.points = o3d.utility.Vector3dVector(target_points)
        candidate = o3d.geometry.PointCloud()
        candidate.points = o3d.utility.Vector3dVector(mesh_points)
        voxel = 0.005
        observed = observed.voxel_down_sample(voxel)
        candidate = candidate.voxel_down_sample(voxel)

        initial = o3d.pipelines.registration.evaluate_registration(
            candidate,
            observed,
            0.03,
            np.eye(4, dtype=np.float64),
        )
        candidate_array = np.asarray(candidate.points, dtype=np.float64)
        observed_array = np.asarray(observed.points, dtype=np.float64)
        coarse_translation = np.median(observed_array, axis=0) - np.median(
            candidate_array,
            axis=0,
        )
        coarse_norm = float(np.linalg.norm(coarse_translation))
        max_coarse = max(0.0, float(config.icp_max_translation_meters))
        if max_coarse > 0.0 and coarse_norm > max_coarse:
            coarse_translation *= max_coarse / coarse_norm
        transform_mesh_to_observed = np.eye(4, dtype=np.float64)
        transform_mesh_to_observed[:3, 3] = coarse_translation
        result = None
        for distance in (0.10, 0.05, 0.025):
            result = o3d.pipelines.registration.registration_icp(
                candidate,
                observed,
                distance,
                transform_mesh_to_observed,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=False
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=80
                ),
            )
            transform_mesh_to_observed = np.asarray(
                result.transformation,
                dtype=np.float64,
            )
        assert result is not None
        correction = transform_mesh_to_observed
        transform_translation = float(np.linalg.norm(correction[:3, 3]))
        candidate_center = np.asarray(mesh_world.bounds, dtype=np.float64).mean(axis=0)
        corrected_center = (
            correction @ np.concatenate([candidate_center, [1.0]])
        )[:3]
        translation = float(np.linalg.norm(corrected_center - candidate_center))
        trace = float(np.trace(correction[:3, :3]))
        rotation_degrees = float(
            np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        )
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)
        initial_fitness = float(initial.fitness)
        initial_rmse = float(initial.inlier_rmse)
        fitness_improvement = fitness - initial_fitness
        rmse_improvement = initial_rmse - rmse
        reasons = _icp_rejection_reasons(
            fitness=fitness,
            rmse=rmse,
            translation=translation,
            rotation_degrees=rotation_degrees,
            fitness_improvement=fitness_improvement,
            rmse_improvement=rmse_improvement,
            config=config,
        )
        report = {
            "attempted": True,
            "accepted": not reasons,
            "registration_mode": "rigid_visible_surface",
            "fitness": fitness,
            "inlier_rmse_meters": rmse,
            "initial_fitness": initial_fitness,
            "initial_inlier_rmse_meters": initial_rmse,
            "fitness_improvement": fitness_improvement,
            "inlier_rmse_improvement_meters": rmse_improvement,
            "translation_meters": translation,
            "transform_translation_meters": transform_translation,
            "rotation_degrees": rotation_degrees,
            "coarse_translation_meters": coarse_translation.tolist(),
            "depth_point_count": int(len(observed.points)),
            "mesh_point_count": int(len(candidate.points)),
            "correction_transform": correction.tolist(),
            "reason": "; ".join(reasons),
        }
        if reasons:
            return mesh_world, np.asarray(T_mesh_world, dtype=float), report
        refined = mesh_world.copy()
        refined.apply_transform(correction)
        return (
            refined,
            correction @ np.asarray(T_mesh_world, dtype=float),
            report,
        )
    except Exception as exc:
        return (
            mesh_world,
            np.asarray(T_mesh_world, dtype=float),
            {
                "attempted": True,
                "accepted": False,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )


def _icp_rejection_reasons(
    *,
    fitness: float,
    rmse: float,
    translation: float,
    rotation_degrees: float,
    fitness_improvement: float,
    rmse_improvement: float,
    config: PipelineConfig,
) -> list[str]:
    reasons: list[str] = []
    if fitness < float(config.icp_min_fitness):
        reasons.append(
            f"fitness {fitness:.3f} < {float(config.icp_min_fitness):.3f}"
        )
    if rmse > float(config.icp_max_inlier_rmse_meters):
        reasons.append(
            f"RMSE {rmse:.6g} m > "
            f"{float(config.icp_max_inlier_rmse_meters):.6g} m"
        )
    if translation > float(config.icp_max_translation_meters):
        reasons.append(
            f"translation {translation:.6g} m > "
            f"{float(config.icp_max_translation_meters):.6g} m"
        )
    if rotation_degrees > float(config.icp_max_rotation_degrees):
        reasons.append(
            f"rotation {rotation_degrees:.3f} deg > "
            f"{float(config.icp_max_rotation_degrees):.3f} deg"
        )
    if fitness_improvement < 0.01 and rmse_improvement < 0.001:
        reasons.append(
            "registration did not materially improve visible-surface agreement"
        )
    return reasons


def _camera_visible_mesh_points(
    *,
    mesh_world: Any,
    K: np.ndarray,
    mask: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
) -> np.ndarray:
    """Sample the candidate's front-most surface in the observed mask region."""

    vertices = np.asarray(mesh_world.vertices, dtype=np.float64)
    faces = np.asarray(mesh_world.faces, dtype=np.int64)
    points_world = vertices
    if faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
        triangles = vertices[faces]
        # Triangle centers are deterministic and give enough raster density for
        # the simplified (normally 10k-face) SAM3D meshes used by registration.
        centers = triangles.mean(axis=1)
        points_world = np.concatenate([vertices, centers], axis=0)
    if not len(points_world):
        return np.empty((0, 3), dtype=np.float64)

    T_camera_from_world = np.linalg.inv(
        np.asarray(T_world_from_camera_opencv, dtype=np.float64)
    )
    rotation = T_camera_from_world[:3, :3]
    translation = T_camera_from_world[:3, 3]
    points_camera = points_world @ rotation.T + translation
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-6)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    points_camera = points_camera[valid]
    points_world = points_world[valid]
    z = points_camera[:, 2]
    intrinsics = np.asarray(K, dtype=np.float64)
    u = intrinsics[0, 0] * points_camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / z + intrinsics[1, 2]

    mask_arr = np.asarray(mask, dtype=bool)
    height, width = mask_arr.shape
    ys, xs = np.nonzero(mask_arr)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    margin_x = max(4.0, 0.10 * float(xs.max() - xs.min() + 1))
    margin_y = max(4.0, 0.10 * float(ys.max() - ys.min() + 1))
    x_min = max(0.0, float(xs.min()) - margin_x)
    x_max = min(float(width - 1), float(xs.max()) + margin_x)
    y_min = max(0.0, float(ys.min()) - margin_y)
    y_max = min(float(height - 1), float(ys.max()) + margin_y)
    in_region = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= x_min)
        & (u <= x_max)
        & (v >= y_min)
        & (v <= y_max)
    )
    if not in_region.any():
        return np.empty((0, 3), dtype=np.float64)

    points_world = points_world[in_region]
    z = z[in_region]
    # A two-pixel z-buffer rejects rear/hidden samples without making the
    # candidate cloud unnecessarily sparse.
    u_bin = np.floor(u[in_region] / 2.0).astype(np.int64)
    v_bin = np.floor(v[in_region] / 2.0).astype(np.int64)
    bin_width = (width + 1) // 2
    pixel_bins = v_bin * bin_width + u_bin
    order = np.lexsort((z, pixel_bins))
    sorted_bins = pixel_bins[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = sorted_bins[1:] != sorted_bins[:-1]
    return points_world[order[keep]]


def _masked_depth_points_world(
    *,
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if depth_arr.shape != mask_arr.shape:
        raise ValueError(
            f"depth/mask shape mismatch: {depth_arr.shape} != {mask_arr.shape}"
        )
    valid = mask_arr & np.isfinite(depth_arr) & (depth_arr > 0.0)
    ys, xs = np.nonzero(valid)
    z = depth_arr[ys, xs]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if z.size >= 100:
        low, high = np.quantile(z, [0.01, 0.99])
        keep = (z >= low) & (z <= high)
        xs = xs[keep]
        ys = ys[keep]
        z = z[keep]
    intrinsics = np.asarray(K, dtype=np.float64)
    x = (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_points = np.stack([x, y, z], axis=1)
    rotation = np.asarray(T_world_from_camera_opencv, dtype=np.float64)[:3, :3]
    translation = np.asarray(T_world_from_camera_opencv, dtype=np.float64)[:3, 3]
    return camera_points @ rotation.T + translation


def _select_sam3d_mesh(
    raw: Any,
    processed: Any,
    scale: np.ndarray,
) -> tuple[Any, str]:
    try:
        mesh = to_trimesh(processed)
        if mesh.vertices.size:
            return mesh, "sam3d_processed_opengl"
    except Exception:
        pass

    mesh = to_trimesh(raw)
    mesh.apply_scale(np.asarray(scale, dtype=float))
    return mesh, "sam3d_raw_scaled_p3d"


def _place_mesh(
    *,
    mesh_for_fp: Any,
    mesh_for_fp_export: Any,
    mesh_fallback: Any,
    mesh_frame: str,
    K: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    T_world_from_camera_opencv: np.ndarray,
    processed_camera_mesh_to_world: np.ndarray,
    raw_camera_mesh_to_world: np.ndarray,
    config: PipelineConfig,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[
    Any,
    str,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
    str,
]:
    if config.use_foundation_pose:
        fp_meta: dict[str, Any] = {}
        try:
            # Foundation Pose registration is intentionally done in the original
            # camera frame: K, RGB, depth, and mask are all camera-frame inputs.
            # Only the returned camera-frame pose is lifted into the chosen
            # support-aligned world frame.
            K_fp, color_fp, depth_fp, mask_fp, fp_meta = _foundation_pose_full_inputs(
                K=K,
                color=color,
                depth=depth,
                mask=mask,
            )
            stage_log(
                log_fn,
                "[real2sim] Foundation Pose input: "
                f"image={fp_meta['input_hw'][0]}x{fp_meta['input_hw'][1]}, "
                f"mask_area={fp_meta['mask_area_px']}, "
                f"mesh_faces={len(mesh_for_fp.faces)}",
            )
            stage_log(
                log_fn,
                "[real2sim] Foundation Pose registration request "
                "(serialized GPU server call)",
            )
            with _FOUNDATION_POSE_LOCK:
                pose_opencv, pose_last = fp_registration_request(
                    mesh_for_fp,
                    K_fp,
                    color_fp,
                    depth_fp,
                    mask=mask_fp,
                    server_url=config.foundation_pose_url,
                )
            # Foundation Pose returns an object pose in the OpenCV camera
            # frame. The support-aligned world transform in this pipeline is
            # also defined from OpenCV camera coordinates, so no OpenCV/OpenGL
            # basis flip belongs in this chain.
            T_mesh_world = (
                np.asarray(T_world_from_camera_opencv, dtype=float) @ pose_opencv
            )
            mesh_world = mesh_for_fp_export.copy()
            mesh_world.apply_transform(T_mesh_world)
            return (
                mesh_world,
                "foundation_pose_camera_frame",
                T_mesh_world,
                pose_last,
                pose_opencv,
                fp_meta,
                "",
            )
        except Exception as exc:
            fp_error = f"{type(exc).__name__}: {exc}"
            stage_log(
                log_fn,
                "[real2sim] Foundation Pose failed; using SAM3D fallback: "
                f"{fp_error}",
            )
            if not config.fallback_without_foundation_pose:
                raise
    else:
        fp_meta = {}
        fp_error = ""

    T_mesh_world = (
        processed_camera_mesh_to_world
        if mesh_frame == "sam3d_processed_opengl"
        else raw_camera_mesh_to_world
    )
    mesh_world = mesh_fallback.copy()
    mesh_world.apply_transform(T_mesh_world)
    return (
        mesh_world,
        f"{mesh_frame}_camera_frame_fallback",
        np.asarray(T_mesh_world),
        None,
        None,
        fp_meta,
        fp_error,
    )


def _foundation_pose_full_inputs(
    *,
    K: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    color_arr = np.asarray(color, dtype=np.uint8)
    depth_arr = np.asarray(depth, dtype=np.float32)
    mask_arr = np.asarray(mask).astype(bool)
    if color_arr.ndim != 3 or color_arr.shape[2] != 3:
        raise ValueError(f"Foundation Pose color must be HxWx3, got {color_arr.shape}")
    if depth_arr.ndim != 2:
        raise ValueError(f"Foundation Pose depth must be HxW, got {depth_arr.shape}")
    if mask_arr.shape != depth_arr.shape or color_arr.shape[:2] != depth_arr.shape:
        raise ValueError(
            "Foundation Pose color/depth/mask shape mismatch: "
            f"color={color_arr.shape}, depth={depth_arr.shape}, mask={mask_arr.shape}"
        )

    height, width = depth_arr.shape
    metadata = {
        "input_hw": [int(height), int(width)],
        "mask_area_px": int(mask_arr.sum()),
        "mode": "full_frame",
    }
    return (
        np.ascontiguousarray(K, dtype=np.float32),
        np.ascontiguousarray(color_arr, dtype=np.uint8),
        np.ascontiguousarray(depth_arr, dtype=np.float32),
        np.ascontiguousarray(mask_arr.astype(np.uint8)),
        metadata,
    )
