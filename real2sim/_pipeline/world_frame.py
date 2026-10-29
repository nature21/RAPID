"""World-frame selection and support-plane fitting for real2sim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from real2sim._pipeline.images import is_near_identity_pose
from real2sim._pipeline.progress import log
from real2sim._pipeline.vlm import (
    accepted_confidences,
    raw_vlm_response_from_error,
)
from real2sim._shared.naming import canonical_object_name
from real2sim._stages.perception import segment_support_surface
from real2sim._stages.vlm import select_support_surface
from real2sim.schema import PipelineConfig
from real2sim.support_plane import (
    SupportFrame,
    fit_support_plane,
    masked_points,
    support_frame_from_plane,
    support_frame_metadata,
)


def build_world_frame(
    *,
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    pointmap_opencv: np.ndarray,
    depth_c2w_opencv: np.ndarray,
    depth_w2c_opencv: np.ndarray,
    depth_extrinsics_source: str,
    masks_dir: Path,
    config: PipelineConfig,
    warnings: list[str],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    mode = normalize_world_frame_mode(config.world_frame_mode)
    depth_metadata = {
        "camera_input_c2w_opencv": depth_c2w_opencv,
        "camera_input_w2c_opencv": depth_w2c_opencv,
        "camera_input_extrinsics_source": depth_extrinsics_source,
        "depth_anything_c2w_opencv": depth_c2w_opencv,
        "depth_anything_w2c_opencv": depth_w2c_opencv,
        "depth_anything_extrinsics_source": depth_extrinsics_source,
    }

    if mode == "camera":
        transform = np.eye(4, dtype=np.float64)
        return (
            transform,
            {
                "mode": mode,
                "source": "camera_identity",
                "T_world_from_camera_opencv": transform,
                **depth_metadata,
            },
            None,
        )

    if mode == "depth_anything":
        if is_near_identity_pose(depth_c2w_opencv):
            warnings.append(
                "Depth Anything camera extrinsics are near identity; "
                "the world frame is effectively the OpenCV camera frame."
            )
        return (
            depth_c2w_opencv,
            {
                "mode": mode,
                "source": depth_extrinsics_source,
                "T_world_from_camera_opencv": depth_c2w_opencv,
                **depth_metadata,
            },
            None,
        )

    if mode not in {"camera_extrinsics", "support_surface"}:
        raise ValueError(f"unsupported world_frame_mode {config.world_frame_mode!r}")
    calibrated_world = mode == "camera_extrinsics"
    if calibrated_world and is_near_identity_pose(depth_c2w_opencv):
        warnings.append(
            "Camera extrinsics are near identity; the calibrated world frame "
            "is effectively the OpenCV camera frame."
        )

    support_metadata: dict[str, Any] = {
        "mode": mode,
        "source": depth_extrinsics_source if calibrated_world else "support_surface",
        "T_world_from_camera_opencv": (
            depth_c2w_opencv if calibrated_world else None
        ),
        **depth_metadata,
    }
    support_mask: np.ndarray | None = None

    try:
        support_selection, support_response = _select_support_surface_with_retry(
            image,
            detected_objects,
            config,
        )
        support_metadata["support_selection"] = support_selection
        support_metadata["support_vlm_raw_response"] = support_response
        log(
            config,
            "[real2sim] support surface selected: "
            f"{support_selection['object_name']!r} via {support_selection['sam_query']!r} "
            f"({support_selection['confidence']})",
        )

        support_mask, support_sam = segment_support_surface(
            image,
            support_selection,
            config=config,
        )
        support_mask_path = masks_dir / "support_surface.png"
        Image.fromarray(support_mask * 255).save(support_mask_path)
        support_sam["mask_file"] = support_mask_path.name
        support_metadata["support_sam"] = support_sam
        log(
            config,
            "[real2sim] support SAM mask: "
            f"query={support_sam['query']!r}, area={support_sam['mask_area_px']}, "
            f"score={support_sam['score']:.3f}",
        )

        frame = _fit_world_frame_from_mask(
            pointmap_opencv,
            support_mask,
            config,
        )
        if calibrated_world:
            support_metadata.update(
                _calibrated_support_plane_metadata(
                    frame,
                    depth_c2w_opencv,
                    source="support_surface",
                )
            )
            return depth_c2w_opencv, support_metadata, support_mask
        support_metadata.update(support_frame_metadata(frame, source="support_surface"))
        return frame.T_world_from_camera_opencv, support_metadata, support_mask
    except Exception as exc:
        if not config.allow_global_support_fallback:
            raise RuntimeError("failed to estimate the support plane") from exc
        warnings.append(
            "Support-plane estimation failed; used global dominant-plane fallback: "
            f"{type(exc).__name__}: {exc}"
        )
        support_metadata["support_surface_error"] = f"{type(exc).__name__}: {exc}"
        support_error_responses = getattr(exc, "vlm_responses", None)
        if isinstance(support_error_responses, list):
            support_metadata["support_vlm_error_responses"] = support_error_responses
        if not calibrated_world:
            support_metadata["source"] = "global_pointcloud_fallback"
        inferred_support = _infer_support_selection_from_detected_objects(
            detected_objects,
        )
        if inferred_support is not None:
            support_metadata["support_selection"] = inferred_support
            support_metadata["support_selection_source"] = (
                "detected_object_heuristic_after_support_failure"
            )
            log(
                config,
                "[real2sim] support object inferred from detected objects "
                f"after fallback: {inferred_support['object_name']!r}",
            )
        frame = _fit_world_frame_from_mask(pointmap_opencv, None, config)
        if calibrated_world:
            support_metadata.update(
                _calibrated_support_plane_metadata(
                    frame,
                    depth_c2w_opencv,
                    source="global_pointcloud_fallback",
                )
            )
            return depth_c2w_opencv, support_metadata, support_mask
        support_metadata.update(
            support_frame_metadata(frame, source="global_pointcloud_fallback")
        )
        return frame.T_world_from_camera_opencv, support_metadata, support_mask


def normalize_world_frame_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"support", "support_surface"}:
        return "support_surface"
    if normalized == "camera":
        return "camera"
    if normalized in {"camera_extrinsics", "extrinsics", "robot"}:
        return "camera_extrinsics"
    if normalized in {"depth", "depth_anything"}:
        return "depth_anything"
    raise ValueError(
        "world_frame_mode must be one of support_surface, camera, "
        f"camera_extrinsics, or depth_anything; got {value!r}"
    )


def _infer_support_selection_from_detected_objects(
    detected_objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[int, dict[str, Any]] | None = None
    for obj in detected_objects:
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
        description = str(obj.get("description", "")).strip()
        text = canonical_object_name(f"{name} {description}")
        score = 0
        if any(
            term in text
            for term in (
                "table",
                "desk",
                "tabletop",
                "counter",
                "floor",
                "workspace",
            )
        ):
            score += 6
        if any(
            term in text
            for term in (
                "platform",
                "tray",
                "shelf",
                "mat",
                "base",
                "support surface",
                "supporting surface",
            )
        ):
            score += 5
        if any(
            term in text
            for term in (
                "supporting",
                "supports",
                "support",
                "surface",
                "top",
                "workspace",
            )
        ):
            score += 2
        if any(
            term in text
            for term in (
                "raised",
                "elevated",
                "resting on",
                "sitting on",
                "placed on",
            )
        ):
            score -= 4
        if any(
            term in text
            for term in (
                "supporting the platform",
                "supports the platform",
                "under the platform",
                "beneath the platform",
            )
        ):
            score += 5
        if any(
            term in text
            for term in (
                "mug",
                "cup",
                "toy",
                "figurine",
                "case",
                "box",
                "package",
                "container",
                "tool",
                "card",
            )
        ):
            score -= 4
        if score <= 0:
            continue
        selection = {
            "object_name": name,
            "sam_query": _support_sam_query_for_name(name, text),
            "reason": (
                "detected object appears to be the foundational ground/workspace "
                "surface; "
                "used after support VLM selection failed"
            ),
            "confidence": "medium" if score < 7 else "high",
            "inferred": True,
        }
        if best is None or score > best[0]:
            best = (score, selection)
    return None if best is None else best[1]


def _support_sam_query_for_name(name: str, canonical_text: str) -> str:
    if any(term in canonical_text for term in ("table", "desk", "counter")):
        return f"{name} tabletop supporting surface"
    if any(
        term in canonical_text
        for term in ("platform", "floor", "tray", "shelf", "mat", "base")
    ):
        return f"{name} top supporting surface"
    return f"{name} supporting surface"


def _select_support_surface_with_retry(
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    config: PipelineConfig,
) -> tuple[dict[str, Any], str]:
    accepted = accepted_confidences(config.support_confidence)
    first_error: Exception | None = None
    first_response = ""
    try:
        return select_support_surface(
            image,
            detected_objects,
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
            accepted_confidences=accepted,
            retry=False,
        )
    except Exception as exc:
        first_error = exc
        first_response = raw_vlm_response_from_error(exc)

    try:
        selection, response = select_support_surface(
            image,
            detected_objects,
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
            accepted_confidences=accepted,
            retry=True,
        )
        selection["retry_used"] = True
        selection["first_error"] = f"{type(first_error).__name__}: {first_error}"
        return selection, response
    except Exception as exc:
        second_response = raw_vlm_response_from_error(exc)
        error = RuntimeError(
            "support VLM selection failed after retry: "
            f"{type(first_error).__name__}: {first_error}; "
            f"{type(exc).__name__}: {exc}"
        )
        error.vlm_responses = [
            {
                "attempt": "initial",
                "stage": "support_surface_selection",
                "error": f"{type(first_error).__name__}: {first_error}",
                "response": first_response,
            },
            {
                "attempt": "retry",
                "stage": "support_surface_selection",
                "error": f"{type(exc).__name__}: {exc}",
                "response": second_response,
            },
        ]
        raise error from exc


def _fit_world_frame_from_mask(
    pointmap_opencv: np.ndarray,
    mask: np.ndarray | None,
    config: PipelineConfig,
):
    points = masked_points(pointmap_opencv, mask)
    plane = fit_support_plane(
        points,
        distance_threshold=config.support_plane_distance_threshold,
        iterations=config.support_plane_ransac_iterations,
        max_points=config.support_plane_max_points,
        min_inlier_ratio=config.support_plane_min_inlier_ratio,
        seed=config.support_plane_seed,
    )
    return support_frame_from_plane(plane)


def _calibrated_support_plane_metadata(
    frame: SupportFrame,
    world_from_camera_opencv: np.ndarray,
    *,
    source: str,
) -> dict[str, Any]:
    """Describe a fitted support plane without changing the calibrated world."""

    camera_metadata = support_frame_metadata(frame, source=source)
    plane_camera = dict(camera_metadata["plane"])
    transform = np.asarray(world_from_camera_opencv, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    normal_world = rotation @ np.asarray(
        plane_camera["normal_camera_opencv"], dtype=np.float64
    )
    normal_world /= max(float(np.linalg.norm(normal_world)), 1e-12)
    centroid_world = (
        rotation
        @ np.asarray(plane_camera["centroid_camera_opencv"], dtype=np.float64)
        + translation
    )
    if normal_world[2] < 0.0:
        normal_world = -normal_world
    d_world = -float(normal_world @ centroid_world)
    tilt_degrees = float(
        np.degrees(np.arccos(np.clip(normal_world[2], -1.0, 1.0)))
    )

    return {
        "support_plane_source": source,
        "plane": plane_camera,
        "support_plane_world": {
            "normal": normal_world,
            "centroid": centroid_world,
            "d": d_world,
            "tilt_degrees_from_world_z": tilt_degrees,
            "inlier_count": plane_camera["inlier_count"],
            "candidate_count": plane_camera["candidate_count"],
            "inlier_ratio": plane_camera["inlier_ratio"],
            "residual_p95": plane_camera["residual_p95"],
        },
        # MuJoCo's floor remains horizontal in the calibrated robot frame. Use
        # the fitted inlier centroid as its robust observed tabletop height.
        "support_floor_z": float(centroid_world[2]),
    }
