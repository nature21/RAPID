"""Depth and support-surface segmentation stages."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from real2sim.schema import PipelineConfig
from real2sim._stages.helpers import list_or_none
from real2sim.vfm_clients.depth_anything import depth_anything_request
from real2sim.vfm_clients.sam3 import sam3_request


def request_depth(
    image: Image.Image,
    config: PipelineConfig,
    use_ray_pose: bool | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return depth_anything_request(
        image,
        focal=config.focal,
        intrinsics=config.intrinsics,
        extrinsics=config.extrinsics,
        use_ray_pose=use_ray_pose,
        process_res=config.process_res,
        process_res_method=config.process_res_method,
        server_url=config.depth_server_url,
        timeout=config.depth_timeout,
        return_metadata=True,
    )


def segment_support_surface(
    image: Image.Image,
    support_selection: dict[str, Any],
    *,
    config: PipelineConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    queries = [str(support_selection["sam_query"]).strip()]
    object_name = str(support_selection["object_name"]).strip()
    if object_name and object_name not in queries:
        queries.append(object_name)

    errors: list[str] = []
    for query_index, query in enumerate(queries):
        if not query:
            continue
        scores, bboxes, masks = sam3_request(
            image=image,
            text=query,
            bbox=None,
            confidence_threshold=config.sam3_confidence_threshold,
            server_url=config.sam3_server_url,
        )
        if not scores:
            errors.append(f"{query!r}: no masks")
            continue

        best_idx = int(np.argmax(np.asarray(scores, dtype=float)))
        mask = (np.asarray(masks[best_idx]) > 0).astype(np.uint8)
        mask_area = int(mask.sum())
        min_area = max(
            int(config.min_mask_area_px),
            int(mask.size * float(config.min_mask_area_fraction)),
        )
        if mask_area < min_area:
            errors.append(f"{query!r}: mask area {mask_area} < {min_area}")
            continue

        bbox = bboxes[best_idx] if best_idx < len(bboxes) else None
        return mask, {
            "query": query,
            "query_source": "sam_query" if query_index == 0 else "object_name_retry",
            "score": float(scores[best_idx]),
            "bbox": list_or_none(bbox),
            "mask_area_px": mask_area,
            "retry_used": query_index > 0,
            "errors": errors,
        }

    raise ValueError("SAM3 could not segment support surface: " + "; ".join(errors))
