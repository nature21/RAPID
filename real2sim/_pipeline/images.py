"""Image and point-map helpers used by the real2sim pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def as_rgb_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    raise TypeError(
        "image must be a PIL image, numpy array, or path; "
        f"got {type(image).__name__}"
    )


def resize_image_for_processing(
    image: Image.Image,
    max_size: int | None,
) -> tuple[Image.Image, dict[str, Any]]:
    width, height = image.size
    metadata = {
        "resized": False,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    if max_size is None or int(max_size) <= 0 or max(width, height) <= int(max_size):
        return image, metadata

    max_size = int(max_size)
    scale = float(max_size) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((new_width, new_height), resampling.LANCZOS)
    metadata.update(
        {
            "resized": True,
            "scale_x": float(new_width) / float(width),
            "scale_y": float(new_height) / float(height),
        }
    )
    return resized, metadata


def scale_intrinsics_for_resize(
    intrinsics: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    K = np.asarray(intrinsics, dtype=np.float64).copy()
    K[0, 0] *= float(scale_x)
    K[0, 2] *= float(scale_x)
    K[1, 1] *= float(scale_y)
    K[1, 2] *= float(scale_y)
    return K


def masked_pointmap(pointmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.full_like(pointmap, np.nan, dtype=np.float32)
    mask_arr = np.asarray(mask).astype(bool)
    masked[mask_arr] = pointmap[mask_arr]
    return masked


def is_near_identity_pose(transform: np.ndarray) -> bool:
    transform = np.asarray(transform, dtype=float)
    return bool(
        np.linalg.norm(transform[:3, :3] - np.eye(3)) < 1e-2
        and np.linalg.norm(transform[:3, 3]) < 1e-2
    )
