"""Point-cloud export helpers for real2sim scene bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_colored_pointcloud_ply(
    path: str | Path,
    pointmap: np.ndarray,
    image_rgb: np.ndarray,
    *,
    stride: int = 1,
) -> int:
    """Save a colored point cloud from an HxWx3 point map as binary PLY.

    Returns the number of written points.
    """

    path = Path(path)
    points = np.asarray(pointmap, dtype=np.float32)
    colors = np.asarray(image_rgb, dtype=np.uint8)

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"pointmap must have shape (H, W, 3), got {points.shape}")
    if colors.shape[:2] != points.shape[:2] or colors.shape[-1] != 3:
        raise ValueError(
            f"image_rgb must have shape (H, W, 3) matching pointmap, got {colors.shape}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    points = points[::stride, ::stride].reshape(-1, 3)
    colors = colors[::stride, ::stride].reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(path, points, colors)
    return int(points.shape[0])


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        vertices.tofile(f)
