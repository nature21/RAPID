"""Camera and point-map utilities for the real2sim pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


T_OPENCV_TO_P3D = np.diag([-1.0, -1.0, 1.0, 1.0])
T_P3D_TO_OPENCV = T_OPENCV_TO_P3D
T_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_intrinsics(
    metadata: dict[str, Any],
    width: int,
    height: int,
    warnings: list[str],
) -> tuple[np.ndarray, str]:
    if "intrinsics" in metadata:
        K = np.asarray(metadata["intrinsics"], dtype=np.float64)
        if K.shape == (3, 3):
            return K, metadata.get("intrinsics_source", "camera_input")
        warnings.append(f"ignored invalid intrinsics shape {K.shape}")

    if "focal_px" in metadata and metadata["focal_px"] is not None:
        focal = float(metadata["focal_px"])
        source = "depth_anything_focal_px"
    else:
        focal = 0.5 * float(width + height)
        source = "image_size_fallback"
        warnings.append(
            "Camera input did not provide intrinsics; used image-size focal fallback"
        )

    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, source


def camera_extrinsics(
    metadata: dict[str, Any],
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    if "extrinsics" not in metadata:
        warnings.append(
            "Camera input did not provide extrinsics; "
            "world frame equals OpenCV camera frame"
        )
        c2w = np.eye(4, dtype=np.float64)
        return c2w, np.linalg.inv(c2w), "identity_fallback"

    extrinsics = as_homogeneous(metadata["extrinsics"])
    convention = str(metadata.get("extrinsics_convention", "opencv_w2c_3x4")).lower()
    if "c2w" in convention or "cam2world" in convention or "camera_to_world" in convention:
        c2w = extrinsics
        w2c = np.linalg.inv(c2w)
        source = metadata.get("extrinsics_source", "camera_input_c2w")
    else:
        w2c = extrinsics
        c2w = np.linalg.inv(w2c)
        source = metadata.get("extrinsics_source", "camera_input_w2c")
    return c2w, w2c, source


def depth_to_pointmap(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth into OpenCV camera coordinates."""

    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth
    x = (u - float(K[0, 2])) * z / float(K[0, 0])
    y = (v - float(K[1, 2])) * z / float(K[1, 1])
    pointmap = np.stack([x, y, z], axis=-1)
    invalid = ~np.isfinite(z) | (z <= 0.0)
    pointmap[invalid] = np.nan
    return pointmap.astype(np.float32)


def transform_pointmap(pointmap: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = as_homogeneous(transform)
    pointmap = np.asarray(pointmap, dtype=np.float32)
    flat = pointmap.reshape(-1, 3)
    out = np.full_like(flat, np.nan, dtype=np.float32)
    valid = np.isfinite(flat).all(axis=1)
    if np.any(valid):
        pts_h = np.concatenate(
            [flat[valid].astype(np.float64), np.ones((int(valid.sum()), 1))],
            axis=1,
        )
        out[valid] = (transform @ pts_h.T).T[:, :3].astype(np.float32)
    return out.reshape(pointmap.shape)


def as_homogeneous(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"expected transform shape (3, 4) or (4, 4), got {matrix.shape}")
