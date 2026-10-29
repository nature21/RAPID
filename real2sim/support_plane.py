"""Support-plane fitting and gravity-frame construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PlaneFit:
    normal: np.ndarray
    centroid: np.ndarray
    d: float
    inlier_count: int
    candidate_count: int
    sampled_count: int
    inlier_ratio: float
    residual_mean: float
    residual_median: float
    residual_p95: float


@dataclass(slots=True)
class SupportFrame:
    T_world_from_camera_opencv: np.ndarray
    plane: PlaneFit
    origin_camera_opencv: np.ndarray
    x_axis_camera_opencv: np.ndarray
    y_axis_camera_opencv: np.ndarray
    z_axis_camera_opencv: np.ndarray
    orientation_source: str


def masked_points(pointmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return finite 3D points from an HxWx3 point map, optionally under a mask."""

    points = np.asarray(pointmap, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"pointmap must have shape (H, W, 3), got {points.shape}")

    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.shape != points.shape[:2]:
            raise ValueError(
                f"mask shape {mask_arr.shape} does not match pointmap {points.shape[:2]}"
            )
        points = points[mask_arr]
    else:
        points = points.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def fit_support_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    max_points: int,
    min_inlier_ratio: float,
    seed: int = 0,
) -> PlaneFit:
    """Fit a plane to 3D points with deterministic RANSAC and SVD refinement."""

    candidates = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    candidate_count = int(candidates.shape[0])
    if candidate_count < 3:
        raise ValueError(f"need at least 3 finite points to fit a plane, got {candidate_count}")
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if max_points < 3:
        raise ValueError("max_points must be >= 3")

    rng = np.random.default_rng(seed)
    if candidate_count > max_points:
        sampled_indices = rng.choice(candidate_count, max_points, replace=False)
        sampled = candidates[sampled_indices]
    else:
        sampled = candidates
    sampled_count = int(sampled.shape[0])

    best_normal: np.ndarray | None = None
    best_d: float | None = None
    best_score = -1
    best_error = np.inf

    for _ in range(iterations):
        sample = sampled[rng.choice(sampled_count, 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        distances = np.abs(sampled @ normal + d)
        inliers = distances < distance_threshold
        score = int(inliers.sum())
        if score == 0:
            continue
        error = float(np.mean(distances[inliers]))
        if score > best_score or (score == best_score and error < best_error):
            best_normal = normal
            best_d = d
            best_score = score
            best_error = error

    if best_normal is None or best_d is None:
        raise ValueError("RANSAC did not find a valid support plane")

    distances = np.abs(candidates @ best_normal + best_d)
    inlier_mask = distances < distance_threshold
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / candidate_count)
    if inlier_ratio < min_inlier_ratio:
        raise ValueError(
            "support plane inlier ratio too low: "
            f"{inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
        )

    inlier_points = candidates[inlier_mask]
    refine_points = _sample_points(inlier_points, max_points, rng)
    centroid = np.mean(refine_points, axis=0)
    _, _, vh = np.linalg.svd(refine_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    d = -float(normal @ centroid)

    refined_distances = np.abs(candidates @ normal + d)
    refined_inlier_mask = refined_distances < distance_threshold
    refined_inlier_count = int(refined_inlier_mask.sum())
    refined_inlier_ratio = float(refined_inlier_count / candidate_count)
    if refined_inlier_count < 3 or refined_inlier_ratio < min_inlier_ratio:
        raise ValueError(
            "refined support plane inlier ratio too low: "
            f"{refined_inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
        )

    refined_inlier_points = candidates[refined_inlier_mask]
    centroid = np.mean(_sample_points(refined_inlier_points, max_points, rng), axis=0)
    signed_centroid_distance = float(normal @ centroid + d)
    centroid = centroid - signed_centroid_distance * normal
    d = -float(normal @ centroid)
    residuals = np.abs(candidates[refined_inlier_mask] @ normal + d)

    return PlaneFit(
        normal=normal.astype(np.float64),
        centroid=centroid.astype(np.float64),
        d=float(d),
        inlier_count=refined_inlier_count,
        candidate_count=candidate_count,
        sampled_count=sampled_count,
        inlier_ratio=refined_inlier_ratio,
        residual_mean=float(np.mean(residuals)),
        residual_median=float(np.median(residuals)),
        residual_p95=float(np.percentile(residuals, 95)),
    )


def support_frame_from_plane(plane: PlaneFit) -> SupportFrame:
    """Create a camera-to-world transform whose +z is the support normal."""

    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    d = float(plane.d)

    # OpenCV camera +z points along the viewing ray. For this tabletop-style
    # setup the camera is above the scene and looking down, so world +z must
    # point back toward the camera, i.e. against OpenCV +z.
    if normal[2] > 0.0:
        normal = -normal
        d = -d

    origin = np.asarray(plane.centroid, dtype=np.float64).reshape(3)
    origin = origin - float(normal @ origin + d) * normal

    x_axis = _project_axis(np.array([1.0, 0.0, 0.0]), normal)
    if x_axis is None:
        x_axis = _project_axis(np.array([0.0, 1.0, 0.0]), normal)
    if x_axis is None:
        x_axis = _orthogonal_axis(normal)

    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if float(y_axis @ camera_up) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
    x_axis = np.cross(y_axis, normal)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)

    rotation = np.stack([x_axis, y_axis, normal], axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin

    return SupportFrame(
        T_world_from_camera_opencv=transform,
        plane=PlaneFit(
            normal=normal,
            centroid=origin,
            d=-float(normal @ origin),
            inlier_count=plane.inlier_count,
            candidate_count=plane.candidate_count,
            sampled_count=plane.sampled_count,
            inlier_ratio=plane.inlier_ratio,
            residual_mean=plane.residual_mean,
            residual_median=plane.residual_median,
            residual_p95=plane.residual_p95,
        ),
        origin_camera_opencv=origin,
        x_axis_camera_opencv=x_axis,
        y_axis_camera_opencv=y_axis,
        z_axis_camera_opencv=normal,
        orientation_source="camera_above_looking_down",
    )


def support_frame_metadata(frame: SupportFrame, *, source: str) -> dict[str, Any]:
    plane = frame.plane
    return {
        "source": source,
        "T_world_from_camera_opencv": frame.T_world_from_camera_opencv,
        "origin_camera_opencv": frame.origin_camera_opencv,
        "x_axis_camera_opencv": frame.x_axis_camera_opencv,
        "y_axis_camera_opencv": frame.y_axis_camera_opencv,
        "z_axis_camera_opencv": frame.z_axis_camera_opencv,
        "orientation_source": frame.orientation_source,
        "plane": {
            "normal_camera_opencv": plane.normal,
            "centroid_camera_opencv": plane.centroid,
            "d_camera_opencv": plane.d,
            "inlier_count": plane.inlier_count,
            "candidate_count": plane.candidate_count,
            "sampled_count": plane.sampled_count,
            "inlier_ratio": plane.inlier_ratio,
            "residual_mean": plane.residual_mean,
            "residual_median": plane.residual_median,
            "residual_p95": plane.residual_p95,
        },
    }


def _sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    return points[rng.choice(points.shape[0], max_points, replace=False)]


def _project_axis(axis: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = axis - float(axis @ normal) * normal
    norm = float(np.linalg.norm(projected))
    if norm < 1e-8:
        return None
    return projected / norm


def _orthogonal_axis(normal: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ normal)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    projected = axis - float(axis @ normal) * normal
    return projected / max(float(np.linalg.norm(projected)), 1e-12)
