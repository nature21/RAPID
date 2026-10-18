"""Loading helpers for captured RGB-D Real2Sim inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COLOR_FILENAMES = ("color.png", "rgb.png")
DEPTH_FILENAME = "depth.npy"
INTRINSICS_FILENAME = "intrinsics.json"
CALIBRATION_GLOB = "*_eob_calib.yaml"


@dataclass(frozen=True, slots=True)
class RGBDSceneInput:
    """A registered RGB-D frame and its supplied camera parameters."""

    directory: Path
    image_path: Path
    depth: np.ndarray
    camera_metadata: dict[str, Any]


def is_rgbd_scene_directory(path: str | Path) -> bool:
    """Return whether ``path`` has the required captured-scene files."""

    root = Path(path)
    return bool(
        root.is_dir()
        and _find_color_path(root) is not None
        and (root / DEPTH_FILENAME).is_file()
        and (root / INTRINSICS_FILENAME).is_file()
        and _find_calibration_path(root) is not None
    )


def load_rgbd_scene(path: str | Path) -> RGBDSceneInput:
    """Load the RGB-D directory format used under ``real_scenes/``.

    Depth values are read from ``depth.npy`` and are expected to be metric and
    registered to the color image. The Easy Handeye eye-on-base transform is a
    ROS parent-to-child transform, so its numeric matrix maps camera points
    into the parent (normally robot-base) frame and is therefore OpenCV c2w.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RGB-D scene directory not found: {root}")

    image_path = _find_color_path(root)
    if image_path is None:
        expected = " or ".join(COLOR_FILENAMES)
        raise FileNotFoundError(f"RGB-D scene is missing {expected}: {root}")

    depth_path = root / DEPTH_FILENAME
    intrinsics_path = root / INTRINSICS_FILENAME
    calibration_path = _find_calibration_path(root)
    for required in (depth_path, intrinsics_path):
        if not required.is_file():
            raise FileNotFoundError(f"RGB-D scene is missing {required.name}: {root}")
    if calibration_path is None:
        raise FileNotFoundError(
            f"RGB-D scene is missing an Easy Handeye {CALIBRATION_GLOB} file: {root}"
        )

    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"{depth_path} must contain an HxW depth map, got {depth.shape}")
    if not np.any(np.isfinite(depth) & (depth > 0.0)):
        raise ValueError(f"{depth_path} contains no finite positive depth values")

    intrinsics = _load_intrinsics(intrinsics_path)
    extrinsics, calibration_metadata = _load_easy_handeye(calibration_path)
    camera_metadata = {
        "provider": "rgbd_scene_directory",
        "input_directory": str(root),
        "depth_source": str(depth_path),
        "depth_units": "meters",
        "intrinsics": intrinsics,
        "intrinsics_source": str(intrinsics_path),
        "extrinsics": extrinsics,
        "extrinsics_convention": "opencv_c2w_4x4",
        "extrinsics_source": (
            f"{calibration_path}: "
            f"{calibration_metadata['parent_frame']}<-"
            f"{calibration_metadata['child_frame']}"
        ),
        "calibration": calibration_metadata,
    }
    return RGBDSceneInput(
        directory=root,
        image_path=image_path,
        depth=depth,
        camera_metadata=camera_metadata,
    )


def _find_color_path(root: Path) -> Path | None:
    for name in COLOR_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _find_calibration_path(root: Path) -> Path | None:
    matches = sorted(root.glob(CALIBRATION_GLOB))
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"RGB-D scene has multiple Easy Handeye calibrations: {names}")
    return matches[0] if matches else None


def _load_intrinsics(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("intrinsics", "K", "camera_matrix"):
            if key in value:
                value = value[key]
                break
    intrinsics = np.asarray(value, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{path} must contain a 3x3 intrinsic matrix, got {intrinsics.shape}")
    if not np.isfinite(intrinsics).all():
        raise ValueError(f"{path} contains non-finite intrinsic values")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError(f"{path} must contain positive fx and fy")
    return intrinsics


def _load_easy_handeye(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in the rapid env
        raise RuntimeError("PyYAML is required to load RGB-D calibration files") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    parameters = payload.get("parameters")
    transform = payload.get("transformation")
    if not isinstance(parameters, dict) or not isinstance(transform, dict):
        raise ValueError(f"{path} must contain parameters and transformation mappings")

    eye_on_hand = parameters.get("eye_on_hand")
    if eye_on_hand is not False:
        raise ValueError(
            f"{path} must be an eye-on-base calibration (eye_on_hand: false)"
        )
    parent_frame = str(parameters.get("robot_base_frame", "")).strip()
    child_frame = str(parameters.get("tracking_base_frame", "")).strip()
    if not parent_frame or not child_frame:
        raise ValueError(f"{path} is missing the calibration parent/child frame names")

    names = ("x", "y", "z", "qx", "qy", "qz", "qw")
    try:
        values = {name: float(transform[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid transformation mapping") from exc
    if not np.isfinite(np.fromiter(values.values(), dtype=np.float64)).all():
        raise ValueError(f"{path} contains non-finite transformation values")

    rotation = _rotation_from_xyzw(
        np.asarray(
            [values["qx"], values["qy"], values["qz"], values["qw"]],
            dtype=np.float64,
        )
    )
    camera_to_parent = np.eye(4, dtype=np.float64)
    camera_to_parent[:3, :3] = rotation
    camera_to_parent[:3, 3] = [values["x"], values["y"], values["z"]]

    metadata: dict[str, Any] = {
        "path": str(path),
        "format": "easy_handeye",
        "eye_on_hand": False,
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "robot_base_frame": str(parameters.get("robot_base_frame", "")),
        "robot_effector_frame": str(parameters.get("robot_effector_frame", "")),
        "tracking_marker_frame": str(parameters.get("tracking_marker_frame", "")),
    }
    calibration = payload.get("calibration")
    residual = payload.get("residual")
    if isinstance(calibration, dict):
        metadata["method"] = calibration.get("method")
        metadata["n_samples"] = calibration.get("n_samples")
    if isinstance(residual, dict):
        metadata["residual"] = residual
    return camera_to_parent, metadata


def _rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Easy Handeye calibration quaternion cannot be zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
