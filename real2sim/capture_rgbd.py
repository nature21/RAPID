#!/usr/bin/env python3
"""Capture a registered RealSense RGB-D observation as a Real2Sim scene input.

By default, each invocation creates the next available ``real_scenes/scene_N``
directory and copies ``calibration/panda_eob_calib.yaml`` into it. Initial
camera frames are deliberately discarded so auto-exposure and depth sensing
can settle before the saved observation is acquired.
Depth is averaged over ten frames by default, using only valid samples at
each pixel. The first frame supplies the RGB image; keep the scene still.
The ``scene`` command then reconstructs the saved capture with Real2Sim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

from vistools.rgbd_format import CALIBRATION_FILENAME, _depth_visualization

# Importing librealsense can initialize native USB/runtime state.  Keep it lazy
# so format helpers, tests, and ``--help`` remain usable on machines without a
# connected RealSense camera.
rs: Any | None = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES_ROOT = REPO_ROOT / "real_scenes"
DEFAULT_CALIBRATION = REPO_ROOT / "calibration" / "panda_eob_calib.yaml"
DEFAULT_WARMUP_FRAMES = 30
DEFAULT_DEPTH_FRAMES = 10


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """One color-aligned metric-depth observation."""

    intrinsics: np.ndarray
    color_rgb: np.ndarray
    depth_meters: np.ndarray
    depth_scale: float
    frame_number: int
    timestamp_ms: float
    camera_serial: str
    depth_frames_averaged: int = 1


class CameraClient:
    """Small RealSense client that returns depth registered to the color image."""

    def __init__(self, *, warmup_frames: int = DEFAULT_WARMUP_FRAMES) -> None:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")

        realsense = _realsense()
        self.pipeline = realsense.pipeline()
        self.config = realsense.config()
        self.profile = self.pipeline.start(self.config)
        self.align = realsense.align(realsense.stream.color)
        self._stopped = False

        try:
            self._warm_up(warmup_frames)
        except Exception:
            self.stop()
            raise

        print(f"Camera initialized; discarded {warmup_frames} warm-up frames.")

    def _warm_up(self, frame_count: int) -> None:
        """Consume and discard complete aligned frames before capture."""

        accepted = 0
        while accepted < frame_count:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            if aligned.get_depth_frame() and aligned.get_color_frame():
                accepted += 1

    def get_observation(self) -> CameraObservation:
        """Capture the next complete color/depth pair after warm-up."""

        while True:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if depth_frame and color_frame:
                break

        color_rgb = _color_frame_as_rgb(color_frame)
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_scale = float(depth_frame.get_units())
        depth_meters = depth_raw.astype(np.float32) * depth_scale
        if color_rgb.shape[:2] != depth_meters.shape:
            raise ValueError(
                "aligned RGB/depth shape mismatch: "
                f"{color_rgb.shape[:2]} versus {depth_meters.shape}"
            )

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        intrinsics = np.asarray(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        device = self.profile.get_device()
        realsense = _realsense()
        camera_serial = (
            device.get_info(realsense.camera_info.serial_number)
            if device.supports(realsense.camera_info.serial_number)
            else ""
        )

        print(
            "Captured observation: "
            f"RGB={color_rgb.shape}, depth={depth_meters.shape}, "
            f"scale={depth_scale:.8g} m/unit"
        )
        return CameraObservation(
            intrinsics=intrinsics,
            color_rgb=color_rgb,
            depth_meters=depth_meters,
            depth_scale=depth_scale,
            frame_number=int(color_frame.get_frame_number()),
            timestamp_ms=float(color_frame.get_timestamp()),
            camera_serial=camera_serial,
        )

    def get_averaged_observation(
        self, *, depth_frames: int = DEFAULT_DEPTH_FRAMES,
    ) -> CameraObservation:
        """Average finite positive depths per pixel, retaining the first RGB.

        Every frame is aligned to the color stream before accumulation. Pixels
        with no valid samples remain zero. Camera and scene must stay still.
        """

        if depth_frames < 1:
            raise ValueError("depth_frames must be positive")
        first = self.get_observation()
        # Own the reference image before the next camera frame is acquired.
        first = replace(first, color_rgb=first.color_rgb.copy())
        depth_sum = np.zeros(first.depth_meters.shape, dtype=np.float64)
        valid_count = np.zeros(first.depth_meters.shape, dtype=np.int32)
        for index in range(depth_frames):
            observation = first if index == 0 else self.get_observation()
            depth = observation.depth_meters
            if (
                depth.shape != depth_sum.shape
                or observation.color_rgb.shape != first.color_rgb.shape
                or not np.array_equal(observation.intrinsics, first.intrinsics)
            ):
                raise ValueError("camera geometry changed during depth averaging")
            valid = np.isfinite(depth) & (depth > 0.0)
            np.add(depth_sum, depth, out=depth_sum, where=valid)
            valid_count += valid
        averaged = np.zeros(depth_sum.shape, dtype=np.float32)
        np.divide(depth_sum, valid_count, out=averaged, where=valid_count > 0)
        print(
            f"Averaged {depth_frames} depth frames using valid samples per pixel; "
            f"valid coverage={np.count_nonzero(valid_count) / valid_count.size:.1%}."
        )
        return replace(
            first, depth_meters=averaged, depth_frames_averaged=depth_frames,
        )

    def stop(self) -> None:
        if not self._stopped:
            self.pipeline.stop()
            self._stopped = True

    def __enter__(self) -> CameraClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _color_frame_as_rgb(color_frame: Any) -> np.ndarray:
    """Return an HxWx3 RGB array without assuming the default stream format."""

    realsense = _realsense()
    image = np.asanyarray(color_frame.get_data())
    pixel_format = color_frame.profile.format()
    if pixel_format == realsense.format.rgb8:
        rgb = image
    elif pixel_format == realsense.format.bgr8:
        rgb = image[..., ::-1]
    elif pixel_format == realsense.format.rgba8:
        rgb = image[..., :3]
    elif pixel_format == realsense.format.bgra8:
        rgb = image[..., [2, 1, 0]]
    else:
        raise ValueError(f"unsupported RealSense color format: {pixel_format}")
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _realsense() -> Any:
    """Load and cache ``pyrealsense2`` only when camera access is requested."""

    global rs
    if rs is None:
        import pyrealsense2 as loaded_realsense

        rs = loaded_realsense
    return rs


def next_scene_name(scenes_root: str | Path) -> str:
    """Return ``scene_N`` using one greater than the largest existing index."""

    root = Path(scenes_root)
    largest = 0
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("scene_"):
                continue
            suffix = child.name.removeprefix("scene_")
            if suffix.isdigit():
                largest = max(largest, int(suffix))
    return f"scene_{largest + 1}"


def resolve_scene_directory(
    scenes_root: str | Path,
    scene_name: str | None,
) -> Path:
    """Resolve a new scene directory and reject ambiguous/nested names."""

    root = Path(scenes_root).expanduser().resolve()
    name = next_scene_name(root) if scene_name is None else str(scene_name).strip()
    if not name or Path(name).name != name:
        raise ValueError("scene_name must be one directory name, such as scene_2")
    if not name.startswith("scene_"):
        raise ValueError("scene_name must start with 'scene_'")
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"scene directory already exists: {destination}")
    return destination


def write_scene_directory(
    destination: str | Path,
    observation: CameraObservation,
    calibration_path: str | Path,
    *,
    warmup_frames: int,
) -> Path:
    """Atomically write one observation in the Real2Sim RGB-D input format."""

    destination = Path(destination).expanduser().resolve()
    calibration = Path(calibration_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"scene directory already exists: {destination}")
    if not calibration.is_file():
        raise FileNotFoundError(f"calibration YAML not found: {calibration}")

    color = np.asarray(observation.color_rgb, dtype=np.uint8)
    depth = np.asarray(observation.depth_meters, dtype=np.float32)
    intrinsics = np.asarray(observation.intrinsics, dtype=np.float64)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color_rgb must be HxWx3, got {color.shape}")
    if depth.shape != color.shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} does not match color {color.shape[:2]}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {intrinsics.shape}")
    if not np.any(np.isfinite(depth) & (depth > 0.0)):
        raise ValueError("captured depth contains no finite positive values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary capture directory already exists: {temporary}")

    temporary.mkdir()
    try:
        Image.fromarray(color, mode="RGB").save(temporary / "color.png")
        np.save(temporary / "depth.npy", depth, allow_pickle=False)
        Image.fromarray(_depth_visualization(depth), mode="L").save(
            temporary / "depth.png"
        )
        (temporary / "intrinsics.json").write_text(
            json.dumps(intrinsics.tolist()) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(calibration, temporary / CALIBRATION_FILENAME)
        capture_metadata = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "camera_serial": observation.camera_serial,
            "color_shape": list(color.shape),
            "depth_scale_meters_per_unit": observation.depth_scale,
            "depth_shape": list(depth.shape),
            "depth_frames_averaged": observation.depth_frames_averaged,
            "frame_number": observation.frame_number,
            "frame_timestamp_ms": observation.timestamp_ms,
            "warmup_frames_discarded": int(warmup_frames),
        }
        (temporary / "capture_metadata.json").write_text(
            json.dumps(capture_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def capture_scene(
    *,
    scenes_root: str | Path = DEFAULT_SCENES_ROOT,
    scene_name: str | None = None,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    depth_frames: int = DEFAULT_DEPTH_FRAMES,
) -> Path:
    """Warm up, average a depth burst, and create a Real2Sim scene directory."""

    if depth_frames < 1:
        raise ValueError("depth_frames must be positive")
    destination = resolve_scene_directory(scenes_root, scene_name)
    calibration = Path(calibration_path).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"calibration YAML not found: {calibration}")

    print(f"Output scene: {destination}")
    print(f"Calibration: {calibration}")
    with CameraClient(warmup_frames=warmup_frames) as camera:
        observation = camera.get_averaged_observation(depth_frames=depth_frames)
    return write_scene_directory(
        destination,
        observation,
        calibration,
        warmup_frames=warmup_frames,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-real2sim-capture scene",
        description=(
            "Capture RealSense RGB with averaged depth as real_scenes/scene_N "
            "and automatically reconstruct it with Real2Sim."
        ),
    )
    parser.add_argument(
        "--scene-name",
        "--name",
        default=None,
        help="Output name such as scene_2; defaults to the next available scene_N.",
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=DEFAULT_SCENES_ROOT,
        help=f"Scene root (default: {DEFAULT_SCENES_ROOT}).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"Easy Handeye YAML to copy (default: {DEFAULT_CALIBRATION}).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=(
            "Number of complete initial frames to discard before capture "
            f"(default: {DEFAULT_WARMUP_FRAMES})."
        ),
    )
    parser.add_argument(
        "--depth-frames",
        type=int,
        default=DEFAULT_DEPTH_FRAMES,
        help=(
            "Frames to average using only finite positive depths per pixel "
            f"(default: {DEFAULT_DEPTH_FRAMES}). Keep camera and scene still; "
            "RGB comes from the first frame. Use 1 for single-frame capture."
        ),
    )
    return parser


def build_mode_parser() -> argparse.ArgumentParser:
    """Build the top-level parser that requires an explicit capture mode."""

    parser = argparse.ArgumentParser(
        prog="rapid-real2sim-capture",
        description="Capture calibrated RealSense RGB-D scenes or demonstrations.",
    )
    modes = parser.add_subparsers(dest="mode", required=True, metavar="MODE")
    modes.add_parser(
        "scene",
        add_help=False,
        help="Capture RGB with averaged depth under real_scenes/scene_N and reconstruct it.",
    )
    modes.add_parser(
        "demo",
        add_help=False,
        help=(
            "Record synchronized L515 RGB-D video under "
            "demos/strategy/demo_N."
        ),
    )
    return parser


def _scene_main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.depth_frames < 1:
        parser.error("--depth-frames must be positive")
    scene_dir = capture_scene(
        scenes_root=args.scenes_root,
        scene_name=args.scene_name,
        calibration_path=args.calibration,
        warmup_frames=args.warmup_frames,
        depth_frames=args.depth_frames,
    )
    print(f"Saved Real2Sim capture: {scene_dir}")
    print(f"Starting Real2Sim reconstruction: {scene_dir.name}", flush=True)
    from real2sim.workflow import main as reconstruct_main

    return reconstruct_main([str(scene_dir), "--name", scene_dir.name])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["scene"]:
        return _scene_main(argv[1:])
    if argv[:1] == ["demo"]:
        from real2sim.capture_rgbd_demo import main as demo_main

        return demo_main(argv[1:])

    # This handles the empty command, top-level --help, and unknown modes with
    # argparse's standard diagnostics. A valid mode is dispatched above so its
    # own --help contains the complete mode-specific option list.
    build_mode_parser().parse_args(argv)
    raise AssertionError("argparse accepted an undispatched capture mode")


if __name__ == "__main__":
    raise SystemExit(main())
