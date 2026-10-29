#!/usr/bin/env python3
"""Record synchronized, calibrated L515 RGB-D demonstration videos.

Each recording is written atomically to ``demos/strategy/demo_N``.
The color stream is an ordinary H.264 MP4, while aligned sensor depth is kept
losslessly as 16-bit FFV1 in Matroska.  A viewable depth-preview MP4 and exact
per-frame timestamps share the same frame indexing.  The first RGB-D frame is
also saved in the existing Real2Sim scene-directory format so the beginning of
the demonstration can be reconstructed without first decoding either video.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import threading
import time
from typing import Any, Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image

from real2sim.capture_rgbd import (
    CALIBRATION_FILENAME,
    DEFAULT_CALIBRATION,
    DEFAULT_WARMUP_FRAMES,
    REPO_ROOT,
    _color_frame_as_rgb,
    _depth_visualization,
    _realsense,
)

from vistools.rgbd_format import (
    RGB_VIDEO_FILENAME,
    DEPTH_VIDEO_FILENAME,
    DEPTH_PREVIEW_FILENAME,
    TIMESTAMPS_FILENAME,
    CAPTURE_METADATA_FILENAME,
)

DEFAULT_DEMOS_ROOT = REPO_ROOT / "demos" / "strategy"
DEFAULT_COLOR_WIDTH = 1280
DEFAULT_COLOR_HEIGHT = 720
DEFAULT_DEPTH_WIDTH = 640
DEFAULT_DEPTH_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_PREVIEW_NEAR_METERS = 0.2
DEFAULT_PREVIEW_FAR_METERS = 2.0
RGB_VIDEO_CRF = 16
RGB_VIDEO_PRESET = "medium"


@dataclass(frozen=True, slots=True)
class RGBDVideoFrame:
    """One registered color/depth pair and its capture-time calibration."""

    intrinsics: np.ndarray
    color_rgb: np.ndarray
    depth_raw: np.ndarray
    depth_scale: float
    color_frame_number: int
    depth_frame_number: int
    color_timestamp_ms: float
    depth_timestamp_ms: float
    timestamp_domain: str
    received_monotonic_seconds: float
    intrinsics_metadata: dict[str, Any] | None = None


class _GracefulSigint:
    """Turn the CLI's first Ctrl-C into a polled, frame-boundary stop request."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: Any | None = None
        self._installed = False

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        previous = signal.getsignal(signal.SIGINT)
        if previous not in (signal.SIG_DFL, signal.default_int_handler):
            return
        self._previous = previous
        signal.signal(signal.SIGINT, self._request_stop)
        self._installed = True

    def _request_stop(self, _signum: int, _frame: Any) -> None:
        if not self.requested:
            self.requested = True
            print(
                "\nStop requested; finishing the current RGB-D frame...",
                flush=True,
            )

    def restore(self) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous)
            self._installed = False


class L515Camera:
    """A RealSense L515 stream with depth registered to the color image."""

    def __init__(
        self,
        *,
        color_width: int = DEFAULT_COLOR_WIDTH,
        color_height: int = DEFAULT_COLOR_HEIGHT,
        depth_width: int = DEFAULT_DEPTH_WIDTH,
        depth_height: int = DEFAULT_DEPTH_HEIGHT,
        fps: int = DEFAULT_FPS,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
        serial: str | None = None,
        allow_non_l515: bool = False,
    ) -> None:
        for name, value in (
            ("color_width", color_width),
            ("color_height", color_height),
            ("depth_width", depth_width),
            ("depth_height", depth_height),
            ("fps", fps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")

        self.rs = _realsense()
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(
            self.rs.stream.depth,
            depth_width,
            depth_height,
            self.rs.format.z16,
            fps,
        )
        config.enable_stream(
            self.rs.stream.color,
            color_width,
            color_height,
            self.rs.format.rgb8,
            fps,
        )

        self.profile = self.pipeline.start(config)
        self.align = self.rs.align(self.rs.stream.color)
        self._stopped = False
        try:
            self.device = self.profile.get_device()
            self.camera_metadata = self._read_camera_metadata()
            camera_name = self.camera_metadata.get("name", "")
            if not allow_non_l515 and "L515" not in camera_name.upper():
                raise RuntimeError(
                    "expected an Intel RealSense L515, but connected device is "
                    f"{camera_name or 'unknown'}; pass --allow-non-l515 to override"
                )
            self._warm_up(warmup_frames)
        except BaseException:
            self.stop()
            raise

        print(
            f"Camera initialized: {camera_name or 'RealSense'} "
            f"(serial {self.camera_metadata.get('serial_number') or 'unknown'}); "
            f"discarded {warmup_frames} warm-up frames."
        )

    def _device_info(self, key: Any) -> str:
        try:
            return str(self.device.get_info(key)) if self.device.supports(key) else ""
        except RuntimeError:
            return ""

    def _read_camera_metadata(self) -> dict[str, str]:
        info = self.rs.camera_info
        return {
            "name": self._device_info(info.name),
            "serial_number": self._device_info(info.serial_number),
            "firmware_version": self._device_info(info.firmware_version),
            "product_line": self._device_info(info.product_line),
            "usb_type_descriptor": self._device_info(info.usb_type_descriptor),
        }

    def _next_complete_frames(self) -> tuple[Any, Any]:
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if depth_frame and color_frame:
                return color_frame, depth_frame

    def _warm_up(self, frame_count: int) -> None:
        for _ in range(frame_count):
            self._next_complete_frames()

    def discard_for(self, seconds: float) -> None:
        """Keep the streams live while waiting for the performer to start."""

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._next_complete_frames()

    def get_frame(self) -> RGBDVideoFrame:
        color_frame, depth_frame = self._next_complete_frames()
        color_rgb = _color_frame_as_rgb(color_frame)
        depth_raw = np.ascontiguousarray(
            np.asanyarray(depth_frame.get_data()), dtype="<u2"
        )
        if depth_raw.shape != color_rgb.shape[:2]:
            raise ValueError(
                "aligned RGB/depth shape mismatch: "
                f"{color_rgb.shape[:2]} versus {depth_raw.shape}"
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
        intrinsics_metadata = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "distortion_model": str(intr.model),
            "distortion_coefficients": [float(value) for value in intr.coeffs],
        }
        return RGBDVideoFrame(
            intrinsics=intrinsics,
            color_rgb=color_rgb,
            depth_raw=depth_raw,
            depth_scale=float(depth_frame.get_units()),
            color_frame_number=int(color_frame.get_frame_number()),
            depth_frame_number=int(depth_frame.get_frame_number()),
            color_timestamp_ms=float(color_frame.get_timestamp()),
            depth_timestamp_ms=float(depth_frame.get_timestamp()),
            timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
            received_monotonic_seconds=time.monotonic(),
            intrinsics_metadata=intrinsics_metadata,
        )

    def stop(self) -> None:
        if not self._stopped:
            self.pipeline.stop()
            self._stopped = True

    def __enter__(self) -> L515Camera:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class RGBDDemoWriter:
    """Atomically encode a synchronized RGB-D demonstration directory."""

    def __init__(
        self,
        destination: str | Path,
        *,
        calibration_path: str | Path,
        fps: int,
        warmup_frames: int,
        camera_metadata: dict[str, Any] | None = None,
        preview_near_meters: float = DEFAULT_PREVIEW_NEAR_METERS,
        preview_far_meters: float = DEFAULT_PREVIEW_FAR_METERS,
    ) -> None:
        self.destination = Path(destination).expanduser().resolve()
        self.calibration_path = Path(calibration_path).expanduser().resolve()
        if self.destination.exists():
            raise FileExistsError(f"demo directory already exists: {self.destination}")
        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"calibration YAML not found: {self.calibration_path}"
            )
        if fps <= 0:
            raise ValueError("fps must be positive")
        if preview_near_meters < 0.0 or preview_far_meters <= preview_near_meters:
            raise ValueError("depth preview range must satisfy 0 <= near < far")

        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self.camera_metadata = dict(camera_metadata or {})
        self.preview_near_meters = float(preview_near_meters)
        self.preview_far_meters = float(preview_far_meters)
        self.temporary = self.destination.with_name(f".{self.destination.name}.tmp")
        if self.temporary.exists():
            raise FileExistsError(
                f"temporary capture directory already exists: {self.temporary}"
            )

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir()
        self._timestamp_file = (self.temporary / TIMESTAMPS_FILENAME).open(
            "w", encoding="utf-8", newline=""
        )
        self._timestamp_writer = csv.writer(self._timestamp_file)
        self._timestamp_writer.writerow(
            (
                "frame_index",
                "relative_seconds",
                "color_frame_number",
                "depth_frame_number",
                "color_timestamp_ms",
                "depth_timestamp_ms",
                "timestamp_domain",
                "received_monotonic_seconds",
            )
        )
        self._rgb_writer: Any | None = None
        self._depth_writer: Any | None = None
        self._preview_writer: Any | None = None
        self._frame_count = 0
        self._shape: tuple[int, int] | None = None
        self._intrinsics: np.ndarray | None = None
        self._intrinsics_metadata: dict[str, Any] = {}
        self._depth_scale: float | None = None
        self._first_color_timestamp_ms: float | None = None
        self._last_color_timestamp_ms: float | None = None
        self._captured_at = datetime.now(timezone.utc)
        self._finished = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _start_encoders(self, frame: RGBDVideoFrame) -> None:
        height, width = frame.depth_raw.shape
        size = (width, height)
        self._rgb_writer = imageio_ffmpeg.write_frames(
            str(self.temporary / RGB_VIDEO_FILENAME),
            size,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=float(self.fps),
            quality=None,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_log_level="warning",
            output_params=[
                "-crf",
                str(RGB_VIDEO_CRF),
                "-preset",
                RGB_VIDEO_PRESET,
                "-movflags",
                "+faststart",
            ],
        )
        self._depth_writer = imageio_ffmpeg.write_frames(
            str(self.temporary / DEPTH_VIDEO_FILENAME),
            size,
            pix_fmt_in="gray16le",
            pix_fmt_out="gray16le",
            fps=float(self.fps),
            quality=None,
            codec="ffv1",
            macro_block_size=1,
            ffmpeg_log_level="warning",
            output_params=["-level", "3"],
        )
        self._preview_writer = imageio_ffmpeg.write_frames(
            str(self.temporary / DEPTH_PREVIEW_FILENAME),
            size,
            pix_fmt_in="gray",
            pix_fmt_out="yuv420p",
            fps=float(self.fps),
            quality=8,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_log_level="warning",
            output_params=["-movflags", "+faststart"],
        )
        for writer in self._encoders():
            writer.send(None)

    def _encoders(self) -> tuple[Any, ...]:
        return tuple(
            writer
            for writer in (
                self._rgb_writer,
                self._depth_writer,
                self._preview_writer,
            )
            if writer is not None
        )

    def _initialize_from_first_frame(self, frame: RGBDVideoFrame) -> None:
        color, depth, intrinsics = _validated_frame_arrays(frame)
        if not np.any(depth > 0):
            raise ValueError("the first frame contains no positive depth values")
        self._shape = depth.shape
        self._intrinsics = intrinsics.copy()
        self._intrinsics_metadata = dict(frame.intrinsics_metadata or {})
        self._depth_scale = float(frame.depth_scale)
        self._first_color_timestamp_ms = float(frame.color_timestamp_ms)

        Image.fromarray(color, mode="RGB").save(self.temporary / "color.png")
        depth_meters = depth.astype(np.float32) * self._depth_scale
        np.save(self.temporary / "depth.npy", depth_meters, allow_pickle=False)
        Image.fromarray(_depth_visualization(depth_meters), mode="L").save(
            self.temporary / "depth.png"
        )
        (self.temporary / "intrinsics.json").write_text(
            json.dumps(intrinsics.tolist()) + "\n", encoding="utf-8"
        )
        shutil.copy2(
            self.calibration_path,
            self.temporary / CALIBRATION_FILENAME,
        )
        self._start_encoders(frame)

    def write_frame(self, frame: RGBDVideoFrame) -> None:
        if self._finished:
            raise RuntimeError("cannot write to a finalized RGB-D demo")
        color, depth, intrinsics = _validated_frame_arrays(frame)
        if self._frame_count == 0:
            self._initialize_from_first_frame(frame)
        else:
            if depth.shape != self._shape:
                raise ValueError(
                    f"frame shape changed from {self._shape} to {depth.shape}"
                )
            if not np.allclose(intrinsics, self._intrinsics, rtol=0.0, atol=1e-6):
                raise ValueError("camera intrinsics changed during the recording")
            if not np.isclose(frame.depth_scale, self._depth_scale, rtol=1e-6):
                raise ValueError("depth scale changed during the recording")

        assert self._rgb_writer is not None
        assert self._depth_writer is not None
        assert self._preview_writer is not None
        self._rgb_writer.send(color)
        self._depth_writer.send(depth)
        self._preview_writer.send(
            _depth_preview(
                depth,
                float(frame.depth_scale),
                near_meters=self.preview_near_meters,
                far_meters=self.preview_far_meters,
            )
        )

        assert self._first_color_timestamp_ms is not None
        relative_seconds = (
            float(frame.color_timestamp_ms) - self._first_color_timestamp_ms
        ) / 1000.0
        self._timestamp_writer.writerow(
            (
                self._frame_count,
                f"{relative_seconds:.9f}",
                frame.color_frame_number,
                frame.depth_frame_number,
                f"{frame.color_timestamp_ms:.6f}",
                f"{frame.depth_timestamp_ms:.6f}",
                frame.timestamp_domain,
                f"{frame.received_monotonic_seconds:.9f}",
            )
        )
        self._last_color_timestamp_ms = float(frame.color_timestamp_ms)
        self._frame_count += 1

    def _close_outputs(self) -> None:
        errors: list[BaseException] = []
        for writer in self._encoders():
            try:
                writer.close()
            except BaseException as exc:  # close all streams before surfacing one
                errors.append(exc)
        for name in ("_rgb_writer", "_depth_writer", "_preview_writer"):
            setattr(self, name, None)
        if not self._timestamp_file.closed:
            try:
                self._timestamp_file.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def finalize(self, *, stop_reason: str) -> Path:
        if self._finished:
            raise RuntimeError("RGB-D demo writer was already finalized")
        if self._frame_count == 0:
            self.abort()
            raise RuntimeError("no complete RGB-D frames were recorded")

        try:
            self._close_outputs()
            assert self._shape is not None
            assert self._intrinsics is not None
            assert self._depth_scale is not None
            assert self._first_color_timestamp_ms is not None
            assert self._last_color_timestamp_ms is not None
            sensor_duration = max(
                0.0,
                (self._last_color_timestamp_ms - self._first_color_timestamp_ms)
                / 1000.0,
            )
            effective_fps = (
                (self._frame_count - 1) / sensor_duration
                if self._frame_count > 1 and sensor_duration > 0.0
                else None
            )
            metadata = {
                "schema_version": 1,
                "capture_type": "aligned_rgbd_demo",
                "captured_at": self._captured_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "stop_reason": stop_reason,
                "camera": self.camera_metadata,
                "registration": "depth_aligned_to_color",
                "color_shape": [self._shape[0], self._shape[1], 3],
                "depth_shape": list(self._shape),
                "intrinsics": self._intrinsics.tolist(),
                "intrinsics_metadata": self._intrinsics_metadata,
                "depth_scale_meters_per_unit": self._depth_scale,
                "depth_units": "uint16_sensor_units",
                "frame_count": self._frame_count,
                "nominal_fps": self.fps,
                "sensor_duration_seconds": sensor_duration,
                "effective_sensor_fps": effective_fps,
                "warmup_frames_discarded": self.warmup_frames,
                "depth_preview_range_meters": [
                    self.preview_near_meters,
                    self.preview_far_meters,
                ],
                "files": {
                    "rgb_video": RGB_VIDEO_FILENAME,
                    "depth_video": DEPTH_VIDEO_FILENAME,
                    "depth_preview_video": DEPTH_PREVIEW_FILENAME,
                    "timestamps": TIMESTAMPS_FILENAME,
                    "first_color": "color.png",
                    "first_metric_depth": "depth.npy",
                    "intrinsics": "intrinsics.json",
                    "extrinsics": CALIBRATION_FILENAME,
                },
                "video_encoding": {
                    "rgb": {
                        "codec": "h264",
                        "pixel_format": "yuv420p",
                        "rate_control": "crf",
                        "crf": RGB_VIDEO_CRF,
                        "preset": RGB_VIDEO_PRESET,
                    },
                    "depth": {
                        "codec": "ffv1",
                        "pixel_format": "gray16le",
                        "lossless": True,
                    },
                    "depth_preview": {
                        "codec": "h264",
                        "pixel_format": "yuv420p",
                        "metric": False,
                    },
                },
            }
            (self.temporary / CAPTURE_METADATA_FILENAME).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.temporary.rename(self.destination)
            self._finished = True
        except BaseException:
            self.abort()
            raise
        return self.destination

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self._close_outputs()
        except BaseException:
            pass
        shutil.rmtree(self.temporary, ignore_errors=True)
        self._finished = True


def _validated_frame_arrays(
    frame: RGBDVideoFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color = np.ascontiguousarray(frame.color_rgb, dtype=np.uint8)
    depth = np.ascontiguousarray(frame.depth_raw, dtype="<u2")
    intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color_rgb must be HxWx3, got {color.shape}")
    if depth.ndim != 2 or depth.shape != color.shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} does not match color {color.shape[:2]}"
        )
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(
            f"intrinsics must be a finite 3x3 matrix, got {intrinsics.shape}"
        )
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("intrinsics must have positive fx and fy")
    if not np.isfinite(frame.depth_scale) or frame.depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    return color, depth, intrinsics


def _depth_preview(
    depth_raw: np.ndarray,
    depth_scale: float,
    *,
    near_meters: float,
    far_meters: float,
) -> np.ndarray:
    """Map metric depth to a stable 8-bit view, with nearer pixels brighter."""

    depth_meters = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    valid = depth_meters > 0.0
    normalized = 1.0 - np.clip(
        (depth_meters - near_meters) / (far_meters - near_meters),
        0.0,
        1.0,
    )
    preview = np.zeros(depth_meters.shape, dtype=np.uint8)
    preview[valid] = np.asarray(normalized[valid] * 255.0, dtype=np.uint8)
    return preview


def next_demo_name(demos_root: str | Path) -> str:
    """Return ``demo_N`` using one more than the largest existing index."""

    root = Path(demos_root)
    largest = 0
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("demo_"):
                continue
            suffix = child.name.removeprefix("demo_")
            if suffix.isdigit():
                largest = max(largest, int(suffix))
    return f"demo_{largest + 1}"


def resolve_demo_directory(
    demos_root: str | Path,
    demo_name: str | None,
) -> Path:
    root = Path(demos_root).expanduser().resolve()
    name = next_demo_name(root) if demo_name is None else str(demo_name).strip()
    suffix = name.removeprefix("demo_") if name.startswith("demo_") else ""
    if Path(name).name != name or not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError("demo_name must be one directory name such as demo_2")
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"demo directory already exists: {destination}")
    return destination


def write_demo_directory(
    destination: str | Path,
    frames: Iterable[RGBDVideoFrame],
    *,
    calibration_path: str | Path,
    fps: int,
    warmup_frames: int = 0,
    camera_metadata: dict[str, Any] | None = None,
    stop_reason: str = "input_exhausted",
    preview_near_meters: float = DEFAULT_PREVIEW_NEAR_METERS,
    preview_far_meters: float = DEFAULT_PREVIEW_FAR_METERS,
) -> Path:
    """Write an iterable of RGB-D frames; useful for imports and tests."""

    writer = RGBDDemoWriter(
        destination,
        calibration_path=calibration_path,
        fps=fps,
        warmup_frames=warmup_frames,
        camera_metadata=camera_metadata,
        preview_near_meters=preview_near_meters,
        preview_far_meters=preview_far_meters,
    )
    try:
        for frame in frames:
            writer.write_frame(frame)
        return writer.finalize(stop_reason=stop_reason)
    except BaseException:
        writer.abort()
        raise


def capture_demo(
    *,
    demos_root: str | Path = DEFAULT_DEMOS_ROOT,
    demo_name: str | None = None,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    color_width: int = DEFAULT_COLOR_WIDTH,
    color_height: int = DEFAULT_COLOR_HEIGHT,
    depth_width: int = DEFAULT_DEPTH_WIDTH,
    depth_height: int = DEFAULT_DEPTH_HEIGHT,
    fps: int = DEFAULT_FPS,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    start_delay_seconds: float = DEFAULT_START_DELAY_SECONDS,
    duration_seconds: float | None = None,
    serial: str | None = None,
    allow_non_l515: bool = False,
    preview_near_meters: float = DEFAULT_PREVIEW_NEAR_METERS,
    preview_far_meters: float = DEFAULT_PREVIEW_FAR_METERS,
) -> Path:
    """Record from an L515 until Ctrl-C or the requested duration elapses."""

    destination = resolve_demo_directory(demos_root, demo_name)
    calibration = Path(calibration_path).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"calibration YAML not found: {calibration}")

    print(f"Output demo: {destination}")
    print(f"Hand-eye calibration: {calibration}")
    writer: RGBDDemoWriter | None = None
    interrupt = _GracefulSigint()
    stop_reason = (
        "duration_elapsed" if duration_seconds is not None else "keyboard_interrupt"
    )
    try:
        try:
            with L515Camera(
                color_width=color_width,
                color_height=color_height,
                depth_width=depth_width,
                depth_height=depth_height,
                fps=fps,
                warmup_frames=warmup_frames,
                serial=serial,
                allow_non_l515=allow_non_l515,
            ) as camera:
                if start_delay_seconds > 0.0:
                    print(
                        f"Recording starts in {start_delay_seconds:g} seconds; "
                        "prepare the demonstration."
                    )
                    camera.discard_for(start_delay_seconds)

                writer = RGBDDemoWriter(
                    destination,
                    calibration_path=calibration,
                    fps=fps,
                    warmup_frames=warmup_frames,
                    camera_metadata=camera.camera_metadata,
                    preview_near_meters=preview_near_meters,
                    preview_far_meters=preview_far_meters,
                )
                interrupt.install()
                print("Recording RGB-D now. Press Ctrl-C once to stop and save.")
                started = time.monotonic()
                while writer.frame_count == 0 or not interrupt.requested:
                    if (
                        duration_seconds is not None
                        and writer.frame_count > 0
                        and time.monotonic() - started >= duration_seconds
                    ):
                        break
                    writer.write_frame(camera.get_frame())
                    if writer.frame_count % fps == 0:
                        print(
                            f"  recorded {writer.frame_count} frames "
                            f"({writer.frame_count / fps:.1f} s)",
                            flush=True,
                        )
        except KeyboardInterrupt:
            if writer is None or writer.frame_count == 0:
                if writer is not None:
                    writer.abort()
                raise
            stop_reason = "keyboard_interrupt"
            print("\nStopping capture and finalizing videos...", flush=True)
        except BaseException:
            if writer is not None:
                writer.abort()
            raise

        assert writer is not None
        if interrupt.requested:
            stop_reason = "keyboard_interrupt"
        return writer.finalize(stop_reason=stop_reason)
    finally:
        interrupt.restore()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-real2sim-capture demo",
        description=(
            "Record synchronized L515 RGB-D video as "
            "demos/strategy/demo_N."
        )
    )
    parser.add_argument(
        "--demo-name",
        "--name",
        default=None,
        help="Output name such as demo_2; defaults to the next demo_N.",
    )
    parser.add_argument(
        "--demos-root",
        type=Path,
        default=DEFAULT_DEMOS_ROOT,
        help=f"Demo root (default: {DEFAULT_DEMOS_ROOT}).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"Easy Handeye YAML to copy (default: {DEFAULT_CALIBRATION}).",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number.")
    parser.add_argument(
        "--allow-non-l515",
        action="store_true",
        help="Allow a different RealSense model (L515 is required by default).",
    )
    parser.add_argument("--color-width", type=int, default=DEFAULT_COLOR_WIDTH)
    parser.add_argument("--color-height", type=int, default=DEFAULT_COLOR_HEIGHT)
    parser.add_argument("--depth-width", type=int, default=DEFAULT_DEPTH_WIDTH)
    parser.add_argument("--depth-height", type=int, default=DEFAULT_DEPTH_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=(
            "Complete frames to discard before capture "
            f"(default: {DEFAULT_WARMUP_FRAMES})."
        ),
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=DEFAULT_START_DELAY_SECONDS,
        help=(
            "Seconds to keep streaming before recording starts "
            f"(default: {DEFAULT_START_DELAY_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds; otherwise use Ctrl-C.",
    )
    parser.add_argument(
        "--depth-preview-near",
        type=float,
        default=DEFAULT_PREVIEW_NEAR_METERS,
        metavar="METERS",
    )
    parser.add_argument(
        "--depth-preview-far",
        type=float,
        default=DEFAULT_PREVIEW_FAR_METERS,
        metavar="METERS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    positive_values = {
        "--color-width": args.color_width,
        "--color-height": args.color_height,
        "--depth-width": args.depth_width,
        "--depth-height": args.depth_height,
        "--fps": args.fps,
    }
    for flag, value in positive_values.items():
        if value <= 0:
            parser.error(f"{flag} must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.start_delay < 0.0:
        parser.error("--start-delay must be non-negative")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if (
        args.depth_preview_near < 0.0
        or args.depth_preview_far <= args.depth_preview_near
    ):
        parser.error("depth preview range must satisfy 0 <= near < far")

    try:
        demo_dir = capture_demo(
            demos_root=args.demos_root,
            demo_name=args.demo_name,
            calibration_path=args.calibration,
            color_width=args.color_width,
            color_height=args.color_height,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            start_delay_seconds=args.start_delay,
            duration_seconds=args.duration,
            serial=args.serial,
            allow_non_l515=args.allow_non_l515,
            preview_near_meters=args.depth_preview_near,
            preview_far_meters=args.depth_preview_far,
        )
    except KeyboardInterrupt:
        print("\nCapture cancelled before the first frame; nothing was saved.")
        return 130

    print(f"Saved RGB-D demonstration: {demo_dir}")
    print(f"RGB video: {demo_dir / RGB_VIDEO_FILENAME}")
    print(f"Lossless metric-depth video: {demo_dir / DEPTH_VIDEO_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
