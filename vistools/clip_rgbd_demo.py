#!/usr/bin/env python3
"""Trim synchronized RGB-D demonstrations using their recorded timestamps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image

from vistools.rgbd_format import CALIBRATION_FILENAME, _depth_visualization
from vistools.rgbd_format import (
    CAPTURE_METADATA_FILENAME,
    DEPTH_PREVIEW_FILENAME,
    DEPTH_VIDEO_FILENAME,
    RGB_VIDEO_FILENAME,
    TIMESTAMPS_FILENAME,
)


TIMESTAMP_FIELDS = (
    "frame_index",
    "relative_seconds",
    "color_frame_number",
    "depth_frame_number",
    "color_timestamp_ms",
    "depth_timestamp_ms",
    "timestamp_domain",
    "received_monotonic_seconds",
)
CLIP_TIMESTAMP_FIELDS = TIMESTAMP_FIELDS + (
    "source_frame_index",
    "source_relative_seconds",
)


def resolve_trim_directory(
    source: str | Path,
    output: str | Path | None = None,
) -> Path:
    """Resolve a non-destructive sibling ``demo_N_trimmed`` directory."""

    source_path = Path(source).expanduser().resolve()
    destination = (
        source_path.with_name(f"{source_path.name}_trimmed")
        if output is None
        else Path(output).expanduser().resolve()
    )
    if destination == source_path:
        raise ValueError("output must differ from the source demo directory")
    if destination.exists():
        raise FileExistsError(f"trimmed demo directory already exists: {destination}")
    return destination


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required RGB-D demo file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_timestamps(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required RGB-D demo file not found: {path}")
    with path.open(encoding="utf-8", newline="") as timestamp_file:
        reader = csv.DictReader(timestamp_file)
        missing = [
            field
            for field in TIMESTAMP_FIELDS
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{path} is missing timestamp columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no frame timestamps")

    previous_time = -np.inf
    for expected_index, row in enumerate(rows):
        try:
            frame_index = int(row["frame_index"])
            relative_seconds = float(row["relative_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} contains an invalid timestamp row") from exc
        if frame_index != expected_index:
            raise ValueError(
                f"{path} frame indices must be contiguous from zero; "
                f"expected {expected_index}, got {frame_index}"
            )
        if not np.isfinite(relative_seconds) or relative_seconds < previous_time:
            raise ValueError(f"{path} relative timestamps must be finite and ordered")
        previous_time = relative_seconds
    return rows


def select_timestamp_range(
    rows: list[dict[str, str]],
    *,
    start_seconds: float,
    end_seconds: float | None,
) -> tuple[int, int]:
    """Map a half-open time range to a half-open recorded frame range."""

    if not np.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError("start_seconds must be finite and non-negative")
    if end_seconds is not None:
        if not np.isfinite(end_seconds) or end_seconds <= start_seconds:
            raise ValueError(
                "end_seconds must be finite and greater than start_seconds"
            )

    relative_times = np.asarray(
        [float(row["relative_seconds"]) for row in rows], dtype=np.float64
    )
    start_frame = int(np.searchsorted(relative_times, start_seconds, side="left"))
    end_frame = (
        len(rows)
        if end_seconds is None
        else int(np.searchsorted(relative_times, end_seconds, side="left"))
    )
    if start_frame >= len(rows):
        raise ValueError(
            f"start time {start_seconds:g}s is beyond the final recorded frame "
            f"at {relative_times[-1]:g}s"
        )
    if end_frame <= start_frame:
        raise ValueError("the requested time range contains no recorded frames")
    return start_frame, end_frame


def _run_ffmpeg(command: list[str], *, description: str) -> bytes:
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed while {description}: {diagnostic or 'no diagnostic'}"
        ) from exc
    return result.stdout


def _clip_video(
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    codec: str,
    pixel_format: str,
    fps: float,
) -> None:
    frame_filter = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS"
    )
    output_options = [
        "-r",
        f"{fps:.12g}",
        "-c:v",
        codec,
        "-pix_fmt",
        pixel_format,
    ]
    if codec == "libx264":
        output_options.extend(["-crf", "10", "-movflags", "+faststart"])
    elif codec == "ffv1":
        output_options.extend(["-level", "3"])
    _run_ffmpeg(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            frame_filter,
            "-an",
            *output_options,
            str(destination),
        ],
        description=f"clipping {source.name}",
    )


def _decode_first_frame(path: Path, *, pixel_format: str) -> bytes:
    return _run_ffmpeg(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-",
        ],
        description=f"decoding the first frame of {path.name}",
    )


def _required_shape(
    metadata: dict[str, Any], key: str, dimensions: int
) -> tuple[int, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != dimensions:
        raise ValueError(
            f"capture metadata must contain {key} with {dimensions} values"
        )
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture metadata has an invalid {key}") from exc
    if any(item <= 0 for item in shape):
        raise ValueError(f"capture metadata {key} values must be positive")
    return shape


def _write_clipped_timestamps(
    path: Path,
    rows: list[dict[str, str]],
    *,
    start_frame: int,
    end_frame: int,
) -> list[dict[str, str]]:
    selected = rows[start_frame:end_frame]
    first_time = float(selected[0]["relative_seconds"])
    clipped_rows: list[dict[str, str]] = []
    with path.open("w", encoding="utf-8", newline="") as timestamp_file:
        writer = csv.DictWriter(timestamp_file, fieldnames=CLIP_TIMESTAMP_FIELDS)
        writer.writeheader()
        for new_index, source_row in enumerate(selected):
            source_time = float(source_row["relative_seconds"])
            root_frame_index = source_row.get("source_frame_index") or source_row[
                "frame_index"
            ]
            root_relative_seconds = source_row.get(
                "source_relative_seconds"
            ) or source_row["relative_seconds"]
            clipped_row = {
                field: source_row[field] for field in TIMESTAMP_FIELDS
            }
            clipped_row.update(
                {
                    "frame_index": str(new_index),
                    "relative_seconds": f"{source_time - first_time:.9f}",
                    "source_frame_index": root_frame_index,
                    "source_relative_seconds": root_relative_seconds,
                }
            )
            writer.writerow(clipped_row)
            clipped_rows.append(clipped_row)
    return clipped_rows


def _source_calibration(source: Path, metadata: dict[str, Any]) -> Path:
    files = metadata.get("files")
    filename = files.get("extrinsics") if isinstance(files, dict) else None
    candidate = source / (str(filename) if filename else CALIBRATION_FILENAME)
    if not candidate.is_file():
        raise FileNotFoundError(f"RGB-D demo calibration not found: {candidate}")
    return candidate


def trim_demo(
    source: str | Path,
    destination: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Path:
    """Remove redundant head/tail content into a synchronized demo directory."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"RGB-D demo directory not found: {source_path}")
    if destination_path.exists():
        raise FileExistsError(
            f"trimmed demo directory already exists: {destination_path}"
        )

    metadata = _load_json_object(source_path / CAPTURE_METADATA_FILENAME)
    rows = _load_timestamps(source_path / TIMESTAMPS_FILENAME)
    declared_frame_count = metadata.get("frame_count")
    if declared_frame_count != len(rows):
        raise ValueError(
            "capture metadata/timestamp frame-count mismatch: "
            f"{declared_frame_count!r} versus {len(rows)}"
        )
    start_frame, end_frame = select_timestamp_range(
        rows,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    required_videos = {
        RGB_VIDEO_FILENAME: ("libx264", "yuv420p"),
        DEPTH_VIDEO_FILENAME: ("ffv1", "gray16le"),
        DEPTH_PREVIEW_FILENAME: ("libx264", "yuv420p"),
    }
    for filename in required_videos:
        if not (source_path / filename).is_file():
            raise FileNotFoundError(
                f"required RGB-D demo video not found: {source_path / filename}"
            )
    intrinsics_path = source_path / "intrinsics.json"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"RGB-D demo intrinsics not found: {intrinsics_path}")
    calibration_path = _source_calibration(source_path, metadata)

    color_shape = _required_shape(metadata, "color_shape", 3)
    depth_shape = _required_shape(metadata, "depth_shape", 2)
    if color_shape[:2] != depth_shape or color_shape[2] != 3:
        raise ValueError(
            f"capture metadata RGB/depth shape mismatch: {color_shape}, {depth_shape}"
        )
    try:
        depth_scale = float(metadata["depth_scale_meters_per_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("capture metadata has an invalid depth scale") from exc
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("capture metadata depth scale must be finite and positive")
    try:
        nominal_fps = float(metadata["nominal_fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("capture metadata has an invalid nominal frame rate") from exc
    if not np.isfinite(nominal_fps) or nominal_fps <= 0.0:
        raise ValueError(
            "capture metadata nominal frame rate must be finite and positive"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary trimmed demo directory already exists: {temporary}"
        )
    temporary.mkdir()
    try:
        for filename, (codec, pixel_format) in required_videos.items():
            _clip_video(
                source_path / filename,
                temporary / filename,
                start_frame=start_frame,
                end_frame=end_frame,
                codec=codec,
                pixel_format=pixel_format,
                fps=nominal_fps,
            )

        clipped_rows = _write_clipped_timestamps(
            temporary / TIMESTAMPS_FILENAME,
            rows,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        shutil.copy2(intrinsics_path, temporary / "intrinsics.json")
        shutil.copy2(calibration_path, temporary / CALIBRATION_FILENAME)

        height, width = depth_shape
        color_bytes = _decode_first_frame(
            temporary / RGB_VIDEO_FILENAME, pixel_format="rgb24"
        )
        expected_color_bytes = height * width * 3
        if len(color_bytes) != expected_color_bytes:
            raise ValueError(
                "decoded first RGB frame has an unexpected size: "
                f"{len(color_bytes)} versus {expected_color_bytes} bytes"
            )
        color_rgb = np.frombuffer(color_bytes, dtype=np.uint8).reshape(
            height, width, 3
        )
        Image.fromarray(color_rgb, mode="RGB").save(temporary / "color.png")

        depth_bytes = _decode_first_frame(
            temporary / DEPTH_VIDEO_FILENAME, pixel_format="gray16le"
        )
        expected_depth_bytes = height * width * 2
        if len(depth_bytes) != expected_depth_bytes:
            raise ValueError(
                "decoded first depth frame has an unexpected size: "
                f"{len(depth_bytes)} versus {expected_depth_bytes} bytes"
            )
        depth_raw = np.frombuffer(depth_bytes, dtype="<u2").reshape(height, width)
        depth_meters = depth_raw.astype(np.float32) * depth_scale
        np.save(temporary / "depth.npy", depth_meters, allow_pickle=False)
        Image.fromarray(_depth_visualization(depth_meters), mode="L").save(
            temporary / "depth.png"
        )

        first_source_time = float(rows[start_frame]["relative_seconds"])
        last_source_time = float(rows[end_frame - 1]["relative_seconds"])
        sensor_duration = max(0.0, last_source_time - first_source_time)
        frame_count = end_frame - start_frame
        clipped_metadata = dict(metadata)
        clipped_metadata.update(
            {
                "capture_type": "aligned_rgbd_demo_trimmed",
                "trimmed_at": datetime.now(timezone.utc).isoformat(),
                "source_stop_reason": metadata.get("stop_reason"),
                "stop_reason": "trimmed_range",
                "frame_count": frame_count,
                "sensor_duration_seconds": sensor_duration,
                "effective_sensor_fps": (
                    (frame_count - 1) / sensor_duration
                    if frame_count > 1 and sensor_duration > 0.0
                    else None
                ),
                "source_demo": str(source_path),
                "source_frame_range": [start_frame, end_frame],
                "source_time_range_seconds": [
                    first_source_time,
                    last_source_time,
                ],
                "requested_time_range_seconds": [
                    float(start_seconds),
                    None if end_seconds is None else float(end_seconds),
                ],
                "root_source_frame_range": [
                    int(clipped_rows[0]["source_frame_index"]),
                    int(clipped_rows[-1]["source_frame_index"]) + 1,
                ],
            }
        )
        (temporary / CAPTURE_METADATA_FILENAME).write_text(
            json.dumps(clipped_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-rgbd-demo-clip",
        description=(
            "Remove redundant beginning/end content from synchronized RGB, "
            "lossless depth, depth preview, and timestamps."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source directory such as demos/strategy/demo_1.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Inclusive start time in source timestamps (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Exclusive end time; omit to clip through the final frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to a sibling <source>_trimmed directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        destination = resolve_trim_directory(args.source, args.output)
        written = trim_demo(
            args.source,
            destination,
            start_seconds=args.start,
            end_seconds=args.end,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved trimmed synchronized RGB-D demo: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
