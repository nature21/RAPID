#!/usr/bin/env python3
"""Build lean, reconstructable RGB-D segments from an annotation manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image

from vistools.rgbd_format import CALIBRATION_FILENAME, _depth_visualization
from vistools.rgbd_format import (
    CAPTURE_METADATA_FILENAME,
    DEPTH_VIDEO_FILENAME,
    RGB_VIDEO_FILENAME,
    TIMESTAMPS_FILENAME,
)
from vistools.clip_rgbd_demo import (
    _clip_video,
    _load_json_object,
    _load_timestamps,
    _required_shape,
    _source_calibration,
    _write_clipped_timestamps,
    select_timestamp_range,
)
from vistools.rgbd_input import load_rgbd_scene


REQUIRED_COLUMNS = (
    "demo",
    "phase",
    "start_s",
    "end_s",
    "primitive",
    "phase_summary",
    "contact_mode",
    "constraints_strategy",
)
SEGMENT_DIRECTORY_PATTERN = re.compile(r"segment_(\d+)$")
MANIFEST_TIME_TOLERANCE_SECONDS = 0.05


class SegmentError(RuntimeError):
    """An actionable segment-generation failure."""


@dataclass(frozen=True, slots=True)
class Segment:
    phase: int
    start_seconds: float
    end_seconds: float
    primitive: str
    phase_summary: str
    contact_mode: str
    constraints_strategy: str

    @property
    def directory_name(self) -> str:
        return f"segment_{self.phase}"


@dataclass(frozen=True, slots=True)
class VideoProbe:
    frame_count: int
    duration_seconds: float
    average_fps: float


def _run(command: list[str], *, description: str) -> bytes:
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SegmentError(
            f"failed while {description}: {diagnostic or 'no diagnostic'}"
        ) from exc
    return result.stdout


def _probe_video(path: Path) -> VideoProbe:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SegmentError("ffprobe is required to validate segment frame counts")
    payload = json.loads(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets,avg_frame_rate:format=duration",
                "-of",
                "json",
                str(path),
            ],
            description=f"probing {path}",
        )
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise SegmentError(f"no video stream found in {path}")
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate", "0/0"))
    numerator, separator, denominator = rate.partition("/")
    try:
        average_fps = (
            float(numerator) / float(denominator)
            if separator
            else float(numerator)
        )
        frame_count = int(stream["nb_read_packets"])
        duration_seconds = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SegmentError(f"invalid ffprobe metadata for {path}") from exc
    if frame_count <= 0 or duration_seconds <= 0.0 or average_fps <= 0.0:
        raise SegmentError(f"invalid video stream metadata for {path}")
    return VideoProbe(frame_count, duration_seconds, average_fps)


def _decode_frame(path: Path, *, frame_index: int, pixel_format: str) -> bytes:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    frame_filter = (
        f"trim=start_frame={frame_index}:end_frame={frame_index + 1},"
        "setpts=PTS-STARTPTS"
    )
    return _run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            frame_filter,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-",
        ],
        description=f"decoding frame {frame_index} from {path.name}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demo_id(demo_dir: Path) -> str:
    match = re.fullmatch(r"demo_(\d+)", demo_dir.name)
    if match is None:
        raise SegmentError(
            f"demo directory must be named demo_N, found {demo_dir.name!r}"
        )
    return str(int(match.group(1)))


def load_segments(manifest: Path, demo_id: str) -> list[Segment]:
    """Load all annotation rows for one demo."""

    if not manifest.is_file():
        raise SegmentError(f"segment manifest not found: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise SegmentError(
                f"manifest is missing required columns: {', '.join(missing)}"
            )
        rows = [row for row in reader if str(row["demo"]).strip() == demo_id]

    if not rows:
        raise SegmentError(f"manifest has no rows for demo {demo_id}")
    parsed: list[Segment] = []
    for row in rows:
        try:
            segment = Segment(
                phase=int(str(row["phase"]).strip()),
                start_seconds=float(str(row["start_s"]).strip()),
                end_seconds=float(str(row["end_s"]).strip()),
                primitive=str(row["primitive"]).strip(),
                phase_summary=str(row["phase_summary"]).strip(),
                contact_mode=str(row["contact_mode"]).strip(),
                constraints_strategy=str(row["constraints_strategy"]).strip(),
            )
        except (TypeError, ValueError) as exc:
            raise SegmentError(f"invalid segment manifest row: {row}") from exc
        text_fields = (
            segment.primitive,
            segment.phase_summary,
            segment.contact_mode,
            segment.constraints_strategy,
        )
        if not all(text_fields):
            raise SegmentError(f"manifest phase {segment.phase} has an empty text field")
        parsed.append(segment)
    return sorted(parsed, key=lambda item: item.phase)


def resolve_frame_ranges(
    segments: list[Segment],
    rows: list[dict[str, str]],
    *,
    media_duration_seconds: float,
    average_fps: float,
) -> list[tuple[int, int]]:
    """Validate complete, contiguous annotation and map it to recorded frames."""

    phases = [segment.phase for segment in segments]
    if phases != list(range(1, len(segments) + 1)):
        raise SegmentError(
            f"phases must be unique and consecutive from 1, found {phases}"
        )
    if abs(segments[0].start_seconds) > 1e-9:
        raise SegmentError("the first phase must start at exactly 0 seconds")
    for previous, current in zip(segments, segments[1:]):
        if abs(previous.end_seconds - current.start_seconds) > 1e-9:
            raise SegmentError(
                f"phases {previous.phase} and {current.phase} are not contiguous: "
                f"{previous.end_seconds:g} != {current.start_seconds:g}"
            )
    if any(
        not np.isfinite(segment.start_seconds)
        or not np.isfinite(segment.end_seconds)
        or segment.start_seconds < 0.0
        or segment.end_seconds <= segment.start_seconds
        for segment in segments
    ):
        raise SegmentError("every segment interval must be finite, positive, and ordered")

    duration_tolerance = max(
        MANIFEST_TIME_TOLERANCE_SECONDS,
        1.5 / average_fps,
    )
    if abs(segments[-1].end_seconds - media_duration_seconds) > duration_tolerance:
        raise SegmentError(
            f"final end_s {segments[-1].end_seconds:.6f} does not cover the "
            f"{media_duration_seconds:.6f}s source video"
        )

    ranges = [
        select_timestamp_range(
            rows,
            start_seconds=segment.start_seconds,
            end_seconds=(None if index == len(segments) - 1 else segment.end_seconds),
        )
        for index, segment in enumerate(segments)
    ]
    if ranges[0][0] != 0 or ranges[-1][1] != len(rows):
        raise SegmentError("manifest does not cover every recorded frame")
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] != current[0]:
            raise SegmentError(
                "timestamp mapping introduced a gap or overlap between phases"
            )
    return ranges


def _write_segment(
    *,
    demo_dir: Path,
    manifest: Path,
    source_metadata: dict[str, Any],
    source_rows: list[dict[str, str]],
    calibration_path: Path,
    segment: Segment,
    frame_range: tuple[int, int],
    output_dir: Path,
    nominal_fps: float,
    color_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
    depth_scale: float,
) -> None:
    start_frame, end_frame = frame_range
    frame_count = end_frame - start_frame
    output_dir.mkdir()

    rgb_output = output_dir / RGB_VIDEO_FILENAME
    _clip_video(
        demo_dir / RGB_VIDEO_FILENAME,
        rgb_output,
        start_frame=start_frame,
        end_frame=end_frame,
        codec="libx264",
        pixel_format="yuv420p",
        fps=nominal_fps,
    )
    rgb_probe = _probe_video(rgb_output)
    if rgb_probe.frame_count != frame_count:
        raise SegmentError(
            f"{output_dir.name} RGB frame count {rgb_probe.frame_count} does not "
            f"match {frame_count} timestamp rows"
        )

    clipped_rows = _write_clipped_timestamps(
        output_dir / TIMESTAMPS_FILENAME,
        source_rows,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if len(clipped_rows) != rgb_probe.frame_count:
        raise SegmentError(f"{output_dir.name} timestamp count mismatch")

    intrinsics_source = demo_dir / "intrinsics.json"
    shutil.copy2(intrinsics_source, output_dir / "intrinsics.json")
    shutil.copy2(calibration_path, output_dir / CALIBRATION_FILENAME)

    height, width = depth_shape
    color_bytes = _decode_frame(rgb_output, frame_index=0, pixel_format="rgb24")
    expected_color_bytes = height * width * 3
    if len(color_bytes) != expected_color_bytes:
        raise SegmentError(
            f"{output_dir.name} first RGB frame has {len(color_bytes)} bytes, "
            f"expected {expected_color_bytes}"
        )
    color = np.frombuffer(color_bytes, dtype=np.uint8).reshape(color_shape)
    Image.fromarray(color, mode="RGB").save(output_dir / "color.png")

    depth_bytes = _decode_frame(
        demo_dir / DEPTH_VIDEO_FILENAME,
        frame_index=start_frame,
        pixel_format="gray16le",
    )
    expected_depth_bytes = height * width * 2
    if len(depth_bytes) != expected_depth_bytes:
        raise SegmentError(
            f"{output_dir.name} source depth frame has {len(depth_bytes)} bytes, "
            f"expected {expected_depth_bytes}"
        )
    depth_raw = np.frombuffer(depth_bytes, dtype="<u2").reshape(depth_shape)
    depth_meters = depth_raw.astype(np.float32) * depth_scale
    np.save(output_dir / "depth.npy", depth_meters, allow_pickle=False)
    Image.fromarray(_depth_visualization(depth_meters), mode="L").save(
        output_dir / "depth.png"
    )

    first_source_time = float(source_rows[start_frame]["relative_seconds"])
    last_source_time = float(source_rows[end_frame - 1]["relative_seconds"])
    root_start_frame = int(clipped_rows[0]["source_frame_index"])
    root_end_frame = int(clipped_rows[-1]["source_frame_index"]) + 1
    root_start_time = float(clipped_rows[0]["source_relative_seconds"])
    root_last_time = float(clipped_rows[-1]["source_relative_seconds"])
    metadata = {
        "schema_version": 1,
        "capture_type": "aligned_rgbd_demo_segment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registration": source_metadata.get("registration"),
        "world_frame": "fr3_link0",
        "camera": source_metadata.get("camera"),
        "frame_count": frame_count,
        "nominal_fps": nominal_fps,
        "sensor_duration_seconds": max(0.0, last_source_time - first_source_time),
        "color_shape": list(color_shape),
        "depth_shape": list(depth_shape),
        "depth_units": "meters",
        "depth_scale_meters_per_unit": depth_scale,
        "depth_preview_range_meters": source_metadata.get(
            "depth_preview_range_meters"
        ),
        "annotation": {
            "demo": int(_demo_id(demo_dir)),
            "phase": segment.phase,
            "primitive": segment.primitive,
            "phase_summary": segment.phase_summary,
            "contact_mode": segment.contact_mode,
            "constraints_strategy": segment.constraints_strategy,
        },
        "provenance": {
            "manifest": str(manifest),
            "source_demo": str(demo_dir),
            "source_capture_metadata": str(
                demo_dir / CAPTURE_METADATA_FILENAME
            ),
            "requested_time_range_seconds": [
                segment.start_seconds,
                segment.end_seconds,
            ],
            "source_frame_range": [start_frame, end_frame],
            "source_time_range_seconds": [first_source_time, last_source_time],
            "root_source_frame_range": [root_start_frame, root_end_frame],
            "root_source_time_range_seconds": [root_start_time, root_last_time],
            "source_depth_video": str(demo_dir / DEPTH_VIDEO_FILENAME),
            "first_depth_source_frame": start_frame,
        },
        "calibration": {
            "intrinsics_supplied": True,
            "extrinsics_supplied": True,
            "extrinsics_parent_frame": "fr3_link0",
            "intrinsics_sha256": _sha256(intrinsics_source),
            "extrinsics_sha256": _sha256(calibration_path),
        },
        "files": {
            "rgb_video": RGB_VIDEO_FILENAME,
            "timestamps": TIMESTAMPS_FILENAME,
            "first_color": "color.png",
            "first_metric_depth": "depth.npy",
            "depth_preview": "depth.png",
            "intrinsics": "intrinsics.json",
            "extrinsics": CALIBRATION_FILENAME,
        },
    }
    (output_dir / CAPTURE_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_rgbd_scene(output_dir)
    if loaded.depth.shape != depth_shape or not np.array_equal(
        loaded.depth, depth_meters
    ):
        raise SegmentError(f"production RGB-D loader changed {output_dir.name} depth")
    parent_frame = loaded.camera_metadata["calibration"]["parent_frame"]
    if parent_frame != "fr3_link0":
        raise SegmentError(
            f"{output_dir.name} calibration parent is {parent_frame!r}, not fr3_link0"
        )
    expected_depth = depth_raw.astype(np.float32) * depth_scale
    if not np.array_equal(np.load(output_dir / "depth.npy"), expected_depth):
        raise SegmentError(
            f"{output_dir.name} metric depth does not match its first source frame"
        )


def _publish_segments(
    demo_dir: Path,
    staging_root: Path,
    expected_names: set[str],
    *,
    force: bool,
) -> None:
    existing = sorted(
        path
        for path in demo_dir.iterdir()
        if path.is_dir() and SEGMENT_DIRECTORY_PATTERN.fullmatch(path.name)
    )
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SegmentError(
            f"segment directories already exist in {demo_dir}: {names}; pass --force"
        )

    backup_root = Path(
        tempfile.mkdtemp(prefix=".segments-backup-", dir=demo_dir)
    )
    published: list[Path] = []
    try:
        for path in existing:
            os.replace(path, backup_root / path.name)
        for name in sorted(expected_names, key=lambda value: int(value.split("_")[1])):
            destination = demo_dir / name
            os.replace(staging_root / name, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            if destination.exists():
                os.replace(destination, staging_root / destination.name)
        for backup in backup_root.iterdir():
            os.replace(backup, demo_dir / backup.name)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def segment_demo(demo_dir: Path, manifest: Path, *, force: bool = False) -> list[Path]:
    """Generate and atomically publish all annotated segments for one demo."""

    demo_dir = demo_dir.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    demo_id = _demo_id(demo_dir)
    if not demo_dir.is_dir():
        raise SegmentError(f"demo directory not found: {demo_dir}")

    source_metadata = _load_json_object(demo_dir / CAPTURE_METADATA_FILENAME)
    source_rows = _load_timestamps(demo_dir / TIMESTAMPS_FILENAME)
    rgb_source = demo_dir / RGB_VIDEO_FILENAME
    depth_source = demo_dir / DEPTH_VIDEO_FILENAME
    intrinsics_source = demo_dir / "intrinsics.json"
    for path in (rgb_source, depth_source, intrinsics_source):
        if not path.is_file():
            raise SegmentError(f"required source file not found: {path}")
    calibration_path = _source_calibration(demo_dir, source_metadata)

    declared_frame_count = source_metadata.get("frame_count")
    if declared_frame_count != len(source_rows):
        raise SegmentError(
            f"capture metadata declares {declared_frame_count!r} frames but "
            f"timestamps contain {len(source_rows)}"
        )
    rgb_probe = _probe_video(rgb_source)
    if rgb_probe.frame_count != len(source_rows):
        raise SegmentError(
            f"source RGB has {rgb_probe.frame_count} frames but timestamps contain "
            f"{len(source_rows)} rows"
        )
    depth_probe = _probe_video(depth_source)
    if depth_probe.frame_count != len(source_rows):
        raise SegmentError(
            f"source depth has {depth_probe.frame_count} frames but timestamps "
            f"contain {len(source_rows)} rows"
        )

    color_shape = _required_shape(source_metadata, "color_shape", 3)
    depth_shape = _required_shape(source_metadata, "depth_shape", 2)
    if color_shape != (*depth_shape, 3):
        raise SegmentError(
            f"source RGB/depth shape mismatch: {color_shape} versus {depth_shape}"
        )
    try:
        nominal_fps = float(source_metadata["nominal_fps"])
        depth_scale = float(source_metadata["depth_scale_meters_per_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SegmentError("source metadata has invalid FPS or depth scale") from exc
    if nominal_fps <= 0.0 or depth_scale <= 0.0:
        raise SegmentError("source FPS and depth scale must be positive")

    segments = load_segments(manifest, demo_id)
    frame_ranges = resolve_frame_ranges(
        segments,
        source_rows,
        media_duration_seconds=rgb_probe.duration_seconds,
        average_fps=rgb_probe.average_fps,
    )
    staging_root = Path(tempfile.mkdtemp(prefix=".segments-build-", dir=demo_dir))
    expected_names = {segment.directory_name for segment in segments}
    try:
        for segment, frame_range in zip(segments, frame_ranges):
            _write_segment(
                demo_dir=demo_dir,
                manifest=manifest,
                source_metadata=source_metadata,
                source_rows=source_rows,
                calibration_path=calibration_path,
                segment=segment,
                frame_range=frame_range,
                output_dir=staging_root / segment.directory_name,
                nominal_fps=nominal_fps,
                color_shape=color_shape,
                depth_shape=depth_shape,
                depth_scale=depth_scale,
            )
        _publish_segments(
            demo_dir,
            staging_root,
            expected_names,
            force=force,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return [demo_dir / segment.directory_name for segment in segments]


def _demo_directories(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SegmentError(f"RGB-D demo root not found: {root}")
    directories = [
        path
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"demo_\d+", path.name)
    ]
    return sorted(directories, key=lambda path: int(path.name.split("_")[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate lean, timestamp-synchronized RGB-D segment directories from "
            "segments.csv annotations."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo-dir",
        type=Path,
        action="append",
        help="A demo_N directory; repeat to process multiple demos.",
    )
    source.add_argument(
        "--root",
        type=Path,
        help="Process every demo_N directory immediately below this root.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing segment_N directories after new outputs validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        demo_dirs = args.demo_dir or _demo_directories(args.root)
        if not demo_dirs:
            raise SegmentError("no demo_N directories found")
        for demo_dir in demo_dirs:
            outputs = segment_demo(demo_dir, args.manifest, force=args.force)
            print(f"Generated {len(outputs)} validated segments for {demo_dir}")
    except (FileExistsError, FileNotFoundError, SegmentError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
