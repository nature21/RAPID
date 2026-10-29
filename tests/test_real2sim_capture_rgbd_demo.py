"""Tests for synchronized RGB-D demonstration capture output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from real2sim.capture_rgbd_demo import (
    CAPTURE_METADATA_FILENAME,
    DEFAULT_DEMOS_ROOT,
    DEPTH_PREVIEW_FILENAME,
    DEPTH_VIDEO_FILENAME,
    RGBDVideoFrame,
    RGB_VIDEO_FILENAME,
    RGB_VIDEO_CRF,
    RGB_VIDEO_PRESET,
    TIMESTAMPS_FILENAME,
    next_demo_name,
    resolve_demo_directory,
    write_demo_directory,
)
from vistools.rgbd_input import is_rgbd_scene_directory, load_rgbd_scene


def _write_calibration(path: Path) -> None:
    path.write_text(
        """\
parameters:
  eye_on_hand: false
  robot_base_frame: fr3_link0
  robot_effector_frame: fr3_hand_tcp
  tracking_base_frame: camera_color_frame
  tracking_marker_frame: marker_0
transformation:
  x: 1.0
  y: 2.0
  z: 3.0
  qx: 0.0
  qy: 0.0
  qz: 0.0
  qw: 1.0
""",
        encoding="utf-8",
    )


def _frame(index: int, *, shape: tuple[int, int] = (4, 4)) -> RGBDVideoFrame:
    height, width = shape
    color = np.empty((height, width, 3), dtype=np.uint8)
    color[...] = [10 + index, 20 + index, 30 + index]
    depth = np.arange(height * width, dtype=np.uint16).reshape(shape) + 1000 + index
    return RGBDVideoFrame(
        intrinsics=np.asarray(
            [[100.0, 0.0, width / 2], [0.0, 110.0, height / 2], [0.0, 0.0, 1.0]]
        ),
        color_rgb=color,
        depth_raw=depth,
        depth_scale=0.001,
        color_frame_number=20 + index,
        depth_frame_number=30 + index,
        color_timestamp_ms=1000.0 + index * 100.0,
        depth_timestamp_ms=1000.5 + index * 100.0,
        timestamp_domain="hardware_clock",
        received_monotonic_seconds=500.0 + index * 0.1,
        intrinsics_metadata={"distortion_model": "none"},
    )


class RGBDDemoCaptureTest(unittest.TestCase):
    def test_default_demo_root_has_the_expected_name(self) -> None:
        self.assertEqual(DEFAULT_DEMOS_ROOT.parts[-2:], ("demos", "strategy"))

    def test_uses_next_numeric_demo_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo_1").mkdir()
            (root / "demo_004").mkdir()
            (root / "demo_notes").mkdir()

            self.assertEqual(next_demo_name(root), "demo_5")
            self.assertEqual(
                resolve_demo_directory(root, None),
                root.resolve() / "demo_5",
            )
            with self.assertRaisesRegex(ValueError, "demo_name"):
                resolve_demo_directory(root, "demo_notes")

    def test_writes_synchronized_lossless_demo_and_first_scene_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.yaml"
            _write_calibration(calibration)
            frames = [_frame(0), _frame(1), _frame(2)]
            destination = root / "demos" / "demo_1"

            written = write_demo_directory(
                destination,
                frames,
                calibration_path=calibration,
                fps=10,
                warmup_frames=7,
                camera_metadata={"name": "Intel RealSense L515", "serial": "abc"},
            )

            self.assertEqual(written, destination.resolve())
            self.assertTrue((written / RGB_VIDEO_FILENAME).is_file())
            self.assertTrue((written / DEPTH_VIDEO_FILENAME).is_file())
            self.assertTrue((written / DEPTH_PREVIEW_FILENAME).is_file())
            self.assertTrue(is_rgbd_scene_directory(written))
            self.assertEqual(
                (written / "panda_eob_calib.yaml").read_text(encoding="utf-8"),
                calibration.read_text(encoding="utf-8"),
            )

            scene = load_rgbd_scene(written)
            np.testing.assert_allclose(
                scene.depth,
                frames[0].depth_raw.astype(np.float32) * frames[0].depth_scale,
            )
            np.testing.assert_allclose(
                scene.camera_metadata["intrinsics"], frames[0].intrinsics
            )

            metadata = json.loads(
                (written / CAPTURE_METADATA_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["nominal_fps"], 10)
            self.assertEqual(metadata["warmup_frames_discarded"], 7)
            self.assertEqual(metadata["depth_scale_meters_per_unit"], 0.001)
            self.assertEqual(
                metadata["video_encoding"]["rgb"],
                {
                    "codec": "h264",
                    "crf": RGB_VIDEO_CRF,
                    "pixel_format": "yuv420p",
                    "preset": RGB_VIDEO_PRESET,
                    "rate_control": "crf",
                },
            )
            self.assertTrue(metadata["video_encoding"]["depth"]["lossless"])

            with (written / TIMESTAMPS_FILENAME).open(
                encoding="utf-8", newline=""
            ) as timestamps_file:
                rows = list(csv.DictReader(timestamps_file))
            self.assertEqual([row["frame_index"] for row in rows], ["0", "1", "2"])
            self.assertEqual(rows[2]["relative_seconds"], "0.200000000")

            decoded = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-i",
                    str(written / DEPTH_VIDEO_FILENAME),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray16le",
                    "-",
                ],
                check=True,
                capture_output=True,
            )
            decoded_frames = [
                frame.reshape(4, 4)
                for frame in np.frombuffer(decoded.stdout, dtype="<u2").reshape(
                    -1, 16
                )
            ]
            self.assertEqual(len(decoded_frames), 3)
            np.testing.assert_array_equal(decoded_frames[0], frames[0].depth_raw)

    def test_removes_partial_directory_when_a_later_frame_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.yaml"
            _write_calibration(calibration)
            destination = root / "demo_1"

            with self.assertRaisesRegex(ValueError, "shape changed"):
                write_demo_directory(
                    destination,
                    [_frame(0), _frame(1, shape=(6, 4))],
                    calibration_path=calibration,
                    fps=10,
                )

            self.assertFalse(destination.exists())
            self.assertFalse((root / ".demo_1.tmp").exists())


if __name__ == "__main__":
    unittest.main()
