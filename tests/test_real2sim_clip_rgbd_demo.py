"""Tests for synchronized RGB-D demonstration clipping."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from real2sim.capture_rgbd_demo import RGBDVideoFrame, write_demo_directory
from vistools.clip_rgbd_demo import resolve_trim_directory, trim_demo
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


def _frame(index: int) -> RGBDVideoFrame:
    color = np.full((4, 4, 3), [20 + index, 40, 60], dtype=np.uint8)
    depth = np.arange(16, dtype=np.uint16).reshape(4, 4) + 1000 + index * 100
    return RGBDVideoFrame(
        intrinsics=np.asarray(
            [[100.0, 0.0, 2.0], [0.0, 110.0, 2.0], [0.0, 0.0, 1.0]]
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
    )


class RGBDDemoClipTest(unittest.TestCase):
    def test_resolves_a_non_destructive_trimmed_sibling(self) -> None:
        source = Path("/tmp/demos/strategy/demo_2")
        self.assertEqual(
            resolve_trim_directory(source),
            Path("/tmp/demos/strategy/demo_2_trimmed"),
        )

    def test_clips_all_streams_and_rebases_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.yaml"
            _write_calibration(calibration)
            frames = [_frame(index) for index in range(6)]
            source = write_demo_directory(
                root / "demo_1",
                frames,
                calibration_path=calibration,
                fps=10,
            )
            destination = root / "demo_1_trimmed"

            written = trim_demo(
                source,
                destination,
                start_seconds=0.15,
                end_seconds=0.45,
            )

            self.assertEqual(written, destination.resolve())
            for filename in ("rgb.mp4", "depth.mkv", "depth_preview.mp4"):
                self.assertTrue((written / filename).is_file())
            self.assertTrue(is_rgbd_scene_directory(written))

            for filename, pixel_format, bytes_per_pixel in (
                ("rgb.mp4", "rgb24", 3),
                ("depth_preview.mp4", "gray", 1),
            ):
                decoded_video = subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-i",
                        str(written / filename),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        pixel_format,
                        "-",
                    ],
                    check=True,
                    capture_output=True,
                )
                self.assertEqual(
                    len(decoded_video.stdout),
                    3 * 4 * 4 * bytes_per_pixel,
                )

            with (written / "timestamps.csv").open(
                encoding="utf-8", newline=""
            ) as timestamps_file:
                rows = list(csv.DictReader(timestamps_file))
            self.assertEqual([row["frame_index"] for row in rows], ["0", "1", "2"])
            self.assertEqual(
                [row["source_frame_index"] for row in rows], ["2", "3", "4"]
            )
            self.assertEqual(
                [row["relative_seconds"] for row in rows],
                ["0.000000000", "0.100000000", "0.200000000"],
            )

            metadata = json.loads(
                (written / "capture_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["capture_type"], "aligned_rgbd_demo_trimmed")
            self.assertEqual(metadata["stop_reason"], "trimmed_range")
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["source_frame_range"], [2, 5])
            self.assertEqual(metadata["root_source_frame_range"], [2, 5])

            decoded = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-i",
                    str(written / "depth.mkv"),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray16le",
                    "-",
                ],
                check=True,
                capture_output=True,
            )
            decoded_depth = np.frombuffer(decoded.stdout, dtype="<u2").reshape(
                3, 4, 4
            )
            np.testing.assert_array_equal(
                decoded_depth,
                np.stack([frames[index].depth_raw for index in (2, 3, 4)]),
            )

            scene = load_rgbd_scene(written)
            np.testing.assert_allclose(
                scene.depth,
                frames[2].depth_raw.astype(np.float32) * frames[2].depth_scale,
            )
            self.assertEqual(
                (written / "panda_eob_calib.yaml").read_text(encoding="utf-8"),
                calibration.read_text(encoding="utf-8"),
            )

    def test_rejects_an_empty_range_without_leaving_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.yaml"
            _write_calibration(calibration)
            source = write_demo_directory(
                root / "demo_1",
                [_frame(index) for index in range(3)],
                calibration_path=calibration,
                fps=10,
            )
            destination = root / "demo_1_trimmed"

            with self.assertRaisesRegex(ValueError, "contains no recorded frames"):
                trim_demo(
                    source,
                    destination,
                    start_seconds=0.01,
                    end_seconds=0.02,
                )

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
