"""Tests for the RealSense-to-Real2Sim capture utility."""

from __future__ import annotations

import io
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from real2sim.capture_rgbd import (
    CALIBRATION_FILENAME,
    CameraClient,
    CameraObservation,
    DEFAULT_CALIBRATION,
    capture_scene,
    main,
    next_scene_name,
    resolve_scene_directory,
    write_scene_directory,
)
from vistools.rgbd_input import load_rgbd_scene


class Real2SimCaptureTest(unittest.TestCase):
    def test_uses_the_shared_repository_calibration_by_default(self) -> None:
        self.assertEqual(
            DEFAULT_CALIBRATION.parts[-2:],
            ("calibration", "panda_eob_calib.yaml"),
        )
        self.assertTrue(DEFAULT_CALIBRATION.is_file())

    def test_requires_an_explicit_capture_mode(self) -> None:
        with (
            mock.patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--scene-name", "scene_2"])
        self.assertEqual(raised.exception.code, 2)

    def test_dispatches_scene_mode(self) -> None:
        destination = Path("/tmp/scene_2")
        for reconstruction_status in (0, 2):
            with (
                self.subTest(reconstruction_status=reconstruction_status),
                mock.patch(
                    "real2sim.capture_rgbd.capture_scene", return_value=destination
                ) as capture_scene,
                mock.patch(
                    "real2sim.workflow.main", return_value=reconstruction_status
                ) as reconstruct_main,
            ):
                result = main(
                    ["scene", "--scene-name", "scene_2", "--warmup-frames", "4"]
                )

                self.assertEqual(result, reconstruction_status)
                capture_scene.assert_called_once_with(
                    scenes_root=mock.ANY,
                    scene_name="scene_2",
                    calibration_path=mock.ANY,
                    warmup_frames=4,
                    depth_frames=10,
                )
                reconstruct_main.assert_called_once_with(
                    [str(destination), "--name", "scene_2"]
                )

    def test_depth_frame_count_cli(self) -> None:
        with (
            mock.patch("real2sim.capture_rgbd.capture_scene", return_value=Path("/tmp/scene_2")) as capture,
            mock.patch("real2sim.workflow.main", return_value=0),
        ):
            main(["scene", "--depth-frames", "3"])
            self.assertEqual(capture.call_args.kwargs["depth_frames"], 3)
            capture.reset_mock()
            for count in ("0", "-1"):
                with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    main(["scene", "--depth-frames", count])
            capture.assert_not_called()

    def test_dispatches_demo_mode(self) -> None:
        with mock.patch("real2sim.capture_rgbd_demo.main", return_value=7) as demo_main:
            result = main(["demo", "--duration", "2"])

        self.assertEqual(result, 7)
        demo_main.assert_called_once_with(["--duration", "2"])

    def test_uses_next_numeric_scene_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene_1").mkdir()
            (root / "scene_003").mkdir()
            (root / "scene_notes").mkdir()

            self.assertEqual(next_scene_name(root), "scene_4")
            self.assertEqual(
                resolve_scene_directory(root, None),
                root.resolve() / "scene_4",
            )

    def test_writes_complete_loadable_rgbd_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "shared.yaml"
            calibration.write_text(
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
            observation = CameraObservation(
                intrinsics=np.asarray(
                    [[100.0, 0.0, 2.0], [0.0, 110.0, 1.5], [0.0, 0.0, 1.0]]
                ),
                color_rgb=np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8),
                depth_meters=np.full((3, 4), 0.75, dtype=np.float32),
                depth_scale=0.001,
                frame_number=42,
                timestamp_ms=1234.5,
                camera_serial="test-camera",
            )
            destination = root / "scenes" / "scene_1"

            written = write_scene_directory(
                destination,
                observation,
                calibration,
                warmup_frames=30,
            )
            loaded = load_rgbd_scene(written)

            self.assertEqual(written, destination.resolve())
            self.assertEqual(loaded.depth.shape, (3, 4))
            np.testing.assert_allclose(loaded.depth, 0.75)
            np.testing.assert_allclose(
                loaded.camera_metadata["intrinsics"], observation.intrinsics
            )
            self.assertTrue((written / "color.png").is_file())
            self.assertTrue((written / "depth.png").is_file())
            self.assertEqual(
                (written / CALIBRATION_FILENAME).read_text(encoding="utf-8"),
                calibration.read_text(encoding="utf-8"),
            )
            capture_metadata = json.loads(
                (written / "capture_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_metadata["warmup_frames_discarded"], 30)
            self.assertEqual(capture_metadata["frame_number"], 42)
            self.assertEqual(capture_metadata["camera_serial"], "test-camera")


class DepthAveragingTest(unittest.TestCase):
    def _observation(self, depth: list[list[float]]) -> CameraObservation:
        array = np.asarray(depth, dtype=np.float32)
        return CameraObservation(
            intrinsics=np.eye(3),
            color_rgb=np.full((*array.shape, 3), 10, dtype=np.uint8),
            depth_meters=array,
            depth_scale=0.001,
            frame_number=42,
            timestamp_ms=1234.5,
            camera_serial="test-camera",
        )

    def test_per_pixel_valid_mean_and_first_rgb(self) -> None:
        first = self._observation([[1, 0, np.nan], [0, 2, np.inf]])
        second = self._observation([[3, 4, 0], [-1, np.nan, -np.inf]])
        third = self._observation([[0, 8, 3], [np.nan, 10, 0]])

        def observations():
            yield first
            # Model a native camera buffer being reused on subsequent frames.
            first.color_rgb[:] = 20
            yield second
            yield third

        camera = object.__new__(CameraClient)
        with mock.patch.object(camera, "get_observation", side_effect=observations()) as get:
            result = camera.get_averaged_observation(depth_frames=3)
        self.assertEqual(get.call_count, 3)
        np.testing.assert_allclose(result.depth_meters, [[2, 6, 3], [0, 6, 0]])
        self.assertEqual(result.depth_meters.dtype, np.float32)
        np.testing.assert_array_equal(result.color_rgb, 10)
        self.assertEqual(result.frame_number, 42)
        self.assertEqual(result.depth_frames_averaged, 3)

    def test_single_frame_and_all_invalid_pixels(self) -> None:
        camera = object.__new__(CameraClient)
        observation = self._observation([[np.nan, 0], [-1, np.inf]])
        with mock.patch.object(camera, "get_observation", return_value=observation):
            result = camera.get_averaged_observation(depth_frames=1)
        np.testing.assert_array_equal(result.depth_meters, np.zeros((2, 2)))
        self.assertEqual(result.depth_frames_averaged, 1)

    def test_rejects_changed_geometry(self) -> None:
        camera = object.__new__(CameraClient)
        first = self._observation([[1, 2]])
        for second in (
            self._observation([[1], [2]]),
            replace(first, intrinsics=first.intrinsics * 2),
        ):
            with (
                mock.patch.object(camera, "get_observation", side_effect=[first, second]),
                self.assertRaisesRegex(ValueError, "geometry changed"),
            ):
                camera.get_averaged_observation(depth_frames=2)

    def test_default_capture_saves_ten_frame_mean_in_existing_format(self) -> None:
        observations = [
            self._observation([[index / 10, 0], [0, 0.5]])
            for index in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(CameraClient, "__init__", return_value=None) as initialize,
                mock.patch.object(CameraClient, "stop") as stop,
                mock.patch.object(CameraClient, "get_observation", side_effect=observations) as get,
            ):
                written = capture_scene(scenes_root=tmp, warmup_frames=4)
            initialize.assert_called_once_with(warmup_frames=4)
            stop.assert_called_once_with()
            self.assertEqual(get.call_count, 10)
            loaded = load_rgbd_scene(written)
            np.testing.assert_allclose(loaded.depth, [[0.55, 0], [0, 0.5]])
            self.assertEqual(np.load(written / "depth.npy").dtype, np.float32)
            self.assertTrue((written / "depth.png").is_file())
            metadata = json.loads((written / "capture_metadata.json").read_text())
            self.assertEqual(metadata["depth_frames_averaged"], 10)
            self.assertEqual(metadata["frame_number"], 42)


if __name__ == "__main__":
    unittest.main()
