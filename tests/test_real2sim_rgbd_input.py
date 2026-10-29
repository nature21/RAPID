"""Tests for supplied RGB-D Real2Sim inputs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from real2sim.adapter import Real2SimTaskConfig, make_default_camera
from real2sim.pipeline import _prepare_provided_depth, run_pipeline
from real2sim._pipeline.world_frame import (
    build_world_frame,
    normalize_world_frame_mode,
)
from real2sim._pipeline.reconstruction import (
    reconstruct_scene_objects,
    selected_support_object_name,
)
from real2sim.mujoco_export import write_mujoco_scene
from vistools.rgbd_input import is_rgbd_scene_directory, load_rgbd_scene
from real2sim.schema import PipelineConfig


class RGBDSceneInputTest(unittest.TestCase):
    def test_robot_mode_does_not_change_legacy_depth_anything_mode(self) -> None:
        self.assertEqual(normalize_world_frame_mode("robot"), "camera_extrinsics")
        self.assertEqual(normalize_world_frame_mode("depth-anything"), "depth_anything")

    def test_loads_depth_intrinsics_and_easy_handeye_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_scene(root)

            self.assertTrue(is_rgbd_scene_directory(root))
            scene = load_rgbd_scene(root)

            self.assertEqual(scene.image_path, root / "color.png")
            self.assertEqual(scene.depth.shape, (3, 4))
            np.testing.assert_allclose(
                scene.camera_metadata["intrinsics"],
                [[100.0, 0.0, 2.0], [0.0, 120.0, 1.5], [0.0, 0.0, 1.0]],
            )
            np.testing.assert_allclose(
                scene.camera_metadata["extrinsics"],
                [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                atol=1e-7,
            )
            self.assertEqual(
                scene.camera_metadata["extrinsics_convention"],
                "opencv_c2w_4x4",
            )
            self.assertEqual(
                scene.camera_metadata["calibration"]["parent_frame"],
                "fr3_link0",
            )

    def test_rejects_depth_that_does_not_match_the_color_image(self) -> None:
        depth = np.ones((2, 4), dtype=np.float32)
        metadata = {"intrinsics": np.eye(3)}

        with self.assertRaisesRegex(ValueError, "must match input image"):
            _prepare_provided_depth(
                depth,
                metadata,
                original_size=(4, 3),
                processing_size=(4, 3),
                resize_metadata={"resized": False, "scale_x": 1.0, "scale_y": 1.0},
            )

    def test_rejects_eye_on_hand_calibration_for_robot_world_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_scene(root)
            calibration = root / "panda_eob_calib.yaml"
            calibration.write_text(
                calibration.read_text(encoding="utf-8").replace(
                    "eye_on_hand: false", "eye_on_hand: true"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "eye-on-base calibration"):
                load_rgbd_scene(root)

    def test_resizes_registered_depth_and_intrinsics_together(self) -> None:
        depth = np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0
        intrinsics = np.asarray(
            [[100.0, 0.0, 2.0], [0.0, 120.0, 1.0], [0.0, 0.0, 1.0]]
        )

        resized_depth, metadata = _prepare_provided_depth(
            depth,
            {"intrinsics": intrinsics},
            original_size=(4, 2),
            processing_size=(2, 1),
            resize_metadata={"resized": True, "scale_x": 0.5, "scale_y": 0.5},
        )

        self.assertEqual(resized_depth.shape, (1, 2))
        np.testing.assert_allclose(
            metadata["intrinsics"],
            [[50.0, 0.0, 1.0], [0.0, 60.0, 0.5], [0.0, 0.0, 1.0]],
        )

    def test_supplied_depth_bypasses_depth_anything(self) -> None:
        class StopAfterDepth(RuntimeError):
            pass

        image = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.float32)
        metadata = {
            "intrinsics": np.eye(3),
            "extrinsics": np.eye(4),
            "extrinsics_convention": "opencv_c2w_4x4",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("real2sim.pipeline.request_depth") as request_depth,
                mock.patch(
                    "real2sim.pipeline.detect_objects",
                    side_effect=StopAfterDepth,
                ),
            ):
                with self.assertRaises(StopAfterDepth):
                    run_pipeline(
                        image,
                        tmp,
                        PipelineConfig(
                            input_max_size=None,
                            pointcloud_stride=1,
                            compile_check=False,
                            verbose=False,
                        ),
                        depth=depth,
                        depth_metadata=metadata,
                    )

            request_depth.assert_not_called()

    def test_robot_world_frame_uses_supplied_camera_to_base_transform(self) -> None:
        camera_to_robot = np.eye(4)
        camera_to_robot[:3, 3] = [1.0, 2.0, 3.0]
        config = PipelineConfig(world_frame_mode="robot", verbose=False)
        pointmap = np.asarray(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        selection = {
            "object_name": "table",
            "sam_query": "tabletop",
            "confidence": "high",
        }
        mask = np.ones((2, 2), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch(
                    "real2sim._pipeline.world_frame._select_support_surface_with_retry",
                    return_value=(selection, "support response"),
                ),
                mock.patch(
                    "real2sim._pipeline.world_frame.segment_support_surface",
                    return_value=(
                        mask,
                        {
                            "query": "tabletop",
                            "score": 1.0,
                            "mask_area_px": 4,
                        },
                    ),
                ),
            ):
                world_from_camera, metadata, support_mask = build_world_frame(
                    image=Image.new("RGB", (2, 2)),
                    detected_objects=[{"name": "table"}],
                    pointmap_opencv=pointmap,
                    depth_c2w_opencv=camera_to_robot,
                    depth_w2c_opencv=np.linalg.inv(camera_to_robot),
                    depth_extrinsics_source="fr3_link0<-camera_color_frame",
                    masks_dir=Path(tmp),
                    config=config,
                    warnings=[],
                )

        np.testing.assert_allclose(world_from_camera, camera_to_robot)
        self.assertEqual(metadata["mode"], "camera_extrinsics")
        self.assertEqual(metadata["source"], "fr3_link0<-camera_color_frame")
        self.assertEqual(selected_support_object_name(metadata), "table")
        self.assertAlmostEqual(metadata["support_floor_z"], 4.0)
        np.testing.assert_array_equal(support_mask, mask)

    def test_mujoco_floor_uses_estimated_robot_frame_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_xml = Path(tmp) / "scene.xml"
            config = PipelineConfig(
                world_frame_mode="camera_extrinsics",
                support_floor_z=0.246,
                verbose=False,
            )

            write_mujoco_scene(scene_xml, [], config)

            floor = ET.parse(scene_xml).getroot().find("./worldbody/geom[@name='floor']")
            self.assertIsNotNone(floor)
            assert floor is not None
            self.assertEqual(floor.get("pos"), "0 0 0.246")

    def test_selected_support_is_not_reconstructed_in_robot_world_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            (root / "masks").mkdir()
            config = PipelineConfig(
                world_frame_mode="camera_extrinsics",
                reconstruct_support_surface=False,
                verbose=False,
            )

            reconstructed, skipped = reconstruct_scene_objects(
                image=Image.new("RGB", (2, 2)),
                detected_objects=[{"name": "wooden table"}],
                K=np.eye(3),
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.float32),
                T_world_from_camera_opencv=np.eye(4),
                pointmap_p3d=np.zeros((2, 2, 3), dtype=np.float32),
                meshes_dir=root / "meshes",
                masks_dir=root / "masks",
                world_frame_metadata={
                    "support_selection": {"object_name": "wooden table"}
                },
                config=config,
            )

            self.assertEqual(reconstructed, [])
            self.assertEqual(len(skipped), 1)
            self.assertIn("selected support surface", skipped[0]["reason"])

    def test_packaged_default_camera_uses_captured_opencv_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_text(
                "<mujoco><worldbody/></mujoco>\n",
                encoding="utf-8",
            )
            transform = np.eye(4)
            transform[:3, :3] = [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
            transform[:3, 3] = [1.0, 2.0, 3.0]
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "depth_input": {"provider": "rgbd_scene_directory"},
                        "camera": {
                            "world_frame_mode": "camera_extrinsics",
                            "T_world_from_camera_opencv": transform.tolist(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            camera = make_default_camera(Real2SimTaskConfig(scene_dir=root))

        self.assertEqual(camera["camera_lookat"], (0.0, 2.0, 3.0))
        self.assertEqual(camera["camera_distance"], 1.0)
        self.assertEqual(camera["camera_azimuth"], 180.0)
        self.assertEqual(camera["camera_elevation"], 0.0)

    def test_packaged_default_camera_recovers_from_depth_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_text(
                "<mujoco><worldbody/></mujoco>\n",
                encoding="utf-8",
            )
            transform = np.eye(4)
            transform[:3, :3] = [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
            transform[:3, 3] = [1.0, 2.0, 3.0]
            np.savez_compressed(root / "depth_camera.npz", c2w_opencv=transform)

            camera = make_default_camera(Real2SimTaskConfig(scene_dir=root))

        self.assertEqual(camera["camera_lookat"], (0.0, 2.0, 3.0))
        self.assertEqual(camera["camera_distance"], 1.0)
        self.assertEqual(camera["camera_azimuth"], 180.0)
        self.assertEqual(camera["camera_elevation"], 0.0)

    @staticmethod
    def _write_scene(root: Path) -> None:
        Image.new("RGB", (4, 3)).save(root / "color.png")
        np.save(root / "depth.npy", np.ones((3, 4), dtype=np.float32))
        (root / "intrinsics.json").write_text(
            json.dumps(
                [[100.0, 0.0, 2.0], [0.0, 120.0, 1.5], [0.0, 0.0, 1.0]]
            ),
            encoding="utf-8",
        )
        (root / "panda_eob_calib.yaml").write_text(
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
  qz: 0.7071067811865476
  qw: 0.7071067811865476
""",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
