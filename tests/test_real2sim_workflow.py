"""Tests for Real2Sim workflow input routing."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from real2sim import workflow
from vistools.rgbd_input import RGBDSceneInput


class ReuseExistingReconstructionTest(unittest.TestCase):
    def test_image_reuses_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.touch()
            output = root / "result"
            output.mkdir()
            (output / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
            parser = workflow._build_parser()
            args = parser.parse_args(
                [
                    str(image),
                    "--output",
                    str(output),
                    "--reuse-existing",
                    "--no-package",
                ]
            )

            with mock.patch.object(workflow, "run_pipeline_from_args") as run_pipeline:
                result = workflow.build_from_input(image, args, parser)

            run_pipeline.assert_not_called()
            self.assertTrue(result.reused_reconstruction)
            self.assertEqual(result.reconstruction_dir, output.resolve())

    def test_incomplete_result_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.touch()
            output = root / "result"
            output.mkdir()
            parser = workflow._build_parser()
            args = parser.parse_args(
                [
                    str(image),
                    "--output",
                    str(output),
                    "--reuse-existing",
                    "--no-package",
                ]
            )

            with mock.patch.object(workflow, "run_pipeline_from_args") as run_pipeline:
                result = workflow.build_from_input(image, args, parser)

            run_pipeline.assert_called_once_with(
                image.resolve(),
                output.resolve(),
                args,
                parser,
            )
            self.assertFalse(result.reused_reconstruction)

    def test_video_reuse_skips_frame_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.touch()
            output = root / "result"
            output.mkdir()
            (output / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
            parser = workflow._build_parser()
            args = parser.parse_args(
                [
                    str(video),
                    "--output",
                    str(output),
                    "--reuse-existing",
                    "--no-package",
                ]
            )

            with (
                mock.patch.object(workflow, "extract_first_frame") as extract,
                mock.patch.object(workflow, "run_pipeline_from_args") as run_pipeline,
            ):
                result = workflow.build_from_input(video, args, parser)

            extract.assert_not_called()
            run_pipeline.assert_not_called()
            self.assertTrue(result.reused_reconstruction)
            self.assertIsNone(result.frame_image)

    def test_reuse_with_force_keeps_the_reconstruction_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.touch()
            output = root / "result"
            output.mkdir()
            (output / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
            stale = output / "preserved_mesh.obj"
            stale.touch()
            parser = workflow._build_parser()
            args = parser.parse_args(
                [
                    str(image),
                    "--output",
                    str(output),
                    "--reuse-existing",
                    "--force",
                    "--no-package",
                ]
            )

            with mock.patch.object(workflow, "run_pipeline_from_args") as run_pipeline:
                result = workflow.build_from_input(image, args, parser)

            run_pipeline.assert_not_called()
            self.assertTrue(result.reused_reconstruction)
            self.assertTrue(stale.exists())


class ForceReconstructionCleanupTest(unittest.TestCase):
    def test_force_removes_entire_result_before_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.touch()
            output = root / "result"
            stale_mesh = output / "meshes" / "stale.obj"
            stale_mesh.parent.mkdir(parents=True)
            stale_mesh.touch()
            (output / "obsolete.json").write_text("{}", encoding="utf-8")
            parser = workflow._build_parser()
            args = parser.parse_args(
                [
                    str(image),
                    "--output",
                    str(output),
                    "--force",
                    "--no-package",
                ]
            )

            def assert_clean(*_args, **_kwargs) -> None:
                self.assertFalse(output.exists())

            with mock.patch.object(
                workflow,
                "run_pipeline_from_args",
                side_effect=assert_clean,
            ) as run_pipeline:
                workflow.build_from_input(image, args, parser)

            run_pipeline.assert_called_once()
            self.assertFalse(stale_mesh.exists())


class RGBDCaptureWorkflowTest(unittest.TestCase):
    def test_directory_routes_supplied_camera_data_to_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "capture"
            source.mkdir()
            image = source / "color.png"
            output = root / "result"
            rgbd = RGBDSceneInput(
                directory=source,
                image_path=image,
                depth=np.ones((2, 3), dtype=np.float32),
                camera_metadata={"provider": "rgbd_scene_directory"},
            )
            parser = workflow._build_parser()
            args = parser.parse_args(
                [str(source), "--output", str(output), "--no-package"]
            )

            with (
                mock.patch.object(
                    workflow,
                    "is_rgbd_scene_directory",
                    return_value=True,
                ),
                mock.patch.object(workflow, "load_rgbd_scene", return_value=rgbd),
                mock.patch.object(workflow, "run_pipeline_from_args") as run_pipeline,
            ):
                result = workflow.build_from_input(source, args, parser)

            run_pipeline.assert_called_once_with(
                image,
                output.resolve(),
                args,
                parser,
                depth=rgbd.depth,
                depth_metadata=rgbd.camera_metadata,
            )
            self.assertEqual(result.reconstruction_dir, output.resolve())
            self.assertFalse(result.reused_reconstruction)
            self.assertEqual(args.world_frame, "robot")


class ScenePreviewWorkflowTest(unittest.TestCase):
    def test_packages_and_previews_fr3_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "result"
            source.mkdir()
            (source / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
            parser = workflow._build_parser()
            args = parser.parse_args([str(source), "--preview"])
            arm = workflow.PackagedScene(root / "arm", "tasks.invention_real2sim.result")
            preview = root / "preview.png"
            with (
                mock.patch.object(
                    workflow,
                    "package_reconstruction",
                    return_value=arm,
                ) as package_arm,
                mock.patch.object(
                    workflow,
                    "_preview_packaged_scene",
                    return_value=preview,
                ) as render,
            ):
                result = workflow.build_from_input(source, args, parser)

            package_arm.assert_called_once()
            render.assert_called_once_with(arm.module, args)
            self.assertEqual(result.packaged_scene, arm)
            self.assertEqual(result.preview_image, preview)

if __name__ == "__main__":
    unittest.main()
