"""Regression tests for Real2Sim object placement."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from real2sim._pipeline.reconstruction import (
    _conflict_resolution,
    _find_grounding_conflicts,
    _mask_overlap_metrics,
    _resolve_stacked_supports,
)
from real2sim._pipeline.world_frame import (
    _infer_support_selection_from_detected_objects,
)
from real2sim._stages.reconstruction import (
    _border_truncation_rejection,
    _support_clearance_lift,
    place_reconstructed_geometry,
    segment_object_grounding,
)
from real2sim.schema import PipelineConfig


class DynamicSupportClearanceLiftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig(
            world_frame_mode="support_surface",
            object_support_clearance=0.002,
        )

    def test_lifts_movable_object_out_of_support_plane(self) -> None:
        lift = _support_clearance_lift(
            min_z=-0.005,
            config=self.config,
        )

        self.assertAlmostEqual(lift, 0.007)

    def test_lifts_fixed_geometry_out_of_foundational_ground(self) -> None:
        lift = _support_clearance_lift(
            min_z=-0.294,
            config=self.config,
        )

        self.assertAlmostEqual(lift, 0.296)

    def test_does_not_lift_in_camera_world_mode(self) -> None:
        self.config.world_frame_mode = "camera"

        lift = _support_clearance_lift(
            min_z=-0.005,
            config=self.config,
        )

        self.assertEqual(lift, 0.0)

    def test_uses_estimated_floor_height_in_robot_world_mode(self) -> None:
        self.config.world_frame_mode = "camera_extrinsics"
        self.config.support_floor_z = 0.25

        lift = _support_clearance_lift(
            min_z=0.24,
            config=self.config,
        )

        self.assertAlmostEqual(lift, 0.012)


class StackedSupportResolutionTest(unittest.TestCase):
    def test_moves_book_with_lifted_platform(self) -> None:
        platform = SimpleNamespace(
            name="white platform",
            safe_name="white_platform",
            description="raised platform supporting the book",
            static_reason="fixed support surface",
            is_static=True,
            world_bounds=[[-0.4, -0.2, 0.002], [0.02, 0.1, 0.097]],
            support_lift_meters=0.0286,
        )
        book = SimpleNamespace(
            name="book",
            safe_name="book",
            description="book on white platform",
            static_reason="",
            is_static=False,
            body_pos=[-0.18, -0.05, 0.069],
            world_bounds=[[-0.29, -0.13, 0.062], [-0.08, 0.02, 0.076]],
            world_min_z_before_lift=0.062,
            world_min_z_after_lift=0.062,
            support_lift_meters=0.0,
            support_object_name="",
            support_stack_lift_meters=0.0,
            T_mesh_world=None,
        )
        config = PipelineConfig(
            world_frame_mode="support_surface",
            object_support_clearance=0.002,
            verbose=False,
        )

        _resolve_stacked_supports([book, platform], config=config)

        self.assertAlmostEqual(book.world_bounds[0][2], 0.099)
        self.assertEqual(book.support_object_name, "white platform")
        self.assertAlmostEqual(book.support_stack_lift_meters, 0.037)


class FoundationalSupportSelectionTest(unittest.TestCase):
    def test_prefers_table_below_raised_platform(self) -> None:
        selection = _infer_support_selection_from_detected_objects(
            [
                {
                    "name": "white platform",
                    "description": "raised platform supporting the book",
                },
                {
                    "name": "wooden table",
                    "description": "tabletop supporting the platform",
                },
            ]
        )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection["object_name"], "wooden table")


class BorderTruncationRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig()

    def test_rejects_medium_confidence_mask_at_image_edge(self) -> None:
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:80, 190:200] = 1

        rejection = _border_truncation_rejection(
            mask=mask,
            object_info={"confidence": "medium"},
            config=self.config,
        )

        self.assertIsNotNone(rejection)
        assert rejection is not None
        self.assertEqual(rejection["touched_edges"], ["right"])

    def test_keeps_high_confidence_mask_at_image_edge(self) -> None:
        mask = np.ones((20, 20), dtype=np.uint8)

        rejection = _border_truncation_rejection(
            mask=mask,
            object_info={"confidence": "high"},
            config=self.config,
        )

        self.assertIsNone(rejection)

    def test_keeps_medium_confidence_interior_mask(self) -> None:
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:80, 50:150] = 1

        rejection = _border_truncation_rejection(
            mask=mask,
            object_info={"confidence": "medium"},
            config=self.config,
        )

        self.assertIsNone(rejection)


class InstanceConditionedGroundingTest(unittest.TestCase):
    @patch("real2sim._stages.reconstruction.sam3_request")
    def test_bbox_agreement_beats_raw_sam_score(self, sam3_request) -> None:
        wrong = np.zeros((66, 66), dtype=np.uint8)
        wrong[1:15, 1:15] = 1
        correct = np.zeros((66, 66), dtype=np.uint8)
        correct[3:63, 3:63] = 1
        sam3_request.return_value = (
            [0.95, 0.60],
            [[1, 1, 15, 15], [3, 3, 63, 63]],
            [wrong, correct],
        )
        config = PipelineConfig(
            use_grounding_alias_fallback=False,
            reject_border_truncated_objects=False,
        )

        grounding = segment_object_grounding(
            image=Image.new("RGB", (100, 100)),
            object_info={
                "name": "center box",
                "description": "",
                "confidence": "high",
                "bbox_2d": [200, 200, 800, 800],
            },
            detected_objects=None,
            config=config,
        )

        self.assertIsNotNone(grounding)
        assert grounding is not None
        expected = np.zeros((100, 100), dtype=np.uint8)
        expected[20:80, 20:80] = 1
        np.testing.assert_array_equal(grounding["mask"], expected)
        self.assertEqual(sam3_request.call_args.kwargs["image"].size, (66, 66))
        self.assertIsNone(sam3_request.call_args.kwargs["bbox"])

    @patch("real2sim._stages.reconstruction.sam3_request")
    def test_rejects_sparse_mask_spanning_the_requested_box(self, sam3_request) -> None:
        sparse = np.zeros((66, 66), dtype=np.uint8)
        sparse[1:9, 1:9] = 1
        sparse[1:9, 57:65] = 1
        sparse[57:65, 1:9] = 1
        sparse[57:65, 57:65] = 1
        sam3_request.return_value = ([0.99], [[1, 1, 65, 65]], [sparse])
        object_info = {
            "name": "brown box",
            "description": "box near the center",
            "confidence": "high",
            "bbox_2d": [200, 200, 800, 800],
        }

        grounding = segment_object_grounding(
            image=Image.new("RGB", (100, 100)),
            object_info=object_info,
            detected_objects=None,
            config=PipelineConfig(
                use_grounding_alias_fallback=False,
                reject_border_truncated_objects=False,
            ),
        )

        self.assertIsNone(grounding)
        attempts = object_info["_grounding"]["attempts"]
        self.assertIn("bbox-fill", attempts[-1]["reason"])
        self.assertLess(
            attempts[-1]["candidate_mask_bbox_fill_ratios"][0],
            0.10,
        )


class GroundingConflictResolutionTest(unittest.TestCase):
    @staticmethod
    def _task(index: int, mask: np.ndarray, *, bbox_iou: float) -> dict:
        return {
            "index": index,
            "name": f"object {index}",
            "object_info": {"confidence": "high"},
            "grounding": {
                "mask": mask,
                "bbox_iou": bbox_iou,
                "score": 0.8,
            },
        }

    def test_near_identical_masks_prefer_instance_bbox_agreement(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        first[5:30, 5:30] = 1
        second = first.copy()
        second[29, 29] = 0
        tasks = [self._task(0, first, bbox_iou=0.95), self._task(1, second, bbox_iou=0.1)]
        config = PipelineConfig()

        conflicts = _find_grounding_conflicts(tasks, config=config)
        self.assertEqual(len(conflicts), 1)
        loser, winner, reason = _conflict_resolution(
            tasks, *conflicts[0], config=config
        )

        self.assertEqual((loser, winner), (1, 0))
        self.assertIn("lower-ranked", reason)

    def test_containment_only_conflict_rejects_larger_superset(self) -> None:
        smaller = np.zeros((40, 40), dtype=np.uint8)
        smaller[10:20, 10:20] = 1
        larger = np.zeros((40, 40), dtype=np.uint8)
        larger[2:32, 2:32] = 1
        metrics = _mask_overlap_metrics(smaller, larger)
        tasks = [self._task(0, smaller, bbox_iou=0.5), self._task(1, larger, bbox_iou=0.9)]

        loser, winner, reason = _conflict_resolution(
            tasks, 0, 1, metrics, config=PipelineConfig()
        )

        self.assertEqual((loser, winner), (1, 0))
        self.assertIn("superset", reason)

    def test_connected_static_boundary_keeps_more_complete_mask(self) -> None:
        smaller = np.zeros((40, 40), dtype=np.uint8)
        smaller[10:20, 10:20] = 1
        larger = np.zeros((40, 40), dtype=np.uint8)
        larger[2:32, 2:32] = 1
        metrics = _mask_overlap_metrics(smaller, larger)
        tasks = [
            self._task(0, smaller, bbox_iou=0.9),
            self._task(1, larger, bbox_iou=0.5),
        ]
        for task in tasks:
            task["object_info"].update(
                {
                    "name": "white divider",
                    "description": "connected tabletop divider panel",
                    "is_static": True,
                }
            )

        loser, winner, reason = _conflict_resolution(
            tasks, 0, 1, metrics, config=PipelineConfig()
        )

        self.assertEqual((loser, winner), (0, 1))
        self.assertIn("connected static boundary", reason)


class ExcessiveFoundationPoseLiftTest(unittest.TestCase):
    @staticmethod
    def _geometry(mask: np.ndarray) -> dict:
        source_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        return {
            "name": "box",
            "safe_name": "box",
            "object_info": {"name": "box", "confidence": "high"},
            "grounding": {
                "mask": mask,
                "bbox": [0, 0, 8, 8],
                "score": 0.9,
                "query": "box",
                "query_source": "object_name",
                "attempts": [],
                "aliases": [],
                "conflicts": [],
            },
            "mask": mask,
            "mask_area_px": 64,
            "mesh_for_fp": source_mesh,
            "mesh_for_fp_export": source_mesh,
            "mesh_fallback": source_mesh,
            "mesh_frame": "sam3d_processed_opengl",
            "scale": np.ones(3),
            "translation": np.zeros(3),
            "quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }

    @patch("real2sim._stages.reconstruction._place_mesh")
    def test_uses_metric_pointmap_fallback_for_excessive_lift(self, place_mesh) -> None:
        fp_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        fp_mesh.apply_translation([0.0, 0.0, -0.10])
        fallback_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.04])
        fallback_mesh.apply_translation([0.0, 0.0, 0.0])
        identity = np.eye(4)
        place_mesh.side_effect = [
            (fp_mesh, "foundation_pose_camera_frame", identity, identity, identity, {"mode": "full_frame"}, ""),
            (fallback_mesh, "sam3d_processed_opengl_camera_frame_fallback", identity, None, None, {}, ""),
        ]
        mask = np.ones((8, 8), dtype=np.uint8)
        geometry = self._geometry(mask)

        with tempfile.TemporaryDirectory() as tmp:
            reconstructed = place_reconstructed_geometry(
                geometry=geometry,
                K=np.eye(3),
                color=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                T_world_from_camera_opencv=identity,
                processed_camera_mesh_to_world=identity,
                raw_camera_mesh_to_world=identity,
                meshes_dir=Path(tmp),
                config=PipelineConfig(
                    max_support_lift_meters=0.10,
                    visual_color_clusters=0,
                ),
            )

        self.assertIn("support_lift_fallback", reconstructed.pose_source)
        self.assertLess(reconstructed.support_lift_meters, 0.10)
        self.assertIn(
            "excessive support-plane correction",
            reconstructed.foundation_pose_error,
        )

    @patch("real2sim._stages.reconstruction._refine_pose_with_icp")
    @patch("real2sim._stages.reconstruction._place_mesh")
    def test_uses_metric_depth_icp_for_suspicious_foundation_pose(
        self,
        place_mesh,
        refine_pose,
    ) -> None:
        fp_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        fp_mesh.apply_translation([0.0, 0.0, -0.10])
        refined_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        refined_mesh.apply_translation([0.0, 0.0, 0.05])
        identity = np.eye(4)
        place_mesh.return_value = (
            fp_mesh,
            "foundation_pose_camera_frame",
            identity,
            identity,
            identity,
            {"mode": "full_frame"},
            "",
        )
        refine_pose.return_value = (
            refined_mesh,
            identity,
            {
                "attempted": True,
                "accepted": True,
                "fitness": 0.9,
                "inlier_rmse_meters": 0.005,
                "translation_meters": 0.04,
                "rotation_degrees": 2.0,
                "reason": "",
            },
        )
        mask = np.ones((8, 8), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            reconstructed = place_reconstructed_geometry(
                geometry=self._geometry(mask),
                K=np.eye(3),
                color=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                T_world_from_camera_opencv=identity,
                processed_camera_mesh_to_world=identity,
                raw_camera_mesh_to_world=identity,
                meshes_dir=Path(tmp),
                config=PipelineConfig(visual_color_clusters=0),
            )

        self.assertEqual(place_mesh.call_count, 1)
        refine_pose.assert_called_once()
        self.assertTrue(reconstructed.pose_source.endswith("metric_depth_icp"))
        self.assertTrue(reconstructed.icp_registration["accepted"])

if __name__ == "__main__":
    unittest.main()
