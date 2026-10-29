"""Command-line interface for the single-image real2sim pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from real2sim.schema import PipelineConfig


def parse_matrix(value: str, rows: int, cols: int) -> np.ndarray:
    vals = [float(v) for v in value.split(",")]
    if len(vals) != rows * cols:
        raise argparse.ArgumentTypeError(
            f"expected {rows * cols} comma-separated values, got {len(vals)}"
        )
    return np.asarray(vals, dtype=np.float64).reshape(rows, cols)


def parse_csv_urls(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def add_pipeline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add single-image reconstruction options to an argument parser."""

    defaults = PipelineConfig()
    parser.add_argument("--vlm-model", default=defaults.vlm_model)
    parser.add_argument(
        "--vlm-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=defaults.vlm_reasoning_effort or "none",
        help=(
            "Reasoning effort sent with every VLM request. Use 'none' to omit "
            "the API parameter."
        ),
    )
    parser.add_argument("--depth-url", default=defaults.depth_server_url)
    parser.add_argument("--sam3-url", default=defaults.sam3_server_url)
    parser.add_argument("--sam3d-url", default=defaults.sam3d_server_url)
    parser.add_argument(
        "--sam3d-urls",
        default=None,
        help=(
            "Comma-separated SAM3D endpoint pool for parallel reconstruction. "
            "Defaults to --sam3d-url."
        ),
    )
    parser.add_argument(
        "--foundation-pose-url",
        default=defaults.foundation_pose_url,
    )
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument(
        "--intrinsics",
        type=lambda s: parse_matrix(s, 3, 3),
        default=None,
        help="Comma-separated 3x3 K matrix in row-major order.",
    )
    parser.add_argument(
        "--extrinsics",
        default=None,
        help="Comma-separated 3x4 or 4x4 OpenCV w2c matrix in row-major order.",
    )
    parser.add_argument("--process-res", type=int, default=None)
    parser.add_argument("--process-res-method", default=None)
    parser.add_argument(
        "--input-max-size",
        type=int,
        default=defaults.input_max_size,
        help=(
            "Resize the input so its longest side is at most this many pixels. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument(
        "--world-frame",
        choices=(
            "support-surface",
            "camera",
            "camera-extrinsics",
            "robot",
            "depth-anything",
        ),
        default=None,
        help=(
            "World-frame source. Defaults to the calibrated robot frame for an "
            "RGB-D capture and support-surface for image/video input. "
            "camera-extrinsics preserves a supplied calibrated frame."
        ),
    )
    parser.add_argument(
        "--support-confidence",
        choices=("high", "medium", "low"),
        default=defaults.support_confidence,
        help="Minimum accepted confidence for the support-surface VLM selection.",
    )
    parser.add_argument(
        "--support-plane-distance-threshold",
        type=float,
        default=defaults.support_plane_distance_threshold,
        help="RANSAC inlier distance threshold for support-plane fitting, in depth units.",
    )
    parser.add_argument(
        "--support-plane-ransac-iterations",
        type=int,
        default=defaults.support_plane_ransac_iterations,
    )
    parser.add_argument(
        "--support-plane-max-points",
        type=int,
        default=defaults.support_plane_max_points,
    )
    parser.add_argument(
        "--support-plane-min-inlier-ratio",
        type=float,
        default=defaults.support_plane_min_inlier_ratio,
    )
    parser.add_argument(
        "--no-global-support-fallback",
        action="store_true",
        help=(
            "Fail instead of using global dominant-plane RANSAC if VLM/SAM "
            "support fitting fails."
        ),
    )
    parser.add_argument(
        "--include-support-object",
        action="store_true",
        help=(
            "Also reconstruct/export the selected support surface object. By default "
            "it is represented by the fitted floor only."
        ),
    )
    parser.add_argument(
        "--no-static-object-selection",
        action="store_true",
        help="Disable the VLM pass that marks fixed obstacles as static.",
    )
    parser.add_argument(
        "--static-confidence",
        choices=("high", "medium", "low"),
        default=defaults.static_confidence,
        help="Minimum accepted confidence for static-object VLM decisions.",
    )
    parser.add_argument(
        "--no-physical-parameters",
        action="store_true",
        help="Disable the VLM pass that estimates material/contact parameters.",
    )
    parser.add_argument(
        "--physical-confidence",
        choices=("high", "medium", "low"),
        default=defaults.physical_confidence,
        help="Minimum accepted confidence for physical-parameter VLM estimates.",
    )
    parser.add_argument(
        "--no-grounding-alias-fallback",
        action="store_true",
        help="Disable the VLM retry that suggests alternative SAM grounding phrases.",
    )
    parser.add_argument(
        "--max-grounding-aliases",
        type=int,
        default=defaults.max_grounding_aliases,
        help="Maximum VLM-proposed grounding aliases to retry per object.",
    )
    parser.add_argument(
        "--grounding-bbox-margin",
        type=float,
        default=defaults.grounding_bbox_margin_fraction,
        help="Fractional margin added around VLM object boxes before SAM grounding.",
    )
    parser.add_argument(
        "--min-grounding-mask-bbox-fill-ratio",
        type=float,
        default=defaults.min_grounding_mask_bbox_fill_ratio,
        help=(
            "Reject sparse SAM masks whose foreground fills less than this "
            "fraction of their own bounding box."
        ),
    )
    parser.add_argument(
        "--duplicate-mask-iou-threshold",
        type=float,
        default=defaults.duplicate_mask_iou_threshold,
        help="Reject unresolved object masks at or above this pairwise IoU.",
    )
    parser.add_argument(
        "--duplicate-mask-containment-threshold",
        type=float,
        default=defaults.duplicate_mask_containment_threshold,
        help=(
            "Reject unresolved object masks when their intersection divided by "
            "the smaller mask area reaches this value."
        ),
    )
    parser.add_argument(
        "--reconstruction-workers",
        type=int,
        default=defaults.reconstruction_workers,
        help="Number of object reconstruction jobs to run in parallel.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=defaults.target_faces,
        help=(
            "Face budget for processed SAM3D object meshes. "
            "Use 0 to disable simplification."
        ),
    )
    parser.add_argument(
        "--visual-color-clusters",
        type=int,
        default=defaults.visual_color_clusters,
        help=(
            "Number of color-clustered visual mesh parts per object. "
            "Use 0 for one uncolored visual mesh."
        ),
    )
    parser.add_argument(
        "--pointcloud-stride",
        type=int,
        default=defaults.pointcloud_stride,
        help="Pixel stride for exported scene point clouds. Use 1 for full resolution.",
    )
    parser.add_argument("--include-low-confidence", action="store_true")
    parser.add_argument(
        "--include-border-truncated-objects",
        action="store_true",
        help=(
            "Reconstruct medium/low-confidence objects whose masks touch an image "
            "boundary. By default these unreliable partial objects are skipped."
        ),
    )
    parser.add_argument(
        "--no-foundation-pose",
        action="store_true",
        help="Skip Foundation Pose registration and use the SAM3D placement fallback.",
    )
    parser.add_argument(
        "--no-fp-fallback",
        action="store_true",
        help="Fail the object reconstruction if Foundation Pose registration fails.",
    )
    parser.add_argument(
        "--foundation-pose-target-faces",
        type=int,
        default=defaults.foundation_pose_target_faces,
        help=(
            "Mesh face budget for Foundation Pose registration. "
            "Use 0 to disable FP-specific simplification."
        ),
    )
    parser.add_argument(
        "--object-support-clearance",
        type=float,
        default=defaults.object_support_clearance,
        help=(
            "Minimum object clearance above the support plane z=0 in support-surface "
            "world mode, in meters."
        ),
    )
    parser.add_argument(
        "--max-support-lift",
        type=float,
        default=defaults.max_support_lift_meters,
        help=(
            "Maximum support-plane correction accepted for an object pose, "
            "in meters."
        ),
    )
    parser.add_argument(
        "--icp-fallback-support-correction",
        type=float,
        default=defaults.icp_fallback_support_correction_meters,
        help=(
            "Run metric-depth ICP when Foundation Pose requires more than this "
            "below-plane correction, in meters."
        ),
    )
    parser.add_argument(
        "--icp-min-fitness",
        type=float,
        default=defaults.icp_min_fitness,
        help="Minimum accepted metric-depth ICP fitness.",
    )
    parser.add_argument(
        "--icp-max-rmse",
        type=float,
        default=defaults.icp_max_inlier_rmse_meters,
        help="Maximum accepted metric-depth ICP inlier RMSE, in meters.",
    )
    parser.add_argument(
        "--icp-max-translation",
        type=float,
        default=defaults.icp_max_translation_meters,
        help="Maximum accepted metric-depth ICP center translation, in meters.",
    )
    parser.add_argument(
        "--icp-max-rotation",
        type=float,
        default=defaults.icp_max_rotation_degrees,
        help="Maximum accepted metric-depth ICP rotation, in degrees.",
    )
    parser.add_argument(
        "--max-initial-object-penetration",
        type=float,
        default=defaults.max_initial_object_penetration_meters,
        help=(
            "Maximum initial MuJoCo penetration allowed between reconstructed "
            "objects, in meters."
        ),
    )
    parser.add_argument(
        "--default-object-mass",
        type=float,
        default=defaults.object_mass,
        help="Fallback object mass in kilograms when no VLM mass estimate is available.",
    )
    dynamics_group = parser.add_mutually_exclusive_group()
    dynamics_group.add_argument(
        "--dynamic-objects",
        dest="dynamic_objects",
        action="store_true",
        default=None,
        help="Export reconstructed objects as freejoint bodies. This is the default.",
    )
    dynamics_group.add_argument(
        "--static-objects",
        dest="dynamic_objects",
        action="store_false",
        help="Export reconstructed objects as fixed visual bodies for pose inspection.",
    )
    parser.add_argument("--no-ray-pose", action="store_true")
    parser.add_argument("--no-compile-check", action="store_true")
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument(
        "--agent-fallback",
        dest="agent_fallback",
        action="store_true",
        default=True,
        help=(
            "On initial scene-health validation failure, run the local "
            "repair-real2sim-scene Codex agent and continue after it validates. "
            "This is the default for CLI runs."
        ),
    )
    repair_group.add_argument(
        "--no-agent-fallback",
        dest="agent_fallback",
        action="store_false",
        help="Fail immediately instead of asking the scene-repair agent to fix MJCF.",
    )
    parser.add_argument(
        "--agent-fallback-model",
        default=defaults.scene_repair_agent_model,
        help="Codex model used by the scene-repair fallback.",
    )
    parser.add_argument(
        "--agent-fallback-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=defaults.scene_repair_agent_reasoning_effort,
    )
    parser.add_argument(
        "--agent-fallback-max-turns",
        type=int,
        default=defaults.scene_repair_agent_max_turns,
        help=(
            "Approximate maximum repair/debug cycles requested from the "
            "fallback agent."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo scene from one RGB image."
    )
    parser.add_argument("image", help="Input RGB image path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output scene bundle directory. Defaults to results/real2sim/<image-stem>.",
    )
    add_pipeline_options(parser)
    return parser


def pipeline_output_dir(image_path: str | Path, output: str | Path | None) -> Path:
    """Return the reconstruction output directory for an image input."""

    image = Path(image_path)
    if output is None:
        return Path("results") / "real2sim" / image.stem
    return Path(output)


def pipeline_config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> PipelineConfig:
    """Build :class:`PipelineConfig` from parsed reconstruction options."""

    extrinsics = None
    if args.extrinsics:
        vals = [float(v) for v in args.extrinsics.split(",")]
        if len(vals) == 12:
            extrinsics = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        elif len(vals) == 16:
            extrinsics = np.asarray(vals, dtype=np.float64).reshape(4, 4)
        else:
            message = "--extrinsics must contain 12 or 16 comma-separated values"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)

    accepted_confidences = (
        ("high", "medium", "low")
        if args.include_low_confidence
        else ("high", "medium")
    )
    default_config = PipelineConfig()
    dynamic_objects = (
        default_config.dynamic_objects
        if args.dynamic_objects is None
        else args.dynamic_objects
    )
    foundation_pose_target_faces = (
        None
        if args.foundation_pose_target_faces is not None
        and args.foundation_pose_target_faces <= 0
        else args.foundation_pose_target_faces
    )
    target_faces = (
        None
        if args.target_faces is not None and args.target_faces <= 0
        else args.target_faces
    )
    input_max_size = (
        None
        if args.input_max_size is not None and args.input_max_size <= 0
        else args.input_max_size
    )
    sam3d_server_urls = parse_csv_urls(args.sam3d_urls)

    return PipelineConfig(
        depth_server_url=args.depth_url,
        sam3_server_url=args.sam3_url,
        sam3d_server_url=args.sam3d_url,
        sam3d_server_urls=sam3d_server_urls,
        foundation_pose_url=args.foundation_pose_url,
        vlm_model=args.vlm_model,
        vlm_reasoning_effort=(
            None
            if args.vlm_reasoning_effort == "none"
            else args.vlm_reasoning_effort
        ),
        use_ray_pose=not args.no_ray_pose,
        focal=args.focal,
        intrinsics=args.intrinsics,
        extrinsics=extrinsics,
        input_max_size=input_max_size,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        world_frame_mode=(args.world_frame or default_config.world_frame_mode).replace(
            "-", "_"
        ),
        support_confidence=args.support_confidence,
        support_plane_distance_threshold=args.support_plane_distance_threshold,
        support_plane_ransac_iterations=args.support_plane_ransac_iterations,
        support_plane_max_points=args.support_plane_max_points,
        support_plane_min_inlier_ratio=args.support_plane_min_inlier_ratio,
        allow_global_support_fallback=not args.no_global_support_fallback,
        reconstruct_support_surface=args.include_support_object,
        use_static_object_selection=not args.no_static_object_selection,
        static_confidence=args.static_confidence,
        use_physical_parameter_selection=not args.no_physical_parameters,
        physical_confidence=args.physical_confidence,
        use_grounding_alias_fallback=not args.no_grounding_alias_fallback,
        max_grounding_aliases=args.max_grounding_aliases,
        grounding_bbox_margin_fraction=max(0.0, args.grounding_bbox_margin),
        min_grounding_mask_bbox_fill_ratio=min(
            1.0, max(0.0, args.min_grounding_mask_bbox_fill_ratio)
        ),
        duplicate_mask_iou_threshold=min(
            1.0, max(0.0, args.duplicate_mask_iou_threshold)
        ),
        duplicate_mask_containment_threshold=min(
            1.0, max(0.0, args.duplicate_mask_containment_threshold)
        ),
        reconstruction_workers=max(1, args.reconstruction_workers),
        visual_color_clusters=max(0, args.visual_color_clusters),
        accepted_confidences=accepted_confidences,
        reject_border_truncated_objects=not args.include_border_truncated_objects,
        max_objects=args.max_objects,
        target_faces=target_faces,
        use_foundation_pose=not args.no_foundation_pose,
        fallback_without_foundation_pose=not args.no_fp_fallback,
        foundation_pose_target_faces=foundation_pose_target_faces,
        pointcloud_stride=args.pointcloud_stride,
        dynamic_objects=dynamic_objects,
        object_mass=max(0.001, args.default_object_mass),
        object_support_clearance=max(0.0, args.object_support_clearance),
        icp_fallback_support_correction_meters=max(
            0.0, args.icp_fallback_support_correction
        ),
        icp_min_fitness=min(1.0, max(0.0, args.icp_min_fitness)),
        icp_max_inlier_rmse_meters=max(0.0, args.icp_max_rmse),
        icp_max_translation_meters=max(0.0, args.icp_max_translation),
        icp_max_rotation_degrees=max(0.0, args.icp_max_rotation),
        max_support_lift_meters=max(0.0, args.max_support_lift),
        max_initial_object_penetration_meters=max(
            0.0, args.max_initial_object_penetration
        ),
        compile_check=not args.no_compile_check,
        use_scene_repair_agent=bool(args.agent_fallback),
        scene_repair_agent_model=str(args.agent_fallback_model).strip(),
        scene_repair_agent_reasoning_effort=(
            args.agent_fallback_reasoning_effort
        ),
        scene_repair_agent_max_turns=max(1, args.agent_fallback_max_turns),
        verbose=not args.quiet,
    )


def run_pipeline_from_args(
    image_path: str | Path,
    output_dir: str | Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    *,
    depth: np.ndarray | None = None,
    depth_metadata: dict | None = None,
):
    """Run reconstruction for ``image_path`` using parsed CLI options."""

    from real2sim.pipeline import run_pipeline

    config = pipeline_config_from_args(args, parser)
    return run_pipeline(
        image_path,
        output_dir,
        config,
        depth=depth,
        depth_metadata=depth_metadata,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    output_dir = pipeline_output_dir(image_path, args.output)
    result = run_pipeline_from_args(image_path, output_dir, args, parser)

    print(f"scene: {result.scene_xml}")
    final_objects = result.scene_object_names
    print(f"objects: {len(result.objects) if final_objects is None else len(final_objects)}")
    if result.skipped_objects:
        print(f"skipped: {len(result.skipped_objects)}")
    if result.warnings:
        print(f"warnings: {len(result.warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
