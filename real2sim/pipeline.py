"""Single-image real2sim pipeline.

The pipeline turns one RGB image or registered RGB-D capture into a MuJoCo
scene bundle:

1. Supplied RGB-D camera data is loaded, or Depth Anything estimates it.
2. A VLM identifies meaningful physical objects.
3. The supplied camera calibration or a selected support surface defines world.
4. A VLM estimates physical material/contact parameters.
5. SAM3 segments the support surface and each object by name.
6. SAM3D reconstructs each segmented object from an RGB image, mask, and
   camera point map.
7. Reconstructed meshes are transformed into the support-aligned world frame
   and exported as a MuJoCo MJCF scene.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from real2sim.camera import (
    T_OPENCV_TO_P3D,
    T_OPENGL_TO_OPENCV,
    camera_extrinsics,
    camera_intrinsics,
    depth_to_pointmap,
    transform_pointmap,
)
from real2sim.mujoco_export import (
    InitialObjectPenetrationError,
    write_mujoco_scene,
)
from real2sim._pipeline.images import (
    as_rgb_image,
    masked_pointmap,
    resize_image_for_processing,
    scale_intrinsics_for_resize,
)
from real2sim._pipeline.progress import (
    export_mode_label,
    log,
    log_depth_camera,
    log_physical_parameters,
    log_static_selections,
    log_world_frame,
)
from real2sim._pipeline.vlm import (
    apply_physical_parameters,
    apply_static_selections,
    select_physical_parameters_for_scene,
    select_static_objects_for_scene,
    write_single_vlm_response,
    write_vlm_response_files,
)
from real2sim.pointcloud import save_colored_pointcloud_ply
from real2sim._pipeline.reconstruction import (
    reconstruct_scene_objects,
    sam3d_urls_for_reconstruction,
)
from real2sim._stages.perception import request_depth
from real2sim.schema import PipelineConfig, PipelineResult, jsonify
from real2sim._stages.vlm import VLMResponseError, detect_objects
from real2sim._pipeline.world_frame import (
    build_world_frame,
    normalize_world_frame_mode,
)


def run_pipeline(
    image: str | Path | Image.Image | np.ndarray,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    *,
    depth: np.ndarray | None = None,
    depth_metadata: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the full real2sim pipeline and write a MuJoCo scene bundle."""

    config = config or PipelineConfig()
    config.world_frame_mode = normalize_world_frame_mode(config.world_frame_mode)
    config.support_confidence = config.support_confidence.strip().lower()
    output_dir = Path(output_dir)
    meshes_dir = output_dir / "meshes"
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []

    original_image = as_rgb_image(image)
    original_width, original_height = original_image.size
    rgb_image, resize_metadata = resize_image_for_processing(
        original_image,
        config.input_max_size,
    )
    if resize_metadata["resized"]:
        original_image.save(output_dir / "input_original.png")
        if config.intrinsics is not None:
            config.intrinsics = scale_intrinsics_for_resize(
                config.intrinsics,
                resize_metadata["scale_x"],
                resize_metadata["scale_y"],
            )
        if config.focal is not None:
            config.focal = float(config.focal) * 0.5 * (
                float(resize_metadata["scale_x"]) + float(resize_metadata["scale_y"])
            )
    rgb_image.save(output_dir / "input.png")
    width, height = rgb_image.size
    if resize_metadata["resized"]:
        log(
            config,
            "[real2sim] input image: "
            f"{original_width}x{original_height} -> {width}x{height} "
            f"(max side {config.input_max_size})",
        )
    else:
        log(config, f"[real2sim] input image: {width}x{height}")
    log(config, f"[real2sim] output directory: {output_dir}")
    log(config, f"[real2sim] MuJoCo export: {export_mode_label(config)}")

    provided_depth = depth is not None
    if provided_depth:
        log(config, "[real2sim] loading supplied RGB-D depth/camera parameters...")
        depth, depth_metadata = _prepare_provided_depth(
            depth,
            depth_metadata,
            original_size=(original_width, original_height),
            processing_size=(width, height),
            resize_metadata=resize_metadata,
        )
        if config.intrinsics is not None:
            depth_metadata["intrinsics"] = np.asarray(
                config.intrinsics, dtype=np.float64
            )
            depth_metadata["intrinsics_source"] = "command_line_override"
        if config.extrinsics is not None:
            depth_metadata["extrinsics"] = np.asarray(
                config.extrinsics, dtype=np.float64
            )
            depth_metadata["extrinsics_convention"] = "opencv_w2c"
            depth_metadata["extrinsics_source"] = "command_line_override"
    else:
        log(config, "[real2sim] requesting Depth Anything depth/camera estimates...")
        try:
            depth, depth_metadata = request_depth(rgb_image, config, config.use_ray_pose)
        except Exception as exc:
            if not (config.use_ray_pose and config.retry_depth_without_ray_pose):
                raise
            warnings.append(
                "Depth Anything ray-pose request failed; retried without ray pose: "
                f"{type(exc).__name__}: {exc}"
            )
            depth, depth_metadata = request_depth(rgb_image, config, False)
            depth_metadata = dict(depth_metadata)
            depth_metadata["ray_pose_retry_error"] = f"{type(exc).__name__}: {exc}"
            depth_metadata["ray_pose_retry_used"] = True

    K, intrinsics_source = camera_intrinsics(depth_metadata, width, height, warnings)
    c2w_opencv, w2c_opencv, extrinsics_source = camera_extrinsics(
        depth_metadata, warnings
    )
    log_depth_camera(config, depth, K, c2w_opencv, intrinsics_source, extrinsics_source)

    log(config, "[real2sim] building camera point map and PLY point cloud...")
    pointmap_opencv = depth_to_pointmap(depth, K)
    pointmap_p3d = transform_pointmap(pointmap_opencv, T_OPENCV_TO_P3D)
    image_np = np.asarray(rgb_image, dtype=np.uint8)

    camera_pcd_path = output_dir / "scene_camera_opencv.ply"
    camera_pcd_points = save_colored_pointcloud_ply(
        camera_pcd_path,
        pointmap_opencv,
        image_np,
        stride=config.pointcloud_stride,
    )
    log(
        config,
        f"[real2sim] saved camera point cloud: "
        f"{camera_pcd_path.name} ({camera_pcd_points} pts)",
    )

    log(config, "[real2sim] querying VLM object detector...")
    try:
        detected_objects, vlm_response = detect_objects(
            rgb_image,
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
            accepted_confidences=config.accepted_confidences,
            max_objects=config.max_objects,
        )
    except VLMResponseError as exc:
        failed_path = write_single_vlm_response(
            output_dir / "vlm_responses",
            "object_detection_failed.txt",
            exc.response,
        )
        if failed_path is not None:
            log(config, f"[real2sim] saved failed VLM response: {failed_path}")
        raise
    log(config, f"[real2sim] detected {len(detected_objects)} accepted objects")
    for i, obj in enumerate(detected_objects, start=1):
        log(
            config,
            "  "
            f"{i}. {obj.get('name')} "
            f"({obj.get('confidence')}) - {obj.get('description', '')}",
        )

    world_from_camera_opencv, world_frame_metadata, support_mask = build_world_frame(
        image=rgb_image,
        detected_objects=detected_objects,
        pointmap_opencv=pointmap_opencv,
        depth_c2w_opencv=c2w_opencv,
        depth_w2c_opencv=w2c_opencv,
        depth_extrinsics_source=extrinsics_source,
        masks_dir=masks_dir,
        config=config,
        warnings=warnings,
    )
    w2c_world_opencv = np.linalg.inv(world_from_camera_opencv)
    support_floor_z = world_frame_metadata.get("support_floor_z")
    if support_floor_z is not None:
        config.support_floor_z = float(support_floor_z)
    log_world_frame(config, world_frame_metadata)

    pointmap_world = transform_pointmap(pointmap_opencv, world_from_camera_opencv)
    world_pcd_path = output_dir / "scene_world.ply"
    world_pcd_points = save_colored_pointcloud_ply(
        world_pcd_path,
        pointmap_world,
        image_np,
        stride=config.pointcloud_stride,
    )
    log(
        config,
        f"[real2sim] saved world point cloud: "
        f"{world_pcd_path.name} ({world_pcd_points} pts)",
    )

    support_camera_pcd_path = None
    support_world_pcd_path = None
    support_camera_pcd_points = 0
    support_world_pcd_points = 0
    if support_mask is not None:
        support_pointmap_opencv = masked_pointmap(pointmap_opencv, support_mask)
        support_pointmap_world = masked_pointmap(pointmap_world, support_mask)
        support_camera_pcd_path = output_dir / "support_surface_camera_opencv.ply"
        support_world_pcd_path = output_dir / "support_surface_world.ply"
        support_camera_pcd_points = save_colored_pointcloud_ply(
            support_camera_pcd_path,
            support_pointmap_opencv,
            image_np,
            stride=config.pointcloud_stride,
        )
        support_world_pcd_points = save_colored_pointcloud_ply(
            support_world_pcd_path,
            support_pointmap_world,
            image_np,
            stride=config.pointcloud_stride,
        )
        log(
            config,
            "[real2sim] saved support point clouds: "
            f"{support_camera_pcd_path.name} ({support_camera_pcd_points} pts), "
            f"{support_world_pcd_path.name} ({support_world_pcd_points} pts)",
        )

    depth_camera_npz = output_dir / "depth_camera.npz"
    depth_camera_arrays = {
        "depth": depth.astype(np.float32),
        "intrinsics": K.astype(np.float32),
        "c2w_opencv": world_from_camera_opencv.astype(np.float32),
        "w2c_opencv": w2c_world_opencv.astype(np.float32),
        "camera_input_c2w_opencv": c2w_opencv.astype(np.float32),
        "camera_input_w2c_opencv": w2c_opencv.astype(np.float32),
        "pointmap_camera_opencv": pointmap_opencv.astype(np.float32),
        "pointmap_camera_p3d": pointmap_p3d.astype(np.float32),
        "pointmap_world": pointmap_world.astype(np.float32),
        "support_mask": (
            np.zeros(depth.shape, dtype=np.uint8)
            if support_mask is None
            else support_mask.astype(np.uint8)
        ),
    }
    if not provided_depth:
        depth_camera_arrays["depth_anything_c2w_opencv"] = c2w_opencv.astype(
            np.float32
        )
        depth_camera_arrays["depth_anything_w2c_opencv"] = w2c_opencv.astype(
            np.float32
        )
    depth_conf = depth_metadata.get("conf")
    if isinstance(depth_conf, np.ndarray):
        depth_camera_arrays["depth_confidence"] = depth_conf.astype(np.float32)
        if not provided_depth:
            depth_camera_arrays["depth_anything_conf"] = depth_conf.astype(np.float32)
    np.savez_compressed(depth_camera_npz, **depth_camera_arrays)

    static_selections, static_vlm_response = select_static_objects_for_scene(
        image=rgb_image,
        detected_objects=detected_objects,
        world_frame_metadata=world_frame_metadata,
        config=config,
        warnings=warnings,
    )
    detected_objects = apply_static_selections(detected_objects, static_selections)
    log_static_selections(config, static_selections)

    physical_parameters, physical_vlm_response = (
        select_physical_parameters_for_scene(
            image=rgb_image,
            detected_objects=detected_objects,
            static_selections=static_selections,
            world_frame_metadata=world_frame_metadata,
            config=config,
            warnings=warnings,
        )
    )
    detected_objects = apply_physical_parameters(detected_objects, physical_parameters)
    config.support_surface_physics = dict(
        physical_parameters.get("support_surface") or {}
    )
    log_physical_parameters(config, physical_parameters)

    reconstructed, reconstruction_skipped = reconstruct_scene_objects(
        image=rgb_image,
        detected_objects=detected_objects,
        K=K,
        color=image_np,
        depth=depth,
        T_world_from_camera_opencv=world_from_camera_opencv,
        pointmap_p3d=pointmap_p3d,
        meshes_dir=meshes_dir,
        masks_dir=masks_dir,
        world_frame_metadata=world_frame_metadata,
        config=config,
    )
    skipped.extend(reconstruction_skipped)

    scene_xml = output_dir / "scene.xml"
    log(config, f"[real2sim] writing MuJoCo scene: {scene_xml}")
    write_mujoco_scene(scene_xml, reconstructed, config)

    physics_preflight = _validate_scene_with_agent_fallback(
        scene_xml=scene_xml,
        reconstructed=reconstructed,
        config=config,
    )

    vlm_response_files = write_vlm_response_files(
        output_dir=output_dir,
        object_detection_response=vlm_response,
        world_frame_metadata=world_frame_metadata,
        static_object_response=static_vlm_response,
        static_selections=static_selections,
        physical_parameter_response=physical_vlm_response,
        physical_parameters=physical_parameters,
        reconstructed=reconstructed,
        skipped=skipped,
    )
    if vlm_response_files["files"]:
        log(
            config,
            "[real2sim] saved VLM responses: "
            f"{vlm_response_files['dir']} ({len(vlm_response_files['files'])} files)",
        )

    metadata = {
        "pipeline": "real2sim.single_image",
        "created_at_unix": started_at,
        "elapsed_seconds": time.time() - started_at,
        "image": {
            "width": width,
            "height": height,
            "processing_width": width,
            "processing_height": height,
            "original_width": original_width,
            "original_height": original_height,
            "input_max_size": config.input_max_size,
            **resize_metadata,
        },
        "servers": {
            "depth_anything": None if provided_depth else config.depth_server_url,
            "sam3": config.sam3_server_url,
            "sam3d": config.sam3d_server_url,
            "sam3d_all": sam3d_urls_for_reconstruction(config),
            "foundation_pose": config.foundation_pose_url,
        },
        "reconstruction": {
            "workers": max(1, int(config.reconstruction_workers)),
            "sam3d_urls": sam3d_urls_for_reconstruction(config),
            "target_faces": config.target_faces,
            "visual_color_clusters": config.visual_color_clusters,
            "reject_border_truncated_objects": (
                config.reject_border_truncated_objects
            ),
            "border_truncation_margin_fraction": (
                config.border_truncation_margin_fraction
            ),
            "grounding_bbox_margin_fraction": (
                config.grounding_bbox_margin_fraction
            ),
            "min_grounding_mask_bbox_fill_ratio": (
                config.min_grounding_mask_bbox_fill_ratio
            ),
            "duplicate_mask_iou_threshold": (
                config.duplicate_mask_iou_threshold
            ),
            "duplicate_mask_containment_threshold": (
                config.duplicate_mask_containment_threshold
            ),
        },
        "mujoco": {
            "dynamic_objects": config.dynamic_objects,
            "object_mass": config.object_mass,
            "object_support_clearance": config.object_support_clearance,
            "icp_fallback_support_correction_meters": (
                config.icp_fallback_support_correction_meters
            ),
            "icp_min_fitness": config.icp_min_fitness,
            "icp_max_inlier_rmse_meters": config.icp_max_inlier_rmse_meters,
            "icp_max_translation_meters": config.icp_max_translation_meters,
            "icp_max_rotation_degrees": config.icp_max_rotation_degrees,
            "max_support_lift_meters": config.max_support_lift_meters,
            "max_initial_object_penetration_meters": (
                config.max_initial_object_penetration_meters
            ),
            "physics_preflight": physics_preflight,
            "scene_repair_agent": {
                "enabled": config.use_scene_repair_agent,
                "model": config.scene_repair_agent_model,
                "reasoning_effort": config.scene_repair_agent_reasoning_effort,
                "max_turns": config.scene_repair_agent_max_turns,
                "used": (output_dir / "agent_repair" / "report.json").is_file(),
                "report": (
                    "agent_repair/report.json"
                    if (output_dir / "agent_repair" / "report.json").is_file()
                    else None
                ),
            },
            "use_physical_parameter_selection": config.use_physical_parameter_selection,
            "physical_confidence": config.physical_confidence,
            "support_surface_physics": config.support_surface_physics,
            "support_floor_z": config.support_floor_z,
            "reconstruct_support_surface": config.reconstruct_support_surface,
            "use_static_object_selection": config.use_static_object_selection,
            "static_confidence": config.static_confidence,
            "use_grounding_alias_fallback": config.use_grounding_alias_fallback,
            "max_grounding_aliases": config.max_grounding_aliases,
            "collision_geometry": "3d_oriented_bounding_box",
        },
        "foundation_pose": {
            "enabled": config.use_foundation_pose,
            "fallback_without_foundation_pose": config.fallback_without_foundation_pose,
            "target_faces": config.foundation_pose_target_faces,
            "input_mode": "full_frame",
        },
        "depth_input": jsonify(depth_metadata),
        "depth_anything": None if provided_depth else jsonify(depth_metadata),
        "camera": {
            "intrinsics": K,
            "intrinsics_source": intrinsics_source,
            "world_frame_mode": config.world_frame_mode,
            "T_world_from_camera_opencv": world_from_camera_opencv,
            "c2w_opencv": world_from_camera_opencv,
            "w2c_opencv": w2c_world_opencv,
            "extrinsics_source": world_frame_metadata.get(
                "source", config.world_frame_mode
            ),
            "camera_input_c2w_opencv": c2w_opencv,
            "camera_input_w2c_opencv": w2c_opencv,
            "camera_input_extrinsics_source": extrinsics_source,
            **(
                {}
                if provided_depth
                else {
                    "depth_anything_c2w_opencv": c2w_opencv,
                    "depth_anything_w2c_opencv": w2c_opencv,
                    "depth_anything_extrinsics_source": extrinsics_source,
                }
            ),
            "pointmap_convention": {
                "opencv": "x-right, y-down, z-forward",
                "pytorch3d": "x-left, y-up, z-forward",
                "opengl": "x-right, y-up, z-backward",
                "T_opencv_to_p3d": T_OPENCV_TO_P3D,
                "T_opengl_to_opencv": T_OPENGL_TO_OPENCV,
            },
        },
        "world_frame": world_frame_metadata,
        "vlm": {
            "model": config.vlm_model,
            "reasoning_effort": config.vlm_reasoning_effort,
            "raw_response": vlm_response,
            "detected_objects": detected_objects,
            "accepted_confidences": config.accepted_confidences,
            "static_object_raw_response": static_vlm_response,
            "static_object_selections": static_selections,
            "physical_parameter_raw_response": physical_vlm_response,
            "physical_parameter_selection": physical_parameters,
            "response_files": vlm_response_files,
        },
        "objects": reconstructed,
        "skipped_objects": skipped,
        "warnings": warnings,
        "outputs": {
            "scene_xml": scene_xml.name,
            "depth_camera_npz": depth_camera_npz.name,
            "scene_camera_opencv_ply": camera_pcd_path.name,
            "scene_camera_opencv_points": camera_pcd_points,
            "scene_world_ply": world_pcd_path.name,
            "scene_world_points": world_pcd_points,
            "scene_world_transform": (
                "support_surface_world_from_camera_opencv @ pointmap_camera_opencv"
                if config.world_frame_mode == "support_surface"
                else "T_world_from_camera_opencv @ pointmap_camera_opencv"
            ),
            "support_surface_camera_opencv_ply": (
                None if support_camera_pcd_path is None else support_camera_pcd_path.name
            ),
            "support_surface_camera_opencv_points": support_camera_pcd_points,
            "support_surface_world_ply": (
                None if support_world_pcd_path is None else support_world_pcd_path.name
            ),
            "support_surface_world_points": support_world_pcd_points,
            "pointcloud_stride": config.pointcloud_stride,
            "meshes_dir": meshes_dir.name,
            "masks_dir": masks_dir.name,
            "vlm_responses_dir": vlm_response_files["dir"],
        },
    }

    metadata = compact_pipeline_metadata(metadata)

    metadata_json = output_dir / "metadata.json"
    metadata_json.write_text(
        json.dumps(jsonify(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (output_dir / "agent_repair" / "report.json").is_file():
        from real2sim.package_scene import _has_validated_scene_repair
        from real2sim.scene_repair import synchronize_scene_metadata

        if _has_validated_scene_repair(output_dir):
            metadata = synchronize_scene_metadata(output_dir)
    log(config, f"[real2sim] wrote metadata: {metadata_json}")
    log(
        config,
        f"[real2sim] done: {len(metadata['objects'])} objects, "
        f"{len(skipped)} skipped, {len(warnings)} warnings",
    )

    return PipelineResult(
        output_dir=output_dir,
        scene_xml=scene_xml,
        metadata_json=metadata_json,
        depth_camera_npz=depth_camera_npz,
        objects=reconstructed,
        skipped_objects=skipped,
        warnings=warnings,
        scene_object_names=list(physics_preflight.get(
            "validated_object_safe_names", [obj.safe_name for obj in reconstructed],
        )),
    )


def _validate_scene_with_agent_fallback(
    *,
    scene_xml: Path,
    reconstructed: list[Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Validate reset contacts and settling stability, repairing failures."""

    if not config.compile_check:
        return {
            "status": "skipped",
            "reason": "compile check disabled",
            "threshold_meters": config.max_initial_object_penetration_meters,
        }

    log(config, "[real2sim] compiling MuJoCo scene for validation...")
    safe_names = [obj.safe_name for obj in reconstructed]
    from real2sim.scene_repair import (
        SceneHealthError,
        repair_scene_with_agent,
        validate_repaired_scene,
    )

    try:
        return validate_repaired_scene(
            scene_xml.parent,
            object_safe_names=safe_names,
            max_initial_object_penetration_meters=(
                config.max_initial_object_penetration_meters
            ),
        )
    except (InitialObjectPenetrationError, SceneHealthError) as exc:
        output_dir = scene_xml.parent
        failure_path = output_dir / "physics_preflight.json"
        failure_path.write_text(
            json.dumps(jsonify(exc.report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not config.use_scene_repair_agent:
            raise
        log(
            config,
            "[real2sim] validation failed; invoking scene-repair agent fallback...",
        )
        repair_source_image = (
            output_dir / "input_original.png"
            if (output_dir / "input_original.png").is_file()
            else output_dir / "input.png"
        )
        return repair_scene_with_agent(
            output_dir,
            source_image=repair_source_image,
            error=exc,
            object_safe_names=safe_names,
            max_initial_object_penetration_meters=(
                config.max_initial_object_penetration_meters
            ),
            model=config.scene_repair_agent_model,
            reasoning_effort=config.scene_repair_agent_reasoning_effort,
            max_turns=config.scene_repair_agent_max_turns,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    from real2sim.cli import build_arg_parser as _build_arg_parser

    return _build_arg_parser()


def compact_pipeline_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove bulky/redundant runtime payloads from persisted metadata.json."""

    compact = dict(metadata)
    compact["depth_input"] = _compact_depth_metadata(compact.get("depth_input"))
    compact["depth_anything"] = _compact_depth_anything_metadata(
        compact.get("depth_anything")
    )
    compact["world_frame"] = _compact_world_frame_metadata(compact.get("world_frame"))
    compact["vlm"] = _compact_vlm_metadata(compact.get("vlm"))
    return compact


def _compact_depth_anything_metadata(value: Any) -> Any:
    return _compact_depth_metadata(value)


def _compact_depth_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    duplicate_or_dense = {"conf", "intrinsics", "extrinsics"}
    return {
        key: item
        for key, item in value.items()
        if key not in duplicate_or_dense and not _is_large_array(item)
    }


def _prepare_provided_depth(
    depth: np.ndarray,
    metadata: dict[str, Any] | None,
    *,
    original_size: tuple[int, int],
    processing_size: tuple[int, int],
    resize_metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and resize registered supplied depth and intrinsics together."""

    depth_array = np.asarray(depth, dtype=np.float32)
    original_width, original_height = original_size
    if depth_array.shape != (original_height, original_width):
        raise ValueError(
            "supplied depth shape "
            f"{depth_array.shape} must match input image (H, W)="
            f"{(original_height, original_width)}"
        )
    if not np.any(np.isfinite(depth_array) & (depth_array > 0.0)):
        raise ValueError("supplied depth contains no finite positive values")

    camera_metadata = dict(metadata or {})
    if resize_metadata["resized"]:
        resampling = getattr(Image, "Resampling", Image)
        depth_image = Image.fromarray(depth_array, mode="F")
        depth_array = np.asarray(
            depth_image.resize(processing_size, resampling.NEAREST),
            dtype=np.float32,
        )
        if "intrinsics" in camera_metadata:
            camera_metadata["intrinsics"] = scale_intrinsics_for_resize(
                np.asarray(camera_metadata["intrinsics"], dtype=np.float64),
                resize_metadata["scale_x"],
                resize_metadata["scale_y"],
            )
    return depth_array, camera_metadata


def _compact_world_frame_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    duplicate_or_raw = {
        "camera_input_c2w_opencv",
        "camera_input_w2c_opencv",
        "depth_anything_c2w_opencv",
        "depth_anything_w2c_opencv",
        "support_vlm_raw_response",
        "support_vlm_error_responses",
    }
    return {
        key: item
        for key, item in value.items()
        if key not in duplicate_or_raw and not _is_large_array(item)
    }


def _compact_vlm_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    raw_responses = {
        "raw_response",
        "static_object_raw_response",
        "physical_parameter_raw_response",
    }
    return {key: item for key, item in value.items() if key not in raw_responses}


def _is_large_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.size > 16


def main(argv: list[str] | None = None) -> int:
    from real2sim.cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
