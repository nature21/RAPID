"""Shared data contracts for real2sim pipeline outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from real2sim.vfm_clients.depth_anything import (
    DEFAULT_SERVER_URL as DEFAULT_DEPTH_SERVER_URL,
)
from real2sim.vfm_clients.foundation_pose import (
    DEFAULT_REGISTRATION_URL as DEFAULT_FP_REGISTRATION_URL,
)
from real2sim.vfm_clients.sam3 import DEFAULT_SERVER_URL as DEFAULT_SAM3_SERVER_URL
from real2sim.vfm_clients.sam3d import DEFAULT_SERVER_URL as DEFAULT_SAM3D_SERVER_URL


@dataclass(slots=True)
class PipelineConfig:
    """Runtime options for :func:`real2sim.pipeline.run_pipeline`."""

    depth_server_url: str = DEFAULT_DEPTH_SERVER_URL
    sam3_server_url: str = DEFAULT_SAM3_SERVER_URL
    sam3d_server_url: str = DEFAULT_SAM3D_SERVER_URL
    foundation_pose_url: str = DEFAULT_FP_REGISTRATION_URL
    vlm_model: str = "gpt-6-astra"
    vlm_reasoning_effort: str | None = "high"
    use_ray_pose: bool = True
    focal: float | None = None
    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    input_max_size: int | None = 720
    process_res: int | None = None
    process_res_method: str | None = None
    depth_timeout: float = 600.0
    retry_depth_without_ray_pose: bool = True
    sam3_confidence_threshold: float = 0.1
    accepted_confidences: tuple[str, ...] = ("high", "medium")
    min_mask_area_px: int = 64
    min_mask_area_fraction: float = 1e-5
    reject_border_truncated_objects: bool = True
    border_truncation_margin_fraction: float = 0.01
    max_objects: int | None = None
    target_faces: int | None = 10000
    world_frame_mode: str = "support_surface"
    support_confidence: str = "medium"
    support_plane_distance_threshold: float = 0.01
    support_plane_ransac_iterations: int = 300
    support_plane_max_points: int = 30000
    support_plane_min_inlier_ratio: float = 0.2
    support_plane_seed: int = 0
    allow_global_support_fallback: bool = True
    reconstruct_support_surface: bool = False
    use_static_object_selection: bool = True
    static_confidence: str = "medium"
    use_physical_parameter_selection: bool = True
    physical_confidence: str = "medium"
    use_grounding_alias_fallback: bool = True
    max_grounding_aliases: int = 6
    grounding_bbox_margin_fraction: float = 0.05
    min_grounding_mask_bbox_fill_ratio: float = 0.10
    duplicate_mask_iou_threshold: float = 0.85
    duplicate_mask_containment_threshold: float = 0.90
    reconstruction_workers: int = 4
    sam3d_server_urls: tuple[str, ...] = ()
    use_foundation_pose: bool = True
    fallback_without_foundation_pose: bool = True
    foundation_pose_target_faces: int | None = 10000
    pointcloud_stride: int = 1
    dynamic_objects: bool = True
    object_mass: float = 0.2
    object_support_clearance: float = 0.002
    icp_fallback_support_correction_meters: float = 0.05
    icp_min_fitness: float = 0.50
    icp_max_inlier_rmse_meters: float = 0.03
    icp_max_translation_meters: float = 0.15
    icp_max_rotation_degrees: float = 30.0
    max_support_lift_meters: float = 0.20
    max_initial_object_penetration_meters: float = 0.03
    default_sliding_friction: float = 0.7
    default_torsional_friction: float = 0.01
    default_rolling_friction: float = 0.001
    default_restitution: float = 0.05
    support_surface_physics: dict[str, Any] = field(default_factory=dict)
    support_floor_z: float | None = None
    visual_color_clusters: int = 8
    compile_check: bool = True
    use_scene_repair_agent: bool = False
    scene_repair_agent_model: str = "gpt-6-astra"
    scene_repair_agent_reasoning_effort: str = "high"
    scene_repair_agent_max_turns: int = 12
    continue_on_object_error: bool = True
    verbose: bool = True


@dataclass(slots=True)
class ReconstructedObject:
    name: str
    safe_name: str
    description: str
    confidence: str
    sam3_score: float
    sam3_bbox: list[float] | None
    mask_area_px: int
    mesh_file: str
    body_pos: list[float]
    local_bounds: list[list[float]]
    world_bounds: list[list[float]]
    box_size: list[float]
    sam3d_scale: list[float]
    sam3d_translation: list[float]
    sam3d_quat: list[float]
    obb_source_mesh_file: str = ""
    visual_mesh_files: list[dict[str, Any]] = field(default_factory=list)
    mesh_frame: str = ""
    pose_source: str = ""
    grounding_query: str = ""
    grounding_query_source: str = ""
    grounding_attempts: list[dict[str, Any]] = field(default_factory=list)
    grounding_aliases: list[str] = field(default_factory=list)
    grounding_alias_raw_response: str = ""
    grounding_conflicts: list[dict[str, Any]] = field(default_factory=list)
    is_static: bool = False
    static_reason: str = ""
    static_confidence: str = ""
    physical_material: str = ""
    physical_mass_kg: float | None = None
    physical_sliding_friction: float | None = None
    physical_torsional_friction: float | None = None
    physical_rolling_friction: float | None = None
    physical_restitution: float | None = None
    physical_confidence: str = ""
    physical_reason: str = ""
    T_mesh_world: list[list[float]] | None = None
    support_lift_meters: float = 0.0
    support_penetration_correction_meters: float = 0.0
    support_object_name: str = ""
    support_stack_lift_meters: float = 0.0
    world_min_z_before_lift: float | None = None
    world_min_z_after_lift: float | None = None
    foundation_pose_opencv: list[list[float]] | None = None
    foundation_pose_last: list[list[float]] | None = None
    foundation_pose_input: dict[str, Any] = field(default_factory=dict)
    foundation_pose_error: str = ""
    icp_registration: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    scene_xml: Path
    metadata_json: Path
    depth_camera_npz: Path
    objects: list[ReconstructedObject] = field(default_factory=list)
    skipped_objects: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Perception records above are historical; repair can change the inventory.
    scene_object_names: list[str] | None = None


def jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonify(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value
