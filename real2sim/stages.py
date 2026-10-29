"""Compatibility facade for network-backed real2sim stages."""

from __future__ import annotations

from real2sim._stages.perception import request_depth, segment_support_surface
from real2sim._stages.reconstruction import (
    place_reconstructed_geometry,
    reconstruct_object,
    reconstruct_object_geometry,
)
from real2sim._stages.vlm import (
    VLMResponseError,
    detect_objects,
    select_physical_parameters,
    select_static_objects,
    select_support_surface,
    suggest_grounding_aliases,
)

__all__ = [
    "VLMResponseError",
    "detect_objects",
    "place_reconstructed_geometry",
    "reconstruct_object",
    "reconstruct_object_geometry",
    "request_depth",
    "segment_support_surface",
    "select_physical_parameters",
    "select_static_objects",
    "select_support_surface",
    "suggest_grounding_aliases",
]
