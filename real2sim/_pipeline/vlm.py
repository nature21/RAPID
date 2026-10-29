"""VLM orchestration helpers used by the high-level pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real2sim._shared.naming import (
    canonical_object_name,
    safe_file_stem,
    unique_file_stem,
)
from real2sim.schema import PipelineConfig, ReconstructedObject, jsonify
from real2sim._stages.vlm import (
    VLMResponseError,
    select_physical_parameters,
    select_static_objects,
)


def write_vlm_response_files(
    *,
    output_dir: Path,
    object_detection_response: str,
    world_frame_metadata: dict[str, Any],
    static_object_response: str | None,
    static_selections: list[dict[str, Any]],
    physical_parameter_response: str | None,
    physical_parameters: dict[str, Any],
    reconstructed: list[ReconstructedObject],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    responses_dir = output_dir / "vlm_responses"
    entries: list[dict[str, Any]] = []

    def add_response(
        *,
        stage: str,
        filename: str,
        response: str | None,
        status: str = "ok",
        object_name: str = "",
        error: str = "",
    ) -> None:
        if not isinstance(response, str) or not response.strip():
            return
        path = write_single_vlm_response(responses_dir, filename, response)
        if path is None:
            return
        entry = {
            "stage": stage,
            "file": str(path.relative_to(output_dir)),
            "status": status,
        }
        if object_name:
            entry["object_name"] = object_name
        if error:
            entry["error"] = error
        entries.append(entry)

    add_response(
        stage="object_detection",
        filename="object_detection.txt",
        response=object_detection_response,
    )
    add_response(
        stage="support_surface_selection",
        filename="support_surface_selection.txt",
        response=world_frame_metadata.get("support_vlm_raw_response"),
    )
    for index, item in enumerate(
        world_frame_metadata.get("support_vlm_error_responses", []),
        start=1,
    ):
        if not isinstance(item, dict):
            continue
        attempt = safe_file_stem(str(item.get("attempt", f"attempt_{index}")))
        add_response(
            stage=str(item.get("stage", "support_surface_selection")),
            filename=f"support_surface_selection_{attempt}_failed.txt",
            response=item.get("response"),
            status="failed",
            error=str(item.get("error", "")),
        )

    static_status = "ok" if static_selections else "failed_or_empty"
    add_response(
        stage="static_object_selection",
        filename="static_object_selection.txt",
        response=static_object_response,
        status=static_status,
    )
    physical_status = (
        "ok"
        if physical_parameters.get("support_surface")
        or physical_parameters.get("objects")
        else "failed_or_empty"
    )
    add_response(
        stage="physical_parameter_selection",
        filename="physical_parameter_selection.txt",
        response=physical_parameter_response,
        status=physical_status,
    )

    used_grounding_files: set[str] = set()
    for obj in reconstructed:
        response = obj.grounding_alias_raw_response
        if not response:
            continue
        stem = unique_file_stem(
            f"object_grounding_aliases_{obj.safe_name}",
            used_grounding_files,
        )
        add_response(
            stage="object_grounding_aliases",
            filename=f"{stem}.txt",
            response=response,
            object_name=obj.name,
        )

    for item in skipped:
        if not isinstance(item, dict):
            continue
        response = item.get("grounding_alias_raw_response")
        if not response:
            continue
        safe_name = str(item.get("safe_name", "")).strip()
        if not safe_name:
            safe_name = safe_file_stem(str(item.get("name", "object")))
        stem = unique_file_stem(
            f"object_grounding_aliases_{safe_name}",
            used_grounding_files,
        )
        add_response(
            stage="object_grounding_aliases",
            filename=f"{stem}.txt",
            response=response,
            status="failed",
            object_name=str(item.get("name", "")),
        )

    manifest = {
        "dir": responses_dir.name,
        "files": entries,
    }
    if entries:
        responses_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = responses_dir / "manifest.json"
        manifest["manifest"] = str(manifest_path.relative_to(output_dir))
        manifest_path.write_text(
            json.dumps(jsonify(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def write_single_vlm_response(
    responses_dir: Path,
    filename: str,
    response: str,
) -> Path | None:
    if not isinstance(response, str) or not response.strip():
        return None
    responses_dir.mkdir(parents=True, exist_ok=True)
    path = responses_dir / filename
    path.write_text(response.rstrip() + "\n", encoding="utf-8")
    return path


def raw_vlm_response_from_error(exc: Exception) -> str:
    if isinstance(exc, VLMResponseError):
        return exc.response
    response = getattr(exc, "response", "")
    return response if isinstance(response, str) else ""


def select_static_objects_for_scene(
    *,
    image: Any,
    detected_objects: list[dict[str, Any]],
    world_frame_metadata: dict[str, Any],
    config: PipelineConfig,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if not config.use_static_object_selection:
        return [], None

    try:
        selections, response = select_static_objects(
            image,
            detected_objects,
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
            accepted_confidences=accepted_confidences(config.static_confidence),
            support_selection=world_frame_metadata.get("support_selection"),
        )
        return selections, response
    except Exception as exc:
        warnings.append(
            "Static-object VLM selection failed; exported all non-support objects "
            f"as dynamic: {type(exc).__name__}: {exc}"
        )
        raw_response = raw_vlm_response_from_error(exc)
        return [], raw_response or None


def select_physical_parameters_for_scene(
    *,
    image: Any,
    detected_objects: list[dict[str, Any]],
    static_selections: list[dict[str, Any]],
    world_frame_metadata: dict[str, Any],
    config: PipelineConfig,
    warnings: list[str],
) -> tuple[dict[str, Any], str | None]:
    if not config.use_physical_parameter_selection:
        return {"support_surface": None, "objects": []}, None

    try:
        parameters, response = select_physical_parameters(
            image,
            detected_objects,
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
            accepted_confidences=accepted_confidences(config.physical_confidence),
            support_selection=world_frame_metadata.get("support_selection"),
            static_selections=static_selections,
        )
        return parameters, response
    except Exception as exc:
        warnings.append(
            "Physical-parameter VLM selection failed; using default MuJoCo "
            f"contact and mass values: {type(exc).__name__}: {exc}"
        )
        raw_response = raw_vlm_response_from_error(exc)
        return {"support_surface": None, "objects": []}, raw_response or None


def apply_static_selections(
    detected_objects: list[dict[str, Any]],
    static_selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {
        canonical_object_name(selection["object_name"]): selection
        for selection in static_selections
    }
    enriched: list[dict[str, Any]] = []
    for obj in detected_objects:
        item = dict(obj)
        selection = by_name.get(canonical_object_name(str(obj.get("name", ""))))
        if selection is None:
            item["is_static"] = False
            item["static_reason"] = ""
            item["static_confidence"] = ""
        else:
            item["is_static"] = bool(selection["is_static"])
            item["static_reason"] = str(selection.get("reason", ""))
            item["static_confidence"] = str(selection.get("confidence", ""))
        enriched.append(item)
    return enriched


def apply_physical_parameters(
    detected_objects: list[dict[str, Any]],
    physical_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    selections = physical_parameters.get("objects", [])
    if not isinstance(selections, list):
        selections = []
    by_name = {
        canonical_object_name(str(selection.get("object_name", ""))): selection
        for selection in selections
        if isinstance(selection, dict)
    }
    enriched: list[dict[str, Any]] = []
    for obj in detected_objects:
        item = dict(obj)
        selection = by_name.get(canonical_object_name(str(obj.get("name", ""))))
        if selection is not None:
            item["physical_material"] = str(selection.get("material", ""))
            item["physical_mass_kg"] = selection.get("mass_kg")
            item["physical_sliding_friction"] = selection.get("sliding_friction")
            item["physical_torsional_friction"] = selection.get("torsional_friction")
            item["physical_rolling_friction"] = selection.get("rolling_friction")
            item["physical_restitution"] = selection.get("restitution")
            item["physical_confidence"] = str(selection.get("confidence", ""))
            item["physical_reason"] = str(selection.get("reason", ""))
        enriched.append(item)
    return enriched


def accepted_confidences(min_confidence: str) -> tuple[str, ...]:
    normalized = min_confidence.strip().lower()
    if normalized == "high":
        return ("high",)
    if normalized == "medium":
        return ("high", "medium")
    if normalized == "low":
        return ("high", "medium", "low")
    raise ValueError(
        "support_confidence must be one of 'high', 'medium', or 'low'; "
        f"got {min_confidence!r}"
    )
