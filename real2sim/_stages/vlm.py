"""VLM-backed scene understanding stages."""

from __future__ import annotations

from json import JSONDecodeError
import json
import re
from typing import Any

import numpy as np
from PIL import Image

from real2sim._stages.helpers import canonical_text, clamp_float, optional_float
from real2sim.vlm_clients.client import (
    PROMPT_DIR,
    load_prompts,
    parse_selection_results,
    query_model,
)


class VLMResponseError(ValueError):
    """VLM response failed parsing or validation, with raw response attached."""

    def __init__(self, stage: str, message: str, response: str):
        super().__init__(message)
        self.stage = stage
        self.response = response


def detect_objects(
    image: Image.Image,
    *,
    model: str,
    reasoning_effort: str | None = None,
    accepted_confidences: tuple[str, ...],
    max_objects: int | None,
) -> tuple[list[dict[str, Any]], str]:
    prompts = load_prompts(PROMPT_DIR)
    instruction = (
        "Analyze the attached image now and follow the system output protocol "
        "exactly. Return only one valid JSON list between literal *** delimiters. "
        "Do not use <result> tags, Markdown fences, analysis, or surrounding prose."
    )
    response = query_model(
        messages=[instruction],
        selected_model=model,
        reasoning_effort=reasoning_effort,
        images=[image],
        meta_prompt=prompts["object_detection"],
    )
    try:
        parsed = parse_selection_results(response)
        if not isinstance(parsed, list):
            raise ValueError(
                f"object_detection prompt returned {type(parsed).__name__}"
            )
    except Exception as exc:
        raise VLMResponseError(
            "object_detection",
            f"object_detection response failed validation: {type(exc).__name__}: {exc}",
            response,
        ) from exc

    accepted = {value.lower() for value in accepted_confidences}
    objects: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        confidence = str(item.get("confidence", "")).lower()
        if confidence not in accepted:
            continue
        if not str(item.get("name", "")).strip():
            continue
        detected = {
            "name": str(item["name"]).strip(),
            "description": str(item.get("description", "")).strip(),
            "confidence": confidence,
        }
        bbox_2d = _normalized_bbox_2d(item.get("bbox_2d"))
        if bbox_2d is not None:
            detected["bbox_2d"] = bbox_2d
        objects.append(detected)

    if max_objects is not None:
        objects = objects[:max_objects]

    return objects, response


def _normalized_bbox_2d(value: Any) -> list[float] | None:
    """Validate one optional 0--1000 image-space object box."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(coords)):
        return None
    x1, y1, x2, y2 = [min(1000.0, max(0.0, item)) for item in coords]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def select_support_surface(
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str | None = None,
    accepted_confidences: tuple[str, ...],
    retry: bool = False,
) -> tuple[dict[str, Any], str]:
    prompts = load_prompts(PROMPT_DIR)
    object_json = json.dumps(detected_objects, indent=2, sort_keys=True)
    instruction = (
        "Select the foundational ground/workspace surface from these detected "
        "objects. If an object rests on a raised platform which rests on a visible "
        "table, select the table so the platform remains elevated scene geometry. "
        "Return only the required JSON result block.\n\n"
        f"Detected objects:\n{object_json}"
    )
    if retry:
        instruction = (
            "Previous support-surface selection failed validation. Choose the broad, "
            "foundational visible ground/workspace surface. Prefer an underlying "
            "floor, table, counter, or fixed shelf over a raised platform, mat, tray, "
            "box, or stand resting on it. "
            "Return only the required JSON result block.\n\n"
            f"Detected objects:\n{object_json}"
        )

    response = query_model(
        messages=[instruction],
        selected_model=model,
        reasoning_effort=reasoning_effort,
        images=[image],
        meta_prompt=prompts["support_surface_selection"],
    )
    try:
        parsed = parse_selection_results(response)
        selection = _validate_support_selection(parsed, accepted_confidences)
    except Exception as exc:
        raise VLMResponseError(
            "support_surface_selection",
            "support_surface_selection response failed parsing/validation: "
            f"{type(exc).__name__}: {exc}",
            response,
        ) from exc
    return selection, response


def select_static_objects(
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str | None = None,
    accepted_confidences: tuple[str, ...],
    support_selection: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    prompts = load_prompts(PROMPT_DIR)
    payload = {
        "detected_objects": detected_objects,
        "support_selection": support_selection,
    }
    instruction = (
        "Classify each detected object as static or dynamic for MuJoCo export. "
        "Return only the required JSON result block.\n\n"
        f"Scene data:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )
    response = query_model(
        messages=[instruction],
        selected_model=model,
        reasoning_effort=reasoning_effort,
        images=[image],
        meta_prompt=prompts["static_object_selection"],
    )
    try:
        parsed = parse_selection_results(response)
        selections = _validate_static_selections(parsed, accepted_confidences)
    except Exception as exc:
        raise VLMResponseError(
            "static_object_selection",
            "static_object_selection response failed parsing/validation: "
            f"{type(exc).__name__}: {exc}",
            response,
        ) from exc
    return selections, response


def select_physical_parameters(
    image: Image.Image,
    detected_objects: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str | None = None,
    accepted_confidences: tuple[str, ...],
    support_selection: dict[str, Any] | None = None,
    static_selections: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    prompts = load_prompts(PROMPT_DIR)
    payload = {
        "detected_objects": detected_objects,
        "support_selection": support_selection,
        "static_selections": static_selections or [],
    }
    instruction = (
        "Estimate physical MuJoCo parameters for the detected scene objects. "
        "Return only the required JSON result block.\n\n"
        f"Scene data:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )
    response = query_model(
        messages=[instruction],
        selected_model=model,
        reasoning_effort=reasoning_effort,
        images=[image],
        meta_prompt=prompts["physical_parameter_selection"],
    )
    try:
        parsed = parse_selection_results(response)
        parameters = _validate_physical_parameters(parsed, accepted_confidences)
    except Exception as exc:
        raise VLMResponseError(
            "physical_parameter_selection",
            "physical_parameter_selection response failed parsing/validation: "
            f"{type(exc).__name__}: {exc}",
            response,
        ) from exc
    return parameters, response


def suggest_grounding_aliases(
    image: Image.Image,
    object_info: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None = None,
    max_aliases: int,
    detected_objects: list[dict[str, Any]] | None = None,
) -> tuple[list[str], str]:
    prompts = load_prompts(PROMPT_DIR)
    max_aliases = max(1, int(max_aliases))
    payload = {
        "target_object": object_info,
        "detected_objects": detected_objects,
    }
    instruction = _grounding_alias_instruction(payload, max_aliases=max_aliases)
    response = query_model(
        messages=[instruction],
        selected_model=model,
        reasoning_effort=reasoning_effort,
        images=[image],
        meta_prompt=prompts["object_grounding_aliases"],
    )
    try:
        parsed = _parse_grounding_alias_response(response)
        aliases = _validate_grounding_aliases(parsed, max_aliases)
    except Exception as first_exc:
        retry_instruction = _grounding_alias_retry_instruction(
            payload,
            max_aliases=max_aliases,
            error=first_exc,
            response=response,
        )
        try:
            retry_response = query_model(
                messages=[retry_instruction],
                selected_model=model,
                reasoning_effort=reasoning_effort,
                images=[image],
                meta_prompt=prompts["object_grounding_aliases"],
            )
        except Exception as retry_call_exc:
            aliases = _heuristic_grounding_aliases(object_info, max_aliases)
            retry_response = (
                "Retry grounding-alias request failed: "
                f"{type(retry_call_exc).__name__}: {retry_call_exc}"
            )
            if aliases:
                return aliases, _combine_alias_responses(response, retry_response)
            raise VLMResponseError(
                "object_grounding_aliases",
                "object_grounding_aliases retry failed and no heuristic aliases "
                f"were available: {type(retry_call_exc).__name__}: {retry_call_exc}",
                _combine_alias_responses(response, retry_response),
            ) from retry_call_exc
        combined_response = _combine_alias_responses(response, retry_response)
        try:
            parsed = _parse_grounding_alias_response(retry_response)
            aliases = _validate_grounding_aliases(parsed, max_aliases)
        except Exception as second_exc:
            aliases = _heuristic_grounding_aliases(object_info, max_aliases)
            if aliases:
                return aliases, combined_response
            raise VLMResponseError(
                "object_grounding_aliases",
                "object_grounding_aliases response failed parsing/validation: "
                f"initial {type(first_exc).__name__}: {first_exc}; "
                f"retry {type(second_exc).__name__}: {second_exc}",
                combined_response,
            ) from second_exc
        response = combined_response

    aliases = _merge_aliases(
        aliases,
        _heuristic_grounding_aliases(object_info, max_aliases),
        max_aliases,
    )
    return aliases, response


def _grounding_alias_instruction(
    payload: dict[str, Any],
    *,
    max_aliases: int,
) -> str:
    target = payload.get("target_object") or {}
    original_name = str(target.get("name", "")).strip()
    description = str(target.get("description", "")).strip()
    return (
        "SAM3 failed to ground the target object with the original text query "
        f"{original_name!r}. Generate stronger text prompts for the exact same "
        "visible target object.\n\n"
        "Use short noun phrases that combine visible appearance, geometry, "
        "material, orientation, location, and role. When nearby instances look "
        "similar, include left/right/front/back or relative location in every "
        "alias. Do not name neighboring objects as the target.\n\n"
        f"Original query: {original_name!r}\n"
        f"Target description: {description!r}\n\n"
        "Return nothing except exactly one result block with valid JSON. "
        "Use double quotes and this schema:\n\n"
        "***\n"
        "{\n"
        f"  \"aliases\": [\"short visual phrase\", \"another phrase\"]\n"
        "}\n"
        "***\n\n"
        f"Provide 1 to {max_aliases} aliases.\n\n"
        f"Grounding context:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _grounding_alias_retry_instruction(
    payload: dict[str, Any],
    *,
    max_aliases: int,
    error: Exception,
    response: str,
) -> str:
    clipped_response = str(response).strip()
    if len(clipped_response) > 1200:
        clipped_response = clipped_response[:1200] + "\n...[truncated]"
    return (
        "Your previous grounding-alias response could not be parsed: "
        f"{type(error).__name__}: {error}\n\n"
        "Reply with no prose and no Markdown fences. Return exactly this shape:\n\n"
        "***\n"
        "{\n"
        "  \"aliases\": [\n"
        "    \"small printed package at back left\",\n"
        "    \"narrow upright carton right of the red box\",\n"
        "    \"back-left retail carton\"\n"
        "  ]\n"
        "}\n"
        "***\n\n"
        f"Use 1 to {max_aliases} aliases for the actual target object in the "
        "image. Keep every alias a short SAM text query for that same object.\n\n"
        f"Grounding context:\n{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        f"Previous invalid response:\n{clipped_response}"
    )


def _parse_grounding_alias_response(response: str) -> Any:
    try:
        return parse_selection_results(response)
    except Exception as block_exc:
        parsed = _extract_json_value(response)
        if parsed is not None:
            return parsed
        aliases = _extract_aliases_from_text(response)
        if aliases:
            return {"aliases": aliases}
        raise block_exc


def _extract_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _extract_aliases_from_text(text: str) -> list[str]:
    aliases: list[str] = []
    for raw_line in str(text).splitlines():
        if not (
            raw_line.lstrip().startswith(("-", "*", "\"", "'", "`"))
            or re.match(r"\s*\d+[.)]", raw_line)
        ):
            continue
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        line = line.strip("\"'` ")
        if not line or ("alias" in line.casefold() and len(line.split()) <= 2):
            continue
        if 1 <= len(line.split()) <= 8:
            aliases.append(line)
    return aliases


def _combine_alias_responses(first: str, second: str) -> str:
    if not str(first).strip():
        return second
    if not str(second).strip():
        return first
    return (
        "Initial grounding-alias response:\n"
        f"{first.rstrip()}\n\n"
        "Retry grounding-alias response:\n"
        f"{second.rstrip()}"
    )


def _heuristic_grounding_aliases(
    object_info: dict[str, Any],
    max_aliases: int,
) -> list[str]:
    name = str(object_info.get("name", "")).strip()
    description = str(object_info.get("description", "")).strip()
    text = f"{name} {description}".casefold()
    words = set(re.findall(r"[a-z][a-z-]*", text))

    color = _first_matching_word(
        words,
        (
            "white",
            "black",
            "gray",
            "grey",
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "brown",
            "tan",
            "silver",
            "clear",
            "transparent",
        ),
    )
    orientation = _first_matching_word(
        words,
        ("slanted", "angled", "upright", "vertical", "horizontal", "long", "thin"),
    )
    location = _first_matching_word(
        words,
        ("left", "right", "front", "back", "rear", "center", "middle"),
    )

    candidates: list[str] = []
    if _contains_any(
        text,
        ("divider", "barrier", "panel", "board", "wall", "rail", "fence"),
    ):
        obstacle_terms = ("divider", "barrier", "panel", "board", "side wall", "rail")
        for term in obstacle_terms:
            candidates.append(_phrase(color, term))
        if orientation:
            candidates.extend(
                _phrase(color, orientation, term) for term in obstacle_terms[:4]
            )
        if location:
            candidates.extend(
                _phrase(location, color, term) for term in obstacle_terms[:4]
            )
        candidates.extend(
            [
                _phrase(color, "obstacle barrier"),
                _phrase(color, "divider board"),
                _phrase(color, "barrier board"),
            ]
        )
    elif _contains_any(text, ("box", "carton", "package", "container")):
        candidates.extend(
            [
                _phrase(color, "box"),
                _phrase(color, "rectangular box"),
                _phrase(color, "cardboard box"),
                _phrase(color, "package"),
                _phrase(location, color, "box"),
            ]
        )
    else:
        candidates.extend(
            [
                name,
                _phrase(color, name),
                _phrase(location, color, name),
                _phrase(orientation, color, name),
            ]
        )

    original = canonical_text(name)
    return [
        alias
        for alias in _merge_aliases([], candidates, max_aliases + 1)
        if canonical_text(alias) != original
    ][:max_aliases]


def _first_matching_word(words: set[str], options: tuple[str, ...]) -> str:
    for option in options:
        if option in words:
            return option
    return ""


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _phrase(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _merge_aliases(
    aliases: list[str],
    fallback_aliases: list[str],
    max_aliases: int,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for alias in [*aliases, *fallback_aliases]:
        text = str(alias).strip()
        canonical = canonical_text(text)
        if not text or not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(text)
        if len(out) >= max_aliases:
            break
    return out


def _validate_support_selection(
    parsed: Any,
    accepted_confidences: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError(
            f"support_surface_selection prompt returned {type(parsed).__name__}"
        )

    required = ("object_name", "sam_query", "confidence")
    missing = [key for key in required if not str(parsed.get(key, "")).strip()]
    if missing:
        raise ValueError(f"support_surface_selection missing fields: {missing}")

    confidence = str(parsed["confidence"]).strip().lower()
    accepted = {value.lower() for value in accepted_confidences}
    if confidence not in accepted:
        raise ValueError(
            f"support surface confidence {confidence!r} not in accepted {sorted(accepted)}"
        )

    return {
        "object_name": str(parsed["object_name"]).strip(),
        "sam_query": str(parsed["sam_query"]).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "confidence": confidence,
    }


def _validate_static_selections(
    parsed: Any,
    accepted_confidences: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(parsed, list):
        raise ValueError(
            f"static_object_selection prompt returned {type(parsed).__name__}"
        )

    accepted = {value.lower() for value in accepted_confidences}
    selections: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"static_object_selection item {index} is not an object")
        object_name = str(item.get("object_name", "")).strip()
        if not object_name:
            raise ValueError(f"static_object_selection item {index} missing object_name")
        if "is_static" not in item or not isinstance(item["is_static"], bool):
            raise ValueError(
                f"static_object_selection item {object_name!r} must include "
                "boolean is_static"
            )
        confidence = str(item.get("confidence", "")).strip().lower()
        if confidence not in accepted:
            raise ValueError(
                f"static selection confidence {confidence!r} for {object_name!r} "
                f"not in accepted {sorted(accepted)}"
            )
        selections.append(
            {
                "object_name": object_name,
                "is_static": bool(item["is_static"]),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": confidence,
            }
        )

    return selections


def _validate_physical_parameters(
    parsed: Any,
    accepted_confidences: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError(
            f"physical_parameter_selection prompt returned {type(parsed).__name__}"
        )

    accepted = {value.lower() for value in accepted_confidences}
    support = parsed.get("support_surface")
    support_params = (
        _validate_physical_parameter_item(
            support,
            accepted,
            require_mass=False,
            context="support_surface",
        )
        if isinstance(support, dict)
        else None
    )

    raw_objects = parsed.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("physical_parameter_selection missing list field 'objects'")

    objects: list[dict[str, Any]] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise ValueError(
                f"physical_parameter_selection object item {index} is not an object"
            )
        params = _validate_physical_parameter_item(
            item,
            accepted,
            require_mass=True,
            context=f"objects[{index}]",
        )
        if params is not None:
            objects.append(params)

    return {"support_surface": support_params, "objects": objects}


def _validate_physical_parameter_item(
    item: dict[str, Any],
    accepted_confidences: set[str],
    *,
    require_mass: bool,
    context: str,
) -> dict[str, Any] | None:
    object_name = str(item.get("object_name", "")).strip()
    if not object_name:
        raise ValueError(f"{context} missing object_name")

    confidence = str(item.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(
            f"physical parameter confidence {confidence!r} for {object_name!r} "
            "must be high, medium, or low"
        )
    if confidence not in accepted_confidences:
        return None

    mass_kg = optional_float(item.get("mass_kg"))
    if require_mass and mass_kg is not None:
        mass_kg = clamp_float(mass_kg, 0.001, 20.0)
    elif require_mass:
        mass_kg = None
    else:
        mass_kg = None if mass_kg is None else clamp_float(mass_kg, 0.001, 200.0)

    static_friction = clamp_float(
        optional_float(item.get("static_friction"), default=0.7),
        0.05,
        3.0,
    )
    dynamic_friction = clamp_float(
        optional_float(item.get("dynamic_friction"), default=static_friction),
        0.03,
        3.0,
    )
    torsional_friction = clamp_float(
        optional_float(item.get("torsional_friction"), default=0.01),
        0.0,
        0.2,
    )
    rolling_friction = clamp_float(
        optional_float(item.get("rolling_friction"), default=0.001),
        0.0,
        0.1,
    )
    restitution = clamp_float(
        optional_float(item.get("restitution"), default=0.05),
        0.0,
        1.0,
    )

    return {
        "object_name": object_name,
        "material": str(item.get("material", "")).strip(),
        "mass_kg": mass_kg,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "sliding_friction": dynamic_friction,
        "torsional_friction": torsional_friction,
        "rolling_friction": rolling_friction,
        "friction": [dynamic_friction, torsional_friction, rolling_friction],
        "restitution": restitution,
        "confidence": confidence,
        "reason": str(item.get("reason", "")).strip(),
    }


def _validate_grounding_aliases(parsed: Any, max_aliases: int) -> list[str]:
    if isinstance(parsed, list):
        aliases = parsed
    elif isinstance(parsed, str):
        aliases = [parsed]
    elif isinstance(parsed, dict):
        aliases = parsed.get("aliases")
    else:
        raise ValueError(
            f"object_grounding_aliases prompt returned {type(parsed).__name__}"
        )
    if not isinstance(aliases, list):
        raise ValueError("object_grounding_aliases missing list field 'aliases'")

    max_aliases = max(1, int(max_aliases))
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        text = str(alias).strip()
        canonical = canonical_text(text)
        if not text or not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(text)
        if len(out) >= max_aliases:
            break

    if not out:
        raise ValueError("object_grounding_aliases returned no usable aliases")
    return out
