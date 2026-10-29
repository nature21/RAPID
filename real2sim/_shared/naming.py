"""Name normalization helpers for real2sim outputs."""

from __future__ import annotations

import re


def safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower()).strip("_")
    return safe or "response"


def unique_file_stem(stem: str, used: set[str]) -> str:
    stem = safe_file_stem(stem)
    unique = stem
    index = 2
    while unique in used:
        unique = f"{stem}_{index}"
        index += 1
    used.add(unique)
    return unique


def safe_identifier(name: str, used: set[str]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not safe:
        safe = "object"
    if safe[0].isdigit():
        safe = f"object_{safe}"
    base = safe
    i = 2
    while safe in used:
        safe = f"{base}_{i}"
        i += 1
    used.add(safe)
    return safe


def same_object_name(left: str, right: str) -> bool:
    return canonical_object_name(left) == canonical_object_name(right)


def canonical_object_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
