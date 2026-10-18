"""Helpers for user-supplied task environment module names."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable


def _flatten(values: Iterable[str | Iterable[str]]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if isinstance(value, str):
            flattened.append(value)
        else:
            flattened.extend(value)
    return flattened


def normalize_env_modules(values: Iterable[str | Iterable[str]] | None) -> list[str]:
    """Return normalized task scene modules from CLI values.

    Accept either an environment package name or its scene module name.
    Append ``.scene`` when the supplied name does not already end with that
    suffix. Ignore empty values and remove duplicates while preserving order.
    """
    if not values:
        return []

    modules: list[str] = []
    seen: set[str] = set()
    for raw_value in _flatten(values):
        module = raw_value.strip()
        if not module:
            continue
        if not module.endswith(".scene"):
            module = f"{module}.scene"
        if module not in seen:
            modules.append(module)
            seen.add(module)
    return modules


def validate_env_modules_importable(env_modules: Iterable[str]) -> list[str]:
    """Return diagnostics for normalized scene modules that cannot be imported."""
    findings: list[str] = []
    for module in env_modules:
        try:
            spec = importlib.util.find_spec(module)
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            findings.append(f"{module}: {exc}")
            continue
        if spec is None:
            findings.append(f"{module}: module not found")
    return findings


def format_env_modules_for_prompt(env_modules: list[str]) -> str:
    """Build the assigned task-environment facts for an invention prompt."""
    if not env_modules:
        return ""

    lines = "\n".join(
        f"{index}. `{module}`" for index, module in enumerate(env_modules, start=1)
    )
    plural = "s" if len(env_modules) != 1 else ""
    return f"""
## Assigned task environment{plural}

Assigned MuJoCo task scene module{plural}, in practice order:

{lines}

These modules are allowed task inputs copied into the isolated workspace by
the launcher.
"""
