"""Shared helpers for network-backed real2sim stages."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def stage_log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def clamp_float(value: float | None, min_value: float, max_value: float) -> float:
    if value is None:
        value = min_value
    return float(np.clip(float(value), min_value, max_value))


def canonical_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1).tolist()
