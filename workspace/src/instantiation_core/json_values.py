"""Strict conversion and atomic writing for JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_json_value(value: Any, *, _path: str = "$") -> Any:
    """Recursively convert supported runtime values into JSON-native values.

    Frozen mappings are deliberately accepted because binding and
    generalization contracts use ``MappingProxyType`` internally. Unknown
    objects fail with their payload path instead of being silently stringified.
    """

    if isinstance(value, Enum):
        return to_json_value(value.value, _path=_path)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return to_json_value(value.item(), _path=_path)
    if isinstance(value, np.ndarray):
        return to_json_value(value.tolist(), _path=_path)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_json_value(
                getattr(value, item.name),
                _path=f"{_path}.{item.name}",
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_json_value(item, _path=f"{_path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            to_json_value(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [
            to_json_value(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    raise TypeError(
        f"unsupported JSON value at {_path}: {type(value).__name__}"
    )


def write_json_file(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    """Normalize and atomically replace one UTF-8 JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        to_json_value(value),
        indent=indent,
        sort_keys=sort_keys,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["to_json_value", "write_json_file"]
