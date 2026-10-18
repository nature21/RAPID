"""Store large numeric diagnostics separately from readable metrics."""

from collections.abc import Mapping
from pathlib import Path
import os
import tempfile

import numpy as np


def externalize_metric_arrays(value, output: Path):
    arrays = {}

    def convert(item):
        candidate = item
        if isinstance(item, (tuple, list)) and item:
            try:
                candidate = np.asarray(item)
            except (ValueError, TypeError):
                candidate = item
        if isinstance(candidate, np.ndarray) and candidate.dtype.kind in "biufc" and candidate.size > 256:
            key = f"array_{len(arrays):04d}"
            arrays[key] = candidate
            return {"array_key": key, "shape": list(candidate.shape), "dtype": str(candidate.dtype)}
        if isinstance(item, Mapping):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(child) for child in item]
        return item

    result = convert(value)
    if not arrays:
        return result
    archive = output.with_suffix(".arrays.npz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=archive.parent, suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        Path(temporary).replace(archive)
    finally:
        Path(temporary).unlink(missing_ok=True)
    result["metric_arrays"] = archive.name
    return result
