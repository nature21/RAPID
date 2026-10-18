"""Curate HOPE visuals into deterministic, simulation-friendly assets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .specs import ALL_APPEARANCES, BENCHMARK_VERSION, DISPLAY_NAMES, safe_appearance_name


TEXTURE_MAX_SIDE = 512
# HOPE source meshes use centimeter-scale dimensions.  The curated meshes are
# normalized per axis for stable storage, so generation must restore this scale
# to avoid stretching every product into the reference collision box.
HOPE_SOURCE_UNIT_METERS = 0.01
# Apply one catalog-wide world-scale correction to every reusable HOPE asset.
# The same product therefore keeps its physical size whether it is used as a
# target, task context, or distractor.
HOPE_REUSABLE_ASSET_SCALE = 1.5
ASSET_UNIFORM_SCALE_RANGE = (0.75, 1.25)
ASSET_ASPECT_JITTER = 0.03

# HOPE's Google-scanned models use +Y as their semantic up direction, while
# MuJoCo scenes use +Z.  Keep this transform explicit because normalized mesh
# bounds preserve the source axes and cannot recover semantic orientation.
HOPE_UPRIGHT_QUATERNION = np.asarray(
    (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
    dtype=np.float64,
)

ASSET_SCALE_RANGES = {
    "box_or_carton": (0.75, 1.25),
    "bottle": (0.80, 1.20),
    "can_or_cup": (0.85, 1.15),
}

ASSET_CATEGORIES = {
    **{
        name: "box_or_carton"
        for name in (
            "Butter",
            "ChocolatePudding",
            "Cookies",
            "CreamCheese",
            "GranolaBars",
            "MacaroniAndCheese",
            "Milk",
            "OrangeJuice",
            "Popcorn",
            "Raisins",
            "Spaghetti",
        )
    },
    **{
        name: "bottle"
        for name in (
            "BBQSauce",
            "Ketchup",
            "Mayo",
            "Mustard",
            "Parmesan",
            "SaladDressing",
        )
    },
    **{
        name: "can_or_cup"
        for name in (
            "AlphabetSoup",
            "Cherries",
            "Corn",
            "GreenBeans",
            "Mushrooms",
            "Peaches",
            "PeasAndCarrots",
            "Pineapple",
            "TomatoSauce",
            "Tuna",
            "Yogurt",
        )
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_obj(source: Path) -> tuple[str, list[float], list[float]]:
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    vertices: list[list[float]] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"HOPE OBJ contains no vertices: {source}")
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    if np.any(extent <= 0.0):
        raise ValueError(f"HOPE OBJ has a degenerate axis: {source}")
    center = 0.5 * (minimum + maximum)
    normalized = 2.0 * (points - center) / extent

    output: list[str] = [
        f"# Normalized for {BENCHMARK_VERSION}; vertices span [-1, 1] on every axis."
    ]
    vertex_index = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("mtllib ") or stripped.startswith("usemtl "):
            continue
        if line.startswith("v "):
            value = normalized[vertex_index]
            vertex_index += 1
            output.append("v " + " ".join(f"{item:.9g}" for item in value))
        else:
            output.append(line)
    return "\n".join(output) + "\n", extent.tolist(), center.tolist()


def curate_hope_assets(raw_root: Path, output_root: Path) -> dict[str, Any]:
    """Create the immutable normalized OBJ/texture pack used by generation."""

    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"HOPE asset root does not exist: {raw_root}")

    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for asset_name in ALL_APPEARANCES:
            source = raw_root / asset_name / "google_16k"
            obj_source = source / "textured_simple.obj"
            texture_source = source / "texture_map.png"
            if not obj_source.is_file() or not texture_source.is_file():
                raise FileNotFoundError(
                    f"HOPE asset {asset_name!r} is missing textured_simple.obj or texture_map.png"
                )
            safe_name = safe_appearance_name(asset_name)
            asset_dir = temporary / safe_name
            asset_dir.mkdir()
            obj_target = asset_dir / "visual.obj"
            texture_target = asset_dir / "texture.png"

            obj_text, source_extent, source_center = _normalized_obj(obj_source)
            obj_target.write_text(obj_text, encoding="utf-8")
            with Image.open(texture_source) as image:
                image = image.convert("RGB")
                image.thumbnail((TEXTURE_MAX_SIDE, TEXTURE_MAX_SIDE), Image.Resampling.LANCZOS)
                image.save(texture_target, format="PNG", optimize=True)

            records.append(
                {
                    "asset_name": asset_name,
                    "safe_name": safe_name,
                    "display_name": DISPLAY_NAMES[asset_name],
                    "source_extent": source_extent,
                    "source_center": source_center,
                    "obj": f"{safe_name}/visual.obj",
                    "texture": f"{safe_name}/texture.png",
                    "obj_sha256": _sha256(obj_target),
                    "texture_sha256": _sha256(texture_target),
                }
            )

        manifest = {
            "benchmark_version": BENCHMARK_VERSION,
            "normalization": "axis-aligned source bounds centered and scaled to [-1, 1]",
            "texture_max_side": TEXTURE_MAX_SIDE,
            "assets": records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_asset_manifest(asset_root: Path) -> dict[str, Any]:
    path = asset_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"curated HOPE manifest is missing: {path}; run the curate-assets command"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(
            f"curated assets use {payload.get('benchmark_version')!r}, expected {BENCHMARK_VERSION!r}"
        )
    return payload


def asset_record(asset_root: Path, asset_name: str) -> dict[str, Any]:
    manifest = load_asset_manifest(asset_root)
    for record in manifest.get("assets", []):
        if record.get("asset_name") == asset_name:
            return dict(record)
    raise KeyError(f"curated HOPE asset is not in manifest: {asset_name}")


def realistic_half_extents(
    record: dict[str, Any],
    size_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Restore a HOPE asset's proportions and keep scale variation plausible.

    The raw three-axis benchmark sample is decomposed into a bounded uniform
    scale and at most three percent of aspect-ratio variation.  This retains a
    meaningful size-generalization dimension without making branded products
    unrecognizable.
    """

    source_extent = np.asarray(record["source_extent"], dtype=np.float64)
    factors = np.asarray(size_scale_xyz, dtype=np.float64)
    if source_extent.shape != (3,) or np.any(source_extent <= 0.0):
        raise ValueError(f"invalid source extent: {record.get('source_extent')!r}")
    if factors.shape != (3,) or np.any(factors <= 0.0):
        raise ValueError(f"invalid size scale: {size_scale_xyz!r}")
    uniform = float(np.exp(np.mean(np.log(factors))))
    asset_name = str(record.get("asset_name", ""))
    try:
        scale_range = ASSET_SCALE_RANGES[ASSET_CATEGORIES[asset_name]]
    except KeyError as exc:
        raise KeyError(f"no scale category is mapped for {asset_name!r}") from exc
    uniform = float(np.clip(uniform, *scale_range))
    residual = factors / float(np.exp(np.mean(np.log(factors))))
    residual = np.clip(residual, 1.0 - ASSET_ASPECT_JITTER, 1.0 + ASSET_ASPECT_JITTER)
    return (
        0.5
        * HOPE_SOURCE_UNIT_METERS
        * HOPE_REUSABLE_ASSET_SCALE
        * source_extent
        * uniform
        * residual
    )


def hope_upright_quaternion() -> np.ndarray:
    """Rotate a HOPE model from source +Y-up coordinates into world +Z-up."""

    return HOPE_UPRIGHT_QUATERNION.copy()


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a MuJoCo wxyz quaternion."""

    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    return quat / np.linalg.norm(quat)


def asset_alignment_quaternion(
    source_half_extents: np.ndarray,
    reference_half_extents: np.ndarray,
) -> np.ndarray:
    """Map the asset's thin/middle/long axes onto the demonstrated OBB axes."""

    source_order = np.argsort(np.asarray(source_half_extents), kind="stable")
    reference_order = np.argsort(np.asarray(reference_half_extents), kind="stable")
    rotation = np.zeros((3, 3), dtype=np.float64)
    for source_axis, reference_axis in zip(source_order, reference_order):
        rotation[reference_axis, source_axis] = 1.0
    if np.linalg.det(rotation) < 0.0:
        rotation[:, int(source_order[0])] *= -1.0
    return matrix_quaternion(rotation)


__all__ = [
    "ASSET_ASPECT_JITTER",
    "ASSET_CATEGORIES",
    "ASSET_SCALE_RANGES",
    "ASSET_UNIFORM_SCALE_RANGE",
    "HOPE_SOURCE_UNIT_METERS",
    "HOPE_REUSABLE_ASSET_SCALE",
    "HOPE_UPRIGHT_QUATERNION",
    "TEXTURE_MAX_SIDE",
    "asset_alignment_quaternion",
    "asset_record",
    "curate_hope_assets",
    "hope_upright_quaternion",
    "load_asset_manifest",
    "matrix_quaternion",
    "realistic_half_extents",
]
