"""Material-driven primitive object catalog for Real2Sim benchmark v3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .materials import MATERIALS, SimulationMaterial


# Target-eligible material primitives are reusable manipulable objects, not
# structural fixtures.  Keep one role-independent catalog scale so the same
# object has the same dimensions as a target, context object, or distractor.
REUSABLE_MANIPULABLE_PRIMITIVE_SCALE = 1.5


@dataclass(frozen=True)
class PrimitiveObjectPreset:
    key: str
    display_name: str
    shape: str
    material_key: str
    dimensions_m: tuple[float, float, float]
    construction: str
    solid_fraction: float
    target_eligible: bool

    @property
    def material(self) -> SimulationMaterial:
        return MATERIALS[self.material_key]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


PRIMITIVE_OBJECTS: Mapping[str, PrimitiveObjectPreset] = {
    "CardboardParcel": PrimitiveObjectPreset(
        "CardboardParcel",
        "corrugated cardboard parcel",
        "box",
        "corrugated_cardboard",
        (0.16, 0.11, 0.055),
        "hollow folded carton",
        0.65,
        True,
    ),
    "MattePlasticCase": PrimitiveObjectPreset(
        "MattePlasticCase",
        "matte plastic case",
        "box",
        "matte_abs",
        (0.14, 0.10, 0.060),
        "hollow molded shell",
        0.18,
        True,
    ),
    "BirchWoodBlock": PrimitiveObjectPreset(
        "BirchWoodBlock",
        "birch wood block",
        "box",
        "birch_wood",
        (0.15, 0.09, 0.045),
        "solid",
        1.0,
        True,
    ),
    "PowderCoatedMetalCase": PrimitiveObjectPreset(
        "PowderCoatedMetalCase",
        "powder-coated metal case",
        "box",
        "powder_coated_steel",
        (0.14, 0.095, 0.050),
        "hollow sheet-metal shell",
        0.08,
        True,
    ),
    "PorcelainTile": PrimitiveObjectPreset(
        "PorcelainTile",
        "glazed porcelain tile",
        "box",
        "glazed_porcelain",
        (0.14, 0.10, 0.025),
        "solid",
        1.0,
        True,
    ),
    "TemperedGlassBlock": PrimitiveObjectPreset(
        "TemperedGlassBlock",
        "tempered glass block",
        "box",
        "tempered_glass",
        (0.13, 0.09, 0.035),
        "partially hollow",
        0.55,
        True,
    ),
    "AluminumCanister": PrimitiveObjectPreset(
        "AluminumCanister",
        "brushed aluminum canister",
        "cylinder",
        "brushed_aluminum",
        (0.070, 0.070, 0.120),
        "hollow metal shell",
        0.12,
        False,
    ),
    "PorcelainCup": PrimitiveObjectPreset(
        "PorcelainCup",
        "glazed porcelain cup",
        "cylinder",
        "glazed_porcelain",
        (0.085, 0.085, 0.090),
        "hollow ceramic shell",
        0.25,
        False,
    ),
    "GlassJar": PrimitiveObjectPreset(
        "GlassJar",
        "tempered glass jar",
        "cylinder",
        "tempered_glass",
        (0.075, 0.075, 0.120),
        "hollow glass shell",
        0.18,
        False,
    ),
    "CastIronPuck": PrimitiveObjectPreset(
        "CastIronPuck",
        "cast iron puck",
        "cylinder",
        "cast_iron",
        (0.075, 0.075, 0.035),
        "solid",
        1.0,
        False,
    ),
    "HDPECylinder": PrimitiveObjectPreset(
        "HDPECylinder",
        "recycled HDPE cylinder",
        "cylinder",
        "recycled_hdpe",
        (0.080, 0.080, 0.100),
        "hollow molded shell",
        0.25,
        False,
    ),
    "LargeCardboardStorageBox": PrimitiveObjectPreset(
        "LargeCardboardStorageBox",
        "large corrugated cardboard storage box",
        "box",
        "corrugated_cardboard",
        (0.26, 0.40, 0.15),
        "hollow folded carton",
        0.20,
        False,
    ),
    "LargePlasticStorageBox": PrimitiveObjectPreset(
        "LargePlasticStorageBox",
        "large recycled plastic storage box",
        "box",
        "recycled_hdpe",
        (0.24, 0.38, 0.14),
        "hollow molded shell",
        0.12,
        False,
    ),
    "LargeWoodStorageCrate": PrimitiveObjectPreset(
        "LargeWoodStorageCrate",
        "large birch wood storage crate",
        "box",
        "birch_wood",
        (0.25, 0.39, 0.14),
        "hollow slatted crate",
        0.18,
        False,
    ),
    "LargeMetalStorageCase": PrimitiveObjectPreset(
        "LargeMetalStorageCase",
        "large powder-coated metal storage case",
        "box",
        "powder_coated_steel",
        (0.23, 0.37, 0.12),
        "hollow sheet-metal shell",
        0.06,
        False,
    ),
    "MediumCardboardBoundary": PrimitiveObjectPreset(
        "MediumCardboardBoundary",
        "medium corrugated cardboard boundary box",
        "box",
        "corrugated_cardboard",
        (0.20, 0.22, 0.105),
        "hollow folded carton",
        0.30,
        False,
    ),
    "MediumPlasticBoundary": PrimitiveObjectPreset(
        "MediumPlasticBoundary",
        "medium recycled plastic boundary box",
        "box",
        "recycled_hdpe",
        (0.19, 0.23, 0.10),
        "hollow molded shell",
        0.16,
        False,
    ),
    "MediumWoodBoundary": PrimitiveObjectPreset(
        "MediumWoodBoundary",
        "medium birch wood boundary box",
        "box",
        "birch_wood",
        (0.21, 0.21, 0.11),
        "hollow slatted crate",
        0.22,
        False,
    ),
    "MediumMetalBoundary": PrimitiveObjectPreset(
        "MediumMetalBoundary",
        "medium powder-coated metal boundary box",
        "box",
        "powder_coated_steel",
        (0.19, 0.22, 0.095),
        "hollow sheet-metal shell",
        0.08,
        False,
    ),
}


STRATEGY_1_LARGE_OBSTACLE_KEYS = (
    "LargeCardboardStorageBox",
    "LargePlasticStorageBox",
    "LargeWoodStorageCrate",
    "LargeMetalStorageCase",
)
STRUCTURAL_CONTEXT_BOX_KEYS = (
    "MediumCardboardBoundary",
    "MediumPlasticBoundary",
    "MediumWoodBoundary",
    "MediumMetalBoundary",
)
PRIMITIVE_TARGET_KEYS = tuple(
    key for key, preset in PRIMITIVE_OBJECTS.items() if preset.target_eligible
)
PRIMITIVE_DISTRACTOR_KEYS = tuple(
    key
    for key in PRIMITIVE_OBJECTS
    if key not in STRATEGY_1_LARGE_OBSTACLE_KEYS
    and key not in STRUCTURAL_CONTEXT_BOX_KEYS
)


def is_primitive_object(identifier: str) -> bool:
    return identifier in PRIMITIVE_OBJECTS


def primitive_preset(identifier: str) -> PrimitiveObjectPreset:
    try:
        return PRIMITIVE_OBJECTS[identifier]
    except KeyError as exc:
        raise KeyError(f"unknown primitive object preset: {identifier!r}") from exc


def primitive_half_extents(
    preset: PrimitiveObjectPreset,
    size_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    broad: bool,
) -> np.ndarray:
    """Return primitive bounding half-extents with legitimate shape variation."""

    factors = np.asarray(size_scale_xyz, dtype=np.float64)
    if factors.shape != (3,) or np.any(factors <= 0.0):
        raise ValueError(f"invalid primitive scale: {size_scale_xyz!r}")
    if broad:
        factors = np.clip(1.0 + 1.45 * (factors - 1.0), 0.65, 1.40)
    else:
        factors = np.clip(factors, 0.85, 1.15)
    catalog_scale = (
        REUSABLE_MANIPULABLE_PRIMITIVE_SCALE
        if preset.target_eligible
        else 1.0
    )
    dimensions = (
        np.asarray(preset.dimensions_m, dtype=np.float64)
        * catalog_scale
        * factors
    )
    if preset.shape == "cylinder":
        diameter = float(np.mean(dimensions[:2]))
        dimensions[:2] = diameter
    return 0.5 * dimensions


def primitive_volume(preset: PrimitiveObjectPreset, half_extents: np.ndarray) -> float:
    half = np.asarray(half_extents, dtype=np.float64)
    if preset.shape == "box":
        return 8.0 * float(np.prod(half))
    if preset.shape == "cylinder":
        return math.pi * float(half[0] ** 2) * float(2.0 * half[2])
    raise ValueError(f"unsupported primitive shape: {preset.shape!r}")


def primitive_mass(
    preset: PrimitiveObjectPreset,
    half_extents: np.ndarray,
    *,
    mass_scale: float = 1.0,
) -> float:
    material = preset.material
    mass = (
        primitive_volume(preset, half_extents)
        * material.effective_density_kg_m3
        * preset.solid_fraction
        * float(mass_scale)
    )
    return float(np.clip(mass, *material.mass_range_kg))


def primitive_inertia(
    preset: PrimitiveObjectPreset,
    half_extents: np.ndarray,
    mass: float,
) -> np.ndarray:
    hx, hy, hz = np.asarray(half_extents, dtype=np.float64)
    if preset.shape == "box":
        return np.asarray(
            (
                mass * (hy * hy + hz * hz) / 3.0,
                mass * (hx * hx + hz * hz) / 3.0,
                mass * (hx * hx + hy * hy) / 3.0,
            )
        )
    if preset.shape == "cylinder":
        radius = hx
        height = 2.0 * hz
        transverse = mass * (3.0 * radius * radius + height * height) / 12.0
        return np.asarray((transverse, transverse, 0.5 * mass * radius * radius))
    raise ValueError(f"unsupported primitive shape: {preset.shape!r}")


__all__ = [
    "PRIMITIVE_DISTRACTOR_KEYS",
    "PRIMITIVE_OBJECTS",
    "PRIMITIVE_TARGET_KEYS",
    "REUSABLE_MANIPULABLE_PRIMITIVE_SCALE",
    "STRATEGY_1_LARGE_OBSTACLE_KEYS",
    "STRUCTURAL_CONTEXT_BOX_KEYS",
    "PrimitiveObjectPreset",
    "is_primitive_object",
    "primitive_half_extents",
    "primitive_inertia",
    "primitive_mass",
    "primitive_preset",
    "primitive_volume",
]
