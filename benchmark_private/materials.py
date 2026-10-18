"""Calibrated visual and physical material presets for benchmark generation.

The values in this module are simulation presets, not claims of exact bulk
material constants.  Effective density accounts for hollow packaging and
contents; contact values are bounded for stable MuJoCo behavior while
preserving physically sensible ordering between material families.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationMaterial:
    key: str
    display_name: str
    family: str
    use: str
    surface_model: str
    base_rgb: tuple[float, float, float]
    accent_rgb: tuple[float, float, float]
    effective_density_kg_m3: float
    sliding_friction: float
    torsional_friction: float
    rolling_friction: float
    restitution: float
    specular: float
    shininess: float
    reflectance: float
    mass_range_kg: tuple[float, float]
    opacity: float = 1.0

    def to_json(self) -> dict[str, object]:
        return asdict(self)


MATERIALS: Mapping[str, SimulationMaterial] = {
    # Structural fixture finishes.
    "birch_wood": SimulationMaterial(
        "birch_wood",
        "birch wood",
        "wood",
        "structural",
        "wood_grain",
        (0.66, 0.49, 0.30),
        (0.38, 0.24, 0.13),
        540.0,
        0.58,
        0.010,
        0.0015,
        0.07,
        0.10,
        0.12,
        0.015,
        (0.10, 8.0),
    ),
    "walnut_wood": SimulationMaterial(
        "walnut_wood",
        "walnut wood",
        "wood",
        "structural",
        "wood_grain",
        (0.31, 0.20, 0.12),
        (0.12, 0.075, 0.045),
        620.0,
        0.61,
        0.011,
        0.0015,
        0.06,
        0.11,
        0.14,
        0.012,
        (0.10, 8.0),
    ),
    "matte_abs": SimulationMaterial(
        "matte_abs",
        "matte ABS plastic",
        "plastic",
        "structural",
        "matte_polymer",
        (0.31, 0.34, 0.35),
        (0.22, 0.24, 0.25),
        1040.0,
        0.39,
        0.007,
        0.0010,
        0.10,
        0.18,
        0.18,
        0.010,
        (0.08, 8.0),
    ),
    "recycled_hdpe": SimulationMaterial(
        "recycled_hdpe",
        "recycled HDPE plastic",
        "plastic",
        "structural",
        "polymer_speckle",
        (0.36, 0.43, 0.39),
        (0.24, 0.30, 0.27),
        950.0,
        0.34,
        0.006,
        0.0010,
        0.12,
        0.16,
        0.16,
        0.008,
        (0.08, 8.0),
    ),
    "brushed_aluminum": SimulationMaterial(
        "brushed_aluminum",
        "brushed aluminum",
        "metal",
        "structural",
        "brushed_metal",
        (0.60, 0.62, 0.63),
        (0.40, 0.42, 0.44),
        2700.0,
        0.31,
        0.005,
        0.0008,
        0.16,
        0.72,
        0.78,
        0.24,
        (0.20, 20.0),
    ),
    "powder_coated_steel": SimulationMaterial(
        "powder_coated_steel",
        "powder-coated steel",
        "metal",
        "structural",
        "powder_coat",
        (0.27, 0.29, 0.30),
        (0.18, 0.19, 0.20),
        7850.0,
        0.43,
        0.007,
        0.0008,
        0.09,
        0.28,
        0.25,
        0.04,
        (0.30, 30.0),
    ),
    "cast_iron": SimulationMaterial(
        "cast_iron",
        "cast iron",
        "metal",
        "structural",
        "cast_metal",
        (0.22, 0.23, 0.23),
        (0.12, 0.13, 0.13),
        7200.0,
        0.48,
        0.008,
        0.0009,
        0.06,
        0.32,
        0.30,
        0.035,
        (0.30, 30.0),
    ),
    "glazed_ceramic": SimulationMaterial(
        "glazed_ceramic",
        "glazed ceramic",
        "ceramic",
        "structural",
        "ceramic_speckle",
        (0.73, 0.71, 0.65),
        (0.56, 0.55, 0.51),
        2300.0,
        0.46,
        0.008,
        0.0010,
        0.13,
        0.52,
        0.62,
        0.10,
        (0.20, 16.0),
    ),
    "corrugated_cardboard_panel": SimulationMaterial(
        "corrugated_cardboard_panel",
        "corrugated cardboard",
        "paper",
        "structural",
        "cardboard_fiber",
        (0.60, 0.46, 0.29),
        (0.39, 0.28, 0.17),
        140.0,
        0.56,
        0.010,
        0.0014,
        0.04,
        0.04,
        0.045,
        0.0,
        (0.05, 5.0),
    ),
    "glazed_porcelain": SimulationMaterial(
        "glazed_porcelain",
        "glazed porcelain",
        "ceramic",
        "structural",
        "porcelain_glaze",
        (0.84, 0.82, 0.77),
        (0.66, 0.68, 0.67),
        2400.0,
        0.42,
        0.007,
        0.0009,
        0.14,
        0.68,
        0.80,
        0.14,
        (0.20, 18.0),
    ),
    "tempered_glass": SimulationMaterial(
        "tempered_glass",
        "tempered glass",
        "glass",
        "structural",
        "glass_haze",
        (0.68, 0.77, 0.78),
        (0.43, 0.57, 0.60),
        2500.0,
        0.38,
        0.006,
        0.0008,
        0.15,
        0.88,
        0.92,
        0.32,
        (0.20, 20.0),
        0.38,
    ),
    # Packaging materials used by HOPE-skinned objects.
    "corrugated_cardboard": SimulationMaterial(
        "corrugated_cardboard",
        "corrugated cardboard",
        "paper",
        "packaging",
        "cardboard_fiber",
        (0.54, 0.42, 0.27),
        (0.38, 0.28, 0.17),
        45.0,
        0.52,
        0.009,
        0.0013,
        0.035,
        0.05,
        0.05,
        0.0,
        (0.06, 0.80),
    ),
    "coated_carton": SimulationMaterial(
        "coated_carton",
        "coated paperboard carton",
        "paper",
        "packaging",
        "coated_paper",
        (0.72, 0.70, 0.65),
        (0.54, 0.52, 0.48),
        75.0,
        0.44,
        0.008,
        0.0011,
        0.05,
        0.09,
        0.08,
        0.0,
        (0.05, 0.70),
    ),
    "rigid_plastic": SimulationMaterial(
        "rigid_plastic",
        "rigid plastic packaging",
        "plastic",
        "packaging",
        "smooth_polymer",
        (0.48, 0.51, 0.52),
        (0.36, 0.38, 0.40),
        430.0,
        0.36,
        0.006,
        0.0009,
        0.11,
        0.22,
        0.24,
        0.012,
        (0.05, 1.20),
    ),
    "tinplate_steel": SimulationMaterial(
        "tinplate_steel",
        "tinplate steel can",
        "metal",
        "packaging",
        "smooth_metal",
        (0.58, 0.60, 0.61),
        (0.42, 0.44, 0.45),
        620.0,
        0.29,
        0.005,
        0.0008,
        0.17,
        0.48,
        0.56,
        0.16,
        (0.08, 1.50),
    ),
    "glass_container": SimulationMaterial(
        "glass_container",
        "glass container",
        "glass",
        "packaging",
        "glass",
        (0.66, 0.70, 0.68),
        (0.48, 0.52, 0.50),
        900.0,
        0.40,
        0.006,
        0.0008,
        0.12,
        0.70,
        0.82,
        0.20,
        (0.10, 1.80),
    ),
}


STRUCTURAL_MATERIAL_KEYS = (
    "birch_wood",
    "walnut_wood",
    "matte_abs",
    "recycled_hdpe",
    "brushed_aluminum",
    "powder_coated_steel",
    "cast_iron",
    "glazed_ceramic",
    "corrugated_cardboard_panel",
    "glazed_porcelain",
    "tempered_glass",
)


WORK_SURFACE_MATERIAL_KEYS = (
    "birch_wood",
    "walnut_wood",
    "recycled_hdpe",
    "brushed_aluminum",
    "powder_coated_steel",
    "glazed_ceramic",
)


APPEARANCE_MATERIAL: Mapping[str, str] = {
    "AlphabetSoup": "tinplate_steel",
    "BBQSauce": "rigid_plastic",
    "Butter": "coated_carton",
    "Cherries": "tinplate_steel",
    "ChocolatePudding": "coated_carton",
    "Cookies": "coated_carton",
    "Corn": "tinplate_steel",
    "CreamCheese": "coated_carton",
    "GranolaBars": "coated_carton",
    "GreenBeans": "tinplate_steel",
    "Ketchup": "rigid_plastic",
    "MacaroniAndCheese": "coated_carton",
    "Mayo": "rigid_plastic",
    "Milk": "coated_carton",
    "Mushrooms": "tinplate_steel",
    "Mustard": "rigid_plastic",
    "OrangeJuice": "coated_carton",
    "Parmesan": "rigid_plastic",
    "Peaches": "tinplate_steel",
    "PeasAndCarrots": "tinplate_steel",
    "Pineapple": "tinplate_steel",
    "Popcorn": "coated_carton",
    "Raisins": "coated_carton",
    "SaladDressing": "rigid_plastic",
    "Spaghetti": "coated_carton",
    "TomatoSauce": "tinplate_steel",
    "Tuna": "tinplate_steel",
    "Yogurt": "rigid_plastic",
}


def structural_material(index: int) -> SimulationMaterial:
    return MATERIALS[STRUCTURAL_MATERIAL_KEYS[index % len(STRUCTURAL_MATERIAL_KEYS)]]


def appearance_material(appearance: str) -> SimulationMaterial:
    try:
        return MATERIALS[APPEARANCE_MATERIAL[appearance]]
    except KeyError as exc:
        raise KeyError(f"no packaging material is mapped for {appearance!r}") from exc


def compatible_work_surface_material(
    fixture_materials: Sequence[SimulationMaterial],
    *,
    seed: int,
) -> SimulationMaterial:
    """Choose a plausible work surface with contrast from nearby fixtures."""

    fixtures = tuple(fixture_materials)
    candidates = [MATERIALS[key] for key in WORK_SURFACE_MATERIAL_KEYS]

    def luminance(material: SimulationMaterial) -> float:
        return float(
            np.dot(np.asarray(material.base_rgb), np.asarray((0.2126, 0.7152, 0.0722)))
        )

    contrasting = [
        material
        for material in candidates
        if all(material.key != fixture.key for fixture in fixtures)
        and all(material.family != fixture.family for fixture in fixtures)
        and all(
            abs(luminance(material) - luminance(fixture)) >= 0.12
            for fixture in fixtures
        )
    ]
    if not contrasting:
        contrasting = [
            material
            for material in candidates
            if all(material.key != fixture.key for fixture in fixtures)
            and all(material.family != fixture.family for fixture in fixtures)
        ]
    if not contrasting:
        contrasting = candidates
    return contrasting[int(seed) % len(contrasting)]


def effective_box_mass(
    material: SimulationMaterial,
    half_extents: np.ndarray,
    *,
    scale: float = 1.0,
) -> float:
    volume = 8.0 * float(np.prod(np.asarray(half_extents, dtype=np.float64)))
    value = volume * material.effective_density_kg_m3 * float(scale)
    return float(np.clip(value, *material.mass_range_kg))


def _noise(x: float, y: float, z: float, seed: int) -> float:
    value = math.sin(
        12.9898 * x + 78.233 * y + 37.719 * z + 0.00137 * seed
    ) * 43758.5453
    return value - math.floor(value)


def _surface_mix(
    material: SimulationMaterial,
    xyz: tuple[float, float, float],
    *,
    seed: int,
) -> float:
    x, y, z = xyz
    noise = _noise(x, y, z, seed)
    phase = (seed % 997) / 997.0 * 2.0 * math.pi
    model = material.surface_model
    if model == "wood_grain":
        grain = 0.5 + 0.5 * math.sin(30.0 * x + 5.0 * math.sin(9.0 * z) + phase)
        return float(np.clip(0.20 + 0.52 * grain + 0.12 * noise, 0.0, 1.0))
    if model == "brushed_metal":
        brush = 0.5 + 0.5 * math.sin(210.0 * z + phase)
        return float(np.clip(0.34 + 0.16 * brush + 0.10 * noise, 0.0, 1.0))
    if model == "powder_coat":
        return 0.34 + 0.18 * noise
    if model == "cast_metal":
        return 0.28 + 0.25 * noise
    if model == "polymer_speckle":
        speck = 0.22 if noise > 0.91 else 0.0
        return 0.28 + 0.13 * noise + speck
    if model == "ceramic_speckle":
        speck = 0.18 if noise > 0.95 else 0.0
        return 0.22 + 0.10 * noise + speck
    if model == "cardboard_fiber":
        corrugation = 0.5 + 0.5 * math.sin(42.0 * x + 2.0 * math.sin(6.0 * y))
        return float(np.clip(0.20 + 0.23 * corrugation + 0.10 * noise, 0.0, 1.0))
    if model == "porcelain_glaze":
        return float(np.clip(0.10 + 0.055 * noise, 0.0, 1.0))
    if model == "glass_haze":
        haze = 0.5 + 0.5 * math.sin(8.0 * x + phase)
        return float(np.clip(0.10 + 0.07 * haze + 0.04 * noise, 0.0, 1.0))
    return 0.26 + 0.10 * noise


def material_vertex_rgb(
    material: SimulationMaterial,
    xyz: tuple[float, float, float],
    *,
    seed: int,
) -> tuple[float, float, float]:
    mix = _surface_mix(material, xyz, seed=seed)
    base = np.asarray(material.base_rgb, dtype=np.float64)
    accent = np.asarray(material.accent_rgb, dtype=np.float64)
    color = (1.0 - mix) * base + mix * accent
    return tuple(float(value) for value in np.clip(color, 0.0, 1.0))


def material_texture_rgb(
    material: SimulationMaterial,
    *,
    seed: int,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Create a deterministic tileable-looking RGB surface texture."""

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    xf = x / max(1, width - 1)
    yf = y / max(1, height - 1)
    noise = rng.normal(0.0, 1.0, size=(height, width))
    phase = (seed % 997) / 997.0 * 2.0 * math.pi
    model = material.surface_model
    if model == "wood_grain":
        warped = 8.0 * xf + 0.42 * np.sin(2.0 * math.pi * (2.2 * yf + phase))
        broad = 0.5 + 0.5 * np.sin(2.0 * math.pi * warped + phase)
        fine = 0.5 + 0.5 * np.sin(2.0 * math.pi * (31.0 * xf + 1.7 * yf))
        mix = 0.20 + 0.48 * broad + 0.10 * fine + 0.035 * noise
    elif model == "brushed_metal":
        bands = 0.5 + 0.5 * np.sin(2.0 * math.pi * (95.0 * yf + phase))
        mix = 0.33 + 0.12 * bands + 0.035 * noise
    elif model == "powder_coat":
        mix = 0.31 + 0.10 * noise
    elif model == "cast_metal":
        coarse = np.sin(2.0 * math.pi * (7.0 * xf + 5.0 * yf + phase))
        mix = 0.29 + 0.08 * coarse + 0.11 * noise
    elif model == "polymer_speckle":
        speckles = (noise > 1.65).astype(np.float64)
        mix = 0.26 + 0.055 * noise + 0.20 * speckles
    elif model == "ceramic_speckle":
        speckles = (noise > 2.0).astype(np.float64)
        mix = 0.19 + 0.035 * noise + 0.16 * speckles
    elif model == "cardboard_fiber":
        corrugation = 0.5 + 0.5 * np.sin(
            2.0 * math.pi * (28.0 * xf + 0.25 * np.sin(2.0 * math.pi * yf))
        )
        fibers = 0.5 + 0.5 * np.sin(2.0 * math.pi * (113.0 * yf + phase))
        mix = 0.21 + 0.15 * corrugation + 0.06 * fibers + 0.055 * noise
    elif model == "porcelain_glaze":
        flecks = (noise > 2.25).astype(np.float64)
        cloud = np.sin(2.0 * math.pi * (1.5 * xf + 1.1 * yf + phase))
        mix = 0.11 + 0.025 * cloud + 0.02 * noise + 0.09 * flecks
    elif model == "glass_haze":
        vertical = 0.5 + 0.5 * np.sin(2.0 * math.pi * (2.0 * xf + phase))
        mix = 0.14 + 0.035 * vertical + 0.025 * noise
    else:
        mix = 0.25 + 0.045 * noise
    mix = np.clip(mix, 0.0, 1.0)[..., None]
    base = np.asarray(material.base_rgb, dtype=np.float64)
    accent = np.asarray(material.accent_rgb, dtype=np.float64)
    rgb = (1.0 - mix) * base + mix * accent
    return np.asarray(np.clip(np.round(255.0 * rgb), 0, 255), dtype=np.uint8)


def work_surface_texture_rgb(
    material: SimulationMaterial,
    *,
    seed: int,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Make a lower-contrast texture suitable for a large work surface."""

    if material.surface_model == "wood_grain":
        rng = np.random.default_rng(seed + 17)
        profile = rng.normal(0.0, 1.0, size=height)
        kernel = np.ones(19, dtype=np.float64) / 19.0
        profile = np.convolve(profile, kernel, mode="same")
        profile -= profile.min()
        profile /= max(1e-9, float(profile.max()))
        grain = np.empty((height, width), dtype=np.float64)
        for column in range(width):
            offset = int(round(3.0 * math.sin(2.0 * math.pi * column / width)))
            grain[:, column] = np.roll(profile, offset)
        grain += 0.035 * rng.normal(0.0, 1.0, size=(height, width))
        mix = np.clip(0.10 + 0.24 * grain, 0.0, 1.0)[..., None]
        base = np.asarray(material.base_rgb, dtype=np.float64)
        accent = np.asarray(material.accent_rgb, dtype=np.float64)
        rgb = (1.0 - mix) * base + mix * accent
        return np.asarray(
            np.clip(np.round(255.0 * rgb), 0, 255),
            dtype=np.uint8,
        )

    texture = material_texture_rgb(
        material,
        seed=seed,
        width=width,
        height=height,
    ).astype(np.float64)
    base = 255.0 * np.asarray(material.base_rgb, dtype=np.float64)
    softened = base + 0.48 * (texture - base)
    return np.asarray(np.clip(np.round(softened), 0, 255), dtype=np.uint8)


def material_table_json() -> list[dict[str, object]]:
    return [MATERIALS[key].to_json() for key in MATERIALS]


__all__ = [
    "APPEARANCE_MATERIAL",
    "MATERIALS",
    "STRUCTURAL_MATERIAL_KEYS",
    "WORK_SURFACE_MATERIAL_KEYS",
    "SimulationMaterial",
    "appearance_material",
    "compatible_work_surface_material",
    "effective_box_mass",
    "material_table_json",
    "material_texture_rgb",
    "material_vertex_rgb",
    "structural_material",
    "work_surface_texture_rgb",
]
