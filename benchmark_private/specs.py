"""Sealed sampling schema and per-strategy phase contracts.

Nothing in this module is copied into invention or deployment workspaces.
Public scene packages contain only the materialized MJCF, object labels, and
the natural-language task goal needed for semantic binding.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import qmc

from .materials import STRUCTURAL_MATERIAL_KEYS
from .primitives import (
    PRIMITIVE_DISTRACTOR_KEYS,
    PRIMITIVE_OBJECTS,
    PRIMITIVE_TARGET_KEYS,
    STRATEGY_1_LARGE_OBSTACLE_KEYS,
)

BENCHMARK_VERSION = "real2sim_generalization_v3"
ROOT_SEED = 20260901
CASES_PER_STRATEGY = 50
DIAGNOSTIC_CASES = 12
MIXED_CASES = 28
STRESS_CASES = 10

BOX_APPEARANCES = (
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

ALL_APPEARANCES = (
    "AlphabetSoup",
    "BBQSauce",
    "Butter",
    "Cherries",
    "ChocolatePudding",
    "Cookies",
    "Corn",
    "CreamCheese",
    "GranolaBars",
    "GreenBeans",
    "Ketchup",
    "MacaroniAndCheese",
    "Mayo",
    "Milk",
    "Mushrooms",
    "Mustard",
    "OrangeJuice",
    "Parmesan",
    "Peaches",
    "PeasAndCarrots",
    "Pineapple",
    "Popcorn",
    "Raisins",
    "SaladDressing",
    "Spaghetti",
    "TomatoSauce",
    "Tuna",
    "Yogurt",
)

TARGET_APPEARANCES = BOX_APPEARANCES + PRIMITIVE_TARGET_KEYS
TASK_CONTEXT_APPEARANCES = BOX_APPEARANCES + PRIMITIVE_TARGET_KEYS
DISTRACTOR_APPEARANCES = ALL_APPEARANCES + PRIMITIVE_DISTRACTOR_KEYS

# Strategy 2 can only exercise its demonstrated wall-assisted flip when the
# flat package exposes a sufficiently tall edge after the flip while its thin
# dimension still fits between the Franka fingers.  Milk/juice cartons are too
# thick, small snack boxes expose too little height, and dense material blocks
# make the contact mode qualitatively different from the demonstrated carton.
# Keep this pool intentionally strategy-specific instead of pretending every
# globally target-eligible appearance is valid for every manipulation skill.
STRATEGY_2_TARGET_APPEARANCES = (
    "Cookies",
    "GranolaBars",
    "MacaroniAndCheese",
)
# Strategies 3 and 8 also require a substantial planar footprint.  Small
# snack cartons can satisfy a generic box schema while removing the corridor
# pull surface or the post-slide grasp overhang that defines these tasks.
STRATEGY_3_TARGET_APPEARANCES = (
    "Cookies",
    "GranolaBars",
    "MacaroniAndCheese",
)
STRATEGY_8_TARGET_APPEARANCES = (
    "Cookies",
    "GranolaBars",
    "MacaroniAndCheese",
)
# Strategy 1 needs a target that can be fully stored inside the finite-depth
# corridor on the benchmark table.  The spaghetti carton is the only
# target-eligible asset whose long axis exceeds that envelope at the sampled
# scale; retaining it creates a case that cannot even be materialized.
STRATEGY_1_TARGET_APPEARANCES = tuple(
    appearance
    for appearance in TARGET_APPEARANCES
    if appearance != "Spaghetti"
)
# Strategies 4, 5, and 7 culminate in an edge grasp or a narrow high contact.
# Reuse the already calibrated thin-carton envelope without changing Strategy
# 2 itself.
STRATEGY_4_TARGET_APPEARANCES = STRATEGY_2_TARGET_APPEARANCES
STRATEGY_5_TARGET_APPEARANCES = STRATEGY_2_TARGET_APPEARANCES
STRATEGY_7_TARGET_APPEARANCES = STRATEGY_2_TARGET_APPEARANCES
# Strategy 6 demonstrates a substantial in-plane reorientation.  Keep boxes
# with a meaningful planar major axis, but exclude the extra-long spaghetti
# carton whose insertion depth approaches the complete structural boundary.
STRATEGY_6_TARGET_APPEARANCES = (
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
    *PRIMITIVE_TARGET_KEYS,
)
STRATEGY_TARGET_APPEARANCE_POOLS: Mapping[int, tuple[str, ...]] = {
    1: STRATEGY_1_TARGET_APPEARANCES,
    2: STRATEGY_2_TARGET_APPEARANCES,
    3: STRATEGY_3_TARGET_APPEARANCES,
    4: STRATEGY_4_TARGET_APPEARANCES,
    5: STRATEGY_5_TARGET_APPEARANCES,
    7: STRATEGY_7_TARGET_APPEARANCES,
    8: STRATEGY_8_TARGET_APPEARANCES,
    6: STRATEGY_6_TARGET_APPEARANCES,
}
MINIMUM_PLANAR_TARGET_STRATEGIES = frozenset((3, 8))
STRATEGY_2_GRIPPER_MAX_WIDTH_M = 0.080
STRATEGY_2_MIN_UPRIGHT_HEIGHT_M = 0.120
STRATEGY_2_GRIPPER_CLEARANCE_M = 0.005
STRATEGY_2_MAX_TARGET_MASS_KG = 0.750
STRATEGY_2_MIN_SLIDING_FRICTION = 0.300
STRATEGY_2_MAX_MASS_FRICTION_RATIO = 2.0
STRATEGY_3_MIN_EXTRACTION_DISTANCE_M = 0.040
STRATEGY_3_MAX_EXTRACTION_DISTANCE_M = 0.300
# Keep generated cases away from the exact learned-primitive boundary.  The
# runtime still validates against the full 0.30 m domain.
STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M = 0.295
# The grasp primitive requires 2 mm of clearance on each side of the target.
STRATEGY_3_GRIPPER_CLEARANCE_M = 0.004
STRATEGY_3_GRIPPER_MAX_WIDTH_M = 0.080

DISPLAY_NAMES: Mapping[str, str] = {
    "AlphabetSoup": "Alphabet Soup can",
    "BBQSauce": "BBQ Sauce bottle",
    "Butter": "Butter box",
    "Cherries": "Cherries can",
    "ChocolatePudding": "Chocolate Pudding box",
    "Cookies": "Cookies box",
    "Corn": "Corn can",
    "CreamCheese": "Cream Cheese box",
    "GranolaBars": "Granola Bars box",
    "GreenBeans": "Green Beans can",
    "Ketchup": "Ketchup bottle",
    "MacaroniAndCheese": "Macaroni and Cheese box",
    "Mayo": "Mayo bottle",
    "Milk": "Milk carton",
    "Mushrooms": "Mushrooms can",
    "Mustard": "Mustard bottle",
    "OrangeJuice": "Orange Juice carton",
    "Parmesan": "Parmesan bottle",
    "Peaches": "Peaches can",
    "PeasAndCarrots": "Peas and Carrots can",
    "Pineapple": "Pineapple can",
    "Popcorn": "Popcorn box",
    "Raisins": "Raisins box",
    "SaladDressing": "Salad Dressing bottle",
    "Spaghetti": "Spaghetti box",
    "TomatoSauce": "Tomato Sauce can",
    "Tuna": "Tuna can",
    "Yogurt": "Yogurt cup",
    **{
        key: preset.display_name
        for key, preset in PRIMITIVE_OBJECTS.items()
    },
}


def safe_appearance_name(asset_name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(asset_name):
        if char.isupper() and index and asset_name[index - 1].islower():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


@dataclass(frozen=True)
class StrategyContract:
    strategy_id: int
    target_body: str
    fixture_bodies: tuple[str, ...]
    secondary_dynamic_bodies: tuple[str, ...]
    phases: tuple[str, ...]
    goal_template: str
    pose_translation_m: tuple[float, float]
    support_body: str | None = None
    approach_direction_xy: tuple[float, float] | None = None
    distractor_slots_xy: tuple[tuple[float, float], ...] = (
        (0.40, -0.25),
        (0.58, -0.28),
        (0.76, -0.25),
    )
    pose_yaw_degrees: float = 20.0
    phase_clearance_m: float = 0.008
    stress_clearance_m: float = 0.002

    @property
    def sampling_id(self) -> int:
        """Stable v3 sampling identity, independent of paper task numbering."""
        return {6: 8, 7: 6, 8: 7}.get(self.strategy_id, self.strategy_id)

    @property
    def key(self) -> str:
        return f"strategy_{self.strategy_id}"


STRATEGY_CONTRACTS: Mapping[int, StrategyContract] = {
    1: StrategyContract(
        1,
        "red_cheez_it_box",
        ("white_divider",),
        ("brown_cardboard_box",),
        ("planar_push", "recontact_push", "stored"),
        "Push the {target} to align it with the corridor between the {secondary_0} and {fixture_0}, then push it into the corridor for storage.",
        (0.030, 0.030),
        distractor_slots_xy=((0.42, -0.30), (0.66, -0.31), (0.88, -0.29)),
    ),
    2: StrategyContract(
        2,
        "cheez_it_box",
        ("white_drawer",),
        (),
        ("push_to_wall", "wall_assisted_flip", "edge_grasp_lift"),
        "Push the {target} to the {fixture_0}, flip it upright against the drawer, then grasp and lift it.",
        (0.050, 0.045),
        distractor_slots_xy=((0.42, -0.30), (0.66, -0.31), (0.88, -0.29)),
        pose_yaw_degrees=90.0,
    ),
    3: StrategyContract(
        3,
        "cheez_it_box",
        ("large_white_panel",),
        ("purple_and_white_carton", "brown_cardboard_box"),
        (
            "obstacle_clearance_pull",
            "push_to_wall",
            "wall_assisted_flip",
            "edge_grasp_lift",
        ),
        "Pull the {target} out from between the {secondary_0} and {secondary_1}, push it to the {fixture_0}, flip it upright, then grasp and lift it.",
        (0.018, 0.015),
        approach_direction_xy=(0.0, -1.0),
        pose_yaw_degrees=12.0,
    ),
    4: StrategyContract(
        4,
        "cheez_it_box",
        ("white_divider",),
        (),
        ("push_to_wall", "wall_assisted_flip", "clearance_pull", "edge_grasp_lift"),
        "Push the {target} to the {fixture_0}, flip it upright, pull it away to create clearance, then grasp and lift it.",
        (0.030, 0.025),
    ),
    5: StrategyContract(
        5,
        "cheez_it_box",
        ("white_divider",),
        ("brown_cardboard_box",),
        ("push_to_wall", "wall_assisted_flip_align", "slot_pull"),
        "Push the {target} to the slot between the {fixture_0} and {secondary_0}, flip it upright to align it, then pull it into the slot.",
        (0.022, 0.018),
        pose_yaw_degrees=14.0,
    ),
    7: StrategyContract(
        7,
        ("cheez_it_box"),
        ("left_white_divider", "right_white_divider"),
        (
            "left_dark_food_box",
            "purple_food_box",
            "tall_red_food_box",
            "brown_food_box",
        ),
        ("topple_expose_edge", "edge_grasp_lift"),
        "Topple the {target} outward from the row of food boxes to expose an edge, then grasp and lift it.",
        (0.008, 0.006),
        approach_direction_xy=(0.0, -1.0),
        pose_yaw_degrees=5.0,
        phase_clearance_m=0.004,
    ),
    8: StrategyContract(
        8,
        "book",
        ("white_drawer",),
        (),
        ("move_to_edge", "edge_grasp_lift"),
        "Slide the {target} toward the front edge of the {fixture_0} until it overhangs, then grasp and lift it.",
        (0.028, 0.020),
        support_body="white_drawer",
        approach_direction_xy=(0.0, -1.0),
        pose_yaw_degrees=15.0,
    ),
    6: StrategyContract(
        6,
        "blue_product_box",
        (),
        ("brown_cardboard_box", "white_product_box"),
        ("reorient", "push_to_slot", "slot_push"),
        "Reorient the {target}, align it with the opening between the {secondary_0} and {secondary_1}, then push it into the slot.",
        (0.030, 0.025),
        pose_yaw_degrees=30.0,
    ),
}


@dataclass(frozen=True)
class VariantParameters:
    case_index: int
    seed: int
    bucket: str
    diagnostic_dimension: str | None
    translation_xy: tuple[float, float]
    yaw_radians: float
    size_scale_xyz: tuple[float, float, float]
    appearance: str
    mass_scale: float
    friction_scale: float
    restitution: float
    distractor_appearances: tuple[str, ...]
    context_layout: tuple[float, ...]
    fixture_material_index: int
    fixture_scale: float
    lighting_intensity: float
    lighting_temperature: float
    secondary_appearances: tuple[str, ...]
    size_band: str
    stress_axis: str | None = None

    @property
    def case_name(self) -> str:
        return f"case_{self.case_index:03d}_scene"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def strategy_2_manipulation_metrics(
    full_dimensions_m: tuple[float, float, float] | np.ndarray,
    *,
    mass_kg: float,
    sliding_friction: float,
    gripper_max_width_m: float,
) -> dict[str, Any]:
    """Evaluate the geometry and physics needed by strategy 2's flip/grasp."""

    dimensions = np.sort(np.asarray(full_dimensions_m, dtype=np.float64))
    if dimensions.shape != (3,) or np.any(~np.isfinite(dimensions)):
        raise ValueError("strategy 2 target dimensions must contain three finite values")
    grasp_thickness = float(dimensions[0])
    upright_height = float(dimensions[1])
    usable_gripper_width = float(gripper_max_width_m) - STRATEGY_2_GRIPPER_CLEARANCE_M
    mass_friction_ratio = float(mass_kg) / max(float(sliding_friction), 1e-12)
    checks = {
        "grasp_thickness": grasp_thickness <= usable_gripper_width,
        "upright_height": upright_height >= STRATEGY_2_MIN_UPRIGHT_HEIGHT_M,
        "target_mass": float(mass_kg) <= STRATEGY_2_MAX_TARGET_MASS_KG,
        "sliding_friction": float(sliding_friction)
        >= STRATEGY_2_MIN_SLIDING_FRICTION,
        "mass_friction_ratio": mass_friction_ratio
        <= STRATEGY_2_MAX_MASS_FRICTION_RATIO,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sorted_dimensions_m": dimensions.tolist(),
        "grasp_thickness_m": grasp_thickness,
        "upright_height_m": upright_height,
        "gripper_max_width_m": float(gripper_max_width_m),
        "gripper_clearance_m": STRATEGY_2_GRIPPER_CLEARANCE_M,
        "usable_gripper_width_m": usable_gripper_width,
        "minimum_upright_height_m": STRATEGY_2_MIN_UPRIGHT_HEIGHT_M,
        "mass_kg": float(mass_kg),
        "maximum_mass_kg": STRATEGY_2_MAX_TARGET_MASS_KG,
        "sliding_friction": float(sliding_friction),
        "minimum_sliding_friction": STRATEGY_2_MIN_SLIDING_FRICTION,
        "mass_friction_ratio": mass_friction_ratio,
        "maximum_mass_friction_ratio": STRATEGY_2_MAX_MASS_FRICTION_RATIO,
    }


def strategy_3_manipulation_metrics(
    full_dimensions_m: tuple[float, float, float] | np.ndarray,
    *,
    gripper_max_width_m: float,
) -> dict[str, Any]:
    """Evaluate the intended upright grasp envelope for strategy 3."""

    dimensions = np.sort(np.asarray(full_dimensions_m, dtype=np.float64))
    if dimensions.shape != (3,) or np.any(~np.isfinite(dimensions)):
        raise ValueError("strategy 3 target dimensions must contain three finite values")
    grasp_thickness = float(dimensions[0])
    usable_gripper_width = (
        float(gripper_max_width_m) - STRATEGY_3_GRIPPER_CLEARANCE_M
    )
    checks = {"post_flip_grasp_thickness": grasp_thickness <= usable_gripper_width}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sorted_dimensions_m": dimensions.tolist(),
        "post_flip_grasp_thickness_m": grasp_thickness,
        "gripper_max_width_m": float(gripper_max_width_m),
        "gripper_clearance_m": STRATEGY_3_GRIPPER_CLEARANCE_M,
        "usable_gripper_width_m": usable_gripper_width,
    }


def _case_seed(strategy_id: int, case_index: int) -> int:
    digest = hashlib.sha256(
        f"{BENCHMARK_VERSION}:{ROOT_SEED}:{STRATEGY_CONTRACTS[strategy_id].sampling_id}:{case_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _diagnostic_variants(contract: StrategyContract) -> list[VariantParameters]:
    dx, dy = contract.pose_translation_m
    common = {
        "translation_xy": (0.0, 0.0),
        "yaw_radians": 0.0,
        "size_scale_xyz": (1.0, 1.0, 1.0),
        "appearance": "GranolaBars",
        "mass_scale": 1.0,
        "friction_scale": 1.0,
        "restitution": 0.05,
        "distractor_appearances": ("Ketchup",),
        "context_layout": (0.0,) * 8,
        "fixture_material_index": 0,
        "fixture_scale": 1.0,
        "lighting_intensity": 1.0,
        "lighting_temperature": 0.0,
        "secondary_appearances": _secondary_appearances(
            contract, start=0, excluded=("GranolaBars",)
        ),
        "size_band": "regular",
        "stress_axis": None,
    }
    negative_context = (-1.0, -0.65, -0.8, 0.55, -0.75, 0.7, -0.45, 0.9)
    positive_context = tuple(-value for value in negative_context)
    small_size = (
        (0.90, 0.92, 0.91)
        if contract.strategy_id in MINIMUM_PLANAR_TARGET_STRATEGIES
        else (0.75, 0.78, 0.77)
    )
    changes: list[tuple[str, dict[str, Any]]] = [
        ("pose", {"translation_xy": (-dx, -dy), "yaw_radians": -math.radians(contract.pose_yaw_degrees), "context_layout": negative_context}),
        ("pose", {"translation_xy": (dx, dy), "yaw_radians": math.radians(contract.pose_yaw_degrees), "context_layout": positive_context}),
        ("size", {"size_scale_xyz": small_size, "size_band": "small"}),
        ("size", {"size_scale_xyz": (1.25, 1.22, 1.24), "size_band": "large"}),
        ("appearance", {"appearance": "Cookies", "fixture_material_index": 8, "fixture_scale": 0.92, "secondary_appearances": _secondary_appearances(contract, start=3, excluded=("Cookies",))}),
        ("appearance", {"appearance": "MacaroniAndCheese" if contract.strategy_id in STRATEGY_TARGET_APPEARANCE_POOLS else "Milk", "fixture_material_index": 10, "fixture_scale": 1.08, "secondary_appearances": _secondary_appearances(contract, start=7, excluded=(("MacaroniAndCheese",) if contract.strategy_id in STRATEGY_TARGET_APPEARANCE_POOLS else ("Milk",)))}),
        ("mass", {"mass_scale": 0.50}),
        ("mass", {"mass_scale": 1.50 if contract.strategy_id == 2 else 2.00}),
        ("friction", {"friction_scale": 0.75 if contract.strategy_id == 2 else 0.50}),
        ("friction", {"friction_scale": 1.50}),
        ("elasticity", {"restitution": 0.02}),
        ("elasticity", {"restitution": 0.30}),
    ]
    result: list[VariantParameters] = []
    for case_index, (dimension, override) in enumerate(changes, start=1):
        values = dict(common)
        values.update(override)
        result.append(
            VariantParameters(
                case_index=case_index,
                seed=_case_seed(contract.strategy_id, case_index),
                bucket="diagnostic",
                diagnostic_dimension=dimension,
                **values,
            )
        )
    return result


def _mixed_variants(contract: StrategyContract) -> list[VariantParameters]:
    sampler = qmc.Sobol(d=20, scramble=True, seed=ROOT_SEED + contract.sampling_id)
    points = sampler.random_base2(m=5)[:MIXED_CASES]
    result: list[VariantParameters] = []
    dx, dy = contract.pose_translation_m
    yaw = math.radians(contract.pose_yaw_degrees)
    for offset, unit in enumerate(points, start=13):
        local_index = offset - 13
        if contract.strategy_id in STRATEGY_TARGET_APPEARANCE_POOLS:
            appearance_pool = STRATEGY_TARGET_APPEARANCE_POOLS[
                contract.strategy_id
            ]
            appearance = appearance_pool[
                min(int(unit[8] * len(appearance_pool)), len(appearance_pool) - 1)
            ]
        elif local_index % 5 in {2, 4}:
            appearance = PRIMITIVE_TARGET_KEYS[
                (contract.sampling_id + local_index) % len(PRIMITIVE_TARGET_KEYS)
            ]
        else:
            appearance = BOX_APPEARANCES[
                min(int(unit[8] * len(BOX_APPEARANCES)), len(BOX_APPEARANCES) - 1)
            ]
        band_selector = local_index % 5
        if band_selector == 0:
            size_band = "small"
            uniform_size = (
                0.90 + 0.06 * float(unit[3])
                if contract.strategy_id in MINIMUM_PLANAR_TARGET_STRATEGIES
                else 0.75 + 0.10 * float(unit[3])
            )
        elif band_selector == 1:
            size_band = "large"
            uniform_size = 1.15 + 0.10 * float(unit[3])
        else:
            size_band = "regular"
            uniform_size = 0.90 + 0.20 * float(unit[3])
        size_scale = (
            uniform_size * (0.92 + 0.16 * float(unit[4])),
            uniform_size * (0.92 + 0.16 * float(unit[5])),
            uniform_size,
        )
        count = 1 + min(int(unit[9] * 3), 2)
        start = min(
            int(unit[10] * len(DISTRACTOR_APPEARANCES)),
            len(DISTRACTOR_APPEARANCES) - 1,
        )
        distractors = tuple(
            DISTRACTOR_APPEARANCES[
                (start + 7 * item + contract.sampling_id)
                % len(DISTRACTOR_APPEARANCES)
            ]
            for item in range(count)
            if DISTRACTOR_APPEARANCES[
                (start + 7 * item + contract.sampling_id)
                % len(DISTRACTOR_APPEARANCES)
            ]
            != appearance
        )
        if not distractors:
            distractors = ("Ketchup",)
        result.append(
            VariantParameters(
                case_index=offset,
                seed=_case_seed(contract.strategy_id, offset),
                bucket="mixed",
                diagnostic_dimension=None,
                translation_xy=(float((2 * unit[0] - 1) * dx), float((2 * unit[1] - 1) * dy)),
                yaw_radians=float((2 * unit[2] - 1) * yaw),
                size_scale_xyz=tuple(float(value) for value in size_scale),
                appearance=appearance,
                mass_scale=float(
                    0.50
                    + (1.00 if contract.strategy_id == 2 else 1.50) * unit[6]
                ),
                friction_scale=float(
                    (0.75 if contract.strategy_id == 2 else 0.50)
                    + (0.75 if contract.strategy_id == 2 else 1.00) * unit[7]
                ),
                restitution=float(0.02 + 0.28 * ((unit[5] + unit[7]) % 1.0)),
                distractor_appearances=distractors,
                context_layout=tuple(float(2 * unit[index] - 1) for index in range(11, 19)),
                fixture_material_index=min(
                    int(unit[19] * len(STRUCTURAL_MATERIAL_KEYS)),
                    len(STRUCTURAL_MATERIAL_KEYS) - 1,
                ),
                fixture_scale=float(0.90 + 0.20 * ((unit[18] + unit[0]) % 1.0)),
                lighting_intensity=float(0.86 + 0.28 * ((unit[17] + unit[7]) % 1.0)),
                lighting_temperature=float(-1.0 + 2.0 * ((unit[16] + unit[6]) % 1.0)),
                secondary_appearances=_secondary_appearances(
                    contract,
                    start=min(
                        int(unit[10] * len(TASK_CONTEXT_APPEARANCES)),
                        len(TASK_CONTEXT_APPEARANCES) - 1,
                    ),
                    excluded=(appearance,),
                ),
                size_band=size_band,
            )
        )
    return result


def _stress_variants(contract: StrategyContract) -> list[VariantParameters]:
    axes = (
        "negative_pose",
        "positive_pose",
        "small_object",
        "large_object",
        "heavy_low_friction",
        "light_high_friction",
        "high_elasticity",
        "pose_size_joint",
        "physics_joint",
        "all_dimensions",
    )
    dx, dy = contract.pose_translation_m
    yaw = math.radians(contract.pose_yaw_degrees)
    result: list[VariantParameters] = []
    for local_index, axis in enumerate(axes):
        case_index = 41 + local_index
        rng = np.random.default_rng(_case_seed(contract.strategy_id, case_index))
        sign = -1.0 if local_index % 2 == 0 else 1.0
        translation = (sign * 0.98 * dx, -sign * 0.98 * dy)
        yaw_value = sign * 0.98 * yaw
        size = (1.0, 1.0, 1.0)
        mass = 1.0
        friction = 1.0
        restitution = 0.05
        size_band = "regular"
        if axis == "small_object":
            size = (
                (0.90, 0.92, 0.91)
                if contract.strategy_id in MINIMUM_PLANAR_TARGET_STRATEGIES
                else (0.75, 0.77, 0.76)
            )
            size_band = "small"
        elif axis == "large_object":
            size = (1.25, 1.23, 1.24)
            size_band = "large"
        elif axis == "heavy_low_friction":
            mass, friction = (
                (1.50, 0.75)
                if contract.strategy_id == 2
                else (2.0, 0.50)
            )
        elif axis == "light_high_friction":
            mass, friction = 0.50, 1.50
        elif axis == "high_elasticity":
            restitution = 0.30
        elif axis == "pose_size_joint":
            size = (1.22, 1.16, 1.20)
            size_band = "large"
        elif axis == "physics_joint":
            mass, friction, restitution = (
                (1.48, 0.77, 0.29)
                if contract.strategy_id == 2
                else (1.95, 0.52, 0.29)
            )
        elif axis == "all_dimensions":
            uniform = float(
                rng.choice(
                    (0.90, 1.22)
                    if contract.strategy_id in MINIMUM_PLANAR_TARGET_STRATEGIES
                    else (0.78, 1.22)
                )
            )
            size = tuple(float(uniform * value) for value in rng.uniform(0.94, 1.06, size=3))
            size_band = "small" if uniform < 1.0 else "large"
            mass, friction, restitution = (
                (1.45, 0.80, 0.28)
                if contract.strategy_id == 2
                else (1.90, 0.55, 0.28)
            )
        if contract.strategy_id in STRATEGY_TARGET_APPEARANCE_POOLS:
            appearance_pool = STRATEGY_TARGET_APPEARANCE_POOLS[
                contract.strategy_id
            ]
            appearance = appearance_pool[local_index % len(appearance_pool)]
        elif local_index % 2:
            appearance = PRIMITIVE_TARGET_KEYS[
                (contract.sampling_id + local_index * 2) % len(PRIMITIVE_TARGET_KEYS)
            ]
        else:
            appearance = BOX_APPEARANCES[
                (contract.sampling_id + local_index * 3) % len(BOX_APPEARANCES)
            ]
        distractors = tuple(
            DISTRACTOR_APPEARANCES[
                (contract.sampling_id * 2 + local_index * 5 + item * 9)
                % len(DISTRACTOR_APPEARANCES)
            ]
            for item in range(2)
        )
        context = (0.0,) * 8
        if axis in {"negative_pose", "positive_pose", "pose_size_joint", "all_dimensions"}:
            context = tuple(float(value) for value in rng.uniform(-1.0, 1.0, size=8))
            if axis == "negative_pose":
                context = tuple(-abs(value) for value in context)
            elif axis == "positive_pose":
                context = tuple(abs(value) for value in context)
        result.append(
            VariantParameters(
                case_index=case_index,
                seed=_case_seed(contract.strategy_id, case_index),
                bucket="stress",
                diagnostic_dimension=None,
                translation_xy=translation,
                yaw_radians=yaw_value,
                size_scale_xyz=size,
                appearance=appearance,
                mass_scale=mass,
                friction_scale=friction,
                restitution=restitution,
                distractor_appearances=distractors,
                context_layout=context,
                fixture_material_index=(
                    contract.sampling_id + 3 * local_index
                )
                % len(STRUCTURAL_MATERIAL_KEYS),
                fixture_scale=(0.90 if local_index % 3 == 0 else 1.10 if local_index % 3 == 1 else 1.0),
                lighting_intensity=(0.86 if local_index % 2 == 0 else 1.14),
                lighting_temperature=(-0.85 if local_index % 2 == 0 else 0.85),
                secondary_appearances=_secondary_appearances(
                    contract,
                    start=contract.sampling_id + 2 * local_index,
                    excluded=(appearance,),
                ),
                size_band=size_band,
                stress_axis=axis,
            )
        )
    return result


def variants_for_strategy(strategy_id: int) -> tuple[VariantParameters, ...]:
    contract = STRATEGY_CONTRACTS[strategy_id]
    variants = (
        _diagnostic_variants(contract)
        + _mixed_variants(contract)
        + _stress_variants(contract)
    )
    if len(variants) != CASES_PER_STRATEGY:
        raise AssertionError(f"expected {CASES_PER_STRATEGY} variants, got {len(variants)}")
    if [item.case_index for item in variants] != list(range(1, CASES_PER_STRATEGY + 1)):
        raise AssertionError("case indices are not contiguous")
    return tuple(variants)


def _secondary_appearances(
    contract: StrategyContract,
    *,
    start: int,
    excluded: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Choose stable, unique box skins for task-relevant movable context."""

    needed = len(contract.secondary_dynamic_bodies)
    role_pools: dict[int, tuple[tuple[str, ...], ...]] = {
        1: (STRATEGY_1_LARGE_OBSTACLE_KEYS,),
        3: (STRATEGY_1_LARGE_OBSTACLE_KEYS, STRATEGY_1_LARGE_OBSTACLE_KEYS),
        5: (STRATEGY_1_LARGE_OBSTACLE_KEYS,),
        6: (STRATEGY_1_LARGE_OBSTACLE_KEYS, STRATEGY_1_LARGE_OBSTACLE_KEYS),
    }
    pools = role_pools.get(
        contract.strategy_id,
        tuple(TASK_CONTEXT_APPEARANCES for _ in range(needed)),
    )
    selected: list[str] = []
    blocked = set(excluded)
    for role_index in range(needed):
        pool = pools[role_index]
        cursor = start + contract.sampling_id + 3 * role_index
        while True:
            appearance = pool[cursor % len(pool)]
            cursor += 1
            if appearance in blocked or appearance in selected:
                continue
            selected.append(appearance)
            break
    return tuple(selected)


def sampling_fingerprint(strategy_id: int) -> str:
    payload = [item.to_json() for item in variants_for_strategy(strategy_id)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "ALL_APPEARANCES",
    "BENCHMARK_VERSION",
    "BOX_APPEARANCES",
    "CASES_PER_STRATEGY",
    "DISPLAY_NAMES",
    "DISTRACTOR_APPEARANCES",
    "ROOT_SEED",
    "STRATEGY_CONTRACTS",
    "STRATEGY_2_GRIPPER_CLEARANCE_M",
    "STRATEGY_2_GRIPPER_MAX_WIDTH_M",
    "STRATEGY_2_MAX_MASS_FRICTION_RATIO",
    "STRATEGY_2_MAX_TARGET_MASS_KG",
    "STRATEGY_2_MIN_SLIDING_FRICTION",
    "STRATEGY_2_MIN_UPRIGHT_HEIGHT_M",
    "STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M",
    "STRATEGY_3_GRIPPER_CLEARANCE_M",
    "STRATEGY_3_GRIPPER_MAX_WIDTH_M",
    "STRATEGY_3_MAX_EXTRACTION_DISTANCE_M",
    "STRATEGY_3_MIN_EXTRACTION_DISTANCE_M",
    "STRATEGY_2_TARGET_APPEARANCES",
    "STRATEGY_3_TARGET_APPEARANCES",
    "STRATEGY_4_TARGET_APPEARANCES",
    "STRATEGY_5_TARGET_APPEARANCES",
    "STRATEGY_7_TARGET_APPEARANCES",
    "STRATEGY_8_TARGET_APPEARANCES",
    "STRATEGY_6_TARGET_APPEARANCES",
    "STRATEGY_1_TARGET_APPEARANCES",
    "STRATEGY_TARGET_APPEARANCE_POOLS",
    "TARGET_APPEARANCES",
    "TASK_CONTEXT_APPEARANCES",
    "StrategyContract",
    "VariantParameters",
    "safe_appearance_name",
    "sampling_fingerprint",
    "strategy_2_manipulation_metrics",
    "strategy_3_manipulation_metrics",
    "variants_for_strategy",
]
