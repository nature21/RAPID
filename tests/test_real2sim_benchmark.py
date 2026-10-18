from __future__ import annotations

import json
import importlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from benchmark_private.certify import (
    _extraction_then_wall_metrics,
    _load_scene,
    _packed_row_metrics,
    _platform_edge_metrics,
    _required_visible_distractor_count,
    _settling_gate,
    _slot_metrics,
    _strategy_1_corridor_metrics,
    _wall_flip_metrics,
)
from benchmark_private.assets import (
    ASSET_ASPECT_JITTER,
    ASSET_CATEGORIES,
    ASSET_SCALE_RANGES,
    ASSET_UNIFORM_SCALE_RANGE,
    HOPE_REUSABLE_ASSET_SCALE,
    asset_record,
    hope_upright_quaternion,
    load_asset_manifest,
    realistic_half_extents,
)
from benchmark_private.primitives import (
    PRIMITIVE_TARGET_KEYS,
    REUSABLE_MANIPULABLE_PRIMITIVE_SCALE,
    STRATEGY_1_LARGE_OBSTACLE_KEYS,
    is_primitive_object,
    primitive_half_extents,
    primitive_inertia,
    primitive_mass,
    primitive_preset,
)
from benchmark_private.generator import (
    _body_collision_footprint_xy,
    materialize_case,
    tree_sha256,
)
from benchmark_private.materials import (
    APPEARANCE_MATERIAL,
    MATERIALS,
    STRUCTURAL_MATERIAL_KEYS,
    WORK_SURFACE_MATERIAL_KEYS,
    appearance_material,
    compatible_work_surface_material,
    effective_box_mass,
    material_texture_rgb,
    structural_material,
)
from benchmark_private.physics import (
    calibrated_drop_rebound,
    elasticity_solref,
)
from benchmark_private.specs import (
    ALL_APPEARANCES,
    BENCHMARK_VERSION,
    CASES_PER_STRATEGY,
    STRATEGY_2_GRIPPER_MAX_WIDTH_M,
    STRATEGY_2_TARGET_APPEARANCES,
    STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M,
    STRATEGY_3_GRIPPER_MAX_WIDTH_M,
    STRATEGY_3_TARGET_APPEARANCES,
    STRATEGY_4_TARGET_APPEARANCES,
    STRATEGY_5_TARGET_APPEARANCES,
    STRATEGY_7_TARGET_APPEARANCES,
    STRATEGY_8_TARGET_APPEARANCES,
    STRATEGY_6_TARGET_APPEARANCES,
    STRATEGY_TARGET_APPEARANCE_POOLS,
    STRATEGY_CONTRACTS,
    TARGET_APPEARANCES,
    sampling_fingerprint,
    strategy_2_manipulation_metrics,
    strategy_3_manipulation_metrics,
    variants_for_strategy,
)
from suites.staged.benchmark_runner import (
    _case_indices,
    _load_seal,
    _remap_authoritative_roles,
    _summary,
    _validated_resume_record,
    _wilson,
    build_parser,
)
from instantiation_core.runner import make_parser as make_instantiation_parser


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "runtime_assets" / "benchmark_hope_v1"

try:
    from primitive_core.native_fr3_osc import _load_native_types

    _load_native_types()
    NATIVE_FR3_OSC_AVAILABLE = True
except RuntimeError:
    NATIVE_FR3_OSC_AVAILABLE = False


class BenchmarkSamplingTests(unittest.TestCase):
    def test_strategy_two_sampling_stays_inside_flip_and_grasp_envelope(self) -> None:
        for parameters in variants_for_strategy(2):
            with self.subTest(case=parameters.case_index):
                self.assertIn(
                    parameters.appearance,
                    STRATEGY_2_TARGET_APPEARANCES,
                )
                material = appearance_material(parameters.appearance)
                half_extents = realistic_half_extents(
                    asset_record(ASSET_ROOT, parameters.appearance),
                    parameters.size_scale_xyz,
                )
                metrics = strategy_2_manipulation_metrics(
                    2.0 * half_extents,
                    mass_kg=effective_box_mass(
                        material,
                        half_extents,
                        scale=parameters.mass_scale,
                    ),
                    sliding_friction=(
                        material.sliding_friction * parameters.friction_scale
                    ),
                    gripper_max_width_m=STRATEGY_2_GRIPPER_MAX_WIDTH_M,
                )
                self.assertTrue(metrics["passed"], metrics)

    def test_strategy_two_gate_rejects_reported_failure_modes(self) -> None:
        examples = {
            "milk_too_thick_and_short": (
                (0.108, 0.110, 0.285),
                0.254,
                0.44,
                {"grasp_thickness", "upright_height"},
            ),
            "small_raisins_box": (
                (0.049, 0.106, 0.151),
                0.082,
                0.582,
                {"upright_height"},
            ),
            "thick_case": (
                (0.093, 0.140, 0.233),
                0.50,
                0.40,
                {"grasp_thickness"},
            ),
            "heavy_smooth_case_042": (
                (0.038, 0.150, 0.210),
                2.835,
                0.42,
                {"target_mass", "mass_friction_ratio"},
            ),
        }
        for name, (dimensions, mass, friction, expected_failures) in examples.items():
            with self.subTest(name=name):
                metrics = strategy_2_manipulation_metrics(
                    dimensions,
                    mass_kg=mass,
                    sliding_friction=friction,
                    gripper_max_width_m=STRATEGY_2_GRIPPER_MAX_WIDTH_M,
                )
                failures = {
                    key for key, passed in metrics["checks"].items() if not passed
                }
                self.assertFalse(metrics["passed"])
                self.assertTrue(expected_failures.issubset(failures), metrics)

    def test_strategy_three_grasp_gate_reserves_finger_clearance(self) -> None:
        fitting = strategy_3_manipulation_metrics(
            (0.075, 0.14, 0.22),
            gripper_max_width_m=STRATEGY_3_GRIPPER_MAX_WIDTH_M,
        )
        too_thick = strategy_3_manipulation_metrics(
            (0.077, 0.14, 0.22),
            gripper_max_width_m=STRATEGY_3_GRIPPER_MAX_WIDTH_M,
        )
        self.assertTrue(fitting["passed"], fitting)
        self.assertFalse(too_thick["passed"], too_thick)

    def test_runner_accepts_one_complete_strategy_from_a_partial_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seal_root = (
                root
                / "benchmark_private"
                / "sealed"
                / BENCHMARK_VERSION
            )
            seal_root.mkdir(parents=True)
            cases = [
                {"case_index": index, "certified": True}
                for index in range(1, CASES_PER_STRATEGY + 1)
            ]
            (seal_root / "seal.json").write_text(
                json.dumps(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "sealed": False,
                        "strategies": {
                            "2": {
                                "case_count": CASES_PER_STRATEGY,
                                "cases": cases,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            seal, records = _load_seal(root, 2)
            self.assertFalse(seal["sealed"])
            self.assertEqual(set(records), set(range(1, 51)))

    def test_runner_accepts_an_explicit_candidate_seal_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seal_root = root / "candidate-seal"
            seal_root.mkdir()
            cases = [
                {"case_index": index, "certified": True}
                for index in range(1, CASES_PER_STRATEGY + 1)
            ]
            (seal_root / "seal.json").write_text(
                json.dumps(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "sealed": False,
                        "strategies": {
                            "8": {
                                "case_count": CASES_PER_STRATEGY,
                                "cases": cases,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            seal, records = _load_seal(root, 8, seal_root=seal_root)

        self.assertFalse(seal["sealed"])
        self.assertEqual(set(records), set(range(1, 51)))

    def test_resume_rejects_a_different_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark_result.json"
            path.write_text(
                json.dumps(
                    {
                        "strategy_id": 1,
                        "case_index": 1,
                        "case_sha256": "case",
                        "checkpoint_sha256": "old",
                        "binding_mode": "authoritative",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                _validated_resume_record(
                    path,
                    strategy_id=1,
                    case_index=1,
                    case_sha256="case",
                    checkpoint_sha256="new",
                    binding_mode="authoritative",
                )

    def test_summary_keeps_diagnostic_and_stress_slices_separate(self) -> None:
        records = [
            {
                "program_success": True,
                "target_bound": True,
                "binding_valid": True,
                "authoritative_scene_role_coverage": 1.0,
                "bucket": "diagnostic",
                "diagnostic_dimension": "pose",
                "stress_axis": None,
            },
            {
                "program_success": False,
                "target_bound": True,
                "binding_valid": True,
                "authoritative_scene_role_coverage": 0.5,
                "bucket": "stress",
                "diagnostic_dimension": None,
                "stress_axis": "pose_size",
            },
        ]
        summary = _summary(records)
        self.assertEqual(summary["by_diagnostic_dimension"]["pose"]["successes"], 1)
        self.assertEqual(summary["by_stress_axis"]["pose_size"]["successes"], 0)
        self.assertEqual(summary["mean_authoritative_scene_role_coverage"], 0.75)

    def test_no_record_explicitly_disables_canonical_recording(self) -> None:
        args = make_instantiation_parser("test").parse_args(["--no-record"])
        self.assertTrue(args.no_record)
        self.assertIsNone(args.record)

    def test_authoritative_binding_is_default_and_remaps_target_entities(self) -> None:
        args = build_parser().parse_args(
            ["some_run", "--strategy-id", "1", "--strategy", "strategy_1"]
        )
        self.assertEqual(args.binding_mode, "authoritative")
        self.assertEqual(args.workspace_mode, "isolated")
        roles = {
            "target": "red_cheez_it_box",
            "geoms": ["red_cheez_it_box_collision"],
            "fixture": "brown_cardboard_box",
        }
        self.assertEqual(
            _remap_authoritative_roles(
                roles,
                source_target="red_cheez_it_box",
                case_target="granola_bars_box",
            ),
            {
                "target": "granola_bars_box",
                "geoms": ["granola_bars_box_collision"],
                "fixture": "brown_cardboard_box",
            },
        )

    def test_authoritative_binding_remaps_context_entities_too(self) -> None:
        roles = {
            "target": "cheez_it_box",
            "fixture": "white_divider",
            "obstacle_geoms": ["brown_cardboard_box_collision"],
        }
        self.assertEqual(
            _remap_authoritative_roles(
                roles,
                source_target="cheez_it_box",
                case_target="cookies_box",
                entity_renames={
                    "white_divider": "orange_divider",
                    "brown_cardboard_box": "milk_carton",
                },
            ),
            {
                "target": "cookies_box",
                "fixture": "orange_divider",
                "obstacle_geoms": ["milk_carton_collision"],
            },
        )
    def test_elasticity_mapping_has_measurable_monotonic_rebound(self) -> None:
        low_solref = elasticity_solref(0.01, 0.02)
        high_solref = elasticity_solref(0.01, 0.30)
        self.assertGreater(low_solref[1], high_solref[1])
        low_rebound = calibrated_drop_rebound(0.02)
        high_rebound = calibrated_drop_rebound(0.30)
        self.assertGreater(high_rebound - low_rebound, 0.002)

    def test_benchmark_runner_case_selection_is_explicit_and_bounded(self) -> None:
        self.assertEqual(_case_indices(None), tuple(range(1, 51)))
        self.assertEqual(_case_indices(["1,3", "5-7"]), (1, 3, 5, 6, 7))
        with self.assertRaisesRegex(ValueError, "1..50"):
            _case_indices(["0,51"])
        with self.assertRaisesRegex(ValueError, "ascending"):
            _case_indices(["7-5"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            _case_indices([""])

    def test_wilson_interval_contains_observed_rate(self) -> None:
        interval = _wilson(37, 50)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLessEqual(interval[0], 37 / 50)
        self.assertGreaterEqual(interval[1], 37 / 50)

    def test_each_strategy_has_fixed_diagnostic_mixed_stress_split(self) -> None:
        for strategy_id in sorted(STRATEGY_CONTRACTS):
            with self.subTest(strategy=strategy_id):
                variants = variants_for_strategy(strategy_id)
                self.assertEqual(len(variants), CASES_PER_STRATEGY)
                self.assertEqual([item.case_index for item in variants], list(range(1, 51)))
                self.assertEqual(sum(item.bucket == "diagnostic" for item in variants), 12)
                self.assertEqual(sum(item.bucket == "mixed" for item in variants), 28)
                self.assertEqual(sum(item.bucket == "stress" for item in variants), 10)
                dimensions = [
                    item.diagnostic_dimension
                    for item in variants
                    if item.bucket == "diagnostic"
                ]
                for dimension in (
                    "pose",
                    "size",
                    "appearance",
                    "mass",
                    "friction",
                    "elasticity",
                ):
                    self.assertEqual(dimensions.count(dimension), 2)

    def test_sampling_is_deterministic_and_strategy_specific(self) -> None:
        first = sampling_fingerprint(1)
        self.assertEqual(first, sampling_fingerprint(1))
        self.assertNotEqual(first, sampling_fingerprint(2))

    def test_parameter_ranges_match_the_sealed_contract(self) -> None:
        for strategy_id in STRATEGY_CONTRACTS:
            for variant in variants_for_strategy(strategy_id):
                self.assertTrue(all(0.68 <= value <= 1.36 for value in variant.size_scale_xyz))
                self.assertGreaterEqual(variant.mass_scale, 0.50)
                self.assertLessEqual(variant.mass_scale, 2.00)
                self.assertGreaterEqual(variant.friction_scale, 0.50)
                self.assertLessEqual(variant.friction_scale, 1.50)
                self.assertGreaterEqual(variant.restitution, 0.02)
                self.assertLessEqual(variant.restitution, 0.30)
                self.assertIn(variant.appearance, TARGET_APPEARANCES)
                self.assertEqual(len(variant.context_layout), 8)
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in variant.context_layout))
                self.assertNotIn(variant.appearance, variant.secondary_appearances)
                self.assertEqual(
                    len(variant.secondary_appearances),
                    len(set(variant.secondary_appearances)),
                )
                self.assertTrue(0 <= variant.fixture_material_index < 11)
                self.assertTrue(0.90 <= variant.fixture_scale <= 1.10)
                self.assertTrue(0.86 <= variant.lighting_intensity <= 1.14)
                self.assertTrue(-1.0 <= variant.lighting_temperature <= 1.0)

    def test_each_strategy_has_stratified_size_and_representation_diversity(self) -> None:
        for strategy_id in STRATEGY_CONTRACTS:
            variants = variants_for_strategy(strategy_id)
            with self.subTest(strategy=strategy_id):
                self.assertGreaterEqual(sum(item.size_band == "small" for item in variants), 8)
                self.assertGreaterEqual(sum(item.size_band == "large" for item in variants), 8)
                if strategy_id in (2, 3, 4, 5, 7, 8):
                    # Dense or thick material primitives violate the
                    # demonstrated manipulation envelopes. Strategies 3 and
                    # 7 likewise require enough planar area for a corridor
                    # pull or a graspable platform overhang.
                    expected_pool = {
                        2: STRATEGY_2_TARGET_APPEARANCES,
                        3: STRATEGY_3_TARGET_APPEARANCES,
                        4: STRATEGY_4_TARGET_APPEARANCES,
                        5: STRATEGY_5_TARGET_APPEARANCES,
                        7: STRATEGY_7_TARGET_APPEARANCES,
                        8: STRATEGY_8_TARGET_APPEARANCES,
                    }[strategy_id]
                    self.assertEqual(
                        {item.appearance for item in variants},
                        set(expected_pool),
                    )
                    self.assertFalse(
                        any(is_primitive_object(item.appearance) for item in variants)
                    )
                else:
                    self.assertGreaterEqual(
                        sum(
                            is_primitive_object(item.appearance)
                            for item in variants
                        ),
                        8,
                    )
                if strategy_id in STRATEGY_TARGET_APPEARANCE_POOLS:
                    self.assertTrue(
                        {item.appearance for item in variants}.issubset(
                            set(STRATEGY_TARGET_APPEARANCE_POOLS[strategy_id])
                        )
                    )
                self.assertEqual(
                    {item.fixture_material_index for item in variants},
                    set(range(11)),
                )
                self.assertTrue(
                    any(
                        any(is_primitive_object(value) for value in item.distractor_appearances)
                        for item in variants
                    )
                )

    def test_each_strategy_has_an_independent_private_generator(self) -> None:
        modules = []
        for strategy_id in sorted(STRATEGY_CONTRACTS):
            module = importlib.import_module(
                f"benchmark_private.strategies.strategy_{strategy_id}"
            )
            self.assertTrue(callable(module.apply))
            modules.append(module.apply.__module__)
        self.assertEqual(len(set(modules)), len(STRATEGY_CONTRACTS))

    def test_strategy_one_samples_only_large_corridor_obstacles(self) -> None:
        variants = variants_for_strategy(1)
        sampled = {
            appearance
            for variant in variants
            for appearance in variant.secondary_appearances
        }
        self.assertEqual(sampled, set(STRATEGY_1_LARGE_OBSTACLE_KEYS))
        for variant in variants:
            self.assertNotIn(
                variant.secondary_appearances[0],
                variant.distractor_appearances,
            )

    def test_strategy_context_pools_match_their_structural_roles(self) -> None:
        large_channel = {
            appearance
            for variant in variants_for_strategy(3)
            for appearance in variant.secondary_appearances
        }
        self.assertEqual(large_channel, set(STRATEGY_1_LARGE_OBSTACLE_KEYS))
        for strategy_id in (5, 6):
            sampled = {
                appearance
                for variant in variants_for_strategy(strategy_id)
                for appearance in variant.secondary_appearances
            }
            self.assertEqual(sampled, set(STRATEGY_1_LARGE_OBSTACLE_KEYS))

    def test_strategy_three_and_seven_targets_keep_a_large_planar_footprint(self) -> None:
        for strategy_id in (3, 8):
            for variant in variants_for_strategy(strategy_id):
                half_extents = np.sort(
                    realistic_half_extents(
                        asset_record(ASSET_ROOT, variant.appearance),
                        variant.size_scale_xyz,
                    )
                )
                planar_area = float(4.0 * half_extents[1] * half_extents[2])
                with self.subTest(
                    strategy=strategy_id,
                    case=variant.case_index,
                ):
                    self.assertGreaterEqual(planar_area, 0.034)


class BenchmarkAssetTests(unittest.TestCase):
    def test_curated_assets_are_complete_and_versioned(self) -> None:
        manifest = load_asset_manifest(ASSET_ROOT)
        self.assertEqual(manifest["benchmark_version"], BENCHMARK_VERSION)
        self.assertEqual(
            {item["asset_name"] for item in manifest["assets"]},
            set(ALL_APPEARANCES),
        )

    def test_curated_mesh_is_centered_and_axis_normalized(self) -> None:
        path = ASSET_ROOT / "granola_bars" / "visual.obj"
        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("v ")
            ]
        )
        np.testing.assert_allclose(vertices.min(axis=0), -np.ones(3), atol=2e-7)
        np.testing.assert_allclose(vertices.max(axis=0), np.ones(3), atol=2e-7)

    def test_hope_upright_transform_maps_source_y_to_world_z(self) -> None:
        quaternion = hope_upright_quaternion()
        w, x, y, z = quaternion
        rotation = np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
        )
        np.testing.assert_allclose(
            rotation @ np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            atol=1e-12,
        )

    def test_realistic_extents_restore_product_proportions_and_bound_jitter(self) -> None:
        milk = asset_record(ASSET_ROOT, "Milk")
        yogurt = asset_record(ASSET_ROOT, "Yogurt")
        milk_half = realistic_half_extents(milk)
        yogurt_half = realistic_half_extents(yogurt)
        np.testing.assert_allclose(
            milk_half / milk_half.max(),
            np.asarray(milk["source_extent"]) / max(milk["source_extent"]),
        )
        self.assertAlmostEqual(
            float(2.0 * milk_half.max()),
            HOPE_REUSABLE_ASSET_SCALE * 0.19028359,
        )
        self.assertAlmostEqual(
            float(2.0 * yogurt_half.max()),
            HOPE_REUSABLE_ASSET_SCALE * 0.06795305,
        )

        distorted_request = realistic_half_extents(milk, (0.80, 1.20, 1.00))
        restored_ratio = distorted_request / milk_half
        self.assertGreaterEqual(
            float(restored_ratio.min()),
            ASSET_UNIFORM_SCALE_RANGE[0] * (1.0 - ASSET_ASPECT_JITTER),
        )
        self.assertLessEqual(
            float(restored_ratio.max()),
            ASSET_UNIFORM_SCALE_RANGE[1] * (1.0 + ASSET_ASPECT_JITTER),
        )

    def test_category_aware_asset_size_ranges_keep_products_recognizable(self) -> None:
        self.assertEqual(set(ASSET_CATEGORIES), set(ALL_APPEARANCES))
        for appearance in ALL_APPEARANCES:
            record = asset_record(ASSET_ROOT, appearance)
            baseline = realistic_half_extents(record)
            lower, upper = ASSET_SCALE_RANGES[ASSET_CATEGORIES[appearance]]
            small = realistic_half_extents(record, (0.60, 0.60, 0.60))
            large = realistic_half_extents(record, (1.50, 1.50, 1.50))
            np.testing.assert_allclose(small / baseline, lower)
            np.testing.assert_allclose(large / baseline, upper)

    def test_reusable_material_primitives_use_role_independent_catalog_scale(self) -> None:
        target_preset = primitive_preset("PowderCoatedMetalCase")
        target_dimensions = 2.0 * primitive_half_extents(
            target_preset,
            broad=False,
        )
        np.testing.assert_allclose(
            target_dimensions,
            REUSABLE_MANIPULABLE_PRIMITIVE_SCALE
            * np.asarray(target_preset.dimensions_m),
        )

        structural_preset = primitive_preset("LargeMetalStorageCase")
        structural_dimensions = 2.0 * primitive_half_extents(
            structural_preset,
            broad=False,
        )
        np.testing.assert_allclose(
            structural_dimensions,
            np.asarray(structural_preset.dimensions_m),
        )


class BenchmarkMaterialCatalogTests(unittest.TestCase):
    def test_every_hope_appearance_has_packaging_physics(self) -> None:
        self.assertEqual(set(APPEARANCE_MATERIAL), set(ALL_APPEARANCES))
        for appearance in ALL_APPEARANCES:
            with self.subTest(appearance=appearance):
                material = appearance_material(appearance)
                self.assertEqual(material.use, "packaging")
                self.assertIn(material.key, MATERIALS)

    def test_structural_catalog_has_bounded_simulation_properties(self) -> None:
        self.assertEqual(len(STRUCTURAL_MATERIAL_KEYS), 11)
        self.assertEqual(len(set(STRUCTURAL_MATERIAL_KEYS)), 11)
        for index, key in enumerate(STRUCTURAL_MATERIAL_KEYS):
            with self.subTest(material=key):
                material = structural_material(index)
                self.assertEqual(material.key, key)
                self.assertEqual(material.use, "structural")
                self.assertGreater(material.effective_density_kg_m3, 0.0)
                self.assertTrue(0.0 < material.sliding_friction < 1.0)
                self.assertTrue(0.0 <= material.restitution <= 0.3)
                self.assertTrue(0.0 <= material.specular <= 1.0)
                self.assertTrue(0.0 <= material.reflectance <= 1.0)
                self.assertTrue(0.0 < material.opacity <= 1.0)

    def test_procedural_material_texture_is_deterministic_and_nonuniform(self) -> None:
        material = MATERIALS["birch_wood"]
        first = material_texture_rgb(material, seed=123, width=64, height=64)
        second = material_texture_rgb(material, seed=123, width=64, height=64)
        other = material_texture_rgb(material, seed=124, width=64, height=64)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(first.std()), 4.0)
        self.assertFalse(np.array_equal(first, other))

    def test_effective_box_mass_respects_material_mass_bounds(self) -> None:
        material = MATERIALS["coated_carton"]
        tiny = effective_box_mass(material, np.asarray((1e-4, 1e-4, 1e-4)))
        large = effective_box_mass(material, np.asarray((1.0, 1.0, 1.0)))
        self.assertEqual(tiny, material.mass_range_kg[0])
        self.assertEqual(large, material.mass_range_kg[1])

    def test_work_surface_selection_avoids_fixture_family_and_low_contrast(self) -> None:
        self.assertGreaterEqual(len(WORK_SURFACE_MATERIAL_KEYS), 5)
        for index in range(len(STRUCTURAL_MATERIAL_KEYS)):
            fixture = structural_material(index)
            surface = compatible_work_surface_material((fixture,), seed=1000 + index)
            with self.subTest(fixture=fixture.key, surface=surface.key):
                self.assertIn(surface.key, WORK_SURFACE_MATERIAL_KEYS)
                self.assertNotEqual(surface.key, fixture.key)
                self.assertNotEqual(surface.family, fixture.family)


class BenchmarkCertifierTests(unittest.TestCase):
    def test_distractor_visibility_is_diagnostic_only(self) -> None:
        self.assertEqual(
            [_required_visible_distractor_count(value) for value in range(5)],
            [0, 0, 0, 0, 0],
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _required_visible_distractor_count(-1)


class BenchmarkMaterializationTests(unittest.TestCase):
    def test_strategy_four_preserves_a_real_post_flip_pull(self) -> None:
        contract = STRATEGY_CONTRACTS[4]
        parameters = variants_for_strategy(4)[0]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            source_half = realistic_half_extents(
                asset_record(ASSET_ROOT, parameters.appearance),
                parameters.size_scale_xyz,
            )
            expected_half = np.sort(source_half)
            expected_half[1] *= 0.65
            np.testing.assert_allclose(
                np.sort(private["target_half_extents"]),
                expected_half,
            )
            expected_wall_center = (
                np.asarray((0.75, 0.19))
                + np.asarray(parameters.translation_xy)
                + np.asarray(
                    (
                        0.018 * parameters.context_layout[0],
                        0.018 * parameters.context_layout[1],
                    )
                )
            )
            np.testing.assert_allclose(
                private["context_layout"]["wall_flip"]["wall_center_xy"],
                expected_wall_center,
            )
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _wall_flip_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["pull_phase_required"], metrics)
            self.assertTrue(
                metrics["checks"]["post_pull_finger_clearance"],
                metrics,
            )
            self.assertTrue(metrics["checks"]["post_pull_in_workspace"], metrics)

    def test_strategy_three_uses_a_full_corridor_and_demo_side_wall(self) -> None:
        contract = STRATEGY_CONTRACTS[3]
        parameters = variants_for_strategy(3)[2]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _extraction_then_wall_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["target_between_obstacles"], metrics)
            self.assertTrue(metrics["checks"]["target_axially_contained"], metrics)
            self.assertTrue(metrics["checks"]["target_planar_area"], metrics)
            self.assertTrue(
                metrics["checks"]["wall_parallel_to_extraction"],
                metrics,
            )
            self.assertTrue(metrics["checks"]["wall_on_expected_side"], metrics)
            self.assertTrue(
                metrics["checks"]["post_flip_grasp_edge_exposed"],
                metrics,
            )
            self.assertGreaterEqual(metrics["post_flip_edge_exposure"], 0.050)
            self.assertLess(metrics["fixture_height"], metrics["post_flip_target_height"])

    def test_strategy_three_camera_is_fixed_across_sampled_poses(self) -> None:
        contract = STRATEGY_CONTRACTS[3]
        cameras = []
        target_centers = []
        with tempfile.TemporaryDirectory() as tmp:
            for parameters in variants_for_strategy(3)[:2]:
                case_dir = Path(tmp) / parameters.case_name
                private = materialize_case(
                    REPO_ROOT,
                    ASSET_ROOT,
                    contract,
                    parameters,
                    destination=case_dir,
                )
                cameras.append(_load_scene(case_dir).DEFAULT_CAMERA)
                target_centers.append(
                    private["context_layout"]["extraction_channel"][
                        "target_initial_center_xy"
                    ]
                )

        self.assertNotEqual(target_centers[0], target_centers[1])
        self.assertEqual(cameras[0], cameras[1])

    def test_strategy_three_oversized_target_is_fitted_to_primitive_domain(self) -> None:
        contract = STRATEGY_CONTRACTS[3]
        parameters = variants_for_strategy(3)[13]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            layout = private["context_layout"]["extraction_channel"]
            relation = private["relation_checks"][0]
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)
            metrics = _extraction_then_wall_metrics(model, data, relation)

        self.assertLess(layout["target_geometry_scale"], 1.0)
        self.assertLessEqual(
            layout["primitive_extraction_distance"],
            STRATEGY_3_DESIGN_MAX_EXTRACTION_DISTANCE_M + 1e-9,
        )
        self.assertTrue(
            metrics["checks"]["primitive_extraction_distance"],
            metrics,
        )
        self.assertAlmostEqual(
            metrics["primitive_extraction_distance"],
            layout["primitive_extraction_distance"],
            places=6,
        )
        np.testing.assert_allclose(
            metrics["primitive_extraction_axis_xy"],
            relation["extraction_axis_xy"],
            atol=1e-8,
        )

    def test_strategy_seven_provides_a_graspable_overhang_area(self) -> None:
        contract = STRATEGY_CONTRACTS[8]
        parameters = variants_for_strategy(8)[2]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _platform_edge_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["initial_lateral_support"], metrics)
            self.assertTrue(metrics["checks"]["initial_forward_support"], metrics)
            self.assertTrue(metrics["checks"]["stable_post_slide_support"], metrics)
            self.assertTrue(metrics["checks"]["graspable_overhang_area"], metrics)
            self.assertTrue(metrics["checks"]["target_planar_area"], metrics)
            self.assertTrue(metrics["checks"]["platform_height"], metrics)
            self.assertGreaterEqual(metrics["exposed_grasp_area"], 0.006)
            self.assertGreaterEqual(metrics["platform_height"], 0.178)

    def test_strategy_five_exposes_a_flip_wall_beside_the_flat_target(self) -> None:
        contract = STRATEGY_CONTRACTS[5]
        parameters = variants_for_strategy(5)[0]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _slot_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["flip_wall_exposed"], metrics)
            self.assertTrue(
                metrics["checks"]["target_overlaps_exposed_flip_wall"],
                metrics,
            )
            self.assertGreater(metrics["flip_wall_exposure"], 0.10)
            self.assertGreater(metrics["target_flip_wall_overlap"], 0.08)
            self.assertTrue(metrics["checks"]["push_phase_required"], metrics)
            self.assertGreaterEqual(metrics["initial_push_distance"], 0.05)

            insufficient_exposure = {
                **relation,
                "minimum_flip_wall_exposure": (
                    metrics["flip_wall_exposure"] + 0.01
                ),
            }
            rejected = _slot_metrics(model, data, insufficient_exposure)
            self.assertFalse(rejected["checks"]["flip_wall_exposed"], rejected)

    def test_strategy_six_packs_short_edges_like_books_on_a_shelf(self) -> None:
        contract = STRATEGY_CONTRACTS[7]
        parameters = variants_for_strategy(7)[0]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _packed_row_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["short_edge_packing"], metrics)
            self.assertTrue(
                metrics["checks"]["broad_faces_point_outward"],
                metrics,
            )
            self.assertEqual(relation["row_axis_dimension_rank"], 0)
            self.assertEqual(relation["outward_axis_dimension_rank"], 1)
            self.assertTrue(metrics["checks"]["fixtures_are_low"], metrics)
            self.assertTrue(metrics["checks"]["fixtures_are_shallow"], metrics)
            self.assertTrue(
                metrics["checks"]["row_is_at_robot_side_opening"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["target_wide_enough_for_topple_contact"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["target_has_broad_topple_face"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["row_height_consistency"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["target_top_not_initially_exposed"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["target_front_not_initially_exposed"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["no_direct_side_grasp_clearance"],
                metrics,
            )
            self.assertGreaterEqual(
                metrics["target_topple_contact_width"],
                0.069,
            )
            self.assertGreaterEqual(metrics["target_topple_face_depth"], 0.178)
            self.assertLessEqual(max(metrics["target_side_gaps"]), 0.005)
            self.assertLessEqual(metrics["target_top_exposure"], 0.002)
            self.assertLessEqual(metrics["target_front_exposure"], 0.002)
            self.assertLessEqual(max(metrics["fixture_heights"]), 0.116)
            for margin in metrics["fixture_opening_edge_margins"]:
                self.assertAlmostEqual(margin, 0.012, places=5)

            wrong_axis_relation = {
                **relation,
                "lateral_xy": relation["outward_xy"],
                "outward_xy": relation["lateral_xy"],
            }
            wrong_metrics = _packed_row_metrics(
                model,
                data,
                wrong_axis_relation,
            )
            self.assertFalse(
                wrong_metrics["checks"]["short_edge_packing"],
                wrong_metrics,
            )

    def test_push_and_horizontal_grasp_paths_keep_distractors_away(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for strategy_id in (1, 8, 6):
                with self.subTest(strategy=strategy_id):
                    contract = STRATEGY_CONTRACTS[strategy_id]
                    parameters = variants_for_strategy(strategy_id)[0]
                    case_dir = Path(tmp) / f"strategy_{strategy_id}"
                    private = materialize_case(
                        REPO_ROOT,
                        ASSET_ROOT,
                        contract,
                        parameters,
                        destination=case_dir,
                    )
                    layout = private["context_layout"]
                    clearance = float(
                        layout["distractor_manipulation_clearance"]
                    )
                    self.assertGreaterEqual(clearance, 0.14)
                    regions = layout["distractor_exclusion_regions"]
                    scene = ET.parse(case_dir / "bundle" / "scene.xml").getroot()
                    for name in private["authoritative_roles"]["distractors"]:
                        center, half = _body_collision_footprint_xy(scene, name)
                        for region in regions:
                            protected_center = np.asarray(region["center_xy"])
                            protected_half = np.asarray(
                                region["half_extents_xy"]
                            )
                            overlaps_clearance = np.all(
                                np.abs(center - protected_center)
                                < half + protected_half + clearance - 1e-8
                            )
                            self.assertFalse(
                                overlaps_clearance,
                                (strategy_id, name, region),
                            )

    def test_strategy_eight_requires_a_quarter_turn_before_insertion(self) -> None:
        contract = STRATEGY_CONTRACTS[6]
        parameters = variants_for_strategy(6)[0]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            relation = private["relation_checks"][0]
            source_half = realistic_half_extents(
                asset_record(ASSET_ROOT, parameters.appearance),
                parameters.size_scale_xyz,
            )
            expected_half = np.sort(source_half)
            expected_half[1] *= 0.5
            expected_half[0] = min(
                expected_half[0],
                0.9 * expected_half[1],
            )
            np.testing.assert_allclose(
                np.sort(private["target_half_extents"]),
                expected_half,
            )
            scene = ET.parse(case_dir / "bundle" / "scene.xml").getroot()
            self.assertEqual(
                private["authoritative_roles"]["fixtures"],
                [relation["left"], relation["right"]],
            )
            self.assertEqual(
                private["authoritative_roles"]["secondary_dynamic"],
                [],
            )
            for boundary_name in (relation["left"], relation["right"]):
                boundary = scene.find(f".//body[@name='{boundary_name}']")
                self.assertIsNotNone(boundary)
                assert boundary is not None
                self.assertIsNone(boundary.find("freejoint"))
                self.assertIsNone(boundary.find("inertial"))
            module = _load_scene(case_dir)
            model = module.make_model()
            data = module.make_data(model)

            metrics = _slot_metrics(model, data, relation)
            self.assertTrue(metrics["checks"]["target_alignment"], metrics)
            self.assertTrue(
                metrics["checks"]["initially_too_wide_for_slot"],
                metrics,
            )
            self.assertTrue(
                metrics["checks"]["aligned_pose_stays_outside"],
                metrics,
            )
            self.assertTrue(metrics["checks"]["pivot_sweep_in_workspace"], metrics)
            self.assertTrue(
                metrics["checks"]["pivot_sweep_obstacle_clearance"],
                metrics,
            )
            self.assertGreaterEqual(
                abs(metrics["target_alignment_error_radians"]),
                np.deg2rad(70.0),
            )
            self.assertGreaterEqual(
                metrics["minimum_pivot_obstacle_clearance_actual"],
                0.147,
            )

    def test_each_strategy_materializes_its_private_relation_contract(self) -> None:
        expected_kinds = {
            1: "strategy_1_corridor",
            2: "wall_flip_setup",
            3: "extraction_then_wall_setup",
            4: "wall_flip_setup",
            5: "slot_setup",
            7: "packed_upright_row",
            8: "platform_edge_setup",
            6: "slot_setup",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for strategy_id, expected_kind in expected_kinds.items():
                with self.subTest(strategy=strategy_id):
                    contract = STRATEGY_CONTRACTS[strategy_id]
                    parameters = variants_for_strategy(strategy_id)[0]
                    case_dir = (
                        Path(tmp)
                        / f"strategy_{strategy_id}"
                        / parameters.case_name
                    )
                    private = materialize_case(
                        REPO_ROOT,
                        ASSET_ROOT,
                        contract,
                        parameters,
                        destination=case_dir,
                    )
                    self.assertEqual(
                        private["relation_checks"][0]["kind"],
                        expected_kind,
                    )
                    self.assertTrue(
                        private["context_layout"].get("canonical_supports")
                    )

    def test_hope_distractors_are_upright_and_static(self) -> None:
        contract = STRATEGY_CONTRACTS[1]
        variants = variants_for_strategy(1)
        parameters = next(
            item
            for item in variants
            if "Ketchup" in item.distractor_appearances
        )
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            scene = ET.parse(case_dir / "bundle" / "scene.xml").getroot()
            distractor = private["authoritative_roles"]["distractors"][0]
            body = scene.find(f".//body[@name='{distractor}']")
            self.assertIsNotNone(body)
            assert body is not None
            visual = body.find("geom[@group='2']")
            collision = body.find("geom[@group='3']")
            inertial = body.find("inertial")
            self.assertIsNotNone(visual)
            self.assertIsNotNone(collision)
            self.assertIsNone(inertial)
            self.assertIsNone(body.find("freejoint"))
            assert visual is not None and collision is not None

            expected = hope_upright_quaternion()
            for element in (visual, collision):
                generated = np.asarray(
                    [float(value) for value in element.get("quat", "").split()]
                )
                np.testing.assert_allclose(generated, expected, atol=1e-8)

            half_extents = np.asarray(
                [float(value) for value in collision.get("size", "").split()]
            )
            body_z = float(body.get("pos", "0 0 0").split()[2])
            self.assertAlmostEqual(body_z, float(half_extents[1] + 1e-4))

    @unittest.skipUnless(
        NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_case_040_runtime_gate_rejects_the_old_dynamic_cylinder(self) -> None:
        contract = STRATEGY_CONTRACTS[2]
        parameters = variants_for_strategy(2)[39]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            scene_path = case_dir / "bundle" / "scene.xml"
            tree = ET.parse(scene_path)
            root = tree.getroot()
            body = root.find(".//body[@name='porcelain_cup_2']")
            self.assertIsNotNone(body)
            assert body is not None
            collision = body.find("geom[@group='3']")
            self.assertIsNotNone(collision)
            assert collision is not None
            raw_size = np.asarray(
                [float(value) for value in collision.get("size", "").split()]
            )
            half_extents = np.asarray(
                (raw_size[0], raw_size[0], raw_size[1])
            )
            preset = primitive_preset("PorcelainCup")
            mass = primitive_mass(preset, half_extents)
            inertia = primitive_inertia(preset, half_extents, mass)
            body.insert(
                0,
                ET.Element(
                    "inertial",
                    mass=f"{mass:.9g}",
                    diaginertia=" ".join(f"{value:.9g}" for value in inertia),
                    pos="0 0 0",
                ),
            )
            body.insert(
                0,
                ET.Element("freejoint", name="porcelain_cup_2_joint"),
            )
            ET.indent(tree, space="  ")
            tree.write(scene_path, encoding="unicode")

            gate, _, _ = _settling_gate(_load_scene(case_dir), private)
            cylinder = gate.metrics["free_bodies"]["porcelain_cup_2"]
            self.assertFalse(gate.passed)
            self.assertGreater(cylinder["max_orientation_drift_radians"], 1.0)
            self.assertGreater(cylinder["max_xy_drift"], 0.05)

    @unittest.skipUnless(
        NATIVE_FR3_OSC_AVAILABLE,
        "native FR3 OSC extension has not been built",
    )
    def test_case_040_static_distractors_pass_runtime_stability(self) -> None:
        contract = STRATEGY_CONTRACTS[2]
        parameters = variants_for_strategy(2)[39]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            gate, _, _ = _settling_gate(_load_scene(case_dir), private)
            self.assertTrue(gate.passed, gate.metrics)
            self.assertTrue(
                all(gate.metrics["static_distractor_checks"].values())
            )

    def test_materialized_case_is_portable_and_does_not_leak_latents(self) -> None:
        contract = STRATEGY_CONTRACTS[1]
        parameters = variants_for_strategy(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private_record = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            tree = ET.parse(case_dir / "bundle" / "scene.xml")
            root = tree.getroot()
            target = private_record["authoritative_roles"]["target"]
            body = root.find(f".//body[@name='{target}']")
            self.assertIsNotNone(body)
            assert body is not None
            visual = body.find("geom[@type='mesh']")
            collision = body.find("geom[@type='box']")
            self.assertIsNotNone(visual)
            self.assertIsNotNone(collision)
            assert visual is not None and collision is not None
            self.assertEqual(visual.get("contype"), "0")
            self.assertEqual(visual.get("conaffinity"), "0")
            self.assertEqual(visual.get("density"), "0")
            self.assertEqual(visual.get("group"), "2")
            self.assertEqual(collision.get("density"), "0")
            self.assertEqual(collision.get("group"), "3")
            self.assertTrue((case_dir / "goal.txt").is_file())
            self.assertTrue((case_dir / "bundle" / "textures").is_dir())
            public = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in case_dir.rglob("*")
                if path.is_file() and path.suffix in {".py", ".json", ".txt", ".xml"}
            )
            self.assertNotIn("authoritative_roles", public)
            self.assertNotIn("mass_scale", public)
            self.assertEqual(tree_sha256(case_dir), tree_sha256(case_dir))
            metadata = json.loads(
                (case_dir / "bundle" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertIn(target, {item["safe_name"] for item in metadata["objects"]})
            target_metadata = next(
                item for item in metadata["objects"] if item["safe_name"] == target
            )
            target_material = appearance_material(parameters.appearance)
            self.assertEqual(target_metadata["material"], target_material.key)
            friction = [float(value) for value in collision.get("friction", "").split()]
            self.assertAlmostEqual(
                friction[0],
                target_material.sliding_friction * parameters.friction_scale,
            )
            source = asset_record(ASSET_ROOT, parameters.appearance)
            generated_half = np.asarray(
                [float(value) for value in collision.get("size", "").split()]
            )
            expected_half = realistic_half_extents(
                source,
                parameters.size_scale_xyz,
            )
            np.testing.assert_allclose(generated_half, expected_half)

            floor = root.find(".//geom[@name='floor']")
            self.assertIsNotNone(floor)
            assert floor is not None
            floor_position = [
                float(value) for value in floor.get("pos", "0 0 0").split()
            ]
            self.assertEqual(floor_position[2], 0.0)
            self.assertEqual(
                private_record["context_layout"]["world_frame"]["floor_z"],
                0.0,
            )
            self.assertEqual(
                private_record["context_layout"]["world_frame"]["robot_base_z"],
                0.0,
            )
            self.assertGreater(
                private_record["context_layout"]["world_frame"]["source_floor_z"],
                0.0,
            )
            floor_material_name = floor.get("material")
            self.assertIsNotNone(floor_material_name)
            self.assertTrue(str(floor_material_name).startswith("work_surface_"))
            self.assertIsNone(root.find("./asset/texture[@name='grid']"))
            self.assertIsNone(root.find("./asset/material[@name='grid']"))
            surface_record = private_record["context_layout"][
                "surface_materials"
            ]["floor"]
            floor_friction = [
                float(value) for value in floor.get("friction", "").split()
            ]
            np.testing.assert_allclose(
                floor_friction,
                (
                    surface_record["sliding_friction"],
                    surface_record["torsional_friction"],
                    surface_record["rolling_friction"],
                ),
            )

    def test_material_primitive_target_uses_matching_visual_physics_and_scale(self) -> None:
        contract = STRATEGY_CONTRACTS[1]
        parameters = next(
            item
            for item in variants_for_strategy(1)
            if item.appearance in PRIMITIVE_TARGET_KEYS and item.size_band == "regular"
        )
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / parameters.case_name
            private = materialize_case(
                REPO_ROOT,
                ASSET_ROOT,
                contract,
                parameters,
                destination=case_dir,
            )
            scene = ET.parse(case_dir / "bundle" / "scene.xml").getroot()
            target = private["authoritative_roles"]["target"]
            body = scene.find(f".//body[@name='{target}']")
            self.assertIsNotNone(body)
            assert body is not None
            visual = body.find("geom[@group='2']")
            collision = body.find("geom[@group='3']")
            self.assertIsNotNone(visual)
            self.assertIsNotNone(collision)
            assert visual is not None and collision is not None
            self.assertEqual(visual.get("type"), "box")
            self.assertEqual(collision.get("type"), "box")
            self.assertIsNone(visual.get("mesh"))
            self.assertEqual(private["target_representation"]["visual_source"], "material_primitive")
            preset = primitive_preset(parameters.appearance)
            expected = primitive_half_extents(
                preset,
                parameters.size_scale_xyz,
                broad=True,
            )
            generated = np.asarray(
                [float(value) for value in collision.get("size", "").split()]
            )
            np.testing.assert_allclose(generated, expected)
            self.assertEqual(
                private["material_assignments"]["target"],
                preset.material_key,
            )
            texture = scene.find(
                f"./asset/texture[@name='{target}_{preset.material_key}_texture']"
            )
            self.assertIsNotNone(texture)
            assert texture is not None
            self.assertTrue(
                (case_dir / "bundle" / str(texture.get("file"))).is_file()
            )

    def test_context_layout_fixture_material_and_obstacle_skin_are_materialized(self) -> None:
        contract = STRATEGY_CONTRACTS[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for parameters in (variants_for_strategy(1)[0], variants_for_strategy(1)[4]):
                case_dir = root / parameters.case_name
                private = materialize_case(
                    REPO_ROOT,
                    ASSET_ROOT,
                    contract,
                    parameters,
                    destination=case_dir,
                )
                tree = ET.parse(case_dir / "bundle" / "scene.xml")
                scene = tree.getroot()
                fixture = private["authoritative_roles"]["fixtures"][0]
                secondary = private["authoritative_roles"]["secondary_dynamic"][0]
                fixture_body = scene.find(f".//body[@name='{fixture}']")
                secondary_body = scene.find(f".//body[@name='{secondary}']")
                self.assertIsNotNone(fixture_body)
                self.assertIsNotNone(secondary_body)
                assert fixture_body is not None and secondary_body is not None
                secondary_visual = secondary_body.find("geom[@group='2']")
                secondary_collision = secondary_body.find("geom[@group='3']")
                self.assertIsNotNone(secondary_visual)
                self.assertIsNotNone(secondary_collision)
                assert secondary_visual is not None
                assert secondary_collision is not None
                self.assertEqual(secondary_visual.get("type"), "box")
                self.assertEqual(secondary_collision.get("type"), "box")
                fixture_visual = fixture_body.find("geom[@group='2']")
                fixture_collision = fixture_body.find("geom[@type='box']")
                self.assertIsNotNone(fixture_visual)
                self.assertIsNotNone(fixture_collision)
                assert fixture_visual is not None and fixture_collision is not None
                self.assertEqual(fixture_visual.get("type"), "box")
                self.assertEqual(
                    fixture_visual.get("size"),
                    fixture_collision.get("size"),
                )
                self.assertEqual(
                    fixture_visual.get("pos"),
                    fixture_collision.get("pos"),
                )
                self.assertEqual(
                    fixture_visual.get("quat"),
                    fixture_collision.get("quat"),
                )
                material_name = fixture_visual.get("material")
                material = scene.find(f"./asset/material[@name='{material_name}']")
                self.assertIsNotNone(material)
                assert material is not None
                self.assertEqual(material.get("rgba"), "1 1 1 1")
                self.assertEqual(material.get("texuniform"), "true")
                texture_name = material.get("texture")
                texture = scene.find(f"./asset/texture[@name='{texture_name}']")
                self.assertIsNotNone(texture)
                assert texture is not None
                texture_file = case_dir / "bundle" / str(texture.get("file"))
                self.assertTrue(texture_file.is_file())
                material_record = private["context_layout"]["fixture_materials"][fixture]
                collision_friction = [
                    float(value)
                    for value in fixture_collision.get("friction", "").split()
                ]
                np.testing.assert_allclose(
                    collision_friction,
                    (
                        material_record["sliding_friction"],
                        material_record["torsional_friction"],
                        material_record["rolling_friction"],
                    ),
                )
                self.assertIn(
                    secondary,
                    private["context_layout"]["object_materials"],
                )
                records.append(
                    (
                        fixture,
                        fixture_body.get("pos"),
                        material_record["key"],
                        texture_file.read_bytes(),
                        secondary,
                    )
                )
            self.assertNotEqual(records[0][1], records[1][1])
            self.assertNotEqual(records[0][0], records[1][0])
            self.assertNotEqual(records[0][2], records[1][2])
            self.assertNotEqual(records[0][3], records[1][3])
            self.assertNotEqual(records[0][4], records[1][4])

    def test_strategy_one_materializes_a_target_sized_parallel_corridor(self) -> None:
        contract = STRATEGY_CONTRACTS[1]
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for case_index in (1, 2, 15, 20):
                parameters = variants_for_strategy(1)[case_index - 1]
                case_dir = Path(tmp) / parameters.case_name
                private = materialize_case(
                    REPO_ROOT,
                    ASSET_ROOT,
                    contract,
                    parameters,
                    destination=case_dir,
                )
                corridor = private["context_layout"]["corridor"]
                self.assertGreaterEqual(
                    corridor["clear_width"],
                    corridor["required_clear_width"],
                )
                self.assertLessEqual(
                    corridor["clear_width"],
                    corridor["maximum_clear_width"],
                )
                self.assertGreaterEqual(
                    corridor["available_store_depth"],
                    corridor["required_store_depth"],
                )
                self.assertLess(corridor["target_initial_lateral_offset"], 0.0)
                requirement = private["relation_checks"][0]
                self.assertEqual(requirement["kind"], "strategy_1_corridor")
                self.assertEqual(
                    requirement["obstacle"],
                    private["authoritative_roles"]["secondary_dynamic"][0],
                )
                module = _load_scene(case_dir)
                model = module.make_model()
                data = module.make_data(model)
                metrics = _strategy_1_corridor_metrics(
                    model,
                    data,
                    requirement,
                )
                self.assertTrue(metrics["checks"]["store_starts_at_entrance"], metrics)
                self.assertTrue(metrics["checks"]["store_depth"], metrics)
                records.append(corridor)
            self.assertLess(records[0]["corridor_yaw_radians"], 0.0)
            self.assertGreater(records[1]["corridor_yaw_radians"], 0.0)

    def test_strategy_one_causal_boxes_start_flush_and_upright(self) -> None:
        def rotation(raw: str | None) -> np.ndarray:
            w, x, y, z = np.asarray(
                [float(value) for value in (raw or "1 0 0 0").split()]
            )
            return np.asarray(
                (
                    (
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w),
                    ),
                    (
                        2 * (x * y + z * w),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),
                    ),
                    (
                        2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x * x + y * y),
                    ),
                )
            )

        contract = STRATEGY_CONTRACTS[1]
        with tempfile.TemporaryDirectory() as tmp:
            for case_index in (1, 15, 24):
                parameters = variants_for_strategy(1)[case_index - 1]
                case_dir = Path(tmp) / parameters.case_name
                private = materialize_case(
                    REPO_ROOT,
                    ASSET_ROOT,
                    contract,
                    parameters,
                    destination=case_dir,
                )
                scene = ET.parse(case_dir / "bundle" / "scene.xml").getroot()
                roles = private["authoritative_roles"]
                causal_roles = (
                    (roles["target"], "thin"),
                    (roles["secondary_dynamic"][0], "thin"),
                    (roles["fixtures"][0], "middle"),
                )
                for body_name, vertical_rank in causal_roles:
                    body = scene.find(f".//body[@name='{body_name}']")
                    self.assertIsNotNone(body)
                    assert body is not None
                    collision = body.find("geom[@group='3']")
                    visual = body.find("geom[@group='2']")
                    self.assertIsNotNone(collision)
                    self.assertIsNotNone(visual)
                    assert collision is not None and visual is not None

                    body_rotation = rotation(body.get("quat"))
                    collision_rotation = body_rotation @ rotation(
                        collision.get("quat")
                    )
                    half_extents = np.asarray(
                        [float(value) for value in collision.get("size", "").split()]
                    )
                    order = np.argsort(half_extents, kind="stable")
                    expected_vertical = int(
                        order[0] if vertical_rank == "thin" else order[1]
                    )
                    self.assertAlmostEqual(
                        abs(float(collision_rotation[2, expected_vertical])),
                        1.0,
                        places=7,
                    )
                    np.testing.assert_allclose(
                        np.delete(collision_rotation[2], expected_vertical),
                        np.zeros(2),
                        atol=1e-7,
                    )

                    body_position = np.asarray(
                        [float(value) for value in body.get("pos", "").split()]
                    )
                    local_position = np.asarray(
                        [
                            float(value)
                            for value in collision.get("pos", "0 0 0").split()
                        ]
                    )
                    center = body_position + body_rotation @ local_position
                    bottom_z = float(
                        center[2]
                        - np.abs(collision_rotation[2]) @ half_extents
                    )
                    self.assertAlmostEqual(bottom_z, 1e-4, places=7)
                    self.assertEqual(
                        visual.get("pos", "0 0 0"),
                        collision.get("pos", "0 0 0"),
                    )
                    self.assertEqual(
                        visual.get("quat", "1 0 0 0"),
                        collision.get("quat", "1 0 0 0"),
                    )


if __name__ == "__main__":
    unittest.main()
