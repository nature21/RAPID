from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from agent.instantiation_validation import validate_generalization_collection
from instantiation_core.generalization import (
    SceneVariantSpec,
    balanced_generalization_samples,
    make_variant_data,
    make_variant_model,
    runtime_scene_fingerprint,
    validate_variant_initial_state,
)
from instantiation_core.scene_scaling import scale_target
from tasks.invention_real2sim.demo_1_scene.scene import (
    make_data as make_baseline_data,
)
from tasks.invention_real2sim.demo_1_scene.scene import (
    make_model as make_baseline_model,
)
from tasks.invention_real2sim.flip_scene.scene import make_data as make_flip_data
from tasks.invention_real2sim.flip_scene.scene import make_model as make_flip_model


BASE_MODULE = "tasks.invention_real2sim.demo_1_scene.scene"


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _inertia(model: mujoco.MjModel, body: int) -> np.ndarray:
    rotation = _rotation(model.body_iquat[body])
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def _mesh_vertices(model: mujoco.MjModel, name: str) -> np.ndarray:
    geom = model.geom(name).id
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return (model.mesh_vert[start:start + count] @ _rotation(model.geom_quat[geom]).T
            + model.geom_pos[geom])


def _spec(index: int, **overrides: object) -> SceneVariantSpec:
    values: dict[str, object] = {
        "base_scene_module": BASE_MODULE,
        "variant_index": index,
        "seed": 7,
        "target_body": "red_cheez_it_box",
        "target_free_joint": "red_cheez_it_box_joint",
        "target_collision_geoms": ("red_cheez_it_box_collision",),
        "support_geom": "floor",
    }
    values.update(overrides)
    return SceneVariantSpec(**values)  # type: ignore[arg-type]


class GeneralizationSamplingTests(unittest.TestCase):
    def test_balanced_samples_are_deterministic_and_cover_full_circle(self) -> None:
        first = balanced_generalization_samples(3, seed=11)
        second = balanced_generalization_samples(3, seed=11)
        different = balanced_generalization_samples(3, seed=12)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len({sample.target_yaw for sample in first}), 3)
        self.assertTrue(any(abs(sample.target_yaw) >= math.pi - 1e-12 for sample in first))
        self.assertTrue(all(all(0.8 <= axis <= 1.2 for axis in sample.size_scale_xyz) for sample in first))
        self.assertTrue(all(0.5 <= sample.mass_scale <= 2.0 for sample in first))
        self.assertTrue(all(0.5 <= sample.friction_scale <= 1.5 for sample in first))
        self.assertTrue(all(len(set(sample.size_scale_xyz)) == 3 for sample in first))
        bins = np.floor((np.array([s.size_scale_xyz for s in first]) - 0.8) / (0.4 / 3))
        for axis in bins.T:
            self.assertEqual(sorted(axis), [0, 1, 2])

    def test_scale_metadata_is_finite_positive_and_three_dimensional(self) -> None:
        for value in [(1, 2), (1, 0, 1), (1, -1, 1), (1, float('nan'), 1),
                      (1, 1, float('inf'))]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _spec(1, size_scale_xyz=value)
        specification = _spec(1, size_scale_xyz=[0.8, 1.0, 1.2])
        self.assertEqual(specification.size_scale_xyz, (0.8, 1.0, 1.2))
        self.assertEqual(specification.to_dict()['size_scale_xyz'], [0.8, 1.0, 1.2])
        self.assertIn('object_size', specification.changed_dimensions)

    def test_collection_requires_joint_physical_and_goal_coverage(self) -> None:
        baseline = _spec(0, goal_parameters={"goal_distance": 0.1})
        variants = [
            _spec(
                index,
                size_scale_xyz=(0.85 + 0.1 * index, 0.9, 1.1),
                target_position_offset=(0.01 * index, -0.01, 0.0),
                target_yaw=(-math.pi + index),
                mass_scale=0.7 + 0.3 * index,
                target_friction_scale=0.7 + 0.2 * index,
                support_friction_scale=1.2,
                goal_parameters={
                    "goal_distance": 0.1 if index == 1 else 0.1 + 0.02 * index
                },
                varied_goal_parameters=(
                    () if index == 1 else ("goal_distance",)
                ),
            )
            for index in range(1, 4)
        ]

        findings = validate_generalization_collection(
            [baseline, *variants],
            ["physical-0", "physical-1", "physical-2", "physical-3"],
            ["runtime-0", "runtime-1", "runtime-2", "runtime-3"],
            num_variants=3,
            goal_parameter_names=("goal_distance",),
        )

        self.assertEqual(findings, [])

    def test_collection_detects_duplicate_physics_and_shallow_variation(self) -> None:
        baseline = _spec(0)
        goal_only = _spec(1, goal_parameters={"goal_distance": 0.2})

        findings = validate_generalization_collection(
            [baseline, goal_only],
            ["same", "same"],
            ["runtime-0", "runtime-1"],
            num_variants=1,
        )

        self.assertTrue(any("at least 3" in finding for finding in findings))
        self.assertTrue(any("at least 2" in finding for finding in findings))
        self.assertTrue(any("duplicate physical" in finding for finding in findings))

    def test_input_free_contract_does_not_require_goal_variation(self) -> None:
        baseline = _spec(0)
        variants = [
            _spec(
                index,
                size_scale_xyz=(0.85 + 0.1 * index, 0.9, 1.1),
                target_position_offset=(0.01 * index, -0.01, 0.0),
                target_yaw=(-math.pi + index),
                mass_scale=0.7 + 0.3 * index,
                target_friction_scale=0.7 + 0.2 * index,
                support_friction_scale=1.2,
            )
            for index in range(1, 3)
        ]

        findings = validate_generalization_collection(
            [baseline, *variants],
            ["physical-0", "physical-1", "physical-2"],
            ["runtime-0", "runtime-1", "runtime-2"],
            num_variants=2,
        )

        self.assertEqual(findings, [])

    def test_declared_primitive_goal_must_be_varied(self) -> None:
        baseline = _spec(0, goal_parameters={"goal_distance": 0.1})
        variants = [
            _spec(
                index,
                size_scale_xyz=(0.85 + 0.1 * index, 0.9, 1.1),
                target_position_offset=(0.01 * index, -0.01, 0.0),
                target_yaw=(-math.pi + index),
                mass_scale=0.7 + 0.3 * index,
                target_friction_scale=0.7 + 0.2 * index,
                support_friction_scale=1.2,
                goal_parameters={"goal_distance": 0.1},
            )
            for index in range(1, 3)
        ]

        findings = validate_generalization_collection(
            [baseline, *variants],
            ["physical-0", "physical-1", "physical-2"],
            ["runtime-0", "runtime-1", "runtime-2"],
            num_variants=2,
            goal_parameter_names=("goal_distance",),
        )

        self.assertTrue(
            any("does not vary declared primitive goal" in item for item in findings)
        )


class GeneralizationSceneTests(unittest.TestCase):
    @staticmethod
    def _flip_specification() -> SceneVariantSpec:
        return SceneVariantSpec(
            base_scene_module="tasks.invention_real2sim.flip_scene.scene",
            variant_index=0,
            seed=7,
            target_body="cheez_it_box",
            target_free_joint="cheez_it_box_joint",
            target_collision_geoms=("cheez_it_box_collision",),
            support_geom="floor",
            fixture_body="white_divider",
            fixture_contact_geoms=("white_divider_collision",),
        )

    def test_fixture_yaw_requires_coupling_or_explicit_relative_variation(self) -> None:
        common = {
            "base_scene_module": "tasks.invention_real2sim.flip_scene.scene",
            "variant_index": 1,
            "seed": 7,
            "target_body": "cheez_it_box",
            "target_free_joint": "cheez_it_box_joint",
            "target_collision_geoms": ("cheez_it_box_collision",),
            "support_geom": "floor",
            "target_yaw": math.pi / 3.0,
            "fixture_body": "white_divider",
            "fixture_contact_geoms": ("white_divider_collision",),
        }

        with self.assertRaisesRegex(ValueError, "must couple the fixture"):
            SceneVariantSpec(**common)  # type: ignore[arg-type]
        specification = SceneVariantSpec(
            **common,  # type: ignore[arg-type]
            allow_independent_fixture_yaw=True,
        )
        self.assertTrue(specification.allow_independent_fixture_yaw)

    def test_neutral_variant_preserves_assigned_baseline(self) -> None:
        specification = _spec(0)
        baseline_model = make_baseline_model()
        baseline_data = make_baseline_data(baseline_model)
        model = make_variant_model(specification)
        data = make_variant_data(specification, model)

        self.assertEqual(
            runtime_scene_fingerprint(model, data),
            runtime_scene_fingerprint(baseline_model, baseline_data),
        )

    def test_variant_scales_physics_visuals_and_resting_pose(self) -> None:
        baseline_model = make_baseline_model()
        specification = _spec(
            1,
            size_scale_xyz=(1.2, 1.2, 1.2),
            target_position_offset=(0.02, -0.01, 0.0),
            target_yaw=math.pi,
            mass_scale=1.5,
            target_friction_scale=0.7,
            support_friction_scale=1.2,
        )

        model = make_variant_model(specification)
        data = make_variant_data(specification, model)

        baseline_target = baseline_model.body("red_cheez_it_box").id
        target = model.body("red_cheez_it_box").id
        np.testing.assert_allclose(
            model.geom("red_cheez_it_box_collision").size,
            baseline_model.geom("red_cheez_it_box_collision").size * 1.2,
        )
        self.assertAlmostEqual(
            float(model.body_mass[target]),
            float(baseline_model.body_mass[baseline_target]) * 1.2**3 * 1.5,
        )
        np.testing.assert_allclose(
            _inertia(model, target),
            _inertia(baseline_model, baseline_target) * 1.2**5 * 1.5,
        )
        np.testing.assert_allclose(
            model.geom("red_cheez_it_box_collision").friction,
            baseline_model.geom("red_cheez_it_box_collision").friction * 0.7,
        )
        np.testing.assert_allclose(
            model.geom("floor").friction,
            baseline_model.geom("floor").friction * 1.2,
        )
        np.testing.assert_allclose(
            _mesh_vertices(model, "red_cheez_it_box_visual"),
            _mesh_vertices(baseline_model, "red_cheez_it_box_visual") * 1.2,
            rtol=1e-6,
            atol=1e-8,
        )

        target_geom = model.geom("red_cheez_it_box_collision").id
        floor = model.geom("floor").id
        floor_height = float(data.geom_xpos[floor, 2])
        rotation = data.geom_xmat[target_geom].reshape(3, 3)
        bottom = float(
            data.geom_xpos[target_geom, 2]
            - np.sum(np.abs(rotation[2, :]) * model.geom_size[target_geom])
        )
        self.assertAlmostEqual(bottom, floor_height, places=9)
        self.assertEqual(validate_variant_initial_state(specification), [])

    def test_anisotropic_scale_preserves_visual_collision_alignment_and_inertia(self) -> None:
        baseline = make_baseline_model()
        factors = np.array([0.8, 1.1, 1.2])
        model = make_variant_model(_spec(1, size_scale_xyz=tuple(factors), mass_scale=1.3))
        name = 'red_cheez_it_box_collision'
        frame = _rotation(baseline.geom(name).quat)
        transform = frame @ np.diag(factors) @ frame.T
        np.testing.assert_allclose(model.geom(name).size, baseline.geom(name).size * factors)
        np.testing.assert_allclose(model.geom(name).pos, transform @ baseline.geom(name).pos)
        np.testing.assert_allclose(
            _mesh_vertices(model, 'red_cheez_it_box_visual'),
            _mesh_vertices(baseline, 'red_cheez_it_box_visual') @ transform.T,
            atol=2e-8,
        )
        body = baseline.body('red_cheez_it_box').id
        # The source collision box carries its explicit uniform-box inertia.
        half = baseline.geom(name).size * factors
        mass = baseline.body_mass[body] * np.prod(factors) * 1.3
        expected = mass / 3 * (np.sum(half**2) - half**2)
        np.testing.assert_allclose(model.body_mass[body], mass)
        np.testing.assert_allclose(_inertia(model, body), frame @ np.diag(expected) @ frame.T,
                                   rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose(model.body_ipos[body], transform @ baseline.body_ipos[body])
        data = make_variant_data(_spec(1, size_scale_xyz=tuple(factors)), model)
        gid = model.geom(name).id
        bottom = data.geom_xpos[gid, 2] - np.abs(data.geom_xmat[gid].reshape(3, 3)[2]) @ model.geom_size[gid]
        self.assertAlmostEqual(bottom, data.geom_xpos[model.geom('floor').id, 2])

    def test_shared_mesh_is_not_resized_on_other_bodies_and_inferred_mass_scales(self) -> None:
        scene = mujoco.MjSpec.from_string('''<mujoco><asset>
          <mesh name="shared" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
          </asset><worldbody>
          <body name="target"><freejoint/>
            <geom name="shape" type="box" size=".1 .2 .3" mass="2"/>
            <geom name="visual" type="mesh" mesh="shared" density="0" contype="0" conaffinity="0"/>
          </body><body name="other" pos="2 0 0">
            <geom name="other_visual" type="mesh" mesh="shared"/>
          </body></worldbody></mujoco>''')
        baseline = scene.compile()
        scale_target(scene, 'target', 'shape', (0.8, 1.1, 1.2), 1.5)
        model = scene.compile()
        np.testing.assert_allclose(_mesh_vertices(model, 'other_visual'),
                                   _mesh_vertices(baseline, 'other_visual'), atol=1e-8)
        np.testing.assert_allclose(_mesh_vertices(model, 'visual'),
                                   _mesh_vertices(baseline, 'visual') * [0.8, 1.1, 1.2], atol=1e-7)
        self.assertAlmostEqual(model.body('target').mass[0], 2 * .8 * 1.1 * 1.2 * 1.5)

    def test_initial_validation_allows_converged_reset_relaxation(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <option timestep="0.01"/>
              <worldbody>
                <geom name="floor" type="plane" size="1 1 .1"/>
                <body name="target" pos="0 0 .06">
                  <freejoint name="target_joint"/>
                  <geom name="target_collision" type="box" size=".02 .02 .02" mass="1"/>
                </body>
              </worldbody>
            </mujoco>
        """)
        data = mujoco.MjData(model)
        initial = data.qpos.copy()
        for _ in range(50):
            mujoco.mj_step(model, data)
        self.assertGreater(float(np.linalg.norm(data.qpos[:3] - initial[:3])), .02)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        specification = SceneVariantSpec(
            base_scene_module="unused", variant_index=1, seed=7,
            target_body="target", target_free_joint="target_joint",
            target_collision_geoms=("target_collision",), support_geom="floor",
        )
        with patch("instantiation_core.generalization.make_variant_model", return_value=model), \
             patch("instantiation_core.generalization.make_variant_data", return_value=data):
            self.assertEqual(validate_variant_initial_state(specification), [])

    def test_initial_validation_rejects_continued_free_fall(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.01" gravity="0 0 -9.81"/>
              <worldbody>
                <geom name="floor" type="plane" pos="0 0 -100" size="1 1 0.1"/>
                <body name="target" pos="0 0 10">
                  <freejoint name="target_joint"/>
                  <geom name="target_collision" type="box"
                        size="0.1 0.1 0.1" mass="1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        specification = SceneVariantSpec(
            base_scene_module="unused",
            variant_index=1,
            seed=7,
            target_body="target",
            target_free_joint="target_joint",
            target_collision_geoms=("target_collision",),
            support_geom="floor",
        )

        with (
            patch(
                "instantiation_core.generalization.make_variant_model",
                return_value=model,
            ),
            patch(
                "instantiation_core.generalization.make_variant_data",
                return_value=data,
            ),
        ):
            findings = validate_variant_initial_state(specification)

        self.assertTrue(
            any("post-relaxation drift" in finding for finding in findings),
            findings,
        )

    def test_initial_validation_rejects_continued_rotation(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.01" gravity="0 0 0"/>
              <worldbody>
                <geom name="floor" type="plane" pos="0 0 -1" size="1 1 0.1"/>
                <body name="target" pos="0 0 1">
                  <freejoint name="target_joint"/>
                  <geom name="target_collision" type="box"
                        size="0.1 0.1 0.1" mass="1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        joint = model.joint("target_joint")
        dof_address = int(model.jnt_dofadr[joint.id])
        data.qvel[dof_address + 5] = 1.0
        mujoco.mj_forward(model, data)
        specification = SceneVariantSpec(
            base_scene_module="unused",
            variant_index=1,
            seed=7,
            target_body="target",
            target_free_joint="target_joint",
            target_collision_geoms=("target_collision",),
            support_geom="floor",
        )

        with (
            patch(
                "instantiation_core.generalization.make_variant_model",
                return_value=model,
            ),
            patch(
                "instantiation_core.generalization.make_variant_data",
                return_value=data,
            ),
        ):
            findings = validate_variant_initial_state(specification)

        self.assertTrue(
            any("post-relaxation rotation" in finding for finding in findings),
            findings,
        )
        self.assertTrue(
            any("terminal settle angular speed" in finding for finding in findings),
            findings,
        )

    def test_yaw_coupled_body_preserves_target_fixture_layout(self) -> None:
        baseline_model = make_flip_model()
        baseline_data = make_flip_data(baseline_model)
        specification = SceneVariantSpec(
            base_scene_module="tasks.invention_real2sim.flip_scene.scene",
            variant_index=1,
            seed=7,
            target_body="cheez_it_box",
            target_free_joint="cheez_it_box_joint",
            target_collision_geoms=("cheez_it_box_collision",),
            support_geom="floor",
            target_yaw=math.pi / 2.0,
            yaw_coupled_bodies=("white_divider",),
            fixture_body="white_divider",
            fixture_contact_geoms=("white_divider_collision",),
        )

        model = make_variant_model(specification)
        data = make_variant_data(specification, model)
        target = baseline_model.body("cheez_it_box").id
        fixture = baseline_model.body("white_divider").id
        variant_target = model.body("cheez_it_box").id
        variant_fixture = model.body("white_divider").id
        yaw_rotation = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

        baseline_delta = baseline_data.xpos[fixture] - baseline_data.xpos[target]
        variant_delta = data.xpos[variant_fixture] - data.xpos[variant_target]
        np.testing.assert_allclose(variant_delta, yaw_rotation @ baseline_delta)
        baseline_relative_rotation = (
            baseline_data.xmat[target].reshape(3, 3).T
            @ baseline_data.xmat[fixture].reshape(3, 3)
        )
        variant_relative_rotation = (
            data.xmat[variant_target].reshape(3, 3).T
            @ data.xmat[variant_fixture].reshape(3, 3)
        )
        np.testing.assert_allclose(
            variant_relative_rotation,
            baseline_relative_rotation,
            atol=1e-12,
        )
        self.assertIn("fixture", specification.changed_dimensions)
        self.assertEqual(
            specification.to_dict()["yaw_coupled_bodies"],
            ["white_divider"],
        )

    def test_runtime_fingerprint_separates_physics_from_goal(self) -> None:
        model = make_baseline_model()
        data = make_baseline_data(model)

        physical = runtime_scene_fingerprint(model, data)
        first_goal = runtime_scene_fingerprint(
            model,
            data,
            goal_parameters={"distance": 0.1},
        )
        second_goal = runtime_scene_fingerprint(
            model,
            data,
            goal_parameters={"distance": 0.2},
        )

        self.assertNotEqual(physical, first_goal)
        self.assertNotEqual(first_goal, second_goal)


if __name__ == "__main__":
    unittest.main()
