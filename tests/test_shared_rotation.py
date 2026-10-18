from __future__ import annotations

import unittest
import numpy as np

from primitive_core.compile import _couple_rotation, PrimitiveCompileSpec
from primitive_core.policy import NominalPlan
from primitive_core.fr3_safety import enforce_pose_trajectory_timing, minimum_pose_trajectory_duration
from tests.test_fr3_safety import _PoseAdapter


class SharedRotationTests(unittest.TestCase):
    def test_contract_rejects_invalid_indices_and_nonconstant_suffix(self):
        knots = np.zeros((4, 7))
        for start in (True, 0, 4, 1.5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                NominalPlan(knots, np.asarray(.1), constant_rotation_from_knot=start)
        knots[3, 4] = .2
        with self.assertRaisesRegex(ValueError, "equal nominal"):
            NominalPlan(knots, np.asarray(.1), constant_rotation_from_knot=1)

    def test_plan_propagates_opt_in_and_default_does_not_project(self):
        plan = NominalPlan(np.zeros((4, 7)), np.asarray(.1), True,
                           constant_rotation_from_knot=2)
        spec = PrimitiveCompileSpec.from_nominal_plan(
            nominal_plan=plan, model=None, initial_state=np.zeros(1), cost_fn=lambda *args: 0,
        )
        self.assertEqual(spec.constant_rotation_from_knot, 2)
        knots = np.random.default_rng(4).normal(size=(5, 4, 7))
        before = knots.copy()
        _couple_rotation(knots, None)
        np.testing.assert_array_equal(knots, before)

    def test_coupling_keeps_search_variation_and_respects_timing(self):
        rng = np.random.default_rng(19)
        adapter = _PoseAdapter()
        fractions = np.asarray([0, .1, .3, .55, 1.])
        for start in (1, 2, 3):
            candidates = rng.uniform(-.8, .8, (12, 5, 7))
            candidates[:, :, 6] = .02
            original = candidates.copy()
            _couple_rotation(candidates, start)
            np.testing.assert_array_equal(candidates[:, :start], original[:, :start])
            np.testing.assert_array_equal(candidates[:, :, :3], original[:, :, :3])
            bounded, _ = enforce_pose_trajectory_timing(adapter, candidates, fractions, .5)
            _couple_rotation(bounded, start)
            self.assertGreater(np.std(bounded[:, start, 3]), .01)
            for trajectory in bounded:
                np.testing.assert_allclose(trajectory[start:, 3:6],
                                           np.broadcast_to(trajectory[start, 3:6], (5-start, 3)))
                self.assertLessEqual(minimum_pose_trajectory_duration(adapter, trajectory, fractions), .5 + 1e-9)
            np.testing.assert_array_equal(bounded[:, 0], original[:, 0])
