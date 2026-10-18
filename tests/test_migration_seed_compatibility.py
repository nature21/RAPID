"""Golden random streams from the pre-migration physical execution baseline."""

import unittest

from primitive_core.transit import _derive_transit_planning_seed


class MigrationSeedCompatibilityTests(unittest.TestCase):
    def test_release_and_handoff_keep_baseline_random_streams(self):
        self.assertEqual(_derive_transit_planning_seed(42, 1, "release"), 1146090052)
        self.assertEqual(_derive_transit_planning_seed(42, 1, "handoff"), 2078500728)


if __name__ == "__main__":
    unittest.main()
