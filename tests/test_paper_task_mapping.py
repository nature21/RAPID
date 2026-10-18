"""Cross-check paper indices against pre-migration media and benchmark samples."""

import csv
import hashlib
import json
from pathlib import Path
import unittest

from benchmark_private.specs import STRATEGY_CONTRACTS, sampling_fingerprint
from benchmark_private.success import PICKUP_STRATEGIES, STORAGE_LAYOUT_KEYS


ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((Path(__file__).parent / "fixtures/paper_task_baseline.json").read_text())


class PaperTaskMappingTests(unittest.TestCase):
    def test_renumbering_preserves_all_sampled_cases(self):
        for task, expected in BASELINE["tasks"].items():
            with self.subTest(task=task):
                contract = STRATEGY_CONTRACTS[int(task)]
                self.assertEqual(contract.target_body, expected["target"])
                self.assertEqual(list(contract.phases), expected["phases"])
                self.assertEqual(sampling_fingerprint(int(task)), expected["fingerprint"])

    def test_demo_media_and_annotations_follow_paper_indices(self):
        with (ROOT / "demos/strategy/segments.csv").open() as handle:
            segments = list(csv.DictReader(handle))
        expected_first_phase = {6: "reorient", 7: "topple_expose_edge", 8: "move_to_edge"}
        for task, phase in expected_first_phase.items():
            with self.subTest(task=task):
                video = ROOT / f"demos/strategy/demo_{task}/rgb.mp4"
                self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(),
                                 BASELINE["tasks"][str(task)]["demo_rgb_sha256"])
                first = next(row for row in segments if int(row["demo"]) == task)
                self.assertEqual(first["primitive"], phase)

    def test_evaluator_uses_task_six_storage_and_seven_eight_pickup(self):
        self.assertIn(6, STORAGE_LAYOUT_KEYS)
        self.assertNotIn(6, PICKUP_STRATEGIES)
        self.assertTrue({7, 8} <= PICKUP_STRATEGIES)
        self.assertTrue({7, 8}.isdisjoint(STORAGE_LAYOUT_KEYS))


if __name__ == "__main__":
    unittest.main()
