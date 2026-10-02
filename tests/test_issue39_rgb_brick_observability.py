from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from breakout_rl.issue39_rgb_brick_observability import (
    BrickRemovalDetector,
    classify,
    load_config,
    match_events,
)


def crop() -> np.ndarray:
    return np.zeros((48, 144, 3), dtype=np.uint8)


def set_cell(image: np.ndarray, row: int, col: int, count: int = 4, value: int = 25) -> None:
    for offset in range(count):
        y, x = divmod(offset, 8)
        image[row * 8 + y, col * 8 + x, :] = value


class DetectorTests(unittest.TestCase):
    def test_empty_and_occupied_threshold_boundary(self):
        detector = BrickRemovalDetector()
        empty = crop()
        self.assertEqual(detector.observe(0, empty), [])
        four = crop()
        set_cell(four, 0, 0, 4, 25)
        detector = BrickRemovalDetector()
        detector.observe(0, four)
        below = crop()
        set_cell(below, 0, 0, 3, 255)
        self.assertEqual(detector.observe(1, below), [])
        detector = BrickRemovalDetector()
        detector.observe(0, four)
        three = crop()
        set_cell(three, 0, 0, 3, 255)
        detector.observe(1, three)
        self.assertEqual(detector.observe(2, three), [{"frame": 2, "cell": 0}])

    def test_pixel_intensity_must_be_strictly_greater_than_24(self):
        at_threshold = crop()
        set_cell(at_threshold, 0, 0, 64, 24)
        above_threshold = crop()
        set_cell(above_threshold, 0, 0, 4, 25)
        detector = BrickRemovalDetector()
        detector.observe(0, at_threshold)
        self.assertEqual(detector.observe(1, above_threshold), [])
        detector.observe(0, above_threshold)
        self.assertEqual(detector.previous[0], True)

    def test_one_frame_occlusion_does_not_emit_and_refill_cancels(self):
        occupied = crop()
        set_cell(occupied, 2, 3)
        empty = crop()
        detector = BrickRemovalDetector()
        detector.observe(0, occupied)
        self.assertEqual(detector.observe(1, empty), [])
        self.assertEqual(detector.observe(2, occupied), [])
        self.assertEqual(detector.observe(3, occupied), [])

    def test_persistent_removal_emits_once_and_refill_reinitializes(self):
        occupied = crop()
        set_cell(occupied, 1, 2)
        empty = crop()
        detector = BrickRemovalDetector()
        detector.observe(0, occupied)
        self.assertEqual(detector.observe(1, empty), [])
        self.assertEqual(detector.observe(2, empty), [{"frame": 2, "cell": 20}])
        self.assertEqual(detector.observe(3, empty), [])
        detector.observe(4, occupied)
        detector.observe(5, empty)
        self.assertEqual(detector.observe(6, empty), [{"frame": 6, "cell": 20}])

    def test_multiple_cells_deduplicate_to_one_event_per_frame(self):
        occupied = crop()
        set_cell(occupied, 0, 1)
        set_cell(occupied, 0, 2)
        detector = BrickRemovalDetector()
        detector.observe(0, occupied)
        empty = crop()
        detector.observe(1, empty)
        self.assertEqual(detector.observe(2, empty), [{"frame": 2, "cell": 1}])


class EvaluatorTests(unittest.TestCase):
    def test_one_to_one_matching_with_plus_minus_two_frame_tolerance(self):
        result = match_events([{"frame": 10}, {"frame": 11}, {"frame": 20}],
                              [{"frame": 8}, {"frame": 22}])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))
        self.assertAlmostEqual(result["f1"], 0.8)
        outside = match_events([{"frame": 13}], [{"frame": 10}])
        self.assertEqual((outside["tp"], outside["fp"], outside["fn"]), (0, 1, 1))

    def test_frozen_classification_boundaries_and_sample_floor(self):
        self.assertEqual(classify({"f1": .90}, 12, 2, True), "PROMOTED")
        self.assertEqual(classify({"f1": .50}, 12, 2, True), "REJECTED")
        self.assertEqual(classify({"f1": .70}, 12, 2, True), "INCONCLUSIVE")
        self.assertEqual(classify({"f1": 1.0}, 11, 2, True), "INCONCLUSIVE")
        self.assertEqual(classify({"f1": 1.0}, 12, 1, True), "INCONCLUSIVE")
        self.assertEqual(classify({"f1": 1.0}, 12, 2, False), "INCONCLUSIVE")

    def test_frozen_config_caps_and_seeds(self):
        cfg = load_config()
        self.assertEqual(cfg.seeds, (707, 808, 909))
        self.assertEqual((cfg.max_frames_per_episode, cfg.max_frames_total), (5000, 15000))
        self.assertEqual(cfg.total_wall_seconds, 600)

    def test_controller_call_is_pixel_only_and_labels_are_created_afterward(self):
        # Regression guard: online loop source uses only select_action(obs), while
        # offline detector's interface accepts only a frame index and RGB crop.
        source = Path(__file__).resolve().parents[1] / "breakout_rl/issue39_rgb_brick_observability.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("decision = controller.select_action(obs)", text)
        marker = "# Both the detector and reward/RAM label construction run after all online actions finish."
        self.assertIn(marker, text)
        call = text.index("decision = controller.select_action(obs)")
        labels = text.index("labels.append(")
        detector = text.index("detector.observe(local_frame, crop)")
        self.assertLess(call, labels)
        self.assertLess(text.index(marker), labels)
        self.assertLess(text.index(marker), detector)


if __name__ == "__main__":
    unittest.main()
