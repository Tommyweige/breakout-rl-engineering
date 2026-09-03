"""Tests for source-backed Day 20 comparison figures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.visualization.visualize_dqn_family_comparison import (
    _completed,
    render_evaluation,
    render_training,
)


def _entry(
    family_id: str,
    seed: int,
    *,
    stage: str = "main",
    target: int = 500_000,
) -> dict:
    return {
        "run_id": f"{family_id}-seed{seed}-{stage}",
        "family_id": family_id,
        "training_seed": seed,
        "stage": stage,
        "target_transitions": target,
        "status": "completed",
        "eligible": True,
        "metrics": [
            {"global_step": str(step), "raw_episode_return": str(seed / 10 + step / 100_000)}
            for step in (100_000, 250_000, target)
        ],
        "evaluation": {
            "status": "completed",
            "summary": {
                "mean_return": seed / 10 + target / 100_000,
                "std_return": 1.0,
            },
        },
        "runtime": {
            "steps_per_second": 300.0,
            "wall_clock_seconds": 100.0,
            "cuda_peak_allocated_bytes": 600_000_000,
            "parameter_count": 1_686_180,
        },
    }


class Day20VisualizationTests(unittest.TestCase):
    def test_training_plot_uses_completed_seed_series_and_writes_alias(self) -> None:
        entries = [
            _entry(family_id, seed)
            for family_id in ("dqn", "double_dqn", "dueling_double_dqn")
            for seed in (11, 22, 33)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            outputs = render_training(manifest, root / "assets", entries)

            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue((root / "assets/dqn-family-training.json").is_file())

    def test_evaluation_plot_accepts_the_optional_1m_extension(self) -> None:
        # Keep the fixture explicit so the test covers both the formal point and
        # the extension point without relying on a production manifest.
        entries = []
        for family_id in ("dqn", "double_dqn", "dueling_double_dqn"):
            entries.append(_entry(family_id, 11, target=500_000))
            if family_id != "dqn":
                entries.append(_entry(family_id, 11, stage="extension_1m", target=1_000_000))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = render_evaluation(manifest, root / "assets", entries)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_incomplete_entries_are_rejected_before_plotting(self) -> None:
        with self.assertRaisesRegex(ValueError, "no completed eligible"):
            _completed([{"status": "running", "eligible": False}])


if __name__ == "__main__":
    unittest.main()
