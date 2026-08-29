"""Tests for the source-backed Day 18 training visualization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.visualization.visualize_day18_comparison import (
    _require_completed,
    render_training,
)


class Day18VisualizationTests(unittest.TestCase):
    def test_training_plot_preserves_seed_level_series_and_writes_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            entries = []
            for algorithm in ("dqn", "double_dqn"):
                entries.append(
                    {
                        "run_id": f"{algorithm}-seed11",
                        "algorithm": algorithm,
                        "training_seed": 11,
                        "stage": "main",
                        "target_transitions": 500_000,
                        "eligible": True,
                        "metrics": [
                            {
                                "global_step": "100000",
                                "raw_episode_return": "1.0",
                            },
                            {
                                "global_step": "200000",
                                "raw_episode_return": "2.0",
                            },
                            {
                                "global_step": "300000",
                                "raw_episode_return": "3.0",
                            },
                        ],
                    }
                )
            outputs = render_training(manifest, root / "assets", entries)

            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue((root / "assets/dqn-vs-double-training.json").is_file())

    def test_incomplete_entries_are_rejected_before_plotting(self) -> None:
        with self.assertRaisesRegex(ValueError, "no completed training"):
            _require_completed(
                [{"eligible": False, "status": "incomplete"}],
                name="training",
            )


if __name__ == "__main__":
    unittest.main()
