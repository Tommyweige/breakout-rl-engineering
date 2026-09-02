"""Tests for the real-output Day 19 visualization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.inspect_dueling_network import collect_dueling_inspection
from scripts.visualization.visualize_dueling_components import create_figure


class DuelingVisualizationTests(unittest.TestCase):
    def test_figure_and_metadata_are_generated_from_inspection_output(self) -> None:
        inspection = collect_dueling_inspection(seed=42, device_name="cpu")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dueling.png"
            metadata = Path(temporary_directory) / "dueling.json"
            image_path, metadata_path = create_figure(
                inspection,
                seed=42,
                output=output,
                metadata_path=metadata,
                command="test visualization command",
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(image_path, output)
            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertEqual(payload["generation_command"], "test visualization command")
            self.assertEqual(payload["figure_question"], "How do the real Value and centered Advantage outputs combine into Q-values?")
            self.assertEqual(payload["model_config"]["architecture"], "dueling")
            self.assertEqual(payload["q_shape"], [1, 4])


if __name__ == "__main__":
    unittest.main()
