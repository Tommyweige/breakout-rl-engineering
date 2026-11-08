"""Tests for the Day 14 gate provenance module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from breakout_rl.artifacts import portable_artifact_path
from breakout_rl.day14_provenance import load_day14_provenance, validate_checkpoint_provenance


def _manifest(directory: str, variant: dict) -> Path:
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps({"status": "completed", "variants": [variant]}), encoding="utf-8")
    return path


class Day14ProvenanceTests(unittest.TestCase):
    def test_step_budget_must_be_a_positive_integer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "Day 14 step budget must be an integer"):
                load_day14_provenance(_manifest(directory, {"step_budget": "100"}))
            with self.assertRaisesRegex(ValueError, "Day 14 step budget must be at least 1"):
                load_day14_provenance(_manifest(directory, {"step_budget": 0}))

    def test_incomplete_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"status": "running", "variants": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not completed"):
                load_day14_provenance(path)

    def test_checkpoint_must_match_the_final_variant(self) -> None:
        provenance = {"run_id": "r1", "expected_checkpoint_step": 400, "config_values": {"seed": 7}}
        training = {"source_day14_run_id": "r1", "training_seed": 7}
        validate_checkpoint_provenance({"step": 400}, training, provenance)
        with self.assertRaisesRegex(ValueError, "selection rule"):
            validate_checkpoint_provenance({"step": 399}, training, provenance)
        with self.assertRaisesRegex(ValueError, "for seed"):
            validate_checkpoint_provenance({"step": 400}, {**training, "training_seed": 8}, provenance)

    def test_portable_artifact_path_starts_at_the_artifact_root(self) -> None:
        path = Path("/some/worktree/evaluations/run/results.json")
        self.assertEqual(portable_artifact_path(path), "evaluations/run/results.json")
        self.assertEqual(portable_artifact_path("plain/file.json"), "plain/file.json")


if __name__ == "__main__":
    unittest.main()
