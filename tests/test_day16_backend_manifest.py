"""Tests for the canonical Day 16 training backend manifest."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from breakout_rl.training.backend_manifest import (
    DAY16_CANONICAL_BACKEND_ID,
    load_day16_backend_manifest,
    validate_day16_backend_manifest,
)


class Day16BackendManifestTests(unittest.TestCase):
    def test_committed_manifest_loads_and_describes_selected_backend(self) -> None:
        manifest = load_day16_backend_manifest()

        self.assertEqual(manifest["backend_id"], DAY16_CANONICAL_BACKEND_ID)
        self.assertEqual(manifest["source_day"], 16)
        self.assertEqual(manifest["trainer"]["num_envs"], 2)
        self.assertTrue(manifest["trainer"]["strict_action_selection_parity"])
        self.assertEqual(manifest["trainer"]["replay_backend"], "gpu")
        self.assertEqual(manifest["selection"]["role"], "selected_systems_backend")
        self.assertEqual(
            manifest["trainer"]["config"]["target_update_interval"],
            500,
        )

    def test_manifest_validation_rejects_trainer_config_drift(self) -> None:
        manifest = load_day16_backend_manifest()
        changed = copy.deepcopy(manifest)
        changed["trainer"]["config"]["num_envs"] = 4

        with self.assertRaisesRegex(ValueError, "trainer.num_envs"):
            validate_day16_backend_manifest(
                changed,
                repository_root=Path.cwd(),
            )


if __name__ == "__main__":
    unittest.main()
