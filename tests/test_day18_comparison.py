"""Tests for the staged Day 18 comparison protocol and evidence gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from breakout_rl.day18_comparison import (
    DAY18_ALGORITHMS,
    DAY18_MILESTONES,
    DAY18_TRAINING_SEEDS,
    build_day18_manifest,
    build_day18_report,
    config_diff,
    load_day18_config,
    read_day18_manifest,
    write_json,
)
from scripts.training.run_day18_comparison import _requested_selection


class Day18ComparisonTests(unittest.TestCase):
    def test_config_reuses_day16_backend_and_contract_v2(self) -> None:
        config = load_day18_config(require_probe_states=True)

        self.assertEqual(config.algorithms, DAY18_ALGORITHMS)
        self.assertEqual(config.training_seeds, DAY18_TRAINING_SEEDS)
        self.assertEqual(dict(config.milestones), DAY18_MILESTONES)
        self.assertEqual(config.backend_config.num_envs, 2)
        self.assertEqual(config.backend_config.replay_backend, "gpu")
        self.assertEqual(config.backend_config.device, "cuda")
        self.assertEqual(config.contract.contract_id, "day15-breakout-evaluation-v2-fire-reset")
        self.assertEqual(config.evaluation_config["epsilon"], 0.0)

    def test_manifest_contains_every_algorithm_seed_and_milestone_pair(self) -> None:
        config = load_day18_config(require_probe_states=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "experiments" / "day18" / "manifest.json"
            manifest = build_day18_manifest(
                config,
                manifest_path=manifest_path,
                runs_root=root / "runs",
                evaluations_root=root / "evaluations",
            )
            write_json(manifest_path, manifest)
            restored = read_day18_manifest(manifest_path)

        self.assertEqual(len(restored["runs"]), 18)
        keys = {
            (
                entry["algorithm"],
                entry["training_seed"],
                entry["stage"],
            )
            for entry in restored["runs"]
        }
        self.assertEqual(
            len(keys),
            len(DAY18_ALGORITHMS) * len(DAY18_TRAINING_SEEDS) * len(DAY18_MILESTONES),
        )
        self.assertTrue(all(entry["status"] == "pending" for entry in restored["runs"]))

    def test_pair_config_diff_ignores_only_intended_variables(self) -> None:
        config = load_day18_config(require_probe_states=True)
        dqn = config.training_config(algorithm="dqn", seed=11, stage="pilot").to_dict()
        double = config.training_config(
            algorithm="double_dqn",
            seed=11,
            stage="pilot",
        ).to_dict()

        self.assertEqual(config_diff(dqn, double), {})
        changed = dict(double)
        changed["batch_size"] = 64
        self.assertIn("batch_size", config_diff(dqn, changed))

    def test_selection_keeps_pilot_to_one_paired_seed_and_main_to_three(self) -> None:
        config = load_day18_config(require_probe_states=True)

        pilot_seeds, pilot_stages = _requested_selection(config, "pilot")
        main_seeds, main_stages = _requested_selection(config, "main")

        self.assertEqual(pilot_seeds, (11,))
        self.assertEqual(pilot_stages, ("screening", "pilot"))
        self.assertEqual(main_seeds, DAY18_TRAINING_SEEDS)
        self.assertEqual(main_stages, ("screening", "pilot", "main"))

    def test_incomplete_runs_are_reported_without_zero_valued_evidence(self) -> None:
        config = load_day18_config(require_probe_states=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "experiments" / "day18" / "manifest.json"
            manifest = build_day18_manifest(
                config,
                manifest_path=manifest_path,
                runs_root=root / "runs",
                evaluations_root=root / "evaluations",
            )
            write_json(manifest_path, manifest)
            report = build_day18_report(manifest_path)

        self.assertEqual(report["training"]["completed_entry_count"], 0)
        self.assertEqual(report["evaluation"]["completed_entry_count"], 0)
        self.assertEqual(report["conclusion"]["code"], "D")
        self.assertFalse(report["comparison_conditions"]["formal_quality_eligible"])
        self.assertTrue(
            all(entry["summary"] == {} for entry in report["training"]["entries"])
        )


if __name__ == "__main__":
    unittest.main()
