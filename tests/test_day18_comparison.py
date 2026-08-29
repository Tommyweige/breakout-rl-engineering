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
    normalize_training_stage_accounting,
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

    def test_historical_resumed_rates_are_rebuilt_from_prior_stage_counters(self) -> None:
        def report(stage: str, target: int, vector: int, updates: int) -> dict:
            return {
                "algorithm": "dqn",
                "training_seed": 11,
                "stage": stage,
                "target_transitions": target,
                "config": {"batch_size": 32},
                "summary": {
                    "status": "completed",
                    "total_transitions": target,
                    "vector_iterations": vector,
                    "optimizer_updates": updates,
                    "runtime": {
                        "wall_clock_seconds": 10.0,
                        "stage_start_step": 0 if stage == "screening" else 100,
                        "physical_environment_steps": target,
                        "action_inference_batches": vector,
                        "action_inference_transitions": target,
                        "replay_insertion_calls": vector,
                        "replay_insertion_transitions": target,
                    },
                },
                "runtime": {},
            }

        reports = [
            report("screening", 100, 50, 25),
            report("pilot", 250, 125, 62),
        ]
        normalize_training_stage_accounting(reports)
        pilot = reports[1]
        self.assertEqual(
            pilot["summary"]["stage_counters"],
            {
                "vector_iterations": 75,
                "optimizer_updates": 37,
                "action_inference_batches": 75,
                "action_inference_transitions": 150,
                "replay_insertion_calls": 75,
                "replay_insertion_transitions": 150,
            },
        )
        self.assertAlmostEqual(
            pilot["runtime"]["optimizer_updates_per_second"],
            3.7,
        )
        self.assertAlmostEqual(
            pilot["runtime"]["training_samples_per_second"],
            118.4,
        )
        self.assertEqual(
            pilot["runtime"]["throughput_accounting"]["stage_start_source"],
            "previous_stage_cumulative_counters",
        )


if __name__ == "__main__":
    unittest.main()
