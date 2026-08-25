"""Tests for Day 14 config inheritance and experiment manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from breakout_rl.experiments import (
    build_manifest,
    config_diff,
    load_experiment_config,
    load_experiment_configs,
    read_json_object,
    update_manifest,
)
from breakout_rl.training.config import DQNConfig
from run_experiments import build_parser, run_batch


class ExperimentConfigTests(unittest.TestCase):
    def _write_config(self, path: Path, values: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values), encoding="utf-8")

    def test_base_plus_override_changes_only_expected_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "dqn_baseline.json"
            baseline_values = DQNConfig(
                total_steps=8,
                batch_size=2,
                replay_capacity=8,
                learning_starts=2,
                device="cpu",
            ).to_dict()
            baseline_values["name"] = "baseline"
            self._write_config(baseline, baseline_values)
            variant = root / "experiments" / "lr-low.json"
            self._write_config(
                variant,
                {
                    "name": "lr-low",
                    "base_config": "../dqn_baseline.json",
                    "overrides": {"learning_rate": 0.00005},
                },
            )

            base = load_experiment_config(baseline)
            low = load_experiment_config(variant)

            self.assertEqual(low.config.learning_rate, 0.00005)
            self.assertEqual(
                config_diff(base.values, low.values),
                {
                    "learning_rate": {
                        "base": base.config.learning_rate,
                        "variant": 0.00005,
                    }
                },
            )
            self.assertEqual(low.config.requested_device, "cpu")
            self.assertEqual(low.config.precision, "float32")

    def test_unknown_config_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.json"
            self._write_config(path, {"learning_rate": 0.001, "not_a_config": True})
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_manifest_is_round_trippable_and_records_changed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = root / "baseline.json"
            variant_path = root / "variant.json"
            values = DQNConfig(
                total_steps=8,
                batch_size=2,
                replay_capacity=8,
                learning_starts=2,
                device="cpu",
            ).to_dict()
            self._write_config(baseline_path, {**values, "name": "baseline"})
            self._write_config(
                variant_path,
                {
                    **values,
                    "name": "variant",
                    "learning_rate": 0.0002,
                },
            )
            configs = load_experiment_configs([baseline_path, variant_path])
            manifest_path = root / "experiments" / "demo" / "manifest.json"
            manifest = build_manifest(
                experiment_id="demo",
                configs=configs,
                manifest_path=manifest_path,
            )
            manifest["variants"][0]["run_dir"] = "../../runs/demo/baseline"
            manifest["variants"][1]["run_dir"] = "../../runs/demo/variant"
            update_manifest(manifest_path, manifest)
            restored = read_json_object(manifest_path)

            self.assertEqual(restored["schema_version"], 1)
            self.assertEqual(restored["base_config"]["values"]["device"], "cpu")
            self.assertEqual(restored["variants"][1]["changed_fields"], ["learning_rate"])
            self.assertEqual(restored["variants"][1]["run_dir"], "../../runs/demo/variant")

    def test_dry_run_does_not_start_training_and_second_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            self._write_config(
                config_path,
                DQNConfig(
                    total_steps=8,
                    batch_size=2,
                    replay_capacity=8,
                    learning_starts=2,
                    device="cpu",
                ).to_dict(),
            )
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--experiment-id",
                    "dry-run",
                    "--dry-run",
                    "--experiments-root",
                    str(root / "experiments"),
                    "--runs-root",
                    str(root / "runs"),
                    str(config_path),
                ]
            )
            exit_code, manifest_path, manifest = run_batch(args)
            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "planned")
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest["variants"][0]["status"], "planned")
            with self.assertRaises(FileExistsError):
                run_batch(args)


if __name__ == "__main__":
    unittest.main()
