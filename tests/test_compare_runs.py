"""Tests for Day 14 run comparison and evidence plotting."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from breakout_rl.experiments import compare_manifest, compare_run_dirs, load_run_report
from visualize_experiment_comparison import render_comparison


class CompareRunsTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        name: str,
        *,
        learning_rate: float,
        status: str = "completed",
        total_steps: int = 4,
        device: str = "cuda",
    ) -> Path:
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "total_steps": total_steps,
                    "seed": 42,
                    "learning_rate": learning_rate,
                    "device": device,
                    "precision": "float32",
                    "runtime": {
                        "requested_device": device,
                        "resolved_device": "cuda:0" if device == "cuda" else device,
                        "precision": "float32",
                        "steps_per_second": 50.0,
                        "wall_clock_seconds": 0.08,
                        "cuda_peak_allocated_bytes": 123,
                        "cuda_peak_reserved_bytes": 456,
                    },
                }
            ),
            encoding="utf-8",
        )
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "global_step",
                    "raw_episode_return",
                    "loss",
                    "q_mean",
                    "q_max",
                    "q_min",
                    "target_mean",
                    "target_max",
                    "sps",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "global_step": 1,
                        "raw_episode_return": "",
                        "loss": 2.0,
                        "q_mean": 0.5,
                        "q_max": 1.0,
                        "q_min": 0.0,
                        "target_mean": 0.7,
                        "target_max": 1.2,
                        "sps": 40.0,
                    },
                    {
                        "global_step": 2,
                        "raw_episode_return": 3.0,
                        "loss": 1.0,
                        "q_mean": 0.6,
                        "q_max": 1.1,
                        "q_min": 0.1,
                        "target_mean": 0.8,
                        "target_max": 1.3,
                        "sps": 45.0,
                    },
                    {
                        "global_step": 3,
                        "raw_episode_return": "",
                        "loss": 0.5,
                        "q_mean": 0.7,
                        "q_max": 1.2,
                        "q_min": 0.2,
                        "target_mean": 0.9,
                        "target_max": 1.4,
                        "sps": 50.0,
                    },
                    {
                        "global_step": 4,
                        "raw_episode_return": 5.0,
                        "loss": 0.25,
                        "q_mean": 0.8,
                        "q_max": 1.3,
                        "q_min": 0.3,
                        "target_mean": 1.0,
                        "target_max": 1.5,
                        "sps": 55.0,
                    },
                ]
            )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "total_steps": 4,
                    "episodes": 2,
                    "steps_per_second": 50.0,
                    "runtime": {
                        "requested_device": device,
                        "resolved_device": "cuda:0" if device == "cuda" else device,
                        "precision": "float32",
                        "steps_per_second": 50.0,
                        "wall_clock_seconds": 0.08,
                    },
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_aggregate_windows_and_runtime_costs_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._write_run(Path(temporary_directory), "baseline", learning_rate=0.1)
            report = load_run_report(run_dir, recent_window=2, rolling_window=2)

        self.assertEqual(report["completed_steps"], 4)
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["recent_window"], 2)
        self.assertEqual(report["rolling_window"], 2)
        self.assertEqual(report["mean_recent_episode_return"], 4.0)
        self.assertEqual(report["median_recent_episode_return"], 4.0)
        self.assertEqual(report["best_rolling_return"], 4.0)
        self.assertEqual(report["loss_summary"]["count"], 4)
        self.assertEqual(report["gradient_summary"]["count"], 0)
        self.assertEqual(report["recent_return_trend"]["direction"], "up")
        self.assertEqual(report["sps"]["runtime"], 50.0)
        self.assertEqual(report["gpu_memory"]["peak_reserved_bytes"], 456)

    def test_manifest_compare_reports_config_diff_and_cuda_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = self._write_run(root / "runs", "baseline", learning_rate=0.1)
            variant = self._write_run(root / "runs", "lr-low", learning_rate=0.05)
            manifest_path = root / "experiments" / "demo" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "demo",
                        "status": "completed",
                        "sequential": True,
                        "base_config": {
                            "values": {
                                "total_steps": 4,
                                "seed": 42,
                                "learning_rate": 0.1,
                                "device": "cuda",
                                "precision": "float32",
                            }
                        },
                        "variants": [
                            {
                                "label": "baseline",
                                "run_dir": "../../runs/baseline",
                                "seed": 42,
                                "requested_device": "cuda",
                            },
                            {
                                "label": "lr-low",
                                "run_dir": "../../runs/lr-low",
                                "seed": 42,
                                "requested_device": "cuda",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = compare_manifest(manifest_path, recent_window=2, rolling_window=2)
            output = root / "comparison.png"
            render_comparison(manifest_path, output)
            output_exists = output.is_file()
            output_size = output.stat().st_size if output_exists else 0
            metadata_exists = output.with_suffix(".json").is_file()

        self.assertEqual(report["experiment_id"], "demo")
        self.assertTrue(report["comparison_conditions"]["formal_cuda_eligible"])
        self.assertEqual(report["runs"][1]["label"], "lr-low")
        self.assertEqual(
            report["runs"][1]["config_diff"]["learning_rate"],
            {"base": 0.1, "variant": 0.05},
        )
        self.assertTrue(output_exists)
        self.assertGreater(output_size, 0)
        self.assertTrue(metadata_exists)

    def test_incomplete_run_is_not_reported_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._write_run(
                Path(temporary_directory),
                "interrupted",
                learning_rate=0.1,
                status="incomplete",
                total_steps=10,
            )
            report = load_run_report(run_dir, recent_window=2, rolling_window=2)

        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["expected_steps"], 10)
        self.assertEqual(report["completed_steps"], 4)

    def test_unreached_main_milestones_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._write_run(
                Path(temporary_directory),
                "interrupted-main",
                learning_rate=0.1,
                status="incomplete",
                total_steps=100_000,
            )
            report = load_run_report(run_dir, recent_window=2, rolling_window=2)

        self.assertEqual(
            report["milestone_snapshots"],
            {"25000": None, "50000": None, "75000": None, "100000": None},
        )

    def test_unequal_step_budgets_are_exposed_as_a_comparison_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self._write_run(root, "first", learning_rate=0.1, total_steps=4)
            second = self._write_run(root, "second", learning_rate=0.2, total_steps=5)
            comparison = compare_run_dirs(
                [first, second],
                recent_window=2,
                rolling_window=2,
            )

        self.assertFalse(comparison["comparison_conditions"]["same_step_budget"])
        self.assertFalse(comparison["comparison_conditions"]["formal_cuda_eligible"])

    def test_screening_and_main_stages_cannot_form_one_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_run(root / "runs", "screening", learning_rate=0.1)
            self._write_run(root / "runs", "main", learning_rate=0.2)
            manifest_path = root / "experiments" / "mixed" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "mixed",
                        "status": "completed",
                        "base_config": {"values": {"total_steps": 4}},
                        "variants": [
                            {
                                "label": "screening",
                                "run_dir": "../../runs/screening",
                                "stage": "screening",
                            },
                            {
                                "label": "main",
                                "run_dir": "../../runs/main",
                                "stage": "main",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_manifest(
                manifest_path,
                recent_window=2,
                rolling_window=2,
            )

        conditions = comparison["comparison_conditions"]
        self.assertFalse(conditions["same_stage"])
        self.assertEqual(conditions["stages"], ["screening", "main"])
        self.assertFalse(conditions["main_comparison_eligible"])

    def test_failed_run_status_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._write_run(
                Path(temporary_directory),
                "failed",
                learning_rate=0.1,
                status="failed_non_finite",
            )
            (run_dir / "failure.json").write_text(
                json.dumps({"status": "failed", "error": "non-finite loss"}),
                encoding="utf-8",
            )
            report = load_run_report(run_dir, recent_window=2, rolling_window=2)

        self.assertEqual(report["status"], "failed_non_finite")

    def test_interrupted_manifest_keeps_not_started_variants_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = self._write_run(root / "runs", "baseline", learning_rate=0.1)
            manifest_path = root / "experiments" / "demo" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "demo",
                        "status": "interrupted",
                        "base_config": {"values": {"learning_rate": 0.1}},
                        "variants": [
                            {
                                "label": "baseline",
                                "run_dir": "../../runs/baseline",
                                "status": "completed",
                                "config_values": {"learning_rate": 0.1},
                                "requested_device": "cuda",
                                "resolved_device": "cuda:0",
                                "step_budget": 4,
                            },
                            {
                                "label": "not-started",
                                "run_dir": None,
                                "status": "pending",
                                "config_values": {"learning_rate": 0.2},
                                "requested_device": "cuda",
                                "resolved_device": None,
                                "step_budget": 4,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = compare_manifest(manifest_path, recent_window=2, rolling_window=2)

        self.assertEqual([run["label"] for run in report["runs"]], ["baseline", "not-started"])
        self.assertEqual(report["runs"][1]["status"], "not_started")
        self.assertFalse(report["comparison_conditions"]["formal_cuda_eligible"])


if __name__ == "__main__":
    unittest.main()
