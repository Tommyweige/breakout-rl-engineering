"""Tests for Day 14 profiling evidence helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from profile_batch_size_experiment import _sample_summary
from summarize_thread_profiles import summarize


class Day14ProfilingTests(unittest.TestCase):
    def test_fixed_interval_sample_summary_aggregates_real_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "samples.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "timestamp_utc",
                        "elapsed_seconds",
                        "gpu_index",
                        "gpu_utilization_percent",
                        "gpu_power_watts",
                        "gpu_memory_used_bytes",
                        "gpu_memory_total_bytes",
                        "process_cpu_percent",
                        "process_rss_bytes",
                        "sample_status",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp_utc": "t1",
                            "elapsed_seconds": 0,
                            "gpu_index": 0,
                            "gpu_utilization_percent": 20,
                            "gpu_power_watts": 25,
                            "gpu_memory_used_bytes": 100,
                            "gpu_memory_total_bytes": 1000,
                            "process_cpu_percent": 5,
                            "process_rss_bytes": 10,
                            "sample_status": "ok",
                        },
                        {
                            "timestamp_utc": "t2",
                            "elapsed_seconds": 1,
                            "gpu_index": 0,
                            "gpu_utilization_percent": 40,
                            "gpu_power_watts": 35,
                            "gpu_memory_used_bytes": 200,
                            "gpu_memory_total_bytes": 1000,
                            "process_cpu_percent": 7,
                            "process_rss_bytes": 11,
                            "sample_status": "ok",
                        },
                    ]
                )

            summary = _sample_summary(path, interval_seconds=1.0)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["gpu_sample_count"], 2)
        self.assertEqual(summary["gpu_utilization_percent"]["mean"], 30.0)
        self.assertEqual(summary["gpu_power_watts"]["max"], 35.0)
        self.assertEqual(summary["process_cpu_percent"]["p50"], 6.0)

    def test_thread_selection_requires_one_two_and_four(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "profiles.json"
            source.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "cpu_threads": 1,
                                "status": "completed",
                                "end_to_end_sps": 10,
                                "run_id": "threads-1",
                                "wall_clock_seconds": 1,
                                "finite_metric_counts": {"loss": 1},
                            },
                            {
                                "cpu_threads": 2,
                                "status": "completed",
                                "end_to_end_sps": 12,
                                "run_id": "threads-2",
                                "wall_clock_seconds": 1,
                                "finite_metric_counts": {"loss": 1},
                            },
                            {
                                "cpu_threads": 4,
                                "status": "completed",
                                "end_to_end_sps": 11,
                                "run_id": "threads-4",
                                "wall_clock_seconds": 1,
                                "finite_metric_counts": {"loss": 1},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "selection.json"
            result = summarize(source, output)
            output_exists = output.is_file()

        self.assertEqual(result["selection"]["selected_cpu_threads"], 2)
        self.assertTrue(output_exists)


if __name__ == "__main__":
    unittest.main()
