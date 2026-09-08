"""Tests for the public Day 24 benchmark measurement and artifact seams."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from breakout_rl.benchmarking import (
    BenchmarkBlockedError,
    BenchmarkConfig,
    InferenceBackend,
    build_benchmark_id,
    load_benchmark_artifacts,
    require_cuda_matrix,
    run_benchmark_target,
    summarize_samples,
    write_benchmark_artifacts,
)


class BenchmarkingTests(unittest.TestCase):
    def test_summary_uses_raw_samples_for_p50_p95_and_throughput(self) -> None:
        summary = summarize_samples(
            [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            batch_size=1,
        )

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min_ns"], 1_000_000)
        self.assertEqual(summary["max_ns"], 4_000_000)
        self.assertEqual(summary["p50_ns"], 2_500_000)
        self.assertAlmostEqual(summary["p95_ns"], 3_850_000)
        self.assertAlmostEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["throughput_per_second"], 400.0)

    def test_benchmark_id_is_deterministic_for_the_same_inputs(self) -> None:
        inputs = {
            "source_model_sha256": "model",
            "onnx_model_sha256": "onnx",
            "batch_sizes": [1, 4],
            "iterations": 10,
        }

        self.assertEqual(build_benchmark_id(inputs), build_benchmark_id(inputs))
        self.assertTrue(build_benchmark_id(inputs).startswith("day24-"))

    def test_cuda_matrix_is_blocked_instead_of_falling_back(self) -> None:
        with self.assertRaisesRegex(BenchmarkBlockedError, "PyTorch CUDA"):
            require_cuda_matrix(
                torch_cuda_available=False,
                onnxruntime_providers=["CPUExecutionProvider"],
            )

        with self.assertRaisesRegex(BenchmarkBlockedError, "ONNX Runtime CUDA"):
            require_cuda_matrix(
                torch_cuda_available=True,
                onnxruntime_providers=["CPUExecutionProvider"],
            )

    def test_model_only_uses_prevalidated_call_but_end_to_end_uses_policy_call(self) -> None:
        observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        calls = {"production": 0, "prevalidated": 0}

        def production_call(value: np.ndarray) -> np.ndarray:
            calls["production"] += 1
            return np.zeros((value.shape[0], 4), dtype=np.float32)

        def prevalidated_call(value: np.ndarray) -> np.ndarray:
            calls["prevalidated"] += 1
            return np.zeros((value.shape[0], 4), dtype=np.float32)

        backend = InferenceBackend(
            runtime="test-runtime",
            requested_provider="cpu",
            actual_provider="test-cpu",
            precision="float32",
            cpu_threads=1,
            thread_setting="1",
            input_ownership="numpy.uint8/cpu",
            output_ownership="numpy.float32/cpu",
            prepare_model_input=lambda value: value.astype(np.float32) / 255.0,
            run_model_input=production_call,
            run_prevalidated_model_input=prevalidated_call,
            materialize_output=lambda value: np.asarray(value),
        )
        config = BenchmarkConfig(warmup_iterations=1, iterations=2, batch_sizes=(1,))

        result = run_benchmark_target(backend, observations, config=config)

        self.assertEqual(calls["prevalidated"], 3)
        self.assertEqual(calls["production"], 3)
        model_only = next(
            value for value in result["summary"]["results"] if value["scope"] == "model_only"
        )
        end_to_end = next(
            value for value in result["summary"]["results"] if value["scope"] == "end_to_end"
        )
        self.assertEqual(model_only["timing_semantics"], "prevalidated_runtime_call")
        self.assertEqual(end_to_end["timing_semantics"], "production_policy_decision_path")

    def test_raw_samples_and_summary_round_trip(self) -> None:
        observations = np.zeros((4, 4, 84, 84), dtype=np.uint8)
        backend = InferenceBackend(
            runtime="test-runtime",
            requested_provider="cpu",
            actual_provider="test-cpu",
            precision="float32",
            cpu_threads=1,
            thread_setting="1",
            input_ownership="numpy.uint8/cpu",
            output_ownership="numpy.float32/cpu",
            prepare_model_input=lambda value: value.astype(np.float32) / 255.0,
            run_model_input=lambda value: np.zeros((value.shape[0], 4), dtype=np.float32),
            materialize_output=lambda value: np.asarray(value),
        )
        config = BenchmarkConfig(warmup_iterations=1, iterations=4, batch_sizes=(1,))

        result = run_benchmark_target(
            backend,
            observations,
            config=config,
            initialization_ns=123,
        )
        payload = {
            "artifact_type": "day24_inference_benchmark",
            "benchmark_id": "day24-test",
            "summary": result["summary"],
            "raw_samples": result["raw_samples"],
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "benchmark"
            write_benchmark_artifacts(output_dir, payload)
            loaded = load_benchmark_artifacts(output_dir)

        self.assertEqual(loaded["benchmark_id"], "day24-test")
        self.assertEqual(len(loaded["raw_samples"]), 8)
        entry = next(
            value
            for value in loaded["results"]
            if value["scope"] == "end_to_end"
        )
        self.assertEqual(entry["batch_size"], 1)
        self.assertEqual(entry["sample_count"], 4)
        self.assertIn("p50_ns", entry["latency"])

    def test_benchmark_artifact_type_can_be_overridden_for_a_new_experiment(self) -> None:
        observations = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        backend = InferenceBackend(
            runtime="test-runtime",
            requested_provider="cpu",
            actual_provider="test-cpu",
            precision="float16",
            cpu_threads=1,
            thread_setting="1",
            input_ownership="numpy.uint8/cpu",
            output_ownership="numpy.float32/cpu",
            prepare_model_input=lambda value: value.astype(np.float32) / 255.0,
            run_model_input=lambda value: np.zeros((value.shape[0], 4), dtype=np.float32),
            materialize_output=lambda value: np.asarray(value),
        )
        measured = run_benchmark_target(
            backend,
            observations,
            config=BenchmarkConfig(warmup_iterations=0, iterations=1, batch_sizes=(1,)),
        )

        with tempfile.TemporaryDirectory() as directory:
            write_benchmark_artifacts(
                Path(directory),
                {
                    "artifact_type": "day25_precision_benchmark",
                    "benchmark_id": "day25-test",
                    "summary": measured["summary"],
                    "raw_samples": measured["raw_samples"],
                },
            )
            summary = json.loads((Path(directory) / "summary.json").read_text())
            raw = json.loads((Path(directory) / "raw-samples.json").read_text())

        self.assertEqual(summary["artifact_type"], "day25_precision_benchmark_summary")
        self.assertEqual(raw["artifact_type"], "day25_precision_benchmark_raw_samples")


if __name__ == "__main__":
    unittest.main()
