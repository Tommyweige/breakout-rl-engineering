"""Tests for the public Day 25 precision-comparison seams."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from breakout_rl.precision import (
    PrecisionBlockedError,
    PrecisionThresholds,
    build_precision_benchmark_id,
    compare_evaluation_parity,
    compare_precision_outputs,
    require_cuda_precision_matrix,
)
from scripts.deployment.convert_onnx_fp16 import convert_onnx_model


class PrecisionTests(unittest.TestCase):
    def test_cuda_matrix_fails_closed_without_both_gpu_paths(self) -> None:
        with self.assertRaisesRegex(PrecisionBlockedError, "PyTorch CUDA"):
            require_cuda_precision_matrix(
                torch_cuda_available=False,
                onnxruntime_providers=["CPUExecutionProvider"],
            )

        with self.assertRaisesRegex(PrecisionBlockedError, "CUDAExecutionProvider"):
            require_cuda_precision_matrix(
                torch_cuda_available=True,
                onnxruntime_providers=["CPUExecutionProvider"],
            )

    def test_precision_outputs_report_numeric_and_action_parity(self) -> None:
        reference = np.array(
            [[1.0, 2.0, 3.0, 0.0], [4.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        candidate = np.array(
            [[1.0, 2.0, 2.9, 0.0], [4.0, 1.1, 0.0, 0.0]],
            dtype=np.float16,
        )
        thresholds = PrecisionThresholds(
            max_absolute_error=0.2,
            mean_absolute_error=0.1,
            max_relative_error=0.1,
            mean_relative_error=0.05,
            action_agreement_rate=1.0,
        )

        result = compare_precision_outputs(
            reference,
            candidate,
            thresholds=thresholds,
            sample_ids=[10, 20],
        )

        self.assertEqual(result["metrics"]["action_agreement_rate"], 1.0)
        self.assertAlmostEqual(result["metrics"]["max_absolute_error"], 0.1, places=3)
        self.assertTrue(result["thresholds"]["passed"])

    def test_evaluation_parity_counts_action_and_episode_matches(self) -> None:
        result = compare_evaluation_parity(
            reference_action_sequences=[[0, 1, 2], [3]],
            candidate_action_sequences=[[0, 1, 2], [3]],
            reference_returns=[4.0, 1.0],
            candidate_returns=[4.0, 1.0],
            reference_lengths=[3, 1],
            candidate_lengths=[3, 1],
        )

        self.assertEqual(result["action_agreement_rate"], 1.0)
        self.assertEqual(result["episode_return_match_rate"], 1.0)
        self.assertEqual(result["episode_length_match_rate"], 1.0)

    def test_evaluation_parity_records_length_mismatch_as_action_disagreement(self) -> None:
        result = compare_evaluation_parity(
            reference_action_sequences=[[0, 1, 2]],
            candidate_action_sequences=[[0, 1]],
            reference_returns=[4.0],
            candidate_returns=[4.0],
            reference_lengths=[3],
            candidate_lengths=[2],
        )

        self.assertEqual(result["action_agreement_rate"], 2 / 3)
        self.assertEqual(result["action_disagreement_count"], 1)
        self.assertEqual(result["episode_length_match_rate"], 0.0)

    def test_benchmark_id_is_deterministic_and_day_specific(self) -> None:
        identity = {"source": "model", "batch_sizes": [1, 4], "iterations": 10}

        self.assertEqual(
            build_precision_benchmark_id(identity),
            build_precision_benchmark_id(identity),
        )
        self.assertTrue(build_precision_benchmark_id(identity).startswith("day25-"))

    def test_fp16_conversion_keeps_declared_float32_io_in_a_separate_file(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["observation", "weight", "bias"], ["q_values"])],
            "tiny-policy",
            [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("q_values", TensorProto.FLOAT, [None, 2])],
            initializer=[
                numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight"),
                numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="bias"),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_operatorsetid("", 17)],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            output = root / "model.fp16.onnx"
            onnx.save(model, source)

            result = convert_onnx_model(source, output, keep_io_types=True)
            converted = onnx.load(output)
            output_size = output.stat().st_size

        self.assertNotEqual(source, output)
        self.assertEqual(result["model_size_bytes"], output_size)
        self.assertEqual(converted.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(converted.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertTrue(
            any(
                initializer.data_type == TensorProto.FLOAT16
                for initializer in converted.graph.initializer
            )
        )


if __name__ == "__main__":
    unittest.main()
