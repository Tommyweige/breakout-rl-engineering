"""Regression tests for the committed Day 22 ONNX evidence."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from breakout_rl.onnx_artifacts import inspect_onnx_model
from scripts.deployment.export_onnx import _cuda_runtime


ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None


@unittest.skipUnless(ONNX_AVAILABLE, "onnx is required for graph inspection")
class ONNXArtifactTests(unittest.TestCase):
    def test_committed_model_has_the_declared_io_and_graph_inventory(self) -> None:
        summary = inspect_onnx_model(
            Path("assets/day22/models/final_model/model.onnx"),
            check=True,
        )

        self.assertEqual(summary["node_count"], 16)
        self.assertEqual(summary["initializer_count"], 14)
        self.assertEqual(summary["opset_imports"], {"ai.onnx": 17})
        self.assertEqual(summary["inputs"][0]["name"], "observation")
        self.assertEqual(summary["inputs"][0]["dtype"], "float32")
        self.assertEqual(summary["inputs"][0]["shape"], ["N", 4, 84, 84])
        self.assertEqual(summary["outputs"][0]["name"], "q_values")
        self.assertEqual(summary["outputs"][0]["dtype"], "float32")
        self.assertEqual(summary["outputs"][0]["shape"], ["N", 4])
        self.assertEqual(
            summary["operator_type_counts"],
            {
                "Add": 1,
                "Conv": 3,
                "Flatten": 1,
                "Gemm": 4,
                "ReduceMean": 1,
                "Relu": 5,
                "Sub": 1,
            },
        )

    def test_export_metadata_keeps_cuda_source_and_host_checks_separate(self) -> None:
        metadata = json.loads(
            Path("assets/day22/models/final_model/model.onnx.metadata.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            metadata["source_model_validation"]["runtime"]["cuda_available"]
        )
        self.assertEqual(
            metadata["source_model_validation"]["runtime"]["device"],
            "cuda:0",
        )
        self.assertTrue(metadata["host_graph_validation"]["checker_passed"])
        self.assertEqual(
            metadata["dynamic_batch_validation"]["tested_batch_sizes"], [1, 4]
        )
        graph_path = metadata["host_graph_validation"]["graph"]["model_path"]
        self.assertFalse(Path(graph_path).is_absolute())
        self.assertNotIn("C:/", graph_path)
        self.assertNotIn("E:/", graph_path)

    def test_formal_cuda_runtime_rejects_unavailable_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA source-model validation"):
                _cuda_runtime(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
