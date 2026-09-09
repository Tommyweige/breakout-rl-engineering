"""Tests for the public TensorRT preflight and memory-policy seams."""

from __future__ import annotations

import unittest

from unittest import mock

import numpy as np
import torch

from breakout_rl.benchmarking import BenchmarkConfig, run_benchmark_target
from breakout_rl.inference import load_inference_spec

from breakout_rl.tensorrt import (
    TENSORRT_STATUS_BLOCKED,
    TENSORRT_STATUS_CLI_ONLY,
    TENSORRT_STATUS_READY,
    TensorRTBlockedError,
    TensorRTSession,
    classify_tensorrt_status,
    create_strongly_typed_network,
    disable_tf32,
    require_gpu_baseline,
    strongly_typed_network_flags,
    validate_workspace_budget,
)


class _FakeNetworkFlag:
    STRONGLY_TYPED = 0


class _FakeBuilderFlag:
    TF32 = "TF32"


class _FakeTensorRT:
    NetworkDefinitionCreationFlag = _FakeNetworkFlag
    BuilderFlag = _FakeBuilderFlag


class _FakeBuilder:
    def __init__(self) -> None:
        self.flags = None

    def create_network(self, flags):
        self.flags = flags
        return object()


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.flags = {_FakeBuilderFlag.TF32}

    def clear_flag(self, flag) -> None:
        self.flags.discard(flag)

    def get_flag(self, flag) -> bool:
        return flag in self.flags


class _FakeTensorRTSession:
    device = torch.device("cpu")
    device_index = 0
    runtime_metadata = {
        "runtime": "TensorRT",
        "tensorrt_version": "test",
    }

    def __init__(self) -> None:
        self.public_calls = 0
        self.prevalidated_calls = 0

    def execute(self, model_input: torch.Tensor) -> torch.Tensor:
        self.public_calls += 1
        return torch.zeros((model_input.shape[0], 4), dtype=torch.float32)

    def execute_prevalidated(self, model_input: torch.Tensor) -> torch.Tensor:
        self.prevalidated_calls += 1
        return torch.zeros((model_input.shape[0], 4), dtype=torch.float32)


class TensorRTTests(unittest.TestCase):
    def test_status_distinguishes_python_api_cli_only_and_blocked(self) -> None:
        self.assertEqual(
            classify_tensorrt_status(
                tensorrt_imported=True,
                onnx_parser_available=True,
                builder_available=True,
                strongly_typed_network_available=True,
                trtexec_available=False,
            ),
            TENSORRT_STATUS_READY,
        )
        self.assertEqual(
            classify_tensorrt_status(
                tensorrt_imported=False,
                onnx_parser_available=True,
                builder_available=False,
                strongly_typed_network_available=False,
                trtexec_available=True,
            ),
            TENSORRT_STATUS_CLI_ONLY,
        )
        self.assertEqual(
            classify_tensorrt_status(
                tensorrt_imported=False,
                onnx_parser_available=False,
                builder_available=False,
                strongly_typed_network_available=False,
                trtexec_available=False,
            ),
            TENSORRT_STATUS_BLOCKED,
        )
        self.assertEqual(
            classify_tensorrt_status(
                tensorrt_imported=True,
                onnx_parser_available=True,
                builder_available=False,
                strongly_typed_network_available=False,
                trtexec_available=False,
            ),
            TENSORRT_STATUS_BLOCKED,
        )
        self.assertEqual(
            classify_tensorrt_status(
                tensorrt_imported=True,
                onnx_parser_available=True,
                builder_available=True,
                strongly_typed_network_available=False,
                trtexec_available=True,
            ),
            TENSORRT_STATUS_BLOCKED,
        )

    def test_workspace_budget_preserves_declared_headroom(self) -> None:
        self.assertEqual(
            validate_workspace_budget(
                workspace_bytes=2 * 1024**3,
                free_bytes=4 * 1024**3,
                headroom_bytes=1 * 1024**3,
            ),
            2 * 1024**3,
        )
        with self.assertRaisesRegex(TensorRTBlockedError, "headroom"):
            validate_workspace_budget(
                workspace_bytes=4 * 1024**3,
                free_bytes=4 * 1024**3,
                headroom_bytes=1 * 1024**3,
            )

    def test_build_seams_require_strong_typing_and_disable_tf32(self) -> None:
        builder = _FakeBuilder()
        self.assertEqual(strongly_typed_network_flags(_FakeTensorRT), 1)
        self.assertIsNotNone(create_strongly_typed_network(builder, _FakeTensorRT))
        self.assertEqual(builder.flags, 1)

        config = _FakeBuilderConfig()
        self.assertFalse(disable_tf32(config, _FakeTensorRT))
        self.assertFalse(config.get_flag(_FakeBuilderFlag.TF32))

    def test_build_seams_fail_closed_when_precision_controls_are_missing(self) -> None:
        with self.assertRaisesRegex(TensorRTBlockedError, "STRONGLY_TYPED"):
            strongly_typed_network_flags(
                mock.Mock(NetworkDefinitionCreationFlag=object())
            )
        with self.assertRaisesRegex(TensorRTBlockedError, "disable TF32"):
            disable_tf32(object(), mock.Mock(BuilderFlag=object()))

    def test_cli_only_engine_build_is_blocked_without_strong_typing_proof(self) -> None:
        from scripts.deployment.build_tensorrt_engine import _build_with_trtexec

        with self.assertRaisesRegex(TensorRTBlockedError, "strongly typed network"):
            _build_with_trtexec(
                executable="trtexec",
                onnx_model=mock.Mock(),
                output=mock.Mock(),
                precision="float32",
                workspace_bytes=1,
                profile={},
                log_path=mock.Mock(),
            )

    def test_tensorrt_benchmark_uses_prevalidated_runtime_for_both_scopes(self) -> None:
        from scripts.analysis.compare_tensorrt import _tensorrt_backend

        session = _FakeTensorRTSession()
        with mock.patch(
            "scripts.analysis.compare_tensorrt._sync_for", return_value=lambda: None
        ):
            backend = _tensorrt_backend(
                session,
                spec=load_inference_spec(),
                precision="float32",
                initialization_ns=0,
                engine_metadata={
                    "network_typing": "strongly_typed",
                    "tf32_enabled": False,
                },
            )
        result = run_benchmark_target(
            backend,
            np.zeros((1, 4, 84, 84), dtype=np.uint8),
            config=BenchmarkConfig(warmup_iterations=0, iterations=1, batch_sizes=(1,)),
        )

        self.assertEqual(session.public_calls, 0)
        self.assertEqual(session.prevalidated_calls, 2)
        for summary in result["summary"]["results"]:
            self.assertEqual(summary["metadata"]["network_typing"], "strongly_typed")
            self.assertFalse(summary["metadata"]["tf32_enabled"])

    def test_public_execute_retains_finite_and_range_validation(self) -> None:
        session = object.__new__(TensorRTSession)
        session.device = torch.device("cpu")
        session.profile_shapes = (
            (1, 4, 84, 84),
            (1, 4, 84, 84),
            (32, 4, 84, 84),
        )
        invalid = torch.full((1, 4, 84, 84), float("nan"), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "finite values"):
            session.execute(invalid)

    def test_gpu_baseline_rejects_missing_or_mislabeled_cuda(self) -> None:
        with self.assertRaisesRegex(TensorRTBlockedError, "PyTorch CUDA"):
            require_gpu_baseline(
                torch_cuda_available=False,
                ort_actual_provider="CUDAExecutionProvider",
                ort_graph_assignment_status="verified",
            )
        with self.assertRaisesRegex(TensorRTBlockedError, "graph assignment"):
            require_gpu_baseline(
                torch_cuda_available=True,
                ort_actual_provider="CUDAExecutionProvider",
                ort_graph_assignment_status="fallback",
            )


if __name__ == "__main__":
    unittest.main()
