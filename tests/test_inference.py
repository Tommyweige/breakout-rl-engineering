"""Tests for the shared native/browser inference contract."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from breakout_rl.inference import (
    EXPECTED_ACTION_MEANINGS,
    InferenceSpec,
    ONNXRuntimePolicy,
    PyTorchPolicy,
    action_meaning,
    load_inference_spec,
    prepare_model_input,
    q_values_to_action,
    validate_action_meanings,
)
from breakout_rl.onnx_parity import compare_q_values


class FixedQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.0, 1.0, 3.0, 2.0]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(observations.shape[0], -1)


class InferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = np.zeros((4, 84, 84), dtype=np.uint8)
        self.observation[0, 0, 0] = 1
        self.observation[1, 0, 1] = 255

    def test_repository_spec_round_trips_and_preserves_action_order(self) -> None:
        spec = load_inference_spec()

        self.assertEqual(spec.input_name, "observation")
        self.assertEqual(spec.input_shape, ("N", 4, 84, 84))
        self.assertEqual(spec.output_shape, ("N", 4))
        self.assertEqual(spec.action_meanings, EXPECTED_ACTION_MEANINGS)
        self.assertEqual(
            InferenceSpec.from_mapping(spec.to_dict()),
            spec,
        )

    def test_prepare_model_input_accepts_chw_and_bchw_once(self) -> None:
        single = prepare_model_input(self.observation, device="cpu")
        batch = prepare_model_input(
            np.stack([self.observation, self.observation], axis=0),
            device="cpu",
        )

        self.assertEqual(tuple(single.shape), (1, 4, 84, 84))
        self.assertEqual(tuple(batch.shape), (2, 4, 84, 84))
        self.assertEqual(single.dtype, torch.float32)
        self.assertAlmostEqual(float(single[0, 0, 0, 0]), 1.0 / 255.0, places=7)
        self.assertEqual(float(single[0, 1, 0, 1]), 1.0)
        self.assertAlmostEqual(float(batch[1, 0, 0, 0]), 1.0 / 255.0, places=7)

    def test_prepare_model_input_rejects_float_or_wrong_shape(self) -> None:
        with self.assertRaises(TypeError):
            prepare_model_input(self.observation.astype(np.float32), device="cpu")
        with self.assertRaises(ValueError):
            prepare_model_input(self.observation[:, :, :83], device="cpu")
        with self.assertRaises(ValueError):
            prepare_model_input(
                np.empty((0, 4, 84, 84), dtype=np.uint8),
                device="cpu",
            )

    def test_q_values_to_action_preserves_argmax_indices_and_tie_rule(self) -> None:
        self.assertEqual(q_values_to_action([0.0, 1.0, 3.0, 2.0]), 2)
        np.testing.assert_array_equal(
            q_values_to_action(
                np.array(
                    [
                        [3.0, 3.0, 1.0, 0.0],
                        [0.0, 1.0, 2.0, 4.0],
                    ],
                    dtype=np.float32,
                )
            ),
            np.array([0, 3], dtype=np.int64),
        )
        self.assertEqual(action_meaning(0), "NOOP")
        self.assertEqual(action_meaning(3), "LEFT")
        self.assertEqual(
            validate_action_meanings(
                ["NOOP", "FIRE", "RIGHT", "LEFT"],
            ),
            EXPECTED_ACTION_MEANINGS,
        )
        with self.assertRaises(ValueError):
            validate_action_meanings(["NOOP", "FIRE", "LEFT", "RIGHT"])

    def test_q_values_to_action_rejects_contract_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            q_values_to_action(np.zeros((1, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            q_values_to_action(np.array([0.0, np.nan, 1.0, 2.0]))

    def test_pytorch_policy_is_eval_no_grad_and_returns_canonical_shapes(self) -> None:
        model = FixedQNetwork().train()
        policy = PyTorchPolicy(model, device="cpu")

        single_q = policy.predict_q_values(self.observation)
        batch_q = policy.predict_q_values(
            np.stack([self.observation, self.observation], axis=0)
        )

        self.assertFalse(model.training)
        self.assertEqual(single_q.shape, (1, 4))
        self.assertEqual(single_q.dtype, np.float32)
        self.assertEqual(policy.select_action(self.observation), 2)
        np.testing.assert_array_equal(
            policy.select_actions(self.observation[None]), [2]
        )
        self.assertEqual(batch_q.shape, (2, 4))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_pytorch_policy_accepts_a_prepared_model_input(self) -> None:
        model = FixedQNetwork()
        policy = PyTorchPolicy(model, device="cpu")
        model_input = prepare_model_input(self.observation, device="cpu")

        q_values = policy.predict_model_input(model_input)

        self.assertIsInstance(q_values, torch.Tensor)
        self.assertEqual(tuple(q_values.shape), (1, 4))
        self.assertEqual(q_values.device.type, "cpu")
        np.testing.assert_allclose(q_values.detach().numpy(), [[0.0, 1.0, 3.0, 2.0]])


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") is not None,
    "onnxruntime is required for ONNX policy tests",
)
class ONNXRuntimePolicyTests(unittest.TestCase):
    MODEL = Path("assets/day22/models/final_model/model.onnx")

    def setUp(self) -> None:
        self.observation = np.zeros((4, 84, 84), dtype=np.uint8)
        self.batch = np.stack([self.observation] * 4, axis=0)

    def test_cpu_policy_validates_dynamic_batch_and_returns_contract_shapes(self) -> None:
        policy = ONNXRuntimePolicy(self.MODEL, provider="cpu")

        single = policy.predict_q_values(self.observation)
        batch = policy.predict_q_values(self.batch)

        self.assertEqual(single.shape, (1, 4))
        self.assertEqual(single.dtype, np.float32)
        self.assertEqual(batch.shape, (4, 4))
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(policy.select_action(self.observation), int(np.argmax(single[0])))
        np.testing.assert_array_equal(
            policy.select_actions(self.batch), np.argmax(batch, axis=1)
        )
        self.assertEqual(policy.runtime_metadata["requested_provider"], "CPUExecutionProvider")
        self.assertEqual(policy.runtime_metadata["actual_provider"], "CPUExecutionProvider")

    def test_cpu_policy_accepts_a_prepared_model_input(self) -> None:
        policy = ONNXRuntimePolicy(self.MODEL, provider="cpu")
        model_input = prepare_model_input(self.observation, device="cpu").numpy()

        q_values = policy.predict_model_input(model_input)

        self.assertEqual(q_values.shape, (1, 4))
        self.assertEqual(q_values.dtype, np.float32)

    def test_requested_cuda_provider_never_silently_falls_back_when_unavailable(self) -> None:
        import onnxruntime as ort

        with patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider.*unavailable"):
                ONNXRuntimePolicy(self.MODEL, provider="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CUDA EP test")
    def test_cuda_policy_records_cuda_primary_and_zero_fallback_nodes(self) -> None:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            self.skipTest("CUDAExecutionProvider is not available")

        policy = ONNXRuntimePolicy(self.MODEL, provider="cuda")

        self.assertEqual(
            policy.runtime_metadata["actual_provider"],
            "CUDAExecutionProvider",
        )
        self.assertEqual(
            policy.runtime_metadata["graph_assignment"]["fallback_node_count"],
            0,
        )
        limits = json.loads(
            Path("configs/inference/parity_validation.json").read_text(
                encoding="utf-8"
            )
        )["thresholds"]["cuda"]
        with np.load("assets/day22/inference/pytorch_reference.npz") as reference:
            observations = reference["observations"]
            reference_q_values = reference["q_values"]
            for batch_size in (1, 4):
                candidate_q_values = np.concatenate(
                    [
                        policy.predict_q_values(
                            observations[start : start + batch_size]
                        )
                        for start in range(0, len(observations), batch_size)
                    ],
                    axis=0,
                )
                metrics = compare_q_values(
                    reference_q_values,
                    candidate_q_values,
                )

                self.assertEqual(candidate_q_values.shape, (len(observations), 4))
                for metric in (
                    "max_absolute_error",
                    "mean_absolute_error",
                    "max_relative_error",
                    "mean_relative_error",
                ):
                    self.assertLessEqual(metrics[metric], limits[metric])
                self.assertGreaterEqual(
                    metrics["action_agreement_rate"], limits["action_agreement_rate"]
                )
                self.assertEqual(metrics["disagreement_sample_ids"], [])


if __name__ == "__main__":
    unittest.main()
