from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from breakout_rl.issue41_onnx_dqn_clear import (
    ACTION_MEANINGS,
    FRAME_LIMIT,
    MODEL_SHA256,
    SEEDS,
    complete_clear_provenance,
    load_frozen_inputs,
    prepare_onnx_input,
    select_greedy_action,
)
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS


ROOT = Path(__file__).resolve().parents[1]


class FrozenInputTests(unittest.TestCase):
    def test_frozen_artifact_contract_and_inference_bindings(self):
        spec, metadata, contract = load_frozen_inputs(
            ROOT / "configs/eval/breakout_contract_v2.json",
            ROOT / "configs/inference/inference_spec.json",
            ROOT / "web/public/models/final_model/model.onnx",
            ROOT / "web/public/models/final_model/model.onnx.metadata.json",
        )
        self.assertEqual(contract.contract_id, "day15-breakout-evaluation-v2-fire-reset")
        self.assertEqual(tuple(spec["actions"]["meanings"]), ACTION_MEANINGS)
        self.assertEqual(metadata["model_sha256"], MODEL_SHA256)

    def test_frozen_seed_order_and_frame_cap(self):
        self.assertEqual(SEEDS, (101, 202, 303))
        self.assertEqual(FRAME_LIMIT, 108000)

    def test_uint8_adapter_normalizes_once_to_batched_float32(self):
        observation = np.full((4, 84, 84), 255, dtype=np.uint8)
        result = prepare_onnx_input(observation)
        self.assertEqual(result.shape, (1, 4, 84, 84))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result == 1.0))
        self.assertEqual(select_greedy_action(np.asarray([[0, 1, 4, 2]], dtype=np.float32)), 2)
        self.assertEqual(ACTION_MEANINGS[select_greedy_action(np.asarray([[0, 1, 4, 2]], dtype=np.float32))], "RIGHT")

    def test_verified_clear_gate_requires_every_repository_provenance_field(self):
        complete = {field: "present" for field in VERIFIED_CLEAR_PROVENANCE_FIELDS}
        complete.update({"checkpoint_id": MODEL_SHA256, "training_seed": 2022,
            "training_transition_count": 2_500_000, "contract_id": "day15-breakout-evaluation-v2-fire-reset",
            "contract_sha256": "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a",
            "source_working_tree_dirty": False, "completion_detector_id": "ale-breakout-two-wall-score-864-v2",
            "contract_validation_status": "canonical_contract_v2", "clear_score": 864.0, "raw_score": 864.0})
        self.assertEqual(complete_clear_provenance(complete), (True, []))
        incomplete = dict(complete)
        incomplete["clear_emulator_frame"] = None
        verified, missing = complete_clear_provenance(incomplete)
        self.assertFalse(verified)
        self.assertIn("clear_emulator_frame", missing)


if __name__ == "__main__":
    unittest.main()
