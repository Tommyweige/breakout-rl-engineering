from __future__ import annotations

import unittest
from collections import deque
import json

import numpy as np

from breakout_rl.issue45_loss_window_audit import (
    FRAME_LIMIT, SEEDS, WINDOW_DECISIONS, append_decision_window,
    serialize_window_decisions, verified_clear_provenance, window_index_entry,
)
from breakout_rl.completion import BREAKOUT_COMPLETION_DETECTOR_ID, BREAKOUT_COMPLETION_SOURCE
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import BREAKOUT_CONTRACT_V2_ID
from breakout_rl.issue41_onnx_dqn_clear import CONTRACT_SHA256, MODEL_SHA256


class Issue45PureTests(unittest.TestCase):
    def test_frozen_seeds_caps_and_window(self):
        self.assertEqual(SEEDS, (103, 204, 305))
        self.assertEqual(FRAME_LIMIT, 108_000)
        self.assertEqual(WINDOW_DECISIONS, 8)

    def test_ring_preserves_exact_uint8_stack_and_only_last_eight(self):
        ring: deque[dict] = deque()
        stacks = []
        for step in range(10):
            stack = np.full((4, 84, 84), step, dtype=np.uint8)
            stacks.append(stack)
            append_decision_window(ring, observation=stack, seed=103, episode_index=1,
                step=step + 1, emulator_frame=step * 4, q_values=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                requested_action=3, ale_input_action=2)
        self.assertEqual(len(ring), 8)
        np.testing.assert_array_equal(ring[0]["observation"], stacks[2])
        np.testing.assert_array_equal(ring[-1]["observation"], stacks[9])
        self.assertEqual(ring[-1]["raw_q_values"].dtype, np.float32)
        self.assertEqual(ring[-1]["requested_action_meaning"], "LEFT")
        self.assertEqual(ring[-1]["ale_input_action_meaning"], "RIGHT")
        encoded = json.dumps(serialize_window_decisions(list(ring)))
        decoded = json.loads(encoded)
        self.assertEqual(decoded[-1]["raw_q_dtype"], "float32")
        self.assertEqual(decoded[-1]["raw_q_values"], [0.0, 1.0, 2.0, 3.0])

    def test_short_window_and_input_validation(self):
        ring: deque[dict] = deque()
        append_decision_window(ring, observation=np.zeros((4, 84, 84), dtype=np.uint8),
            seed=103, episode_index=1, step=1, emulator_frame=4,
            q_values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), requested_action=0)
        self.assertEqual(len(ring), 1)
        with self.assertRaises(ValueError):
            append_decision_window(ring, observation=np.zeros((84, 84), dtype=np.uint8),
                seed=103, episode_index=1, step=2, emulator_frame=8,
                q_values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), requested_action=0)
        with self.assertRaises(ValueError):
            append_decision_window(ring, observation=np.zeros((4, 84, 84), dtype=np.uint8),
                seed=103, episode_index=1, step=2, emulator_frame=8,
                q_values=np.array([1.0, 2.0, float("nan"), 4.0], dtype=np.float32), requested_action=0)

    def test_loss_window_array_index_binds_event_and_json_metadata(self):
        decision = {"seed": 204, "episode_index": 2, "agent_step": 12,
            "emulator_frame": 48, "observation_sha256": "a" * 64,
            "observation": np.zeros((4, 84, 84), dtype=np.uint8),
            "raw_q_values": np.asarray([1, 2, 3, 4], dtype=np.float32),
            "requested_action": 2, "requested_action_meaning": "RIGHT",
            "ale_input_action": 1, "ale_input_action_meaning": "FIRE"}
        event = {"seed": 204, "episode_index": 2, "life_loss_event": 3,
            "life_loss_count": 1, "agent_step": 12, "emulator_frame": 52,
            "decisions": [decision], "post_step_observation": np.zeros((4, 84, 84), dtype=np.uint8),
            "post_step_observation_sha256": "b" * 64,
            "post_step_observation_unavailable_reason": None}
        entry = window_index_entry(event, "window_0002")
        self.assertEqual({k: entry[k] for k in ("seed", "episode_index", "life_loss_event",
            "life_loss_count", "loss_agent_step", "loss_emulator_frame")},
            {"seed": 204, "episode_index": 2, "life_loss_event": 3,
             "life_loss_count": 1, "loss_agent_step": 12, "loss_emulator_frame": 52})
        encoded = json.dumps(entry)
        self.assertIn('"array_key": "window_0002"', encoded)
        self.assertIn('"post_step_array_key": "window_0002_post_step"', encoded)

    def test_clear_gate_accepts_full_issue45_seed_mapping_and_fails_closed(self):
        commit, digest = "a" * 40, "b" * 64
        provenance = {field: "present" for field in VERIFIED_CLEAR_PROVENANCE_FIELDS}
        provenance.update({
            "checkpoint_id": MODEL_SHA256, "training_seed": 2022,
            "training_transition_count": 2_500_000, "evaluation_seed": 204,
            "episode_seed": 204, "episode_index": 2,
            "contract_id": BREAKOUT_CONTRACT_V2_ID, "contract_sha256": CONTRACT_SHA256,
            "source_commit": commit, "source_working_tree_dirty": False,
            "completion_source_sha256": digest, "raw_score": 864.0, "clear_score": 864.0,
            "clear_agent_step": 100, "clear_emulator_frame": 400, "lives_remaining_at_clear": 2,
            "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
            "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "contract_validation_status": "canonical_contract_v2",
        })
        self.assertEqual(verified_clear_provenance(provenance, commit=commit, digest=digest,
            seed=204, episode_index=2), (True, []))
        altered = dict(provenance, episode_index=3)
        valid, missing = verified_clear_provenance(altered, commit=commit, digest=digest,
            seed=204, episode_index=3)
        self.assertFalse(valid)
        self.assertIn("episode_index", missing)


if __name__ == "__main__":
    unittest.main()
