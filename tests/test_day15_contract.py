"""Tests for the machine-readable Day 15/16 environment contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from breakout_rl.evaluation import load_evaluation_config
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.evaluation_artifacts import summary_from_episode_rows
from evaluate_dqn import (
    CONTRACT_V2_OUTPUT_DIRS,
    FORMAL_DQN_OUTPUT_DIR,
    _validate_contract_for_config,
    _output_destination,
)


class Day15ContractTests(unittest.TestCase):
    def test_concrete_seed_expansion_is_stable_and_traceable(self) -> None:
        self.assertEqual(
            expand_concrete_episode_seeds([101, 202, 303], episodes_per_seed=5),
            (101, 102, 103, 104, 105, 202, 203, 204, 205, 206, 303, 304, 305, 306, 307),
        )

    def test_contract_round_trip_preserves_environment_semantics(self) -> None:
        contract = BreakoutEvaluationContractV2.from_mapping(
            {
                "schema_version": 2,
                "contract_id": "day15-breakout-v2",
                "environment_id": "ALE/Breakout-v5",
                "frame_skip": 4,
                "frame_stack": 4,
                "sticky_action_probability": 0.25,
                "fire_reset": True,
                "terminal_on_life_loss": False,
                "time_limit_semantics": {
                    "source": "ale.game_truncated",
                    "max_num_frames_per_episode": 108000,
                    "agent_step_limit": 27000,
                    "truncated_is_finished": True,
                },
                "concrete_episode_seeds": [101, 102, 202],
                "evaluation_epsilon": 0.0,
                "raw_reward_rule": "sum environment rewards without clipping",
            }
        )

        self.assertEqual(contract.schema_version, 2)
        self.assertTrue(contract.fire_reset)
        self.assertEqual(contract.time_limit_semantics["agent_step_limit"], 27000)
        self.assertEqual(contract.to_dict()["concrete_episode_seeds"], [101, 102, 202])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(
                json.dumps(contract.to_dict()),
                encoding="utf-8",
            )
            loaded = load_evaluation_contract(path)

        self.assertEqual(loaded, contract)

    def test_contract_rejects_incomplete_time_limit_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_limit_semantics"):
            BreakoutEvaluationContractV2.from_mapping(
                {
                    "schema_version": 2,
                    "contract_id": "broken",
                    "environment_id": "ALE/Breakout-v5",
                    "frame_skip": 4,
                    "frame_stack": 4,
                    "sticky_action_probability": 0.25,
                    "fire_reset": False,
                    "terminal_on_life_loss": False,
                    "time_limit_semantics": {"source": "unknown"},
                    "concrete_episode_seeds": [101],
                    "evaluation_epsilon": 0.0,
                    "raw_reward_rule": "raw",
                }
            )

    def test_committed_contract_v2_is_day16_reusable(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )

        self.assertEqual(contract.contract_id, "day15-breakout-evaluation-v2-fire-reset")
        self.assertTrue(contract.fire_reset)
        self.assertEqual(len(contract.concrete_episode_seeds), 15)
        self.assertEqual(contract.time_limit_semantics["source"], "ale.game_truncated")
        validate_breakout_runtime_contract(contract)

    def test_time_limit_summary_separates_finished_episode_outcomes(self) -> None:
        summary = summary_from_episode_rows(
            [
                {
                    "episode_return": 2.0,
                    "episode_length": 10,
                    "terminated": True,
                    "truncated": False,
                    "time_limit": False,
                    "complete": True,
                },
                {
                    "episode_return": 4.0,
                    "episode_length": 27000,
                    "terminated": False,
                    "truncated": True,
                    "time_limit": True,
                    "complete": True,
                },
            ]
        )

        self.assertEqual(summary["finished_episode_count"], 2)
        self.assertEqual(summary["terminated_count"], 1)
        self.assertEqual(summary["truncated_count"], 1)
        self.assertEqual(summary["time_limit_truncated_count"], 1)
        self.assertEqual(summary["mean_return_terminated"], 2.0)
        self.assertEqual(summary["mean_return_truncated"], 4.0)

    def test_contract_v2_output_cannot_overwrite_v1_artifacts(self) -> None:
        args = Namespace(device="cuda", output_dir=None, evaluation_id=None)

        output_dir, evaluation_id = _output_destination(
            "dqn",
            args,
            contract_id="day15-breakout-evaluation-v2-fire-reset",
        )

        self.assertEqual(output_dir, CONTRACT_V2_OUTPUT_DIRS["dqn"])
        self.assertEqual(evaluation_id, "day15-contract-v2-dqn")
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            _output_destination(
                "dqn",
                Namespace(
                    device="cuda",
                    output_dir=FORMAL_DQN_OUTPUT_DIR,
                    evaluation_id=None,
                ),
                contract_id="day15-breakout-evaluation-v2-fire-reset",
            )

    def test_contract_v2_matches_the_fixed_evaluation_protocol(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )
        evaluation_config = load_evaluation_config(
            Path("configs/eval/breakout_eval.json")
        )

        _validate_contract_for_config(contract, evaluation_config)

    def test_runtime_validator_rejects_noncanonical_stack_or_fire_reset(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )
        with self.assertRaisesRegex(ValueError, "frame_stack=4"):
            validate_breakout_runtime_contract(replace(contract, frame_stack=3))
        with self.assertRaisesRegex(ValueError, "fire_reset=true"):
            validate_breakout_runtime_contract(replace(contract, fire_reset=False))

    def test_runtime_validator_rejects_noncanonical_evaluation_scoring(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )
        with self.assertRaisesRegex(ValueError, "evaluation_epsilon=0"):
            validate_breakout_runtime_contract(
                replace(contract, evaluation_epsilon=0.1)
            )
        with self.assertRaisesRegex(ValueError, "raw_reward_rule"):
            validate_breakout_runtime_contract(
                replace(contract, raw_reward_rule="clip")
            )


if __name__ == "__main__":
    unittest.main()
