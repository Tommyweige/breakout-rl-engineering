"""Tests for the machine-readable Day 15/16 environment contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import read_ale_episode_frame
from breakout_rl.evaluation import load_evaluation_config
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.evaluation_artifacts import summary_from_episode_rows
from scripts.evaluation.evaluate_dqn import (
    CONTRACT_V2_OUTPUT_DIRS,
    CONTRACT_V3_OUTPUT_DIRS,
    FORMAL_DQN_OUTPUT_DIR,
    _validate_contract_for_config,
    _output_destination,
    run_evaluation,
)


class Day15ContractTests(unittest.TestCase):
    def test_v3_contract_selects_single_frame_decisions_without_changing_v2(self) -> None:
        v2_path = Path("configs/eval/breakout_contract_v2.json")
        v2 = load_evaluation_contract(v2_path)
        v3 = load_evaluation_contract("configs/eval/breakout_contract_v3.json")

        self.assertEqual(
            hashlib.sha256(v2_path.read_bytes()).hexdigest(),
            "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a",
        )
        self.assertEqual(v2.frame_skip, 4)
        self.assertEqual(v3.contract_id, "breakout-evaluation-v3-frame-skip-1")
        self.assertEqual(v3.frame_skip, 1)
        self.assertEqual(v3.frame_stack, v2.frame_stack)
        self.assertEqual(
            v3.sticky_action_probability,
            v2.sticky_action_probability,
        )
        self.assertEqual(v3.fire_reset, v2.fire_reset)
        self.assertEqual(v3.terminal_on_life_loss, v2.terminal_on_life_loss)
        self.assertEqual(
            v3.time_limit_semantics["max_num_frames_per_episode"],
            v2.time_limit_semantics["max_num_frames_per_episode"],
        )
        self.assertEqual(v3.time_limit_semantics["agent_step_limit"], 108000)

        kwargs = breakout_environment_kwargs(v3, allow_contract_v3=True)
        self.assertEqual(kwargs["frame_skip"], 1)
        validate_breakout_runtime_contract(v3, allow_contract_v3=True)

        env = make_breakout_env(**kwargs)
        try:
            observation, _ = env.reset(seed=123)
            self.assertEqual(observation.shape, (4, 84, 84))
            self.assertEqual(observation.dtype, np.uint8)
            for action in (0, 2, 3, 0, 3, 2, 0, 2):
                before = read_ale_episode_frame(env)
                self.assertIsNotNone(before)
                env.step(action)
                after = read_ale_episode_frame(env)
                self.assertEqual(after - before, 1)
            self.assertEqual(getattr(env.unwrapped, "_frameskip", None), 1)
        finally:
            env.close()

    def test_contract_v3_requires_explicit_precision_opt_in(self) -> None:
        contract = load_evaluation_contract("configs/eval/breakout_contract_v3.json")

        with self.assertRaisesRegex(ValueError, "allow_contract_v3"):
            validate_breakout_runtime_contract(contract)
        with self.assertRaisesRegex(ValueError, "frame_skip=1"):
            validate_breakout_runtime_contract(
                replace(contract, frame_skip=4),
                allow_contract_v3=True,
            )

    def test_contract_v2_keeps_four_native_frames_per_policy_decision(self) -> None:
        contract = load_evaluation_contract("configs/eval/breakout_contract_v2.json")
        env = make_breakout_env(**breakout_environment_kwargs(contract))
        try:
            env.reset(seed=123)
            before = read_ale_episode_frame(env)
            self.assertIsNotNone(before)
            env.step(0)
            after = read_ale_episode_frame(env)
            self.assertEqual(after - before, 4)
            self.assertEqual(getattr(env.unwrapped, "_frameskip", None), 1)
        finally:
            env.close()

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
                "fire_reset_confirmation": {
                    "max_fire_attempts": 8,
                    "confirmation_steps": 2,
                    "min_observation_change_fraction": 1e-4,
                    "confirmation_operator": "any",
                    "confirmation_signals": [
                        "raw_reward",
                        "observation_activity_streak",
                    ],
                },
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
        self.assertEqual(contract.fire_reset_confirmation.max_fire_attempts, 8)
        self.assertEqual(contract.fire_reset_confirmation.confirmation_steps, 2)
        self.assertEqual(
            contract.fire_reset_confirmation.min_observation_change_fraction,
            1e-4,
        )
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
                    "fire_reset_confirmation": {
                        "max_fire_attempts": 8,
                        "confirmation_steps": 2,
                        "min_observation_change_fraction": 1e-4,
                        "confirmation_operator": "any",
                        "confirmation_signals": [
                            "raw_reward",
                            "observation_activity_streak",
                        ],
                    },
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
        self.assertEqual(
            contract.fire_reset_confirmation.confirmation_operator,
            "any",
        )
        self.assertEqual(
            contract.fire_reset_confirmation.confirmation_signals,
            ("raw_reward", "observation_activity_streak"),
        )
        self.assertEqual(len(contract.concrete_episode_seeds), 15)
        self.assertEqual(contract.time_limit_semantics["source"], "ale.game_truncated")
        environment_kwargs = breakout_environment_kwargs(contract)
        self.assertEqual(environment_kwargs["fire_reset_max_attempts"], 8)
        self.assertEqual(environment_kwargs["fire_confirmation_steps"], 2)
        self.assertEqual(environment_kwargs["fire_confirmation_change_fraction"], 1e-4)
        self.assertEqual(environment_kwargs["sticky_action_probability"], 0.25)
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

    def test_contract_v3_uses_separate_evaluation_output_identity(self) -> None:
        args = Namespace(device="cpu", output_dir=None, evaluation_id=None)

        output_dir, evaluation_id = _output_destination(
            "dqn",
            args,
            contract_id="breakout-evaluation-v3-frame-skip-1",
        )

        self.assertEqual(output_dir, CONTRACT_V3_OUTPUT_DIRS["dqn"])
        self.assertEqual(evaluation_id, "contract-v3-dqn")
        with self.assertRaisesRegex(ValueError, "another contract"):
            _output_destination(
                "dqn",
                Namespace(
                    device="cpu",
                    output_dir=CONTRACT_V2_OUTPUT_DIRS["dqn"],
                    evaluation_id=None,
                ),
                contract_id="breakout-evaluation-v3-frame-skip-1",
            )

    def test_contract_v3_matches_the_shared_seed_and_scoring_protocol(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v3.json")
        )
        evaluation_config = load_evaluation_config(
            Path("configs/eval/breakout_eval.json")
        )

        _validate_contract_for_config(contract, evaluation_config)

    def test_contract_v3_dqn_evaluation_uses_checkpoint_contract_provenance(self) -> None:
        loaded = SimpleNamespace(
            model=object(),
            model_id="v3-checkpoint",
            training_metadata={
                "contract_id": "breakout-evaluation-v3-frame-skip-1",
                "contract_path": "configs/eval/breakout_contract_v3.json",
                "training_seed": 11,
            },
            checkpoint_metadata={
                "path": "runs/v3/checkpoints/step-00000001.pth",
                "sha256": "a" * 64,
                "step": 1,
            },
        )
        result = SimpleNamespace(
            to_dict=lambda: {"model_id": "v3-checkpoint", "summary": {}}
        )
        args = Namespace(
            policy="dqn",
            checkpoint=Path("runs/v3/checkpoints/step-00000001.pth"),
            config=Path("configs/eval/breakout_eval.json"),
            contract=Path("configs/eval/breakout_contract_v3.json"),
            device="cpu",
            output_dir=None,
            evaluation_id=None,
            source_day14_manifest=None,
            source_day14_profiling_report=None,
        )

        with (
            patch(
                "scripts.evaluation.evaluate_dqn.load_dqn_checkpoint",
                return_value=loaded,
            ) as load_checkpoint,
            patch(
                "scripts.evaluation.evaluate_dqn.load_day14_provenance",
                side_effect=AssertionError("v3 evaluation must not use Day 14 v2 evidence"),
            ),
            patch(
                "scripts.evaluation.evaluate_dqn.evaluate_policy",
                return_value=result,
            ) as evaluate,
            patch(
                "scripts.evaluation.evaluate_dqn.write_evaluation_artifacts",
                return_value=(Path("results.json"), Path("episodes.csv")),
            ) as write_artifacts,
        ):
            results_path, episodes_path, payload = run_evaluation(args)

        self.assertEqual(results_path, Path("results.json"))
        self.assertEqual(episodes_path, Path("episodes.csv"))
        self.assertEqual(payload["model_id"], "v3-checkpoint")
        checkpoint_env_factory = load_checkpoint.call_args.kwargs["env_factory"]
        env = checkpoint_env_factory()
        try:
            self.assertEqual(env.env.env.frame_skip, 1)
        finally:
            env.close()
        self.assertEqual(
            evaluate.call_args.kwargs["training_metadata"]["source_of_truth"],
            "checkpoint training contract metadata",
        )
        self.assertEqual(
            evaluate.call_args.kwargs["evaluation_id"],
            "contract-v3-dqn",
        )
        self.assertEqual(write_artifacts.call_args.args[1], CONTRACT_V3_OUTPUT_DIRS["dqn"])

    def test_runtime_validator_rejects_noncanonical_stack_or_fire_reset(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )
        with self.assertRaisesRegex(ValueError, "frame_skip=4"):
            validate_breakout_runtime_contract(replace(contract, frame_skip=1))
        with self.assertRaisesRegex(ValueError, "frame_stack=4"):
            validate_breakout_runtime_contract(replace(contract, frame_stack=3))
        with self.assertRaisesRegex(ValueError, "fire_reset=true"):
            validate_breakout_runtime_contract(replace(contract, fire_reset=False))

        with self.assertRaisesRegex(ValueError, "max_fire_attempts=8"):
            validate_breakout_runtime_contract(
                replace(
                    contract,
                    fire_reset_confirmation=replace(
                        contract.fire_reset_confirmation,
                        max_fire_attempts=4,
                    ),
                )
            )

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

    def test_runtime_validator_rejects_external_time_limit_semantics(self) -> None:
        contract = load_evaluation_contract(
            Path("configs/eval/breakout_contract_v2.json")
        )
        time_limit = dict(contract.time_limit_semantics)
        time_limit["external_time_limit_wrapper"] = True

        with self.assertRaisesRegex(ValueError, "external TimeLimit"):
            validate_breakout_runtime_contract(
                replace(contract, time_limit_semantics=time_limit)
            )


if __name__ == "__main__":
    unittest.main()
