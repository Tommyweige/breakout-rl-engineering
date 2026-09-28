"""Evaluation integration tests for clear results and their provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_COMPLETION_SOURCE,
    CompletionSupport,
)
from breakout_rl.evaluation import (
    _contract_provenance,
    _contract_runtime_binding,
    evaluate_policy,
    write_evaluation_artifacts,
)
from breakout_rl.evaluation_artifacts import read_evaluation_results
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
)


class SingleClearEnv(gym.Env):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(4, 84, 84),
        dtype=np.uint8,
    )

    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            id="ALE/Breakout-v5",
            kwargs={"game": "breakout", "mode": 0, "difficulty": 0},
        )
        self.step_count = 0

    @property
    def unwrapped(self) -> "SingleClearEnv":
        return self

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.step_count = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action: int):
        del action
        self.step_count += 1
        return (
            np.zeros((4, 84, 84), dtype=np.uint8),
            864.0,
            True,
            False,
            {"lives": 4},
        )


class EvaluationCompletionPipelineTests(unittest.TestCase):
    def test_contract_v3_provenance_and_runtime_binding_are_versioned(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        contract_path = Path("configs/eval/breakout_contract_v3.json")
        contract = load_evaluation_contract(repository_root / contract_path)
        metadata = {
            "evaluation_contract_path": contract_path.as_posix(),
            "evaluation_contract": contract.to_dict(),
        }
        provenance = _contract_provenance(
            metadata,
            repository_root=repository_root,
        )

        self.assertEqual(provenance["definition_status"], "validated")
        self.assertEqual(
            provenance["contract_id"],
            "breakout-evaluation-v3-frame-skip-1",
        )
        self.assertEqual(provenance["validation_status"], "canonical_contract_v3")
        self.assertNotEqual(
            provenance["contract_sha256"],
            "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a",
        )

        env = make_breakout_env(
            **breakout_environment_kwargs(contract, allow_contract_v3=True)
        )
        try:
            validated, reason = _contract_runtime_binding(
                env,
                metadata,
                provenance,
                evaluation_seeds=contract.concrete_episode_seeds,
                episodes_per_seed=1,
                epsilon=contract.evaluation_epsilon,
            )
        finally:
            env.close()

        self.assertTrue(validated, reason)

        v2_contract_path = Path("configs/eval/breakout_contract_v2.json")
        v2_contract = load_evaluation_contract(repository_root / v2_contract_path)
        v2_metadata = {
            "evaluation_contract_path": v2_contract_path.as_posix(),
            "evaluation_contract": v2_contract.to_dict(),
        }
        v2_provenance = _contract_provenance(
            v2_metadata,
            repository_root=repository_root,
        )
        mismatched_env = make_breakout_env(
            **breakout_environment_kwargs(v2_contract)
        )
        try:
            validated, reason = _contract_runtime_binding(
                mismatched_env,
                metadata,
                provenance,
                evaluation_seeds=contract.concrete_episode_seeds,
                episodes_per_seed=1,
                epsilon=contract.evaluation_epsilon,
            )
        finally:
            mismatched_env.close()
        self.assertFalse(validated)
        self.assertIn("preprocessing semantics", reason)

        mismatched_env = make_breakout_env(
            **breakout_environment_kwargs(contract, allow_contract_v3=True)
        )
        try:
            validated, reason = _contract_runtime_binding(
                mismatched_env,
                v2_metadata,
                v2_provenance,
                evaluation_seeds=v2_contract.concrete_episode_seeds,
                episodes_per_seed=1,
                epsilon=v2_contract.evaluation_epsilon,
            )
        finally:
            mismatched_env.close()
        self.assertFalse(validated)
        self.assertIn("preprocessing semantics", reason)

    def test_verified_clear_preserves_checkpoint_and_protocol_provenance(self) -> None:
        support = CompletionSupport(
            supported=True,
            environment_id="ALE/Breakout-v5",
            game="breakout",
            mode=0,
            difficulty=0,
            ale_py_version="0.12.0",
            rom_sha256="audited-rom-sha256",
        )
        with (
            patch(
                "breakout_rl.evaluation.inspect_breakout_completion_support",
                return_value=support,
            ),
            patch(
                "breakout_rl.evaluation.read_ale_episode_frame",
                side_effect=[100, 104],
            ),
            patch(
                "breakout_rl.evaluation.read_ale_lives",
                side_effect=[4],
            ),
            patch(
                "breakout_rl.evaluation.read_breakout_score",
                return_value=864,
            ),
            patch(
                "breakout_rl.evaluation._capture_source_provenance",
                return_value={
                    "source_commit": "a" * 40,
                    "working_tree_dirty": False,
                    "completion_source_sha256": "d" * 64,
                },
            ),
            patch(
                "breakout_rl.evaluation._contract_provenance",
                return_value={
                    "contract_id": "contract-v2",
                    "contract_path": "configs/eval/breakout_contract_v2.json",
                    "contract_sha256": "b" * 64,
                    "hash_semantics": "exact file bytes",
                    "definition_status": "validated",
                    "validation_status": "canonical_contract_v2",
                },
            ),
            patch(
                "breakout_rl.evaluation._contract_runtime_binding",
                return_value=(True, None),
            ),
        ):
            result = evaluate_policy(
                None,
                episodes=1,
                seeds=[101],
                device="cpu",
                model_id="checkpoint-10000",
                training_metadata={
                    "training_seed": 7,
                    "training_steps": 10000,
                },
                checkpoint_metadata={"sha256": "c" * 64},
                env_factory=SingleClearEnv,
            )

        episode = result.episodes[0]
        self.assertTrue(episode.complete)
        self.assertTrue(episode.cleared)
        self.assertEqual(episode.stop_reason, "terminated")
        self.assertEqual(episode.completion_outcome, "cleared")
        self.assertEqual(episode.clear_agent_step, 1)
        self.assertEqual(episode.clear_emulator_frame, 4)
        self.assertEqual(episode.total_emulator_frames, 4)
        self.assertEqual(episode.clear_score, 864.0)
        self.assertEqual(episode.lives_remaining_at_clear, 4)
        self.assertEqual(
            episode.completion_detection_source,
            BREAKOUT_COMPLETION_SOURCE,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path, _ = write_evaluation_artifacts(
                result,
                Path(temporary_directory) / "clear-evaluation",
            )
            loaded = read_evaluation_results(results_path)

        self.assertEqual(loaded["summary"]["clear_count"], 1)
        self.assertEqual(loaded["summary"]["clear_rate"], 1.0)
        self.assertEqual(loaded["summary"]["clear_time_sample_count"], 1)
        self.assertEqual(loaded["summary"]["p90_clear_steps"], 1.0)
        self.assertEqual(loaded["summary"]["p90_clear_emulator_frame"], 4.0)
        self.assertEqual(loaded["total_agent_steps"], 1)
        self.assertEqual(loaded["total_emulator_frames"], 4)
        self.assertEqual(loaded["summary"]["total_agent_steps"], 1)
        self.assertEqual(loaded["summary"]["total_emulator_frames"], 4)
        self.assertEqual(loaded["evaluation_status"], "completed")
        self.assertEqual(len(loaded["verified_clears"]), 1)
        provenance = loaded["verified_clears"][0]
        self.assertEqual(provenance["checkpoint_id"], "c" * 64)
        self.assertEqual(provenance["training_seed"], 7)
        self.assertEqual(provenance["training_transition_count"], 10000)
        self.assertEqual(provenance["evaluation_seed"], 101)
        self.assertEqual(provenance["contract_id"], "contract-v2")
        self.assertEqual(provenance["contract_sha256"], "b" * 64)
        self.assertEqual(provenance["source_commit"], "a" * 40)
        self.assertEqual(provenance["provenance_status"], "complete")

        v3_result = replace(
            result,
            contract_provenance={
                **dict(result.contract_provenance or {}),
                "contract_id": "breakout-evaluation-v3-frame-skip-1",
                "validation_status": "canonical_contract_v3",
            },
        )
        v3_payload = v3_result.to_dict()
        self.assertEqual(v3_payload["per_episode"][0]["total_agent_steps"], 1)
        self.assertEqual(v3_payload["per_episode"][0]["total_emulator_frames"], 4)
        self.assertEqual(len(v3_payload["verified_clears"]), 1)
        self.assertEqual(
            v3_payload["verified_clears"][0]["contract_id"],
            "breakout-evaluation-v3-frame-skip-1",
        )
        self.assertEqual(
            v3_payload["verified_clears"][0]["contract_validation_status"],
            "canonical_contract_v3",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            v3_path, _ = write_evaluation_artifacts(
                v3_result,
                Path(temporary_directory) / "v3-clear-evaluation",
            )
            v3_loaded = read_evaluation_results(v3_path)
        self.assertEqual(len(v3_loaded["verified_clears"]), 1)

        without_git_source = replace(
            result,
            source_provenance={
                "source_commit": None,
                "working_tree_dirty": None,
                "completion_source_sha256": "d" * 64,
            },
        ).to_dict()
        incomplete_provenance = without_git_source["per_episode"][0][
            "completion_provenance"
        ]
        self.assertEqual(incomplete_provenance["provenance_status"], "incomplete")
        self.assertIn("source_commit", incomplete_provenance["missing_provenance_fields"])
        self.assertIn(
            "source_working_tree_dirty",
            incomplete_provenance["missing_provenance_fields"],
        )
        self.assertEqual(without_git_source["verified_clears"], [])

    def test_contract_runtime_binding_rejects_default_fire_reset_semantics(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        contract_path = Path("configs/eval/breakout_contract_v2.json")
        contract = load_evaluation_contract(repository_root / contract_path)
        metadata = {
            "evaluation_contract_path": contract_path.as_posix(),
            "evaluation_contract": contract.to_dict(),
        }
        contract_provenance = _contract_provenance(
            metadata,
            repository_root=repository_root,
        )
        self.assertEqual(contract_provenance["definition_status"], "validated")

        noncanonical_env = make_breakout_env(fire_reset=False)
        try:
            validated, reason = _contract_runtime_binding(
                noncanonical_env,
                metadata,
                contract_provenance,
                evaluation_seeds=contract.concrete_episode_seeds,
                episodes_per_seed=1,
                epsilon=contract.evaluation_epsilon,
            )
        finally:
            noncanonical_env.close()

        self.assertFalse(validated)
        self.assertIn("FIRE-reset", reason)

        canonical_env = make_breakout_env(
            **breakout_environment_kwargs(contract)
        )
        try:
            validated, reason = _contract_runtime_binding(
                canonical_env,
                metadata,
                contract_provenance,
                evaluation_seeds=contract.concrete_episode_seeds,
                episodes_per_seed=1,
                epsilon=contract.evaluation_epsilon,
            )
        finally:
            canonical_env.close()

        self.assertTrue(validated, reason)

    def test_noncanonical_positive_clear_is_not_listed_as_verified(self) -> None:
        support = CompletionSupport(
            supported=True,
            environment_id="ALE/Breakout-v5",
            game="breakout",
            mode=0,
            difficulty=0,
            ale_py_version="0.12.0",
            rom_sha256="audited-rom-sha256",
        )
        with (
            patch(
                "breakout_rl.evaluation.inspect_breakout_completion_support",
                return_value=support,
            ),
            patch(
                "breakout_rl.evaluation.read_ale_episode_frame",
                side_effect=[100, 104],
            ),
            patch(
                "breakout_rl.evaluation.read_ale_lives",
                side_effect=[4],
            ),
            patch(
                "breakout_rl.evaluation.read_breakout_score",
                return_value=864,
            ),
            patch(
                "breakout_rl.evaluation._capture_source_provenance",
                return_value={
                    "source_commit": "a" * 40,
                    "working_tree_dirty": False,
                    "completion_source_sha256": "d" * 64,
                },
            ),
        ):
            result = evaluate_policy(
                None,
                episodes=1,
                seeds=[101],
                device="cpu",
                model_id="checkpoint-10000",
                training_metadata={
                    "training_seed": 7,
                    "training_steps": 10000,
                },
                checkpoint_metadata={"sha256": "c" * 64},
                env_factory=SingleClearEnv,
            )

        payload = result.to_dict()
        episode = payload["per_episode"][0]
        self.assertTrue(episode["cleared"])
        self.assertEqual(payload["verified_clears"], [])
        self.assertEqual(
            episode["completion_provenance"]["provenance_status"],
            "incomplete",
        )
        self.assertIsNone(payload["contract_provenance"]["contract_id"])
        self.assertIsNone(payload["contract_provenance"]["contract_sha256"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path, _ = write_evaluation_artifacts(
                result,
                Path(temporary_directory) / "noncanonical-clear",
            )
            loaded = read_evaluation_results(results_path)

        self.assertEqual(loaded["summary"]["clear_count"], 1)
        self.assertEqual(loaded["verified_clears"], [])

    def test_seeded_real_game_over_trajectory_is_not_a_clear(self) -> None:
        result = evaluate_policy(
            None,
            episodes=1,
            seeds=[101],
            device="cpu",
            env_factory=lambda: make_breakout_env(fire_reset=True),
            max_steps_per_episode=5000,
        )

        episode = result.episodes[0]
        self.assertTrue(result.completion_detector["supported"])
        self.assertTrue(episode.terminated)
        self.assertFalse(episode.truncated)
        self.assertEqual(episode.completion_outcome, "game_over")
        self.assertFalse(episode.cleared)
        self.assertEqual(episode.episode_return, 1.0)
        self.assertEqual(episode.life_loss_count, 5)
        self.assertEqual(result.to_dict()["summary"]["clear_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
