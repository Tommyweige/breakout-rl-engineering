"""Tests for Issue #9 reward shaping and raw-score separation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.evaluation import evaluate_policy, write_evaluation_artifacts
from breakout_rl.reward_shaping_experiment import (
    compare_evaluation_payloads,
    score_statistics,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import DQNTrainer
from breakout_rl.training.reward_shaping import shape_training_reward
from breakout_rl.training.vectorized import VectorizedDQNTrainer


OBSERVATION_SHAPE = (4, 84, 84)


class _ActionSpace:
    n = 2


class _ObservationSpace:
    shape = OBSERVATION_SHAPE


class _TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(int(np.prod(OBSERVATION_SHAPE)), 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(observations))


class _LifeLossEnv:
    action_space = _ActionSpace()
    observation_space = _ObservationSpace()

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        del seed
        self.step_count = 0
        return np.zeros(OBSERVATION_SHAPE, dtype=np.uint8), {}

    def step(self, action: int):
        del action
        self.step_count += 1
        rewards = {1: 0.0, 2: 0.0, 3: 2.0, 4: 0.0}
        return (
            np.full(OBSERVATION_SHAPE, self.step_count, dtype=np.uint8),
            rewards[self.step_count],
            self.step_count == 4,
            False,
            {"fire_reset_life_loss": self.step_count in {2, 3}},
        )


class _LifeLossVectorEnv:
    num_envs = 2
    single_action_space = _ActionSpace()
    single_observation_space = _ObservationSpace()

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.step_count = 0
        return np.zeros((2, *OBSERVATION_SHAPE), dtype=np.uint8), {}

    def step(self, actions: np.ndarray):
        del actions
        self.step_count += 1
        return (
            np.full((2, *OBSERVATION_SHAPE), self.step_count, dtype=np.uint8),
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([True, True]),
            np.array([False, False]),
            {
                "fire_reset_life_loss": np.array([True, False]),
                "fire_reset_auto": np.array([False, False]),
                "fire_reset_executed_action": np.array([0, 0]),
            },
        )


class _EvaluationEnv:
    action_space = _ActionSpace()
    observation_space = _ObservationSpace()

    def reset(self, *, seed: int | None = None):
        del seed
        return np.zeros(OBSERVATION_SHAPE, dtype=np.uint8), {}

    def step(self, action: int):
        del action
        return (
            np.zeros(OBSERVATION_SHAPE, dtype=np.uint8),
            2.0,
            True,
            False,
            {"fire_reset_life_loss": True},
        )

    def close(self) -> None:
        pass


class RewardShapingTests(unittest.TestCase):
    def test_score_statistics_and_paired_comparison_are_raw_score_based(self) -> None:
        self.assertEqual(score_statistics([0.0, 1.0, 2.0])["p10"], 0.2)
        baseline = {
            "model_id": "baseline",
            "per_episode": [
                {
                    "evaluation_seed": 1,
                    "episode_index": 1,
                    "episode_seed": 1,
                    "episode_return": 2.0,
                    "episode_length": 10,
                    "terminated": True,
                    "truncated": False,
                    "life_loss_count": 1,
                },
                {
                    "evaluation_seed": 2,
                    "episode_index": 1,
                    "episode_seed": 2,
                    "episode_return": 0.0,
                    "episode_length": 8,
                    "terminated": True,
                    "truncated": False,
                    "life_loss_count": 1,
                },
            ],
        }
        shaped = {
            "model_id": "shaped",
            "per_episode": [
                {**baseline["per_episode"][0], "episode_return": 3.0},
                {**baseline["per_episode"][1], "episode_return": 0.0},
            ],
        }
        comparison = compare_evaluation_payloads(baseline, shaped)
        self.assertEqual(comparison["evaluation_seed_count"], 2)
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["ties"], 1)
        self.assertEqual(comparison["losses"], 0)
        self.assertEqual(comparison["shaped"]["mean_raw_score"], 1.5)

    def test_shape_training_reward_clips_game_reward_before_penalty(self) -> None:
        self.assertEqual(
            shape_training_reward(
                0.0,
                reward_clip=True,
                life_loss=False,
                life_loss_penalty=-1.0,
            ),
            0.0,
        )
        self.assertEqual(
            shape_training_reward(
                1.0,
                reward_clip=True,
                life_loss=False,
                life_loss_penalty=-1.0,
            ),
            1.0,
        )
        self.assertEqual(
            shape_training_reward(
                0.0,
                reward_clip=True,
                life_loss=True,
                life_loss_penalty=-1.0,
            ),
            -1.0,
        )
        self.assertEqual(
            shape_training_reward(
                1.0,
                reward_clip=True,
                life_loss=True,
                life_loss_penalty=-1.0,
            ),
            0.0,
        )
        self.assertEqual(
            shape_training_reward(
                2.5,
                reward_clip=False,
                life_loss=True,
                life_loss_penalty=-1.0,
            ),
            1.5,
        )

    def test_invalid_penalties_fail_closed(self) -> None:
        for penalty in (1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(penalty=penalty):
                with self.assertRaises((TypeError, ValueError)):
                    DQNConfig(life_loss_penalty=penalty)
                with self.assertRaises((TypeError, ValueError)):
                    shape_training_reward(
                        0.0,
                        reward_clip=True,
                        life_loss=True,
                        life_loss_penalty=penalty,
                    )

    def test_single_trainer_stores_shaped_reward_but_reports_raw_episode_score(self) -> None:
        config = DQNConfig(
            total_steps=4,
            batch_size=4,
            replay_capacity=8,
            learning_starts=8,
            checkpoint_interval=4,
            device="cpu",
            life_loss_penalty=-1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "single"
            trainer = DQNTrainer(
                _LifeLossEnv(),
                config,
                run_dir=run_dir,
                online_network=_TinyNetwork(),
            )
            summary = trainer.train()
            with (run_dir / "metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))

        np.testing.assert_allclose(
            trainer.replay.rewards[: len(trainer.replay)],
            [0.0, -1.0, 0.0, 0.0],
        )
        self.assertEqual(float(rows[-1]["raw_episode_return"]), 2.0)
        self.assertEqual(float(rows[-1]["training_episode_return"]), -1.0)
        self.assertEqual(int(rows[-1]["episode_life_loss_count"]), 2)
        self.assertEqual(int(rows[-1]["life_loss_count"]), 2)
        self.assertEqual(float(rows[-1]["life_loss_penalty_total"]), -2.0)
        self.assertEqual(summary["life_loss_count"], 2)
        self.assertEqual(summary["life_loss_penalty_total"], -2.0)

    def test_vectorized_trainer_applies_life_loss_per_environment(self) -> None:
        config = DQNConfig(
            total_steps=2,
            num_envs=2,
            batch_size=2,
            replay_capacity=4,
            learning_starts=4,
            checkpoint_interval=2,
            device="cpu",
            life_loss_penalty=-1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "vectorized"
            trainer = VectorizedDQNTrainer(
                _LifeLossVectorEnv(),
                config,
                run_dir=run_dir,
                online_network=_TinyNetwork(),
            )
            summary = trainer.train()
            with (run_dir / "metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))

        np.testing.assert_allclose(
            trainer.replay.rewards[: len(trainer.replay)],
            [-1.0, 1.0],
        )
        self.assertEqual(summary["life_loss_count"], 1)
        self.assertEqual(summary["life_loss_penalty_total"], -1.0)
        self.assertEqual(int(rows[0]["life_loss"] == "True"), 1)
        self.assertEqual(float(rows[0]["raw_reward"]), 0.0)
        self.assertEqual(float(rows[1]["raw_reward"]), 2.0)
        self.assertEqual(float(rows[1]["training_reward"]), 1.0)

    def test_evaluation_keeps_raw_score_when_life_loss_is_reported(self) -> None:
        result = evaluate_policy(
            None,
            episodes=1,
            seeds=[7],
            device="cpu",
            env_factory=_EvaluationEnv,
        )
        self.assertEqual(result.episodes[0].episode_return, 2.0)
        self.assertEqual(result.episodes[0].life_loss_count, 1)
        self.assertEqual(result.life_loss_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            results_path, episodes_path = write_evaluation_artifacts(
                result,
                Path(directory) / "evaluation",
            )
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            with episodes_path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(payload["per_episode"][0]["episode_return"], 2.0)
        self.assertEqual(payload["summary"]["life_loss_count"], 1)
        self.assertEqual(int(row["life_loss_count"]), 1)


if __name__ == "__main__":
    unittest.main()
