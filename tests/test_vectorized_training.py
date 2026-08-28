"""Correctness tests for transition-counted vectorized DQN training."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.training.config import DQNConfig
from breakout_rl.training.vectorized import (
    VectorizedDQNTrainer,
    crossed_transition_boundaries,
)


OBSERVATION_SHAPE = (4, 84, 84)


class FakeActionSpace:
    n = 2


class FakeObservationSpace:
    shape = OBSERVATION_SHAPE


class DeterministicVectorEnv:
    """A vector-environment-shaped seam with independent episode lifecycles."""

    num_envs = 3
    single_action_space = FakeActionSpace()
    single_observation_space = FakeObservationSpace()

    def __init__(self) -> None:
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.reset_history: list[np.ndarray] = []

    def _observations(self) -> np.ndarray:
        observations = np.zeros(
            (self.num_envs, *OBSERVATION_SHAPE),
            dtype=np.uint8,
        )
        for index, step in enumerate(self.steps):
            observations[index].fill(int(step))
        return observations

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed
        if options is None or "reset_mask" not in options:
            self.steps.fill(0)
            mask = np.ones(self.num_envs, dtype=np.bool_)
        else:
            mask = np.asarray(options["reset_mask"], dtype=np.bool_)
            self.steps[mask] = 0
        self.reset_history.append(mask.copy())
        return self._observations(), {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        del actions
        self.steps += 1
        terminated = self.steps == np.array([2, 3, 99], dtype=np.int64)
        truncated = self.steps == np.array([99, 99, 4], dtype=np.int64)
        observations = self._observations()
        return (
            observations,
            np.array([1.0, 0.0, 2.0], dtype=np.float32),
            terminated,
            truncated,
            {},
        )


class CountingQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(int(np.prod(OBSERVATION_SHAPE)), 2)
        self.forward_batch_sizes: list[int] = []

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.forward_batch_sizes.append(int(observations.shape[0]))
        return self.head(self.flatten(observations))


class VectorizedTrainingTests(unittest.TestCase):
    def test_batched_forward_matches_independent_forward_actions(self) -> None:
        network = CountingQNetwork().eval()
        observations = torch.rand(3, *OBSERVATION_SHAPE)
        with torch.no_grad():
            independent = torch.cat(
                [network(observations[index : index + 1]) for index in range(3)],
                dim=0,
            )
            batched = network(observations)

        torch.testing.assert_close(batched, independent)
        torch.testing.assert_close(batched.argmax(dim=1), independent.argmax(dim=1))

    def test_scheduler_reports_every_boundary_crossed_by_a_vector_step(self) -> None:
        self.assertEqual(
            crossed_transition_boundaries(0, 8, 4),
            (4, 8),
        )
        self.assertEqual(
            crossed_transition_boundaries(8, 15, 4),
            (12,),
        )

    def test_trainer_counts_transitions_and_keeps_per_env_episode_state(self) -> None:
        env = DeterministicVectorEnv()
        network = CountingQNetwork()
        config = DQNConfig(
            total_steps=12,
            num_envs=3,
            batch_size=3,
            replay_capacity=20,
            learning_starts=3,
            train_frequency=2,
            target_update_interval=4,
            checkpoint_interval=12,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=12,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as directory:
            trainer = VectorizedDQNTrainer(
                env,
                config,
                run_dir=Path(directory) / "vectorized",
                online_network=network,
            )
            summary = trainer.train()

            with (Path(directory) / "vectorized/metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(summary["total_transitions"], 12)
        self.assertEqual(summary["vector_iterations"], 4)
        self.assertEqual(summary["physical_environment_steps"], 12)
        self.assertEqual(summary["optimizer_updates"], 5)
        self.assertEqual(summary["target_sync_count"], 4)
        self.assertEqual(summary["per_environment_episode_counts"], [2, 1, 1])
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            [int(row["global_step"]) for row in rows],
            list(range(1, 13)),
        )
        self.assertEqual(
            [int(row["vector_iteration"]) for row in rows],
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        )
        self.assertEqual(network.forward_batch_sizes[:4], [3, 3, 3, 3])
        self.assertTrue(all(size == 3 for size in network.forward_batch_sizes))
        self.assertEqual(env.reset_history[0].tolist(), [True, True, True])
        self.assertEqual(env.reset_history[1].tolist(), [True, False, False])
        self.assertEqual(env.reset_history[2].tolist(), [False, True, False])
        self.assertEqual(env.reset_history[3].tolist(), [True, False, True])

        # Env 0 ends its second episode at the second vector iteration. Its
        # replay next_state must be the terminal observation (value 2), not the
        # reset observation (value 0) used for the following episode.
        np.testing.assert_array_equal(
            trainer.replay.next_states[3],
            np.full(OBSERVATION_SHAPE, 2, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            trainer.replay.states[6],
            np.zeros(OBSERVATION_SHAPE, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            trainer.replay.truncated[[2, 5, 8, 11]],
            np.array([False, False, False, True]),
        )

    def test_terminal_final_observation_from_same_step_info_wins(self) -> None:
        class SameStepFinalObservationEnv(DeterministicVectorEnv):
            def step(self, actions: np.ndarray):
                observations, rewards, terminated, truncated, _ = super().step(actions)
                final_observations = np.empty(self.num_envs, dtype=object)
                final_observations[:] = None
                final_observations[0] = np.full(OBSERVATION_SHAPE, 77, dtype=np.uint8)
                observations[0].fill(0)
                return (
                    observations,
                    rewards,
                    np.array([True, False, False]),
                    truncated,
                    {"final_obs": final_observations, "_final_obs": np.array([True, False, False])},
                )

        env = SameStepFinalObservationEnv()
        config = DQNConfig(
            total_steps=3,
            num_envs=3,
            batch_size=3,
            replay_capacity=4,
            learning_starts=4,
            checkpoint_interval=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = VectorizedDQNTrainer(
                env,
                config,
                run_dir=Path(directory) / "same-step",
                online_network=CountingQNetwork(),
            )
            trainer.train()

        np.testing.assert_array_equal(
            trainer.replay.next_states[0],
            np.full(OBSERVATION_SHAPE, 77, dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
