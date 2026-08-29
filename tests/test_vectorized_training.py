"""Correctness tests for transition-counted vectorized DQN training."""

from __future__ import annotations

import csv
import tempfile
import unittest
import warnings
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


class SwitchingQNetwork(nn.Module):
    """Network whose preferred action changes when the fake optimizer runs."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.preferred_action = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(
            (observations.shape[0], 2),
            dtype=torch.float32,
            device=observations.device,
        )
        values[:, self.preferred_action] = 1.0
        return values + self.anchor * 0.0


class SwitchingVectorEnv:
    """Minimal vector seam for exposing action-selection/update ordering."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.single_action_space = FakeActionSpace()
        self.single_observation_space = FakeObservationSpace()
        self.step_index = 0
        self.action_batches: list[np.ndarray] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        self.step_index = 0
        return np.zeros((self.num_envs, *OBSERVATION_SHAPE), dtype=np.uint8), {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        self.action_batches.append(np.asarray(actions, dtype=np.int64).copy())
        self.step_index += 1
        observations = np.full(
            (self.num_envs, *OBSERVATION_SHAPE),
            self.step_index,
            dtype=np.uint8,
        )
        return (
            observations,
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )


class SwitchingTrainer(VectorizedDQNTrainer):
    """Change the network's preferred action at the scheduled update boundary."""

    def _update_once(self):
        self.online_network.preferred_action = 1  # type: ignore[attr-defined]
        self.optimizer_updates += 1
        return None


class VectorizedTrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_loaded_rng_states_are_normalized_before_restore(self) -> None:
        config = DQNConfig(
            total_steps=3,
            num_envs=3,
            batch_size=3,
            replay_capacity=3,
            learning_starts=3,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = VectorizedDQNTrainer(
                DeterministicVectorEnv(),
                config,
                run_dir=Path(directory) / "rng-restore",
                online_network=CountingQNetwork(),
            )
            trainer._restore_rng_state(  # type: ignore[attr-defined]
                {
                    "torch_cpu": torch.get_rng_state().to("cuda"),
                    "torch_cuda": [torch.cuda.get_rng_state()],
                    "action_rng": trainer.rng.bit_generator.state,
                }
            )
            trainer.metrics.close()

    def test_total_transition_budget_must_be_a_full_vector_step_count(self) -> None:
        config = DQNConfig(
            total_steps=4,
            num_envs=3,
            batch_size=3,
            replay_capacity=4,
            learning_starts=4,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "divisible"):
                VectorizedDQNTrainer(
                    DeterministicVectorEnv(),
                    config,
                    run_dir=Path(directory) / "partial-vector-step",
                    online_network=CountingQNetwork(),
                )

    def test_contract_provenance_is_written_to_run_and_checkpoint_metadata(self) -> None:
        config = DQNConfig(
            total_steps=3,
            num_envs=3,
            batch_size=3,
            replay_capacity=3,
            learning_starts=3,
            checkpoint_interval=3,
            device="cpu",
        )
        contract = {
            "contract_id": "test-contract-v2",
            "contract_path": "configs/eval/breakout_contract_v2.json",
            "contract_sha256": "a" * 64,
            "semantics": {"environment_id": "ALE/Breakout-v5"},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "contract-provenance"
            trainer = VectorizedDQNTrainer(
                DeterministicVectorEnv(),
                config,
                run_dir=run_dir,
                online_network=CountingQNetwork(),
                environment_contract=contract,
            )
            summary = trainer.train()
            payload = torch.load(
                run_dir / "checkpoints/step-00000003.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(summary["environment_contract"], contract)
        self.assertEqual(summary["runtime"]["environment_contract"], contract)
        self.assertEqual(payload["environment_contract"], contract)

    def test_checkpoint_boundaries_are_captured_inside_a_vector_step(self) -> None:
        config = DQNConfig(
            total_steps=6,
            num_envs=3,
            batch_size=3,
            replay_capacity=6,
            learning_starts=6,
            checkpoint_interval=2,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "checkpoint-boundaries"
            trainer = VectorizedDQNTrainer(
                DeterministicVectorEnv(),
                config,
                run_dir=run_dir,
                online_network=CountingQNetwork(),
            )
            trainer.train()

            checkpoint_names = sorted(
                path.name for path in (run_dir / "checkpoints").glob("*.pt")
            )

        self.assertEqual(
            checkpoint_names,
            [
                "step-00000002.pt",
                "step-00000004.pt",
                "step-00000006.pt",
            ],
        )

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

    def test_strict_parity_rule_rejects_batches_that_cross_update_boundaries(self) -> None:
        config = DQNConfig(
            total_steps=8,
            num_envs=8,
            batch_size=4,
            replay_capacity=8,
            learning_starts=4,
            train_frequency=4,
            target_update_interval=100,
            checkpoint_interval=8,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
            strict_action_selection_parity=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "strict action-selection parity"):
                SwitchingTrainer(
                    SwitchingVectorEnv(8),
                    config,
                    run_dir=Path(directory) / "strict-reject",
                    online_network=SwitchingQNetwork(),
                )

    def test_action_selection_metadata_exposes_crossing_batch_lag(self) -> None:
        config = DQNConfig(
            total_steps=8,
            num_envs=8,
            batch_size=4,
            replay_capacity=8,
            learning_starts=4,
            train_frequency=4,
            target_update_interval=100,
            checkpoint_interval=8,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
        )
        env = SwitchingVectorEnv(8)
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trainer = SwitchingTrainer(
                    env,
                    config,
                    run_dir=Path(directory) / "lag",
                    online_network=SwitchingQNetwork(),
                )
                summary = trainer.train()

        self.assertTrue(
            any("behavior-policy lag" in str(warning.message) for warning in caught)
        )
        self.assertEqual(len(env.action_batches), 1)
        np.testing.assert_array_equal(env.action_batches[0], np.zeros(8, dtype=np.int64))
        self.assertEqual(trainer.online_network.preferred_action, 1)
        self.assertFalse(summary["strict_action_selection_parity_satisfied"])
        self.assertIn("pre-update", summary["action_selection_batch_semantics"])

    def test_strict_parity_keeps_later_batch_actions_after_the_update(self) -> None:
        config = DQNConfig(
            total_steps=8,
            num_envs=4,
            batch_size=4,
            replay_capacity=8,
            learning_starts=4,
            train_frequency=4,
            target_update_interval=100,
            checkpoint_interval=8,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
            strict_action_selection_parity=True,
        )
        env = SwitchingVectorEnv(4)
        with tempfile.TemporaryDirectory() as directory:
            trainer = SwitchingTrainer(
                env,
                config,
                run_dir=Path(directory) / "strict",
                online_network=SwitchingQNetwork(),
            )
            summary = trainer.train()

        self.assertEqual(len(env.action_batches), 2)
        np.testing.assert_array_equal(env.action_batches[0], np.zeros(4, dtype=np.int64))
        np.testing.assert_array_equal(env.action_batches[1], np.ones(4, dtype=np.int64))
        self.assertTrue(summary["strict_action_selection_parity_satisfied"])

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
                    {
                        "final_obs": final_observations,
                        "_final_obs": np.array([True, False, False]),
                        "fire_reset_auto": np.array([True, False, False]),
                        "fire_reset_executed_action": np.array([1, 0, 0]),
                        "fire_reset_reason": np.array(
                            ["initial_serve", None, None],
                            dtype=object,
                        ),
                    },
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
            network = CountingQNetwork()
            with torch.no_grad():
                network.head.weight.zero_()
                network.head.bias.copy_(torch.tensor([1.0, -1.0]))
            trainer = VectorizedDQNTrainer(
                env,
                config,
                run_dir=Path(directory) / "same-step",
                online_network=network,
            )
            trainer.train()
            with (Path(directory) / "same-step/metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))

        np.testing.assert_array_equal(
            trainer.replay.next_states[0],
            np.full(OBSERVATION_SHAPE, 77, dtype=np.uint8),
        )
        self.assertEqual(rows[0]["requested_action"], "0")
        self.assertEqual(rows[0]["action"], "1")
        self.assertEqual(rows[0]["action_overridden"], "True")
        self.assertEqual(rows[0]["fire_reset_reason"], "initial_serve")

    def test_vectorized_trainer_switches_to_double_dqn_and_records_metadata(self) -> None:
        config = DQNConfig(
            algorithm="double_dqn",
            total_steps=6,
            num_envs=3,
            batch_size=3,
            replay_capacity=6,
            learning_starts=3,
            train_frequency=3,
            target_update_interval=3,
            checkpoint_interval=6,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "double-vectorized"
            trainer = VectorizedDQNTrainer(
                DeterministicVectorEnv(),
                config,
                run_dir=run_dir,
                online_network=CountingQNetwork(),
            )
            summary = trainer.train()
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

        self.assertEqual(summary["algorithm"], "double_dqn")
        self.assertEqual(summary["num_envs"], 3)
        self.assertEqual(payload["algorithm"], "double_dqn")
        self.assertEqual(payload["num_envs"], 3)


if __name__ == "__main__":
    unittest.main()
