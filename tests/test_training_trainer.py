"""Short deterministic trainer checks at the environment boundary."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import DQNTrainer, TrainingStepSnapshot


class FakeActionSpace:
    n = 2


class FakeObservationSpace:
    shape = (4, 84, 84)


class ShortEpisodeEnv:
    """Small environment-shaped seam that emits real uint8 observations."""

    action_space = FakeActionSpace()
    observation_space = FakeObservationSpace()

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
        if seed is not None:
            self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, int]]:
        del action
        self.steps += 1
        observation = np.full(
            self.observation_space.shape,
            self.steps % 4,
            dtype=np.uint8,
        )
        terminated = self.steps % 5 == 0
        return observation, 2.0 if self.steps == 3 else 0.0, terminated, False, {}

    def render(self) -> np.ndarray:
        return np.full((210, 160, 3), self.steps % 256, dtype=np.uint8)


class TinyImageQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(4 * 84 * 84, 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(observations))


class DQNTrainerTests(unittest.TestCase):
    def test_step_callback_receives_runtime_snapshot_and_rendered_frame(self) -> None:
        snapshots: list[TrainingStepSnapshot] = []
        frames: list[np.ndarray] = []

        def on_step(snapshot: TrainingStepSnapshot, frame: np.ndarray | None) -> None:
            snapshots.append(snapshot)
            self.assertIsNotNone(frame)
            frames.append(frame)  # type: ignore[arg-type]

        config = DQNConfig(
            total_steps=4,
            batch_size=4,
            replay_capacity=8,
            learning_starts=8,
            train_frequency=2,
            target_update_interval=4,
            checkpoint_interval=4,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = DQNTrainer(
                ShortEpisodeEnv(),
                config,
                run_dir=Path(temporary_directory) / "callback-smoke",
                online_network=TinyImageQNetwork(),
                on_step=on_step,
            )
            trainer.train()

        self.assertEqual(len(snapshots), 4)
        self.assertEqual(len(frames), 4)
        self.assertEqual(snapshots[0].global_step, 1)
        self.assertFalse(snapshots[0].warmup_complete)
        self.assertFalse(snapshots[0].optimizer_updated)
        self.assertEqual(frames[0].shape, (210, 160, 3))

    def test_same_seed_reproduces_cpu_action_and_replay_sequence(self) -> None:
        config = DQNConfig(
            total_steps=12,
            batch_size=4,
            replay_capacity=12,
            learning_starts=4,
            train_frequency=2,
            target_update_interval=4,
            checkpoint_interval=12,
            device="cpu",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                config,
                run_dir=Path(temporary_directory) / "first",
            )
            first_trainer.train()
            second_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                config,
                run_dir=Path(temporary_directory) / "second",
            )
            second_trainer.train()

            self.assertGreater(first_trainer.optimizer_updates, 0)
            self.assertGreater(first_trainer.target_sync_count, 1)
            np.testing.assert_array_equal(
                first_trainer.replay.actions,
                second_trainer.replay.actions,
            )

            def action_sources(path: Path) -> list[str]:
                with path.open(newline="", encoding="utf-8") as stream:
                    return [row["action_source"] for row in csv.DictReader(stream)]

            first_sources = action_sources(
                Path(temporary_directory) / "first/metrics.csv"
            )
            second_sources = action_sources(
                Path(temporary_directory) / "second/metrics.csv"
            )
            self.assertEqual(first_sources, second_sources)

            def losses(path: Path) -> list[str]:
                with path.open(newline="", encoding="utf-8") as stream:
                    return [
                        row["loss"]
                        for row in csv.DictReader(stream)
                        if row["loss"]
                    ]

            self.assertEqual(
                losses(Path(temporary_directory) / "first/metrics.csv"),
                losses(Path(temporary_directory) / "second/metrics.csv"),
            )

    def test_reward_clipping_keeps_raw_metric_separate_from_replay_reward(self) -> None:
        common_values = {
            "total_steps": 4,
            "batch_size": 4,
            "replay_capacity": 8,
            "learning_starts": 8,
            "train_frequency": 2,
            "target_update_interval": 4,
            "checkpoint_interval": 4,
            "device": "cpu",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            clipped_run = Path(temporary_directory) / "clipped"
            clipped_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                DQNConfig(reward_clip=True, **common_values),
                run_dir=clipped_run,
                online_network=TinyImageQNetwork(),
            )
            clipped_trainer.train()
            self.assertIn(1.0, clipped_trainer.replay.rewards.tolist())

            with (clipped_run / "metrics.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as stream:
                clipped_rows = list(csv.DictReader(stream))
            reward_row = clipped_rows[2]
            self.assertEqual(float(reward_row["raw_reward"]), 2.0)
            self.assertEqual(float(reward_row["training_reward"]), 1.0)

            raw_run = Path(temporary_directory) / "raw"
            raw_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                DQNConfig(reward_clip=False, **common_values),
                run_dir=raw_run,
                online_network=TinyImageQNetwork(),
            )
            raw_trainer.train()
            self.assertIn(2.0, raw_trainer.replay.rewards.tolist())

    def test_short_run_writes_metrics_summary_and_checkpoint(self) -> None:
        config = DQNConfig(
            total_steps=16,
            batch_size=4,
            replay_capacity=16,
            learning_starts=4,
            train_frequency=2,
            target_update_interval=4,
            checkpoint_interval=8,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "trainer-smoke"
            env = ShortEpisodeEnv()
            trainer = DQNTrainer(
                env,
                config,
                run_dir=run_dir,
                online_network=TinyImageQNetwork(),
            )
            summary = trainer.train()

            self.assertEqual(summary["total_steps"], 16)
            self.assertGreater(summary["optimizer_updates"], 0)
            self.assertGreater(summary["target_sync_count"], 1)
            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue(list((run_dir / "checkpoints").glob("*.pt")))

            with (run_dir / "metrics.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            losses = [float(row["loss"]) for row in rows if row["loss"]]
            self.assertTrue(losses)
            self.assertTrue(all(np.isfinite(loss) for loss in losses))

            saved_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_summary["optimizer_updates"], summary["optimizer_updates"])

    def test_resume_restores_model_state_and_rewarms_replay(self) -> None:
        base_values = {
            "batch_size": 4,
            "replay_capacity": 16,
            "learning_starts": 4,
            "train_frequency": 2,
            "target_update_interval": 4,
            "checkpoint_interval": 8,
            "device": "cpu",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "resume-smoke"
            first_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                DQNConfig(total_steps=8, **base_values),
                run_dir=run_dir,
                online_network=TinyImageQNetwork(),
            )
            first_summary = first_trainer.train()
            checkpoint = Path(first_summary["last_checkpoint"])

            resumed_trainer = DQNTrainer(
                ShortEpisodeEnv(),
                DQNConfig(total_steps=16, **base_values),
                run_dir=run_dir,
                online_network=TinyImageQNetwork(),
                resume_from=checkpoint,
            )
            resumed_summary = resumed_trainer.train()

            self.assertEqual(resumed_summary["total_steps"], 16)
            self.assertGreater(
                resumed_summary["optimizer_updates"],
                first_summary["optimizer_updates"],
            )
            self.assertEqual(resumed_summary["replay_size"], 8)


if __name__ == "__main__":
    unittest.main()
