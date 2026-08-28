"""Behavioral tests for the Day 15 FIRE/TimeLimit diagnostic seam."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.day15_diagnostics import (
    EpisodeSpec,
    run_diagnostic_evaluation,
    write_diagnostic_artifacts,
)


class _ActionSpace:
    n = 4

    def seed(self, seed: int) -> None:
        del seed


class _ObservationSpace:
    shape = (4, 84, 84)


class _Spec:
    id = "Test/Breakout-v0"
    kwargs = {"max_num_frames_per_episode": 8}


class _ALE:
    def __init__(self, *, time_limit: bool) -> None:
        self.current_lives = 3
        self.time_limit = time_limit

    def lives(self) -> int:
        return self.current_lives

    def game_truncated(self) -> bool:
        return self.time_limit


class DiagnosticEnv:
    def __init__(self, *, finish_mode: str = "terminated") -> None:
        self.action_space = _ActionSpace()
        self.observation_space = _ObservationSpace()
        self.spec = _Spec()
        self.ale = _ALE(time_limit=finish_mode == "time_limit")
        self.finish_mode = finish_mode
        self.step_count = 0
        self.actions: list[int] = []

    @property
    def unwrapped(self) -> "DiagnosticEnv":
        return self

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict]:
        del seed
        self.step_count = 0
        self.ale.current_lives = 3
        self.ale.time_limit = False
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.actions.append(action)
        self.step_count += 1
        if self.step_count == 2:
            self.ale.current_lives = 2
        terminated = self.finish_mode == "terminated" and self.step_count >= 4
        truncated = self.finish_mode == "time_limit" and self.step_count >= 2
        self.ale.time_limit = truncated
        value = 0 if self.step_count == 1 else self.step_count
        observation = np.full((4, 84, 84), value, dtype=np.uint8)
        return observation, 1.0 if self.step_count == 3 else 0.0, terminated, truncated, {}

    def close(self) -> None:
        pass


class ConstantQNetwork(nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]],
            device=observations.device,
        ).expand(observations.shape[0], -1)


class Day15DiagnosticTests(unittest.TestCase):
    def test_fire_assist_records_initial_and_life_loss_serves(self) -> None:
        env = DiagnosticEnv()
        payload = run_diagnostic_evaluation(
            ConstantQNetwork(),
            env_factory=lambda: env,
            device="cpu",
            episode_specs=[EpisodeSpec(101, 1, 101)],
            mode="fire_assist",
            trace_seeds=[101],
        )

        row = payload["per_episode"][0]
        self.assertEqual(env.actions, [1, 3, 1, 3])
        self.assertEqual(row["auto_fire_count"], 2)
        self.assertEqual(row["life_loss_count"], 1)
        self.assertEqual(row["life_loss_to_fire_latencies"], [1])
        self.assertEqual(row["first_fire_after_life_loss_steps"], [3])
        self.assertEqual(row["stop_reason"], "terminated")
        self.assertEqual(
            payload["trace"][0]["steps"][0]["q_values"],
            [0.0, 1.0, 2.0, 3.0],
        )

    def test_timeout_is_time_limit_and_records_compact_observation_signal(self) -> None:
        payload = run_diagnostic_evaluation(
            ConstantQNetwork(),
            env_factory=lambda: DiagnosticEnv(finish_mode="time_limit"),
            device="cpu",
            episode_specs=[EpisodeSpec(101, 1, 101)],
            mode="v1",
        )

        row = payload["per_episode"][0]
        self.assertTrue(row["truncated"])
        self.assertTrue(row["time_limit"])
        self.assertEqual(row["stop_reason"], "time_limit")
        self.assertGreater(row["max_consecutive_unchanged_observation"], 0)
        self.assertEqual(payload["summary"]["finished_episode_count"], 1)
        self.assertEqual(payload["summary"]["terminated_count"], 0)
        self.assertEqual(payload["summary"]["truncated_count"], 1)
        self.assertEqual(payload["summary"]["time_limit_truncated_count"], 1)

    def test_diagnostic_artifacts_write_time_limit_schema_and_trace(self) -> None:
        payload = run_diagnostic_evaluation(
            ConstantQNetwork(),
            env_factory=lambda: DiagnosticEnv(finish_mode="time_limit"),
            device="cpu",
            episode_specs=[EpisodeSpec(101, 1, 101)],
            mode="v1",
            trace_seeds=[101],
        )

        with tempfile.TemporaryDirectory() as directory:
            results_path, episodes_path, trace_path = write_diagnostic_artifacts(
                payload,
                Path(directory) / "diagnostic",
            )
            written = json.loads(results_path.read_text(encoding="utf-8"))
            csv_header = episodes_path.read_text(encoding="utf-8").splitlines()[0]
            trace_size = trace_path.stat().st_size

        self.assertEqual(written["summary"]["time_limit_truncated_count"], 1)
        self.assertIn("time_limit", csv_header)
        self.assertGreater(trace_size, 0)


if __name__ == "__main__":
    unittest.main()
