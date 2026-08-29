"""Behavioral tests for the Day 15 frozen-policy evaluation contract."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from breakout_rl.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationConfig,
    evaluate_policy,
    load_day14_provenance,
    load_dqn_checkpoint,
    load_evaluation_config,
    summarize_returns,
    write_evaluation_artifacts,
)
from breakout_rl.models import DQNNetwork
from breakout_rl.evaluation_artifacts import validate_episode_rows
from breakout_rl.evaluation_artifacts import read_evaluation_results
from breakout_rl.training.config import DQNConfig
from scripts.evaluation.evaluate_dqn import (
    DQN_REFERENCE_EVALUATION_ID,
    DQN_REFERENCE_OUTPUT_DIR,
    FORMAL_DQN_EVALUATION_ID,
    FORMAL_DQN_OUTPUT_DIR,
    _output_destination,
)
from scripts.analysis.generate_dqn_milestone_report import build_report
from scripts.visualization.visualize_day15_evaluation import render_evaluation_comparison


class _ActionSpace:
    n = 4

    def __init__(self) -> None:
        self.seed_values: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_values.append(seed)


class _ObservationSpace:
    shape = (4, 84, 84)


class _Spec:
    id = "Test/Breakout-v0"


class ScriptedEvaluationEnv:
    """Small deterministic env that preserves the real observation shape."""

    def __init__(
        self,
        *,
        return_by_seed: dict[int, float] | None = None,
        finish_mode: str = "terminated",
    ) -> None:
        self.action_space = _ActionSpace()
        self.observation_space = _ObservationSpace()
        self.spec = _Spec()
        self.return_by_seed = return_by_seed or {}
        self.finish_mode = finish_mode
        self.reset_seeds: list[int] = []
        self.actions: list[int] = []
        self.current_seed = 0
        self.step_count = 0
        self.closed = False

    @property
    def unwrapped(self) -> "ScriptedEvaluationEnv":
        return self

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is None:
            raise AssertionError("evaluation must explicitly seed every reset")
        self.current_seed = seed
        self.reset_seeds.append(seed)
        self.step_count = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.actions.append(action)
        self.step_count += 1
        if self.current_seed in self.return_by_seed:
            reward = self.return_by_seed[self.current_seed] if self.step_count == 1 else 0.0
        else:
            reward = 0.5 if self.step_count == 1 else 2.0
        finished = self.step_count >= 2
        terminated = finished and self.finish_mode == "terminated"
        truncated = finished and self.finish_mode == "truncated"
        return (
            np.zeros((4, 84, 84), dtype=np.uint8),
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self) -> None:
        self.closed = True


class FireOverrideEvaluationEnv(ScriptedEvaluationEnv):
    """Scripted seam exposing the same provenance keys as the FIRE wrapper."""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = super().step(action)
        if self.step_count == 1:
            info = {
                "fire_reset_auto": True,
                "fire_reset_requested_action": action,
                "fire_reset_executed_action": 1,
                "fire_reset_reason": "initial_serve",
            }
        else:
            info = {
                "fire_reset_auto": False,
                "fire_reset_requested_action": action,
                "fire_reset_executed_action": action,
                "fire_reset_reason": None,
            }
        return observation, reward, terminated, truncated, info


class ConstantQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_values = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))
        self.grad_enabled: list[bool] = []

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.q_values.expand(observations.shape[0], -1)


class EvaluationTests(unittest.TestCase):
    def test_dqn_evaluation_is_greedy_no_grad_and_does_not_change_parameters(self) -> None:
        env = ScriptedEvaluationEnv()
        model = ConstantQNetwork()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        result = evaluate_policy(
            model,
            episodes=1,
            seeds=[11],
            device="cpu",
            epsilon=0.0,
            env_factory=lambda: env,
        )

        self.assertEqual(result.policy_type, "dqn")
        self.assertFalse(model.training)
        self.assertTrue(model.grad_enabled)
        self.assertTrue(all(enabled is False for enabled in model.grad_enabled))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        self.assertEqual(env.actions, [3, 3])
        self.assertEqual(result.episodes[0].episode_return, 2.5)
        self.assertTrue(result.episodes[0].complete)
        self.assertEqual(result.action_distribution["LEFT"], 2)
        self.assertTrue(env.closed)

    def test_evaluation_preserves_requested_and_wrapper_resolved_actions(self) -> None:
        env = FireOverrideEvaluationEnv()
        model = ConstantQNetwork()
        with torch.no_grad():
            model.q_values.copy_(torch.tensor([0.0, 1.0, 3.0, 2.0]))

        result = evaluate_policy(
            model,
            episodes=1,
            seeds=[11],
            device="cpu",
            epsilon=0.0,
            env_factory=lambda: env,
        )

        episode = result.episodes[0]
        self.assertEqual(episode.requested_action_distribution["RIGHT"], 2)
        self.assertEqual(episode.executed_action_distribution["FIRE"], 1)
        self.assertEqual(episode.executed_action_distribution["RIGHT"], 1)
        self.assertEqual(result.action_distribution["FIRE"], 1)
        self.assertEqual(result.requested_action_distribution["RIGHT"], 2)
        self.assertEqual(result.auto_fire_count, 1)
        self.assertEqual(result.auto_fire_reason_counts, {"initial_serve": 1})

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path, episodes_path = write_evaluation_artifacts(
                result,
                Path(temporary_directory) / "evaluation",
            )
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            with episodes_path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            loaded_v2 = read_evaluation_results(results_path)

        self.assertEqual(payload["schema_version"], EVALUATION_SCHEMA_VERSION)
        self.assertEqual(loaded_v2["schema_version"], EVALUATION_SCHEMA_VERSION)
        self.assertEqual(row["schema_version"], str(EVALUATION_SCHEMA_VERSION))
        self.assertEqual(
            row["action_distribution_semantics"],
            "executed/wrapper-resolved action",
        )
        self.assertEqual(
            payload["action_distribution_semantics"],
            "executed/wrapper-resolved action",
        )
        self.assertEqual(json.loads(row["requested_action_distribution_json"])["RIGHT"], 2)
        self.assertEqual(json.loads(row["executed_action_distribution_json"])["FIRE"], 1)
        self.assertEqual(int(row["auto_fire_count"]), 1)
        self.assertEqual(
            json.loads(row["auto_fire_reason_counts_json"]),
            {"initial_serve": 1},
        )

    def test_v1_evaluation_results_remain_readable(self) -> None:
        payload = {
            "schema_version": 1,
            "per_episode": [
                {
                    "evaluation_seed": 101,
                    "episode_index": 1,
                    "episode_seed": 101,
                    "episode_return": 2.0,
                    "episode_length": 4,
                    "terminated": True,
                    "truncated": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "v1-results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_evaluation_results(path)

        self.assertEqual(loaded["schema_version"], 1)

    def test_v2_evaluation_results_require_provenance_fields(self) -> None:
        payload = {
            "schema_version": 2,
            "per_episode": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid-v2-results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema v2 is missing"):
                read_evaluation_results(path)

    def test_seed_groups_episode_count_and_raw_reward_are_preserved(self) -> None:
        env = ScriptedEvaluationEnv()

        result = evaluate_policy(
            None,
            episodes=2,
            seeds=[101, 202],
            device="cpu",
            env_factory=lambda: env,
        )

        self.assertEqual(env.reset_seeds, [101, 102, 202, 203])
        self.assertEqual(result.evaluation_seeds, (101, 202))
        self.assertEqual(result.episodes_per_seed, 2)
        self.assertEqual(len(result.episodes), 4)
        self.assertEqual(
            [episode.episode_seed for episode in result.episodes],
            [101, 102, 202, 203],
        )
        self.assertEqual(result.to_dict()["per_episode_returns"], [2.5] * 4)
        self.assertEqual(sum(result.action_distribution.values()), 8)

    def test_truncated_episodes_are_not_reported_as_terminated(self) -> None:
        env = ScriptedEvaluationEnv(finish_mode="truncated")

        result = evaluate_policy(
            None,
            episodes=1,
            seeds=[101],
            device="cpu",
            env_factory=lambda: env,
        )

        episode = result.episodes[0]
        self.assertFalse(episode.terminated)
        self.assertTrue(episode.truncated)
        self.assertTrue(episode.complete)

    def test_evaluator_limit_fails_without_emitting_a_partial_episode(self) -> None:
        env = ScriptedEvaluationEnv()

        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            evaluate_policy(
                None,
                episodes=1,
                seeds=[101],
                device="cpu",
                env_factory=lambda: env,
                max_steps_per_episode=1,
            )

        self.assertTrue(env.closed)

    def test_random_policy_only_emits_legal_actions(self) -> None:
        env = ScriptedEvaluationEnv()

        result = evaluate_policy(
            None,
            episodes=3,
            seeds=[101],
            device="cpu",
            env_factory=lambda: env,
        )

        self.assertEqual(result.policy_type, "random")
        self.assertTrue(all(0 <= action < 4 for action in env.actions))
        self.assertEqual(sum(result.action_distribution.values()), len(env.actions))

    def test_aggregation_uses_population_std_and_keeps_distribution(self) -> None:
        summary = summarize_returns([1.0, 2.0, 3.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean_return"], 2.0)
        self.assertEqual(summary["median_return"], 2.0)
        self.assertAlmostEqual(summary["std_return"], math.sqrt(2.0 / 3.0))
        self.assertEqual(summary["min_return"], 1.0)
        self.assertEqual(summary["max_return"], 3.0)

    def test_cuda_request_fails_without_falling_back_to_cpu(self) -> None:
        with patch(
            "breakout_rl.training.dqn_trainer.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to fall back to CPU"):
                evaluate_policy(
                    None,
                    episodes=1,
                    seeds=[101],
                    device="cuda",
                    env_factory=ScriptedEvaluationEnv,
                )

    def test_config_and_json_csv_artifacts_round_trip(self) -> None:
        config = EvaluationConfig.from_mapping(
            {
                "seeds": [101, 202, 303],
                "episodes_per_seed": 5,
                "epsilon": 0.0,
            }
        )
        self.assertEqual(config.total_episodes, 15)

        result = evaluate_policy(
            None,
            episodes=config.episodes_per_seed,
            seeds=config.seeds,
            device="cpu",
            epsilon=config.epsilon,
            env_factory=ScriptedEvaluationEnv,
            evaluation_id="test-random",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path, episodes_path = write_evaluation_artifacts(
                result,
                Path(temporary_directory) / "evaluation",
            )
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            with episodes_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(payload["evaluation_seeds"], [101, 202, 303])
        self.assertEqual(payload["total_episodes"], 15)
        self.assertEqual(len(payload["per_episode"]), 15)
        self.assertEqual(len(rows), 15)
        self.assertIn("action_distribution_json", rows[0])

    def test_evaluation_config_file_loads_fixed_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "eval.json"
            path.write_text(
                json.dumps(
                    {
                        "seeds": [101, 202, 303],
                        "episodes_per_seed": 5,
                        "epsilon": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            config = load_evaluation_config(path)

        self.assertEqual(config.seeds, (101, 202, 303))
        self.assertEqual(config.total_episodes, 15)

    def test_config_records_explicit_day14_profiling_source(self) -> None:
        config = EvaluationConfig.from_mapping(
            {
                "seeds": [101],
                "source_day14_profiling_report": "experiments/profile.json",
            }
        )

        self.assertEqual(
            config.to_dict()["source_day14_profiling_report"],
            "experiments/profile.json",
        )

    def test_day14_provenance_uses_manifest_config_and_explicit_profile(self) -> None:
        provenance = load_day14_provenance(
            Path("experiments/day14-final-frozen-100k/manifest.json"),
            profiling_report_path=Path(
                "experiments/day14-batch-size-profiling-final/batch-size-comparison.json"
            ),
        )

        self.assertEqual(provenance["replay_backend"], "cpu")
        self.assertEqual(
            provenance["source_day14_profiling_report"],
            "experiments/day14-batch-size-profiling-final/batch-size-comparison.json",
        )
        self.assertEqual(
            provenance["gpu_profiling_summary"]["selected_run_id"],
            "batch-size-32-seed42",
        )

    def test_cpu_dqn_reference_cannot_overwrite_formal_cuda_artifacts(self) -> None:
        default_args = Namespace(device="cpu", output_dir=None, evaluation_id=None)
        output_dir, evaluation_id = _output_destination("dqn", default_args)
        self.assertEqual(output_dir, DQN_REFERENCE_OUTPUT_DIR)
        self.assertEqual(evaluation_id, DQN_REFERENCE_EVALUATION_ID)

        with self.assertRaisesRegex(ValueError, "formal CUDA output directory"):
            _output_destination(
                "dqn",
                Namespace(
                    device="cpu",
                    output_dir=FORMAL_DQN_OUTPUT_DIR,
                    evaluation_id=None,
                ),
            )
        with self.assertRaisesRegex(ValueError, "formal CUDA evaluation id"):
            _output_destination(
                "dqn",
                Namespace(
                    device="cpu",
                    output_dir=None,
                    evaluation_id=FORMAL_DQN_EVALUATION_ID,
                ),
            )

    def test_checkpoint_load_matches_environment_action_count(self) -> None:
        model = DQNNetwork(num_actions=4).cpu()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory)
                / "day14-run"
                / "checkpoints"
                / "step-00100000.pt"
            )
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "format_version": 1,
                    "online_network": model.state_dict(),
                    "config": DQNConfig(
                        total_steps=100_000,
                        seed=42,
                        learning_rate=2e-4,
                        device="cpu",
                    ).to_dict(),
                    "global_step": 100_000,
                },
                checkpoint_path,
            )

            loaded = load_dqn_checkpoint(
                checkpoint_path,
                device="cpu",
                env_factory=ScriptedEvaluationEnv,
            )

        self.assertEqual(loaded.model_id, "day14-run@step-00100000")
        self.assertEqual(loaded.checkpoint_metadata["step"], 100_000)
        self.assertEqual(loaded.training_metadata["training_seed"], 42)

    def test_report_recomputes_statistics_and_visualization_uses_artifacts(self) -> None:
        random_result = evaluate_policy(
            None,
            episodes=1,
            seeds=[101],
            device="cpu",
            env_factory=ScriptedEvaluationEnv,
        )
        dqn_result = evaluate_policy(
            ConstantQNetwork(),
            episodes=1,
            seeds=[101],
            device="cpu",
            env_factory=ScriptedEvaluationEnv,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            random_path, _ = write_evaluation_artifacts(random_result, root / "random")
            dqn_path, _ = write_evaluation_artifacts(dqn_result, root / "dqn")
            report = build_report(random_path, dqn_path, require_cuda=False)
            tampered = json.loads(dqn_path.read_text(encoding="utf-8"))
            tampered["summary"]["mean_return"] = 999.0
            tampered_path = root / "tampered.json"
            tampered_path.write_text(
                json.dumps(tampered),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_report(random_path, tampered_path, require_cuda=False)
            output = root / "returns.png"
            metadata = root / "returns.json"
            render_evaluation_comparison(
                random_path,
                dqn_path,
                output,
                metadata_path=metadata,
            )
            output_size = output.stat().st_size
            metadata_exists = metadata.is_file()

        self.assertIn("Random 的平均 raw Atari return", report)
        self.assertGreater(output_size, 0)
        self.assertTrue(metadata_exists)

    def test_artifact_validator_derives_completion_and_rejects_duplicate_identity(self) -> None:
        payload = {
            "per_episode": [
                {
                    "evaluation_seed": 101,
                    "episode_index": 1,
                    "episode_seed": 101,
                    "episode_return": 1.0,
                    "episode_length": 2,
                    "terminated": True,
                    "truncated": False,
                    "complete": True,
                }
            ]
        }
        rows = validate_episode_rows(
            payload,
            source="memory",
            expected_seeds=[101],
            expected_episodes_per_seed=1,
        )
        self.assertTrue(rows[0]["complete"])

        duplicate = {"per_episode": [*payload["per_episode"], *payload["per_episode"]]}
        with self.assertRaisesRegex(ValueError, "duplicate episode identity"):
            validate_episode_rows(duplicate, source="memory")

        inconsistent = {
            "per_episode": [{**payload["per_episode"][0], "complete": False}]
        }
        with self.assertRaisesRegex(ValueError, "complete disagrees"):
            validate_episode_rows(inconsistent, source="memory")


if __name__ == "__main__":
    unittest.main()
