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
    summarize_training_run,
)
from breakout_rl.training.survival import compute_episode_survival_metrics
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import DQNTrainer
from breakout_rl.training.reward_shaping import shape_training_reward
from breakout_rl.training.vectorized import VectorizedDQNTrainer
from scripts.training.run_reward_shaping_sweep import build_parser, run_sweep
from scripts.analysis.audit_reward_shaping_schedule import build_parity_audit


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
    def test_schedule_audit_separates_learning_parity_from_full_run_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = Path("configs/eval/breakout_contract_v2.json").resolve()

            def write_config(
                name: str,
                *,
                total_steps: int,
                checkpoint_interval: int,
                penalty: float,
            ) -> Path:
                path = root / f"{name}.json"
                config = DQNConfig(
                    total_steps=total_steps,
                    seed=2022,
                    algorithm="double_dqn",
                    architecture="dueling",
                    checkpoint_interval=checkpoint_interval,
                    life_loss_penalty=penalty,
                    device="cpu",
                ).to_dict()
                path.write_text(
                    json.dumps(
                        {
                            "training_config": config,
                            "contract": str(contract_path),
                        }
                    ),
                    encoding="utf-8",
                )
                return path

            stage2_config = write_config(
                "stage2",
                total_steps=1000,
                checkpoint_interval=250,
                penalty=0.0,
            )
            stage3_baseline = write_config(
                "stage3-baseline",
                total_steps=2000,
                checkpoint_interval=500,
                penalty=0.0,
            )
            stage3_penalty = write_config(
                "stage3-penalty",
                total_steps=2000,
                checkpoint_interval=500,
                penalty=-1.0,
            )
            stage2_manifest = root / "stage2-manifest.json"
            stage2_manifest.write_text(
                json.dumps(
                    {
                        "variants": [
                            {
                                "label": "penalty-0.0",
                                "config_path": str(stage2_config),
                                "status": "completed",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stage3_manifest = root / "stage3-manifest.json"
            stage3_manifest.write_text(
                json.dumps(
                    {
                        "parallel_execution": True,
                        "variants": [
                            {
                                "label": "penalty-0.0",
                                "config_path": str(stage3_baseline),
                                "status": "completed",
                            },
                            {
                                "label": "penalty-minus-1.0",
                                "config_path": str(stage3_penalty),
                                "status": "completed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            resume_validation = root / "resume-validation.json"
            resume_validation.write_text(
                json.dumps(
                    {
                        "exact_continuation_available": False,
                        "checkpoints": [
                            {
                                "exact_continuation_available": False,
                                "blockers": ["replay buffer contents are not checkpointed"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = build_parity_audit(
                stage2_config_path=stage2_config,
                stage3_config_path=stage3_baseline,
                stage2_manifest_path=stage2_manifest,
                stage3_manifest_path=stage3_manifest,
                resume_validation_path=resume_validation,
            )

        self.assertTrue(
            audit["schedule_audit"]["stage2_vs_stage3_250k_schedule_equivalent"]
        )
        self.assertFalse(audit["full_runtime_reproduction_equivalent"])
        self.assertFalse(
            audit["does_this_invalidate_stage3_internal_baseline_vs_minus1"]
        )

    def test_sweep_supports_non_default_training_seed_for_replication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "stage3b"
            args = build_parser().parse_args(
                [
                    "--dry-run",
                    "--parallel",
                    "--training-seed",
                    "2023",
                    "--expected-steps",
                    "1000000",
                    "--output-dir",
                    str(output_dir),
                    "--config",
                    "configs/issue9_reward_shaping_stage3b_seed2023_baseline.json",
                    "--config",
                    "configs/issue9_reward_shaping_stage3b_seed2023_penalty_minus1.json",
                ]
            )
            manifest = run_sweep(args)

        self.assertEqual(manifest["training_seed"], 2023)
        self.assertEqual(
            manifest["artifact_type"],
            "issue9_reward_shaping_stage3b_1m_training_sweep",
        )
        self.assertTrue(manifest["parallel_execution"])

    def test_training_summary_keeps_q_and_td_statistics_at_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            config = DQNConfig(total_steps=100, device="cpu").to_dict()
            config["run_id"] = "test-run"
            (run_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            with (run_dir / "metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "global_step",
                        "raw_episode_return",
                        "training_episode_return",
                        "episode_length",
                        "episode_life_loss_count",
                        "q_mean",
                        "q_max",
                        "q_min",
                        "target_mean",
                        "target_max",
                        "td_error_mean_abs",
                        "td_error_max_abs",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "global_step": 25,
                        "raw_episode_return": 3,
                        "training_episode_return": 2,
                        "episode_length": 10,
                        "episode_life_loss_count": 1,
                        "q_mean": 0.5,
                        "q_max": 1.0,
                        "q_min": 0.0,
                        "target_mean": 0.4,
                        "target_max": 0.8,
                        "td_error_mean_abs": 0.2,
                        "td_error_max_abs": 0.6,
                    }
                )
            report = summarize_training_run(run_dir)

        milestone = report["milestones"]["25_percent"]
        self.assertEqual(
            milestone["recent_q_value_statistics"]["q_mean"]["mean"],
            0.5,
        )
        self.assertEqual(
            milestone["recent_td_error_statistics"]["td_error_mean_abs"]["mean"],
            0.2,
        )

    def test_sweep_rejects_non_empty_rerun_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "sweep"
            run_dir = output_dir / "runs" / "penalty-0.0"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.csv").write_text("existing\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--config",
                    "configs/issue9_reward_shaping_250k_baseline.json",
                    "--config",
                    "configs/issue9_reward_shaping_250k_penalty_minus1.json",
                ]
            )
            with self.assertRaises(FileExistsError):
                run_sweep(args)

    def test_sweep_rejects_explicit_config_hyperparameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            altered_config = Path(directory) / "altered.json"
            payload = json.loads(
                Path("configs/issue9_reward_shaping_250k_penalty_minus1.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["training_config"]["learning_rate"] = 0.0002
            altered_config.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--dry-run",
                    "--output-dir",
                    str(Path(directory) / "sweep"),
                    "--config",
                    "configs/issue9_reward_shaping_250k_baseline.json",
                    "--config",
                    str(altered_config),
                ]
            )
            with self.assertRaises(ValueError):
                run_sweep(args)

    def test_survival_metrics_use_life_loss_timing_not_only_episode_count(self) -> None:
        metrics = compute_episode_survival_metrics(
            raw_score=10.0,
            episode_length=100,
            life_loss_steps=[20, 70],
        )

        self.assertEqual(metrics["score_per_life"], 5.0)
        self.assertEqual(metrics["frames_between_life_losses"], 50.0)
        self.assertEqual(metrics["time_to_first_life_loss"], 20)
        self.assertEqual(metrics["life_losses_per_1000_steps"], 20.0)

    def test_survival_metrics_handle_an_episode_without_life_loss(self) -> None:
        metrics = compute_episode_survival_metrics(
            raw_score=4.0,
            episode_length=100,
            life_loss_steps=[],
        )

        self.assertEqual(metrics["score_per_life"], 4.0)
        self.assertIsNone(metrics["frames_between_life_losses"])
        self.assertIsNone(metrics["time_to_first_life_loss"])
        self.assertEqual(metrics["life_losses_per_1000_steps"], 0.0)

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
        self.assertEqual(float(rows[-1]["score_per_life"]), 1.0)
        self.assertEqual(float(rows[-1]["episode_frames_between_life_losses"]), 1.0)
        self.assertEqual(int(rows[-1]["time_to_first_life_loss"]), 2)
        self.assertEqual(float(rows[-1]["episode_life_losses_per_1000_steps"]), 500.0)
        self.assertEqual(float(rows[-1]["life_losses_per_1000_steps"]), 500.0)
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
        self.assertEqual(float(rows[0]["score_per_life"]), 0.0)
        self.assertEqual(int(rows[0]["time_to_first_life_loss"]), 1)
        self.assertEqual(float(rows[1]["life_losses_per_1000_steps"]), 500.0)

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
        self.assertEqual(result.episodes[0].score_per_life, 2.0)
        self.assertEqual(result.episodes[0].time_to_first_life_loss, 1)
        self.assertIsNone(result.episodes[0].frames_between_life_losses)
        self.assertEqual(result.episodes[0].life_losses_per_1000_steps, 1000.0)
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
