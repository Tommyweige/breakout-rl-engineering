"""Contract and decision tests for the Day 21 final-training protocol."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.day21_final_training import (
    DAY21_STAGE_TARGETS,
    assess_evaluation_contract_health,
    assess_training_health,
    build_day21_manifest,
    compact_metrics,
    load_day21_config,
    render_day21_markdown,
    select_extension_candidates,
    select_final_checkpoint,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.vectorized import VectorizedDQNTrainer


class TinyActionSpace:
    n = 2


class TinyObservationSpace:
    shape = (4, 84, 84)


class TinyVectorEnv:
    """Small vector seam used to verify continuous milestone training."""

    num_envs = 2
    single_action_space = TinyActionSpace()
    single_observation_space = TinyObservationSpace()

    def __init__(self) -> None:
        self.step_count = 0
        self.reset_count = 0
        self.action_batches: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: object = None):
        del seed, options
        self.reset_count += 1
        self.step_count = 0
        return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8), {}

    def step(self, actions: np.ndarray):
        self.action_batches.append(np.asarray(actions, dtype=np.int64).copy())
        self.step_count += 1
        observations = np.full(
            (self.num_envs, 4, 84, 84),
            self.step_count,
            dtype=np.uint8,
        )
        return (
            observations,
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )

    def close(self) -> None:
        return None


class TinyQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(4 * 84 * 84, 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(observations))


class Day21FinalTrainingTests(unittest.TestCase):
    def test_day21_config_freezes_day20_winner_and_fresh_protocol(self) -> None:
        config = load_day21_config("configs/final-training/manifest.json")

        self.assertEqual(config.winner_family_id, "dueling_double_dqn")
        self.assertEqual(config.algorithm, "double_dqn")
        self.assertEqual(config.architecture, "dueling")
        self.assertEqual(config.training_seeds, (1011, 2022, 3033))
        self.assertTrue(set(config.training_seeds).isdisjoint({11, 22, 33}))
        self.assertEqual(config.stage_targets, DAY21_STAGE_TARGETS)
        self.assertTrue(
            set(config.selection_concrete_seeds).isdisjoint(
                config.holdout_concrete_seeds
            )
        )
        self.assertEqual(
            config.contract.contract_id,
            "day15-breakout-evaluation-v2-fire-reset",
        )
        stage_c_policy = config.protocol()["stage_c_policy"]
        self.assertEqual(
            stage_c_policy["primary_trigger"],
            "2.5M evaluation showed substantial improvement, so 5M continuation remained justified.",
        )
        self.assertEqual(
            stage_c_policy["trigger_evidence"],
            {
                "training_seed": 2022,
                "stage_a_1m_mean_return": 34.86666666666667,
                "stage_b_2_5m_mean_return": 51.4,
                "mean_return_improvement": 16.53333333333333,
            },
        )
        self.assertTrue(stage_c_policy["user_requested_5m"])
        self.assertTrue(stage_c_policy["request_is_supplemental_provenance"])

    def test_manifest_predeclares_stage_targets_and_frozen_sources(self) -> None:
        config = load_day21_config("configs/final-training/manifest.json")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            manifest = build_day21_manifest(
                config,
                manifest_path=destination,
                runs_root=Path(directory) / "runs",
                evaluations_root=Path(directory) / "evaluations",
            )

        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(len(manifest["runs"]), 3)
        self.assertEqual(
            manifest["protocol"]["stage_targets"],
            DAY21_STAGE_TARGETS,
        )
        self.assertEqual(
            manifest["source_of_truth"]["day20_selection"]["sha256"],
            config.source_hashes["day20_selection"],
        )
        self.assertEqual(
            manifest["protocol"]["day20_training_seeds"],
            list(config.day20_training_seeds),
        )
        for entry in manifest["runs"]:
            self.assertEqual(
                set(entry["stages"]),
                set(DAY21_STAGE_TARGETS),
            )

    def test_compact_metrics_is_bounded_to_one_stage_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "metrics.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["global_step", "raw_episode_return", "loss"],
                )
                writer.writeheader()
                for step in (5_000, 10_000, 15_000):
                    writer.writerow(
                        {
                            "global_step": step,
                            "raw_episode_return": step / 1_000,
                            "loss": "",
                        }
                    )
            destination = root / "metrics-stage_a_1m.csv"
            info = compact_metrics(
                source,
                destination,
                max_global_step=10_000,
            )
            with destination.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(info["max_global_step"], 10_000)
        self.assertEqual([int(row["global_step"]) for row in rows], [5_000, 10_000])

    def test_markdown_reads_nested_final_checkpoint_selection(self) -> None:
        report = {
            "status": "completed",
            "protocol": {},
            "selection_decisions": {
                "final_checkpoint": {
                    "selected": {
                        "stage": "stage_b_2_5m",
                        "training_seed": 2022,
                    }
                }
            },
            "canonical_final_model": {
                "model_path": "assets/day21/models/final_model/model.pt",
                "model_sha256": "model-sha",
                "source_checkpoint": {"path": "checkpoint.pt"},
            },
            "final_holdout": {"status": "completed"},
            "training": [],
            "limitations": [],
        }

        markdown = render_day21_markdown(report)

        self.assertIn("selected stage/seed: `stage_b_2_5m` / `2022`", markdown)

    def test_extension_selection_uses_aggregate_mean_not_single_episode_peak(self) -> None:
        records = [
            {
                "training_seed": 1011,
                "stage": "stage_a_1m",
                "target_transitions": 1_000_000,
                "health": {"healthy": True},
                "evaluation": {
                    "summary": {
                        "mean_return": 12.0,
                        "median_return": 12.0,
                        "std_return": 1.0,
                        "max_return": 999.0,
                        "count": 15,
                    }
                },
            },
            {
                "training_seed": 2022,
                "stage": "stage_a_1m",
                "target_transitions": 1_000_000,
                "health": {"healthy": True},
                "evaluation": {
                    "summary": {
                        "mean_return": 20.0,
                        "median_return": 19.0,
                        "std_return": 2.0,
                        "max_return": 25.0,
                        "count": 15,
                    }
                },
            },
            {
                "training_seed": 3033,
                "stage": "stage_a_1m",
                "target_transitions": 1_000_000,
                "health": {"healthy": True},
                "evaluation": {
                    "summary": {
                        "mean_return": 18.0,
                        "median_return": 18.0,
                        "std_return": 2.0,
                        "max_return": 18.0,
                        "count": 15,
                    }
                },
            },
        ]

        selected = select_extension_candidates(records, limit=2)

        self.assertEqual(
            [item["training_seed"] for item in selected],
            [2022, 3033],
        )

    def test_final_selection_prefers_earlier_checkpoint_when_quality_is_near(self) -> None:
        candidates = [
            {
                "training_seed": 1011,
                "stage": "stage_a_1m",
                "target_transitions": 1_000_000,
                "checkpoint": {"path": "early.pt", "step": 1_000_000},
                "health": {"healthy": True},
                "evaluation": {
                    "summary": {
                        "mean_return": 20.0,
                        "median_return": 20.0,
                        "std_return": 1.0,
                        "count": 15,
                    }
                },
            },
            {
                "training_seed": 1011,
                "stage": "stage_c_5m",
                "target_transitions": 5_000_000,
                "checkpoint": {"path": "late.pt", "step": 5_000_000},
                "health": {"healthy": True},
                "evaluation": {
                    "summary": {
                        "mean_return": 20.4,
                        "median_return": 20.0,
                        "std_return": 1.0,
                        "count": 15,
                    }
                },
            },
        ]

        selected = select_final_checkpoint(
            candidates,
            near_equal_absolute_gap=0.5,
        )

        self.assertEqual(selected["stage"], "stage_a_1m")
        self.assertEqual(selected["target_transitions"], 1_000_000)

    def test_health_gate_rejects_non_finite_metric(self) -> None:
        summary = {
            "status": "completed",
            "total_transitions": 4,
            "contract_id": "contract-v2",
            "runtime": {
                "requested_device": "cuda",
                "resolved_device": "cuda:0",
                "cuda_available": True,
            },
            "action_distribution": {"NOOP": 2, "FIRE": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "global_step",
                        "loss",
                        "q_mean",
                        "q_max",
                        "target_mean",
                        "td_error_mean_abs",
                        "gradient_norm",
                        "sps",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "global_step": 4,
                        "loss": "nan",
                        "q_mean": 1.0,
                        "q_max": 1.0,
                        "target_mean": 1.0,
                        "td_error_mean_abs": 1.0,
                        "gradient_norm": 1.0,
                        "sps": 1.0,
                    }
                )
            health = assess_training_health(
                summary,
                metrics_path,
                expected_transitions=4,
                contract_id="contract-v2",
                health_rule={
                    "required_metric_fields": [
                        "loss",
                        "q_mean",
                        "q_max",
                        "target_mean",
                        "td_error_mean_abs",
                        "gradient_norm",
                        "sps",
                    ],
                    "min_distinct_executed_actions": 2,
                },
            )

        self.assertFalse(health["healthy"])
        self.assertIn("non-finite", " ".join(health["failures"]).lower())

    def test_evaluation_health_checks_contract_v2_episode_semantics(self) -> None:
        result = {
            "environment_id": "ALE/Breakout-v5",
            "evaluation_epsilon": 0.0,
            "requested_device": "cuda",
            "resolved_device": "cuda:0",
            "runtime": {
                "requested_device": "cuda",
                "resolved_device": "cuda:0",
            },
            "training": {"contract_id": "contract-v2"},
            "checkpoint": {"contract_id": "contract-v2"},
            "action_distribution_semantics": "executed/wrapper-resolved action",
            "per_episode": [
                {
                    "evaluation_seed": 101,
                    "episode_index": 1,
                    "episode_seed": 101,
                    "terminated": True,
                    "truncated": False,
                    "time_limit": False,
                    "action_distribution_semantics": "executed/wrapper-resolved action",
                    "requested_action_distribution": {"NOOP": 1},
                    "executed_action_distribution": {"FIRE": 1},
                    "auto_fire_reason_counts": {"initial_serve": 1},
                }
            ],
            "summary": {
                "count": 1,
                "complete_episodes": 1,
                "finished_episode_count": 1,
                "terminated_count": 1,
                "truncated_count": 0,
                "time_limit_truncated_count": 0,
                "truncation_rate": 0.0,
            },
        }

        health = assess_evaluation_contract_health(
            result,
            contract_id="contract-v2",
            expected_seeds=[101],
            episodes_per_seed=1,
            expected_concrete_seeds=[101],
        )

        self.assertTrue(health["healthy"])

    def test_train_until_keeps_one_environment_session_across_milestones(self) -> None:
        config = DQNConfig(
            total_steps=8,
            seed=7,
            batch_size=2,
            replay_capacity=8,
            learning_starts=2,
            train_frequency=2,
            target_update_interval=4,
            checkpoint_interval=4,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
            num_envs=2,
            strict_action_selection_parity=True,
        )
        env = TinyVectorEnv()
        with tempfile.TemporaryDirectory() as directory:
            trainer = VectorizedDQNTrainer(
                env,
                config,
                run_dir=Path(directory) / "continuous",
                online_network=TinyQNetwork(),
            )
            first = trainer.train_until(4)
            second = trainer.train_until(8, close=True)

            checkpoint_names = sorted(
                path.name
                for path in (Path(directory) / "continuous" / "checkpoints").glob(
                    "*.pt"
                )
            )

        self.assertEqual(first["total_transitions"], 4)
        self.assertEqual(second["total_transitions"], 8)
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(len(env.action_batches), 4)
        self.assertEqual(
            checkpoint_names,
            ["step-00000004.pt", "step-00000008.pt"],
        )


if __name__ == "__main__":
    unittest.main()
