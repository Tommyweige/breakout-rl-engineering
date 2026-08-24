"""Public behavior tests for the Day 13 debugging workflow."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from analyze_training_run import analyze_run
from breakout_rl.replay_tensors import ReplayTensorBatch
from breakout_rl.training.diagnostics import (
    aggregate_training_metrics,
    check_finite,
    collect_runtime_metadata,
    run_fixed_batch_overfit,
)


class TinyOverfitNetwork(nn.Module):
    """Small model with the same ``(batch, actions)`` output contract as DQN."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(states)


def make_overfit_batch() -> ReplayTensorBatch:
    return ReplayTensorBatch(
        states=torch.eye(2),
        actions=torch.tensor([0, 1], dtype=torch.long),
        rewards=torch.zeros(2),
        next_states=torch.zeros(2, 2),
        terminated=torch.ones(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
    )


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_finite_check_distinguishes_finite_and_non_finite_tensors(self) -> None:
        finite = check_finite(torch.tensor([1.0, -2.0]), name="loss")
        non_finite = check_finite(torch.tensor([1.0, float("nan")]), name="loss")

        self.assertTrue(finite.is_finite)
        self.assertEqual(finite.non_finite_count, 0)
        self.assertFalse(non_finite.is_finite)
        self.assertEqual(non_finite.non_finite_count, 1)

    def test_metric_aggregation_reports_actions_and_decision_ratio(self) -> None:
        rows = [
            {
                "global_step": "1",
                "episode": "0",
                "raw_episode_return": "",
                "loss": "2.0",
                "q_mean": "1.0",
                "q_max": "2.0",
                "gradient_norm": "3.0",
                "epsilon": "0.9",
                "replay_size": "1",
                "sps": "10.0",
                "action": "0",
                "action_source": "random",
            },
            {
                "global_step": "2",
                "episode": "1",
                "raw_episode_return": "4.0",
                "loss": "1.0",
                "q_mean": "2.0",
                "q_max": "3.0",
                "gradient_norm": "2.0",
                "epsilon": "0.8",
                "replay_size": "2",
                "sps": "12.0",
                "action": "1",
                "action_source": "greedy",
            },
        ]

        report = aggregate_training_metrics(rows)

        self.assertEqual(report["step_range"], [1, 2])
        self.assertEqual(report["episodes_completed"], 1)
        self.assertAlmostEqual(report["return_summary"]["mean"], 4.0)
        self.assertEqual(report["action_distribution"]["counts"]["NOOP"], 1)
        self.assertEqual(report["action_distribution"]["counts"]["FIRE"], 1)
        self.assertAlmostEqual(report["decision_distribution"]["random_ratio"], 0.5)
        self.assertEqual(report["non_finite_count"], 0)

    def test_fixed_batch_overfit_reduces_loss_with_fixed_targets(self) -> None:
        torch.manual_seed(42)
        model = TinyOverfitNetwork()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

        result = run_fixed_batch_overfit(
            model,
            optimizer,
            make_overfit_batch(),
            torch.tensor([1.0, -1.0]),
            updates=40,
        )

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.losses), 40)

    def test_analyzer_reads_minimal_fixture_and_writes_required_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "fixture"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"run_id": "fixture", "runtime": {"seed": 42}}),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"episodes": 1, "replay_size": 2}),
                encoding="utf-8",
            )
            with (run_dir / "metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "global_step",
                        "episode",
                        "raw_episode_return",
                        "loss",
                        "q_mean",
                        "q_max",
                        "gradient_norm",
                        "epsilon",
                        "replay_size",
                        "sps",
                        "action",
                        "action_source",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "global_step": 1,
                        "episode": 0,
                        "raw_episode_return": "",
                        "loss": 2.0,
                        "q_mean": 1.0,
                        "q_max": 2.0,
                        "gradient_norm": 3.0,
                        "epsilon": 0.9,
                        "replay_size": 1,
                        "sps": 10.0,
                        "action": 0,
                        "action_source": "random",
                    }
                )
                writer.writerow(
                    {
                        "global_step": 2,
                        "episode": 1,
                        "raw_episode_return": 4.0,
                        "loss": 1.0,
                        "q_mean": 2.0,
                        "q_max": 3.0,
                        "gradient_norm": 2.0,
                        "epsilon": 0.8,
                        "replay_size": 2,
                        "sps": 12.0,
                        "action": 1,
                        "action_source": "greedy",
                    }
                )

            report = analyze_run(run_dir)

            self.assertEqual(report["step_range"], [1, 2])
            for filename in (
                "return-curve.png",
                "loss-curve.png",
                "q-values.png",
                "gradient-norm.png",
            ):
                plot = run_dir / "plots" / filename
                self.assertTrue(plot.is_file())
                self.assertGreater(plot.stat().st_size, 0)

    def test_runtime_metadata_is_best_effort_without_cuda_or_git(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata = collect_runtime_metadata(
                seed=42,
                device="cpu",
                run_dir=Path(temporary_directory),
            )

        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["device"], "cpu")
        self.assertIn("cuda_available", metadata)
        self.assertIn("git_commit_sha", metadata)


if __name__ == "__main__":
    unittest.main()
