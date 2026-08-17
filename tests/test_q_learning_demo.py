"""Behavioral tests for the Day 6 tabular Q-Learning demo."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from q_learning_demo import (
    TrainingStep,
    epsilon_greedy_action,
    q_learning_update,
    train_q_learning,
    write_trace_csv,
    write_trace_json,
)


class QLearningUpdateTests(unittest.TestCase):
    def test_non_terminal_update_bootstraps_from_next_state(self) -> None:
        updated_q, td_error, target = q_learning_update(
            current_q=1.0,
            reward=2.0,
            next_q_max=4.0,
            alpha=0.5,
            gamma=0.9,
            terminated=False,
        )

        self.assertAlmostEqual(target, 5.6)
        self.assertAlmostEqual(td_error, 4.6)
        self.assertAlmostEqual(updated_q, 3.3)

    def test_terminal_update_does_not_bootstrap(self) -> None:
        updated_q, td_error, target = q_learning_update(
            current_q=0.5,
            reward=1.0,
            next_q_max=999.0,
            alpha=0.1,
            gamma=0.99,
            terminated=True,
        )

        self.assertEqual(target, 1.0)
        self.assertAlmostEqual(td_error, 0.5)
        self.assertAlmostEqual(updated_q, 0.55)

    def test_zero_learning_rate_keeps_the_current_value(self) -> None:
        updated_q, td_error, target = q_learning_update(
            current_q=7.0,
            reward=2.0,
            next_q_max=4.0,
            alpha=0.0,
            gamma=0.9,
            terminated=False,
        )

        self.assertEqual(updated_q, 7.0)
        self.assertAlmostEqual(td_error, -1.4)
        self.assertAlmostEqual(target, 5.6)

    def test_update_hyperparameters_must_be_in_unit_interval(self) -> None:
        for parameter, value in (
            ("alpha", -0.1),
            ("alpha", 1.1),
            ("gamma", -0.1),
            ("gamma", 1.1),
        ):
            with self.subTest(parameter=parameter, value=value):
                kwargs = {
                    "current_q": 0.0,
                    "reward": 0.0,
                    "next_q_max": 0.0,
                    "alpha": 0.1,
                    "gamma": 0.9,
                    "terminated": False,
                }
                kwargs[parameter] = value
                with self.assertRaises(ValueError):
                    q_learning_update(**kwargs)


class EpsilonGreedyTests(unittest.TestCase):
    def test_zero_epsilon_selects_the_highest_q_value(self) -> None:
        action = epsilon_greedy_action(
            q_values=[0.2, 1.5, 0.8],
            epsilon=0.0,
            rng=random.Random(42),
        )

        self.assertEqual(action, 1)

    def test_full_epsilon_uses_a_reproducible_random_branch(self) -> None:
        first_rng = random.Random(42)
        second_rng = random.Random(42)

        first_actions = [
            epsilon_greedy_action([0.0, 10.0], epsilon=1.0, rng=first_rng)
            for _ in range(20)
        ]
        second_actions = [
            epsilon_greedy_action([0.0, 10.0], epsilon=1.0, rng=second_rng)
            for _ in range(20)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertGreater(len(set(first_actions)), 1)

    def test_epsilon_must_be_in_unit_interval(self) -> None:
        for epsilon in (-0.1, 1.1):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    epsilon_greedy_action([0.0, 1.0], epsilon, random.Random(42))


class TrainingDemoTests(unittest.TestCase):
    def test_same_seed_and_hyperparameters_reproduce_the_q_table(self) -> None:
        first_table = train_q_learning(
            episodes=20,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.2,
            seed=42,
        )
        second_table = train_q_learning(
            episodes=20,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.2,
            seed=42,
        )

        self.assertEqual(first_table, second_table)

    def test_demo_learns_the_rewarding_path(self) -> None:
        q_table = train_q_learning(
            episodes=20,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.2,
            seed=42,
        )

        self.assertGreater(q_table[0]["RIGHT"], q_table[0]["LEFT"])
        self.assertGreater(q_table[1]["RIGHT"], q_table[1]["LEFT"])


class TraceExportTests(unittest.TestCase):
    def test_trace_exports_keep_update_values_and_run_metadata(self) -> None:
        trace = [
            TrainingStep(
                episode=19,
                step=0,
                state=0,
                action="RIGHT",
                reward=0.0,
                next_state=1,
                terminated=False,
                current_q=0.0,
                next_q_max=0.1,
                target=0.099,
                td_error=0.099,
                updated_q=0.0099,
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path = output_dir / "trace.csv"
            json_path = output_dir / "trace.json"

            write_trace_csv(csv_path, trace)
            write_trace_json(
                json_path,
                trace,
                episodes=20,
                alpha=0.1,
                gamma=0.99,
                epsilon=0.2,
                seed=42,
            )

            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            json_payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_rows[0]["updated_q"], "0.0099")
        self.assertEqual(csv_rows[0]["next_q_max"], "0.1")
        self.assertEqual(json_payload["metadata"]["seed"], 42)
        self.assertEqual(json_payload["updates"][0]["target"], 0.099)


if __name__ == "__main__":
    unittest.main()
