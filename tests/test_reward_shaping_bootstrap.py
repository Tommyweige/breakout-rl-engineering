"""Tests for Issue #10 paired and hierarchical bootstrap analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from breakout_rl.evaluation_contract import load_evaluation_contract
from scripts.analysis.bootstrap_reward_shaping import (
    BASELINE_LABEL,
    CHECKPOINT_STEP,
    EXPECTED_SCORE_DEFINITION,
    SHAPED_LABEL,
    PairedRecord,
    build_analysis,
    hierarchical_bootstrap,
    load_paired_artifacts,
    paired_bootstrap,
    render_analysis_report,
    training_seed_level_bootstrap,
    validate_paired_rows,
    write_analysis_artifacts,
)


def _records(training_seed: int, delta: float) -> list[PairedRecord]:
    return [
        PairedRecord(
            training_seed=training_seed,
            evaluation_seed=1001 + index,
            baseline_raw_score=10.0,
            shaped_raw_score=10.0 + delta,
        )
        for index in range(50)
    ]


def _variable_records(training_seed: int, shift: int) -> list[PairedRecord]:
    return [
        PairedRecord(
            training_seed=training_seed,
            evaluation_seed=1001 + index,
            baseline_raw_score=10.0,
            shaped_raw_score=10.0 + float((index % 9) - 4 + shift),
        )
        for index in range(50)
    ]


def _artifact_payload(training_seed: int) -> dict[str, object]:
    contract_path = Path("configs/eval/breakout_contract_v2.json")
    contract = load_evaluation_contract(contract_path)
    canonical_contract = json.dumps(
        contract.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(canonical_contract).hexdigest()
    paired_rows = [
        {
            "evaluation_seed": record.evaluation_seed,
            "baseline_raw_score": record.baseline_raw_score,
            "shaped_raw_score": record.shaped_raw_score,
            "score_delta": record.delta,
        }
        for record in _records(training_seed, 1.0)
    ]
    variants = [
        {
            "label": label,
            "config": {
                "life_loss_penalty": 0.0 if label == BASELINE_LABEL else -1.0,
                "seed": training_seed,
                "total_steps": CHECKPOINT_STEP,
                "algorithm": "double_dqn",
                "architecture": "dueling",
            },
            "contract_path": contract_path.as_posix(),
            "contract_sha256": contract_sha256,
        }
        for label in (BASELINE_LABEL, SHAPED_LABEL)
    ]
    return {
        "training_seed": training_seed,
        "training_transitions": CHECKPOINT_STEP,
        "checkpoint_steps": [250_000, 500_000, 750_000, CHECKPOINT_STEP],
        "contract_id": "day15-breakout-evaluation-v2-fire-reset",
        "contract_path": contract_path.as_posix(),
        "eval_seeds": list(range(1001, 1051)),
        "evaluation_seed_count": 50,
        "evaluation_episodes_per_seed": 1,
        "evaluation_score_definition": EXPECTED_SCORE_DEFINITION,
        "variants": variants,
        "comparisons": {
            str(CHECKPOINT_STEP): {
                SHAPED_LABEL: {
                    "baseline": {
                        "episode_count": 50,
                        "raw_score": {"count": 50},
                        "score_definition": EXPECTED_SCORE_DEFINITION,
                    },
                    "shaped": {
                        "episode_count": 50,
                        "raw_score": {"count": 50},
                        "score_definition": (
                            "raw Atari game reward sum; no training shaping"
                        )
                    },
                    "paired_rows": paired_rows,
                }
            }
        },
    }


def _standalone_paired_payload(training_seed: int) -> dict[str, object]:
    comparison = _artifact_payload(training_seed)["comparisons"][
        str(CHECKPOINT_STEP)
    ][SHAPED_LABEL]
    return {
        SHAPED_LABEL: {
            "evaluation_seed_count": 50,
            "baseline": comparison["baseline"],
            "shaped": comparison["shaped"],
            "paired_rows": comparison["paired_rows"],
        }
    }


class RewardShapingBootstrapTests(unittest.TestCase):
    def test_pair_loader_fails_closed_on_missing_pair(self) -> None:
        rows = [
            {
                "evaluation_seed": 1001,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 11.0,
                "score_delta": 1.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing paired"):
            validate_paired_rows(
                rows,
                training_seed=2022,
                expected_evaluation_seeds=[1001, 1002],
            )

    def test_pair_loader_rejects_inconsistent_delta(self) -> None:
        rows = [
            {
                "evaluation_seed": 1001,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 11.0,
                "score_delta": 99.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "score_delta"):
            validate_paired_rows(
                rows,
                training_seed=2022,
                expected_evaluation_seeds=[1001],
            )

    def test_pair_loader_rejects_duplicate_and_unexpected_pairs(self) -> None:
        duplicate_rows = [
            {
                "evaluation_seed": 1001,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 11.0,
            },
            {
                "evaluation_seed": 1001,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 12.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate evaluation seed"):
            validate_paired_rows(
                duplicate_rows,
                training_seed=2022,
                expected_evaluation_seeds=[1001, 1002],
            )

        unexpected_rows = [
            {
                "evaluation_seed": 1001,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 11.0,
            },
            {
                "evaluation_seed": 999,
                "baseline_raw_score": 10.0,
                "shaped_raw_score": 11.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "unexpected evaluation seeds"):
            validate_paired_rows(
                unexpected_rows,
                training_seed=2022,
                expected_evaluation_seeds=[1001],
            )

    def test_build_analysis_fails_closed_on_incomplete_records(self) -> None:
        records = {
            2022: _records(2022, 1.0),
            2023: _records(2023, 1.0)[:-1],
            2024: _records(2024, 1.0),
        }

        with self.assertRaisesRegex(ValueError, "missing paired"):
            build_analysis(records, iterations=10_000, bootstrap_seed=3)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        records = _records(2022, 2.0)

        first, first_means, first_medians = paired_bootstrap(
            records,
            iterations=10_000,
            rng_seed=1234,
        )
        second, second_means, second_medians = paired_bootstrap(
            records,
            iterations=10_000,
            rng_seed=1234,
        )

        self.assertEqual(first, second)
        np.testing.assert_array_equal(first_means, second_means)
        np.testing.assert_array_equal(first_medians, second_medians)
        self.assertEqual(first["point_mean_delta"], 2.0)
        self.assertEqual(first["wins"], 50)

    def test_seed_level_and_hierarchical_bootstrap_are_deterministic(self) -> None:
        records = {
            2022: _variable_records(2022, 0),
            2023: _variable_records(2023, -1),
            2024: _variable_records(2024, 1),
        }
        effects = {
            seed: float(np.mean([record.delta for record in records[seed]]))
            for seed in records
        }
        first_seed, first_means, first_medians = training_seed_level_bootstrap(
            effects,
            iterations=10_000,
            rng_seed=4,
        )
        second_seed, second_means, second_medians = training_seed_level_bootstrap(
            effects,
            iterations=10_000,
            rng_seed=4,
        )
        first_hierarchical, first_h_means, first_h_medians = hierarchical_bootstrap(
            records,
            iterations=10_000,
            rng_seed=5,
        )
        second_hierarchical, second_h_means, second_h_medians = hierarchical_bootstrap(
            records,
            iterations=10_000,
            rng_seed=5,
        )

        self.assertEqual(first_seed, second_seed)
        self.assertEqual(first_hierarchical, second_hierarchical)
        np.testing.assert_array_equal(first_means, second_means)
        np.testing.assert_array_equal(first_medians, second_medians)
        np.testing.assert_array_equal(first_h_means, second_h_means)
        np.testing.assert_array_equal(first_h_medians, second_h_medians)
        self.assertGreater(np.ptp(first_h_means), 0.0)

    def test_analysis_schema_keeps_training_seed_as_independent_unit(self) -> None:
        records = {
            2022: _records(2022, 9.48),
            2023: _records(2023, -1.74),
            2024: _records(2024, 2.04),
        }

        summary, distributions = build_analysis(
            records,
            iterations=10_000,
            bootstrap_seed=20260916,
        )

        self.assertEqual(summary["n_train"], 3)
        self.assertEqual(summary["n_eval_per_train"], 50)
        self.assertTrue(summary["pairing"]["pairs_validated"])
        self.assertTrue(
            summary["cross_training_seed_inference"][
                "pseudo_replication_used"
            ]
            is False
        )
        self.assertEqual(
            summary["training_seed_level_bootstrap"]["training_seed_ids"],
            [2022, 2023, 2024],
        )
        self.assertEqual(
            summary["hierarchical_bootstrap"]["n_eval_per_train"],
            50,
        )
        self.assertEqual(len(distributions), 50_000)

    def test_analysis_accepts_future_training_seed_without_code_change(self) -> None:
        records = {
            2022: _records(2022, 1.0),
            2023: _records(2023, 1.0),
            2024: _records(2024, 1.0),
            2025: _records(2025, 3.0),
        }

        summary, _ = build_analysis(
            records,
            iterations=10_000,
            bootstrap_seed=11,
        )

        self.assertEqual(summary["training_seed_ids"], [2022, 2023, 2024, 2025])
        self.assertEqual(summary["n_train"], 4)
        self.assertEqual(
            summary["cross_training_seed_inference"]["observed_effect_range"],
            {"minimum": 1.0, "maximum": 3.0, "span": 2.0},
        )

    def test_written_cross_and_hierarchical_artifacts_are_json(self) -> None:
        records = {
            2022: _records(2022, 1.0),
            2023: _records(2023, 0.0),
            2024: _records(2024, -1.0),
        }
        summary, distributions = build_analysis(
            records,
            iterations=10_000,
            bootstrap_seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_analysis_artifacts(
                Path(directory),
                summary,
                distributions,
            )
            cross_seed = json.loads(
                paths["cross-seed-bootstrap.json"].read_text(encoding="utf-8")
            )
            hierarchical = json.loads(
                paths["hierarchical-bootstrap.json"].read_text(encoding="utf-8")
            )
            summary_json = json.loads(
                paths["bootstrap-summary.json"].read_text(encoding="utf-8")
            )
            per_seed_rows = list(
                csv.DictReader(
                    paths["per-seed-bootstrap.csv"]
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )
            distribution_rows = list(
                csv.DictReader(
                    paths["bootstrap-distribution.csv"]
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )
            report = paths["analysis-report.md"].read_text(encoding="utf-8")

        self.assertEqual(
            set(paths),
            {
                "bootstrap-summary.json",
                "per-seed-bootstrap.csv",
                "cross-seed-bootstrap.json",
                "hierarchical-bootstrap.json",
                "bootstrap-distribution.csv",
                "analysis-report.md",
            },
        )
        self.assertEqual(cross_seed["n_train"], 3)
        self.assertEqual(hierarchical["n_eval_per_train"], 50)
        self.assertEqual(summary_json["n_train"], 3)
        self.assertEqual(len(per_seed_rows), 3)
        self.assertEqual(len(distribution_rows), 50_000)
        for field in (
            "mean_delta_95_ci_lower",
            "mean_delta_95_ci_upper",
            "median_delta_95_ci_lower",
            "median_delta_95_ci_upper",
            "p_bootstrap_mean_effect_gt_zero",
            "bootstrap_iterations",
            "bootstrap_rng_seed",
            "n_eval",
            "wins",
            "ties",
            "losses",
        ):
            self.assertIn(field, per_seed_rows[0])
        for field in (
            "point_mean_delta",
            "mean_delta_95_ci",
            "p_bootstrap_mean_effect_gt_zero",
            "bootstrap_iterations",
            "bootstrap_rng_seed",
            "n_train",
        ):
            self.assertIn(field, cross_seed)
            self.assertIn(field, hierarchical)
        self.assertIn("analysis_level", distribution_rows[0])
        self.assertIn("effect_mean", distribution_rows[0])
        self.assertIn("Pooled 150-episode bootstrap: **not used**", report)
        self.assertIn("Candidate: `life_loss_penalty=-1.0`", report)

    def test_loader_validates_contract_and_one_million_transition_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation_dir = Path(directory) / "evaluation"
            evaluation_dir.mkdir()
            summary_path = evaluation_dir / "sweep-summary.json"
            summary_path.write_text(
                json.dumps(_artifact_payload(2022)),
                encoding="utf-8",
            )
            paired_path = evaluation_dir / "paired-evaluation-1000000.json"
            paired_path.write_text(
                json.dumps(_standalone_paired_payload(2022)),
                encoding="utf-8",
            )

            records, sources, contract = load_paired_artifacts(
                {2022: evaluation_dir}
            )

            self.assertEqual(len(records[2022]), 50)
            self.assertEqual(sources[2022], summary_path.as_posix())
            self.assertEqual(
                contract["contract_id"], "day15-breakout-evaluation-v2-fire-reset"
            )
            self.assertTrue(contract["source_artifacts_include_contract_sha256"])

            mismatched_paired = _standalone_paired_payload(2022)
            mismatched_paired[SHAPED_LABEL]["paired_rows"][0][
                "shaped_raw_score"
            ] = 999.0
            mismatched_paired[SHAPED_LABEL]["paired_rows"][0]["score_delta"] = 989.0
            paired_path.write_text(
                json.dumps(mismatched_paired),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not match sweep summary"):
                load_paired_artifacts({2022: evaluation_dir})

            malformed = _artifact_payload(2022)
            malformed["training_transitions"] = 999_999
            summary_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_transitions"):
                load_paired_artifacts({2022: evaluation_dir})

    def test_negative_cross_seed_report_is_not_called_positive(self) -> None:
        records = {
            2022: _records(2022, -1.0),
            2023: _records(2023, -2.0),
            2024: _records(2024, -3.0),
        }
        summary, _ = build_analysis(
            records,
            iterations=10_000,
            bootstrap_seed=12,
        )

        report = summary["training_seed_level_bootstrap"]
        self.assertLess(report["point_mean_delta"], 0)
        rendered = render_analysis_report(summary)
        self.assertIn("Cross-seed point-estimate direction: `negative`", rendered)
        self.assertNotIn("suggestive positive point estimate", rendered)


if __name__ == "__main__":
    unittest.main()
