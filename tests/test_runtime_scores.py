"""Tests for Day 26 multi-episode score evidence and visualization seams."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from breakout_rl.runtime_scores import (
    PAIRING_DEFINITIONS,
    RUNTIME_SCORE_REQUIRED_TARGETS,
    RUNTIME_SCORE_TARGETS,
    RuntimeScoreConfig,
    aggregate_runtime_scores,
    paired_score_differences,
    score_statistics,
    validate_runtime_episode_rows,
)
from scripts.visualization.visualize_runtime_score_distribution import (
    load_runtime_score_visualization_data,
    render_runtime_score_distribution,
)


def _episode_rows(runtime: str, seeds: list[int], offset: float = 0.0) -> list[dict]:
    return [
        {
            "runtime": runtime,
            "seed": seed,
            "evaluation_seed": seed,
            "episode_index": 1,
            "episode_return": float(index + offset),
            "score": float(index + offset),
            "episode_length": 10 + index,
            "termination_reason": "terminated",
            "time_limit_source": None,
            "terminated": True,
            "truncated": False,
            "runtime_error": None,
            "failure": False,
        }
        for index, seed in enumerate(seeds)
    ]


class RuntimeScoreTests(unittest.TestCase):
    def test_score_config_predeclares_thirty_unique_extended_seeds(self) -> None:
        payload = json.loads(
            Path("configs/eval/day26_runtime_score_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        config = RuntimeScoreConfig.from_mapping(payload)

        self.assertEqual(len(config.concrete_episode_seeds), 30)
        self.assertEqual(len(set(config.concrete_episode_seeds)), 30)
        self.assertEqual(
            config.seed_selection_rule, "predeclared_before_runtime_results"
        )
        self.assertTrue(
            set(payload["source_protocol"]["base_concrete_episode_seeds"]).issubset(
                config.concrete_episode_seeds
            )
        )
        self.assertNotIn("results", payload)

    def test_raw_episode_rows_rebuild_requested_statistics(self) -> None:
        seeds = [101, 102, 103]
        rows = _episode_rows("pytorch_cuda_fp32", seeds)

        aggregate = aggregate_runtime_scores(
            rows,
            runtime="pytorch_cuda_fp32",
            expected_seeds=seeds,
        )
        expected = score_statistics([0.0, 1.0, 2.0])

        self.assertEqual(aggregate["episode_count"], 3)
        self.assertEqual(aggregate["failure_count"], 0)
        self.assertEqual(aggregate["episode_length_statistics"]["mean"], 11.0)
        for field in ("mean", "median", "std", "p10", "p90", "min", "max"):
            self.assertEqual(aggregate[field], expected[field])

    def test_optional_fp16_can_be_omitted_from_the_declared_runtime_set(self) -> None:
        payload = json.loads(
            Path("configs/eval/day26_runtime_score_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        payload["runtime_targets"] = list(RUNTIME_SCORE_REQUIRED_TARGETS)

        config = RuntimeScoreConfig.from_mapping(payload)

        self.assertEqual(config.runtime_targets, RUNTIME_SCORE_REQUIRED_TARGETS)

    def test_seed_selection_is_rejected_when_not_predeclared(self) -> None:
        payload = json.loads(
            Path("configs/eval/day26_runtime_score_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        payload["seed_selection_rule"] = "selected_after_results"

        with self.assertRaisesRegex(ValueError, "predeclared"):
            RuntimeScoreConfig.from_mapping(payload)

    def test_duplicate_declared_seed_is_rejected(self) -> None:
        payload = json.loads(
            Path("configs/eval/day26_runtime_score_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        payload["concrete_episode_seeds"][1] = payload["concrete_episode_seeds"][0]

        with self.assertRaisesRegex(ValueError, "unique"):
            RuntimeScoreConfig.from_mapping(payload)

    def test_paired_comparison_requires_the_same_seed_set(self) -> None:
        seeds = [101, 102, 103]
        left = _episode_rows("onnx_cuda_fp32", seeds, offset=1.0)
        right = _episode_rows("tensorrt_cuda_fp32", seeds)

        paired = paired_score_differences(
            left,
            right,
            left_runtime="onnx_cuda_fp32",
            right_runtime="tensorrt_cuda_fp32",
            expected_seeds=seeds,
        )

        self.assertEqual(paired["declared_seed_count"], 3)
        self.assertEqual(paired["paired_episode_count"], 3)
        self.assertEqual(
            [row["difference"] for row in paired["differences"]],
            [1.0, 1.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "declared evaluation set"):
            paired_score_differences(
                left,
                right[:-1],
                left_runtime="onnx_cuda_fp32",
                right_runtime="tensorrt_cuda_fp32",
                expected_seeds=seeds,
            )

    def test_duplicate_episode_seed_is_rejected(self) -> None:
        rows = _episode_rows("pytorch_cuda_fp32", [101, 102, 103])
        rows[1]["seed"] = 101
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            validate_runtime_episode_rows(
                rows,
                runtime="pytorch_cuda_fp32",
                expected_seeds=[101, 102, 103],
            )

    def test_visualization_reads_scores_and_pairs_from_formal_json(self) -> None:
        seeds = list(range(100, 130))
        per_runtime = {}
        for index, runtime in enumerate(RUNTIME_SCORE_TARGETS):
            rows = _episode_rows(runtime, seeds, offset=float(index))
            per_runtime[runtime] = {
                "runtime_metadata": {},
                "episodes": rows,
                "aggregate": aggregate_runtime_scores(
                    rows,
                    runtime=runtime,
                    expected_seeds=seeds,
                ),
            }
        rows_by_runtime = {
            runtime: per_runtime[runtime]["episodes"]
            for runtime in RUNTIME_SCORE_TARGETS
        }
        paired = {
            name: paired_score_differences(
                rows_by_runtime[left],
                rows_by_runtime[right],
                left_runtime=left,
                right_runtime=right,
                expected_seeds=seeds,
            )
            for name, (left, right) in PAIRING_DEFINITIONS.items()
        }
        payload = {
            "schema_version": 1,
            "artifact_type": "day26_runtime_score_comparison",
            "status": "completed",
            "seed_manifest": {
                "episode_seeds": seeds,
                "count": len(seeds),
                "unique": True,
                "selection_rule": "predeclared_before_runtime_results",
            },
            "per_runtime": per_runtime,
            "paired_score_differences": paired,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "runtime-score-comparison.json"
            output = Path(temporary_directory) / "runtime-score-distribution.png"
            source.write_text(json.dumps(payload), encoding="utf-8")

            data = load_runtime_score_visualization_data(source)
            rendered = render_runtime_score_distribution(source, output)
            rendered_exists = rendered.is_file()
            rendered_size = rendered.stat().st_size

        self.assertEqual(list(data["episode_seeds"]), seeds)
        self.assertEqual(len(data["scores"]["tensorrt_cuda_fp32"]), 30)
        self.assertEqual(
            data["paired_differences"]["tensorrt_fp32_minus_pytorch_fp32"],
            [2.0] * 30,
        )
        self.assertTrue(rendered_exists)
        self.assertGreater(rendered_size, 0)


if __name__ == "__main__":
    unittest.main()
