"""Tests for Day 26 multi-episode score evidence and visualization seams."""

from __future__ import annotations

import unittest

from breakout_rl.runtime_scores import (
    aggregate_runtime_scores,
    paired_score_differences,
    score_statistics,
    validate_runtime_episode_rows,
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


if __name__ == "__main__":
    unittest.main()
