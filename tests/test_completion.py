"""Regression tests for canonical Breakout completion detection and reporting."""

from __future__ import annotations

import unittest

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_ALE_PY_VERSION,
    BREAKOUT_COMPLETION_SOURCE,
    BREAKOUT_DIFFICULTY,
    BREAKOUT_ENVIRONMENT_ID,
    BREAKOUT_FULL_CLEAR_SCORE,
    BREAKOUT_MODE,
    BREAKOUT_ROM_SHA256,
    BreakoutCompletionDetector,
    CompletionSupport,
    inspect_breakout_completion_support,
    read_breakout_score,
)
from breakout_rl.evaluation_artifacts import (
    summary_from_episode_rows,
    validate_episode_rows,
)


def _supported_runtime() -> CompletionSupport:
    return CompletionSupport(
        supported=True,
        environment_id=BREAKOUT_ENVIRONMENT_ID,
        game="breakout",
        mode=BREAKOUT_MODE,
        difficulty=BREAKOUT_DIFFICULTY,
        ale_py_version=BREAKOUT_ALE_PY_VERSION,
        rom_sha256=BREAKOUT_ROM_SHA256,
    )


def _episode_row(
    *,
    seed: int,
    cleared: bool,
    episode_return: float,
    episode_length: int,
    outcome: str,
    clear_agent_step: int | None = None,
    clear_emulator_frame: int | None = None,
) -> dict:
    return {
        "evaluation_seed": seed,
        "episode_index": 1,
        "episode_seed": seed,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "terminated": outcome == "game_over",
        "truncated": outcome == "time_limit",
        "time_limit": outcome == "time_limit",
        "complete": outcome in {"game_over", "time_limit"},
        "stop_reason": (
            "time_limit"
            if outcome == "time_limit"
            else "terminated"
            if outcome == "game_over"
            else "incomplete"
        ),
        "completion_outcome": outcome,
        "cleared": cleared,
        "clear_agent_step": clear_agent_step,
        "clear_emulator_frame": clear_emulator_frame,
        "clear_score": BREAKOUT_FULL_CLEAR_SCORE if cleared else None,
        "lives_remaining_at_clear": 2 if cleared else None,
        "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
    }


class BreakoutCompletionDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = BreakoutCompletionDetector(_supported_runtime())

    def test_life_loss_and_first_wall_are_not_clear(self) -> None:
        state = self.detector.reset()
        self.assertFalse(state.cleared)

        for score, step, lives in ((0.0, 1, 5), (7.0, 2, 5), (432.0, 100, 4)):
            state = self.detector.observe(
                cumulative_score=score,
                ram_score=int(score),
                agent_step=step,
                emulator_frame=step * 4,
                lives_remaining=lives,
            )
            self.assertFalse(state.cleared)
            self.assertIsNone(state.clear_agent_step)

    def test_two_wall_maximum_latches_with_environment_native_timing(self) -> None:
        before_maximum = self.detector.observe(
            cumulative_score=863.0,
            ram_score=863,
            agent_step=119,
            emulator_frame=476,
            lives_remaining=3,
        )
        self.assertFalse(before_maximum.cleared)

        clear = self.detector.observe(
            cumulative_score=864.0,
            ram_score=864,
            agent_step=120,
            emulator_frame=480,
            lives_remaining=3,
        )
        self.assertTrue(clear.cleared)
        self.assertEqual(clear.clear_agent_step, 120)
        self.assertEqual(clear.clear_emulator_frame, 480)
        self.assertEqual(clear.clear_score, 864.0)
        self.assertEqual(clear.lives_remaining_at_clear, 3)
        self.assertEqual(clear.completion_detection_source, BREAKOUT_COMPLETION_SOURCE)

        after_clear = self.detector.observe(
            cumulative_score=864.0,
            ram_score=864,
            agent_step=121,
            emulator_frame=484,
            lives_remaining=2,
        )
        self.assertEqual(after_clear, clear)

    def test_game_over_and_truncation_scores_do_not_imply_a_clear(self) -> None:
        game_over = self.detector.observe(
            cumulative_score=87.0,
            ram_score=87,
            agent_step=80,
            emulator_frame=320,
            lives_remaining=0,
        )
        self.assertFalse(game_over.cleared)

        self.detector.reset()
        time_limit = self.detector.observe(
            cumulative_score=250.0,
            ram_score=250,
            agent_step=27000,
            emulator_frame=108000,
            lives_remaining=1,
        )
        self.assertFalse(time_limit.cleared)

    def test_unsupported_runtime_stays_unknown_at_score_threshold(self) -> None:
        support = CompletionSupport(
            supported=False,
            environment_id=BREAKOUT_ENVIRONMENT_ID,
            game="breakout",
            mode=BREAKOUT_MODE,
            difficulty=BREAKOUT_DIFFICULTY,
            ale_py_version=BREAKOUT_ALE_PY_VERSION,
            rom_sha256="unrecognized-rom",
            reason="ROM checksum mismatch",
        )
        detector = BreakoutCompletionDetector(support)

        state = detector.observe(
            cumulative_score=864.0,
            ram_score=864,
            agent_step=100,
            emulator_frame=400,
            lives_remaining=2,
        )

        self.assertIsNone(state.cleared)
        self.assertIsNone(state.clear_score)
        self.assertIsNone(state.completion_detection_source)

    def test_ram_and_raw_reward_disagreement_fails_closed_for_the_episode(self) -> None:
        mismatch = self.detector.observe(
            cumulative_score=10.0,
            ram_score=9,
            agent_step=1,
            emulator_frame=4,
            lives_remaining=5,
        )
        self.assertIsNone(mismatch.cleared)
        self.assertIsNone(mismatch.completion_detection_source)
        self.assertIn("disagrees", mismatch.unavailable_reason)

        later = self.detector.observe(
            cumulative_score=864.0,
            ram_score=864,
            agent_step=100,
            emulator_frame=400,
            lives_remaining=4,
        )
        self.assertIsNone(later.cleared)

    def test_ale_scoreboard_ram_fixture_emits_the_documented_maximum_reward(self) -> None:
        env = make_breakout_env(fire_reset=True)
        try:
            env.reset(seed=101)
            support = inspect_breakout_completion_support(env)
            ale = env.unwrapped.ale
            frame_origin = ale.getEpisodeFrameNumber()
            env.step(2)  # The wrapper supplies the initial FIRE serve.
            ale.setRAM(76, 0x08)
            ale.setRAM(77, 0x64)
            self.assertEqual(read_breakout_score(env), BREAKOUT_FULL_CLEAR_SCORE)

            _, reward, terminated, truncated, _ = env.step(0)
            score_from_ram = read_breakout_score(env)
            emulator_frame = ale.getEpisodeFrameNumber() - frame_origin
            lives = ale.lives()
        finally:
            env.close()

        self.assertTrue(support.supported, support.reason)
        self.assertEqual(reward, float(BREAKOUT_FULL_CLEAR_SCORE))
        self.assertFalse(terminated or truncated)
        detector = BreakoutCompletionDetector(support)
        state = detector.observe(
            cumulative_score=float(reward),
            ram_score=score_from_ram,
            agent_step=2,
            emulator_frame=emulator_frame,
            lives_remaining=lives,
        )
        self.assertTrue(state.cleared)
        self.assertEqual(state.clear_score, float(BREAKOUT_FULL_CLEAR_SCORE))

    def test_repository_environment_matches_the_audited_runtime(self) -> None:
        env = make_breakout_env(fire_reset=True)
        try:
            env.reset(seed=101)
            support = inspect_breakout_completion_support(env)
        finally:
            env.close()

        self.assertTrue(support.supported, support.reason)
        self.assertEqual(support.mode, 0)
        self.assertEqual(support.difficulty, 0)
        self.assertEqual(support.rom_sha256, BREAKOUT_ROM_SHA256)


class CompletionSummaryTests(unittest.TestCase):
    def test_zero_clear_evaluation_has_zero_rate_and_no_time_percentiles(self) -> None:
        summary = summary_from_episode_rows(
            [
                _episode_row(
                    seed=101,
                    cleared=False,
                    episode_return=18.0,
                    episode_length=200,
                    outcome="game_over",
                ),
                _episode_row(
                    seed=202,
                    cleared=False,
                    episode_return=50.0,
                    episode_length=27000,
                    outcome="time_limit",
                ),
            ]
        )

        self.assertTrue(summary["clear_detection_available"])
        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["clear_count"], 0)
        self.assertEqual(summary["clear_rate"], 0.0)
        self.assertEqual(summary["clear_time_sample_count"], 0)
        self.assertIsNone(summary["best_clear_steps"])
        self.assertIsNone(summary["p90_clear_emulator_frame"])
        self.assertEqual(
            summary["failure_stop_reasons"],
            {"game_over": 1, "time_limit": 1},
        )

    def test_clear_time_statistics_use_successes_and_nearest_rank_p90(self) -> None:
        summary = summary_from_episode_rows(
            [
                _episode_row(
                    seed=101,
                    cleared=True,
                    episode_return=864.0,
                    episode_length=400,
                    outcome="cleared",
                    clear_agent_step=120,
                    clear_emulator_frame=480,
                ),
                _episode_row(
                    seed=202,
                    cleared=True,
                    episode_return=864.0,
                    episode_length=500,
                    outcome="cleared",
                    clear_agent_step=200,
                    clear_emulator_frame=800,
                ),
                _episode_row(
                    seed=303,
                    cleared=False,
                    episode_return=400.0,
                    episode_length=900,
                    outcome="game_over",
                ),
            ]
        )

        self.assertEqual(summary["clear_count"], 2)
        self.assertAlmostEqual(summary["clear_rate"], 2 / 3)
        self.assertEqual(summary["clear_time_sample_count"], 2)
        self.assertEqual(summary["best_clear_steps"], 120)
        self.assertEqual(summary["median_clear_steps"], 160.0)
        self.assertEqual(summary["mean_clear_steps"], 160.0)
        self.assertEqual(summary["p90_clear_steps"], 200.0)
        self.assertEqual(summary["p90_clear_emulator_frame"], 800.0)
        self.assertEqual(summary["per_seed_clears"][0]["clear_count"], 1)

    def test_legacy_complete_does_not_become_a_clear(self) -> None:
        rows = validate_episode_rows(
            {
                "per_episode": [
                    {
                        "evaluation_seed": 101,
                        "episode_index": 1,
                        "episode_seed": 101,
                        "episode_return": 30.0,
                        "episode_length": 200,
                        "terminated": True,
                        "truncated": False,
                        "complete": True,
                    }
                ]
            },
            source="legacy-results",
        )

        self.assertTrue(rows[0]["complete"])
        self.assertIsNone(rows[0]["cleared"])
        summary = summary_from_episode_rows(rows)
        self.assertEqual(summary["clear_detection_unknown_count"], 1)
        self.assertIsNone(summary["clear_rate"])

    def test_verified_clear_rows_validate_timing_and_score(self) -> None:
        clear_row = _episode_row(
            seed=101,
            cleared=True,
            episode_return=864.0,
            episode_length=400,
            outcome="game_over",
            clear_agent_step=120,
            clear_emulator_frame=480,
        )
        clear_row["completion_outcome"] = "cleared"
        rows = validate_episode_rows(
            {"per_episode": [clear_row]},
            source="new-results",
        )
        self.assertTrue(rows[0]["cleared"])
        self.assertTrue(rows[0]["complete"])
        self.assertEqual(rows[0]["clear_agent_step"], 120)

        invalid_clear_row = _episode_row(
            seed=101,
            cleared=True,
            episode_return=864.0,
            episode_length=20,
            outcome="game_over",
            clear_agent_step=21,
            clear_emulator_frame=84,
        )
        invalid_clear_row["completion_outcome"] = "cleared"
        invalid = {
            "per_episode": [invalid_clear_row]
        }
        with self.assertRaisesRegex(ValueError, "outside episode bounds"):
            validate_episode_rows(invalid, source="invalid-results")


if __name__ == "__main__":
    unittest.main()
