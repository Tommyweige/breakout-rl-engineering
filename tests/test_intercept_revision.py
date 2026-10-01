from __future__ import annotations

import unittest

from breakout_rl.vision_controller import PlayfieldBounds, predict_paddle_intercept
from scripts.analysis.analyze_intercept_revision import analyze_episode, classify


def window(gap: int = 5, *, previous: dict | None = None, stale: dict | None = None,
           reacquisition: dict | None = None) -> list[dict]:
    direct = {
        "episode_seed": 101,
        "episode_ordinal": 1,
        "ball_directly_detected": True,
        "ball_y": 100.0,
        "ball_vy": 2.0,
        "predicted_paddle_intercept_horizon_frames": 20.0,
        "predicted_intercept_x": 50.0,
    }
    rows = [{**direct, **(previous or {}), "observation_ale_frame": 10}]
    for offset in range(gap):
        rows.append({
            "episode_seed": 101,
            "episode_ordinal": 1,
            "observation_ale_frame": 11 + offset,
            "ball_directly_detected": False,
            "predicted_intercept_x": 50.0,
            "predicted_paddle_intercept_horizon_frames": 20.0,
        })
    rows[-1].update(stale or {})
    rows.append({**direct, **(reacquisition or {}), "observation_ale_frame": 11 + gap})
    return rows


class InterceptRevisionEventTests(unittest.TestCase):
    def test_complete_dropout_windows_accept_5_and_12_and_reject_4_and_13(self) -> None:
        for gap, eligible in ((4, False), (5, True), (12, True), (13, False)):
            with self.subTest(gap=gap):
                events, excluded, _ = analyze_episode(window(gap))
                self.assertEqual(len(events), int(eligible))
                self.assertEqual(len(excluded), int(not eligible))

    def test_requires_immediate_direct_rows_around_dropout(self) -> None:
        rows = window()
        rows[0]["ball_directly_detected"] = False
        self.assertEqual(len(analyze_episode(rows)[0]), 0)
        rows = window()
        rows[-1]["ball_directly_detected"] = False
        events, excluded, _ = analyze_episode(rows)
        self.assertEqual(events, [])
        self.assertIn("immediately following", excluded[0]["reason"])

    def test_preceding_direct_physics_and_horizon_boundaries(self) -> None:
        for changes in (
            {"ball_y": float("nan")},
            {"ball_vy": float("inf")},
            {"ball_vy": 0.0},
            {"predicted_paddle_intercept_horizon_frames": -0.01},
            {"predicted_paddle_intercept_horizon_frames": 30.01},
        ):
            with self.subTest(changes=changes):
                events, excluded, _ = analyze_episode(window(previous=changes))
                self.assertEqual(events, [])
                self.assertEqual(len(excluded), 1)
        for horizon in (0.0, 30.0):
            with self.subTest(horizon=horizon):
                self.assertEqual(len(analyze_episode(window(previous={"predicted_paddle_intercept_horizon_frames": horizon}))[0]), 1)

    def test_both_measured_targets_and_horizons_must_be_valid(self) -> None:
        for stale, reacquisition in (
            ({"predicted_intercept_x": float("nan")}, {}),
            ({"predicted_paddle_intercept_horizon_frames": 30.01}, {}),
            ({}, {"predicted_intercept_x": None}),
            ({}, {"predicted_paddle_intercept_horizon_frames": -1.0}),
        ):
            with self.subTest(stale=stale, reacquisition=reacquisition):
                self.assertEqual(len(analyze_episode(window(stale=stale, reacquisition=reacquisition))[0]), 0)
        events, _, _ = analyze_episode(window(
            stale={"predicted_paddle_intercept_horizon_frames": 0.0},
            reacquisition={"predicted_paddle_intercept_horizon_frames": 30.0},
        ))
        self.assertEqual(len(events), 1)

    def test_revision_uses_last_non_direct_row_and_strictly_exceeds_8px(self) -> None:
        rows = window(stale={"predicted_intercept_x": 60.0}, reacquisition={"predicted_intercept_x": 52.0})
        rows[2]["predicted_intercept_x"] = 999.0  # Earlier stale predictions do not define the metric.
        events, _, _ = analyze_episode(rows)
        self.assertEqual(events[0]["intercept_target_revision_px"], 8.0)
        self.assertFalse(events[0]["over_8px"])
        rows[-1]["predicted_intercept_x"] = 51.999
        events, _, _ = analyze_episode(rows)
        self.assertGreater(events[0]["intercept_target_revision_px"], 8.0)
        self.assertTrue(events[0]["over_8px"])

    def test_frame_integrity_issues_are_recorded(self) -> None:
        rows = window()
        rows[3]["observation_ale_frame"] = rows[2]["observation_ale_frame"]
        _, _, issues = analyze_episode(rows)
        self.assertIn("duplicate observation_ale_frame", issues)
        self.assertIn("missing or invalid observation frame", issues)


class InterceptRevisionClassificationTests(unittest.TestCase):
    @staticmethod
    def events(n: int, high: int, seeds: tuple[int, ...] = (101, 202)) -> list[dict]:
        return [
            {"episode_seed": seeds[i % len(seeds)], "over_8px": i < high}
            for i in range(n)
        ]

    def test_sample_floor_and_seed_coverage(self) -> None:
        self.assertEqual(classify(self.events(5, 5))[0], "INCONCLUSIVE")
        self.assertEqual(classify(self.events(6, 4, (101,)))[0], "INCONCLUSIVE")

    def test_promoted_rejected_and_inconclusive_boundaries(self) -> None:
        self.assertEqual(classify(self.events(6, 4)), ("PROMOTED", 2 / 3, 2))
        self.assertEqual(classify(self.events(6, 1)), ("REJECTED", 1 / 6, 2))
        self.assertEqual(classify(self.events(6, 2))[0], "INCONCLUSIVE")
        self.assertEqual(classify(self.events(6, 4), invalid=True)[0], "INCONCLUSIVE")


class InterceptRevisionSanityTests(unittest.TestCase):
    def test_constant_velocity_one_step_no_bounce_preserves_intercept_x(self) -> None:
        bounds = PlayfieldBounds(0.0, 200.0, 0.0, 200.0, 1.0)
        before = predict_paddle_intercept(
            ball_x=60.0, ball_y=80.0, ball_vx=0.75, ball_vy=2.0,
            paddle_plane_y=140.0, bounds=bounds, ball_radius=2.0,
        )
        after = predict_paddle_intercept(
            ball_x=60.75, ball_y=82.0, ball_vx=0.75, ball_vy=2.0,
            paddle_plane_y=140.0, bounds=bounds, ball_radius=2.0,
        )
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(before[0], after[0], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
