"""Frozen event-classifier boundary tests for Issue 27."""

from __future__ import annotations

import unittest

from breakout_rl.ball_dropout_audit import classify_dropout_events


def _trace(
    *,
    event_frame: int = 20,
    dropout_start: int = 10,
    dropout_length: int = 5,
    vy: float = 1.0,
    horizon: float = 30.0,
    ambiguous: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(event_frame):
        missing = dropout_start <= frame < dropout_start + dropout_length
        rows.append(
            {
                "episode_seed": 101,
                "observation_ale_frame": frame,
                "action_result_ale_frame": frame + 1,
                "ball_directly_detected": not missing,
                "last_directly_observed_vy": vy,
                "predicted_paddle_intercept_horizon_frames": horizon,
                "wrapper_life_loss_event": frame + 1 == event_frame,
                "trace_semantics_valid": not ambiguous,
            }
        )
    return rows


class BallDropoutAuditTests(unittest.TestCase):
    def test_four_misses_do_not_associate_but_five_do(self) -> None:
        four = classify_dropout_events(_trace(dropout_length=4))
        five = classify_dropout_events(_trace(dropout_length=5))
        self.assertFalse(four["events"][0]["associated"])
        self.assertTrue(five["events"][0]["associated"])

    def test_only_directly_observed_descending_estimate_can_anchor_run(self) -> None:
        descending = classify_dropout_events(_trace(vy=0.1))
        non_descending = classify_dropout_events(_trace(vy=0.0))
        upward = classify_dropout_events(_trace(vy=-1.0))
        self.assertTrue(descending["events"][0]["associated"])
        self.assertFalse(non_descending["events"][0]["associated"])
        self.assertFalse(upward["events"][0]["associated"])

    def test_intercept_horizon_is_inclusive_at_30(self) -> None:
        at_boundary = classify_dropout_events(_trace(horizon=30.0))
        beyond_boundary = classify_dropout_events(_trace(horizon=30.0001))
        self.assertTrue(at_boundary["events"][0]["associated"])
        self.assertFalse(beyond_boundary["events"][0]["associated"])

    def test_dropout_must_begin_within_event_window(self) -> None:
        inside = classify_dropout_events(
            _trace(event_frame=40, dropout_start=10, dropout_length=5)
        )
        outside = classify_dropout_events(
            _trace(event_frame=40, dropout_start=9, dropout_length=5)
        )
        self.assertTrue(inside["events"][0]["associated"])
        self.assertFalse(outside["events"][0]["associated"])

    def test_ambiguous_event_stays_in_denominator(self) -> None:
        result = classify_dropout_events(_trace(ambiguous=True))
        self.assertEqual(result["all_wrapper_reported_life_loss_events"], 1)
        self.assertEqual(result["ambiguous_life_loss_events"], 1)
        self.assertEqual(result["associated_life_loss_fraction"], 0.0)
        self.assertEqual(result["classification"], "INCONCLUSIVE")

    def test_no_event_sample_is_inconclusive_with_undefined_fraction(self) -> None:
        trace = _trace()
        trace[-1]["wrapper_life_loss_event"] = False
        result = classify_dropout_events(trace)
        self.assertEqual(result["all_wrapper_reported_life_loss_events"], 0)
        self.assertIsNone(result["associated_life_loss_fraction"])
        self.assertEqual(result["classification"], "INCONCLUSIVE")

    def test_missing_frame_inside_event_window_is_ambiguous(self) -> None:
        trace = _trace()
        trace.pop(12)
        result = classify_dropout_events(trace)
        self.assertEqual(result["ambiguous_life_loss_events"], 1)

    def test_multiple_seed_traces_keep_episode_local_frame_boundaries(self) -> None:
        trace = _trace() + [
            {**row, "episode_seed": 202} for row in _trace()
        ]
        result = classify_dropout_events(trace)
        self.assertEqual(result["all_wrapper_reported_life_loss_events"], 2)
        self.assertEqual(
            result["associated_events_by_episode_seed"], {"101": 1, "202": 1}
        )


if __name__ == "__main__":
    unittest.main()
