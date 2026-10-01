import unittest

from scripts.analysis.analyze_reacquisition_drift import analyze_episode, classify


def episode(gap, *, vy=2.0, horizon=10.0, reacquired_y=None):
    rows = [{"observation_ale_frame": 10, "episode_seed": 101, "ball_directly_detected": True,
             "ball_y": 100.0, "ball_vy": vy, "predicted_paddle_intercept_horizon_frames": horizon}]
    rows += [{"observation_ale_frame": 11 + i, "episode_seed": 101, "ball_directly_detected": False,
              "ball_y": 100 + 2 * (i + 1), "ball_vy": vy} for i in range(gap)]
    reacq_frame = 11 + gap
    rows.append({"observation_ale_frame": reacq_frame, "episode_seed": 101, "ball_directly_detected": True,
                 "ball_y": 100 + 2 * (gap + 1) if reacquired_y is None else reacquired_y})
    return rows


class ReacquisitionDriftTests(unittest.TestCase):
    def test_contiguous_dropout_reacquisition_and_boundaries(self):
        events, excluded, _ = analyze_episode(episode(5))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gap_length"], 5)
        self.assertEqual(events[0]["frame_delta"], 6)
        for n in (4, 13):
            events, excluded, _ = analyze_episode(episode(n))
            self.assertEqual(events, [])
            self.assertEqual(excluded[0]["reason"], "gap length outside 5-12")
        self.assertEqual(len(analyze_episode(episode(12))[0]), 1)

    def test_descending_and_horizon_eligibility(self):
        for opts in ({"vy": 0}, {"vy": -1}, {"horizon": -0.1}, {"horizon": 30.1}):
            self.assertEqual(analyze_episode(episode(5, **opts))[0], [])
        self.assertEqual(len(analyze_episode(episode(5, horizon=30))[0]), 1)

    def test_exact_frame_delta_forecast_error_and_strict_threshold(self):
        self.assertEqual(analyze_episode(episode(5))[0][0]["vertical_error_px"], 0.0)
        e8 = analyze_episode(episode(5, reacquired_y=120.0))[0][0]
        self.assertEqual(e8["vertical_error_px"], 8.0)
        self.assertFalse(e8["over_8px"])
        e9 = analyze_episode(episode(5, reacquired_y=121.0))[0][0]
        self.assertEqual(e9["vertical_error_px"], 9.0)
        self.assertTrue(e9["over_8px"])

    def test_invalid_forecast_and_frame_integrity_are_retained(self):
        rows = episode(5, vy=float("nan"))
        events, excluded, issues = analyze_episode(rows)
        self.assertEqual(events, [])
        self.assertTrue(issues)
        self.assertIn("invalid preceding direct y/vy", excluded[0]["reason"])
        missing = episode(5)
        missing.pop(2)
        _, _, issues = analyze_episode(missing)
        self.assertTrue(any("missing or invalid observation frame" in x for x in issues))

    def test_denominator_sample_floor_and_classification_boundaries(self):
        def events(n, high, seeds=(101, 202)):
            return [{"episode_seed": seeds[i % len(seeds)], "over_8px": i < high} for i in range(n)]
        self.assertEqual(classify(events(5, 5), [101, 202]), ("INCONCLUSIVE", 1.0))
        self.assertEqual(classify(events(6, 4), [101, 202])[0], "PROMOTED")
        self.assertEqual(classify(events(6, 1), [101, 202])[0], "REJECTED")
        self.assertEqual(classify(events(6, 2), [101, 202])[0], "INCONCLUSIVE")
        self.assertEqual(classify(events(6, 6, (101,)), [101])[0], "INCONCLUSIVE")
        self.assertEqual(classify(events(6, 6), [101, 202], invalid=True)[0], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
