from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from pathlib import Path
from unittest.mock import patch

import numpy as np

from breakout_rl.issue54_q_replay_consistency import (
    STACK_SHA256, classify_diagnostic, match_rows, max_q_error, prepare_model_input,
    aggregate_q_error, attach_log_capture_hashes, create_cpu_session, remaining_budget,
    validate_capture_offsets, verify_stack_row,
)
from scripts.analysis.run_issue54_once import run_captured_command


def record(step: int = 1, digest: str = "abc") -> tuple[dict, dict]:
    key = {"seed": 105, "arm": "baseline", "episode_index": 1,
           "agent_step": step, "emulator_frame": step * 4 + 1}
    return ({**key, "observation_sha256": digest},
            {**key, "observation_sha256": digest,
             "q_values": [1.0, 2.0, 3.0, 4.0]})


class Issue54PureTests(unittest.TestCase):
    def test_formal_supervisor_timeout_kills_group_and_preserves_partial_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout, stderr = root / "stdout", root / "stderr"
            command = [sys.executable, "-c",
                       "import sys,time; print('started', flush=True); print('err', file=sys.stderr, flush=True); time.sleep(3)"]
            return_code, timed_out, _ = run_captured_command(
                command, root, stdout, stderr, timeout_seconds=0.1)
            self.assertTrue(timed_out)
            self.assertNotEqual(return_code, 0)
            self.assertIn(b"started", stdout.read_bytes())
            self.assertIn(b"err", stderr.read_bytes())

    def test_unique_exact_join_and_logged_q_integrity(self):
        pairs = [record(1), record(2)]
        joined = match_rows([x[0] for x in pairs] + [record(i)[0] for i in range(3, 301)],
                            [x[1] for x in pairs] + [record(i)[1] for i in range(3, 301)])
        self.assertEqual(len(joined), 300)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            match_rows([record()[0]] + [record(i)[0] for i in range(2, 301)],
                       [record()[1], record()[1]] + [record(i)[1] for i in range(2, 301)])
        bad = record()
        bad[1]["q_values"] = [1.0, 2.0, float("nan"), 4.0]
        with self.assertRaisesRegex(ValueError, "four finite"):
            match_rows([bad[0]] + [record(i)[0] for i in range(2, 301)],
                       [bad[1]] + [record(i)[1] for i in range(2, 301)])

    def test_index_stack_hash_rejection(self):
        stream = bytes(range(256)) * (1_411_200 // 256) + bytes(range(1_411_200 % 256))
        sample = stream[:28_224]
        row = {"seed": 105, "arm": "baseline", "episode_index": 1, "agent_step": 402,
               "emulator_frame": 1609, "file": "decision_stacks/seed105_baseline_ep1.uint8",
               "byte_offset": 0, "bytes": 28_224, "dtype": "uint8", "shape": [4, 84, 84],
               "sha256": hashlib.sha256(sample).hexdigest(),
               "observation_sha256": hashlib.sha256(sample).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "decision_stacks").mkdir()
            (root / "decision_stacks" / "seed105_baseline_ep1.uint8").write_bytes(stream)
            with patch.dict(STACK_SHA256, {"seed105_baseline_ep1.uint8": hashlib.sha256(stream).hexdigest()}):
                self.assertEqual(verify_stack_row(root, row)[1], row["sha256"])
                row["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "observation hash"):
                    verify_stack_row(root, row)
                row["sha256"] = hashlib.sha256(sample).hexdigest()
                row["byte_offset"] = 1
                with self.assertRaisesRegex(ValueError, "invalid indexed stack metadata"):
                    verify_stack_row(root, row)

    def test_sparse_capture_steps_have_contiguous_file_offsets(self):
        first = {"file": "decision_stacks/seed105_baseline_ep1.uint8", "agent_step": 402,
                 "byte_offset": 0}
        second = {"file": first["file"], "agent_step": 407, "byte_offset": 28_224}
        validate_capture_offsets([first, second], complete=False)
        second["byte_offset"] = 402 * 28_224
        with self.assertRaisesRegex(ValueError, "capture offset"):
            validate_capture_offsets([first, second], complete=False)

    def test_uint8_stack_is_normalized_once_to_nchw_float32(self):
        stack = np.full((4, 84, 84), 128, dtype=np.uint8)
        prepared = prepare_model_input(stack)
        self.assertEqual(prepared.shape, (1, 4, 84, 84))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertAlmostEqual(float(prepared[0, 0, 0, 0]), 128.0 / 255.0, places=7)
        with self.assertRaises(ValueError):
            prepare_model_input(prepared)

    def test_cpu_runtime_uses_issue50_default_session_options(self):
        class FakeOrt:
            calls = []
            @classmethod
            def InferenceSession(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return object()
        create_cpu_session(FakeOrt, Path("model.onnx"))
        self.assertEqual(FakeOrt.calls, [(('model.onnx',), {"providers": ["CPUExecutionProvider"]})])

    def test_q_error_is_one_max_absolute_vector_metric(self):
        self.assertEqual(max_q_error([0, 1, 2, 3], [0.25, 0.5, 2, 3]), 0.5)
        self.assertEqual(aggregate_q_error([0.1] * 299 + [0.5]), 0.5)
        with self.assertRaises(ValueError):
            aggregate_q_error([0.1] * 299)
        with self.assertRaises(ValueError):
            max_q_error([1, 2, 3], [1, 2, 3])

    def test_three_diagnostic_classifications(self):
        self.assertEqual(classify_diagnostic(True, 1e-6), "PROMOTED")
        self.assertEqual(classify_diagnostic(True, 1.000001e-6), "REJECTED")
        self.assertEqual(classify_diagnostic(False, 0.0), "INCONCLUSIVE")
        self.assertEqual(classify_diagnostic(True, None), "INCONCLUSIVE")

    def test_pre_run_validation_reserves_combined_cap(self):
        self.assertEqual(remaining_budget(1.25), 18.75)
        with self.assertRaises(TimeoutError):
            remaining_budget(20.0)

    def test_post_run_capture_attachment_preserves_and_hashes_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts"
            output.mkdir()
            (output / "q_comparisons.json").write_text("[]\n")
            (output / "results.json").write_text("{}\n")
            (output / "report.md").write_text("report\n")
            (output / "manifest.json").write_text(json.dumps({"output_hashes": {}}))
            stdout, stderr = root / "stdout", root / "stderr"
            stdout.write_bytes(b"full stdout\n")
            stderr.write_bytes(b"full stderr\n")
            captures = attach_log_capture_hashes(output, stdout, stderr)
            self.assertEqual((output / "formal.stdout").read_bytes(), stdout.read_bytes())
            self.assertEqual((output / "formal.stderr").read_bytes(), stderr.read_bytes())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["output_hashes"]["formal.stdout"], captures["stdout_sha256"])
            self.assertEqual(manifest["output_hashes"]["formal.stderr"], captures["stderr_sha256"])
            self.assertTrue(manifest["manifest_self_hash_excluded"])


if __name__ == "__main__":
    unittest.main()
