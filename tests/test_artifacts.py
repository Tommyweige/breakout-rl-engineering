"""Tests for shared repository artifact helpers."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file


class ArtifactHelperTests(unittest.TestCase):
    def test_repository_root_and_relative_path_use_the_current_repository(self) -> None:
        root = repository_root()

        self.assertTrue((root / "AGENTS.md").is_file())
        self.assertEqual(
            repository_relative_path(root / "AGENTS.md", root=root),
            "AGENTS.md",
        )

    def test_sha256_file_can_hash_raw_and_lf_normalized_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_bytes(b"one\r\ntwo\r\n")

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(b"one\r\ntwo\r\n").hexdigest(),
            )
            self.assertEqual(
                sha256_file(path, normalize_text=True),
                hashlib.sha256(b"one\ntwo\n").hexdigest(),
            )

    def test_relative_path_rejects_files_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                repository_relative_path(Path(directory) / "external.txt", root=repository_root())


if __name__ == "__main__":
    unittest.main()
