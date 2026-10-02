"""Attach preserved stdout/stderr from the single Issue #54 replay command."""
import argparse
import json
from pathlib import Path

from breakout_rl.issue54_q_replay_consistency import attach_log_capture_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--validation-record", type=Path, required=True)
    parser.add_argument("--formal-command-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(attach_log_capture_hashes(args.output_dir, args.stdout, args.stderr,
                                               args.validation_record,
                                               args.formal_command_wall_seconds), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
