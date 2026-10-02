"""Run the frozen nine-episode Issue #35 paddle-contact diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from breakout_rl.paddle_contact_probe import DEFAULT_OUTPUT_DIR, DEFAULT_PROBE_CONFIG, run_probe

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROBE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths, payload = run_probe(args.config, args.output_dir)
    print(json.dumps({"artifacts": [p.as_posix() for p in paths], "completed_episodes": len(payload["runs"]),
                      "aggregate_native_frames": payload["aggregate_native_frames"], "primary_result": payload["primary_result"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
