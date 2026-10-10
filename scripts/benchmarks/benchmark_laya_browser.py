"""Compare frozen reference and fixed-input browser inference on the local GPU."""

import argparse
import json
from pathlib import Path

from breakout_rl.laya_benchmark import run_laya_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--stress", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 100 or args.rounds < 3 or args.stress < 1000:
        parser.error("acceptance run requires samples >=100, rounds >=3, stress >=1000")
    result = run_laya_benchmark(samples=args.samples, rounds=args.rounds,
                                stress=args.stress, url=args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "summary": result["summary"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
