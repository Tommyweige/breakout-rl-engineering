"""Generate native-derived reset seeds for the Day 30 50-seed Browser pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.generate_day29_native_noop_manifest import native_seed_config


DAY29_SEEDS = tuple(range(101, 106)) + tuple(range(202, 207)) + tuple(range(303, 323))
DAY30_NEW_SEEDS = tuple(range(323, 343))
SEEDS = DAY29_SEEDS + DAY30_NEW_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/day30/native-noop-reset-manifest.json"))
    args = parser.parse_args()
    manifest = {
        "schema_version": 3,
        "artifact_type": "day30_native_noop_reset_manifest",
        "source": "Gymnasium AtariPreprocessing reset / env.unwrapped.np_random.integers(1, noop_max+1), captured through ALE 0.12.0",
        "seed_pool": "Day 29 30 unique seeds plus new Day 30 seeds 323-342",
        "noop_max": 30,
        "seeds": {str(seed): native_seed_config(seed) for seed in SEEDS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seedCount": len(SEEDS), "newSeeds": list(DAY30_NEW_SEEDS)}))


if __name__ == "__main__":
    main()
