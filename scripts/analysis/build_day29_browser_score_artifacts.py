"""Build the checked-in Day 29 score/timing artifacts from live Browser JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OVERLAP_SEEDS = tuple(range(101, 106)) + tuple(range(202, 207)) + tuple(range(303, 308))


def native_scores(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    differences = payload["paired_score_differences"]["onnx_fp32_minus_pytorch_fp32"]["differences"]
    return {int(item["seed"]): float(item["left_score"]) for item in differences}


def browser_scores(result: dict[str, Any]) -> dict[int, float]:
    return {int(episode["seed"]): float(episode["episodeReturn"]) for episode in result["episodes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-input", type=Path, default=Path("assets/day29/browser-overlap-evaluation.json"))
    parser.add_argument("--native-input", type=Path, default=Path("assets/day26/runtime-score-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/day29/browser-policy-score-comparison.json"))
    parser.add_argument("--timing-output", type=Path, default=Path("assets/day29/agent-loop-timing.json"))
    parser.add_argument("--page-url", default="http://127.0.0.1:5174/")
    parser.add_argument("--smoke", type=Path, default=Path("assets/day29/production-smoke.json"))
    args = parser.parse_args()

    source = json.loads(args.browser_input.read_text(encoding="utf-8"))
    native = native_scores(args.native_input)
    wasm = source["wasm"]
    webgpu = source["webgpu"]
    wasm_by_seed = browser_scores(wasm)
    webgpu_by_seed = browser_scores(webgpu)
    paired = []
    for seed in wasm_by_seed:
        paired.append({
            "seed": seed,
            "native": native.get(seed),
            "wasm": wasm_by_seed[seed],
            "webgpu": webgpu_by_seed.get(seed),
            "wasmMinusNative": None if seed not in native else wasm_by_seed[seed] - native[seed],
            "webgpuMinusNative": None if seed not in native or seed not in webgpu_by_seed else webgpu_by_seed[seed] - native[seed],
            "webgpuMinusWasm": None if seed not in webgpu_by_seed else webgpu_by_seed[seed] - wasm_by_seed[seed],
        })
    overlap = [item for item in paired if item["seed"] in OVERLAP_SEEDS]
    smoke = json.loads(args.smoke.read_text(encoding="utf-8")) if args.smoke.exists() else {
        "backend": "webgpu",
        "completedEpisode": False,
        "state": {},
        "productValidation": {},
        "runtimeDiagnostics": {},
    }
    environment_contract = dict(webgpu["environmentContract"])
    environment_contract["status"] = "partial"
    environment_contract["unsupportedFields"] = [
        "Contract v2 JSON omits Gymnasium AtariPreprocessing noop_max and the derived NumPy seed-stream rule; Day 29 records an explicit adapter and fixed-seed manifest",
        "Browser arbitrary interactive seeds use a deterministic seeded fallback rather than a full NumPy SeedSequence/PCG64 implementation",
    ]
    environment_contract["note"] = (
        "Fixed Day 29 evaluation seeds use native-derived NOOP/ALE seed manifests; the representative raw-frame fixture "
        "for seeds 101, 202, and 303 is byte-identical; arbitrary interactive seeds still use a documented seeded "
        "fallback, so general seed-stream parity remains partial."
    )
    artifact = {
        "schemaVersion": 2,
        "artifactType": "day29_browser_policy_score_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pageUrl": args.page_url,
        "browser": webgpu["browser"],
        "modelSha256": webgpu["modelSha256"],
        "environmentContract": environment_contract,
        "validation": source.get("validation", {}),
        "results": {"wasm": wasm, "webgpu": webgpu},
        "pairedScoreDifference": {
            "count": len([item for item in paired if item["webgpuMinusWasm"] is not None]),
            "mean": _mean([item["webgpuMinusWasm"] for item in paired if item["webgpuMinusWasm"] is not None]),
            "bySeed": [{
                "seed": item["seed"],
                "webgpu": item["webgpu"],
                "wasm": item["wasm"],
                "difference": item["webgpuMinusWasm"],
            } for item in paired],
        },
        "nativeComparison": {
            "runtime": "onnx_fp32",
            "source": str(args.native_input).replace("\\", "/"),
            "overlapSeeds": list(OVERLAP_SEEDS),
            "overlap": overlap,
            "meanNative": _mean([item["native"] for item in overlap]),
            "meanWasm": _mean([item["wasm"] for item in overlap]),
            "meanWebgpu": _mean([item["webgpu"] for item in overlap]),
        },
        "gameplaySmoke": smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    timing = {
        "schemaVersion": 1,
        "artifactType": "day29_agent_loop_timing",
        "modelSha256": webgpu["modelSha256"],
        "browser": webgpu["browser"],
        "backends": {"wasm": wasm["timing"], "webgpu": webgpu["timing"]},
    }
    args.timing_output.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "timingOutput": str(args.timing_output), "overlap": overlap}, indent=2))


def _mean(values: list[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


if __name__ == "__main__":
    main()
