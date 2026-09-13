"""Validate the Day 21→Day 30 final-model evidence lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path("configs/eval/breakout_contract_v2.json")
FINAL_SCORE_PATH = Path("assets/day30/final-browser-score-comparison.json")
FINAL_VALIDATION_PATH = Path("assets/day30/final-validation.json")

REQUIRED_ARTIFACTS = (
    ("contract_v2", CONTRACT_PATH, "canonical RL task definition"),
    ("day21_final_holdout", Path("assets/day21/final-holdout.json"), "final model holdout evidence"),
    ("day21_final_model_metadata", Path("assets/day21/models/final_model/metadata.json"), "frozen final model metadata"),
    ("day22_onnx_model", Path("assets/day22/models/final_model/model.onnx"), "PyTorch to ONNX export"),
    ("day22_onnx_metadata", Path("assets/day22/models/final_model/model.onnx.metadata.json"), "ONNX export metadata"),
    ("day23_onnx_runtime_parity", Path("assets/day23/onnx-runtime-parity.json"), "ONNX Runtime parity"),
    ("day24_latency", Path("assets/day24/batch1-latency.json"), "native inference timing"),
    ("day25_precision_decision", Path("assets/day25/fp32-vs-fp16.json"), "FP32/FP16 decision"),
    ("day26_tensorrt_comparison", Path("assets/day26/tensorrt-comparison.json"), "TensorRT comparison"),
    ("day27_browser_wasm", Path("assets/day27/browser-wasm-validation.json"), "Browser WASM validation"),
    ("day28_browser_webgpu", Path("assets/day28/webgpu-validation.json"), "Browser WebGPU validation"),
    ("day28_browser_benchmark", Path("assets/day28/web-benchmark.json"), "Browser backend benchmark"),
    ("day29_browser_comparison", Path("assets/day29/browser-policy-score-comparison.json"), "dual Browser score comparison"),
    ("day29_environment_parity", Path("assets/day29/environment-parity.json"), "Browser environment parity"),
    ("day30_final_screenshot", Path("assets/day30/final-human-vs-rl.png"), "real final product screenshot"),
    ("day30_score_comparison", FINAL_SCORE_PATH, "at least 50-episode Browser scores"),
    ("day30_final_validation", FINAL_VALIDATION_PATH, "final validation summary"),
    ("day30_score_distribution", Path("assets/day30/final-browser-score-distribution.png"), "score distribution figure"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--production-url", default=None)
    parser.add_argument("--score-comparison", type=Path, default=FINAL_SCORE_PATH)
    parser.add_argument("--final-validation", type=Path, default=FINAL_VALIDATION_PATH)
    parser.add_argument("--output", type=Path, default=Path("assets/day30/final-artifacts.json"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "median": percentile(values, 0.5),
        "std": statistics.pstdev(values) if values else 0.0,
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(episode["episodeReturn"]) for episode in episodes]
    result = aggregate(values)
    result.update(
        {
            "uniqueSeedCount": len({episode.get("seed") for episode in episodes}),
            "successCount": sum(episode.get("runtimeError") is None for episode in episodes),
            "crashCount": sum(episode.get("runtimeError") is not None for episode in episodes),
        }
    )
    return result


def close_enough(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(float(left) - float(right)) <= tolerance


def validate_score_comparison(payload: dict[str, Any], contract_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    check("final artifact type", payload.get("artifactType") == "day30_browser_policy_score_comparison", str(payload.get("artifactType")))
    contract = payload.get("contract")
    check("Contract v2 identity", isinstance(contract, dict) and contract.get("contractId") == "day15-breakout-evaluation-v2-fire-reset", str(contract))
    check("Contract v2 path", isinstance(contract, dict) and contract.get("contractPath") == "configs/eval/breakout_contract_v2.json", str(contract.get("contractPath") if isinstance(contract, dict) else None))
    check("Contract v2 SHA256", isinstance(contract, dict) and contract.get("contractSha256") == contract_sha, f"expected {contract_sha}, got {contract.get('contractSha256') if isinstance(contract, dict) else None}")
    check("Contract fire reset", isinstance(contract, dict) and contract.get("fireReset") is True, str(contract.get("fireReset") if isinstance(contract, dict) else None))
    check("Contract life-loss semantics", isinstance(contract, dict) and contract.get("terminalOnLifeLoss") is False, str(contract.get("terminalOnLifeLoss") if isinstance(contract, dict) else None))
    check("Contract frame semantics", isinstance(contract, dict) and contract.get("frameSkip") == 4 and contract.get("frameStack") == 4, str(contract))
    check("Contract sticky action", isinstance(contract, dict) and contract.get("stickyActionProbability") == 0.25, str(contract.get("stickyActionProbability") if isinstance(contract, dict) else None))
    check("Contract raw reward", isinstance(contract, dict) and contract.get("rawRewardRule") == "sum environment rewards without clipping", str(contract.get("rawRewardRule") if isinstance(contract, dict) else None))
    check("Contract final seeds", isinstance(contract, dict) and isinstance(contract.get("finalHoldoutConcreteSeeds"), list) and len(contract["finalHoldoutConcreteSeeds"]) >= 15, str(contract.get("finalHoldoutConcreteSeeds") if isinstance(contract, dict) else None))
    check("final unique seed count", payload.get("uniqueSeedCount") == 50, str(payload.get("uniqueSeedCount")))

    results = payload.get("results")
    if not isinstance(results, dict):
        check("backend results", False, "results.wasm/results.webgpu are missing")
        return checks, {"wasm": {}, "webgpu": {}}

    strategy = payload.get("finalBackendStrategy")
    selected_backend = strategy.get("selectedForFormalEvaluation") if isinstance(strategy, dict) else "webgpu"
    reconstructed: dict[str, Any] = {}
    for backend in ("wasm", "webgpu"):
        result = results.get(backend)
        episodes = result.get("episodes") if isinstance(result, dict) else None
        episode_records = [episode for episode in episodes if isinstance(episode, dict)] if isinstance(episodes, list) else []
        returns = [float(episode["episodeReturn"]) for episode in episode_records if isinstance(episode.get("episodeReturn"), (int, float))]
        observed = result.get("aggregate") if isinstance(result, dict) else None
        expected = aggregate_episodes([episode for episode in episode_records if isinstance(episode.get("episodeReturn"), (int, float))])
        reconstructed[backend] = {"episodeCount": len(returns), "aggregate": expected}
        required_count = 50 if backend == selected_backend else 30
        check(f"{backend} episode count", len(returns) == required_count, f"{len(returns)} episodes; required exactly {required_count}")
        required_unique_seed_count = 50 if backend == selected_backend else 30
        check(f"{backend} unique seed count", isinstance(observed, dict) and observed.get("uniqueSeedCount") == required_unique_seed_count and expected["uniqueSeedCount"] == required_unique_seed_count, f"observed={observed.get('uniqueSeedCount') if isinstance(observed, dict) else None}, reconstructed={expected['uniqueSeedCount']}")
        check(f"{backend} raw score reconstruction", isinstance(observed, dict) and all(close_enough(observed.get(key), expected[key]) for key in ("count", "uniqueSeedCount", "successCount", "crashCount", "mean", "median", "std", "p10", "p90", "min", "max")), f"observed={observed}, reconstructed={expected}")
        check(f"{backend} successful episodes", isinstance(observed, dict) and observed.get("successCount") == required_count and observed.get("crashCount") == 0, str(observed.get("crashCount") if isinstance(observed, dict) else None))
        check(f"{backend} contract hash", isinstance(result, dict) and result.get("environmentContract", {}).get("sha256") == contract_sha, str(result.get("environmentContract", {}).get("sha256") if isinstance(result, dict) else None))

    check("backend strategy", isinstance(strategy, dict) and strategy.get("preferred") == "webgpu" and strategy.get("fallback") == "wasm", str(strategy))
    check("production URL recorded", isinstance(payload.get("productUrl", payload.get("pageUrl")), str) and bool(payload.get("productUrl", payload.get("pageUrl"))), str(payload.get("productUrl", payload.get("pageUrl"))))
    return checks, reconstructed


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    score_path = (root / args.score_comparison).resolve() if not args.score_comparison.is_absolute() else args.score_comparison.resolve()
    validation_path = (root / args.final_validation).resolve() if not args.final_validation.is_absolute() else args.final_validation.resolve()
    output_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    contract_path = root / CONTRACT_PATH
    contract_sha = sha256(contract_path) if contract_path.exists() else ""

    artifact_entries = []
    checks: list[dict[str, Any]] = []
    for key, relative_path, purpose in REQUIRED_ARTIFACTS:
        path = root / relative_path
        exists = path.exists() and path.is_file() and path.stat().st_size > 0
        artifact_entries.append({"key": key, "path": relative_path.as_posix(), "purpose": purpose, "exists": exists, "sha256": sha256(path) if exists else None, "sizeBytes": path.stat().st_size if exists else 0})
        checks.append({"name": f"artifact present: {key}", "passed": exists, "detail": relative_path.as_posix()})

    score_payload = read_json(score_path) if score_path.exists() else {}
    score_checks, reconstructed = validate_score_comparison(score_payload, contract_sha) if score_payload else ([{"name": "final score comparison readable", "passed": False, "detail": str(score_path)}], {})
    checks.extend(score_checks)

    validation_payload = read_json(validation_path) if validation_path.exists() else {}
    validation_contract = validation_payload.get("contract") if isinstance(validation_payload, dict) else None
    checks.append({"name": "final validation contract", "passed": isinstance(validation_contract, dict) and validation_contract.get("contractSha256") == contract_sha, "detail": str(validation_contract)})
    validation_webgpu = validation_payload.get("evaluation", {}).get("webgpu") if isinstance(validation_payload.get("evaluation"), dict) else None
    checks.append({"name": "final validation unique seed count", "passed": isinstance(validation_payload, dict) and validation_payload.get("uniqueSeedCount") == 50 and isinstance(validation_webgpu, dict) and validation_webgpu.get("count") == 50 and validation_webgpu.get("uniqueSeedCount") == 50, "detail": str({"top": validation_payload.get("uniqueSeedCount") if isinstance(validation_payload, dict) else None, "webgpu": validation_webgpu})})
    production_url = args.production_url or validation_payload.get("productionUrl") or score_payload.get("pageUrl")
    checks.append({"name": "production URL", "passed": isinstance(production_url, str) and production_url.startswith("https://"), "detail": str(production_url)})

    passed = all(item["passed"] for item in checks)
    output = {
        "schemaVersion": 1,
        "artifactType": "day30_final_artifacts",
        "generatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {"id": "day15-breakout-evaluation-v2-fire-reset", "path": CONTRACT_PATH.as_posix(), "sha256": contract_sha},
        "productionUrl": production_url,
        "requiredArtifacts": artifact_entries,
        "checks": checks,
        "passed": passed,
        "reconstructedFinalAggregates": reconstructed,
        "finalEvaluation": {
            "backend": "webgpu",
            "episodeCount": reconstructed.get("webgpu", {}).get("aggregate", {}).get("count", 0),
            "uniqueSeedCount": reconstructed.get("webgpu", {}).get("aggregate", {}).get("uniqueSeedCount", 0),
            "successCount": reconstructed.get("webgpu", {}).get("aggregate", {}).get("successCount", 0),
            "crashCount": reconstructed.get("webgpu", {}).get("aggregate", {}).get("crashCount", 0),
            "mean": reconstructed.get("webgpu", {}).get("aggregate", {}).get("mean", 0),
            "median": reconstructed.get("webgpu", {}).get("aggregate", {}).get("median", 0),
            "std": reconstructed.get("webgpu", {}).get("aggregate", {}).get("std", 0),
            "p10": reconstructed.get("webgpu", {}).get("aggregate", {}).get("p10", 0),
            "p90": reconstructed.get("webgpu", {}).get("aggregate", {}).get("p90", 0),
            "min": reconstructed.get("webgpu", {}).get("aggregate", {}).get("min", 0),
            "max": reconstructed.get("webgpu", {}).get("aggregate", {}).get("max", 0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(output, indent=2, ensure_ascii=False)}\n", encoding="utf-8")

    if validation_payload:
        validation_payload["artifactValidation"] = {"validator": "scripts/analysis/validate_final_artifacts.py", "passed": passed, "checks": checks, "finalArtifacts": output_path.relative_to(root).as_posix()}
        if production_url:
            validation_payload["productionUrl"] = production_url
        validation_path.write_text(f"{json.dumps(validation_payload, indent=2, ensure_ascii=False)}\n", encoding="utf-8")

    print(json.dumps({"passed": passed, "productionUrl": production_url, "output": output_path.as_posix(), "failedChecks": [item for item in checks if not item["passed"]]}, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
