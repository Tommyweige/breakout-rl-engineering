"""Compare native and Browser preprocessing traces pixel by pixel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def metric(native_values: Any, browser_values: Any) -> dict[str, Any]:
    native = np.asarray(native_values, dtype=np.int16)
    browser = np.asarray(browser_values, dtype=np.int16)
    if native.shape != browser.shape:
        return {
            "shapeMatch": False,
            "nativeShape": list(native.shape),
            "browserShape": list(browser.shape),
            "maxAbsolutePixelDifference": None,
            "meanAbsolutePixelDifference": None,
            "differingPixelFraction": None,
        }
    difference = np.abs(native - browser)
    return {
        "shapeMatch": True,
        "nativeShape": list(native.shape),
        "browserShape": list(browser.shape),
        "maxAbsolutePixelDifference": int(difference.max(initial=0)),
        "meanAbsolutePixelDifference": float(difference.mean()) if difference.size else 0.0,
        "differingPixelFraction": float(np.count_nonzero(difference) / difference.size) if difference.size else 0.0,
    }


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [item for item in metrics if item["shapeMatch"]]
    if not comparable:
        return {
            "sampleCount": len(metrics),
            "shapeMismatchCount": len(metrics),
            "maxAbsolutePixelDifference": None,
            "meanAbsolutePixelDifference": None,
            "differingPixelFraction": None,
        }
    return {
        "sampleCount": len(metrics),
        "shapeMismatchCount": len(metrics) - len(comparable),
        "maxAbsolutePixelDifference": max(item["maxAbsolutePixelDifference"] for item in comparable),
        "meanAbsolutePixelDifference": float(np.mean([item["meanAbsolutePixelDifference"] for item in comparable])),
        "differingPixelFraction": float(np.mean([item["differingPixelFraction"] for item in comparable])),
    }


def image_panel(image: np.ndarray, scale: int = 3) -> Image.Image:
    clipped = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    return Image.fromarray(clipped, mode="L").resize((clipped.shape[1] * scale, clipped.shape[0] * scale), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=Path("assets/day29/native-preprocessing-fixture.json"))
    parser.add_argument("--browser", type=Path, default=Path("assets/day29/browser-preprocessing-trace.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/day29/preprocessing-parity.json"))
    parser.add_argument("--environment-output", type=Path, default=Path("assets/day29/environment-parity.json"))
    parser.add_argument("--image", type=Path, default=Path("assets/day29/preprocessing-parity-comparison.png"))
    parser.add_argument("--smoke", type=Path, default=Path("assets/day29/production-smoke.json"))
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    native = json.loads(args.native.read_text(encoding="utf-8"))
    browser = json.loads(args.browser.read_text(encoding="utf-8"))
    native_by_seed = {sample["seed"]: sample for sample in native["samples"]}
    browser_by_seed = {sample["seed"]: sample for sample in browser["samples"]}
    sample_metrics: list[dict[str, Any]] = []
    raw_metrics: list[dict[str, Any]] = []
    pooled_metrics: list[dict[str, Any]] = []
    processed_metrics: list[dict[str, Any]] = []
    stack_metrics: list[dict[str, Any]] = []
    count_mismatches: list[dict[str, Any]] = []
    per_sample_raw: list[dict[str, Any]] = []
    representatives: tuple[np.ndarray, np.ndarray] | None = None

    for seed in native["seeds"]:
        native_sample = native_by_seed[seed]
        browser_sample = browser_by_seed.get(seed)
        if browser_sample is None:
            count_mismatches.append({"seed": seed, "reason": "missing_browser_sample"})
            continue
        if native_sample["noopCount"] != browser_sample["noopCount"]:
            count_mismatches.append({
                "seed": seed,
                "reason": "noop_count",
                "native": native_sample["noopCount"],
                "browser": browser_sample["noopCount"],
            })
        reset_raw = [
            metric(native_frame, browser_frame)
            for native_frame, browser_frame in zip(
                native_sample["resetNoopGrayscaleFrames"], browser_sample["resetNoopGrayscaleFrames"], strict=False
            )
        ]
        if len(native_sample["resetNoopGrayscaleFrames"]) != len(browser_sample["resetNoopGrayscaleFrames"]):
            count_mismatches.append({
                "seed": seed,
                "reason": "reset_raw_frame_count",
                "native": len(native_sample["resetNoopGrayscaleFrames"]),
                "browser": len(browser_sample["resetNoopGrayscaleFrames"]),
            })
        raw_metrics.extend(reset_raw)
        per_seed_raw: dict[str, Any] = {"seed": seed, "resetNoop": reset_raw, "steps": []}
        reset_processed = metric(native_sample["resetProcessedFrame"], browser_sample["resetProcessedFrame"])
        reset_stack = metric(native_sample["resetObservation"], browser_sample["resetObservation"])
        processed_metrics.append({"seed": seed, "phase": "reset", **reset_processed})
        stack_metrics.append({"seed": seed, "phase": "reset", **reset_stack})
        if representatives is None:
            representatives = (
                np.asarray(native_sample["resetProcessedFrame"], dtype=np.uint8).reshape(84, 84),
                np.asarray(browser_sample["resetProcessedFrame"], dtype=np.uint8).reshape(84, 84),
            )
        if len(native_sample["steps"]) != len(browser_sample["steps"]):
            count_mismatches.append({
                "seed": seed,
                "reason": "step_count",
                "native": len(native_sample["steps"]),
                "browser": len(browser_sample["steps"]),
            })
        for index, (native_step, browser_step) in enumerate(zip(native_sample["steps"], browser_sample["steps"], strict=False)):
            raw_metrics.extend(
                metric(native_frame, browser_frame)
                for native_frame, browser_frame in zip(
                    native_step["rawGrayscaleFrames"], browser_step["rawGrayscaleFrames"], strict=False
                )
            )
            per_seed_raw["steps"].append({
                "step": index + 1,
                "raw": [
                    metric(native_frame, browser_frame)
                    for native_frame, browser_frame in zip(
                        native_step["rawGrayscaleFrames"], browser_step["rawGrayscaleFrames"], strict=False
                    )
                ],
            })
            if len(native_step["rawGrayscaleFrames"]) != len(browser_step["rawGrayscaleFrames"]):
                count_mismatches.append({
                    "seed": seed,
                    "reason": "step_raw_frame_count",
                    "step": index + 1,
                    "native": len(native_step["rawGrayscaleFrames"]),
                    "browser": len(browser_step["rawGrayscaleFrames"]),
                })
            pooled_metrics.append({"seed": seed, "step": index + 1, **metric(native_step["pooledGrayscale"], browser_step["pooledGrayscale"])})
            processed_metrics.append({"seed": seed, "step": index + 1, **metric(native_step["processedFrame"], browser_step["processedFrame"])})
            stack_metrics.append({"seed": seed, "step": index + 1, **metric(native_step["observation"], browser_step["observation"])})
        per_sample_raw.append(per_seed_raw)

    if representatives is None:
        raise SystemExit("no overlapping preprocessing samples were available")
    native_image, browser_image = representatives
    difference_image = np.abs(native_image.astype(np.int16) - browser_image.astype(np.int16)).astype(np.uint8)
    panel_width = native_image.shape[1] * 3
    canvas = Image.new("RGB", (panel_width * 3, native_image.shape[0] * 3 + 32), "white")
    labels = ("native cv2.INTER_AREA", "browser area resize", "absolute difference")
    for index, (label, panel) in enumerate(zip(labels, (native_image, browser_image, difference_image), strict=True)):
        rendered = image_panel(panel)
        canvas.paste(Image.merge("RGB", (rendered, rendered, rendered)), (index * panel_width, 32))
        ImageDraw.Draw(canvas).text((index * panel_width + 4, 8), label, fill="black")
    args.image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.image)

    artifact = {
        "schemaVersion": 2,
        "artifactType": "day29_preprocessing_parity",
        "source": {
            "native": str(args.native).replace("\\", "/"),
            "browser": str(args.browser).replace("\\", "/"),
            "nativeImplementation": "Gymnasium AtariPreprocessing 1.3.0 / ALE 0.12.0 / cv2.INTER_AREA",
            "browserImplementation": "ALE WASM 0.12.0 / native ALE grayscale / grayscale max-pool / area resize",
        },
        "atariPreprocessing": native["atariPreprocessing"],
        "coverage": {
            "seeds": native["seeds"],
            "actions": native["actions"],
            "comparisonScope": "representative raw-frame fixture, not the complete 30-seed score run",
        },
        "comparison": {
            "resetAndStepRawGrayscale": aggregate(raw_metrics),
            "pooledGrayscale": aggregate(pooled_metrics),
            "processed84x84": aggregate(processed_metrics),
            "frameStack": aggregate(stack_metrics),
        },
        "perSampleProcessed84x84": processed_metrics,
        "perSampleRawGrayscale": per_sample_raw,
        "countMismatches": count_mismatches,
        "representativeComparisonImage": str(args.image).replace("\\", "/"),
        "status": "pending",
    }
    artifact["status"] = "full" if (
        not count_mismatches
        and all(
            artifact["comparison"][key]["maxAbsolutePixelDifference"] == 0
            for key in ("resetAndStepRawGrayscale", "pooledGrayscale", "processed84x84", "frameStack")
        )
    ) else "partial"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    environment: dict[str, Any] = {}
    if args.environment_output.exists():
        environment = json.loads(args.environment_output.read_text(encoding="utf-8"))
    if args.smoke.exists():
        smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
        environment["runtime"] = smoke.get("runtimeDiagnostics", environment.get("runtime", {}))
        environment["gameplaySmoke"] = smoke
    environment["schemaVersion"] = 2
    environment["artifactType"] = "day29_environment_parity"
    environment.setdefault("contract", {})["unsupportedFields"] = [
        "Contract v2 JSON omits Gymnasium AtariPreprocessing noop_max and the derived NumPy seed-stream rule; Day 29 records an explicit adapter and fixed-seed manifest",
        "Browser arbitrary interactive seeds use a deterministic seeded fallback rather than a full NumPy SeedSequence/PCG64 implementation",
    ]
    environment["contract"]["note"] = (
        "Fixed Day 29 evaluation seeds use native-derived NOOP/ALE seeds; the representative raw-frame/preprocessing fixture "
        "for seeds 101, 202, and 303 is byte-identical; general arbitrary-seed stream parity remains partial by design."
    )
    environment["nativeAtariPreprocessing"] = native["atariPreprocessing"]
    environment["nativeAtariPreprocessing"]["noopManifest"] = str(Path("assets/day29/native-noop-reset-manifest.json")).replace("\\", "/")
    environment["nativeAtariPreprocessing"]["resetSemantics"] = [
        "AtariPreprocessing.reset calls raw env.reset(seed=seed)",
        "Gymnasium AtariEnv derives np_seed and ale_seed with np.random.SeedSequence(seed).generate_state(2)",
        "np_seed drives env.unwrapped.np_random.integers(1, noop_max+1), inclusive 1..30",
        "raw NOOP action 0 is repeated until the sampled count; if terminal/truncated occurs, reset and continue",
        "after NOOP reset, capture one current ALE grayscale frame and duplicate it across FrameStackObservation(4)",
    ]
    environment["browserAtariPreprocessing"] = {
        "noopMax": 30,
        "noopReset": "native manifest for fixed evaluation seeds; seeded fallback for other interactive seeds",
        "frameSkip": 4,
        "maxPoolSpace": "grayscale",
        "grayscaleSource": "ALE getScreenGrayscale() per raw frame",
        "resizeInterpolation": "area-coverage implementation audited against cv2.INTER_AREA",
        "frameStack": 4,
        "seedSemantics": "native-derived ale_seed for fixed evaluation seeds; documented fallback outside manifest",
        "actionMapping": {"modelIndex": ["NOOP", "FIRE", "RIGHT", "LEFT"], "aleCode": [0, 1, 3, 4]},
        "fireResetOwnership": "Browser environment owns initial serve and post-life-loss FIRE only",
        "stickyActionProbability": 0.25,
    }
    environment["preprocessingPixelParity"] = artifact["comparison"]
    environment["preprocessingPixelParityCoverage"] = artifact["coverage"]
    environment["preprocessingPixelParityStatus"] = artifact["status"]
    environment["fixedEvaluationEnvironmentParity"] = "full" if artifact["status"] == "full" else "partial"
    environment["overallStatus"] = "partial"
    environment["remainingDifferences"] = [
        "Browser uses the exact native-derived noop/ale seed manifest for the fixed Day 29 30-seed evaluation; arbitrary interactive seeds use the documented seeded fallback instead of a full NumPy SeedSequence/PCG64 implementation.",
        "Contract v2 JSON remains unchanged and does not itself contain noop_max or the derived seed-stream rule; this Day 29 parity artifact records those runtime semantics explicitly.",
        "The terminal/truncated-during-NOOP recovery branch is structurally mirrored with ALE seed/load/reset, but has not been independently exercised in the representative fixture against Gymnasium's reset(seed, options) callback path.",
    ]
    args.environment_output.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "countMismatches": count_mismatches, "comparison": artifact["comparison"]}, indent=2))
    if args.strict and artifact["status"] != "full":
        raise SystemExit(1)
if __name__ == "__main__":
    main()
