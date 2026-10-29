#!/usr/bin/env python3
"""Reproduce the historical "137.47" RGB-controller record.

The figure quoted for the #25/#26 pixel controller (mean 137.47, median 95,
max 341, 0 canonical clears over 15 seeds) comes from the controller as it was
before the #67/#68 paddle-control fix. This script rebuilds that exact
controller from git history, runs it through the current evaluation pipeline
with the repository's 15 contract seeds, and prints the same statistics.

It does not modify the repository. Requires the evaluation environment:
    ale-py==0.12.0, gymnasium==1.3.0, opencv-python (or opencv-python-headless)

Usage (from the repository root):
    python audits/browser-clear/reproduce-137-47.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_CONTROLLER_COMMIT = "50ee352"  # "feat: add pixel-based predictive Breakout controller (#26)"
CONFIG = REPO_ROOT / "configs" / "eval" / "breakout_vision_controller_v1.json"


def load_old_controller():
    source = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{OLD_CONTROLLER_COMMIT}:breakout_rl/vision_controller.py"],
        text=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "old_vision_controller.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("old_vision_controller", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the historical controller module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve their module by name
        spec.loader.exec_module(module)
        return module.PredictiveBreakoutController


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    import breakout_rl.vision_evaluation as evaluation

    protocol = evaluation.load_vision_evaluation_protocol(CONFIG)
    evaluation.PredictiveBreakoutController = load_old_controller()
    subset = replace(protocol, trace_episodes=0)
    payload = evaluation.evaluate_predictive_controller(subset)
    rows = payload["episodes"]
    scores = [row["raw_score"] for row in rows]
    summary = {
        "controller_commit": OLD_CONTROLLER_COMMIT,
        "seeds": list(protocol.seeds),
        "episodes": len(rows),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "min": min(scores),
        "canonical_clears": sum(1 for row in rows if row["cleared"]),
        "scores": scores,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
