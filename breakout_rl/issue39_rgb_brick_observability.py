"""Frozen Issue #39 RGB brick removal observability diagnostic."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium
import numpy as np

from breakout_env import ENVIRONMENT_ID
from breakout_rl.completion import (
    BreakoutCompletionDetector,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.evaluation_contract import load_evaluation_contract, validate_breakout_runtime_contract
from breakout_rl.vision_controller import PredictiveBreakoutController
from breakout_rl.vision_evaluation import make_vision_breakout_env

DEFAULT_CONFIG = Path("configs/eval/issue39_rgb_brick_observability_v1.json")
DEFAULT_OUTPUT_DIR = Path("outputs/issue-39-rgb-brick-observability")
EXPECTED_SEEDS = (707, 808, 909)
PREDECLARED_TEST_NATIVE_FRAMES = 350
TOTAL_NATIVE_FRAME_CEILING = 15350
TOTAL_WALL_SECONDS_LIMIT = 600
FINALIZATION_WALL_RESERVE_SECONDS = 2.0
GRID_COLUMNS = 18
GRID_ROWS = 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProbeConfig:
    config_path: Path
    contract_path: Path
    controller_config_path: Path
    seeds: tuple[int, ...]
    max_frames_per_episode: int
    max_frames_total: int
    total_wall_seconds: int
    controller_options: Mapping[str, Any]


def load_config(path: str | Path = DEFAULT_CONFIG) -> ProbeConfig:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("probe_id") != "issue-39-rgb-brick-observability-v1":
        raise ValueError("unsupported Issue #39 config")
    contract_path = (path.parent / data["base_environment_contract"]).resolve()
    controller_path = (path.parent / data["base_controller_config"]).resolve()
    if sha256(contract_path) != data["frozen_contract_sha256"]:
        raise ValueError("frozen Contract v3 hash mismatch")
    if sha256(controller_path) != data["frozen_controller_config_sha256"]:
        raise ValueError("frozen controller config hash mismatch")
    controller_json = json.loads(controller_path.read_text(encoding="utf-8"))
    options = controller_json["controller"]
    controller_options = {"deadband": float(options["deadband_pixels"]),
                          "hysteresis": float(options["hysteresis_pixels"]),
                          "max_missing_frames": int(options["max_missing_frames"]),
                          "max_ball_speed_pixels_per_frame": float(options["max_ball_speed_pixels_per_frame"])}
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    if contract.contract_id != "breakout-evaluation-v3-frame-skip-1" or contract.frame_skip != 1:
        raise ValueError("Issue #39 requires unchanged Contract v3 semantics")
    if tuple(data["seeds"]) != EXPECTED_SEEDS or data["episodes"] != 1:
        raise ValueError("frozen Issue #39 episode seeds/count changed")
    if data["max_native_frames_per_episode"] != 5000 or data["max_native_frames_total"] != 15000:
        raise ValueError("frozen Issue #39 frame caps changed")
    if data["predeclared_test_native_frames_reserved"] != PREDECLARED_TEST_NATIVE_FRAMES or data["total_native_frames_ceiling"] != TOTAL_NATIVE_FRAME_CEILING:
        raise ValueError("frozen Issue #39 combined frame ceiling changed")
    if data["total_wall_seconds_limit"] != TOTAL_WALL_SECONDS_LIMIT:
        raise ValueError("frozen Issue #39 wall-clock cap changed")
    return ProbeConfig(path, contract_path, controller_path, EXPECTED_SEEDS, 5000, 15000, 600,
                       controller_options)


def cell_occupancy(rgb_crop: np.ndarray) -> tuple[bool, ...]:
    """Apply the frozen 18x6, >=4 bright-pixel cell rule to one raw RGB crop."""
    crop = np.asarray(rgb_crop)
    if crop.shape != (48, 144, 3) or crop.dtype != np.uint8:
        raise ValueError("RGB brick crop must be uint8 with shape (48, 144, 3)")
    occupied = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLUMNS):
            cell = crop[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
            bright = np.max(cell, axis=2) > 24
            occupied.append(int(np.count_nonzero(bright)) >= 4)
    return tuple(occupied)


class BrickRemovalDetector:
    """Stateful fixed detector; consumes captured RGB crops only."""
    def __init__(self) -> None:
        self.previous: tuple[bool, ...] | None = None
        self.pending: dict[int, int] = {}
        self.emitted: set[int] = set()

    def observe(self, frame_index: int, rgb_crop: np.ndarray) -> list[dict[str, int]]:
        current = cell_occupancy(rgb_crop)
        if self.previous is None:
            self.previous = current
            return []
        for cell, (was_occupied, is_occupied) in enumerate(zip(self.previous, current)):
            if was_occupied and not is_occupied:
                self.pending[cell] = frame_index
            elif is_occupied:
                self.pending.pop(cell, None)
        confirmed = []
        for cell, first_empty in list(self.pending.items()):
            if frame_index - first_empty >= 1 and not current[cell]:
                confirmed.append(cell)
                self.emitted.add(cell)
                del self.pending[cell]
        self.previous = current
        # The issue freezes one event maximum per native frame.
        return [{"frame": int(frame_index), "cell": min(confirmed)}] if confirmed else []


def match_events(predicted: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], tolerance: int = 2) -> dict[str, Any]:
    """Maximum-cardinality matching for timestamp events with a symmetric window."""
    predictions = sorted((int(x["frame"]) for x in predicted))
    truth = sorted((int(x["frame"]) for x in labels))
    pairs = []
    prediction_index = label_index = 0
    while prediction_index < len(predictions) and label_index < len(truth):
        frame, target = predictions[prediction_index], truth[label_index]
        if frame < target - tolerance:
            prediction_index += 1
        elif target < frame - tolerance:
            label_index += 1
        else:
            pairs.append({"predicted_frame": frame, "label_frame": target})
            prediction_index += 1
            label_index += 1
    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(truth) - tp
    denom = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": (2 * tp / denom if denom else 0.0), "matches": pairs}


def aggregate_episode_metrics(per_episode: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum already-isolated episode metrics so timestamps cannot cross-match seeds."""
    tp = sum(int(item["metrics"]["tp"]) for item in per_episode)
    fp = sum(int(item["metrics"]["fp"]) for item in per_episode)
    fn = sum(int(item["metrics"]["fn"]) for item in per_episode)
    denominator = 2 * tp + fp + fn
    matches = [{"episode_seed": item["seed"], **pair}
               for item in per_episode for pair in item["metrics"]["matches"]]
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": (2 * tp / denominator if denominator else 0.0), "matches": matches}


def classify(metrics: Mapping[str, Any], label_count: int, episodes_with_labels: int,
             labels_consistent: bool, completed_episodes: int, expected_episodes: int = 3,
             all_episodes_complete: bool = True) -> str:
    if (not labels_consistent or label_count < 12 or episodes_with_labels < 2
            or completed_episodes != expected_episodes or not all_episodes_complete):
        return "INCONCLUSIVE"
    f1 = float(metrics["f1"])
    if f1 >= 0.90:
        return "PROMOTED"
    if f1 <= 0.50:
        return "REJECTED"
    return "INCONCLUSIVE"


def bounded_formal_wall_budget(pre_run_wall_seconds: float,
                               requested_wall_seconds: float = TOTAL_WALL_SECONDS_LIMIT) -> float:
    """Subtract measured pre-run time and reserve room for final artifact writes."""
    return min(float(requested_wall_seconds), max(0.0, TOTAL_WALL_SECONDS_LIMIT
               - float(pre_run_wall_seconds) - FINALIZATION_WALL_RESERVE_SECONDS))


def _crop(obs: np.ndarray) -> np.ndarray:
    image = np.asarray(obs)
    if image.shape != (210, 160, 3) or image.dtype != np.uint8:
        raise RuntimeError("Contract v3 observation must be uint8 raw RGB 210x160x3")
    return np.ascontiguousarray(image[32:80, 8:152, :])


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment_versions() -> dict[str, Any]:
    try:
        ale_version = version("ale-py")
    except PackageNotFoundError:
        ale_version = None
    return {"python": platform.python_version(), "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "ale_py": ale_version}


def run_probe(config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path = DEFAULT_OUTPUT_DIR,
              *, env_factory=None, wall_budget_seconds: float = TOTAL_WALL_SECONDS_LIMIT,
              pre_run_wall_seconds: float = 0.0) -> tuple[list[Path], dict[str, Any]]:
    started = time.perf_counter()
    cfg = load_config(config_path)
    root = cfg.config_path.parents[2]
    contract = load_evaluation_contract(cfg.contract_path)
    budget = min(float(wall_budget_seconds), bounded_formal_wall_budget(pre_run_wall_seconds,
                                                                         wall_budget_seconds))
    deadline = started + budget
    env = (env_factory or (lambda: make_vision_breakout_env(contract)))()
    support = inspect_breakout_completion_support(env)
    if not support.supported:
        env.close()
        raise RuntimeError(f"canonical completion support unavailable: {support.reason}")
    base = getattr(env, "unwrapped", env)
    names = tuple(str(name) for name in base.get_action_meanings())
    indices = {name: i for i, name in enumerate(names)}
    if not {"NOOP", "FIRE", "LEFT", "RIGHT"}.issubset(indices):
        env.close()
        raise RuntimeError(f"unsupported action set {names}")
    runs: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    detector_outputs: list[dict[str, int]] = []
    labels: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    total_frames = 0
    label_consistent = True
    completion = BreakoutCompletionDetector(support)
    try:
        for seed in cfg.seeds:
            if time.perf_counter() >= deadline or total_frames + PREDECLARED_TEST_NATIVE_FRAMES >= cfg.max_frames_total:
                break
            obs, _ = env.reset(seed=seed)
            controller = PredictiveBreakoutController(**cfg.controller_options)
            completion.reset()
            origin = read_ale_episode_frame(env)
            if origin is None:
                raise RuntimeError("ALE native episode frame counter unavailable")
            start = len(crops)
            crop = _crop(obs)
            crops.append(crop)
            # The existing completion evaluator may read score/RAM for canonical clear stopping.
            # Offline label construction remains deferred until every action has been selected.
            raw_return = 0.0
            rows: list[dict[str, Any]] = []
            initial_score = read_breakout_score(env)
            stop_reason = "episode_frame_cap"
            for step in range(1, cfg.max_frames_per_episode + 1):
                if time.perf_counter() >= deadline:
                    stop_reason = "wall_clock_cap"
                    break
                # Controller receives only observation pixels; labels never enter this call.
                decision = controller.select_action(obs)
                requested = decision.action
                obs, reward, terminated, truncated, info = env.step(indices[requested])
                info = info if isinstance(info, Mapping) else {}
                ale_action_idx = int(info.get("fire_reset_executed_action", indices[requested]))
                frame = read_ale_episode_frame(env)
                if frame is None or frame - origin != step:
                    raise RuntimeError("expected one ALE-native frame per requested action")
                crop = _crop(obs)
                crops.append(crop)
                raw_return += float(reward)
                total_frames += 1
                score = read_breakout_score(env)
                lives = read_ale_lives(env)
                completion.observe(cumulative_score=raw_return, ram_score=score, agent_step=step,
                                   emulator_frame=step, lives_remaining=lives)
                rows.append({"episode_seed": seed, "frame": step, "requested_action": requested,
                             "ale_input_action": names[ale_action_idx], "raw_reward": float(reward),
                             "score_reading_for_completion_and_offline_label": score,
                             "lives_remaining_offline": lives, "canonical_clear_offline": completion.state.cleared})
                if completion.state.cleared:
                    stop_reason = "canonical_clear"
                    break
                if terminated or truncated:
                    stop_reason = "game_terminated" if terminated else "game_truncated"
                    break
            runs.append({"seed": seed, "capture_start_index": start, "capture_end_index": len(crops),
                         "episode_frames": len(rows), "raw_score_offline": raw_return,
                         "initial_score_reading_for_completion_and_offline_label": initial_score,
                         "canonical_clear_offline": completion.state.cleared,
                         "stop_reason": stop_reason, "trace": rows})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    # Both the detector and reward/RAM label construction run after all online actions finish.
    for run in runs:
        previous_score = run["initial_score_reading_for_completion_and_offline_label"]
        for row in run["trace"]:
            score = row.pop("score_reading_for_completion_and_offline_label")
            score_delta = None if score is None or previous_score is None else score - previous_score
            agrees = score_delta is not None and np.isclose(float(row["raw_reward"]), score_delta, rtol=0, atol=1e-6)
            row["score_delta_offline"] = score_delta
            row["label_agrees_offline"] = bool(agrees)
            if not agrees:
                label_consistent = False
            if row["raw_reward"] > 0 or (score_delta is not None and score_delta > 0):
                labels.append({"episode_seed": run["seed"], "frame": row["frame"],
                               "raw_reward": float(row["raw_reward"]), "score_delta": score_delta,
                               "agrees": bool(agrees)})
            previous_score = score
    for run in runs:
        detector = BrickRemovalDetector()
        start, end = run["capture_start_index"], run["capture_end_index"]
        for local_frame, crop in enumerate(crops[start:end]):
            detector_outputs.extend({"episode_seed": run["seed"], **event} for event in detector.observe(local_frame, crop))
    per_episode = []
    for run in runs:
        seed = run["seed"]
        pred = [x for x in detector_outputs if x["episode_seed"] == seed]
        truth = [x for x in labels if x["episode_seed"] == seed]
        per_episode.append({"seed": seed, "predicted_events": pred, "label_events": truth,
                            "metrics": match_events(pred, truth)})
    all_truth = [event for result in per_episode for event in result["label_events"]]
    metrics = aggregate_episode_metrics(per_episode)
    label_count = len(all_truth)
    episodes_with_labels = sum(bool(x["label_events"]) for x in per_episode)
    classification = classify(metrics, label_count, episodes_with_labels, label_consistent,
                               completed_episodes=len(runs), expected_episodes=len(cfg.seeds),
                               all_episodes_complete=all(run["stop_reason"] != "wall_clock_cap" for run in runs))
    command = ("python -m scripts.evaluation.run_issue39_rgb_brick_observability "
               "--config configs/eval/issue39_rgb_brick_observability_v1.json "
               "--output-dir outputs/issue-39-rgb-brick-observability "
               f"--pre-run-wall-seconds {float(pre_run_wall_seconds):.2f} "
               f"--wall-budget-seconds {budget:.2f}")
    payload = {"schema_version": 1, "issue": 39, "probe_id": "issue-39-rgb-brick-observability-v1",
        "requested_model": "GPT-6 Luna / High", "model_routing_verification": "UNAVAILABLE",
        "command": command,
        "seeds": list(cfg.seeds), "completed_seeds": [r["seed"] for r in runs], "runs": runs,
        "predictions": detector_outputs, "labels_offline_only": all_truth,
        "labels_consistent": label_consistent, "label_count": label_count,
        "episodes_with_labels": episodes_with_labels, "primary_metric": metrics,
        "classification": classification, "native_frames": total_frames,
        "frame_caps": {"per_episode": cfg.max_frames_per_episode,
                       "formal_total": cfg.max_frames_total,
                       "test_reserve": PREDECLARED_TEST_NATIVE_FRAMES,
                       "combined_total": TOTAL_NATIVE_FRAME_CEILING,
                       "wall_seconds": cfg.total_wall_seconds},
        "predeclared_test_native_frames_reserved": PREDECLARED_TEST_NATIVE_FRAMES,
        "pre_run_test_native_frames_upper_bound": 42,
        "remaining_test_native_frame_reserve": PREDECLARED_TEST_NATIVE_FRAMES - 42,
        "combined_native_frame_ceiling": TOTAL_NATIVE_FRAME_CEILING, "elapsed_wall_seconds": 0.0,
        "pre_run_wall_seconds": float(pre_run_wall_seconds),
        "formal_wall_budget_seconds": budget,
        "finalization_wall_reserve_seconds": FINALIZATION_WALL_RESERVE_SECONDS,
        "environment": {"id": ENVIRONMENT_ID, "contract_id": contract.contract_id,
                        "frame_skip": contract.frame_skip, "frame_stack": contract.frame_stack,
                        "sticky_action_probability": contract.sticky_action_probability,
                        "rom_sha256": support.rom_sha256, "versions": _environment_versions()},
        "source": {"revision": _git_revision(root), "branch": subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"], capture_output=True, text=True).stdout.strip(),
            "config_sha256": sha256(cfg.config_path), "contract_sha256": sha256(cfg.contract_path),
            "controller_config_sha256": sha256(cfg.controller_config_path)}}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "rgb_brick_crops.npz"
    np.savez_compressed(npz_path, crops=np.stack(crops) if crops else np.empty((0, 48, 144, 3), dtype=np.uint8),
                        episode_seed=np.array([seed for run in runs for seed in [run["seed"]] * (run["capture_end_index"] - run["capture_start_index"])], dtype=np.int32),
                        native_frame=np.array([frame for run in runs for frame in range(run["capture_end_index"] - run["capture_start_index"])], dtype=np.int32))
    payload["artifacts"] = {"rgb_crops_file": npz_path.name, "rgb_crops_sha256": sha256(npz_path)}
    result_path = out / "results.json"
    report_path = out / "report.md"
    payload["elapsed_wall_seconds"] = time.perf_counter() - started
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload, npz_path.name), encoding="utf-8")
    payload["elapsed_wall_seconds"] = time.perf_counter() - started + FINALIZATION_WALL_RESERVE_SECONDS
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload, npz_path.name), encoding="utf-8")
    return [npz_path, result_path, report_path], payload


def render_report(payload: Mapping[str, Any], crop_artifact: str) -> str:
    m = payload["primary_metric"]
    source = payload["source"]
    return "\n".join(["# Issue #39 RGB Brick Removal Observability", "",
        f"Classification: **{payload['classification']}**", "",
        f"Event F1: {m['f1']:.4f} (TP {m['tp']}, FP {m['fp']}, FN {m['fn']})", "",
        f"Label events: {payload['label_count']} across {payload['episodes_with_labels']} episodes; labels consistent: {payload['labels_consistent']}.",
        f"Frames: {payload['native_frames']} formal + at most {payload['pre_run_test_native_frames_upper_bound']} prior smoke/test frames (<= {payload['remaining_test_native_frame_reserve']} reserved frames remain) = at most {payload['combined_native_frame_ceiling']} native frames.",
        f"Wall time: {payload['pre_run_wall_seconds'] + payload['elapsed_wall_seconds']:.2f}s total ({payload['pre_run_wall_seconds']:.2f}s setup/validation/smoke + {payload['elapsed_wall_seconds']:.2f}s formal runner and finalization). Formal budget: {payload['formal_wall_budget_seconds']:.2f}s; finalization reserve: {payload['finalization_wall_reserve_seconds']:.2f}s; seeds completed: {payload['completed_seeds']}.", "",
        f"Command: `{payload['command']}`.",
        f"Source revision: `{source['revision']}`; branch: `{source['branch']}`.",
        f"Config SHA-256: `{source['config_sha256']}`; Contract v3 SHA-256: `{source['contract_sha256']}`; controller config SHA-256: `{source['controller_config_sha256']}`.",
        f"Lossless captured RGB crops: `{crop_artifact}` (NPZ compressed, uint8; SHA-256 `{payload['artifacts']['rgb_crops_sha256']}`). Detector input is these crops only. The existing canonical completion evaluator reads score/RAM only for the frozen clear-stop rule; evaluator label rows are joined after all action selection. Neither value enters the controller, detector, or action choice.", "",
        "No clear probability or controller-benefit claim is made.", ""])


__all__ = ["ProbeConfig", "load_config", "cell_occupancy", "BrickRemovalDetector", "match_events", "aggregate_episode_metrics", "classify", "bounded_formal_wall_budget", "run_probe", "render_report"]
