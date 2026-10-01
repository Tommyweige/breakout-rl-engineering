"""Run the frozen paired Issue 29 visual track-retention experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import operator
import platform
import signal
import sys
import time
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.completion import (
    BreakoutCompletionDetector,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.evaluation_contract import BREAKOUT_CONTRACT_V3_ID
from breakout_rl.vision_controller import PredictiveBreakoutController
from breakout_rl.vision_evaluation import (
    DEFAULT_VISION_CONFIG,
    ENVIRONMENT_ID,
    load_vision_evaluation_protocol,
    make_vision_breakout_env,
)

FROZEN_SEEDS = (101, 202, 303)
FROZEN_ARMS = (4, 12)
FROZEN_STEP_CAP = 10_000
FRAME_BUDGET = 60_000
WALL_CLOCK_CEILING_SECONDS = 600
_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: Any) -> None:
    """Let the frozen outer timeout trigger a clean partial-artifact checkpoint."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def validate_arms(baseline: int, candidate: int) -> tuple[int, int]:
    """Reject any experiment variant other than the two frozen settings."""
    parsed: list[int] = []
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if isinstance(value, bool):
            raise TypeError(f"{name} retention setting must be an integer")
        try:
            parsed.append(operator.index(value))
        except TypeError as error:
            raise TypeError(f"{name} retention setting must be an integer") from error
    resolved = (parsed[0], parsed[1])
    if resolved != FROZEN_ARMS:
        raise ValueError(f"Issue 29 requires arms {FROZEN_ARMS}, got {resolved}")
    return resolved


def controller_options_for_arm(options: Mapping[str, Any], arm: int) -> dict[str, Any]:
    """Copy the validated config options and replace only the frozen retention value."""
    if arm not in FROZEN_ARMS:
        raise ValueError(f"unsupported Issue 29 arm: {arm}")
    if options.get("max_missing_frames") != 4:
        raise ValueError("baseline config must resolve to max_missing_frames=4")
    resolved = dict(options)
    resolved["max_missing_frames"] = arm
    return resolved


def paired_survival_gain_seed_count(episodes: Sequence[dict[str, Any]]) -> int:
    """Count completed pairs with at least 500 additional native frames."""
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for episode in episodes:
        key = (int(episode["max_missing_frames"]), int(episode["episode_seed"]))
        if key in by_key:
            raise ValueError(f"duplicate episode arm/seed: {key}")
        by_key[key] = episode
    count = 0
    for seed in FROZEN_SEEDS:
        baseline = by_key.get((4, seed))
        candidate = by_key.get((12, seed))
        if not baseline or not candidate:
            continue
        if baseline.get("stop_reason") != "game_over" or candidate.get("stop_reason") != "game_over":
            continue
        if int(candidate["game_over_ale_frame"]) - int(baseline["game_over_ale_frame"]) >= 500:
            count += 1
    return count


def classify_result(episodes: Sequence[dict[str, Any]], *, provenance_valid: bool) -> tuple[str, int, dict[str, int]]:
    """Apply only the frozen Issue 29 completion and paired-delta criteria."""
    gain_count = paired_survival_gain_seed_count(episodes)
    deltas: dict[str, int] = {}
    for seed in FROZEN_SEEDS:
        base = next((e for e in episodes if e["max_missing_frames"] == 4 and e["episode_seed"] == seed), None)
        cand = next((e for e in episodes if e["max_missing_frames"] == 12 and e["episode_seed"] == seed), None)
        if base and cand and base.get("game_over_ale_frame") is not None and cand.get("game_over_ale_frame") is not None:
            deltas[str(seed)] = int(cand["game_over_ale_frame"]) - int(base["game_over_ale_frame"])
    complete_keys = {(e["max_missing_frames"], e["episode_seed"]) for e in episodes}
    exact_six = complete_keys == {(arm, seed) for arm in FROZEN_ARMS for seed in FROZEN_SEEDS} and len(episodes) == 6
    no_clear_or_cap = all(e.get("stop_reason") == "game_over" for e in episodes)
    regressed_too_far = any(delta < -1000 for delta in deltas.values())
    if not provenance_valid or not exact_six or not no_clear_or_cap or regressed_too_far:
        return "INCONCLUSIVE", gain_count, deltas
    if gain_count >= 2:
        return "PROMOTED", gain_count, deltas
    if gain_count == 0:
        return "REJECTED", gain_count, deltas
    return "INCONCLUSIVE", gain_count, deltas


def _git_value(root: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True, timeout=10)
    return result.stdout.strip()


def _longest_dropout(rows: Sequence[dict[str, Any]]) -> int:
    longest = current = 0
    for row in rows:
        if row["ball_directly_detected"]:
            current = 0
        else:
            current += 1
        longest = max(longest, current)
    return longest


def _write_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["trace"]:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("max_missing_frames", "episode_seed", "stop_reason", "game_over_ale_frame", "total_agent_steps", "raw_score", "wrapper_life_loss_frames", "direct_ball_detection_rate", "longest_dropout_frames")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in result["episodes"]:
            writer.writerow({key: json.dumps(episode.get(key), sort_keys=True) if isinstance(episode.get(key), (list, dict)) else episode.get(key) for key in fields})
    report = [
        "# Issue 29 paired track-retention comparison", "",
        f"- Classification: **{result['classification']}**",
        f"- Paired survival gain seed count: **{result['primary_metric']['paired_survival_gain_seed_count']} / 3**",
        f"- Completed episodes: {len(result['episodes'])} / 6; native frames: {result['provenance']['total_ale_native_frames']}",
        f"- Stop reason: `{result['stop_reason']}`", "",
        "## Per-episode results", "",
        "| Arm | Seed | Game-over frame | Score | Life-loss frames | Detection rate | Longest dropout | Stop reason |",
        "|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for e in result["episodes"]:
        report.append(f"| {e['max_missing_frames']} | {e['episode_seed']} | {e.get('game_over_ale_frame')} | {e['raw_score']} | {e['wrapper_life_loss_frames']} | {e['direct_ball_detection_rate']} | {e['longest_dropout_frames']} | {e['stop_reason']} |")
    report.extend(["", "The controller receives raw RGB only. Completion, score, and life reads remain outside the controller; requested and ALE-input actions are recorded separately, and sticky-resolved physical actions remain unknown.", "", "See `results.json`, `episodes.csv`, and `trace.jsonl` for full provenance and frame evidence."])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _checkpoint_frame(result: dict[str, Any], output_dir: Path, row: dict[str, Any]) -> None:
    """Durably append each action trace and record exact partial-episode resume state."""
    with (output_dir / "trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
    resume = {
        "next_or_partial_arm": row["max_missing_frames"],
        "next_or_partial_seed": row["episode_seed"],
        "partial_agent_step": row["agent_step"],
        "episodes_completed": len(result["episodes"]),
        "total_ale_native_frames": result["provenance"]["total_ale_native_frames"],
        "last_trace_key": [row["max_missing_frames"], row["episode_seed"], row["agent_step"]],
        "stop_requested": _STOP_REQUESTED,
    }
    temporary = output_dir / ".resume.json.tmp"
    temporary.write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "resume.json")


def _checkpoint_episode(result: dict[str, Any], output_dir: Path, *, next_arm: int | None, next_seed: int | None) -> None:
    resume = {
        "partial_episode": None,
        "completed_pairs": [[e["max_missing_frames"], e["episode_seed"]] for e in result["episodes"]],
        "episodes_completed": len(result["episodes"]),
        "total_ale_native_frames": result["provenance"]["total_ale_native_frames"],
        "next_arm": next_arm,
        "next_seed": next_seed,
        "goal_reached": result["goal_reached"],
        "stop_reason": result["stop_reason"],
    }
    temporary = output_dir / ".resume.json.tmp"
    temporary.write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "resume.json")


def run_comparison(*, config_path: str | Path = DEFAULT_VISION_CONFIG, episode_seeds: Sequence[int] = FROZEN_SEEDS,
                   baseline_max_missing_frames: int = 4, candidate_max_missing_frames: int = 12,
                   max_steps_per_episode: int = FROZEN_STEP_CAP,
                   output_dir: str | Path = "outputs/issue-29-track-retention") -> dict[str, Any]:
    validate_arms(baseline_max_missing_frames, candidate_max_missing_frames)
    if tuple(episode_seeds) != FROZEN_SEEDS:
        raise ValueError(f"Issue 29 requires exactly seeds {FROZEN_SEEDS}")
    if max_steps_per_episode != FROZEN_STEP_CAP:
        raise ValueError(f"Issue 29 requires a {FROZEN_STEP_CAP}-step cap")
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_vision_evaluation_protocol(config_path)
    if protocol.controller_options["max_missing_frames"] != 4:
        raise RuntimeError("baseline config must resolve to max_missing_frames=4")
    if not set(FROZEN_SEEDS).issubset(protocol.seeds):
        raise ValueError("frozen comparison seeds must be present in the canonical config")
    contract = protocol.contract
    frozen_runtime = (
        contract.contract_id == BREAKOUT_CONTRACT_V3_ID
        and contract.frame_skip == 1 and contract.frame_stack == 4
        and contract.sticky_action_probability == 0.25 and contract.fire_reset is True
        and contract.terminal_on_life_loss is False
        and contract.raw_reward_rule == "sum environment rewards without clipping"
    )
    if not frozen_runtime:
        raise RuntimeError("runtime contract differs from frozen Breakout Contract v3 semantics")
    protocol = replace(protocol, seeds=FROZEN_SEEDS, trace_episodes=len(FROZEN_SEEDS))
    root = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        "issue": 29,
        "classification": "INCONCLUSIVE",
        "hypothesis": "With only max_missing_frames changed 4 to 12, at least two paired seeds gain >=500 ALE-native game-over frames and no pair loses >1,000 frames.",
        "arm_order": [4, 12],
        "episodes": [], "trace": [], "stop_reason": "not_started", "goal_reached": False,
        "primary_metric": {"name": "paired_survival_gain_seed_count", "paired_survival_gain_seed_count": 0, "paired_deltas_frames": {}},
        "secondary_metrics": {},
        "provenance": {
            "issue": 29, "branch": "codex/issue-29-track-retention", "base_sha": "f8311b55a05f4f901d63e70a89a49d6dc39a02c8",
            "source_sha": _git_value(root, "rev-parse", "HEAD"), "source_dirty_at_run_start": bool(_git_value(root, "status", "--porcelain")),
            "config_path": config_path.as_posix(), "config_sha256": _sha256(config_path),
            "contract_path": protocol.contract_path.as_posix(), "contract_sha256": _sha256(protocol.contract_path),
            "completion_audit_config_sha256": _sha256(root / "configs/eval/breakout_completion_audit_v1.json"),
            "source_sha256": None,
            "seeds": list(FROZEN_SEEDS), "episodes_per_seed_per_arm": 1, "arm_values": list(FROZEN_ARMS),
            "step_cap_per_episode": FROZEN_STEP_CAP, "native_frame_budget": FRAME_BUDGET,
            "wall_clock_ceiling_seconds": WALL_CLOCK_CEILING_SECONDS,
            "command": "timeout --signal=TERM --kill-after=5s 600s python -m scripts.analysis.compare_vision_dropout_retention --config configs/eval/breakout_vision_controller_v1.json --episode-seeds 101,202,303 --baseline-max-missing-frames 4 --candidate-max-missing-frames 12 --max-steps-per-episode 10000 --output-dir outputs/issue-29-track-retention",
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "gymnasium": _version("gymnasium"), "ale_py": _version("ale-py"), "opencv_python": _version("opencv-python")},
            "contract": {
                "id": contract.contract_id, "environment_id": ENVIRONMENT_ID,
                "frame_skip": contract.frame_skip, "frame_stack": contract.frame_stack,
                "sticky_action_probability": contract.sticky_action_probability,
                "fire_reset": contract.fire_reset,
                "fire_reset_confirmation": {
                    "max_fire_attempts": contract.fire_reset_confirmation.max_fire_attempts,
                    "confirmation_steps": contract.fire_reset_confirmation.confirmation_steps,
                    "min_observation_change_fraction": contract.fire_reset_confirmation.min_observation_change_fraction,
                    "confirmation_operator": contract.fire_reset_confirmation.confirmation_operator,
                    "confirmation_signals": list(contract.fire_reset_confirmation.confirmation_signals),
                },
                "terminal_on_life_loss": contract.terminal_on_life_loss,
                "time_limit_semantics": dict(contract.time_limit_semantics),
                "concrete_episode_seeds": list(contract.concrete_episode_seeds),
                "evaluation_epsilon": contract.evaluation_epsilon,
                "raw_reward_rule": contract.raw_reward_rule,
                "observation_shape": list(protocol.expected_observation_shape),
                "control_inputs": ["RGB observations only"],
                "sticky_resolved_physical_action": "unknown; not exposed by ALE wrapper",
            },
            "runtime_contract_valid": frozen_runtime, "cadence_valid": True, "rgb_only_valid": True,
            "completion_detector_available": False, "completion_detector": None,
            "total_ale_native_frames": 0, "episodes_completed_or_stopped": 0,
        },
    }
    digest = hashlib.sha256()
    source_files = ("breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation_contract.py", "configs/eval/breakout_completion_audit_v1.json", "breakout_rl/vision_controller.py", "breakout_rl/vision_evaluation.py", "scripts/analysis/compare_vision_dropout_retention.py", "configs/eval/breakout_vision_controller_v1.json", "configs/eval/breakout_contract_v3.json")
    for relative in source_files:
        digest.update(relative.encode() + b"\0" + (root / relative).read_bytes() + b"\0")
    result["provenance"]["source_sha256"] = digest.hexdigest()
    _write_artifacts(result, output_dir)
    env = make_vision_breakout_env(contract)
    try:
        support = inspect_breakout_completion_support(env)
        result["provenance"]["completion_detector"] = support.to_dict()
        result["provenance"]["completion_detector_available"] = bool(support.supported)
        if not support.supported:
            result["stop_reason"] = "completion_detector_unavailable"
            _write_artifacts(result, output_dir)
            raise RuntimeError("canonical Breakout clear detector unavailable: " + (support.reason or "unsupported runtime"))
        base = getattr(env, "unwrapped", env)
        if getattr(getattr(env, "spec", None), "id", None) != ENVIRONMENT_ID:
            raise RuntimeError(f"expected {ENVIRONMENT_ID}")
        names_getter = getattr(base, "get_action_meanings", None)
        if not callable(names_getter):
            raise RuntimeError("ALE action meanings unavailable")
        action_names = tuple(str(name) for name in names_getter())
        if not {"NOOP", "FIRE", "LEFT", "RIGHT"}.issubset(action_names):
            raise RuntimeError(f"unsupported ALE action meanings: {action_names}")
        action_index = {name: i for i, name in enumerate(action_names)}
        completion = BreakoutCompletionDetector(support)
        result["provenance"]["ale_action_meanings"] = list(action_names)
        result["stop_reason"] = "all_episodes_finished"
        _write_artifacts(result, output_dir)
        ordinal = 0
        for max_missing_frames in FROZEN_ARMS:
            for seed in FROZEN_SEEDS:
                if time.monotonic() - started >= WALL_CLOCK_CEILING_SECONDS:
                    result["stop_reason"] = "wall_clock_ceiling"
                    break
                if result["provenance"]["total_ale_native_frames"] >= FRAME_BUDGET:
                    result["stop_reason"] = "total_frame_budget"
                    break
                ordinal += 1
                observation, _ = env.reset(seed=seed)
                if tuple(np.asarray(observation).shape) != protocol.expected_observation_shape:
                    raise RuntimeError("runtime observation is not raw RGB (210, 160, 3)")
                controller_options = controller_options_for_arm(protocol.controller_options, max_missing_frames)
                controller = PredictiveBreakoutController(**controller_options)
                completion.reset()
                frame_origin = read_ale_episode_frame(env)
                if frame_origin is None:
                    raise RuntimeError("ALE episode-frame counter unavailable")
                current_frame = frame_origin
                score = 0.0
                lives = read_ale_lives(env)
                previous_lives = lives
                life_loss_frames: list[int] = []
                requested_actions: Counter[str] = Counter({name: 0 for name in action_names})
                ale_input_actions: Counter[str] = Counter({name: 0 for name in action_names})
                rows: list[dict[str, Any]] = []
                episode_stop = "step_cap"
                terminated = truncated = clear_detected = False
                for step in range(1, FROZEN_STEP_CAP + 1):
                    if _STOP_REQUESTED or time.monotonic() - started >= WALL_CLOCK_CEILING_SECONDS:
                        episode_stop = "wall_clock_ceiling"
                        result["stop_reason"] = "wall_clock_ceiling"
                        break
                    observation_frame = current_frame - frame_origin
                    decision = controller.select_action(observation)
                    requested_actions[decision.action] += 1
                    observation, reward, terminated_raw, truncated_raw, info = env.step(action_index[decision.action])
                    info = info if isinstance(info, dict) else {}
                    next_frame = read_ale_episode_frame(env)
                    if next_frame is None or next_frame - current_frame != 1:
                        result["provenance"]["cadence_valid"] = False
                        raise RuntimeError("each agent action must advance exactly one ALE-native frame")
                    current_frame = next_frame
                    result_frame = current_frame - frame_origin
                    ale_input = int(info.get("fire_reset_executed_action", action_index[decision.action]))
                    if not 0 <= ale_input < len(action_names):
                        raise RuntimeError("ALE input action is outside action meanings")
                    ale_input_name = action_names[ale_input]
                    ale_input_actions[ale_input_name] += 1
                    reward_value = float(reward)
                    if not math.isfinite(reward_value):
                        raise RuntimeError("ALE returned non-finite reward")
                    score += reward_value
                    lives = read_ale_lives(env)
                    if bool(info.get("fire_reset_life_loss", False)):
                        life_loss_frames.append(result_frame)
                    completion.observe(cumulative_score=score, ram_score=read_breakout_score(env), agent_step=step, emulator_frame=result_frame, lives_remaining=lives)
                    ball = decision.observation.ball
                    row = {
                        "arm": "baseline" if max_missing_frames == 4 else "candidate", "max_missing_frames": max_missing_frames,
                        "episode_seed": seed, "episode_ordinal": ordinal, "agent_step": step,
                        "observation_ale_frame": observation_frame, "action_result_ale_frame": result_frame,
                        "requested_action": decision.action, "ale_input_action": ale_input_name,
                        "sticky_resolved_physical_action": "unknown", "ball_directly_detected": bool(ball.directly_detected),
                        "ball_age_frames": ball.age_frames, "ball_x": ball.x, "ball_y": ball.y, "ball_vx": ball.vx, "ball_vy": ball.vy,
                        "ball_confidence": ball.confidence, "predicted_paddle_intercept_horizon_frames": decision.time_to_intercept_frames,
                        "paddle_center_x": decision.observation.paddle.center_x if decision.observation.paddle else None,
                        "predicted_intercept_x": decision.predicted_intercept_x, "paddle_error": decision.paddle_error,
                        "raw_reward": reward_value, "raw_score": score, "lives_remaining": lives,
                        "lives_decreased_on_transition": lives < previous_lives if lives is not None and previous_lives is not None else None,
                        "wrapper_life_loss_event": bool(info.get("fire_reset_life_loss", False)),
                        "completion_cleared": completion.state.cleared,
                    }
                    rows.append(row)
                    result["trace"].append(row)
                    result["provenance"]["total_ale_native_frames"] += 1
                    _checkpoint_frame(result, output_dir, row)
                    previous_lives = lives
                    terminated, truncated = bool(terminated_raw), bool(truncated_raw)
                    if completion.state.cleared is True:
                        clear_detected = True
                        episode_stop = "canonical_clear_detected"
                        result["stop_reason"] = "canonical_clear_detected"
                        result["goal_reached"] = True
                        break
                    if terminated or truncated:
                        episode_stop = "game_over" if terminated else "time_limit"
                        break
                else:
                    episode_stop = "step_cap"
                if episode_stop == "wall_clock_ceiling":
                    # Do not add an empty episode; the current episode trace is still preserved.
                    result["provenance"]["partial_episode"] = {"max_missing_frames": max_missing_frames, "episode_seed": seed, "agent_steps": len(rows), "stop_reason": episode_stop}
                    _write_artifacts(result, output_dir)
                    break
                if rows:
                    game_over_frame = rows[-1]["action_result_ale_frame"] if episode_stop == "game_over" else None
                    episode = {
                        "max_missing_frames": max_missing_frames, "episode_seed": seed, "raw_score": score,
                        "total_agent_steps": len(rows), "total_ale_native_frames": rows[-1]["action_result_ale_frame"],
                        "game_over_ale_frame": game_over_frame, "wrapper_life_loss_frames": life_loss_frames,
                        "wrapper_life_loss_count": len(life_loss_frames), "terminated": terminated, "truncated": truncated,
                        "canonical_clear_detected": clear_detected, "completion_clear_ale_frame": completion.state.clear_emulator_frame,
                        "completion_clear_score": completion.state.clear_score, "stop_reason": episode_stop,
                        "direct_ball_detection_rate": sum(r["ball_directly_detected"] for r in rows) / len(rows),
                        "longest_dropout_frames": _longest_dropout(rows), "requested_action_counts": dict(requested_actions),
                        "ale_input_action_counts": dict(ale_input_actions), "trace_rows": len(rows),
                    }
                    result["episodes"].append(episode)
                    result["provenance"]["episodes_completed_or_stopped"] = len(result["episodes"])
                _finish_classification(result)
                next_position = ordinal
                has_next = next_position < 6 and not clear_detected and episode_stop == "game_over"
                next_arm = FROZEN_ARMS[next_position // len(FROZEN_SEEDS)] if has_next else None
                next_seed = FROZEN_SEEDS[next_position % len(FROZEN_SEEDS)] if has_next else None
                _checkpoint_episode(result, output_dir, next_arm=next_arm, next_seed=next_seed)
                _write_artifacts(result, output_dir)
                if clear_detected or episode_stop in {"step_cap", "wall_clock_ceiling"}:
                    result["stop_reason"] = "canonical_clear_detected" if clear_detected else episode_stop
                    break
            if result["goal_reached"] or result["stop_reason"] in {"wall_clock_ceiling", "step_cap", "canonical_clear_detected", "total_frame_budget"}:
                break
        if result["stop_reason"] == "all_episodes_finished":
            _finish_classification(result)
            if len(result["episodes"]) == 6:
                result["stop_reason"] = "all_episodes_finished"
            else:
                result["stop_reason"] = "incomplete_episode_set"
        result["provenance"]["wall_clock_seconds"] = time.monotonic() - started
        result["provenance"]["exact_arm_seed_pairs"] = len(result["episodes"]) == 6 and {(e["max_missing_frames"], e["episode_seed"]) for e in result["episodes"]} == {(a, s) for a in FROZEN_ARMS for s in FROZEN_SEEDS}
        _finish_classification(result)
        _write_artifacts(result, output_dir)
    finally:
        env.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return result


def _finish_classification(result: dict[str, Any]) -> None:
    prov = result["provenance"]
    valid = bool(prov.get("runtime_contract_valid") and prov.get("cadence_valid") and prov.get("rgb_only_valid") and prov.get("completion_detector_available") and prov.get("exact_arm_seed_pairs"))
    classification, count, deltas = classify_result(result["episodes"], provenance_valid=valid)
    result["classification"] = classification
    if result.get("goal_reached"):
        result["classification"] = "GOAL_REACHED"
    result["primary_metric"] = {"name": "paired_survival_gain_seed_count", "paired_survival_gain_seed_count": count, "paired_deltas_frames": deltas}
    result["secondary_metrics"] = {"total_ale_native_frames": prov.get("total_ale_native_frames", 0), "episodes": result["episodes"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--baseline-max-missing-frames", type=int, required=True)
    parser.add_argument("--candidate-max-missing-frames", type=int, required=True)
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = tuple(int(part) for part in args.episode_seeds.split(","))
        result = run_comparison(config_path=args.config, episode_seeds=seeds,
            baseline_max_missing_frames=args.baseline_max_missing_frames,
            candidate_max_missing_frames=args.candidate_max_missing_frames,
            max_steps_per_episode=args.max_steps_per_episode, output_dir=args.output_dir)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"classification": result["classification"], "primary_metric": result["primary_metric"], "stop_reason": result["stop_reason"], "goal_reached": result["goal_reached"], "episodes": len(result["episodes"]), "artifacts": str(args.output_dir)}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
