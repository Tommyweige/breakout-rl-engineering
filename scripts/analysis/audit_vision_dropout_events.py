"""Run the frozen Issue 27 RGB ball-dropout association diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from breakout_rl.ball_dropout_audit import classify_dropout_events
from breakout_rl.completion import (
    BreakoutCompletionDetector,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.vision_controller import PredictiveBreakoutController
from breakout_rl.vision_evaluation import (
    DEFAULT_VISION_CONFIG,
    ENVIRONMENT_ID,
    load_vision_evaluation_protocol,
    make_vision_breakout_env,
)


FROZEN_SEEDS = (101, 202, 303)
FROZEN_STEP_CAP = 10_000
WALL_CLOCK_CEILING_SECONDS = 600


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _action_name(action: int, action_names: tuple[str, ...]) -> str:
    if action < 0 or action >= len(action_names):
        raise RuntimeError(f"ALE input action {action} is outside action meanings")
    return action_names[action]


def run_dropout_audit(
    *,
    config_path: str | Path = DEFAULT_VISION_CONFIG,
    episode_seeds: Sequence[int] = FROZEN_SEEDS,
    max_steps_per_episode: int = FROZEN_STEP_CAP,
    output_dir: str | Path = "outputs/issue-27-ball-dropout-audit",
) -> dict[str, Any]:
    """Run each frozen seed once, stopping on either frozen compute ceiling."""

    if tuple(episode_seeds) != FROZEN_SEEDS:
        raise ValueError(f"Issue 27 requires exactly seeds {FROZEN_SEEDS}")
    if max_steps_per_episode != FROZEN_STEP_CAP:
        raise ValueError(f"Issue 27 requires a {FROZEN_STEP_CAP}-step cap")
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_vision_evaluation_protocol(config_path)
    if not set(FROZEN_SEEDS).issubset(protocol.seeds):
        raise ValueError("frozen diagnostic seeds must be present in the canonical config")
    protocol = replace(protocol, seeds=FROZEN_SEEDS, trace_episodes=len(FROZEN_SEEDS))
    if protocol.contract.frame_skip != 1 or protocol.contract.sticky_action_probability != 0.25:
        raise RuntimeError("runtime contract differs from the frozen v3 task contract")

    env = make_vision_breakout_env(protocol.contract)
    root = Path(__file__).resolve().parents[2]
    support = inspect_breakout_completion_support(env)
    if not support.supported:
        env.close()
        raise RuntimeError(
            "canonical Breakout clear detector unavailable: "
            f"{support.reason or 'unsupported runtime'}"
        )
    base = getattr(env, "unwrapped", env)
    if getattr(getattr(env, "spec", None), "id", None) != ENVIRONMENT_ID:
        env.close()
        raise RuntimeError(f"expected {ENVIRONMENT_ID}")
    names_getter = getattr(base, "get_action_meanings", None)
    if not callable(names_getter):
        env.close()
        raise RuntimeError("ALE action meanings unavailable")
    action_names = tuple(str(name) for name in names_getter())
    if not {"NOOP", "FIRE", "LEFT", "RIGHT"}.issubset(action_names):
        env.close()
        raise RuntimeError(f"unsupported ALE action meanings: {action_names}")
    action_index = {name: index for index, name in enumerate(action_names)}
    completion = BreakoutCompletionDetector(support)
    episodes: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    stop_reason = "all_episodes_finished"
    goal_reached = False
    cadence_valid = True
    rgb_only_valid = True
    provenance: dict[str, Any] = {
        "issue": 27,
        "branch": "codex/issue-27-ball-dropout-audit",
        "source_sha": None,
        "source_dirty": None,
        "source_sha256": None,
        "config_path": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "contract_path": protocol.contract_path.as_posix(),
        "contract_sha256": _sha256(protocol.contract_path),
        "completion_audit_config_sha256": _sha256(
            root / "configs/eval/breakout_completion_audit_v1.json"
        ),
        "seeds": list(FROZEN_SEEDS),
        "episodes_per_seed": 1,
        "step_cap_per_episode": FROZEN_STEP_CAP,
        "total_native_frame_budget": 30_000,
        "wall_clock_ceiling_seconds": WALL_CLOCK_CEILING_SECONDS,
        "command": (
            "timeout --signal=TERM --kill-after=5s 600s python -m "
            "scripts.analysis.audit_vision_dropout_events --config "
            "configs/eval/breakout_vision_controller_v1.json --episode-seeds "
            "101,202,303 --max-steps-per-episode 10000 --output-dir "
            "outputs/issue-27-ball-dropout-audit"
        ),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": _version("gymnasium"),
            "ale_py": _version("ale-py"),
            "opencv_python": _version("opencv-python"),
        },
        "contract": {
            "id": protocol.contract.contract_id,
            "environment_id": ENVIRONMENT_ID,
            "frame_skip": protocol.contract.frame_skip,
            "frame_stack": protocol.contract.frame_stack,
            "sticky_action_probability": protocol.contract.sticky_action_probability,
            "fire_reset": protocol.contract.fire_reset,
            "terminal_on_life_loss": protocol.contract.terminal_on_life_loss,
            "raw_reward_rule": protocol.contract.raw_reward_rule,
            "observation_shape": list(protocol.expected_observation_shape),
            "control_inputs": ["RGB observations only"],
            "sticky_resolved_physical_action": "unknown; not exposed by ALE wrapper",
        },
        "completion_detector": support.to_dict(),
    }

    try:
        for seed in FROZEN_SEEDS:
            if time.monotonic() - started >= WALL_CLOCK_CEILING_SECONDS:
                stop_reason = "wall_clock_ceiling"
                break
            observation, _reset_info = env.reset(seed=seed)
            if tuple(np.asarray(observation).shape) != protocol.expected_observation_shape:
                raise RuntimeError("runtime observation is not raw RGB (210, 160, 3)")
            controller = PredictiveBreakoutController(**protocol.controller_options)
            completion.reset()
            frame_origin = read_ale_episode_frame(env)
            if frame_origin is None:
                raise RuntimeError("ALE episode-frame counter unavailable")
            current_frame = frame_origin
            lives = read_ale_lives(env)
            previous_lives = lives
            score = 0.0
            life_loss_count = 0
            last_direct_vy = None
            last_direct_horizon = None
            requested_actions: Counter[str] = Counter({name: 0 for name in action_names})
            ale_input_actions: Counter[str] = Counter({name: 0 for name in action_names})
            rows: list[dict[str, Any]] = []
            terminated = truncated = False
            clear_detected = False
            episode_stop_reason = "step_cap"
            for step in range(1, FROZEN_STEP_CAP + 1):
                if time.monotonic() - started >= WALL_CLOCK_CEILING_SECONDS:
                    stop_reason = "wall_clock_ceiling"
                    episode_stop_reason = "wall_clock_ceiling"
                    break
                if tuple(np.asarray(observation).shape) != protocol.expected_observation_shape:
                    rgb_only_valid = False
                    raise RuntimeError("controller received a non-RGB observation")
                decision = controller.select_action(observation)
                obs_frame = current_frame - frame_origin
                requested_action = decision.action
                requested_actions[requested_action] += 1
                observation, reward, terminated_raw, truncated_raw, info = env.step(
                    action_index[requested_action]
                )
                info = info if isinstance(info, dict) else {}
                next_frame = read_ale_episode_frame(env)
                if next_frame is None or next_frame - current_frame != 1:
                    cadence_valid = False
                    raise RuntimeError(
                        "each agent action must advance exactly one ALE-native frame"
                    )
                current_frame = next_frame
                result_frame = current_frame - frame_origin
                ale_input = int(info.get("fire_reset_executed_action", action_index[requested_action]))
                ale_input_action = _action_name(ale_input, action_names)
                ale_input_actions[ale_input_action] += 1
                reward_value = float(reward)
                if not math.isfinite(reward_value):
                    raise RuntimeError("ALE returned non-finite reward")
                score += reward_value
                lives = read_ale_lives(env)
                life_event = bool(info.get("fire_reset_life_loss", False))
                if life_event:
                    life_loss_count += 1
                completion.observe(
                    cumulative_score=score,
                    ram_score=read_breakout_score(env),
                    agent_step=step,
                    emulator_frame=result_frame,
                    lives_remaining=lives,
                )
                ball = decision.observation.ball
                directly_detected = bool(ball.directly_detected)
                if directly_detected:
                    last_direct_vy = ball.vy
                    last_direct_horizon = decision.time_to_intercept_frames
                row = {
                    "episode_seed": seed,
                    "agent_step": step,
                    "observation_ale_frame": obs_frame,
                    "action_result_ale_frame": result_frame,
                    "requested_action": requested_action,
                    "ale_input_action": ale_input_action,
                    "sticky_resolved_physical_action": "unknown",
                    "ball_directly_detected": directly_detected,
                    "ball_age_frames": ball.age_frames,
                    "last_directly_observed_vy": last_direct_vy,
                    "predicted_paddle_intercept_horizon_frames": last_direct_horizon,
                    "ball_x": ball.x,
                    "ball_y": ball.y,
                    "ball_vx": ball.vx,
                    "ball_vy": ball.vy,
                    "ball_confidence": ball.confidence,
                    "paddle_center_x": (
                        decision.observation.paddle.center_x
                        if decision.observation.paddle is not None
                        else None
                    ),
                    "predicted_intercept_x": decision.predicted_intercept_x,
                    "paddle_error": decision.paddle_error,
                    "raw_reward": reward_value,
                    "raw_score": score,
                    "lives_remaining": lives,
                    "lives_before_action": previous_lives,
                    "lives_decreased_on_transition": (
                        lives < previous_lives
                        if lives is not None and previous_lives is not None
                        else None
                    ),
                    "wrapper_life_loss_event": life_event,
                    "auto_fire": bool(info.get("fire_reset_auto", False)),
                    "completion_cleared": completion.state.cleared,
                    "trace_semantics_valid": True,
                }
                rows.append(row)
                all_trace.append(row)
                previous_lives = lives
                terminated = bool(terminated_raw)
                truncated = bool(truncated_raw)
                if completion.state.cleared is True:
                    clear_detected = True
                    episode_stop_reason = "canonical_clear_detected"
                    stop_reason = "canonical_clear_detected"
                    goal_reached = True
                    break
                if terminated or truncated:
                    episode_stop_reason = "game_over" if terminated else "time_limit"
                    break
            else:
                episode_stop_reason = "step_cap"
            if not rows and episode_stop_reason == "wall_clock_ceiling":
                stop_reason = "wall_clock_ceiling"
                break
            episode = {
                "episode_seed": seed,
                "raw_score": score,
                "total_agent_steps": len(rows),
                "total_ale_native_frames": rows[-1]["action_result_ale_frame"] if rows else 0,
                "wrapper_life_loss_event_count": life_loss_count,
                "terminated": terminated,
                "truncated": truncated,
                "canonical_clear_detected": clear_detected,
                "completion_clear_ale_frame": completion.state.clear_emulator_frame,
                "completion_clear_score": completion.state.clear_score,
                "lives_remaining_at_clear": completion.state.lives_remaining_at_clear,
                "stop_reason": episode_stop_reason,
                "direct_ball_detection_rate": (
                    sum(bool(row["ball_directly_detected"]) for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "longest_direct_detection_dropout_frames": _longest_dropout(rows),
                "requested_action_counts": dict(requested_actions),
                "ale_input_action_counts": dict(ale_input_actions),
                "trace_rows": len(rows),
            }
            episodes.append(episode)
            if episode_stop_reason == "step_cap":
                stop_reason = "per_episode_step_cap"
                break
            if episode_stop_reason == "wall_clock_ceiling" or goal_reached:
                break
    finally:
        env.close()

    event_result = classify_dropout_events(all_trace)
    total_frames = sum(int(row["total_ale_native_frames"]) for row in episodes)
    if total_frames > 30_000:
        event_result["classification"] = "INCONCLUSIVE"
        stop_reason = "total_frame_budget_exceeded"
    if any(row["stop_reason"] in {"step_cap", "wall_clock_ceiling"} for row in episodes):
        event_result["classification"] = "INCONCLUSIVE"
    if not cadence_valid or not rgb_only_valid or len(episodes) != 3:
        event_result["classification"] = "INCONCLUSIVE"
    elapsed = time.monotonic() - started
    provenance.update(
        {
            "source_sha": _git_value(root, "rev-parse", "HEAD"),
            "source_dirty": bool(_git_value(root, "status", "--porcelain")),
            "source_sha256": _source_digest(root),
            "runtime_contract_valid": True,
            "rgb_only_action_path_valid": rgb_only_valid,
            "completion_detector_available": bool(support.supported),
            "one_native_frame_per_action": cadence_valid,
            "episodes_completed_or_stopped": len(episodes),
            "total_ale_native_frames": total_frames,
            "exactly_one_episode_for_each_frozen_seed": [
                row["episode_seed"] for row in episodes
            ] == list(FROZEN_SEEDS),
            "stop_reason": stop_reason,
            "wall_clock_seconds": elapsed,
            "ale_action_meanings": list(action_names),
        }
    )
    return {
        "issue": 27,
        "hypothesis": (
            "At least two-thirds of wrapper-reported life-loss events across three episodes "
            "are preceded within 30 ALE frames by a qualifying five-frame visual dropout."
        ),
        "classification": event_result["classification"],
        "primary_metric": event_result,
        "secondary_metrics": {
            "episodes": episodes,
            "wrapper_life_loss_events_by_seed": {
                str(row["episode_seed"]): row["wrapper_life_loss_event_count"]
                for row in episodes
            },
            "longest_direct_detection_dropout_frames_by_seed": {
                str(row["episode_seed"]): row["longest_direct_detection_dropout_frames"]
                for row in episodes
            },
            "total_ale_native_frames": total_frames,
            "requested_and_ale_input_action_counts": {
                str(row["episode_seed"]): {
                    "requested_action": row["requested_action_counts"],
                    "ale_input_action": row["ale_input_action_counts"],
                }
                for row in episodes
            },
        },
        "stop_reason": stop_reason,
        "goal_reached": goal_reached,
        "trace": all_trace,
        "provenance": provenance,
    }


def _longest_dropout(rows: Sequence[dict[str, Any]]) -> int:
    longest = current = 0
    previous_frame: int | None = None
    for row in rows:
        frame = int(row["observation_ale_frame"])
        if not row["ball_directly_detected"] and (
            previous_frame is None or frame == previous_frame + 1
        ):
            current += 1
        elif not row["ball_directly_detected"]:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous_frame = frame
    return longest


def _git_value(root: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


def _source_digest(root: Path) -> str:
    paths = (
        "breakout_env.py",
        "breakout_rl/completion.py",
        "breakout_rl/evaluation_contract.py",
        "configs/eval/breakout_completion_audit_v1.json",
        "breakout_rl/vision_controller.py",
        "breakout_rl/vision_evaluation.py",
        "breakout_rl/ball_dropout_audit.py",
        "scripts/analysis/audit_vision_dropout_events.py",
        "configs/eval/breakout_vision_controller_v1.json",
        "configs/eval/breakout_contract_v3.json",
    )
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _write_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = result.pop("trace")
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in trace),
        encoding="utf-8",
    )
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode_seed", "event_ale_frame", "status", "associated", "qualifying_dropout_runs"))
        writer.writeheader()
        for event in result["primary_metric"]["events"]:
            writer.writerow({"episode_seed": "", "event_ale_frame": event["event_ale_frame"], "status": event["status"], "associated": event["associated"], "qualifying_dropout_runs": json.dumps(event.get("qualifying_dropout_runs", []), sort_keys=True)})
    report = [
        "# Issue 27 ball-dropout audit",
        "",
        f"- Classification: **{result['classification']}**",
        f"- Associated wrapper life-loss events: {result['primary_metric']['associated_life_loss_events']} / {result['primary_metric']['all_wrapper_reported_life_loss_events']}",
        f"- Associated fraction: {result['primary_metric']['associated_life_loss_fraction']}",
        f"- Ambiguous events: {result['primary_metric']['ambiguous_life_loss_events']}",
        f"- Episodes: {len(result['secondary_metrics']['episodes'])}; ALE frames: {result['provenance']['total_ale_native_frames']}",
        f"- Stop reason: {result['stop_reason']}",
        "",
        "## Per-episode summary",
        "",
        "| Seed | Score | ALE frames | Life-loss events | Ball detection rate | Longest dropout | Stop reason |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["secondary_metrics"]["episodes"]:
        report.append(
            f"| {row['episode_seed']} | {row['raw_score']} | {row['total_ale_native_frames']} | {row['wrapper_life_loss_event_count']} | {row['direct_ball_detection_rate']} | {row['longest_direct_detection_dropout_frames']} | {row['stop_reason']} |"
        )
    report.extend(
        [
            "",
            "RAM score/lives and the canonical completion detector were used only outside the controller for audit and completion evidence. The controller action path consumed raw RGB observations only. `requested_action` and `ale_input_action` are distinct; sticky-resolved physical actions remain unknown.",
            "",
            "See `results.json`, `events.csv`, and `trace.jsonl` for complete metrics, provenance, classifier output, and per-frame evidence.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result["trace"] = trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = tuple(int(part) for part in args.episode_seeds.split(","))
        result = run_dropout_audit(
            config_path=args.config,
            episode_seeds=seeds,
            max_steps_per_episode=args.max_steps_per_episode,
            output_dir=args.output_dir,
        )
        _write_artifacts(result, args.output_dir)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "primary_metric": result["primary_metric"],
                "stop_reason": result["stop_reason"],
                "goal_reached": result["goal_reached"],
                "artifacts": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
