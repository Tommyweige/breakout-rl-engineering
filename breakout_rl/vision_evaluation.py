"""Fixed-seed evaluation for the pixel-only predictive Breakout controller."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import operator
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from breakout_env import BreakoutFireResetWrapper, ENVIRONMENT_ID, make_breakout_raw_env
from breakout_rl.completion import (
    BREAKOUT_FULL_CLEAR_SCORE,
    BreakoutCompletionDetector,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V3_ID,
    BreakoutEvaluationContractV2,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.vision_controller import (
    CONTROLLER_ACTIONS,
    PredictiveBreakoutController,
)


VISION_CONFIG_SCHEMA_VERSION = 1
DEFAULT_VISION_CONFIG = Path("configs/eval/breakout_vision_controller_v1.json")
BASELINE_REPORT_URL = (
    "https://github.com/Tommyweige/breakout-rl-engineering/blob/"
    "docs-ironman-series/articles/day21-final-long-training.md"
)
EnvironmentFactory = Callable[[], Any]


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _integer_sequence(values: Any, *, name: str, minimum: int = 0) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise TypeError(f"{name} must be a non-empty sequence of integers")
    parsed = tuple(_positive_integer(value, name=name, minimum=minimum) for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique values")
    return parsed


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a JSON object")
    return dict(payload)


@dataclass(frozen=True)
class VisionEvaluationProtocol:
    config_path: Path
    contract_path: Path
    contract: BreakoutEvaluationContractV2
    contract_id: str
    seeds: tuple[int, ...]
    episodes_per_seed: int
    trace_episodes: int
    expected_observation_shape: tuple[int, ...]
    controller_options: Mapping[str, Any]

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.episodes_per_seed

    @property
    def max_steps_per_episode(self) -> int:
        return int(self.contract.time_limit_semantics["agent_step_limit"])


def load_vision_evaluation_protocol(
    path: str | Path = DEFAULT_VISION_CONFIG,
) -> VisionEvaluationProtocol:
    config_path = Path(path).resolve()
    values = _read_json_mapping(config_path)
    if values.get("schema_version") != VISION_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{config_path}: schema_version must be {VISION_CONFIG_SCHEMA_VERSION}"
        )
    contract_id = values.get("contract_id")
    if contract_id != "breakout-vision-controller-v1-frame-skip-1":
        raise ValueError(f"{config_path}: unsupported vision contract_id {contract_id!r}")
    contract_reference = values.get("base_environment_contract")
    if not isinstance(contract_reference, str) or not contract_reference.strip():
        raise ValueError(f"{config_path}: base_environment_contract is required")
    contract_path = (config_path.parent / contract_reference).resolve()
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    if contract.contract_id != BREAKOUT_CONTRACT_V3_ID:
        raise ValueError("the pixel controller requires the frame-skip-1 v3 task semantics")

    seeds = _integer_sequence(values.get("concrete_episode_seeds"), name="seed")
    if seeds != contract.concrete_episode_seeds:
        raise ValueError("vision controller seeds must match the v3 contract's concrete seeds")
    episodes_per_seed = _positive_integer(
        values.get("episodes_per_seed"), name="episodes_per_seed"
    )
    if episodes_per_seed != 1:
        raise ValueError(
            "the v3 contract lists concrete episode seeds; episodes_per_seed must be 1"
        )
    trace_episodes = _positive_integer(
        values.get("trace_episodes", 1), name="trace_episodes", minimum=0
    )
    if trace_episodes > len(seeds):
        raise ValueError("trace_episodes cannot exceed the number of concrete episode seeds")

    observation = values.get("observation")
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be an object")
    if observation.get("type") != "ale_raw_rgb":
        raise ValueError("vision controller observations must use ale_raw_rgb")
    expected_shape = tuple(
        _integer_sequence(
            observation.get("expected_shape"),
            name="observation.expected_shape",
            minimum=1,
        )
    )
    if expected_shape != (210, 160, 3):
        raise ValueError("the audited ALE raw RGB observation shape must be (210, 160, 3)")
    history_frames = _positive_integer(
        observation.get("controller_history_frames"),
        name="controller_history_frames",
    )
    if history_frames < 2:
        raise ValueError("controller_history_frames must include current and previous screens")

    controller_options = values.get("controller")
    if not isinstance(controller_options, Mapping):
        raise TypeError("controller must be an object")
    normalized_options = {
        "deadband": controller_options.get("deadband_pixels", 3.0),
        "hysteresis": controller_options.get("hysteresis_pixels", 1.0),
        "max_missing_frames": controller_options.get("max_missing_frames", 4),
        "max_ball_speed_pixels_per_frame": controller_options.get(
            "max_ball_speed_pixels_per_frame", 8.0
        ),
    }
    # Constructor validation keeps the file and implementation on the same contract.
    PredictiveBreakoutController(**normalized_options)
    return VisionEvaluationProtocol(
        config_path=config_path,
        contract_path=contract_path,
        contract=contract,
        contract_id=contract_id,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        trace_episodes=trace_episodes,
        expected_observation_shape=expected_shape,
        controller_options=normalized_options,
    )


def make_vision_breakout_env(contract: BreakoutEvaluationContractV2) -> Any:
    """Create raw RGB ALE with the v3 sticky-action and environment FIRE rules."""

    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    if contract.contract_id != BREAKOUT_CONTRACT_V3_ID or contract.frame_skip != 1:
        raise ValueError("raw pixel control requires Breakout contract v3 with frame_skip=1")
    confirmation = contract.fire_reset_confirmation
    raw = make_breakout_raw_env(
        sticky_action_probability=contract.sticky_action_probability
    )
    return BreakoutFireResetWrapper(
        raw,
        max_fire_attempts=confirmation.max_fire_attempts,
        confirmation_steps=confirmation.confirmation_steps,
        min_observation_change_fraction=confirmation.min_observation_change_fraction,
        confirmation_operator=confirmation.confirmation_operator,
        confirmation_signals=confirmation.confirmation_signals,
    )


def _source_provenance(repository_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        revision = None
    try:
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty: bool | None = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        dirty = None
    source_files = (
        Path("breakout_env.py"),
        Path("breakout_rl/completion.py"),
        Path("breakout_rl/evaluation_contract.py"),
        Path("configs/eval/breakout_completion_audit_v1.json"),
        Path("breakout_rl/vision_controller.py"),
        Path("breakout_rl/vision_evaluation.py"),
        Path("configs/eval/breakout_vision_controller_v1.json"),
        Path("configs/eval/breakout_contract_v3.json"),
    )
    digest = hashlib.sha256()
    try:
        for source_file in source_files:
            digest.update(source_file.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update((repository_root / source_file).read_bytes())
            digest.update(b"\0")
        source_sha256: str | None = digest.hexdigest()
    except OSError:
        source_sha256 = None
    return {
        "source_commit": revision or None,
        "working_tree_dirty": dirty,
        "source_sha256": source_sha256,
        "source_files": [source_file.as_posix() for source_file in source_files],
    }


def _action_name(action: int, action_names: Sequence[str]) -> str:
    if isinstance(action, bool) or not 0 <= int(action) < len(action_names):
        raise ValueError(f"environment action {action!r} is outside the legal action set")
    return action_names[int(action)]


def evaluate_predictive_controller(
    protocol: VisionEvaluationProtocol,
    *,
    env_factory: EnvironmentFactory | None = None,
    max_steps_per_episode: int | None = None,
) -> dict[str, Any]:
    """Run every concrete seed and use the canonical completion detector."""

    if not isinstance(protocol, VisionEvaluationProtocol):
        raise TypeError("protocol must be a VisionEvaluationProtocol")
    step_limit = protocol.max_steps_per_episode
    if max_steps_per_episode is not None:
        requested_limit = _positive_integer(
            max_steps_per_episode, name="max_steps_per_episode"
        )
        if requested_limit > step_limit:
            raise ValueError("max_steps_per_episode cannot exceed the v3 contract limit")
        step_limit = requested_limit

    factory = env_factory or (lambda: make_vision_breakout_env(protocol.contract))
    env = factory()
    root = Path(__file__).resolve().parents[1]
    support = inspect_breakout_completion_support(env)
    if not support.supported:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise RuntimeError(
            "canonical Breakout clear detection is unavailable: "
            f"{support.reason or 'unsupported runtime'}"
        )

    base = getattr(env, "unwrapped", env)
    spec = getattr(env, "spec", None)
    environment_id = getattr(spec, "id", None)
    if environment_id != ENVIRONMENT_ID:
        env.close()
        raise RuntimeError(f"expected {ENVIRONMENT_ID}, received {environment_id!r}")
    get_action_meanings = getattr(base, "get_action_meanings", None)
    if not callable(get_action_meanings):
        env.close()
        raise RuntimeError("ALE action meanings are unavailable")
    action_names = tuple(str(value) for value in get_action_meanings())
    if not {"NOOP", "FIRE", "RIGHT", "LEFT"}.issubset(action_names):
        env.close()
        raise RuntimeError(f"unsupported ALE action meanings: {action_names}")
    action_index = {name: index for index, name in enumerate(action_names)}
    completion = BreakoutCompletionDetector(support)
    episode_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    traced_seed_count = 0
    started_at = time.perf_counter()
    all_latencies: list[float] = []
    try:
        for seed_index, seed in enumerate(protocol.seeds):
            for episode_index in range(protocol.episodes_per_seed):
                episode_seed = seed + episode_index
                observation, _reset_info = env.reset(seed=episode_seed)
                if tuple(np.asarray(observation).shape) != protocol.expected_observation_shape:
                    raise RuntimeError(
                        "raw ALE observation shape mismatch: "
                        f"expected {protocol.expected_observation_shape}, got {np.asarray(observation).shape}"
                    )
                controller = PredictiveBreakoutController(**protocol.controller_options)
                completion.reset()
                frame_origin = read_ale_episode_frame(env)
                if frame_origin is None:
                    raise RuntimeError("ALE episode-frame counter is unavailable")
                episode_return = 0.0
                requested_actions: Counter[str] = Counter({name: 0 for name in action_names})
                ale_input_actions: Counter[str] = Counter({name: 0 for name in action_names})
                auto_fire_reasons: Counter[str] = Counter()
                life_loss_count = 0
                terminated = False
                truncated = False
                current_frame = frame_origin
                requested_trace: list[dict[str, Any]] = []
                has_trace = traced_seed_count < protocol.trace_episodes
                if has_trace:
                    traced_seed_count += 1

                for step_number in range(1, step_limit + 1):
                    decision = controller.select_action(observation)
                    requested_index = action_index[decision.action]
                    requested_actions[decision.action] += 1
                    observation, reward, terminated_raw, truncated_raw, info = env.step(
                        requested_index
                    )
                    reward_value = float(reward)
                    if not math.isfinite(reward_value):
                        raise ValueError("environment reward must be finite")
                    episode_return += reward_value
                    info = info if isinstance(info, Mapping) else {}
                    # Wrapper metadata identifies ALE input, before ALE sticky actions.
                    ale_input_index = info.get("fire_reset_executed_action", requested_index)
                    ale_input_name = _action_name(int(ale_input_index), action_names)
                    ale_input_actions[ale_input_name] += 1
                    auto_fire = bool(info.get("fire_reset_auto", False))
                    if auto_fire:
                        reason = info.get("fire_reset_reason")
                        auto_fire_reasons[str(reason or "unknown")] += 1
                    if bool(info.get("fire_reset_life_loss", False)):
                        life_loss_count += 1

                    terminated = bool(terminated_raw)
                    truncated = bool(truncated_raw)
                    next_frame = read_ale_episode_frame(env)
                    if next_frame is None or next_frame - current_frame != 1:
                        raise RuntimeError(
                            "raw RGB control requires exactly one ALE-native frame per agent action"
                        )
                    current_frame = next_frame
                    elapsed_frames = current_frame - frame_origin
                    lives = read_ale_lives(env)
                    completion.observe(
                        cumulative_score=episode_return,
                        ram_score=read_breakout_score(env),
                        agent_step=step_number,
                        emulator_frame=elapsed_frames,
                        lives_remaining=lives,
                    )

                    if has_trace:
                        ball = decision.observation.ball
                        paddle = decision.observation.paddle
                        requested_trace.append(
                            {
                                "episode_seed": episode_seed,
                                "agent_step": step_number,
                                "ale_emulator_frame": elapsed_frames,
                                "requested_action": decision.action,
                                "ale_input_action": ale_input_name,
                                "auto_fire": auto_fire,
                                "raw_reward": reward_value,
                                "cumulative_score": episode_return,
                                "lives_remaining": lives,
                                "ball_x": ball.x,
                                "ball_y": ball.y,
                                "ball_vx": ball.vx,
                                "ball_vy": ball.vy,
                                "ball_confidence": ball.confidence,
                                "ball_directly_detected": ball.directly_detected,
                                "ball_age_frames": ball.age_frames,
                                "paddle_center_x": paddle.center_x if paddle else None,
                                "predicted_intercept_x": decision.predicted_intercept_x,
                                "time_to_intercept_frames": decision.time_to_intercept_frames,
                                "paddle_error": decision.paddle_error,
                                "decision_latency_ms": decision.decision_latency_ms,
                                "completion_cleared": completion.state.cleared,
                            }
                        )
                    all_latencies.append(decision.decision_latency_ms)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError(
                        "evaluation episode did not finish within "
                        f"{step_limit} steps; refusing to emit a partial result"
                    )

                if completion.state.cleared is True:
                    failure_reason = "canonical_clear_detected"
                elif completion.state.cleared is None:
                    failure_reason = "completion_status_unavailable"
                elif truncated:
                    failure_reason = "time_limit"
                elif terminated:
                    failure_reason = "game_over"
                else:
                    failure_reason = "incomplete"
                diagnostics = controller.diagnostics
                elapsed_frames = current_frame - frame_origin
                row = {
                    "evaluation_seed": seed,
                    "episode_seed": episode_seed,
                    "seed_index": seed_index,
                    "episode_index": episode_index + 1,
                    "raw_score": float(episode_return),
                    "episode_return": float(episode_return),
                    "total_agent_steps": step_number,
                    "total_emulator_frames": elapsed_frames,
                    "terminated": terminated,
                    "truncated": truncated,
                    "time_limit": truncated,
                    "failure_reason": failure_reason,
                    "cleared": completion.state.cleared,
                    "clear_agent_step": completion.state.clear_agent_step,
                    "clear_emulator_frame": completion.state.clear_emulator_frame,
                    "clear_score": completion.state.clear_score,
                    "lives_remaining_at_clear": completion.state.lives_remaining_at_clear,
                    "completion_detection_source": completion.state.completion_detection_source,
                    "auto_fire_count": sum(auto_fire_reasons.values()),
                    "auto_fire_reason_counts": dict(auto_fire_reasons),
                    "life_loss_count": life_loss_count,
                    "requested_action_distribution": dict(requested_actions),
                    "ale_input_action_distribution": dict(ale_input_actions),
                    "ball_detected_frames": diagnostics["ball_detected_frames"],
                    "ball_detection_success_rate": diagnostics["ball_detection_success_rate"],
                    "paddle_detected_frames": diagnostics["paddle_detected_frames"],
                    "paddle_detection_success_rate": diagnostics["paddle_detection_success_rate"],
                    "lost_ball_frames": diagnostics["lost_ball_frames"],
                    "trajectory_prediction_count": diagnostics["trajectory_prediction_count"],
                    "left_right_reversal_count": diagnostics["left_right_reversal_count"],
                    "left_right_reversal_rate": diagnostics["left_right_reversal_rate"],
                    "noop_rate": diagnostics["noop_rate"],
                    "controller_decision_latency_ms": diagnostics["controller_decision_latency_ms"],
                }
                episode_rows.append(row)
                if has_trace:
                    traces.append({"episode_seed": episode_seed, "rows": requested_trace})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    scores = [float(row["raw_score"]) for row in episode_rows]
    clear_rows = [row for row in episode_rows if row["cleared"] is True]
    clear_frames = [int(row["clear_emulator_frame"]) for row in clear_rows]
    total_observations = sum(int(row["ball_detected_frames"] + row["lost_ball_frames"]) for row in episode_rows)
    total_ball_detections = sum(int(row["ball_detected_frames"]) for row in episode_rows)
    total_paddle_observations = sum(int(row["total_agent_steps"]) for row in episode_rows)
    total_paddle_detections = sum(int(row["paddle_detected_frames"]) for row in episode_rows)
    total_predictions = sum(int(row["trajectory_prediction_count"]) for row in episode_rows)
    total_reversals = sum(int(row["left_right_reversal_count"]) for row in episode_rows)
    ale_input_distribution: Counter[str] = Counter({name: 0 for name in action_names})
    requested_distribution: Counter[str] = Counter({name: 0 for name in action_names})
    for row in episode_rows:
        ale_input_distribution.update(row["ale_input_action_distribution"])
        requested_distribution.update(row["requested_action_distribution"])
    failure_counts = Counter(str(row["failure_reason"]) for row in episode_rows)
    latency_values = np.asarray(all_latencies, dtype=np.float64)
    clear_rate = (
        len(clear_rows) / len(episode_rows)
        if all(row["cleared"] is not None for row in episode_rows)
        else None
    )
    config_bytes = protocol.config_path.read_bytes()
    contract_bytes = protocol.contract_path.read_bytes()
    provenance = _source_provenance(root)
    return {
        "schema_version": 1,
        "evaluation_id": protocol.contract_id,
        "policy_type": "deterministic_pixel_predictive_controller",
        "model_id": None,
        "training_required": False,
        "training_seed": None,
        "training_transitions": 0,
        "environment_stochasticity": {
            "episode_seeds": list(protocol.seeds),
            "sticky_action_probability": protocol.contract.sticky_action_probability,
            "note": "No training seed exists for this method; variability comes from seeded ALE episodes and sticky actions.",
        },
        "environment": {
            "environment_id": ENVIRONMENT_ID,
            "contract_id": protocol.contract_id,
            "base_environment_contract_id": protocol.contract.contract_id,
            "base_environment_contract_path": protocol.contract_path.as_posix(),
            "base_environment_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "frame_skip": protocol.contract.frame_skip,
            "frame_stack": None,
            "observation_type": "raw ALE RGB pixels (210, 160, 3)",
            "sticky_action_probability": protocol.contract.sticky_action_probability,
            "fire_reset": True,
            "terminal_on_life_loss": protocol.contract.terminal_on_life_loss,
            "time_limit_semantics": dict(protocol.contract.time_limit_semantics),
            "raw_reward_rule": protocol.contract.raw_reward_rule,
        },
        "evaluation_protocol": {
            "config_path": protocol.config_path.as_posix(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "evaluation_seeds": list(protocol.seeds),
            "episodes_per_seed": protocol.episodes_per_seed,
            "episode_count": len(episode_rows),
            "evaluation_epsilon": 0.0,
            "max_steps_per_episode": step_limit,
            "completion_detector": support.to_dict(),
            "clear_count_semantics": (
                "Audited BreakoutCompletionDetector clear events; repository verified_clears "
                "requires separate contract/source/provenance validation, which this evaluator does not perform."
            ),
            "action_semantics": {
                "requested_action": "Controller decision before FIRE wrapper overrides.",
                "ale_input_action": (
                    "Action sent to ALE after FIRE wrapper overrides, before sticky-action stochasticity; "
                    "the physical action resolved by ALE is not observed."
                ),
            },
        },
        "controller": {
            "name": "visual ball tracking with side-wall-aware paddle-plane intercept",
            "actions": list(CONTROLLER_ACTIONS),
            "options": dict(protocol.controller_options),
            "control_inputs": ["current and previous raw RGB screen pixels"],
            "forbidden_inputs": ["ALE RAM", "emulator object coordinates", "completion detector state"],
            "completion_evaluator_boundary": "BreakoutCompletionDetector is called only by the external evaluation loop.",
        },
        "summary": {
            "mean_score": fmean(scores) if scores else None,
            "median_score": median(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "clear_count": len(clear_rows),
            "clear_rate": clear_rate,
            "best_clear_ale_frame": min(clear_frames) if clear_frames else None,
            "median_clear_ale_frame": median(clear_frames) if clear_frames else None,
            "mean_lives_remaining_at_clear": (
                fmean(int(row["lives_remaining_at_clear"]) for row in clear_rows)
                if clear_rows
                else None
            ),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "ball_detection_success_rate": (
                total_ball_detections / total_observations if total_observations else None
            ),
            "lost_ball_frames": sum(int(row["lost_ball_frames"]) for row in episode_rows),
            "trajectory_prediction_count": total_predictions,
            "paddle_detection_success_rate": (
                total_paddle_detections / total_paddle_observations
                if total_paddle_observations
                else None
            ),
            "requested_action_distribution": dict(requested_distribution),
            "ale_input_action_distribution": dict(ale_input_distribution),
            "left_right_reversal_count": total_reversals,
            "left_right_reversal_rate": (
                total_reversals / sum(requested_distribution.values())
                if sum(requested_distribution.values())
                else 0.0
            ),
            "noop_rate": (
                requested_distribution["NOOP"] / sum(requested_distribution.values())
                if sum(requested_distribution.values())
                else 0.0
            ),
            "controller_decision_latency_ms": {
                "mean": float(np.mean(latency_values)) if latency_values.size else None,
                "median": float(np.median(latency_values)) if latency_values.size else None,
                "p95": float(np.percentile(latency_values, 95)) if latency_values.size else None,
                "max": float(np.max(latency_values)) if latency_values.size else None,
            },
            "evaluation_wall_clock_seconds": elapsed_seconds,
            "evaluation_agent_steps_per_second": (
                sum(int(row["total_agent_steps"]) for row in episode_rows) / elapsed_seconds
            ),
        },
        "episodes": episode_rows,
        "step_traces": traces,
        "source_provenance": provenance,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_clock_seconds": elapsed_seconds,
        },
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    env = payload["environment"]
    lines = [
        "# Issue 25: predictive vision controller evaluation",
        "",
        f"- Evaluation: `{payload['evaluation_id']}`",
        f"- Environment contract: `{env['contract_id']}` (task semantics `{env['base_environment_contract_id']}`)",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['evaluation_protocol']['evaluation_seeds'])}`",
        f"- Episodes: {payload['evaluation_protocol']['episode_count']}",
        f"- Observation: {env['observation_type']}; one ALE frame per controller action",
        "- Training seed: none; all runs use the same deterministic controller configuration.",
        "",
        "Canonical clear detections are events from the audited `BreakoutCompletionDetector`. "
        "They do not establish repository `verified_clears`, which requires separate contract/source/provenance validation.",
        "",
        "`requested_action` is the controller decision; `ale_input_action` is the action sent to ALE after "
        "FIRE wrapper overrides. With sticky-action probability "
        f"{env['sticky_action_probability']}, ALE may repeat the previous action; the physical action resolved by ALE is not observed.",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Mean raw score | {_format_number(summary['mean_score'])} |",
        f"| Median raw score | {_format_number(summary['median_score'])} |",
        f"| Maximum raw score | {_format_number(summary['max_score'])} |",
        f"| Canonical clear detections | {summary['clear_count']} / {payload['evaluation_protocol']['episode_count']} |",
        f"| Canonical clear detection rate | {_format_number(summary['clear_rate'] * 100 if summary['clear_rate'] is not None else None)}% |",
        f"| Best clear ALE frame | {_format_number(summary['best_clear_ale_frame'], 0)} |",
        f"| Median clear ALE frame | {_format_number(summary['median_clear_ale_frame'], 0)} |",
        f"| Mean lives remaining at clear | {_format_number(summary['mean_lives_remaining_at_clear'])} |",
        f"| Ball detection success | {_format_number(summary['ball_detection_success_rate'] * 100 if summary['ball_detection_success_rate'] is not None else None)}% |",
        f"| Lost-ball observations | {summary['lost_ball_frames']} |",
        f"| Paddle detection success | {_format_number(summary['paddle_detection_success_rate'] * 100 if summary['paddle_detection_success_rate'] is not None else None)}% |",
        f"| Intercept predictions | {summary['trajectory_prediction_count']} |",
        f"| Requested actions | `{json.dumps(summary['requested_action_distribution'], sort_keys=True)}` |",
        f"| ALE input actions (after FIRE wrapper) | `{json.dumps(summary['ale_input_action_distribution'], sort_keys=True)}` |",
        f"| LEFT↔RIGHT reversals / rate | {summary['left_right_reversal_count']} / {_format_number(summary['left_right_reversal_rate'] * 100)}% |",
        f"| NOOP rate | {_format_number(summary['noop_rate'] * 100)}% |",
        f"| Controller decision latency, CPU mean / p95 | {_format_number(summary['controller_decision_latency_ms']['mean'], 4)} / {_format_number(summary['controller_decision_latency_ms']['p95'], 4)} ms |",
        f"| Evaluation throughput | {_format_number(summary['evaluation_agent_steps_per_second'], 1)} agent steps/s |",
        "",
        "## Episode outcomes",
        "",
        "| Seed | Score | Clear | ALE frames | Lives at clear | Ball detect | Predictions | Failure reason |",
        "|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["episodes"]:
        lines.append(
            "| {seed} | {score} | {clear} | {frames} | {lives} | {detect} | {predictions} | {reason} |".format(
                seed=row["episode_seed"],
                score=_format_number(row["raw_score"], 1),
                clear="yes" if row["cleared"] is True else "no" if row["cleared"] is False else "unknown",
                frames=_format_number(row["clear_emulator_frame"], 0),
                lives=_format_number(row["lives_remaining_at_clear"], 0),
                detect=_format_number(
                    row["ball_detection_success_rate"] * 100
                    if row["ball_detection_success_rate"] is not None
                    else None,
                    1,
                )
                + "%",
                predictions=row["trajectory_prediction_count"],
                reason=row["failure_reason"],
            )
        )
    lines.extend(
        [
            "",
            "## RL baseline comparison",
            "",
            f"The strongest published RL checkpoint in the repository's experiment record is the Dueling Double DQN trained with seed 2022 and selected at 2.5M environment transitions. Its reported model-selection evaluation mean was 51.4 over 15 episodes; the later 15-episode holdout mean was 30.933 (median 32). Source: [Day 21 final long training]({BASELINE_REPORT_URL}).",
            "",
            "| Metric | Predictive controller | Dueling Double DQN | Comparability |",
            "|---|---:|---:|---|",
            f"| Training required / compute | no / 0 transitions | yes / 2.5M transitions | comparable |",
            f"| Mean raw score | {_format_number(summary['mean_score'])} (this run) | 51.4 selected eval; 30.933 holdout | not directly comparable: RL results use Contract v2 (frame skip 4), controller uses v3 task semantics (frame skip 1) and different seed roles |",
            f"| Maximum score | {_format_number(summary['max_score'])} (this run) | not reported in the source summary | unavailable |",
            f"| Canonical clear detections / clear rate | {summary['clear_count']} / {_format_number(summary['clear_rate'])} | not recorded with the canonical clear detector | unavailable |",
            f"| Best clear ALE frame | {_format_number(summary['best_clear_ale_frame'], 0)} | not recorded | unavailable |",
            f"| Controller / inference cost | CPU mean {_format_number(summary['controller_decision_latency_ms']['mean'], 4)} ms per screen | not benchmarked here on the same machine and evaluation harness | no matched benchmark |",
            "",
            "The score figures are kept as historical context. They do not support a ranking because the older RL evaluation used frame skip 4 and selected/holdout seed sets, while this controller uses one ALE frame per decision and the fixed v3 concrete episode seeds. No DQN checkpoint or compatible v3 RL evaluation artifact is present in this checkout, so the report does not infer an RL clear rate or matched inference cost.",
            "",
            f"This run recorded {summary['clear_count']} / {payload['evaluation_protocol']['episode_count']} canonical clear detections. The controller collected points reliably enough to show that pixel perception and receding-horizon movement work, but these results do not establish that learning is unnecessary for the full-clear objective. The next controller experiment should target paddle-contact angles and brick-lane creation, following the issue's proposed path.",
            "",
            "## Artifacts and provenance",
            "",
            "- `results.json`: protocol, per-episode outcomes, aggregate metrics, detector support, and source provenance.",
            "- `episodes.csv`: one row per fixed episode.",
            "- `diagnostics.json`: perception/control metrics and preserved trace file names.",
            "- `controller_trace_seed_<seed>.csv`: first real ALE episode's per-step pixels-derived state, `requested_action` / `ale_input_action`, score, lives, and external clear outcome.",
            f"- Canonical completion detector: `{payload['evaluation_protocol']['completion_detector']['detector_id']}`; supported={payload['evaluation_protocol']['completion_detector']['supported']}; clear threshold={BREAKOUT_FULL_CLEAR_SCORE}.",
            "",
        ]
    )
    return "\n".join(lines)


def write_vision_evaluation_artifacts(
    payload: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.json"
    episodes_path = destination / "episodes.csv"
    diagnostics_path = destination / "diagnostics.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    episode_rows = list(payload["episodes"])
    episode_columns = (
        "evaluation_seed",
        "episode_seed",
        "seed_index",
        "episode_index",
        "raw_score",
        "total_agent_steps",
        "total_emulator_frames",
        "terminated",
        "truncated",
        "failure_reason",
        "cleared",
        "clear_agent_step",
        "clear_emulator_frame",
        "clear_score",
        "lives_remaining_at_clear",
        "ball_detection_success_rate",
        "paddle_detection_success_rate",
        "lost_ball_frames",
        "trajectory_prediction_count",
        "left_right_reversal_count",
        "left_right_reversal_rate",
        "noop_rate",
        "requested_action_distribution",
        "ale_input_action_distribution",
    )
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=episode_columns)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in episode_columns
                }
            )

    trace_files: list[str] = []
    for trace in payload.get("step_traces", []):
        seed = int(trace["episode_seed"])
        trace_path = destination / f"controller_trace_seed_{seed}.csv"
        rows = list(trace["rows"])
        columns = tuple(rows[0].keys()) if rows else ()
        with trace_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        trace_files.append(trace_path.name)
    diagnostics = {
        "summary": payload["summary"],
        "episodes": [
            {
                "episode_seed": row["episode_seed"],
                "ball_detected_frames": row["ball_detected_frames"],
                "ball_detection_success_rate": row["ball_detection_success_rate"],
                "paddle_detected_frames": row["paddle_detected_frames"],
                "paddle_detection_success_rate": row["paddle_detection_success_rate"],
                "lost_ball_frames": row["lost_ball_frames"],
                "trajectory_prediction_count": row["trajectory_prediction_count"],
                "left_right_reversal_count": row["left_right_reversal_count"],
                "left_right_reversal_rate": row["left_right_reversal_rate"],
                "noop_rate": row["noop_rate"],
                "controller_decision_latency_ms": row["controller_decision_latency_ms"],
            }
            for row in episode_rows
        ],
        "trace_files": trace_files,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "report.md").write_text(_render_report(payload), encoding="utf-8")
    return results_path, episodes_path, diagnostics_path


def run_vision_evaluation(
    *,
    config_path: str | Path = DEFAULT_VISION_CONFIG,
    output_dir: str | Path = "outputs/issue25-vision-controller",
    env_factory: EnvironmentFactory | None = None,
    max_steps_per_episode: int | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    protocol = load_vision_evaluation_protocol(config_path)
    payload = evaluate_predictive_controller(
        protocol,
        env_factory=env_factory,
        max_steps_per_episode=max_steps_per_episode,
    )
    results_path, episodes_path, diagnostics_path = write_vision_evaluation_artifacts(
        payload, output_dir
    )
    return results_path, episodes_path, diagnostics_path, payload


__all__ = [
    "DEFAULT_VISION_CONFIG",
    "VisionEvaluationProtocol",
    "evaluate_predictive_controller",
    "load_vision_evaluation_protocol",
    "make_vision_breakout_env",
    "run_vision_evaluation",
    "write_vision_evaluation_artifacts",
]
