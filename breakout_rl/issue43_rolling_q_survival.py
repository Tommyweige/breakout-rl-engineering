"""Frozen Issue #43 paired raw-greedy versus rolling-Q4 ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BreakoutCompletionDetector, BREAKOUT_COMPLETION_DETECTOR_ID,
    BREAKOUT_COMPLETION_SOURCE, inspect_breakout_completion_support,
    read_ale_episode_frame, read_ale_lives, read_breakout_score,
)
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V2_ID, breakout_environment_kwargs,
)
from breakout_rl.issue41_onnx_dqn_clear import (
    ACTION_MEANINGS, AUDIT_SHA256, CONTRACT_SHA256, DEFAULT_AUDIT,
    DEFAULT_METADATA, FRAME_LIMIT as ISSUE41_FRAME_LIMIT, METADATA_SHA256,
    MODEL_SHA256, SPEC_CURRENT_SHA256, SPEC_METADATA_SHA256,
    CpuOnnxPolicy, load_frozen_inputs, sha256, source_identity,
)

SEEDS = (102, 203, 304)
ARM_ORDER = ("raw-greedy", "rolling-q4")
FRAME_LIMIT = 108_000
STEP_LIMIT = 27_000
TOTAL_FRAME_LIMIT = 648_000
TOTAL_WALL_LIMIT = 600.0
COLLECTION_WALL_LIMIT = 530.0
FINALIZATION_RESERVE_SECONDS = 30.0
SMOOTHING_WINDOW = 4
REQUESTED_MODEL = "gpt-6-luna"
REQUESTED_REASONING_EFFORT = "high"
MODEL_ROUTING_VERIFICATION = "UNAVAILABLE"
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "breakout_rl/issue41_onnx_dqn_clear.py",
    "breakout_rl/issue43_rolling_q_survival.py",
    "scripts/evaluation/run_issue43_rolling_q_survival_eval.py",
    "configs/eval/breakout_completion_audit_v1.json",
)


def smooth_q_history(history: list[np.ndarray], current: np.ndarray) -> np.ndarray:
    """Append one finite float32 [1,4] vector and return the rolling mean."""
    q = np.asarray(current)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    history.append(q.copy())
    if len(history) > SMOOTHING_WINDOW:
        del history[:-SMOOTHING_WINDOW]
    # Accumulate and retain float32; no normalization or score-based adjustment.
    return np.mean(np.stack(history, axis=0), axis=0, dtype=np.float32)


def clear_provenance_check(provenance: dict[str, Any], *, commit: str,
                           digest: str, seed: int, episode_index: int) -> tuple[bool, list[str]]:
    missing = [key for key in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(key) is None]
    expected = {
        "checkpoint_id": MODEL_SHA256, "training_seed": 2022,
        "training_transition_count": 2_500_000, "contract_id": BREAKOUT_CONTRACT_V2_ID,
        "contract_sha256": CONTRACT_SHA256, "source_working_tree_dirty": False,
        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2", "source_commit": commit,
        "completion_source_sha256": digest, "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
        "clear_score": 864.0, "raw_score": 864.0,
    }
    missing += [key for key, value in expected.items() if provenance.get(key) != value and key not in missing]
    expected_indexes = ({2 * SEEDS.index(seed) + 1, 2 * SEEDS.index(seed) + 2}
                        if seed in SEEDS else set())
    if episode_index not in expected_indexes:
        missing.append("episode_index")
    return not missing, missing


def completion_source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in COMPLETION_SOURCE_FILES:
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update((root / relative).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def preflight(contract_path: Path, spec_path: Path, model_path: Path) -> dict[str, Any]:
    """Validate the frozen runtime without resetting or stepping ALE."""
    root = Path(__file__).resolve().parents[1]
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, DEFAULT_METADATA)
    if sha256(root / DEFAULT_AUDIT) != AUDIT_SHA256:
        raise ValueError("completion audit hash mismatch")
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    try:
        support = inspect_breakout_completion_support(env)
        meanings = tuple(env.unwrapped.get_action_meanings())
        if meanings != ACTION_MEANINGS or not support.supported:
            raise RuntimeError(f"canonical runtime preflight failed: meanings={meanings}, completion={support.reason}")
    finally:
        env.close()
    return {
        "model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
        "inference_spec_sha256": SPEC_CURRENT_SHA256,
        "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
        "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
        "contract_id": contract.contract_id,
        "runtime_contract_validation": "canonical_contract_v2",
        "onnxruntime": policy.ort.__version__,
        "providers": list(policy.session.get_providers()),
        "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
        "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
        "actual_ale_action_meanings": list(ACTION_MEANINGS),
        "completion_support": support.to_dict(), "native_frames_used": 0,
        "metadata_declared_training_seed": metadata["source_model"]["training_seed"],
        "metadata_declared_training_transitions": metadata["source_model"]["training_transitions"],
    }


def _action_switch_rate(actions: list[int]) -> float | None:
    if len(actions) < 2:
        return None
    return sum(left != right for left, right in zip(actions, actions[1:])) / (len(actions) - 1)


def _top_two_margin(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(np.float32, copy=False))
    return float(np.float32(ordered[-1] - ordered[-2]))


def run(contract_path: Path, spec_path: Path, model_path: Path,
        episode_seeds: tuple[int, ...], arm_order: tuple[str, ...],
        smoothing_window: int, max_frames_per_episode: int, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    if (episode_seeds, arm_order, smoothing_window, max_frames_per_episode) != (
        SEEDS, ARM_ORDER, SMOOTHING_WINDOW, FRAME_LIMIT
    ):
        raise ValueError("Issue #43 frozen seeds, arm order, smoothing window, or frame cap changed")
    if ISSUE41_FRAME_LIMIT != FRAME_LIMIT:
        raise RuntimeError("shared Issue #41 frame contract drifted")
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, DEFAULT_METADATA)
    if sha256(root / DEFAULT_AUDIT) != AUDIT_SHA256:
        raise ValueError("completion audit hash mismatch")
    commit, dirty = source_identity(root)
    if dirty:
        raise RuntimeError("formal evaluation requires a clean committed source tree")
    digest = completion_source_digest(root)
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.jsonl"
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    support = inspect_breakout_completion_support(env)
    meanings = tuple(env.unwrapped.get_action_meanings())
    if meanings != ACTION_MEANINGS or not support.supported:
        env.close()
        raise RuntimeError(f"canonical runtime preflight failed: meanings={meanings}, completion={support.reason}")

    episodes: list[dict[str, Any]] = []
    total_native = 0
    status = "completed"
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for seed_index, seed in enumerate(episode_seeds):
                for arm in arm_order:
                    episode_index = len(episodes) + 1
                    if time.perf_counter() - started >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                        status = "incomplete_run"
                        break
                    observation, _ = env.reset(seed=seed)
                    detector = BreakoutCompletionDetector(support)
                    history: list[np.ndarray] = []
                    actions: list[int] = []
                    raw_margins: list[float] = []
                    smoothed_margins: list[float] = []
                    requested_counts = [0] * 4
                    ale_input_counts = [0] * 4
                    reward_sum = 0.0
                    life_losses = 0
                    previous_lives = read_ale_lives(env)
                    initial_frame = read_ale_episode_frame(env)
                    episode_native = 0
                    steps = 0
                    stop_reason = "frame_limit"
                    while steps < STEP_LIMIT and episode_native < max_frames_per_episode:
                        if time.perf_counter() - started >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                            status, stop_reason = "incomplete_run", "wall_or_aggregate_cap"
                            break
                        _, raw_list = policy.act(np.asarray(observation))
                        raw_q = np.asarray([raw_list], dtype=np.float32)
                        raw_margins.append(_top_two_margin(raw_q[0]))
                        if arm == "rolling-q4":
                            smooth_q = smooth_q_history(history, raw_q)
                            action = int(np.argmax(smooth_q[0]))
                            smooth_list: list[float] | None = [float(v) for v in smooth_q[0]]
                            history_size = len(history)
                            smoothed_margins.append(_top_two_margin(smooth_q[0]))
                        else:
                            action = int(np.argmax(raw_q[0]))
                            smooth_list = None
                            history_size = 0
                        actions.append(action)
                        requested_counts[action] += 1
                        next_observation, reward, terminated, truncated, _ = env.step(action)
                        steps += 1
                        reward_sum += float(reward)
                        ale_input_action = getattr(env, "last_executed_action", action)
                        if ale_input_action is None:
                            ale_input_action = action
                        ale_input_action = int(ale_input_action)
                        ale_input_counts[ale_input_action] += 1
                        frame = read_ale_episode_frame(env)
                        native = max(0, frame - initial_frame) if frame is not None and initial_frame is not None else 0
                        episode_native = native
                        total_native = sum(int(row["native_frames"]) for row in episodes) + episode_native
                        lives = read_ale_lives(env)
                        if previous_lives is not None and lives is not None and lives < previous_lives:
                            life_losses += previous_lives - lives
                        previous_lives = lives
                        ram_score = read_breakout_score(env)
                        clear = detector.observe(cumulative_score=reward_sum, ram_score=ram_score,
                            agent_step=steps, emulator_frame=frame, lives_remaining=lives)
                        trajectory.write(json.dumps({
                            "seed": seed, "arm": arm, "episode_index": episode_index,
                            "agent_step": steps, "emulator_frame": frame, "native_frames": native,
                            "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                            "requested_action": action, "requested_action_meaning": ACTION_MEANINGS[action],
                            "ale_input_action": ale_input_action,
                            "ale_input_action_meaning": ACTION_MEANINGS[ale_input_action],
                            "sticky_resolved_physical_action": None,
                            "raw_q_values": raw_list, "smoothed_q_values": smooth_list,
                            "raw_top_two_q_margin": raw_margins[-1],
                            "smoothed_top_two_q_margin": smoothed_margins[-1] if smooth_list is not None else None,
                            "smoothing_history_vectors": history_size,
                            "raw_reward": float(reward), "cumulative_raw_reward": reward_sum,
                            "ram_score": ram_score, "lives": lives,
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "completion_cleared": clear.cleared,
                        }, sort_keys=True) + "\n")
                        observation = next_observation
                        if clear.cleared is True:
                            stop_reason = "canonical_clear"
                            break
                        if terminated or truncated:
                            stop_reason = "terminated" if terminated else "time_limit"
                            break
                    if steps >= STEP_LIMIT and stop_reason == "frame_limit":
                        stop_reason = "agent_step_limit"
                    clear_state = detector.state
                    episode = {
                        "seed": seed, "evaluation_seed": seed, "episode_seed": seed,
                        "episode_index": episode_index, "arm": arm, "agent_steps": steps,
                        "native_frames": episode_native, "initial_native_frame": initial_frame,
                        "stop_reason": stop_reason, "raw_score": reward_sum,
                        "ram_score": read_breakout_score(env), "life_losses": life_losses,
                        "lives_remaining": read_ale_lives(env),
                        "requested_action_counts": requested_counts,
                        "ale_input_action_counts": ale_input_counts,
                        "requested_action_switch_rate": _action_switch_rate(actions),
                        "median_raw_top_two_q_margin": float(np.median(raw_margins)) if raw_margins else None,
                        "median_smoothed_top_two_q_margin": float(np.median(smoothed_margins)) if smoothed_margins else None,
                        "canonical_clear": clear_state.cleared,
                        "clear_agent_step": clear_state.clear_agent_step,
                        "clear_emulator_frame": clear_state.clear_emulator_frame,
                        "clear_score": clear_state.clear_score,
                        "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                        "completion_detection_source": clear_state.completion_detection_source,
                        "completion_unavailable_reason": clear_state.unavailable_reason,
                    }
                    if clear_state.cleared is True:
                        provenance = {
                            "checkpoint_id": MODEL_SHA256,
                            "training_seed": metadata["source_model"]["training_seed"],
                            "training_transition_count": metadata["source_model"]["training_transitions"],
                            "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
                            "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                            "source_commit": commit, "source_working_tree_dirty": dirty,
                            "completion_source_sha256": digest, "raw_score": reward_sum,
                            "clear_score": clear_state.clear_score,
                            "clear_agent_step": clear_state.clear_agent_step,
                            "clear_emulator_frame": clear_state.clear_emulator_frame,
                            "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                            "completion_detection_source": clear_state.completion_detection_source,
                            "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                            "contract_validation_status": "canonical_contract_v2",
                        }
                        complete, missing = clear_provenance_check(provenance, commit=commit,
                            digest=digest, seed=seed, episode_index=episode_index)
                        episode["verified_clear_provenance"] = provenance
                        episode["missing_provenance_fields"] = missing
                        episode["clear_status"] = "VERIFIED_CLEAR" if complete else "CANONICAL_CLEAR_UNVERIFIED"
                    else:
                        episode["clear_status"] = "NO_CLEAR" if clear_state.cleared is False else "CLEAR_STATUS_UNAVAILABLE"
                    episodes.append(episode)
                    if stop_reason == "wall_or_aggregate_cap":
                        break
                if status != "completed":
                    break
    finally:
        env.close()

    verified_clears = [row["verified_clear_provenance"] for row in episodes if row["clear_status"] == "VERIFIED_CLEAR"]
    has_verified_clear = bool(verified_clears)
    classification = "GOAL_REACHED" if has_verified_clear else "INCONCLUSIVE"
    if len(episodes) != len(SEEDS) * len(ARM_ORDER):
        status = "incomplete_run"
    pairs = []
    by_key = {(row["seed"], row["arm"]): row for row in episodes}
    for seed in SEEDS:
        raw, smooth = by_key.get((seed, "raw-greedy")), by_key.get((seed, "rolling-q4"))
        if raw is None or smooth is None:
            pairs.append({"seed": seed, "status": "incomplete_pair", "raw_greedy": raw, "rolling_q4": smooth})
            continue
        censored = raw["canonical_clear"] is True or smooth["canonical_clear"] is True or raw["stop_reason"] != "terminated" or smooth["stop_reason"] != "terminated"
        diff = smooth["native_frames"] - raw["native_frames"]
        pairs.append({"seed": seed, "status": "duration_censored" if censored else "paired_complete",
            "raw_greedy_native_frames": raw["native_frames"], "rolling_q4_native_frames": smooth["native_frames"],
            "native_frame_difference_rolling_minus_raw": diff,
            "native_frame_percent_difference": (100.0 * diff / raw["native_frames"]) if raw["native_frames"] else None,
            "raw_greedy_life_losses": raw["life_losses"], "rolling_q4_life_losses": smooth["life_losses"],
            "raw_greedy_raw_score": raw["raw_score"], "rolling_q4_raw_score": smooth["raw_score"],
            "raw_greedy_action_switch_rate": raw["requested_action_switch_rate"],
            "rolling_q4_action_switch_rate": smooth["requested_action_switch_rate"],
            "raw_greedy_median_q_margin": raw["median_raw_top_two_q_margin"],
            "rolling_q4_median_q_margin": smooth["median_smoothed_top_two_q_margin"],
            "censoring_explanation": "At least one duration ended at clear, frame/step cap, or runtime cap; survival difference is descriptive only." if censored else None})
    result = {
        "schema_version": 1, "issue": 43, "evaluation_status": status,
        "classification": classification, "has_verified_clear": has_verified_clear,
        "execution_context": {"requested_model": REQUESTED_MODEL,
            "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
            "MODEL_ROUTING_VERIFICATION": MODEL_ROUTING_VERIFICATION},
        "model": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_CURRENT_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
            "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
            "metadata_lineage": {"training_seed": metadata["source_model"]["training_seed"],
                "training_transitions": metadata["source_model"]["training_transitions"],
                "source_model_sha256": metadata["source_model"]["model_sha256"],
                "source_checkpoint_sha256": metadata["source_model"]["source_checkpoint_sha256"],
                "original_pt_independently_rehashed": False},
            "contract_id": contract.contract_id, "detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "contract_validation_status": "canonical_contract_v2"},
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "providers": list(policy.session.get_providers()), "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "model_input_preflight": "zero uint8 [4,84,84], cast float32 and divide by 255 once; no ALE frames"},
        "source_provenance": {"source_commit": commit, "source_working_tree_dirty": dirty,
            "completion_source_sha256": digest, "completion_source_files": list(COMPLETION_SOURCE_FILES),
            "completion_audit_sha256": AUDIT_SHA256},
        "evaluation_protocol": {"contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
            "episode_seeds": list(SEEDS), "episodes_per_seed": 1, "arm_order": list(ARM_ORDER),
            "max_native_frames_per_episode": FRAME_LIMIT, "max_agent_steps_per_episode": STEP_LIMIT,
            "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT, "wall_clock_cap_seconds": TOTAL_WALL_LIMIT,
            "collection_wall_cap_seconds": COLLECTION_WALL_LIMIT,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "smoothing_window": SMOOTHING_WINDOW,
            "rolling_mean_definition": "float32 mean of current raw Q-vector and up to three preceding raw Q-vectors; history resets at each episode",
            "environment_action_meanings": list(ACTION_MEANINGS),
            "observation_contract": "AtariPreprocessing grayscale uint8 84x84; frame_skip=4; stack uint8 [4,84,84]; adapter /255 once",
            "policy_inputs": "pixels only; RAM/score/lives/completion evaluator only"},
        "completion_support": support.to_dict(), "episodes": episodes, "paired_diagnostics": pairs,
        "verified_clears": verified_clears,
        "canonical_clear_unverified": [row for row in episodes if row["clear_status"] == "CANONICAL_CLEAR_UNVERIFIED"],
        "native_frames": total_native, "wall_seconds": time.perf_counter() - started,
        "artifacts": {"trajectory": trajectory_path.name, "trajectory_sha256": sha256(trajectory_path)},
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Issue #43: Rolling Q4 Survival Ablation", "",
        f"**{classification}** — Has Verified Clear: **{'YES' if has_verified_clear else 'NO'}**.",
        "", f"Status: `{status}`; episodes: `{len(episodes)}/6`; native frames: `{total_native}`; wall seconds: `{result['wall_seconds']:.2f}`.",
        "", "A verified clear reaches the Phase 1 milestone after independent Planner validation. If none was verified, the first-clear result is INCONCLUSIVE.",
        "Survival, score, action-switch rate, and Q-vector differences are descriptive diagnostics only; they do not rank a champion or replace the clear gate.",
        "", f"Model `{MODEL_SHA256}`; metadata `{METADATA_SHA256}`; current inference spec `{SPEC_CURRENT_SHA256}` (metadata-declared older spec `{SPEC_METADATA_SHA256}`); Contract v2 `{CONTRACT_SHA256}`; audit `{AUDIT_SHA256}`.",
        f"", f"ONNX Runtime `{policy.ort.__version__}`, providers `{policy.session.get_providers()}`. Source commit `{commit}`, dirty `{dirty}`. Original PyTorch `.pt` bytes were absent and lineage remains metadata-declared.",
        f"Requested execution context: `{REQUESTED_MODEL}` / `{REQUESTED_REASONING_EFFORT}`; `MODEL_ROUTING_VERIFICATION: {MODEL_ROUTING_VERIFICATION}` (no identity introspection is available).",
        "", "## Paired descriptive diagnostics", "", "| Seed | Status | Raw frames | Q4 frames | Difference | Raw lives lost | Q4 lives lost | Raw score | Q4 score | Raw switch rate | Q4 switch rate | Raw median Q margin | Q4 median Q margin |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        report.append(f"| {pair['seed']} | {pair['status']} | {pair.get('raw_greedy_native_frames', '—')} | {pair.get('rolling_q4_native_frames', '—')} | {pair.get('native_frame_difference_rolling_minus_raw', '—')} | {pair.get('raw_greedy_life_losses', '—')} | {pair.get('rolling_q4_life_losses', '—')} | {pair.get('raw_greedy_raw_score', '—')} | {pair.get('rolling_q4_raw_score', '—')} | {pair.get('raw_greedy_action_switch_rate', '—')} | {pair.get('rolling_q4_action_switch_rate', '—')} | {pair.get('raw_greedy_median_q_margin', '—')} | {pair.get('rolling_q4_median_q_margin', '—')} |")
    report += ["", "Requested and wrapper ALE-input actions are stored separately; sticky-resolved physical actions remain unknown. Raw and smoothed float32 Q-vectors and per-step trajectories are in `trajectory.jsonl`; full episode details are in `results.json`.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inference-spec", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--arm-order", required=True)
    parser.add_argument("--smoothing-window", type=int, required=True)
    parser.add_argument("--max-native-frames-per-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight(args.contract, args.inference_spec, args.model), indent=2, sort_keys=True))
        return 0
    result = run(args.contract, args.inference_spec, args.model,
        tuple(int(item) for item in args.episode_seeds.split(",")),
        tuple(args.arm_order.split(",")), args.smoothing_window,
        args.max_native_frames_per_episode, args.output_dir)
    print(json.dumps({"results": str(args.output_dir / "results.json"),
        "classification": result["classification"], "native_frames": result["native_frames"],
        "episodes": [{"seed": row["seed"], "arm": row["arm"], "clear_status": row["clear_status"]} for row in result["episodes"]]}, indent=2))
    return 0
