"""Frozen Issue #45 raw-greedy Contract v2 life-loss observation audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BreakoutCompletionDetector, BREAKOUT_COMPLETION_DETECTOR_ID,
    BREAKOUT_COMPLETION_SOURCE, inspect_breakout_completion_support,
    read_ale_episode_frame, read_ale_lives, read_breakout_score,
)
from breakout_rl.evaluation_contract import BREAKOUT_CONTRACT_V2_ID, breakout_environment_kwargs
from breakout_rl.issue41_onnx_dqn_clear import (
    ACTION_MEANINGS, AUDIT_SHA256, CONTRACT_SHA256, DEFAULT_AUDIT,
    DEFAULT_METADATA, FRAME_LIMIT as ISSUE41_FRAME_LIMIT, METADATA_SHA256,
    MODEL_SHA256, SPEC_CURRENT_SHA256, SPEC_METADATA_SHA256,
    CpuOnnxPolicy, load_frozen_inputs,
    sha256, source_identity,
)

SEEDS = (103, 204, 305)
FRAME_LIMIT = 108_000
STEP_LIMIT = 27_000
TOTAL_FRAME_LIMIT = 324_000
TOTAL_WALL_LIMIT = 600.0
COLLECTION_WALL_LIMIT = 530.0
FINALIZATION_RESERVE_SECONDS = 30.0
WINDOW_DECISIONS = 8
REQUESTED_MODEL = "gpt-6-luna"
REQUESTED_REASONING_EFFORT = "high"
MODEL_ROUTING_VERIFICATION = "UNAVAILABLE"
FORMAL_COMMAND = (
    "timeout --signal=TERM --kill-after=5s 560s env PYTHONPATH=/tmp/issue41-onnxruntime "
    "python -m scripts.evaluation.run_issue45_loss_window_audit_eval "
    "--contract configs/eval/breakout_contract_v2.json "
    "--inference-spec configs/inference/inference_spec.json "
    "--model web/public/models/final_model/model.onnx "
    "--episode-seeds 103,204,305 --loss-window-decisions 8 "
    "--max-native-frames-per-episode 108000 "
    "--output-dir outputs/issue-45-loss-window-audit"
)


def verified_clear_provenance(provenance: dict[str, Any], *, commit: str,
                              digest: str, seed: int, episode_index: int) -> tuple[bool, list[str]]:
    """Apply the frozen full-provenance gate with this issue's seed/index mapping."""
    from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
    from breakout_rl.completion import BREAKOUT_COMPLETION_SOURCE
    missing = [key for key in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(key) is None]
    expected = {
        "checkpoint_id": MODEL_SHA256, "training_seed": 2022,
        "training_transition_count": 2_500_000, "evaluation_seed": seed,
        "episode_seed": seed, "episode_index": episode_index,
        "contract_id": BREAKOUT_CONTRACT_V2_ID, "contract_sha256": CONTRACT_SHA256,
        "source_commit": commit, "source_working_tree_dirty": False,
        "completion_source_sha256": digest, "raw_score": 864.0, "clear_score": 864.0,
        "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2",
    }
    missing.extend(key for key, value in expected.items() if provenance.get(key) != value and key not in missing)
    if seed not in SEEDS or episode_index != SEEDS.index(seed) + 1:
        missing.append("episode_index")
    return not missing, missing


def append_decision_window(ring: deque[dict[str, Any]], *, observation: np.ndarray,
                           seed: int, episode_index: int, step: int,
                           emulator_frame: int | None, q_values: list[float],
                           requested_action: int, ale_input_action: int | None = None) -> None:
    """Keep a lossless copy of one validated model-visible stack and decision metadata."""
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("window observation must be uint8 [4,84,84]")
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("window Q-values must be finite float32 [4]")
    if not 0 <= requested_action < len(ACTION_MEANINGS):
        raise ValueError("requested action outside frozen action meanings")
    ring.append({
        "seed": seed, "episode_index": episode_index, "agent_step": step,
        "emulator_frame": emulator_frame,
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "observation": observation.copy(), "raw_q_values": q.copy(),
        "requested_action": requested_action, "requested_action_meaning": ACTION_MEANINGS[requested_action],
        "ale_input_action": ale_input_action,
        "ale_input_action_meaning": ACTION_MEANINGS[ale_input_action] if ale_input_action is not None else None,
    })
    while len(ring) > WINDOW_DECISIONS:
        ring.popleft()


def serialize_window_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert metadata to JSON-safe values without changing stored uint8 stacks."""
    rows = []
    for row in decisions:
        metadata = {k: v for k, v in row.items() if k != "observation"}
        metadata["raw_q_values"] = np.asarray(metadata["raw_q_values"], dtype=np.float32).tolist()
        metadata["raw_q_dtype"] = "float32"
        rows.append(metadata)
    return rows


def window_index_entry(event: dict[str, Any], array_key: str) -> dict[str, Any]:
    """Bind a pixel array to its complete life-loss event identity and decisions."""
    post = event.get("post_step_observation")
    return {
        "array_key": array_key, "seed": event["seed"], "episode_index": event["episode_index"],
        "life_loss_event": event["life_loss_event"], "life_loss_count": event["life_loss_count"],
        "loss_agent_step": event["agent_step"], "loss_emulator_frame": event["emulator_frame"],
        "decision_metadata": serialize_window_decisions(event["decisions"]),
        "post_step_array_key": array_key + "_post_step" if post is not None else None,
        "post_step_observation_sha256": event.get("post_step_observation_sha256"),
        "post_step_observation_unavailable_reason": event.get("post_step_observation_unavailable_reason"),
    }


def preflight(contract_path: Path, spec_path: Path, model_path: Path) -> dict[str, Any]:
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
        "contract_id": contract.contract_id, "runtime_contract_validation": "canonical_contract_v2",
        "onnxruntime": policy.ort.__version__, "providers": list(policy.session.get_providers()),
        "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
        "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
        "actual_ale_action_meanings": list(meanings), "completion_support": support.to_dict(),
        "native_frames_used": 0,
        "metadata_declared_training_seed": metadata["source_model"]["training_seed"],
        "metadata_declared_training_transitions": metadata["source_model"]["training_transitions"],
    }


def _pack_windows(events: list[dict[str, Any]], output_dir: Path) -> tuple[Path, str]:
    arrays: dict[str, np.ndarray] = {}
    index: list[dict[str, Any]] = []
    for event in events:
        key = f"window_{event['array_index']:04d}"
        arrays[key] = np.stack([row["observation"] for row in event["decisions"]], axis=0).astype(np.uint8, copy=False)
        post = event.get("post_step_observation")
        if post is not None:
            arrays[key + "_post_step"] = post
        index.append(window_index_entry(event, key))
    path = output_dir / "loss_windows.npz"
    np.savez_compressed(path, **arrays)
    (output_dir / "loss_windows_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from PIL import Image, ImageDraw
    for event, meta in zip(events, index):
        stack_array = arrays[meta["array_key"]]
        frames_by_step = stack_array
        tile_w = tile_h = 168
        canvas = Image.new("RGB", (tile_w * 4, (tile_h + 28) * len(frames_by_step)), "white")
        draw = ImageDraw.Draw(canvas)
        for step_index, stack in enumerate(frames_by_step):
            y0 = step_index * (tile_h + 28)
            decision = event["decisions"][step_index]
            draw.text((4, y0 + 4), f"step {decision['agent_step']} action {decision['requested_action']} Q {decision['raw_q_values']}", fill="black")
            for frame_index, frame in enumerate(stack):
                tile = Image.fromarray(frame, mode="L").resize((tile_w, tile_h), Image.Resampling.NEAREST).convert("RGB")
                x, y = frame_index * tile_w, y0 + 28
                canvas.paste(tile, (x, y))
                draw.text((x + 4, y - 22), f"frame {frame_index}", fill="black")
        name = f"loss_window_seed{event['seed']}_event{event['life_loss_event']}.png"
        canvas.save(output_dir / name)
        event["contact_sheet"] = name
    return path, sha256(path)


def run(contract_path: Path, spec_path: Path, model_path: Path, episode_seeds: tuple[int, ...],
        loss_window_decisions: int, max_frames_per_episode: int, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, DEFAULT_METADATA)
    if (episode_seeds, loss_window_decisions, max_frames_per_episode) != (SEEDS, WINDOW_DECISIONS, FRAME_LIMIT):
        raise ValueError("Issue #45 frozen seed order, window size, or frame cap changed")
    if ISSUE41_FRAME_LIMIT != FRAME_LIMIT or contract.contract_id != BREAKOUT_CONTRACT_V2_ID:
        raise RuntimeError("shared Contract v2 frame or identity contract drifted")
    if sha256(root / DEFAULT_AUDIT) != AUDIT_SHA256:
        raise ValueError("completion audit hash mismatch")
    commit, dirty = source_identity(root)
    if dirty:
        raise RuntimeError("formal evaluation requires a clean committed source tree")
    # Reuse the exact source digest fields used by Issue #41's canonical clear gate.
    from breakout_rl.issue41_onnx_dqn_clear import completion_source_digest
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
    loss_events: list[dict[str, Any]] = []
    total_native = 0
    status = "completed"
    verified = False
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for episode_index, seed in enumerate(episode_seeds, start=1):
                if time.perf_counter() - started >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                    status = "incomplete_run"
                    break
                observation, _ = env.reset(seed=seed)
                detector = BreakoutCompletionDetector(support)
                reward_sum = 0.0
                requested_counts, ale_input_counts = [0] * 4, [0] * 4
                life_losses = 0
                previous_lives = read_ale_lives(env)
                initial_frame = read_ale_episode_frame(env)
                episode_native = steps = 0
                stop_reason = "frame_limit"
                ring: deque[dict[str, Any]] = deque()
                episode_events = 0
                while steps < STEP_LIMIT and episode_native < max_frames_per_episode:
                    if time.perf_counter() - started >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                        status, stop_reason = "incomplete_run", "wall_or_aggregate_cap"
                        break
                    raw_action, q_values = policy.act(np.asarray(observation))
                    q = np.asarray(q_values, dtype=np.float32)
                    pre_frame = read_ale_episode_frame(env)
                    next_observation, reward, terminated, truncated, _ = env.step(raw_action)
                    steps += 1
                    reward_sum += float(reward)
                    ale_input_action = getattr(env, "last_executed_action", raw_action)
                    if ale_input_action is None:
                        ale_input_action = raw_action
                    ale_input_action = int(ale_input_action)
                    requested_counts[raw_action] += 1
                    ale_input_counts[ale_input_action] += 1
                    frame = read_ale_episode_frame(env)
                    native = max(0, frame - initial_frame) if frame is not None and initial_frame is not None else 0
                    episode_native = native
                    total_native = sum(int(row["native_frames"]) for row in episodes) + episode_native
                    lives = read_ale_lives(env)
                    append_decision_window(ring, observation=np.asarray(observation), seed=seed,
                        episode_index=episode_index, step=steps, emulator_frame=pre_frame,
                        q_values=q, requested_action=raw_action, ale_input_action=ale_input_action)
                    life_loss_count = (previous_lives - lives) if previous_lives is not None and lives is not None and lives < previous_lives else 0
                    if life_loss_count:
                        life_losses += life_loss_count
                        event_no = episode_events + 1
                        post = np.asarray(next_observation)
                        post_valid = post.dtype == np.uint8 and post.shape == (4, 84, 84)
                        decisions = list(ring)
                        loss_events.append({
                            "seed": seed, "episode_index": episode_index, "life_loss_event": event_no,
                            "life_loss_count": life_loss_count, "agent_step": steps, "emulator_frame": frame,
                            "array_index": len(loss_events), "decisions": decisions,
                            "post_step_observation": post.copy() if post_valid else None,
                            "post_step_observation_sha256": hashlib.sha256(post.tobytes()).hexdigest() if post_valid else None,
                            "post_step_observation_unavailable_reason": None if post_valid else "returned observation did not match uint8 [4,84,84]",
                        })
                        episode_events += 1
                        ring.clear()
                    previous_lives = lives
                    ram_score = read_breakout_score(env)
                    clear = detector.observe(cumulative_score=reward_sum, ram_score=ram_score,
                        agent_step=steps, emulator_frame=frame, lives_remaining=lives)
                    trajectory.write(json.dumps({
                        "seed": seed, "episode_index": episode_index, "agent_step": steps,
                        "emulator_frame": frame, "native_frames": native,
                        "observation_sha256": hashlib.sha256(np.asarray(observation).tobytes()).hexdigest(),
                        "requested_action": raw_action, "requested_action_meaning": ACTION_MEANINGS[raw_action],
                        "ale_input_action": ale_input_action, "ale_input_action_meaning": ACTION_MEANINGS[ale_input_action],
                        "sticky_resolved_physical_action": None, "raw_q_values": q_values,
                        "raw_q_dtype": "float32",
                        "raw_reward": float(reward), "cumulative_raw_reward": reward_sum,
                        "ram_score": ram_score, "lives": lives, "life_loss_count": life_loss_count,
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
                    "episode_index": episode_index, "agent_steps": steps, "native_frames": episode_native,
                    "initial_native_frame": initial_frame, "stop_reason": stop_reason, "raw_score": reward_sum,
                    "ram_score": read_breakout_score(env), "life_losses": life_losses,
                    "captured_loss_windows": episode_events, "lives_remaining": read_ale_lives(env),
                    "canonical_clear": clear_state.cleared, "clear_agent_step": clear_state.clear_agent_step,
                    "clear_emulator_frame": clear_state.clear_emulator_frame, "clear_score": clear_state.clear_score,
                    "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                    "completion_detection_source": clear_state.completion_detection_source,
                    "completion_unavailable_reason": clear_state.unavailable_reason,
                }
                if clear_state.cleared is True:
                    provenance = {
                        "checkpoint_id": MODEL_SHA256, "training_seed": metadata["source_model"]["training_seed"],
                        "training_transition_count": metadata["source_model"]["training_transitions"],
                        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
                        "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                        "source_commit": commit, "source_working_tree_dirty": dirty,
                        "completion_source_sha256": digest, "raw_score": reward_sum,
                        "clear_score": clear_state.clear_score, "clear_agent_step": clear_state.clear_agent_step,
                        "clear_emulator_frame": clear_state.clear_emulator_frame,
                        "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                        "completion_detection_source": clear_state.completion_detection_source,
                        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                        "contract_validation_status": "canonical_contract_v2",
                    }
                    complete, missing = verified_clear_provenance(provenance, commit=commit,
                        digest=digest, seed=seed, episode_index=episode_index)
                    episode["verified_clear_provenance"] = provenance
                    episode["missing_provenance_fields"] = missing
                    episode["clear_status"] = "VERIFIED_CLEAR" if complete else "CANONICAL_CLEAR_UNVERIFIED"
                    verified = complete
                else:
                    episode["clear_status"] = "NO_CLEAR" if clear_state.cleared is False else "CLEAR_STATUS_UNAVAILABLE"
                episodes.append(episode)
                if verified:
                    break
                if stop_reason == "wall_or_aggregate_cap":
                    break
    finally:
        env.close()

    collection_end = time.perf_counter()
    window_path, window_sha = _pack_windows(loss_events, output_dir)
    if verified:
        classification = "PLANNER_VALIDATION_PENDING"
    elif status == "completed" and len(episodes) == len(SEEDS):
        classification = "INCONCLUSIVE"
    else:
        classification, status = "INCONCLUSIVE", "incomplete_run"
    verified_clears = [row["verified_clear_provenance"] for row in episodes if row["clear_status"] == "VERIFIED_CLEAR"]
    result = {
        "schema_version": 1, "issue": 45, "evaluation_status": status,
        "classification": classification, "has_verified_clear": verified,
        "planner_validation_status": "PENDING" if verified else "NOT_REQUIRED",
        "execution_context": {"requested_model": REQUESTED_MODEL, "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
            "MODEL_ROUTING_VERIFICATION": MODEL_ROUTING_VERIFICATION},
        "model": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_CURRENT_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
            "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
            "metadata_lineage": {"training_seed": metadata["source_model"]["training_seed"],
                "training_transitions": metadata["source_model"]["training_transitions"],
                "source_model_sha256": metadata["source_model"]["model_sha256"],
                "source_checkpoint_sha256": metadata["source_model"]["source_checkpoint_sha256"],
                "original_pt_independently_rehashed": False},
            "inference_spec_lineage_limitation": "The metadata-declared older spec hash is preserved; Contract v2 digest-only update leaves input/preprocessing/output/action sections unchanged.",
            "contract_id": contract.contract_id, "detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "contract_validation_status": "canonical_contract_v2"},
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "providers": list(policy.session.get_providers()), "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "model_input_preflight": "zero uint8 [4,84,84], cast float32 and divide by 255 once; no ALE frames"},
        "source_provenance": {"source_commit": commit, "source_working_tree_dirty": dirty,
            "completion_source_sha256": digest, "completion_audit_sha256": AUDIT_SHA256},
        "evaluation_protocol": {"policy": "raw-greedy argmax(raw_q)", "contract_id": contract.contract_id,
            "contract_sha256": CONTRACT_SHA256, "episode_seeds": list(SEEDS), "episodes_per_seed": 1,
            "max_native_frames_per_episode": FRAME_LIMIT, "max_agent_steps_per_episode": STEP_LIMIT,
            "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT, "wall_clock_cap_seconds": TOTAL_WALL_LIMIT,
            "collection_wall_cap_seconds": COLLECTION_WALL_LIMIT, "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "loss_window_decisions": WINDOW_DECISIONS, "formal_command": FORMAL_COMMAND,
            "environment_action_meanings": list(ACTION_MEANINGS),
            "observation_contract": "uint8 [4,84,84] model stacks; adapter divides by 255 once",
            "policy_inputs": "pixels only; RAM/score/lives/completion evaluator only"},
        "completion_support": support.to_dict(), "episodes": episodes,
        "scheduled_seeds_remaining": list(SEEDS[len(episodes):]) if not verified else list(SEEDS[len(episodes):]),
        "verified_clears": verified_clears,
        "canonical_clear_unverified": [row for row in episodes if row["clear_status"] == "CANONICAL_CLEAR_UNVERIFIED"],
        "life_loss_events": [{k: v for k, v in event.items() if k not in ("decisions", "post_step_observation")} for event in loss_events],
        "captured_window_count": len(loss_events), "captured_life_loss_count": sum(e["life_loss_count"] for e in loss_events),
        "native_frames": total_native, "collection_wall_seconds": collection_end - started,
        "wall_seconds": None, "finalization_wall_seconds": None,
        "artifacts": {"trajectory": trajectory_path.name, "trajectory_sha256": sha256(trajectory_path),
            "loss_windows": window_path.name, "loss_windows_sha256": window_sha,
            "loss_windows_index": "loss_windows_index.json", "manifest": "manifest.json"},
    }
    result_path = output_dir / "results.json"
    result["finalization_wall_seconds"] = time.perf_counter() - collection_end
    result["wall_seconds"] = time.perf_counter() - started
    report = ["# Issue #45: Contract v2 Life-Loss Observation Windows", "",
        f"**{classification}** — Has Verified Clear: **{'YES' if verified else 'NO'}**.", "",
        f"Status: `{status}`; episodes: `{len(episodes)}/3`; life-loss windows: `{len(loss_events)}`; native frames: `{total_native}`; wall seconds: `{result['wall_seconds']:.2f}`.",
        (f"Schedule stopped after the first provenance-verified candidate at seed {episodes[-1]['seed']}; remaining seeds `{result['scheduled_seeds_remaining']}` were not run."
         if verified else f"Seeds not run because collection did not complete: `{result['scheduled_seeds_remaining']}`."),
        "If no clear has complete provenance, the primary result is INCONCLUSIVE. This runner records a provenance-complete candidate as pending Planner validation; only independent Planner review may set GOAL_REACHED.",
        "Life-loss windows contain exact uint8 model-input stacks and per-step raw Q/action metadata. Contact sheets are derived offline; they are diagnostic and never clear evidence.",
        f"Model `{MODEL_SHA256}`; metadata `{METADATA_SHA256}`; current inference spec `{SPEC_CURRENT_SHA256}` (metadata-declared older spec `{SPEC_METADATA_SHA256}`); Contract v2 `{CONTRACT_SHA256}`; audit `{AUDIT_SHA256}`.",
        f"ONNX Runtime `{policy.ort.__version__}`, providers `{policy.session.get_providers()}`. Source commit `{commit}`, dirty `{dirty}`. Original PyTorch `.pt` bytes were absent and lineage remains metadata-declared.",
        f"Requested execution context: `{REQUESTED_MODEL}` / `{REQUESTED_REASONING_EFFORT}`; `MODEL_ROUTING_VERIFICATION: {MODEL_ROUTING_VERIFICATION}`.",
        "", "## Life-loss windows", "", "| Seed | Episode | Event | Loss step | Frame | Decisions captured | Post-step observation |", "|---:|---:|---:|---:|---:|---:|:---:|"]
    for event in loss_events:
        report.append(f"| {event['seed']} | {event['episode_index']} | {event['life_loss_event']} | {event['agent_step']} | {event['emulator_frame']} | {len(event['decisions'])} | {'yes' if event['post_step_observation'] is not None else 'no'} |")
    report += ["", f"Lossless arrays: `{window_path.name}`; index binds each stack to seed, episode, event, step/frame, Q-values, requested action, and wrapper ALE-input action.", "RAM, score, lives, completion, and screenshots are evaluator diagnostics only; sticky-resolved physical actions remain unknown.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    result["finalization_wall_seconds"] = time.perf_counter() - collection_end
    result["wall_seconds"] = time.perf_counter() - started
    result["artifacts"]["loss_windows_index_sha256"] = sha256(output_dir / "loss_windows_index.json")
    result["artifacts"]["contact_sheets"] = [event["contact_sheet"] for event in loss_events]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contact_sheet_hashes = {event["contact_sheet"]: sha256(output_dir / event["contact_sheet"])
                            for event in loss_events}
    manifest = {
        "issue": 45, "classification": classification,
        "planner_validation_status": result["planner_validation_status"],
        "formal_command": FORMAL_COMMAND, "source_commit": commit,
        "completion_source_sha256": digest, "model_sha256": MODEL_SHA256,
        "metadata_sha256": METADATA_SHA256, "inference_spec_sha256": SPEC_CURRENT_SHA256,
        "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
        "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
        "runtime": result["runtime"], "native_frames": total_native,
        "collection_wall_seconds": result["collection_wall_seconds"],
        "finalization_wall_seconds": result["finalization_wall_seconds"],
        "total_wall_seconds": result["wall_seconds"],
        "episodes": [{"seed": row["seed"], "episode_index": row["episode_index"],
            "native_frames": row["native_frames"], "clear_status": row["clear_status"],
            "stop_reason": row["stop_reason"]} for row in episodes],
        "scheduled_seeds_remaining": result["scheduled_seeds_remaining"],
        "artifacts_sha256": {"trajectory.jsonl": result["artifacts"]["trajectory_sha256"],
            "loss_windows.npz": window_sha,
            "loss_windows_index.json": result["artifacts"]["loss_windows_index_sha256"],
            "results.json": sha256(result_path), "report.md": sha256(output_dir / "report.md"),
            **contact_sheet_hashes},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inference-spec", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--loss-window-decisions", type=int, required=True)
    parser.add_argument("--max-native-frames-per-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight(args.contract, args.inference_spec, args.model), indent=2, sort_keys=True))
        return 0
    result = run(args.contract, args.inference_spec, args.model,
        tuple(int(item) for item in args.episode_seeds.split(",")), args.loss_window_decisions,
        args.max_native_frames_per_episode, args.output_dir)
    print(json.dumps({"results": str(args.output_dir / "results.json"),
        "classification": result["classification"], "native_frames": result["native_frames"],
        "episodes": [{"seed": row["seed"], "clear_status": row["clear_status"]} for row in result["episodes"]]}, indent=2))
    return 0
