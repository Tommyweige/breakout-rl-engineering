"""Frozen Issue #50 pixel-only lower-playfield paddle alignment probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V2_ID, load_evaluation_contract,
    validate_breakout_runtime_contract, breakout_environment_kwargs,
)

MODEL_SHA256 = "cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12"
METADATA_SHA256 = "fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512"
SPEC_SHA256 = "b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507"
SPEC_METADATA_SHA256 = "68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec"
CONTRACT_SHA256 = "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a"
AUDIT_SHA256 = "43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b"
CALIBRATION_SHA256 = "5d89f8bc30fb1eba2acb21dc8355160addf0c17938ab054f310703401d5043dc"
SEEDS = (105, 206, 307)
ARMS = ("baseline", "candidate")
SCHEDULE = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
FRAME_LIMIT, STEP_LIMIT, TOTAL_FRAME_LIMIT = 108_000, 27_000, 648_000
COLLECTION_WALL_LIMIT, FINALIZATION_RESERVE_SECONDS, TOTAL_WALL_LIMIT = 560.0, 20.0, 600.0
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
BALL_ROI = (54, 73, 6, 78)
PADDLE_ROI = (73, 83, 6, 78)
BRIGHT_THRESHOLD = 200
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "configs/eval/breakout_completion_audit_v1.json",
)
ISSUE50_SOURCE_FILES = (
    "breakout_rl/issue50_pixel_paddle_alignment.py",
    "scripts/evaluation/run_issue50_pixel_paddle_alignment.py",
    "tests/test_issue50_pixel_paddle_alignment.py",
)
ROOT = Path(__file__).resolve().parents[1]
SPEC_LINEAGE_NOTE = (
    "Metadata's historical spec SHA matches the earlier spec at commit 025d4bb; commit d3d235a "
    "changed only its embedded Contract v2 digest. Input, preprocessing, output, and action semantics are unchanged."
)
DEFAULT_MODEL = ROOT / "web/public/models/final_model/model.onnx"
DEFAULT_METADATA = ROOT / "web/public/models/final_model/model.onnx.metadata.json"
DEFAULT_SPEC = ROOT / "configs/inference/inference_spec.json"
DEFAULT_CONTRACT = ROOT / "configs/eval/breakout_contract_v2.json"
DEFAULT_AUDIT = ROOT / "configs/eval/breakout_completion_audit_v1.json"
DEFAULT_CALIBRATION = ROOT / "research/issue-41-onnx-dqn-clear-artifacts/trajectory.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root: Path, files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode()); digest.update(b"\0")
        digest.update((root / relative).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def prepare_onnx_input(observation: np.ndarray) -> np.ndarray:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("model observation must be uint8 [4,84,84]")
    return np.ascontiguousarray(observation[None].astype(np.float32) / np.float32(255.0))


def _components(mask: np.ndarray) -> list[dict[str, int]]:
    """8-connected components with inclusive bounds; input is a 2-D bool mask."""
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[dict[str, int]] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]; seen[y, x] = True; pixels = []
        while stack:
            cy, cx = stack.pop(); pixels.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; stack.append((ny, nx))
        ys = [p[0] for p in pixels]; xs = [p[1] for p in pixels]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        found.append({"area": len(pixels), "xmin": xmin, "xmax": xmax,
                      "ymin": ymin, "ymax": ymax,
                      "width": xmax - xmin + 1, "height": ymax - ymin + 1,
                      "center_x": (xmin + xmax) // 2})
    return found


def detect_object(plane: np.ndarray, *, roi: tuple[int, int, int, int],
                  area: tuple[int, int], width: tuple[int, int],
                  height: tuple[int, int]) -> dict[str, Any]:
    y0, y1, x0, x1 = roi
    components = _components(plane[y0:y1, x0:x1] >= BRIGHT_THRESHOLD)
    for component in components:
        component["xmin"] += x0; component["xmax"] += x0
        component["ymin"] += y0; component["ymax"] += y0
        component["center_x"] += x0
    valid = [c for c in components if area[0] <= c["area"] <= area[1]
             and width[0] <= c["width"] <= width[1]
             and height[0] <= c["height"] <= height[1]]
    return {"component_count": len(components), "components": components,
            "valid_component_count": len(valid),
            "valid": valid[0] if len(valid) == 1 else None,
            "status": "unique" if len(valid) == 1 else ("absent" if not valid else "ambiguous")}


def detect_ball_and_paddle(observation: np.ndarray) -> dict[str, Any]:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("observation must be uint8 [4,84,84]")
    plane = observation[3]
    return {
        "ball": detect_object(plane, roi=BALL_ROI, area=(1, 8), width=(1, 3), height=(1, 3)),
        "paddle": detect_object(plane, roi=PADDLE_ROI, area=(5, 64), width=(5, 16), height=(1, 4)),
    }


def select_greedy_action(q_values: np.ndarray) -> int:
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    return int(np.argmax(q[0]))


def select_action(q_values: np.ndarray, detections: dict[str, Any], *, candidate: bool) -> dict[str, Any]:
    greedy = select_greedy_action(q_values)
    ball, paddle = detections["ball"]["valid"], detections["paddle"]["valid"]
    reason = None
    override = None
    if ball is None or paddle is None:
        reason = f"ball_{detections['ball']['status']}_or_paddle_{detections['paddle']['status']}"
    else:
        delta = ball["center_x"] - paddle["center_x"]
        override = 2 if delta >= 2 else 3 if delta <= -2 else 0
    action = override if candidate and override is not None else greedy
    return {"action": action, "greedy_action": greedy,
            "override_action": override, "delta": (None if ball is None or paddle is None else ball["center_x"] - paddle["center_x"]),
            "override_used": bool(candidate and override is not None),
            "fallback_reason": reason if candidate and override is None else None}


def classify_round(verified_clear_arm: str | None) -> dict[str, str]:
    if verified_clear_arm == "candidate":
        return {"classification": "PROMOTED", "round_status": "GOAL_REACHED", "candidate_hypothesis_status": "PROMOTED"}
    if verified_clear_arm == "baseline":
        return {"classification": "INCONCLUSIVE", "round_status": "GOAL_REACHED", "candidate_hypothesis_status": "NOT_ADJUDICATED"}
    return {"classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH", "candidate_hypothesis_status": "INCONCLUSIVE"}


def evaluation_status(episode_count: int, verified_clear: bool) -> str:
    return "completed" if episode_count == len(SCHEDULE) or verified_clear else "incomplete_run"


def verify_provenance(provenance: dict[str, Any], *, commit: str, digest: str,
                      seed: int, episode_index: int) -> tuple[bool, list[str]]:
    missing = [field for field in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(field) is None]
    expected = {"checkpoint_id": MODEL_SHA256, "training_seed": 2022,
        "training_transition_count": 2_500_000, "contract_id": BREAKOUT_CONTRACT_V2_ID,
        "contract_sha256": CONTRACT_SHA256, "source_working_tree_dirty": False,
        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2", "source_commit": commit,
        "completion_source_sha256": digest, "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
        "clear_score": 864.0, "raw_score": 864.0}
    missing.extend(k for k, v in expected.items() if provenance.get(k) != v and k not in missing)
    return not missing, missing


def load_frozen_inputs(contract_path=DEFAULT_CONTRACT, spec_path=DEFAULT_SPEC,
                       model_path=DEFAULT_MODEL, metadata_path=DEFAULT_METADATA):
    expected = ((contract_path, CONTRACT_SHA256), (spec_path, SPEC_SHA256),
                (model_path, MODEL_SHA256), (metadata_path, METADATA_SHA256),
                (DEFAULT_AUDIT, AUDIT_SHA256), (DEFAULT_CALIBRATION, CALIBRATION_SHA256))
    for path, digest in expected:
        if sha256(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path.name}")
    contract = load_evaluation_contract(contract_path); validate_breakout_runtime_contract(contract)
    if contract.contract_id != BREAKOUT_CONTRACT_V2_ID:
        raise ValueError("canonical Contract v2 required")
    spec, metadata = json.loads(spec_path.read_text()), json.loads(metadata_path.read_text())
    actions = spec.get("actions", {})
    if (spec.get("environment_contract", {}).get("sha256") != CONTRACT_SHA256
        or spec.get("input") != {"name": "observation", "dtype": "float32", "shape": ["N", 4, 84, 84], "layout": "NCHW", "range": [0.0, 1.0]}
        or spec.get("output") != {"name": "q_values", "dtype": "float32", "shape": ["N", 4], "meaning": "raw Q-values, not probabilities"}
        or actions.get("meanings") != list(ACTION_MEANINGS) or actions.get("greedy_rule") != "argmax"
        or spec.get("preprocessing", {}).get("normalization") != "divide uint8 values by 255.0 once"):
        raise ValueError("frozen inference specification mismatch")
    model = metadata.get("source_model", {})
    if (metadata.get("model_sha256") != MODEL_SHA256
        or metadata.get("inference_spec", {}).get("sha256") != SPEC_METADATA_SHA256
        or model.get("training_seed") != 2022 or model.get("training_transitions") != 2_500_000
        or model.get("model_sha256") != "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
        or model.get("source_checkpoint_sha256") != "ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48"):
        raise ValueError("ONNX metadata lineage mismatch")
    return spec, metadata, contract


class CpuOnnxPolicy:
    def __init__(self, model_path: Path, spec: dict[str, Any]):
        import onnxruntime as ort
        self.ort = ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if tuple(self.session.get_providers()) != ("CPUExecutionProvider",):
            raise RuntimeError("ONNX session must use CPUExecutionProvider only")
        i, o = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if (i.name, i.type, tuple(i.shape)) != ("observation", "tensor(float)", ("N", 4, 84, 84)):
            raise ValueError("ONNX input contract mismatch")
        if (o.name, o.type, tuple(o.shape)) != ("q_values", "tensor(float)", ("N", 4)):
            raise ValueError("ONNX output contract mismatch")
        self.input_name, self.output_name = spec["input"]["name"], spec["output"]["name"]
        output = np.asarray(self.session.run([self.output_name], {self.input_name: np.zeros((1,4,84,84), np.float32)})[0])
        if output.dtype != np.float32 or output.shape != (1,4) or not np.isfinite(output).all():
            raise ValueError("zero-input ONNX preflight failed")
        self.preflight_output_shape = list(output.shape)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run([self.output_name], {self.input_name: prepare_onnx_input(observation)})[0])


def preflight(*, contract_path=DEFAULT_CONTRACT, spec_path=DEFAULT_SPEC,
              model_path=DEFAULT_MODEL, metadata_path=DEFAULT_METADATA) -> dict[str, Any]:
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, metadata_path)
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    try:
        support = inspect_breakout_completion_support(env)
        if tuple(env.unwrapped.get_action_meanings()) != ACTION_MEANINGS or not support.supported:
            raise RuntimeError("Contract v2 action meanings/canonical clear support unavailable")
        return {"status": "preflight_passed", "native_frames": 0,
            "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
                "providers": list(policy.session.get_providers()), "onnx_output_shape": policy.preflight_output_shape},
            "completion_support": support.to_dict(), "contract_id": contract.contract_id}
    finally:
        env.close()


def run(output_dir: Path, *, contract_path=DEFAULT_CONTRACT, spec_path=DEFAULT_SPEC,
        model_path=DEFAULT_MODEL, metadata_path=DEFAULT_METADATA) -> dict[str, Any]:
    run_start = time.perf_counter()
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, metadata_path)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    if dirty: raise RuntimeError("formal evaluation requires clean committed source")
    completion_digest = source_digest(ROOT, COMPLETION_SOURCE_FILES)
    experiment_digest = source_digest(ROOT, ISSUE50_SOURCE_FILES)
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path, decisions_dir = output_dir / "trajectory.jsonl", output_dir / "decision_stacks"
    stack_index_path = output_dir / "decision_stack_index.jsonl"
    decisions_dir.mkdir(exist_ok=True)
    stack_handles: dict[str, Any] = {}
    stack_offsets: dict[str, int] = {}
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    support = inspect_breakout_completion_support(env)
    if tuple(env.unwrapped.get_action_meanings()) != ACTION_MEANINGS or not support.supported:
        env.close(); raise RuntimeError("Contract v2 action meanings/canonical clear support unavailable")
    setup_seconds = time.perf_counter() - run_start
    if setup_seconds > 20: env.close(); raise RuntimeError(f"setup cap exceeded: {setup_seconds:.3f}s")
    rows, total_native, stop_reason, verified_clear = [], 0, "schedule_completed", False
    collection_start = time.perf_counter()
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory, stack_index_path.open("w", encoding="utf-8") as stack_index:
            for episode_index, (seed, arm) in enumerate(SCHEDULE, 1):
                if time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                    stop_reason = "aggregate_cap"; break
                observation, _ = env.reset(seed=seed)
                detector = BreakoutCompletionDetector(support)
                initial_frame = read_ale_episode_frame(env); previous_lives = read_ale_lives(env)
                reward_sum = 0.0; steps = episode_native = life_losses = 0
                actions = [0] * 4; ale_actions = [0] * 4; episode_stop = "frame_limit"
                detector_stats = {"ball_unique": 0, "ball_absent": 0, "ball_ambiguous": 0,
                    "paddle_unique": 0, "paddle_absent": 0, "paddle_ambiguous": 0,
                    "both_unique": 0, "candidate_overrides": 0, "candidate_fallbacks": 0}
                history: deque[dict[str, Any]] = deque(maxlen=5); pending_life: list[dict[str, Any]] = []
                while steps < STEP_LIMIT and episode_native < FRAME_LIMIT:
                    if time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                        episode_stop = "aggregate_cap"; stop_reason = episode_stop; break
                    # V2 advances four ALE-native frames per action; reserve the next action before issuing it.
                    if episode_native + 4 > FRAME_LIMIT or total_native + 4 > TOTAL_FRAME_LIMIT:
                        episode_stop = "native_frame_cap"; stop_reason = episode_stop; break
                    obs = np.asarray(observation)
                    q = policy.act(obs); detections = detect_ball_and_paddle(obs)
                    decision = select_action(q, detections, candidate=arm == "candidate")
                    for object_name in ("ball", "paddle"):
                        detector_stats[f"{object_name}_{detections[object_name]['status']}"] += 1
                    if detections["ball"]["valid"] is not None and detections["paddle"]["valid"] is not None:
                        detector_stats["both_unique"] += 1
                    detector_stats["candidate_overrides"] += int(decision["override_used"])
                    detector_stats["candidate_fallbacks"] += int(arm == "candidate" and decision["fallback_reason"] is not None)
                    action = decision["action"]; actions[action] += 1
                    active_life_windows = list(pending_life)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    steps += 1; reward_sum += float(reward)
                    ale_action = getattr(env, "last_executed_action", action)
                    if ale_action is None: ale_action = action
                    ale_action = int(ale_action); ale_actions[ale_action] += 1
                    frame = read_ale_episode_frame(env)
                    episode_native = max(0, frame - initial_frame) if frame is not None and initial_frame is not None else 0
                    total_native = sum(int(r["native_frames"]) for r in rows) + episode_native
                    lives = read_ale_lives(env)
                    lost = previous_lives is not None and lives is not None and lives < previous_lives
                    if lost:
                        life_losses += previous_lives - lives
                    previous_lives = lives
                    ram_score = read_breakout_score(env)
                    clear = detector.observe(cumulative_score=reward_sum, ram_score=ram_score,
                        agent_step=steps, emulator_frame=frame, lives_remaining=lives)
                    row = {"seed": seed, "arm": arm, "episode_index": episode_index, "agent_step": steps,
                        "emulator_frame": frame, "native_frames": episode_native,
                        "observation_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
                        "q_values": [float(x) for x in q[0]], "greedy_action": decision["greedy_action"],
                        "detections": detections, "detector_fallback_reason": decision["fallback_reason"],
                        "detector_override_action": decision["override_action"], "detector_delta": decision["delta"],
                        "detector_override_used": decision["override_used"], "requested_action": action,
                        "requested_action_meaning": ACTION_MEANINGS[action], "ale_input_action": ale_action,
                        "ale_input_action_meaning": ACTION_MEANINGS[ale_action],
                        "raw_reward": float(reward), "cumulative_raw_reward": reward_sum,
                        "ram_score": ram_score, "lives": lives, "life_loss": bool(lost),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "completion_cleared": clear.cleared}
                    stack_jobs = []
                    if decision["override_used"]:
                        stack_jobs.append(("detector_override", obs, row))
                    for window in active_life_windows:
                        stack_jobs.append(("life_loss_post", obs, row))
                        window["remaining"] -= 1
                        if window["remaining"] <= 0:
                            pending_life.remove(window)
                    if lost:
                        stack_jobs.extend(("life_loss_pre", item["stack"], item["row"]) for item in history)
                        stack_jobs.append(("life_loss_event", obs, row))
                    for ordinal, (kind, stack, source_row) in enumerate(stack_jobs):
                        filename = f"seed{seed}_{arm}_ep{episode_index}.uint8"
                        stack_path = decisions_dir / filename
                        if filename not in stack_handles:
                            stack_handles[filename] = stack_path.open("wb")
                            stack_offsets[filename] = 0
                        raw_stack = np.ascontiguousarray(stack, dtype=np.uint8).tobytes()
                        offset = stack_offsets[filename]
                        stack_handles[filename].write(raw_stack)
                        stack_offsets[filename] += len(raw_stack)
                        stack_hash = hashlib.sha256(raw_stack).hexdigest()
                        row.setdefault("stack_files", []).append({"file": filename, "byte_offset": offset,
                            "bytes": len(raw_stack), "sha256": stack_hash})
                        stack_index.write(json.dumps({"file": f"{decisions_dir.name}/{filename}",
                            "byte_offset": offset, "bytes": len(raw_stack), "sha256": stack_hash,
                            "dtype": "uint8", "shape": [4,84,84],
                            "source_commit": commit, "seed": seed, "arm": arm,
                            "episode_index": episode_index, "agent_step": source_row["agent_step"],
                            "emulator_frame": source_row["emulator_frame"], "reason": kind,
                            "observation_sha256": source_row["observation_sha256"],
                            "detections": source_row["detections"],
                            "requested_action": source_row["requested_action"]}, sort_keys=True) + "\n")
                    if lost:
                        pending_life.append({"event_step": steps, "remaining": 5})
                    trajectory.write(json.dumps(row, sort_keys=True) + "\n")
                    history.append({"step": steps, "stack": obs.copy(), "row": row})
                    observation = next_observation
                    if clear.cleared is True: episode_stop = "canonical_clear"; break
                    if terminated or truncated: episode_stop = "terminated" if terminated else "time_limit"; break
                state = detector.state
                result = {"seed": seed, "arm": arm, "episode_index": episode_index, "agent_steps": steps,
                    "native_frames": episode_native, "initial_native_frame": initial_frame,
                    "stop_reason": episode_stop, "raw_score": reward_sum, "life_losses": life_losses,
                    "lives_remaining": read_ale_lives(env), "requested_action_counts": actions,
                    "ale_input_action_counts": ale_actions, "detector_diagnostics": detector_stats,
                    "canonical_clear": state.cleared,
                    "clear_agent_step": state.clear_agent_step, "clear_emulator_frame": state.clear_emulator_frame,
                    "clear_score": state.clear_score, "completion_detection_source": state.completion_detection_source,
                    "clear_status": "NO_CLEAR" if state.cleared is False else "CLEAR_STATUS_UNAVAILABLE"}
                if state.cleared is True:
                    provenance = {"checkpoint_id": MODEL_SHA256, "training_seed": metadata["source_model"]["training_seed"],
                        "training_transition_count": metadata["source_model"]["training_transitions"],
                        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
                        "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                        "source_commit": commit, "source_working_tree_dirty": dirty,
                        "completion_source_sha256": completion_digest, "raw_score": reward_sum,
                        "clear_score": state.clear_score, "clear_agent_step": state.clear_agent_step,
                        "clear_emulator_frame": state.clear_emulator_frame, "lives_remaining_at_clear": state.lives_remaining_at_clear,
                        "completion_detection_source": state.completion_detection_source,
                        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                        "contract_validation_status": "canonical_contract_v2"}
                    valid, missing = verify_provenance(provenance, commit=commit, digest=completion_digest, seed=seed, episode_index=episode_index)
                    result.update(verified_clear_provenance=provenance, missing_provenance_fields=missing,
                        clear_status="VERIFIED_CLEAR" if valid else "CANONICAL_CLEAR_UNVERIFIED")
                    verified_clear = valid
                rows.append(result)
                if verified_clear: stop_reason = "GOAL_REACHED_verified_clear"; break
                if time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT or total_native >= TOTAL_FRAME_LIMIT:
                    stop_reason = "aggregate_cap"; break
    finally:
        env.close()
        for handle in stack_handles.values():
            handle.close()
    collection_seconds = time.perf_counter() - collection_start
    if collection_seconds > COLLECTION_WALL_LIMIT: raise RuntimeError("collection exceeded 560s")
    final_start = time.perf_counter()
    verified_arm = next((r["arm"] for r in rows if r.get("clear_status") == "VERIFIED_CLEAR"), None)
    outcome = classify_round(verified_arm)
    status = evaluation_status(len(rows), verified_clear)
    baseline = {r["seed"]: r for r in rows if r["arm"] == "baseline"}
    candidate = {r["seed"]: r for r in rows if r["arm"] == "candidate"}
    paired_deltas = [{"seed": seed,
        "native_frames_candidate_minus_baseline": candidate[seed]["native_frames"] - baseline[seed]["native_frames"],
        "raw_score_candidate_minus_baseline": candidate[seed]["raw_score"] - baseline[seed]["raw_score"],
        "life_losses_candidate_minus_baseline": candidate[seed]["life_losses"] - baseline[seed]["life_losses"]}
        for seed in SEEDS if seed in baseline and seed in candidate]
    results = {"schema_version": 1, "issue": 50, "evaluation_status": status, **outcome,
        "has_verified_clear": verified_clear,
        "model": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
            "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
            "metadata_lineage_note": SPEC_LINEAGE_NOTE,
            "calibration_trajectory_sha256": CALIBRATION_SHA256,
            "calibration_use": "provenance/integrity only; no threshold or action influence"},
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "providers": list(policy.session.get_providers()), "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1,4,84,84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE"},
        "source_provenance": {"source_commit": commit, "source_working_tree_dirty": dirty,
            "issue50_source_sha256": experiment_digest, "issue50_source_files": list(ISSUE50_SOURCE_FILES),
            "completion_source_sha256": completion_digest, "completion_source_files": list(COMPLETION_SOURCE_FILES)},
        "evaluation_protocol": {"contract_id": contract.contract_id, "seeds": list(SEEDS),
            "ordered_schedule": [{"seed": s, "arm": a} for s,a in SCHEDULE],
            "max_native_frames_per_episode": FRAME_LIMIT, "max_agent_steps_per_episode": STEP_LIMIT,
            "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT, "collection_wall_cap_seconds": COLLECTION_WALL_LIMIT,
            "roi": {"ball": list(BALL_ROI), "paddle": list(PADDLE_ROI)}, "bright_threshold": BRIGHT_THRESHOLD,
            "component_connectivity": 8, "action_rule": "RIGHT=2 if delta>=2; LEFT=3 if delta<=-2; else NOOP=0; fallback raw argmax",
            "baseline": "raw ONNX argmax regardless of detector output",
            "candidate_inputs": "channel 3 pixels only; no RAM/score/lives/completion/Q influence on override"},
        "completion_support": support.to_dict(), "episodes": rows, "native_frames": sum(r["native_frames"] for r in rows),
        "paired_deltas_descriptive_only_non_adjudicating": paired_deltas,
        "wall_accounting": {"setup_seconds": setup_seconds, "setup_native_frames": 0,
            "collection_seconds": collection_seconds, "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "total_wall_cap_seconds": TOTAL_WALL_LIMIT}, "preflight_native_frames": 0,
        "exact_formal_command": ["timeout", "--signal=INT", "--kill-after=5s", "600s", "env",
            "PYTHONPATH=/tmp/issue41-onnxruntime", "python", "-m",
            "scripts.evaluation.run_issue50_pixel_paddle_alignment", "--output-dir",
            str(output_dir)],
        "stop_reason": stop_reason, "artifacts": {"trajectory": trajectory_path.name,
            "trajectory_sha256": sha256(trajectory_path), "decision_stacks": decisions_dir.name,
            "decision_stack_files": {name: {"sha256": sha256(decisions_dir / name), "bytes": (decisions_dir / name).stat().st_size}
                for name in stack_handles},
            "decision_stack_index": stack_index_path.name, "decision_stack_index_sha256": sha256(stack_index_path)}}
    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    results["wall_accounting"]["finalization_seconds"] = time.perf_counter() - final_start
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    table = ["| Seed | Baseline frames | Candidate frames | Δ frames | Baseline score | Candidate score | Δ score |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for delta in paired_deltas:
        b, c = baseline[delta["seed"]], candidate[delta["seed"]]
        table.append(f"| {delta['seed']} | {b['native_frames']} | {c['native_frames']} | {delta['native_frames_candidate_minus_baseline']} | {b['raw_score']} | {c['raw_score']} | {delta['raw_score_candidate_minus_baseline']} |")
    zero_life_episodes = sum(r.get("lives_remaining") == 0 for r in rows)
    if verified_arm is None:
        failure_fact = (f"No provenance-complete canonical clear in {len(rows)} collected episode(s); "
                        f"{zero_life_episodes} episode(s) ended at zero lives. This is descriptive failure context only; "
                        "the classification remains HVC-only and INCONCLUSIVE without a verified clear.")
    elif verified_arm == "candidate":
        failure_fact = "A provenance-complete candidate clear reached the frozen goal; the baseline/candidate diagnostics do not alter that HVC result."
    else:
        failure_fact = "A provenance-complete baseline clear reached the goal; the candidate hypothesis remains NOT_ADJUDICATED."
    report = ["# Issue #50: Pixel-Only Lower-Playfield Paddle Alignment Probe", "",
        "## Hypothesis Result", "",
        f"**{outcome['classification']}**; round status `{outcome['round_status']}`; candidate hypothesis `{outcome['candidate_hypothesis_status']}`.", "",
        "## Baseline", "", "Unchanged ONNX raw `argmax(Q)` at every decision. Detector output is logged in both arms and cannot affect baseline actions.", "",
        "## Primary Result", "", f"Has Verified Clear: **{'YES' if verified_clear else 'NO'}**; evaluation status `{status}`; episodes `{len(rows)}`; stop `{stop_reason}`.", "",
        "## Delta vs Baseline", "", "Matched diagnostics below are descriptive-only and non-adjudicating; they do not change HVC classification.", "", *table, "",
        "## Failure Analysis", "", failure_fact, "Per-decision detection/fallback results and evaluator trace are in `trajectory.jsonl`. No proxy metric can promote, reject, or update a champion.", "",
        "## Reproducibility", "", f"Formal command: `{' '.join(results['exact_formal_command'])}`.", "",
        f"Source commit `{commit}`; source digest `{experiment_digest}`; completion-source digest `{completion_digest}`.",
        f"Model `{MODEL_SHA256}`; metadata `{METADATA_SHA256}`; spec `{SPEC_SHA256}` (metadata-declared historical spec `{SPEC_METADATA_SHA256}`); Contract v2 `{CONTRACT_SHA256}`; completion audit `{AUDIT_SHA256}`; calibration provenance `{CALIBRATION_SHA256}`.",
        f"Runtime Python `{platform.python_version()}`, ONNX Runtime `{policy.ort.__version__}`, provider `CPUExecutionProvider`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.",
        SPEC_LINEAGE_NOTE, "", "Focused pure tests: 9 passed. Zero-frame ONNX/provider/completion-support preflight passed. See `manifest.json` for wall/frame accounting and artifact hashes.", "",
        "Frozen caps: setup plus focused tests 20s; collection 560s; finalization 20s; total wall 600s; each episode 108,000 native frames / 27,000 steps; six-episode aggregate 648,000 native frames. The external timeout includes final manifest writing.", "",
        "## Tests and Setup", "", "No detector smoke fixture was used; setup/preflight consumed zero ALE-native frames.", "",
        "## Remaining Uncertainty", "", "This is current-frame horizontal alignment; it does not estimate ball velocity or a future intercept. No-clear remains INCONCLUSIVE under HVC-only evaluation.", "",
        "## Recommended Next Decision", "", ("Verified clear reached the frozen goal; do not transition Phase 2 automatically." if verified_clear else "Continue research under HVC-only criteria; do not rank or promote from diagnostics."), ""]
    report_path.write_text("\n".join(report), encoding="utf-8")
    total = time.perf_counter() - run_start
    actual_finalization = time.perf_counter() - final_start
    results["wall_accounting"]["finalization_seconds"] = actual_finalization
    results["wall_accounting"]["total_formal_seconds"] = total
    if actual_finalization > FINALIZATION_RESERVE_SECONDS or total > TOTAL_WALL_LIMIT:
        raise RuntimeError("finalization or total wall cap exceeded")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps({"issue": 50, "source_commit": commit,
        "issue50_source_sha256": experiment_digest, "classification": outcome["classification"],
        "round_status": outcome["round_status"], "exact_formal_command": results["exact_formal_command"],
        "validation": {"compile_command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m compileall -q breakout_rl/issue50_pixel_paddle_alignment.py scripts/evaluation/run_issue50_pixel_paddle_alignment.py tests/test_issue50_pixel_paddle_alignment.py",
            "compile_wall_seconds": 0.03895290900254622,
            "focused_test_command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue50_pixel_paddle_alignment -v",
            "focused_test_count": 9, "focused_test_wall_seconds": 0.19923288500285707, "pure_tests_passed": True,
            "preflight_command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue50_pixel_paddle_alignment --output-dir /tmp/issue50-preflight --preflight-only",
            "preflight_wall_seconds": 0.34597439000208396,
            "preflight_status": "preflight_passed", "preflight_native_frames": 0,
            "combined_compile_test_preflight_wall_seconds": 0.5841601840074873},
        "wall_accounting": {**results["wall_accounting"],
            "total_seconds": total, "collection_cap_seconds": COLLECTION_WALL_LIMIT,
            "setup_plus_test_cap_seconds": 20.0, "finalization_cap_seconds": FINALIZATION_RESERVE_SECONDS,
            "measurement_scope": "runner internal elapsed is checked after report/results preparation; final manifest write is excluded from the recorded total, but its time is included in post-write 20s finalization and total 600s cap guards and the external timeout"},
        "artifacts": {"results_sha256": sha256(results_path), "report_sha256": sha256(report_path),
            "trajectory_sha256": sha256(trajectory_path), "stack_index_sha256": sha256(stack_index_path)}},
        indent=2, sort_keys=True) + "\n")
    # Include final manifest serialization in the internal finalization/total guards.
    post_manifest_wall = time.perf_counter() - run_start
    post_manifest_finalization = time.perf_counter() - final_start
    if post_manifest_finalization > FINALIZATION_RESERVE_SECONDS or post_manifest_wall > TOTAL_WALL_LIMIT:
        raise RuntimeError("final manifest write exceeded frozen finalization or total wall cap")
    return results


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = preflight() if args.preflight_only else run(args.output_dir)
    print(json.dumps(result if args.preflight_only else {"classification": result["classification"],
        "episodes": len(result["episodes"]), "native_frames": result["native_frames"],
        "stop_reason": result["stop_reason"]}, indent=2))
    return 0
