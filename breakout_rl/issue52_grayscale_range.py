"""Frozen Issue #52 Contract v2 grayscale range reachability diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_COMPLETION_DETECTOR_ID, BREAKOUT_COMPLETION_SOURCE,
    BreakoutCompletionDetector, inspect_breakout_completion_support,
    read_ale_episode_frame, read_ale_lives, read_breakout_score,
)
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V2_ID, breakout_environment_kwargs,
    load_evaluation_contract, validate_breakout_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12"
METADATA_SHA256 = "fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512"
SPEC_SHA256 = "b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507"
SPEC_METADATA_SHA256 = "68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec"
CONTRACT_SHA256 = "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a"
SEED = 408
FRAME_CAP = 350
DECISION_CAP = 87
THRESHOLD = 200
BALL_ROI = (54, 73, 6, 78)
PADDLE_ROI = (73, 83, 6, 78)
QUANTILES = (0.50, 0.90, 0.95, 0.99)
SOURCE_FILES = (
    "breakout_rl/issue52_grayscale_range.py",
    "scripts/evaluation/run_issue52_grayscale_range.py",
    "tests/test_issue52_grayscale_range.py",
    "breakout_env.py", "breakout_rl/completion.py",
    "breakout_rl/evaluation_contract.py", "breakout_rl/evaluation_artifacts.py",
)
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "configs/eval/breakout_completion_audit_v1.json",
)
EXACT_COMMAND = (
    "timeout --signal=INT --kill-after=5s 90s env PYTHONPATH=/tmp/issue41-onnxruntime "
    "python -m scripts.evaluation.run_issue52_grayscale_range "
    "--output-dir research/issue-52-grayscale-range-artifacts"
)
TEST_COMMAND = (
    "env PYTHONPATH=/tmp/issue41-onnxruntime python -c 'import time,unittest; "
    "start=time.perf_counter(); result=unittest.TextTestRunner(verbosity=2).run("
    "unittest.defaultTestLoader.loadTestsFromName(\"tests.test_issue52_grayscale_range\")); "
    "print(f\"focused_test_elapsed_seconds={time.perf_counter()-start:.3f}\"); "
    "raise SystemExit(not result.wasSuccessful())'"
)
PREFLIGHT_COMMAND = (
    "timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime "
    "python -m scripts.evaluation.run_issue52_grayscale_range "
    "--output-dir research/issue-52-grayscale-range-artifacts --preflight-only"
)
PRE_RUN_VALIDATION = {
    "focused_test_command": TEST_COMMAND,
    "focused_test_count": 8,
    "focused_test_wall_seconds": 0.161,
    "compile_command": "python -m py_compile breakout_rl/issue52_grayscale_range.py scripts/evaluation/run_issue52_grayscale_range.py tests/test_issue52_grayscale_range.py",
    "compile_result": "passed",
    "diff_check_command": "git diff --check",
    "diff_check_result": "passed",
    "zero_frame_preflight_command": PREFLIGHT_COMMAND,
    "zero_frame_preflight_wall_seconds": 0.369,
    "zero_frame_preflight_native_frames": 0,
    "zero_frame_preflight_result": "passed",
    "zero_frame_preflight_runtime": {"python": "3.12.14", "numpy": "2.3.5",
        "onnxruntime": "1.22.1", "providers": ["CPUExecutionProvider"],
        "input": ["N", 4, 84, 84], "output": [1, 4]},
    "formal_collection_started_before_checkpoint": False,
}
DEFAULT_MODEL = ROOT / "web/public/models/final_model/model.onnx"
DEFAULT_METADATA = ROOT / "web/public/models/final_model/model.onnx.metadata.json"
DEFAULT_SPEC = ROOT / "configs/inference/inference_spec.json"
DEFAULT_CONTRACT = ROOT / "configs/eval/breakout_contract_v2.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_digest(root: Path = ROOT, files: tuple[str, ...] = SOURCE_FILES) -> str:
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode()); digest.update(b"\0")
        digest.update((root / name).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def roi_slice(plane: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    if not isinstance(plane, np.ndarray) or plane.dtype != np.uint8 or plane.shape != (84, 84):
        raise ValueError("plane must be uint8 [84,84]")
    y0, y1, x0, x1 = roi
    if not (0 <= y0 < y1 <= 84 and 0 <= x0 < x1 <= 84):
        raise ValueError("ROI must be a non-empty in-bounds half-open rectangle")
    return plane[y0:y1, x0:x1]


def threshold_reached(plane: np.ndarray, threshold: int = THRESHOLD) -> bool:
    return bool(np.any(roi_slice(plane, BALL_ROI) >= threshold)
                or np.any(roi_slice(plane, PADDLE_ROI) >= threshold))


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.dtype != np.uint8 or arr.size == 0:
        raise ValueError("quantile input must be non-empty uint8")
    quantiles = np.quantile(arr, QUANTILES, method="linear")
    return {"min": int(arr.min()), "max": int(arr.max()),
            "p50": float(quantiles[0]), "p90": float(quantiles[1]),
            "p95": float(quantiles[2]), "p99": float(quantiles[3]),
            "quantile_method": "linear"}


def observation_summaries(observation: np.ndarray) -> dict[str, Any]:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("observation must be uint8 [4,84,84]")
    plane = observation[3]
    return {"channel_3_full_plane": quantile_summary(plane),
            "channel_3_ball_roi": quantile_summary(roi_slice(plane, BALL_ROI)),
            "channel_3_paddle_roi": quantile_summary(roi_slice(plane, PADDLE_ROI))}


def verify_stack_index(index: list[dict[str, Any]], stack_stream: bytes) -> bool:
    expected_offset = 0
    width = 4 * 84 * 84
    for row in index:
        offset, length = row.get("byte_offset"), row.get("byte_length")
        if offset != expected_offset or length != width:
            return False
        stack = stack_stream[offset:offset + length]
        if len(stack) != width or sha256_bytes(stack) != row.get("stack_sha256"):
            return False
        expected_offset += width
    return expected_offset == len(stack_stream)


def prepare_onnx_input(observation: np.ndarray) -> np.ndarray:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("model observation must be uint8 [4,84,84]")
    return np.ascontiguousarray(observation[None].astype(np.float32) / np.float32(255.0))


def select_greedy_action(q_values: np.ndarray) -> int:
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    return int(np.argmax(q[0]))


def complete_clear_provenance(provenance: dict[str, Any], *, commit: str,
                              digest: str, seed: int, episode_index: int) -> tuple[bool, list[str]]:
    missing = [field for field in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(field) is None]
    expected = {"checkpoint_id": MODEL_SHA256, "training_seed": 2022,
        "training_transition_count": 2_500_000, "evaluation_seed": seed,
        "episode_seed": seed, "episode_index": episode_index,
        "contract_id": BREAKOUT_CONTRACT_V2_ID, "contract_sha256": CONTRACT_SHA256,
        "source_commit": commit, "source_working_tree_dirty": False,
        "completion_source_sha256": digest, "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2", "clear_score": 864.0,
        "raw_score": 864.0}
    invalid = [key for key, value in expected.items() if provenance.get(key) != value]
    missing.extend(key for key in invalid if key not in missing)
    return not missing, missing


def classify_clear_state(canonical_clear_detected: bool, verified_clear: bool) -> dict[str, str]:
    if verified_clear and not canonical_clear_detected:
        raise ValueError("a verified clear requires a canonical clear detector result")
    if verified_clear:
        round_status = "GOAL_REACHED"
    elif canonical_clear_detected:
        round_status = "HUMAN_REVIEW_REQUIRED"
    else:
        round_status = "CONTINUE_RESEARCH"
    return {"classification": "INCONCLUSIVE", "round_status": round_status}


def setup_test_budget(formal_setup_seconds: float) -> dict[str, float]:
    cap = 20.0
    pre_run = (PRE_RUN_VALIDATION["focused_test_wall_seconds"]
        + PRE_RUN_VALIDATION["zero_frame_preflight_wall_seconds"])
    consumed = pre_run + formal_setup_seconds
    return {"cap": cap, "pre_run_consumed": pre_run,
        "formal_setup_preflight": formal_setup_seconds, "total_consumed": consumed,
        "remaining": cap - consumed}


def load_frozen_inputs(contract_path: Path = DEFAULT_CONTRACT, spec_path: Path = DEFAULT_SPEC,
                       model_path: Path = DEFAULT_MODEL, metadata_path: Path = DEFAULT_METADATA):
    expected = ((contract_path, CONTRACT_SHA256), (spec_path, SPEC_SHA256),
                (model_path, MODEL_SHA256), (metadata_path, METADATA_SHA256))
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path.name}")
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != BREAKOUT_CONTRACT_V2_ID:
        raise ValueError("canonical Contract v2 required")
    spec, metadata = json.loads(spec_path.read_text()), json.loads(metadata_path.read_text())
    if (spec.get("environment_contract", {}).get("sha256") != CONTRACT_SHA256
            or spec.get("input") != {"name": "observation", "dtype": "float32", "shape": ["N", 4, 84, 84], "layout": "NCHW", "range": [0.0, 1.0]}
            or spec.get("output") != {"name": "q_values", "dtype": "float32", "shape": ["N", 4], "meaning": "raw Q-values, not probabilities"}
            or spec.get("actions", {}).get("meanings") != ["NOOP", "FIRE", "RIGHT", "LEFT"]
            or spec.get("actions", {}).get("greedy_rule") != "argmax"
            or spec.get("preprocessing", {}).get("normalization") != "divide uint8 values by 255.0 once"):
        raise ValueError("frozen inference specification mismatch")
    source_model = metadata.get("source_model", {})
    if (metadata.get("model_sha256") != MODEL_SHA256
            or metadata.get("inference_spec", {}).get("sha256") != SPEC_METADATA_SHA256
            or source_model.get("training_seed") != 2022
            or source_model.get("training_transitions") != 2_500_000
            or source_model.get("model_sha256") != "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
            or source_model.get("source_checkpoint_sha256") != "ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48"):
        raise ValueError("frozen model metadata lineage mismatch")
    return spec, metadata, contract


class CpuOnnxPolicy:
    def __init__(self, model_path: Path, spec: dict[str, Any]):
        import onnxruntime as ort
        self.ort = ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if tuple(self.session.get_providers()) != ("CPUExecutionProvider",):
            raise RuntimeError("ONNX Runtime must use CPUExecutionProvider only")
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if (inp.name, inp.type, tuple(inp.shape)) != ("observation", "tensor(float)", ("N", 4, 84, 84)):
            raise ValueError("ONNX input contract mismatch")
        if (out.name, out.type, tuple(out.shape)) != ("q_values", "tensor(float)", ("N", 4)):
            raise ValueError("ONNX output contract mismatch")
        self.input_name, self.output_name = spec["input"]["name"], spec["output"]["name"]
        zero = np.asarray(self.session.run([self.output_name], {self.input_name: np.zeros((1, 4, 84, 84), np.float32)})[0])
        if zero.dtype != np.float32 or zero.shape != (1, 4) or not np.isfinite(zero).all():
            raise ValueError("zero-input ONNX preflight failed")
        self.output_shape = list(zero.shape)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run([self.output_name], {self.input_name: prepare_onnx_input(observation)})[0])


def preflight() -> dict[str, Any]:
    spec, _, contract = load_frozen_inputs()
    policy = CpuOnnxPolicy(DEFAULT_MODEL.resolve(), spec)
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    try:
        support = inspect_breakout_completion_support(env)
        meanings = tuple(env.unwrapped.get_action_meanings())
        if meanings != ("NOOP", "FIRE", "RIGHT", "LEFT") or not support.supported:
            raise RuntimeError("Contract v2 runtime/completion preflight failed")
        return {"status": "passed", "ale_native_frames": 0,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                "onnxruntime": policy.ort.__version__, "providers": list(policy.session.get_providers()),
                "onnx_input": ["N", 4, 84, 84], "onnx_output": policy.output_shape},
            "action_meanings": meanings, "completion_support": support.to_dict(),
            "contract_id": contract.contract_id}
    finally:
        env.close()


def run(output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    source_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True,
                                capture_output=True, text=True).stdout.strip())
    if dirty:
        raise RuntimeError("formal run requires clean committed source")
    spec, metadata, contract = load_frozen_inputs()
    policy = CpuOnnxPolicy(DEFAULT_MODEL.resolve(), spec)
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    support = inspect_breakout_completion_support(env)
    if tuple(env.unwrapped.get_action_meanings()) != ("NOOP", "FIRE", "RIGHT", "LEFT") or not support.supported:
        env.close()
        raise RuntimeError("canonical runtime preflight failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.jsonl"
    stack_path = output_dir / "observation_stacks.uint8"
    index_path = output_dir / "observation_index.jsonl"
    observations: list[dict[str, Any]] = []
    clear_provenance: dict[str, Any] | None = None
    completion_digest = source_digest(ROOT, COMPLETION_SOURCE_FILES)
    setup_seconds = time.perf_counter() - started
    setup_budget = setup_test_budget(setup_seconds)
    if setup_budget["total_consumed"] > setup_budget["cap"]:
        env.close()
        raise RuntimeError(f"focused-test plus setup/preflight cap exceeded: {setup_budget['total_consumed']:.3f}s")
    collection_start = time.perf_counter()
    stop_reason = "decision_cap"
    native_frames = decisions = 0
    raw_score = 0.0
    final_lives = None
    initial_frame = None
    final_frame = None
    verified = False
    state = None
    try:
        observation, reset_info = env.reset(seed=SEED)
        detector = BreakoutCompletionDetector(support)
        initial_frame = read_ale_episode_frame(env)
        lives = read_ale_lives(env)
        cumulative_score = 0.0
        with trajectory_path.open("w", encoding="utf-8") as trajectory, stack_path.open("wb") as stacks, index_path.open("w", encoding="utf-8") as index_file:
            while decisions < DECISION_CAP:
                if time.perf_counter() - collection_start >= 60:
                    stop_reason = "collection_wall_cap"
                    break
                if native_frames + 4 > FRAME_CAP:
                    stop_reason = "native_frame_cap"
                    break
                obs = np.asarray(observation)
                if obs.dtype != np.uint8 or obs.shape != (4, 84, 84):
                    raise ValueError("environment observation violated uint8 [4,84,84]")
                q = policy.act(obs)
                action = select_greedy_action(q)
                stack_bytes = obs.tobytes(order="C")
                summary = observation_summaries(obs)
                idx = {"source_commit": source_commit, "seed": SEED, "episode_index": 1,
                    "decision_index": decisions, "agent_step": decisions + 1,
                    "emulator_frame": read_ale_episode_frame(env), "byte_offset": len(observations) * len(stack_bytes),
                    "byte_length": len(stack_bytes), "stack_sha256": sha256_bytes(stack_bytes),
                    "q_values": [float(v) for v in q[0]], "raw_greedy_action": action,
                    "pixel_summaries": summary, "completion_state_before_action":
                        None if detector.state.cleared is None else bool(detector.state.cleared)}
                stacks.write(stack_bytes); index_file.write(json.dumps(idx, sort_keys=True) + "\n")
                observations.append(idx)
                threshold = threshold_reached(obs[3])
                next_observation, reward, terminated, truncated, info = env.step(action)
                decisions += 1; cumulative_score += float(reward)
                frame = read_ale_episode_frame(env); final_frame = frame
                native_frames = max(0, frame - initial_frame) if frame is not None and initial_frame is not None else decisions * 4
                current_lives = read_ale_lives(env)
                ram_score = read_breakout_score(env)
                state = detector.observe(cumulative_score=cumulative_score, ram_score=ram_score,
                    agent_step=decisions, emulator_frame=frame, lives_remaining=current_lives)
                row = {**idx, "threshold_reached_in_frozen_roi": threshold,
                    "reward": float(reward), "cumulative_raw_score": cumulative_score,
                    "ram_score": ram_score, "lives": current_lives,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "canonical_clear_state": state.cleared}
                trajectory.write(json.dumps(row, sort_keys=True) + "\n")
                observation = next_observation; lives = current_lives
                if state.cleared is True:
                    stop_reason = "canonical_clear"
                    break
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break
            else:
                stop_reason = "decision_cap"
        final_lives = lives
        if state is not None and state.cleared is True:
            clear_provenance = {"checkpoint_id": MODEL_SHA256, "training_seed": metadata["source_model"]["training_seed"],
                "training_transition_count": metadata["source_model"]["training_transitions"],
                "evaluation_seed": SEED, "episode_seed": SEED, "episode_index": 1,
                "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                "source_commit": source_commit, "source_working_tree_dirty": dirty,
                "completion_source_sha256": completion_digest, "raw_score": cumulative_score,
                "clear_score": state.clear_score, "clear_agent_step": state.clear_agent_step,
                "clear_emulator_frame": state.clear_emulator_frame,
                "lives_remaining_at_clear": state.lives_remaining_at_clear,
                "completion_detection_source": state.completion_detection_source,
                "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                "contract_validation_status": "canonical_contract_v2"}
            verified, missing = complete_clear_provenance(clear_provenance, commit=source_commit,
                digest=completion_digest, seed=SEED, episode_index=1)
            clear_provenance["missing_provenance_fields"] = missing
    finally:
        env.close()
    collection_seconds = time.perf_counter() - collection_start
    if collection_seconds > 60 or time.perf_counter() - started > 80:
        raise RuntimeError("collection or finalization reserve cap exceeded")
    finalization_start = time.perf_counter()
    stack_bytes = stack_path.read_bytes()
    if not verify_stack_index(observations, stack_bytes):
        raise RuntimeError("observation index does not bind exact stack stream")
    reach_count = sum(bool(row["threshold_reached_in_frozen_roi"]) for row in _read_jsonl(trajectory_path))
    n = len(observations)
    all_pixels = np.frombuffer(stack_bytes, dtype=np.uint8).reshape(n, 4, 84, 84)[:, 3]
    aggregate = {"channel_3_full_plane": quantile_summary(all_pixels),
        "channel_3_ball_roi": quantile_summary(all_pixels[:, BALL_ROI[0]:BALL_ROI[1], BALL_ROI[2]:BALL_ROI[3]]),
        "channel_3_paddle_roi": quantile_summary(all_pixels[:, PADDLE_ROI[0]:PADDLE_ROI[1], PADDLE_ROI[2]:PADDLE_ROI[3]])} if n else None
    result = {"schema_version": 1, "issue": 52, "seed": SEED, "decisions": n,
        "native_frames": native_frames, "frame_cap": FRAME_CAP, "decision_cap": DECISION_CAP,
        "stop_reason": stop_reason, "threshold": THRESHOLD,
        "frozen_rois": {"ball": list(BALL_ROI), "paddle": list(PADDLE_ROI)},
        "reachability": {"R": reach_count, "N": n, "fraction": reach_count / n if n else None,
            "interpretation": "not reached in sample" if n and reach_count == 0 else "reached at least once in sample" if n else "unavailable"},
        "aggregate_channel_3_summaries": aggregate, "canonical_clear_state": None if state is None else state.cleared,
        "canonical_clear_detected": bool(state is not None and state.cleared is True),
        "has_verified_clear": verified,
        "hvc": "YES" if verified else "NO",
        **classify_clear_state(bool(state is not None and state.cleared is True), verified),
        "diagnostic_result": "INCONCLUSIVE", "clear_provenance": clear_provenance,
        "exact_run_command": EXACT_COMMAND,
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
            "onnxruntime": policy.ort.__version__, "providers": list(policy.session.get_providers())},
        "source": {"commit": source_commit, "dirty": dirty, "source_digest": source_digest(),
            "source_files": list(SOURCE_FILES), "completion_source_digest": completion_digest,
            "completion_source_files": list(COMPLETION_SOURCE_FILES)},
        "input_hashes": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_SHA256, "contract_sha256": CONTRACT_SHA256},
        "pre_run_validation": PRE_RUN_VALIDATION,
        "setup_test_budget_seconds": setup_budget,
        "timing_seconds": {"setup_preflight": setup_seconds, "collection": collection_seconds,
            "finalization_before_artifact_writes": time.perf_counter() - finalization_start,
            "total_before_artifact_writes": time.perf_counter() - started,
            "measurement_scope": "pre-write values stop before results/report/manifest serialization; post-manifest wall cap guard is sampled after the final manifest and includes all artifact writes"},
        "stream_sha256": sha256_file(stack_path), "index_sha256": sha256_file(index_path),
        "trajectory_sha256": sha256_file(trajectory_path)}
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path = output_dir / "report.md"
    report_path.write_text(_report(result) + "\n")
    manifest = {name: sha256_file(output_dir / name) for name in
        ("observation_stacks.uint8", "observation_index.jsonl", "trajectory.jsonl", "results.json", "report.md")}
    manifest["manifest_sha256_self_excluded"] = "self hash intentionally omitted"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    final_elapsed = time.perf_counter() - started
    post_write_finalization = final_elapsed - setup_seconds - collection_seconds
    result["post_manifest_wall_cap_guard"] = {
        "sampled_after_final_manifest_write": True,
        "total_seconds": final_elapsed,
        "finalization_seconds_including_artifact_writes": post_write_finalization,
        "setup_plus_tests_cap_seconds": setup_budget["cap"],
        "collection_cap_seconds": 60.0,
        "finalization_cap_seconds": 10.0,
        "total_cap_seconds": 90.0,
        "passed": setup_budget["total_consumed"] <= setup_budget["cap"]
            and collection_seconds <= 60 and post_write_finalization <= 10 and final_elapsed <= 90,
    }
    if (final_elapsed > 90 or post_write_finalization > 10
            or setup_budget["total_consumed"] > setup_budget["cap"] or collection_seconds > 60):
        raise RuntimeError(f"total wall cap exceeded: {final_elapsed:.3f}s")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _report(result: dict[str, Any]) -> str:
    reach = result["reachability"]
    return ("# Issue 52: Contract v2 model-input grayscale range audit\n\n"
        f"Classification: **{result['classification']}**; round status: **{result['round_status']}**; HVC: **{result['hvc']}**. The diagnostic itself is **INCONCLUSIVE**.\n\n"
        f"Seed {result['seed']}: {reach['R']}/{reach['N']} decisions ({reach['fraction']}) had a channel-3 uint8 pixel >= {result['threshold']} in either frozen ROI. This is sample reachability only.\n\n"
        f"Captured {result['decisions']} pre-action stacks over {result['native_frames']} ALE-native frames; stop reason: `{result['stop_reason']}`.\n\n"
        f"Channel-3 summaries (NumPy linear quantiles): `{json.dumps(result['aggregate_channel_3_summaries'], sort_keys=True)}`\n\n"
        f"Pre-run validation: {result['pre_run_validation']['focused_test_count']} focused tests passed in {result['pre_run_validation']['focused_test_wall_seconds']:.3f}s; zero-frame preflight passed in {result['pre_run_validation']['zero_frame_preflight_wall_seconds']:.3f}s with {result['pre_run_validation']['zero_frame_preflight_native_frames']} ALE frames. Setup/test cap: {result['setup_test_budget_seconds']['cap']:.1f}s; recorded pre-run use {result['setup_test_budget_seconds']['pre_run_consumed']:.3f}s plus formal setup {result['setup_test_budget_seconds']['formal_setup_preflight']:.3f}s, leaving {result['setup_test_budget_seconds']['remaining']:.3f}s. Compile and diff checks passed.\n\n"
        f"Model and spec hashes were verified before collection. Runtime: Python {result['runtime']['python']}, ONNX Runtime {result['runtime']['onnxruntime']}, providers {result['runtime']['providers']}.\n\n"
        f"Source commit `{result['source']['commit']}`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.\n")


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    return 0
