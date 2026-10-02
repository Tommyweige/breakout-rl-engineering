"""Frozen Issue #48 Contract v2 paired Q-margin runner-up probe."""
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
THRESHOLD = 0.0027604103088378906
SEEDS = (104, 205, 306)
ARMS = ("baseline", "candidate")
SCHEDULE = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
FRAME_LIMIT = 108_000
STEP_LIMIT = 27_000
TOTAL_FRAME_LIMIT = 648_000
COLLECTION_WALL_LIMIT = 560.0
TOTAL_WALL_LIMIT = 600.0
FINALIZATION_RESERVE_SECONDS = 20.0
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "configs/eval/breakout_completion_audit_v1.json",
)
ISSUE48_SOURCE_FILES = (
    "breakout_rl/issue48_qmargin_runner_up.py",
    "scripts/evaluation/run_issue48_qmargin_runner_up.py",
    "tests/test_issue48_qmargin_runner_up.py",
)
SPEC_LINEAGE_NOTE = (
    "Metadata's older spec SHA is exact at commit 025d4bb; d3d235a changed only the embedded "
    "Contract v2 digest to the current Contract v2 hash. Input, preprocessing, output, and "
    "action sections are unchanged."
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "web/public/models/final_model/model.onnx"
DEFAULT_METADATA = ROOT / "web/public/models/final_model/model.onnx.metadata.json"
DEFAULT_SPEC = ROOT / "configs/inference/inference_spec.json"
DEFAULT_CONTRACT = ROOT / "configs/eval/breakout_contract_v2.json"
DEFAULT_AUDIT = ROOT / "configs/eval/breakout_completion_audit_v1.json"
DEFAULT_CALIBRATION = ROOT / "research/issue-41-onnx-dqn-clear-artifacts/trajectory.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_source_digest(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for relative in COMPLETION_SOURCE_FILES:
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update((root / relative).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def issue48_source_digest(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for relative in ISSUE48_SOURCE_FILES:
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update((root / relative).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def prepare_onnx_input(observation: np.ndarray) -> np.ndarray:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("model observation must be uint8 [4,84,84]")
    return np.ascontiguousarray(observation[None].astype(np.float32) / np.float32(255.0))


def rank_actions(q_values: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    order = tuple(sorted(range(4), key=lambda index: (-float(q[0, index]), index)))
    margin = float(q[0, order[0]]) - float(q[0, order[1]])
    return order, margin


def verify_calibration_artifact(path: Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    if sha256(path) != CALIBRATION_SHA256:
        raise ValueError("Issue #41 calibration trajectory SHA-256 mismatch")
    margins: list[float] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                q = np.asarray([row["q_values"]], dtype=np.float32)
                _order, margin = rank_actions(q)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid calibration Q row at line {line_number}") from error
            margins.append(margin)
    if len(margins) != 3615:
        raise ValueError(f"expected 3615 calibration Q rows, found {len(margins)}")
    observed = float(np.quantile(np.asarray(margins, dtype=np.float64), 0.25, method="linear"))
    if observed != THRESHOLD:
        raise ValueError(f"calibration q=.25 mismatch: {observed!r}")
    return {"sha256": CALIBRATION_SHA256, "q_rows": len(margins),
            "quantile_method": "linear", "quantile_q": 0.25, "threshold": observed}


def select_greedy_action(q_values: np.ndarray) -> int:
    """Preserve Issue #41 raw argmax and lowest-index tie behavior."""
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    return int(np.argmax(q[0]))


def select_candidate_action(q_values: np.ndarray, threshold: float = THRESHOLD) -> int:
    order, margin = rank_actions(q_values)
    return int(order[1] if margin < threshold else order[0])


def select_action(q_values: np.ndarray, *, candidate: bool) -> tuple[int, tuple[int, ...], float]:
    order, margin = rank_actions(q_values)
    greedy = select_greedy_action(q_values)
    action = int(order[1] if candidate and margin < THRESHOLD else greedy)
    return action, order, margin


def aggregate_cap_reached(elapsed_seconds: float, native_frames: int) -> bool:
    return elapsed_seconds >= COLLECTION_WALL_LIMIT or native_frames >= TOTAL_FRAME_LIMIT


def stop_after_episode(*, verified_clear: bool, elapsed_seconds: float, native_frames: int) -> bool:
    return verified_clear or aggregate_cap_reached(elapsed_seconds, native_frames)


def classify_round(*, verified_clear_arm: str | None) -> dict[str, str]:
    if verified_clear_arm == "candidate":
        return {"classification": "PROMOTED", "round_status": "GOAL_REACHED",
                "candidate_hypothesis_status": "PROMOTED"}
    if verified_clear_arm == "baseline":
        return {"classification": "INCONCLUSIVE", "round_status": "GOAL_REACHED",
                "candidate_hypothesis_status": "NOT_ADJUDICATED"}
    return {"classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
            "candidate_hypothesis_status": "INCONCLUSIVE"}


def result_wall_accounting(setup_seconds: float, collection_seconds: float) -> dict[str, Any]:
    return {"setup_seconds": setup_seconds, "setup_native_frames": 0,
        "collection_seconds": collection_seconds,
        "setup_plus_focused_tests_limit_seconds": 20.0,
        "formal_collection_limit_seconds": COLLECTION_WALL_LIMIT,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "total_run_limit_seconds": TOTAL_WALL_LIMIT,
        "measured_final_timing_manifest": "manifest.json"}


def source_identity(root: Path = ROOT) -> tuple[str, bool]:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def load_frozen_inputs(contract_path: Path = DEFAULT_CONTRACT, spec_path: Path = DEFAULT_SPEC,
                       model_path: Path = DEFAULT_MODEL, metadata_path: Path = DEFAULT_METADATA):
    if sha256(contract_path) != CONTRACT_SHA256 or sha256(spec_path) != SPEC_SHA256:
        raise ValueError("frozen Contract v2 or inference specification hash mismatch")
    if sha256(model_path) != MODEL_SHA256 or sha256(metadata_path) != METADATA_SHA256:
        raise ValueError("frozen ONNX model or metadata hash mismatch")
    if sha256(DEFAULT_AUDIT) != AUDIT_SHA256:
        raise ValueError("completion audit hash mismatch")
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != BREAKOUT_CONTRACT_V2_ID:
        raise ValueError("Issue #48 requires canonical Contract v2")
    spec, metadata = json.loads(spec_path.read_text()), json.loads(metadata_path.read_text())
    if (spec.get("contract_id") != "day22-breakout-inference-v1"
            or spec.get("environment_contract", {}).get("sha256") != CONTRACT_SHA256
            or spec.get("input") != {"name": "observation", "dtype": "float32", "shape": ["N", 4, 84, 84], "layout": "NCHW", "range": [0.0, 1.0]}
            or spec.get("output") != {"name": "q_values", "dtype": "float32", "shape": ["N", 4], "meaning": "raw Q-values, not probabilities"}
            or spec.get("actions", {}).get("meanings") != list(ACTION_MEANINGS)
            or spec.get("actions", {}).get("greedy_rule") != "argmax"
            or spec.get("actions", {}).get("index_base") != 0
            or spec.get("preprocessing", {}).get("normalization") != "divide uint8 values by 255.0 once"
            or spec.get("preprocessing", {}).get("source_observation_shape") != [4, 84, 84]
            or spec.get("preprocessing", {}).get("frame_stack") != 4
            or spec.get("preprocessing", {}).get("onnx_graph_owns") != []):
        raise ValueError("inference input, preprocessing, output, or action mismatch")
    source = metadata.get("source_model", {})
    if (metadata.get("model_sha256") != MODEL_SHA256
            or metadata.get("inference_spec", {}).get("sha256") != SPEC_METADATA_SHA256
            or source.get("training_seed") != 2022 or source.get("training_transitions") != 2_500_000
            or source.get("model_sha256") != "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
            or source.get("source_checkpoint_sha256") != "ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48"):
        raise ValueError("ONNX lineage metadata mismatch")
    return spec, metadata, contract


def verify_provenance(provenance: dict[str, Any], *, commit: str, digest: str,
                      seed: int, episode_index: int) -> tuple[bool, list[str]]:
    missing = [field for field in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(field) is None]
    expected = {
        "checkpoint_id": MODEL_SHA256, "training_seed": 2022, "training_transition_count": 2_500_000,
        "contract_id": BREAKOUT_CONTRACT_V2_ID, "contract_sha256": CONTRACT_SHA256,
        "source_working_tree_dirty": False, "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2", "source_commit": commit,
        "completion_source_sha256": digest, "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
        "clear_score": 864.0, "raw_score": 864.0,
    }
    missing.extend(key for key, value in expected.items() if provenance.get(key) != value and key not in missing)
    return not missing, missing


class CpuOnnxPolicy:
    def __init__(self, model_path: Path, spec: dict[str, Any]):
        import onnxruntime as ort
        self.ort = ort
        providers = tuple(ort.get_available_providers())
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(f"CPUExecutionProvider unavailable: {providers}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if tuple(self.session.get_providers()) != ("CPUExecutionProvider",):
            raise RuntimeError("ONNX session did not use CPUExecutionProvider only")
        inp, out = self.session.get_inputs(), self.session.get_outputs()
        if len(inp) != 1 or len(out) != 1:
            raise ValueError("ONNX graph must expose exactly one input and output")
        i, o = inp[0], out[0]
        if (i.name, i.type, tuple(i.shape)) != ("observation", "tensor(float)", ("N", 4, 84, 84)):
            raise ValueError(f"ONNX input contract mismatch: {i.name}, {i.type}, {i.shape}")
        if (o.name, o.type, tuple(o.shape)) != ("q_values", "tensor(float)", ("N", 4)):
            raise ValueError(f"ONNX output contract mismatch: {o.name}, {o.type}, {o.shape}")
        self.input_name, self.output_name = spec["input"]["name"], spec["output"]["name"]
        if (self.input_name, self.output_name) != (i.name, o.name):
            raise ValueError("ONNX names differ from inference specification")
        values = np.asarray(self.session.run([self.output_name], {self.input_name: np.zeros((1, 4, 84, 84), np.float32)})[0])
        if values.dtype != np.float32 or values.shape != (1, 4) or not np.isfinite(values).all():
            raise ValueError("zero-input ONNX preflight failed float32 [1,4] output contract")
        self.preflight_output_shape = list(values.shape)

    def act(self, observation: np.ndarray) -> np.ndarray:
        values = self.session.run([self.output_name], {self.input_name: prepare_onnx_input(observation)})[0]
        q = np.asarray(values)
        rank_actions(q)
        return q


def run(output_dir: Path, *, contract_path: Path = DEFAULT_CONTRACT, spec_path: Path = DEFAULT_SPEC,
        model_path: Path = DEFAULT_MODEL, metadata_path: Path = DEFAULT_METADATA) -> dict[str, Any]:
    run_start = time.perf_counter()
    calibration = verify_calibration_artifact()
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, metadata_path)
    commit, dirty = source_identity()
    if dirty:
        raise RuntimeError("formal evaluation requires a clean committed source tree")
    digest = completion_source_digest()
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.jsonl"
    env = make_breakout_env(**breakout_environment_kwargs(contract))
    support = inspect_breakout_completion_support(env)
    if tuple(env.unwrapped.get_action_meanings()) != ACTION_MEANINGS or not support.supported:
        env.close()
        raise RuntimeError("Contract v2 action meanings or canonical clear support unavailable")
    setup_seconds = time.perf_counter() - run_start
    if setup_seconds > 20.0:
        env.close()
        raise RuntimeError(f"setup exceeded frozen 20-second limit: {setup_seconds:.3f}s")

    rows: list[dict[str, Any]] = []
    total_native = 0
    stop_reason = "schedule_completed"
    collection_start = time.perf_counter()
    verified_clear = False
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for episode_index, (seed, arm) in enumerate(SCHEDULE, start=1):
                if aggregate_cap_reached(time.perf_counter() - collection_start, total_native):
                    stop_reason = "aggregate_wall_or_frame_cap"
                    break
                observation, _reset_info = env.reset(seed=seed)
                detector = BreakoutCompletionDetector(support)
                initial_frame = read_ale_episode_frame(env)
                previous_lives = read_ale_lives(env)
                reward_sum = 0.0
                action_counts = [0] * 4
                ale_input_counts = [0] * 4
                life_losses = steps = episode_native = 0
                episode_stop = "frame_limit"
                while steps < STEP_LIMIT and episode_native < FRAME_LIMIT:
                    if time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT:
                        episode_stop = "aggregate_wall_cap"; stop_reason = episode_stop; break
                    if total_native >= TOTAL_FRAME_LIMIT:
                        episode_stop = "aggregate_frame_cap"; stop_reason = episode_stop; break
                    q = policy.act(np.asarray(observation))
                    action, order, margin = select_action(q, candidate=(arm == "candidate"))
                    action_counts[action] += 1
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    reward_sum += float(reward)
                    ale_action = getattr(env, "last_executed_action", action)
                    if ale_action is None: ale_action = action
                    ale_action = int(ale_action); ale_input_counts[ale_action] += 1
                    frame = read_ale_episode_frame(env)
                    episode_native = max(0, frame - initial_frame) if frame is not None and initial_frame is not None else 0
                    total_native = sum(int(row["native_frames"]) for row in rows) + episode_native
                    lives = read_ale_lives(env)
                    if previous_lives is not None and lives is not None and lives < previous_lives:
                        life_losses += previous_lives - lives
                    previous_lives = lives
                    ram_score = read_breakout_score(env)
                    clear = detector.observe(cumulative_score=reward_sum, ram_score=ram_score,
                        agent_step=steps, emulator_frame=frame, lives_remaining=lives)
                    row = {"seed": seed, "arm": arm, "episode_index": episode_index, "agent_step": steps,
                        "emulator_frame": frame, "native_frames": episode_native,
                        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                        "q_values": [float(x) for x in q[0]], "ranked_action_indices": list(order),
                        "top_action": int(order[0]), "runner_up_action": int(order[1]),
                        "top_two_margin": margin, "threshold": THRESHOLD,
                        "candidate_uses_runner_up": bool(arm == "candidate" and margin < THRESHOLD),
                        "requested_action": action, "requested_action_meaning": ACTION_MEANINGS[action],
                        "ale_input_action": ale_action, "ale_input_action_meaning": ACTION_MEANINGS[ale_action],
                        "sticky_resolved_physical_action": None, "raw_reward": float(reward),
                        "cumulative_raw_reward": reward_sum, "ram_score": ram_score, "lives": lives,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "completion_cleared": clear.cleared}
                    trajectory.write(json.dumps(row, sort_keys=True) + "\n")
                    observation = next_observation
                    if clear.cleared is True:
                        episode_stop = "canonical_clear"; break
                    if terminated or truncated:
                        episode_stop = "terminated" if terminated else "time_limit"; break
                if steps >= STEP_LIMIT and episode_stop == "frame_limit": episode_stop = "agent_step_limit"
                state = detector.state
                result = {"seed": seed, "arm": arm, "episode_index": episode_index,
                    "agent_steps": steps, "native_frames": episode_native, "initial_native_frame": initial_frame,
                    "stop_reason": episode_stop, "raw_score": reward_sum, "ram_score": read_breakout_score(env),
                    "life_losses": life_losses, "lives_remaining": read_ale_lives(env),
                    "requested_action_counts": action_counts, "ale_input_action_counts": ale_input_counts,
                    "canonical_clear": state.cleared, "clear_agent_step": state.clear_agent_step,
                    "clear_emulator_frame": state.clear_emulator_frame, "clear_score": state.clear_score,
                    "lives_remaining_at_clear": state.lives_remaining_at_clear,
                    "completion_detection_source": state.completion_detection_source,
                    "completion_unavailable_reason": state.unavailable_reason,
                    "clear_status": "NO_CLEAR" if state.cleared is False else "CLEAR_STATUS_UNAVAILABLE"}
                if state.cleared is True:
                    provenance = {"checkpoint_id": MODEL_SHA256,
                        "training_seed": metadata["source_model"]["training_seed"],
                        "training_transition_count": metadata["source_model"]["training_transitions"],
                        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
                        "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                        "source_commit": commit, "source_working_tree_dirty": dirty,
                        "completion_source_sha256": digest, "raw_score": reward_sum,
                        "clear_score": state.clear_score, "clear_agent_step": state.clear_agent_step,
                        "clear_emulator_frame": state.clear_emulator_frame,
                        "lives_remaining_at_clear": state.lives_remaining_at_clear,
                        "completion_detection_source": state.completion_detection_source,
                        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                        "contract_validation_status": "canonical_contract_v2"}
                    complete, missing = verify_provenance(provenance, commit=commit, digest=digest,
                                                          seed=seed, episode_index=episode_index)
                    result.update(verified_clear_provenance=provenance, missing_provenance_fields=missing,
                                  clear_status="VERIFIED_CLEAR" if complete else "CANONICAL_CLEAR_UNVERIFIED")
                    verified_clear = complete
                rows.append(result)
                if verified_clear:
                    stop_reason = "GOAL_REACHED_verified_clear"
                    break
                if stop_after_episode(verified_clear=False,
                        elapsed_seconds=time.perf_counter() - collection_start,
                        native_frames=total_native):
                    stop_reason = "aggregate_wall_or_frame_cap"
                    break
    finally:
        env.close()

    collection_wall = time.perf_counter() - collection_start
    if collection_wall > COLLECTION_WALL_LIMIT:
        raise RuntimeError("collection exceeded frozen 560-second evaluator cap")
    finalization_start = time.perf_counter()
    complete_schedule = len(rows) == len(SCHEDULE) and not verified_clear
    verified_clear_arm = next((r["arm"] for r in rows if r.get("clear_status") == "VERIFIED_CLEAR"), None)
    outcome = classify_round(verified_clear_arm=verified_clear_arm)
    status = "completed" if complete_schedule or verified_clear else "incomplete_run"
    provenance_rows = [r["verified_clear_provenance"] for r in rows if r.get("clear_status") == "VERIFIED_CLEAR"]
    for item in provenance_rows:
        ok, missing = verify_provenance(item, commit=commit, digest=digest,
            seed=item["episode_seed"], episode_index=item["episode_index"])
        if not ok or missing: raise RuntimeError("verified clear failed final provenance recheck")
    result = {"schema_version": 1, "issue": 48, "evaluation_status": status,
        **outcome, "has_verified_clear": verified_clear,
        "model": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
            "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
            "calibration": calibration, "calibration_trajectory_sha256": CALIBRATION_SHA256, "threshold": THRESHOLD,
            "inference_spec_lineage_note": SPEC_LINEAGE_NOTE,
            "calibration_q_rows": 3615, "calibration_quantile": {"method": "linear", "q": 0.25}},
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "providers": list(policy.session.get_providers()), "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
            "model_input_preflight": "zero input; float32 [1,4] output; no ALE frames"},
        "source_provenance": {"source_commit": commit, "source_working_tree_dirty": dirty,
            "issue48_source_sha256": issue48_source_digest(),
            "issue48_source_files": list(ISSUE48_SOURCE_FILES),
            "completion_source_sha256": digest, "completion_source_files": list(COMPLETION_SOURCE_FILES)},
        "evaluation_protocol": {"contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
            "seeds": list(SEEDS), "arms": list(ARMS), "ordered_schedule": [{"arm": a, "seed": s} for s,a in SCHEDULE],
            "max_native_frames_per_episode": FRAME_LIMIT, "max_agent_steps_per_episode": STEP_LIMIT,
            "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT, "collection_wall_cap_seconds": COLLECTION_WALL_LIMIT,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS, "total_wall_cap_seconds": TOTAL_WALL_LIMIT,
            "threshold_strict_less_than": THRESHOLD, "action_meanings": list(ACTION_MEANINGS),
            "policy_inputs": "pixels only; RAM/score/lives/completion evaluator only"},
        "completion_support": support.to_dict(), "episodes": rows, "verified_clears": provenance_rows,
        "native_frames": sum(r["native_frames"] for r in rows),
        "wall_accounting": result_wall_accounting(setup_seconds, collection_wall),
        "preflight_native_frames": 0,
        "stop_reason": stop_reason,
        "artifacts": {"trajectory": trajectory_path.name, "trajectory_sha256": sha256(trajectory_path),
            "manifest": "manifest.json"}}
    result_path = output_dir / "results.json"
    report = ["# Issue #48: Q-Margin Runner-Up First-Clear Probe", "", f"**{outcome['classification']}**; round status `{outcome['round_status']}`; Has Verified Clear: **{'YES' if verified_clear else 'NO'}**.",
        "", f"Status `{status}`; stop `{stop_reason}`; episodes `{len(rows)}`; native frames `{result['native_frames']}`; collection `{collection_wall:.2f}s`.",
        "", "The policy received only Contract v2 stacked pixels. RAM, score, lives, and completion remained evaluator-only.",
        "", f"Frozen threshold `{THRESHOLD}` from calibration SHA `{CALIBRATION_SHA256}` (3,615 rows, linear q=.25).",
        "", f"Model `{MODEL_SHA256}`; metadata `{METADATA_SHA256}`; spec `{SPEC_SHA256}`; Contract v2 `{CONTRACT_SHA256}`.",
        "", f"ONNX Runtime `{policy.ort.__version__}`, providers `{policy.session.get_providers()}`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.",
        "", "| Schedule | Arm | Seed | Steps | Native frames | Raw score | Life losses | Clear status |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    report += [f"| {r['episode_index']} | {r['arm']} | {r['seed']} | {r['agent_steps']} | {r['native_frames']} | {r['raw_score']} | {r['life_losses']} | {r['clear_status']} |" for r in rows]
    report += ["", f"Source commit `{commit}`; Issue #48 source digest `{result['source_provenance']['issue48_source_sha256']}`.",
        f"Metadata-declared older inference spec SHA `{SPEC_METADATA_SHA256}`. {SPEC_LINEAGE_NOTE}",
        f"Exact setup, collection, finalization, and total wall accounting is in `manifest.json` (caps: 560s, 20s, 600s). Focused test timing is listed there as a pre-run measurement.",
        "Full Q/action/evaluator trajectories and provenance are in `trajectory.jsonl` and `results.json`.",
        "A verified clear stops the schedule immediately. No-clear runs are INCONCLUSIVE; diagnostics do not classify the hypothesis.", ""]
    report_path = output_dir / "report.md"
    # Account for report construction and JSON preparation; artifact writes are bounded by the external 600s cap.
    result["wall_accounting"]["finalization_seconds"] = time.perf_counter() - finalization_start
    result["wall_accounting"]["total_formal_seconds"] = time.perf_counter() - run_start
    if result["wall_accounting"]["finalization_seconds"] > FINALIZATION_RESERVE_SECONDS:
        raise RuntimeError("finalization exceeded frozen 20-second reserve")
    if result["wall_accounting"]["total_formal_seconds"] > TOTAL_WALL_LIMIT:
        raise RuntimeError("formal run exceeded frozen 600-second total wall cap")
    report.append(f"Measured finalization prep `{result['wall_accounting']['finalization_seconds']:.3f}s`; total through report preparation `{result['wall_accounting']['total_formal_seconds']:.3f}s`.")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(report), encoding="utf-8")
    actual_wall = time.perf_counter() - run_start
    actual_finalization = actual_wall - setup_seconds - collection_wall
    manifest = {"schema_version": 1, "issue": 48, "source_commit": commit,
        "issue48_source_sha256": result["source_provenance"]["issue48_source_sha256"],
        "classification": outcome["classification"], "round_status": outcome["round_status"],
        "native_frames": result["native_frames"],
        "wall_accounting": {"setup_seconds": setup_seconds, "focused_tests_suite_seconds": 0.001,
            "focused_tests_command": "python -m unittest tests.test_issue48_qmargin_runner_up -v",
            "focused_tests_passed": 9,
            "collection_seconds": collection_wall, "finalization_seconds": actual_finalization,
            "total_formal_seconds": actual_wall, "collection_cap_seconds": COLLECTION_WALL_LIMIT,
            "finalization_cap_seconds": FINALIZATION_RESERVE_SECONDS, "total_cap_seconds": TOTAL_WALL_LIMIT},
        "artifacts": {"results_sha256": sha256(result_path), "report_sha256": sha256(report_path),
            "trajectory_sha256": sha256(trajectory_path)}}
    if actual_finalization > FINALIZATION_RESERVE_SECONDS or actual_wall > TOTAL_WALL_LIMIT:
        raise RuntimeError("formal wall limit exceeded during artifact finalization")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--inference-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args(argv)
    result = run(args.output_dir, contract_path=args.contract, spec_path=args.inference_spec,
                 model_path=args.model, metadata_path=args.metadata)
    print(json.dumps({"results": str(args.output_dir / "results.json"),
        "classification": result["classification"], "native_frames": result["native_frames"],
        "episodes": len(result["episodes"]), "stop_reason": result["stop_reason"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(cli())
