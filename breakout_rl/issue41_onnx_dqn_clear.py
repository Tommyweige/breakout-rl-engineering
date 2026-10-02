"""Frozen Issue #41 Contract v2 ONNX first-clear probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BreakoutCompletionDetector,
    BREAKOUT_COMPLETION_DETECTOR_ID,
    BREAKOUT_COMPLETION_SOURCE,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V2_ID,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)

MODEL_SHA256 = "cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12"
METADATA_SHA256 = "fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512"
SPEC_CURRENT_SHA256 = "b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507"
SPEC_METADATA_SHA256 = "68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec"
CONTRACT_SHA256 = "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a"
AUDIT_SHA256 = "43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b"
SEEDS = (101, 202, 303)
FRAME_LIMIT = 108_000
STEP_LIMIT = 27_000
TOTAL_FRAME_LIMIT = 324_000
TOTAL_WALL_LIMIT = 560.0
FINALIZATION_RESERVE_SECONDS = 30.0
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "configs/eval/breakout_completion_audit_v1.json",
)
DEFAULT_MODEL = Path("web/public/models/final_model/model.onnx")
DEFAULT_METADATA = Path("web/public/models/final_model/model.onnx.metadata.json")
DEFAULT_AUDIT = Path("configs/eval/breakout_completion_audit_v1.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in COMPLETION_SOURCE_FILES:
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update((root / relative).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def prepare_onnx_input(observation: np.ndarray) -> np.ndarray:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("model observation must be uint8 [4,84,84]")
    return np.ascontiguousarray(observation[None].astype(np.float32) / np.float32(255.0))


def select_greedy_action(q_values: np.ndarray) -> int:
    q = np.asarray(q_values)
    if q.dtype != np.float32 or q.shape != (1, 4) or not np.isfinite(q).all():
        raise ValueError("Q-values must be finite float32 [1,4]")
    return int(np.argmax(q[0]))


def complete_clear_provenance(provenance: dict[str, Any], *, source_commit: str,
                              completion_digest: str, episode_seed: int,
                              episode_index: int) -> tuple[bool, list[str]]:
    missing = [field for field in VERIFIED_CLEAR_PROVENANCE_FIELDS if provenance.get(field) is None]
    expected = {
        "checkpoint_id": MODEL_SHA256, "training_seed": 2022,
        "training_transition_count": 2_500_000, "contract_id": BREAKOUT_CONTRACT_V2_ID,
        "contract_sha256": CONTRACT_SHA256, "source_working_tree_dirty": False,
        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2",
        "source_commit": source_commit,
        "completion_source_sha256": completion_digest,
        "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
        "evaluation_seed": episode_seed,
        "episode_seed": episode_seed,
        "episode_index": episode_index,
        "clear_score": 864.0, "raw_score": 864.0,
    }
    invalid = [field for field, value in expected.items() if provenance.get(field) != value]
    expected_index = SEEDS.index(episode_seed) + 1 if episode_seed in SEEDS else None
    if episode_index != expected_index:
        invalid.append("episode_index")
    missing.extend(field for field in invalid if field not in missing)
    return not missing, missing


def source_identity(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def load_frozen_inputs(contract_path: Path, spec_path: Path, model_path: Path,
                       metadata_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    contract_path, spec_path = contract_path.resolve(), spec_path.resolve()
    model_path, metadata_path = model_path.resolve(), metadata_path.resolve()
    if sha256(contract_path) != CONTRACT_SHA256:
        raise ValueError("frozen Contract v2 hash mismatch")
    if sha256(spec_path) != SPEC_CURRENT_SHA256:
        raise ValueError("current inference specification hash mismatch")
    if sha256(model_path) != MODEL_SHA256:
        raise ValueError("frozen ONNX model hash mismatch")
    if sha256(metadata_path) != METADATA_SHA256:
        raise ValueError("frozen ONNX metadata hash mismatch")
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != BREAKOUT_CONTRACT_V2_ID:
        raise ValueError("Issue #41 requires canonical Contract v2")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (spec.get("contract_id") != "day22-breakout-inference-v1"
            or spec.get("environment_contract", {}).get("sha256") != CONTRACT_SHA256
            or spec.get("input") != {"name": "observation", "dtype": "float32",
                "shape": ["N", 4, 84, 84], "layout": "NCHW", "range": [0.0, 1.0]}
            or spec.get("output") != {"name": "q_values", "dtype": "float32",
                "shape": ["N", 4], "meaning": "raw Q-values, not probabilities"}
            or spec.get("actions", {}).get("meanings") != list(ACTION_MEANINGS)
            or spec.get("actions", {}).get("greedy_rule") != "argmax"
            or spec.get("actions", {}).get("index_base") != 0
            or spec.get("preprocessing", {}).get("normalization") != "divide uint8 values by 255.0 once"
            or spec.get("preprocessing", {}).get("source_observation_shape") != [4, 84, 84]
            or spec.get("preprocessing", {}).get("frame_stack") != 4
            or spec.get("preprocessing", {}).get("onnx_graph_owns") != []):
        raise ValueError("inference input, preprocessing, output, or actions mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("model_sha256") != MODEL_SHA256
            or metadata.get("inference_spec", {}).get("sha256") != SPEC_METADATA_SHA256
            or metadata.get("source_model", {}).get("training_seed") != 2022
            or metadata.get("source_model", {}).get("training_transitions") != 2_500_000
            or metadata.get("source_model", {}).get("model_sha256") != "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
            or metadata.get("source_model", {}).get("source_checkpoint_sha256") != "ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48"):
        raise ValueError("ONNX lineage metadata mismatch")
    return spec, metadata, contract


class CpuOnnxPolicy:
    """Minimal Runtime adapter; deliberately avoids the torch-importing inference module."""

    def __init__(self, model_path: Path, spec: dict[str, Any]):
        import onnxruntime as ort
        self.ort = ort
        providers = tuple(ort.get_available_providers())
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(f"CPUExecutionProvider unavailable: {providers}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if tuple(self.session.get_providers()) != ("CPUExecutionProvider",):
            raise RuntimeError("ONNX session did not use CPUExecutionProvider only")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX graph must expose exactly one input and output")
        inp, out = inputs[0], outputs[0]
        if (inp.name, inp.type, tuple(inp.shape)) != ("observation", "tensor(float)", ("N", 4, 84, 84)):
            raise ValueError(f"ONNX input contract mismatch: {inp.name}, {inp.type}, {inp.shape}")
        if (out.name, out.type, tuple(out.shape)) != ("q_values", "tensor(float)", ("N", 4)):
            raise ValueError(f"ONNX output contract mismatch: {out.name}, {out.type}, {out.shape}")
        self.input_name, self.output_name = spec["input"]["name"], spec["output"]["name"]
        if (self.input_name, self.output_name) != (inp.name, out.name):
            raise ValueError("ONNX names differ from the inference specification")
        probe = np.zeros((1, 4, 84, 84), dtype=np.float32)
        values = np.asarray(self.session.run([self.output_name], {self.input_name: probe})[0])
        if values.dtype != np.float32 or values.shape != (1, 4) or not np.isfinite(values).all():
            raise ValueError("ONNX zero-input preflight failed float32 [1,4] output contract")
        self.preflight_output_shape = list(values.shape)

    def act(self, observation: np.ndarray) -> tuple[int, list[float]]:
        model_input = prepare_onnx_input(observation)
        q = np.asarray(self.session.run([self.output_name], {self.input_name: model_input})[0])
        return select_greedy_action(q), [float(value) for value in q[0]]


def run(contract_path: Path, spec_path: Path, model_path: Path, episode_seeds: tuple[int, ...],
        max_frames_per_episode: int, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, DEFAULT_METADATA)
    if episode_seeds != SEEDS or max_frames_per_episode != FRAME_LIMIT:
        raise ValueError("frozen Issue #41 seed order or frame cap changed")
    audit_path = root / DEFAULT_AUDIT
    if sha256(audit_path) != AUDIT_SHA256:
        raise ValueError("completion audit hash mismatch")
    commit, dirty = source_identity(root)
    if dirty:
        raise RuntimeError("formal evaluation requires a clean committed source tree")
    completion_digest = completion_source_digest(root)
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.jsonl"
    env = make_breakout_env(**__import__("breakout_rl.evaluation_contract", fromlist=["breakout_environment_kwargs"]).breakout_environment_kwargs(contract))
    support = inspect_breakout_completion_support(env)
    meanings = tuple(env.unwrapped.get_action_meanings())
    if meanings != ACTION_MEANINGS:
        env.close()
        raise RuntimeError(f"ALE action meanings mismatch: {meanings}")
    if not support.supported:
        env.close()
        raise RuntimeError(f"canonical clear evaluator unavailable: {support.reason}")
    episodes: list[dict[str, Any]] = []
    total_native = 0
    verified = False
    status = "completed"
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for episode_index, seed in enumerate(episode_seeds, start=1):
                if time.perf_counter() - start >= TOTAL_WALL_LIMIT - FINALIZATION_RESERVE_SECONDS or total_native >= TOTAL_FRAME_LIMIT:
                    status = "incomplete_run"
                    break
                observation, reset_info = env.reset(seed=seed)
                detector = BreakoutCompletionDetector(support)
                reward_sum = 0.0
                requested_counts = [0] * 4
                ale_input_counts = [0] * 4
                life_losses = 0
                previous_lives = read_ale_lives(env)
                stop_reason = "frame_limit"
                steps = 0
                initial_frame = read_ale_episode_frame(env)
                episode_native = 0
                while steps < STEP_LIMIT and episode_native < max_frames_per_episode:
                    if time.perf_counter() - start >= TOTAL_WALL_LIMIT - FINALIZATION_RESERVE_SECONDS or total_native >= TOTAL_FRAME_LIMIT:
                        stop_reason, status = "wall_or_aggregate_cap", "incomplete_run"
                        break
                    action, q_values = policy.act(np.asarray(observation))
                    requested_counts[action] += 1
                    next_observation, reward, terminated, truncated, info = env.step(action)
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
                    action_row = {"seed": seed, "episode_index": episode_index, "agent_step": steps,
                        "emulator_frame": frame, "native_frames": native, "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                        "requested_action": action, "requested_action_meaning": ACTION_MEANINGS[action],
                        "ale_input_action": ale_input_action, "ale_input_action_meaning": ACTION_MEANINGS[ale_input_action],
                        "sticky_resolved_physical_action": None, "q_values": q_values, "raw_reward": float(reward),
                        "cumulative_raw_reward": reward_sum, "ram_score": ram_score, "lives": lives,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "completion_cleared": clear.cleared}
                    trajectory.write(json.dumps(action_row, sort_keys=True) + "\n")
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
                episode = {"seed": seed, "evaluation_seed": seed, "episode_seed": seed,
                    "episode_index": episode_index, "agent_steps": steps, "native_frames": episode_native,
                    "initial_native_frame": initial_frame, "stop_reason": stop_reason,
                    "raw_score": reward_sum, "ram_score": read_breakout_score(env), "life_losses": life_losses,
                    "lives_remaining": read_ale_lives(env), "requested_action_counts": requested_counts,
                    "ale_input_action_counts": ale_input_counts, "canonical_clear": clear_state.cleared,
                    "clear_agent_step": clear_state.clear_agent_step,
                    "clear_emulator_frame": clear_state.clear_emulator_frame,
                    "clear_score": clear_state.clear_score,
                    "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                    "completion_detection_source": clear_state.completion_detection_source,
                    "completion_unavailable_reason": clear_state.unavailable_reason}
                if clear_state.cleared is True:
                    provenance = {"checkpoint_id": MODEL_SHA256,
                        "training_seed": metadata["source_model"]["training_seed"],
                        "training_transition_count": metadata["source_model"]["training_transitions"],
                        "evaluation_seed": seed, "episode_seed": seed, "episode_index": episode_index,
                        "contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
                        "source_commit": commit, "source_working_tree_dirty": dirty,
                        "completion_source_sha256": completion_digest, "raw_score": reward_sum,
                        "clear_score": clear_state.clear_score, "clear_agent_step": clear_state.clear_agent_step,
                        "clear_emulator_frame": clear_state.clear_emulator_frame,
                        "lives_remaining_at_clear": clear_state.lives_remaining_at_clear,
                        "completion_detection_source": clear_state.completion_detection_source,
                        "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
                        "contract_validation_status": "canonical_contract_v2"}
                    complete, missing_fields = complete_clear_provenance(
                        provenance, source_commit=commit, completion_digest=completion_digest,
                        episode_seed=seed, episode_index=episode_index)
                    episode["verified_clear_provenance"] = provenance
                    episode["missing_provenance_fields"] = missing_fields
                    episode["clear_status"] = "VERIFIED_CLEAR" if complete else "CANONICAL_CLEAR_UNVERIFIED"
                    verified = episode["clear_status"] == "VERIFIED_CLEAR"
                else:
                    episode["clear_status"] = "NO_CLEAR" if clear_state.cleared is False else "CLEAR_STATUS_UNAVAILABLE"
                episodes.append(episode)
                if verified:
                    break
                if stop_reason == "wall_or_aggregate_cap":
                    break
    finally:
        env.close()
    if verified:
        classification = "GOAL_REACHED"
    elif status == "completed" and len(episodes) == len(SEEDS):
        classification = "INCONCLUSIVE"
    else:
        classification, status = "INCONCLUSIVE", "incomplete_run"
    model_meta = {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
        "inference_spec_sha256": SPEC_CURRENT_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
        "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
        "metadata_lineage": {"training_seed": metadata["source_model"]["training_seed"],
            "training_transitions": metadata["source_model"]["training_transitions"],
            "source_model_sha256": metadata["source_model"]["model_sha256"],
            "source_checkpoint_sha256": metadata["source_model"]["source_checkpoint_sha256"],
            "original_pt_independently_rehashed": False},
        "inference_spec_lineage_limitation": "Metadata's older spec SHA is exact at commit 025d4bb; d3d235a changed only embedded Contract v2 digest to current Contract v2 hash. Input/preprocessing/output/action sections are unchanged.",
        "contract_id": contract.contract_id, "detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
        "contract_validation_status": "canonical_contract_v2"}
    verified_clears = [row["verified_clear_provenance"] for row in episodes
                       if row["canonical_clear"] is True
                       and row["clear_status"] == "VERIFIED_CLEAR"]
    for provenance in verified_clears:
        complete, missing_fields = complete_clear_provenance(
            provenance, source_commit=commit, completion_digest=completion_digest,
            episode_seed=provenance["episode_seed"], episode_index=provenance["episode_index"])
        if not complete or missing_fields:
            raise RuntimeError("verified_clears contains incomplete provenance")
    result = {"schema_version": 1, "issue": 41, "evaluation_status": status,
        "classification": classification, "has_verified_clear": verified,
        "model": model_meta, "runtime": {"python": platform.python_version(),
            "onnxruntime": policy.ort.__version__, "providers": list(policy.session.get_providers()),
            "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1, 4, 84, 84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "model_input_preflight": "zero uint8 [4,84,84], cast float32 and divide by 255 once; no ALE frames"},
        "source_provenance": {"source_commit": commit, "source_working_tree_dirty": dirty,
            "completion_source_sha256": completion_digest,
            "completion_source_files": list(COMPLETION_SOURCE_FILES),
            "completion_audit_sha256": AUDIT_SHA256},
        "evaluation_protocol": {"contract_id": contract.contract_id, "contract_sha256": CONTRACT_SHA256,
            "episode_seeds": list(SEEDS), "episodes_per_seed": 1, "max_native_frames_per_episode": FRAME_LIMIT,
            "max_agent_steps_per_episode": STEP_LIMIT, "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT,
            "wall_clock_cap_seconds": TOTAL_WALL_LIMIT, "collection_wall_cap_seconds": TOTAL_WALL_LIMIT - FINALIZATION_RESERVE_SECONDS,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "environment_action_meanings": list(ACTION_MEANINGS),
            "observation_contract": "AtariPreprocessing grayscale uint8 84x84; frame_skip=4; stack uint8 [4,84,84]; adapter /255 once",
            "policy_inputs": "pixels only; RAM/score/lives/completion evaluator only"},
        "completion_support": support.to_dict(), "episodes": episodes,
        "verified_clears": verified_clears,
        "canonical_clear_unverified": [row for row in episodes if row["clear_status"] == "CANONICAL_CLEAR_UNVERIFIED"],
        "native_frames": total_native, "wall_seconds": time.perf_counter() - start,
        "artifacts": {"trajectory": trajectory_path.name, "trajectory_sha256": sha256(trajectory_path)}}
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Issue #41: Existing ONNX DQN First-Clear Probe", "", f"**{classification}** — Has Verified Clear: **{'YES' if verified else 'NO'}**.",
        "", f"Status: `{status}`; seeds run: `{[row['seed'] for row in episodes]}`; native frames: `{total_native}`; wall seconds: `{result['wall_seconds']:.2f}`.",
        "", "The ONNX policy received only uint8 stacked pixels converted once to float32 by `/255`. RAM, score, lives and completion state were evaluator-only. Contract v2 was used unchanged; this result is not compared with Contract v3.",
        "", f"Model SHA-256: `{MODEL_SHA256}`; metadata SHA-256: `{METADATA_SHA256}`; inference spec current SHA-256: `{SPEC_CURRENT_SHA256}` (metadata-declared older SHA-256: `{SPEC_METADATA_SHA256}`); Contract v2 SHA-256: `{CONTRACT_SHA256}`; completion audit SHA-256: `{AUDIT_SHA256}`.",
        "", f"ONNX Runtime `{policy.ort.__version__}`, providers `{policy.session.get_providers()}`. Source commit `{commit}`, dirty `{dirty}`. Original PyTorch `.pt` bytes were absent and could not be independently rehashed; lineage is metadata-declared.",
        "", "| Seed | Stop reason | Agent steps | ALE-native frames | Raw score | Life losses | Clear status |", "|---:|---|---:|---:|---:|---:|---|"]
    report += [f"| {row['seed']} | {row['stop_reason']} | {row['agent_steps']} | {row['native_frames']} | {row['raw_score']} | {row['life_losses']} | {row['clear_status']} |" for row in episodes]
    report += ["", "## Provenance", "", "Full per-step action/evaluator rows and the complete result payload are in `trajectory.jsonl` and `results.json`. Requested and wrapper ALE-input actions are distinct; sticky-resolved physical actions are not observable.",
        "", "## Decision", "", "A verified clear is reported as GOAL_REACHED / PROMOTED after independent Planner validation. If all three seeds finish without one, the frozen decision is INCONCLUSIVE; there is no rejection threshold.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inference-spec", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--max-native-frames-per-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.contract, args.inference_spec, args.model,
        tuple(int(item) for item in args.episode_seeds.split(",")),
        args.max_native_frames_per_episode, args.output_dir)
    print(json.dumps({"results": str(args.output_dir / "results.json"),
        "classification": result["classification"], "native_frames": result["native_frames"],
        "seeds": [episode["seed"] for episode in result["episodes"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
