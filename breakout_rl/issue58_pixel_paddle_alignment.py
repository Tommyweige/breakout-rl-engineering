"""Frozen Issue #58 pixel-only lower-playfield paddle alignment probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
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
CALIBRATION_INDEX_SHA256 = "757eab8a54327e0663e2d83e5156958b776c1c5e7a09521a3355a9af43746978"
CALIBRATION_RESULTS_SHA256 = "6954370d9270b154de55c38c26d107b25347a82d3ecf86bc723cdf03b68cf582"
CALIBRATION_TRAJECTORY_SHA256 = "48c3ad056962862ba8a80219909caa6f4349a704642cef65b3210f62b47a2398"
CALIBRATION_DETECTOR_SOURCE_SHA256 = "9c58280398839cde85f554a79193e28e562c4c72a5101ff2be9def731b000c04"
CALIBRATION_STACK_SHA256S = {
    "seed105_baseline_ep1.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa",
    "seed105_candidate_ep2.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa",
    "seed206_baseline_ep3.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55",
    "seed206_candidate_ep4.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55",
    "seed307_baseline_ep5.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95",
    "seed307_candidate_ep6.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95",
}
CALIBRATION_THRESHOLDS = (80, 90, 95, 100, 105, 110, 115, 120, 130, 148, 200)
EXPECTED_ACTIVATIONS = {80: 64, 90: 52, 95: 50, 100: 42, 105: 12, 110: 2, 115: 0, 120: 0, 130: 0, 148: 0, 200: 0}
CALIBRATION_SHA256 = CALIBRATION_INDEX_SHA256
SEEDS = (510, 511, 512)
ARMS = ("baseline", "candidate")
SCHEDULE = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
FRAME_LIMIT, STEP_LIMIT, TOTAL_FRAME_LIMIT = 108_000, 27_000, 648_000
TOTAL_STEP_LIMIT = 162_000
COLLECTION_WALL_LIMIT, FINALIZATION_RESERVE_SECONDS, TOTAL_WALL_LIMIT = 560.0, 20.0, 600.0
# Covers process launch and Python startup before the executable module can timestamp itself.
PROCESS_LAUNCH_RESERVE_SECONDS = 5.0
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
BALL_ROI = (54, 73, 6, 78)
PADDLE_ROI = (73, 83, 6, 78)
BRIGHT_THRESHOLD = 80
COMPLETION_SOURCE_FILES = (
    "breakout_env.py", "breakout_rl/completion.py", "breakout_rl/evaluation.py",
    "breakout_rl/evaluation_artifacts.py", "breakout_rl/evaluation_contract.py",
    "configs/eval/breakout_completion_audit_v1.json",
)
ISSUE58_SOURCE_FILES = (
    "breakout_rl/issue58_pixel_paddle_alignment.py",
    "scripts/evaluation/run_issue58_pixel_paddle_alignment.py",
    "tests/test_issue58_pixel_paddle_alignment.py",
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
DEFAULT_CALIBRATION = ROOT / "research/issue-50-pixel-paddle-alignment-artifacts"
DEFAULT_PRE_RUN_VALIDATION = Path("/tmp/issue58-pre-run-validation.json")


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
                  height: tuple[int, int], threshold: int = BRIGHT_THRESHOLD) -> dict[str, Any]:
    y0, y1, x0, x1 = roi
    components = _components(plane[y0:y1, x0:x1] >= threshold)
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


def detect_ball_and_paddle(observation: np.ndarray, *, threshold: int = BRIGHT_THRESHOLD) -> dict[str, Any]:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (4, 84, 84):
        raise ValueError("observation must be uint8 [4,84,84]")
    plane = observation[3]
    return {
        "ball": detect_object(plane, roi=BALL_ROI, area=(1, 8), width=(1, 3), height=(1, 3), threshold=threshold),
        "paddle": detect_object(plane, roi=PADDLE_ROI, area=(5, 64), width=(5, 16), height=(1, 4), threshold=threshold),
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
        return {"classification": "GOAL_REACHED", "round_status": "GOAL_REACHED", "candidate_hypothesis_status": "GOAL_REACHED", "hvc_status": "YES"}
    if verified_clear_arm == "baseline":
        return {"classification": "GOAL_REACHED", "round_status": "GOAL_REACHED", "candidate_hypothesis_status": "NOT_ADJUDICATED", "hvc_status": "YES"}
    return {"classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH", "candidate_hypothesis_status": "INCONCLUSIVE", "hvc_status": "NO"}


def evaluation_status(episode_count: int, verified_clear: bool) -> str:
    return "completed" if episode_count == len(SCHEDULE) or verified_clear else "incomplete_run"


def clear_stop_outcome(canonical_clear_detected: bool, provenance_complete: bool,
                       clear_arm: str | None) -> dict[str, Any]:
    """Resolve the first-clear gate; every canonical clear signal stops collection."""
    if not canonical_clear_detected:
        return {"stop": False, "classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
                "hvc_status": "NO", "candidate_hypothesis_status": "INCONCLUSIVE"}
    if not provenance_complete:
        return {"stop": True, "classification": "HUMAN_REVIEW_REQUIRED",
                "round_status": "HUMAN_REVIEW_REQUIRED", "hvc_status": "UNVERIFIED_CLEAR_SIGNAL",
                "candidate_hypothesis_status": "NOT_ADJUDICATED"}
    outcome = classify_round(clear_arm)
    return {"stop": True, **outcome, "hvc_status": "YES"}


def continue_after_episode(canonical_clear_detected: bool) -> bool:
    """Return false on every canonical clear signal, verified or not."""
    return not canonical_clear_detected


def stack_capture_reasons(*, override: bool, life_loss: bool,
                          canonical_clear: bool, life_window_active: bool) -> list[str]:
    reasons = []
    if override:
        reasons.append("detector_override")
    if life_window_active:
        reasons.append("life_loss_post")
    if life_loss:
        reasons.append("life_loss_event")
    if canonical_clear:
        reasons.append("canonical_clear")
    return reasons


def verify_calibration_assets(calibration_dir: Path = DEFAULT_CALIBRATION) -> None:
    index = calibration_dir / "decision_stack_index.jsonl"
    results = calibration_dir / "results.json"
    trajectory = calibration_dir / "trajectory.jsonl"
    for path, expected in ((index, CALIBRATION_INDEX_SHA256),
                           (results, CALIBRATION_RESULTS_SHA256),
                           (trajectory, CALIBRATION_TRAJECTORY_SHA256),
                           (ROOT / "breakout_rl/issue50_pixel_paddle_alignment.py",
                            CALIBRATION_DETECTOR_SOURCE_SHA256)):
        if sha256(path) != expected:
            raise ValueError(f"pinned Issue #50 calibration hash mismatch: {path}")
    for name, expected in CALIBRATION_STACK_SHA256S.items():
        path = calibration_dir / "decision_stacks" / name
        if sha256(path) != expected:
            raise ValueError(f"pinned Issue #50 calibration stack hash mismatch: {name}")


def scan_calibration(calibration_dir: Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    """Reproduce the frozen scan from Issue #50's 300 indexed archived stacks."""
    verify_calibration_assets(calibration_dir)
    index = calibration_dir / "decision_stack_index.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 300:
        raise ValueError(f"expected 300 indexed Issue #50 calibration stacks, found {len(rows)}")
    observations: list[np.ndarray] = []
    unique_hashes: set[str] = set()
    for row in rows:
        if row.get("shape") != [4, 84, 84] or row.get("dtype") != "uint8":
            raise ValueError("unexpected Issue #50 calibration stack shape or dtype")
        path = calibration_dir / row["file"]
        raw = path.read_bytes()[int(row["byte_offset"]):int(row["byte_offset"]) + int(row["bytes"])]
        if len(raw) != 4 * 84 * 84 or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Issue #50 indexed stack bytes do not match their pinned observation hash")
        observations.append(np.frombuffer(raw, dtype=np.uint8).reshape(4, 84, 84))
        unique_hashes.add(row["observation_sha256"])
    if len(unique_hashes) != 150:
        raise ValueError(f"expected 150 unique paired observations, found {len(unique_hashes)}")
    activation_counts = {}
    for threshold in CALIBRATION_THRESHOLDS:
        count = 0
        for observation in observations:
            detections = detect_ball_and_paddle(observation, threshold=threshold)
            count += int(detections["ball"]["status"] == "unique"
                         and detections["paddle"]["status"] == "unique")
        activation_counts[threshold] = count
    selected = max(CALIBRATION_THRESHOLDS,
                   key=lambda threshold: (activation_counts[threshold], threshold))
    if activation_counts != EXPECTED_ACTIVATIONS or selected != 80:
        raise ValueError(f"pinned calibration selection mismatch: {activation_counts}, selected {selected}")
    return {"source_issue": 50, "index_sha256": CALIBRATION_INDEX_SHA256,
            "results_sha256": CALIBRATION_RESULTS_SHA256,
            "trajectory_sha256": CALIBRATION_TRAJECTORY_SHA256,
            "detector_source_sha256": CALIBRATION_DETECTOR_SOURCE_SHA256,
            "stack_sha256s": dict(CALIBRATION_STACK_SHA256S),
            "indexed_rows": len(rows), "unique_pixel_observations": len(unique_hashes),
            "activation_counts_by_threshold": activation_counts, "selection_rule":
            "maximize indexed stacks with exactly one valid ball and one valid paddle; ties choose higher threshold",
            "selected_threshold": selected, "candidate_activation_at_selected_threshold": "64/300 indexed rows = 32/150 unique observations",
            "activation_is_object_truth": False}


def frozen_calibration_record() -> dict[str, Any]:
    return {"source_issue": 50, "index_sha256": CALIBRATION_INDEX_SHA256,
            "results_sha256": CALIBRATION_RESULTS_SHA256,
            "trajectory_sha256": CALIBRATION_TRAJECTORY_SHA256,
            "detector_source_sha256": CALIBRATION_DETECTOR_SOURCE_SHA256,
            "stack_sha256s": dict(CALIBRATION_STACK_SHA256S),
            "indexed_rows": 300, "unique_pixel_observations": 150,
            "activation_counts_by_threshold": dict(EXPECTED_ACTIVATIONS),
            "selection_rule": "maximize indexed stacks with exactly one valid ball and one valid paddle; ties choose higher threshold",
            "selected_threshold": 80,
            "candidate_activation_at_selected_threshold": "64/300 indexed rows = 32/150 unique observations",
            "activation_is_object_truth": False}


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
                (DEFAULT_AUDIT, AUDIT_SHA256))
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
    verify_calibration_assets()
    calibration = frozen_calibration_record()
    policy = CpuOnnxPolicy(model_path.resolve(), spec)
    audit = json.loads(DEFAULT_AUDIT.read_text(encoding="utf-8"))
    if contract.contract_id != BREAKOUT_CONTRACT_V2_ID or audit.get("detector_id") != BREAKOUT_COMPLETION_DETECTOR_ID:
        raise RuntimeError("pinned Contract v2 or canonical completion audit mismatch")
    return {"status": "preflight_passed", "native_frames": 0, "ale_environment_created": False,
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"), "ale_py": importlib.metadata.version("ale-py"),
            "providers": list(policy.session.get_providers()), "onnx_output_shape": policy.preflight_output_shape},
        "completion_audit_sha256": AUDIT_SHA256, "calibration": calibration,
        "contract_id": contract.contract_id}


def run(output_dir: Path, *, contract_path=DEFAULT_CONTRACT, spec_path=DEFAULT_SPEC,
        model_path=DEFAULT_MODEL, metadata_path=DEFAULT_METADATA) -> dict[str, Any]:
    run_start = time.perf_counter()
    if output_dir.exists():
        raise RuntimeError("formal output directory must not be pre-created")
    spec, metadata, contract = load_frozen_inputs(contract_path, spec_path, model_path, metadata_path)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    if dirty: raise RuntimeError("formal evaluation requires clean committed source")
    completion_digest = source_digest(ROOT, COMPLETION_SOURCE_FILES)
    experiment_digest = source_digest(ROOT, ISSUE58_SOURCE_FILES)
    validation_record = json.loads(DEFAULT_PRE_RUN_VALIDATION.read_text(encoding="utf-8"))
    if validation_record.get("source_sha256") != experiment_digest:
        raise RuntimeError("pre-run validation package does not match frozen Issue #58 source digest")
    verify_calibration_assets()
    calibration = validation_record.get("calibration_scan", {})
    frozen_calibration = frozen_calibration_record()
    if calibration != json.loads(json.dumps(frozen_calibration)):
        raise RuntimeError("pre-run package calibration scan does not match pinned Issue #58 result")
    validation_seconds = float(validation_record["validation"]["combined_wall_seconds"])
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
    if validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + setup_seconds > 20:
        env.close(); raise RuntimeError(f"combined tests/preflight/process-launch/setup cap exceeded: {validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + setup_seconds:.3f}s")
    rows, total_native, stop_reason, verified_clear = [], 0, "schedule_completed", False
    canonical_clear_detected = False
    collection_start = time.perf_counter()
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory, stack_index_path.open("w", encoding="utf-8") as stack_index:
            for episode_index, (seed, arm) in enumerate(SCHEDULE, 1):
                total_decisions = sum(int(r["agent_steps"]) for r in rows)
                if (time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT
                    or total_native >= TOTAL_FRAME_LIMIT or total_decisions >= TOTAL_STEP_LIMIT):
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
                    total_decisions = sum(int(r["agent_steps"]) for r in rows) + steps
                    if (time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT
                        or total_native >= TOTAL_FRAME_LIMIT or total_decisions >= TOTAL_STEP_LIMIT):
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
                        "source_commit": commit, "source_branch": branch,
                        "emulator_frame": frame, "native_frames": episode_native,
                        "observation_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
                        "stack_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
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
                    stack_jobs = [(reason, obs, row) for reason in stack_capture_reasons(
                        override=decision["override_used"], life_loss=lost,
                        canonical_clear=clear.cleared is True, life_window_active=bool(active_life_windows))]
                    for window in active_life_windows:
                        window["remaining"] -= 1
                        if window["remaining"] <= 0:
                            pending_life.remove(window)
                    if lost:
                        stack_jobs.extend(("life_loss_pre", item["stack"], item["row"]) for item in history)
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
                            "source_commit": commit, "source_branch": branch, "seed": seed, "arm": arm,
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
                if not continue_after_episode(state.cleared is True):
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
                if state.cleared is True:
                    canonical_clear_detected = True
                    gate = clear_stop_outcome(True, verified_clear, arm)
                    stop_reason = "GOAL_REACHED_verified_clear" if verified_clear else "HUMAN_REVIEW_REQUIRED_unverified_clear"
                    break
                total_decisions = sum(int(r["agent_steps"]) for r in rows)
                if (time.perf_counter() - collection_start >= COLLECTION_WALL_LIMIT
                    or total_native >= TOTAL_FRAME_LIMIT or total_decisions >= TOTAL_STEP_LIMIT):
                    stop_reason = "aggregate_cap"; break
    finally:
        env.close()
        for handle in stack_handles.values():
            handle.close()
    collection_seconds = time.perf_counter() - collection_start
    if collection_seconds > COLLECTION_WALL_LIMIT: raise RuntimeError("collection exceeded 560s")
    final_start = time.perf_counter()
    verified_arm = next((r["arm"] for r in rows if r.get("clear_status") == "VERIFIED_CLEAR"), None)
    outcome = (clear_stop_outcome(True, False, rows[-1]["arm"]) if canonical_clear_detected and not verified_clear
               else classify_round(verified_arm))
    status = "human_review_required" if canonical_clear_detected and not verified_clear else evaluation_status(len(rows), verified_clear)
    baseline = {r["seed"]: r for r in rows if r["arm"] == "baseline"}
    candidate = {r["seed"]: r for r in rows if r["arm"] == "candidate"}
    paired_deltas = [{"seed": seed,
        "native_frames_candidate_minus_baseline": candidate[seed]["native_frames"] - baseline[seed]["native_frames"],
        "raw_score_candidate_minus_baseline": candidate[seed]["raw_score"] - baseline[seed]["raw_score"],
        "life_losses_candidate_minus_baseline": candidate[seed]["life_losses"] - baseline[seed]["life_losses"]}
        for seed in SEEDS if seed in baseline and seed in candidate]
    results = {"schema_version": 1, "issue": 58, "evaluation_status": status, **outcome,
        "has_verified_clear": verified_clear,
        "canonical_clear_detected": canonical_clear_detected,
        "model": {"model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
            "inference_spec_sha256": SPEC_SHA256, "metadata_inference_spec_sha256": SPEC_METADATA_SHA256,
            "contract_sha256": CONTRACT_SHA256, "completion_audit_sha256": AUDIT_SHA256,
            "metadata_lineage_note": SPEC_LINEAGE_NOTE,
            "calibration_index_sha256": CALIBRATION_INDEX_SHA256,
            "calibration_results_sha256": CALIBRATION_RESULTS_SHA256,
            "calibration_trajectory_sha256": CALIBRATION_TRAJECTORY_SHA256,
            "calibration_use": "T=80 selected from archived Issue #50 artifacts; this threshold feeds the candidate detector and action rule; activation is not object truth"},
        "calibration_selection": calibration,
        "runtime": {"python": platform.python_version(), "onnxruntime": policy.ort.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"), "ale_py": importlib.metadata.version("ale-py"),
            "providers": list(policy.session.get_providers()), "requested_providers": ["CPUExecutionProvider"],
            "onnx_input": {"name": policy.input_name, "dtype": "float32", "shape": [1,4,84,84]},
            "onnx_output": {"name": policy.output_name, "dtype": "float32", "shape": policy.preflight_output_shape},
            "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE"},
        "source_provenance": {"source_commit": commit, "source_branch": branch, "source_working_tree_dirty": dirty,
            "issue58_source_sha256": experiment_digest, "issue58_source_files": list(ISSUE58_SOURCE_FILES),
            "completion_source_sha256": completion_digest, "completion_source_files": list(COMPLETION_SOURCE_FILES)},
        "pre_run_validation": {"package": str(DEFAULT_PRE_RUN_VALIDATION),
            "package_sha256": sha256(DEFAULT_PRE_RUN_VALIDATION), "source_sha256": experiment_digest,
            "validation": validation_record["validation"]},
        "evaluation_protocol": {"contract_id": contract.contract_id, "seeds": list(SEEDS),
            "ordered_schedule": [{"seed": s, "arm": a} for s,a in SCHEDULE],
            "max_native_frames_per_episode": FRAME_LIMIT, "max_agent_steps_per_episode": STEP_LIMIT,
            "aggregate_native_frame_cap": TOTAL_FRAME_LIMIT, "aggregate_agent_decision_cap": TOTAL_STEP_LIMIT,
            "collection_wall_cap_seconds": COLLECTION_WALL_LIMIT,
            "roi": {"ball": list(BALL_ROI), "paddle": list(PADDLE_ROI)}, "bright_threshold": BRIGHT_THRESHOLD,
            "component_connectivity": 8, "action_rule": "RIGHT=2 if delta>=2; LEFT=3 if delta<=-2; else NOOP=0; fallback raw argmax",
            "baseline": "raw ONNX argmax regardless of detector output",
        "candidate_inputs": "channel 3 pixels only; no RAM/score/lives/completion/Q influence on override"},
        "sticky_action_note": "Contract v2 sticky-action probability is 0.25; sticky resolution can make the physical ALE action differ from the requested policy/ALE-input action, so controller effects retain this uncertainty.",
        "completion_support": support.to_dict(), "episodes": rows, "native_frames": sum(r["native_frames"] for r in rows),
        "agent_decisions": sum(r["agent_steps"] for r in rows),
        "paired_deltas_descriptive_only_non_adjudicating": paired_deltas,
        "wall_accounting": {"setup_seconds": setup_seconds, "setup_native_frames": 0,
            "collection_seconds": collection_seconds, "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "total_wall_cap_seconds": TOTAL_WALL_LIMIT}, "preflight_native_frames": 0,
        "exact_formal_command": ["timeout", "--signal=INT", "--kill-after=5s", "600s", "env",
            "PYTHONPATH=/tmp/issue41-onnxruntime", "python", "-m",
            "scripts.evaluation.run_issue58_pixel_paddle_alignment", "--output-dir",
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
    if canonical_clear_detected and not verified_clear:
        failure_fact = "A canonical clear signal occurred but provenance validation failed. Collection stopped immediately and HUMAN_REVIEW_REQUIRED; no verified clear is claimed."
    elif verified_arm is None:
        failure_fact = (f"No provenance-complete canonical clear in {len(rows)} collected episode(s); "
                        f"{zero_life_episodes} episode(s) ended at zero lives. This is descriptive failure context only; "
                        "the classification remains HVC-only and INCONCLUSIVE without a verified clear.")
    elif verified_arm == "candidate":
        failure_fact = "A provenance-complete candidate clear reached the frozen goal; the baseline/candidate diagnostics do not alter that HVC result."
    else:
        failure_fact = "A provenance-complete baseline clear reached the goal; the candidate hypothesis remains NOT_ADJUDICATED."
    earlier_validation_attempts = validation_record["validation"].get("earlier_validation_attempts", [])
    if earlier_validation_attempts:
        attempts_wall = sum(float(item["wall_seconds"]) for item in earlier_validation_attempts)
        validation_history = (
            f"Earlier validation work totaled {validation_record['validation']['earlier_validation_work_wall_seconds']:.6f}s, including the two listed failed assertions, whose measured durations sum to {attempts_wall:.6f}s: "
            f"`{json.dumps(earlier_validation_attempts, sort_keys=True)}`. These were focused-code checks only; none created, reset, or stepped ALE. "
            f"A prior exact-head passing validation charged {validation_record['validation']['other_earlier_validation_work_wall_seconds']:.6f}s; this exact-head passing validation charged {validation_record['validation']['current_final_validation_wall_seconds']:.6f}s; "
            f"cumulative validation cap charge is {validation_record['validation']['combined_wall_seconds']:.6f}s."
        )
    else:
        validation_history = "No earlier validation failures were recorded."
    report = ["# Issue #58: Calibrated Grayscale Paddle Alignment First-Clear Probe", "",
        "## Hypothesis Result", "",
        f"**{outcome['classification']}**; HVC `{outcome['hvc_status']}`; round status `{outcome['round_status']}`; candidate hypothesis `{outcome['candidate_hypothesis_status'] }`.", "",
        "## Baseline", "", "Unchanged ONNX raw `argmax(Q)` at every decision. Detector output is logged in both arms and cannot affect baseline actions.", "",
        "## Primary Result", "", f"Has Verified Clear: **{'YES' if verified_clear else 'NO'}**; evaluation status `{status}`; episodes `{len(rows)}`; stop `{stop_reason}`.", "",
        "## Delta vs Baseline", "", "Matched diagnostics below are descriptive-only and non-adjudicating; they do not change HVC classification.", "", *table, "",
        "## Failure Analysis", "", failure_fact, "Per-decision detection/fallback results and evaluator trace are in `trajectory.jsonl`. No proxy metric can promote, reject, or update a champion.", "",
        "## Reproducibility", "", f"Formal command: `{' '.join(results['exact_formal_command'])}`.", "",
        f"Source branch `{branch}`, commit `{commit}`; source digest `{experiment_digest}`; completion-source digest `{completion_digest}`.",
        f"Model `{MODEL_SHA256}`; metadata `{METADATA_SHA256}`; spec `{SPEC_SHA256}` (metadata-declared historical spec `{SPEC_METADATA_SHA256}`); Contract v2 `{CONTRACT_SHA256}`; completion audit `{AUDIT_SHA256}`; Issue #50 index `{CALIBRATION_INDEX_SHA256}`, results `{CALIBRATION_RESULTS_SHA256}`, trajectory `{CALIBRATION_TRAJECTORY_SHA256}`.",
        f"Issue #50 calibration stack hashes `{json.dumps(calibration['stack_sha256s'], sort_keys=True)}`.",
        f"Runtime Python `{platform.python_version()}`, gymnasium `{importlib.metadata.version('gymnasium')}`, ALE `{importlib.metadata.version('ale-py')}`, ONNX Runtime `{policy.ort.__version__}`, provider `CPUExecutionProvider`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.",
        f"Frozen calibration scan from 300 indexed rows / 150 unique observations: selected T=80; activation counts {calibration['activation_counts_by_threshold']}; this is a detector diagnostic, not object truth.",
        SPEC_LINEAGE_NOTE, "", f"Focused test command `{validation_record['validation']['test_command']}` passed {validation_record['validation']['test_count']} tests in {validation_record['validation']['test_wall_seconds']:.3f}s. Compile took {validation_record['validation']['compile_wall_seconds']:.3f}s; zero-frame CPU preflight took {validation_record['validation']['preflight_wall_seconds']:.3f}s with `ale_environment_created=false`.",
        validation_history, "",
        "Contract v2 sticky-action probability is 0.25; sticky resolution can make the physical ALE action differ from the requested policy/ALE-input action, so controller effects retain this uncertainty.",
        "Frozen caps: validation plus process-launch reserve and formal setup 20s; collection 560s; finalization 20s; total wall 600s; each episode 108,000 native frames / 27,000 decisions; six-episode aggregate 648,000 native frames / 162,000 decisions. The external timeout includes final manifest writing.", "",
        "## Tests and Setup", "", "No detector smoke fixture was used; setup/preflight consumed zero ALE-native frames.", "",
        "## Remaining Uncertainty", "", ("A canonical clear was detected without complete provenance; human review is required, and no HVC yes/no classification is claimed." if canonical_clear_detected and not verified_clear else "This is current-frame horizontal alignment; it does not estimate ball velocity or a future intercept. No-clear remains INCONCLUSIVE under HVC-only evaluation."), "",
        "## Recommended Next Decision", "", ("Verified clear reached the frozen goal; do not transition Phase 2 automatically." if verified_clear else ("Review the unverified canonical clear signal; do not continue collection." if canonical_clear_detected else "Continue research under HVC-only criteria; do not rank or promote from diagnostics.")), ""]
    report_path.write_text("\n".join(report), encoding="utf-8")
    total = time.perf_counter() - run_start
    actual_finalization = time.perf_counter() - final_start
    results["wall_accounting"]["finalization_seconds"] = actual_finalization
    results["wall_accounting"]["total_formal_seconds"] = total
    if (actual_finalization > FINALIZATION_RESERVE_SECONDS
        or validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + total > TOTAL_WALL_LIMIT):
        raise RuntimeError("finalization or total wall cap exceeded")
    results["wall_accounting"]["total_including_validation_seconds"] = validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + total
    results["wall_accounting"]["combined_validation_setup_seconds"] = validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + setup_seconds
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps({"issue": 58, "source_commit": commit, "source_branch": branch,
        "issue58_source_sha256": experiment_digest, "classification": outcome["classification"],
        "round_status": outcome["round_status"], "exact_formal_command": results["exact_formal_command"],
        "pre_run_validation_package": results["pre_run_validation"],
        "calibration_selection": calibration,
        "validation": validation_record["validation"],
        "wall_accounting": {**results["wall_accounting"],
            "formal_seconds": total, "total_seconds": validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + total,
            "process_launch_reserve_seconds": PROCESS_LAUNCH_RESERVE_SECONDS,
            "collection_cap_seconds": COLLECTION_WALL_LIMIT,
            "setup_plus_test_cap_seconds": 20.0, "finalization_cap_seconds": FINALIZATION_RESERVE_SECONDS,
            "measurement_scope": "runner elapsed and finalization are measured through the completed runner manifest write; CLI elapsed includes runner work, console summary, artifact packaging, and terminal report/results/manifest/sidecar writes"},
        "artifacts": {"results_sha256": sha256(results_path), "report_sha256": sha256(report_path),
            "trajectory_sha256": sha256(trajectory_path), "stack_index_sha256": sha256(stack_index_path)}},
        indent=2, sort_keys=True) + "\n")
    # Include final manifest serialization in the internal finalization/total guards.
    post_manifest_wall = time.perf_counter() - run_start
    post_manifest_finalization = time.perf_counter() - final_start
    if (post_manifest_finalization > FINALIZATION_RESERVE_SECONDS
        or validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + post_manifest_wall > TOTAL_WALL_LIMIT):
        raise RuntimeError("final manifest write exceeded frozen finalization or total wall cap")
    results["wall_accounting"].update({
        "runner_elapsed_through_manifest_seconds": post_manifest_wall,
        "runner_finalization_through_manifest_seconds": post_manifest_finalization,
        "process_launch_reserve_seconds": PROCESS_LAUNCH_RESERVE_SECONDS,
        "formal_seconds": post_manifest_wall,
        "total_seconds": validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + post_manifest_wall,
        "finalization_seconds": post_manifest_finalization,
        "total_formal_seconds": post_manifest_wall,
        "total_including_validation_seconds": validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + post_manifest_wall,
        "combined_validation_setup_seconds": validation_seconds + PROCESS_LAUNCH_RESERVE_SECONDS + setup_seconds,
        "measurement_complete_through": "runner manifest write",
    })
    return results


def cli(argv: list[str] | None = None, *, process_start: float | None = None) -> int:
    cli_start = process_start if process_start is not None else time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2))
        return 0
    stdout_path, stderr_path = Path("/tmp/issue58-formal.stdout"), Path("/tmp/issue58-formal.stderr")
    if stdout_path.exists() or stderr_path.exists():
        raise RuntimeError("formal console capture path already exists; frozen run cannot be retried")
    class Tee:
        def __init__(self, console, capture): self.console, self.capture = console, capture
        def write(self, value):
            self.console.write(value); self.capture.write(value); self.capture.flush(); return len(value)
        def flush(self): self.console.flush(); self.capture.flush()
        def __getattr__(self, name): return getattr(self.console, name)
    console_out, console_err = sys.stdout, sys.stderr
    with stdout_path.open("w", encoding="utf-8") as captured_out, stderr_path.open("w", encoding="utf-8") as captured_err:
        sys.stdout, sys.stderr = Tee(console_out, captured_out), Tee(console_err, captured_err)
        try:
            result = run(args.output_dir)
            print(json.dumps({"classification": result["classification"],
                "episodes": len(result["episodes"]), "native_frames": result["native_frames"],
                "stop_reason": result["stop_reason"]}, indent=2))
        finally:
            sys.stdout, sys.stderr = console_out, console_err
    wrapup_start = time.perf_counter()
    output_stdout, output_stderr = args.output_dir / "formal.stdout", args.output_dir / "formal.stderr"
    shutil.copyfile(stdout_path, output_stdout); shutil.copyfile(stderr_path, output_stderr)
    results_path = args.output_dir / "results.json"
    result_data = json.loads(results_path.read_text(encoding="utf-8"))
    runner_wall = result["wall_accounting"]
    for key in ("runner_elapsed_through_manifest_seconds", "runner_finalization_through_manifest_seconds",
                "process_launch_reserve_seconds", "finalization_seconds", "formal_seconds", "total_seconds",
                "total_formal_seconds", "total_including_validation_seconds", "combined_validation_setup_seconds",
                "measurement_complete_through"):
        result_data["wall_accounting"][key] = runner_wall[key]
    result_data["artifacts"]["formal_stdout"] = {"file": output_stdout.name, "sha256": sha256(output_stdout), "bytes": output_stdout.stat().st_size}
    result_data["artifacts"]["formal_stderr"] = {"file": output_stderr.name, "sha256": sha256(output_stderr), "bytes": output_stderr.stat().st_size}
    result_data["artifacts"]["manifest_sha256_sidecar"] = "/tmp/issue58-formal-manifest.sha256"
    validation_seconds = float(result["pre_run_validation"]["validation"]["combined_wall_seconds"])
    launch_charge = PROCESS_LAUNCH_RESERVE_SECONDS
    report_path = args.output_dir / "report.md"
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(f"\nRaw formal stdout: `formal.stdout` SHA-256 `{sha256(output_stdout)}`.\n")
        report_file.write(f"Raw formal stderr: `formal.stderr` SHA-256 `{sha256(output_stderr)}`.\n")
    wrapup_seconds = time.perf_counter() - wrapup_start
    inner_finalization_seconds = float(runner_wall["runner_finalization_through_manifest_seconds"])
    result_data["wall_accounting"]["post_run_artifact_packaging_seconds"] = wrapup_seconds
    result_data["wall_accounting"]["total_including_validation_and_packaging_seconds"] = validation_seconds + launch_charge + time.perf_counter() - cli_start
    result_data["wall_accounting"]["finalization_seconds"] = inner_finalization_seconds
    if inner_finalization_seconds + wrapup_seconds > FINALIZATION_RESERVE_SECONDS or result_data["wall_accounting"]["total_including_validation_and_packaging_seconds"] > TOTAL_WALL_LIMIT:
        raise RuntimeError("outer artifact packaging exceeded finalization or total wall cap")
    results_path.write_text(json.dumps(result_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(f"Pre-run validation plus formal command and artifact packaging elapsed: {result_data['wall_accounting']['total_including_validation_and_packaging_seconds']:.3f}s.\n")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["wall_accounting"].update({key: runner_wall[key] for key in (
        "runner_elapsed_through_manifest_seconds", "runner_finalization_through_manifest_seconds",
        "process_launch_reserve_seconds", "formal_seconds", "total_seconds", "total_formal_seconds", "total_including_validation_seconds",
        "combined_validation_setup_seconds", "measurement_complete_through")})
    manifest["raw_console_outputs"] = {
        "stdout_path": str(stdout_path), "stdout_sha256": sha256(stdout_path),
        "stderr_path": str(stderr_path), "stderr_sha256": sha256(stderr_path),
        "capture_location": "outside the isolated worktree; byte-identical copies are included in the run artifacts",
        "artifact_stdout": output_stdout.name, "artifact_stdout_sha256": sha256(output_stdout),
        "artifact_stderr": output_stderr.name, "artifact_stderr_sha256": sha256(output_stderr)}
    manifest["manifest_sha256_sidecar"] = "/tmp/issue58-formal-manifest.sha256"
    manifest["wrapper_timing_sidecar"] = "/tmp/issue58-formal-wrapper-timing.json"
    manifest["wall_accounting"]["post_run_artifact_packaging_seconds"] = wrapup_seconds
    manifest["wall_accounting"]["total_including_validation_and_packaging_seconds"] = result_data["wall_accounting"]["total_including_validation_and_packaging_seconds"]
    manifest["wall_accounting"]["finalization_seconds"] = result_data["wall_accounting"]["finalization_seconds"]
    manifest["artifacts"].update({"results_sha256": sha256(results_path), "report_sha256": sha256(report_path),
        "formal_stdout_sha256": sha256(output_stdout), "formal_stderr_sha256": sha256(output_stderr)})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path)
    Path("/tmp/issue58-formal-manifest.sha256").write_text(f"{manifest_sha}  {manifest_path}\n", encoding="utf-8")
    actual_elapsed = time.perf_counter() - cli_start
    actual_wrapup = time.perf_counter() - wrapup_start
    actual_finalization = float(runner_wall["runner_finalization_through_manifest_seconds"]) + actual_wrapup
    if validation_seconds + launch_charge + actual_elapsed > TOTAL_WALL_LIMIT or actual_finalization > FINALIZATION_RESERVE_SECONDS:
        raise RuntimeError("actual outer wrapper elapsed exceeded the frozen finalization or total wall cap")
    timing_path = args.output_dir / "wrapper_timing.json"
    timing = {"outer_command_elapsed_seconds": actual_elapsed,
        "validation_plus_outer_command_elapsed_seconds": validation_seconds + launch_charge + actual_elapsed,
        "process_launch_reserve_seconds": launch_charge,
        "outer_artifact_finalization_seconds": actual_finalization,
        "measurement_complete_through": "manifest hash sidecar write before binding wrapper_timing.json",
        "final_manifest_sha256_sidecar": "/tmp/issue58-formal-manifest.sha256"}
    timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_data["wall_accounting"]["outer_command_elapsed_seconds"] = actual_elapsed
    result_data["wall_accounting"]["post_run_artifact_packaging_seconds"] = actual_wrapup
    result_data["wall_accounting"]["total_including_validation_and_packaging_seconds"] = validation_seconds + launch_charge + actual_elapsed
    result_data["wall_accounting"]["finalization_seconds"] = actual_finalization
    result_data["artifacts"]["wrapper_timing"] = {"file": timing_path.name, "sha256": sha256(timing_path)}
    results_path.write_text(json.dumps(result_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(f"\nRunner elapsed through manifest {runner_wall['runner_elapsed_through_manifest_seconds']:.3f}s; runner finalization through manifest {runner_wall['runner_finalization_through_manifest_seconds']:.3f}s. Wrapper timing artifact: `wrapper_timing.json` SHA-256 `{sha256(timing_path)}`; outer elapsed {actual_elapsed:.3f}s, launch reserve {launch_charge:.3f}s, combined validation/command {validation_seconds + launch_charge + actual_elapsed:.3f}s, finalization {actual_finalization:.3f}s.\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["wrapper_timing"] = {"file": timing_path.name, "sha256": sha256(timing_path)}
    manifest["wall_accounting"].update({"outer_command_elapsed_seconds": actual_elapsed,
        "total_including_validation_and_packaging_seconds": validation_seconds + launch_charge + actual_elapsed,
        "finalization_seconds": actual_finalization})
    manifest["artifacts"].update({"results_sha256": sha256(results_path), "report_sha256": sha256(report_path),
        "wrapper_timing_sha256": sha256(timing_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path)
    Path("/tmp/issue58-formal-manifest.sha256").write_text(f"{manifest_sha}  {manifest_path}\n", encoding="utf-8")
    final_elapsed = time.perf_counter() - cli_start
    final_outer_wrapup = time.perf_counter() - wrapup_start
    terminal_write_seconds = max(0.0, final_outer_wrapup - actual_wrapup)
    finalization_total = actual_finalization + terminal_write_seconds
    if validation_seconds + launch_charge + final_elapsed > TOTAL_WALL_LIMIT or finalization_total > FINALIZATION_RESERVE_SECONDS:
        raise RuntimeError("terminal report/manifest writes exceeded frozen total or finalization cap")
    Path("/tmp/issue58-formal-wrapper-timing.json").write_text(json.dumps({
        "outer_command_elapsed_seconds_through_terminal_writes": final_elapsed,
        "validation_plus_outer_command_elapsed_seconds": validation_seconds + launch_charge + final_elapsed,
        "outer_artifact_finalization_seconds_through_terminal_writes": finalization_total,
        "artifact_wrapper_timing_sha256": sha256(timing_path),
        "final_manifest_sha256": manifest_sha,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
