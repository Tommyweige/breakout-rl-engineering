"""Offline Issue #54 ONNX Q replay consistency audit."""
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

ROOT = Path(__file__).resolve().parents[1]
ISSUE50_DIR = ROOT / "research/issue-50-pixel-paddle-alignment-artifacts"
OUTPUT_RELATIVE = "research/issue-54-q-replay-consistency-artifacts"
BASE_COMMIT = "a8641d99d6a0eb5b7226dae2b82ed3479f35e55f"
ISSUE50_SOURCE_DIGEST = "e32be297ba474635708fefd7da311e5db37b6076439e759be2701d4aaf2fee2f"
INDEX_SHA256 = "757eab8a54327e0663e2d83e5156958b776c1c5e7a09521a3355a9af43746978"
TRAJECTORY_SHA256 = "48c3ad056962862ba8a80219909caa6f4349a704642cef65b3210f62b47a2398"
RESULTS_SHA256 = "6954370d9270b154de55c38c26d107b25347a82d3ecf86bc723cdf03b68cf582"
REPORT_SHA256 = "1c9a57bf3ce850b836f5768276b64829f29faf6842089c37022f80318b2247d1"
ISSUE50_MANIFEST_SHA256 = "91d7e7528d12644a5de154400ea479581d68b9d5e44650709b635568bbce7063"
MODEL_SHA256 = "cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12"
METADATA_SHA256 = "fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512"
SPEC_SHA256 = "b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507"
CONTRACT_SHA256 = "7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a"
STACK_SHA256 = {
    "seed105_baseline_ep1.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa",
    "seed105_candidate_ep2.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa",
    "seed206_baseline_ep3.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55",
    "seed206_candidate_ep4.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55",
    "seed307_baseline_ep5.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95",
    "seed307_candidate_ep6.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95",
}
ISSUE50_SOURCE_FILES = (
    "breakout_rl/issue50_pixel_paddle_alignment.py",
    "scripts/evaluation/run_issue50_pixel_paddle_alignment.py",
    "tests/test_issue50_pixel_paddle_alignment.py",
)
STACK_FILENAMES = tuple(STACK_SHA256)
REPLAY_SOURCE_FILES = (
    "breakout_rl/issue54_q_replay_consistency.py",
    "scripts/analysis/run_issue54_q_replay_consistency.py",
    "tests/test_issue54_q_replay_consistency.py",
    "breakout_rl/issue50_pixel_paddle_alignment.py",
)
PRE_RUN_VALIDATION = {
    "focused_test_command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue54_q_replay_consistency -v",
    "focused_test_count": 7,
    "focused_test_status": "pending final timed validation",
    "focused_test_wall_seconds": None,
    "compile_command": "python -m py_compile breakout_rl/issue54_q_replay_consistency.py scripts/analysis/run_issue54_q_replay_consistency.py tests/test_issue54_q_replay_consistency.py",
    "compile_status": "pending final timed validation",
    "compile_wall_seconds": None,
    "diff_check_command": "git diff --check",
    "diff_check_status": "pending final timed validation",
    "diff_check_wall_seconds": None,
    "preflight_command": "timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.analysis.run_issue54_q_replay_consistency --output-dir /tmp/issue54-preflight --preflight-only",
    "preflight_status": "pending final timed validation",
    "preflight_wall_seconds": None,
    "preflight_matched_rows": 300,
    "preflight_replay_started": False,
    "preflight_native_frames": 0,
}
PRE_RUN_VALIDATION_SECONDS = 0.0
KEY_FIELDS = ("seed", "arm", "episode_index", "agent_step", "emulator_frame")
EXPECTED_FILES = {
    "web/public/models/final_model/model.onnx": MODEL_SHA256,
    "web/public/models/final_model/model.onnx.metadata.json": METADATA_SHA256,
    "configs/inference/inference_spec.json": SPEC_SHA256,
    "configs/eval/breakout_contract_v2.json": CONTRACT_SHA256,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_digest(root: Path = ROOT, files: tuple[str, ...] = ISSUE50_SOURCE_FILES) -> str:
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode()); digest.update(b"\0")
        digest.update((root / name).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def replay_source_digest(root: Path = ROOT) -> str:
    return source_digest(root, REPLAY_SOURCE_FILES)


def observation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def match_rows(index_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return a unique exact key join and verify both artifacts' observation hashes."""
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        by_key.setdefault(observation_key(row), []).append(row)
    seen: set[tuple[Any, ...]] = set()
    pairs = []
    for indexed in index_rows:
        key = observation_key(indexed)
        if key in seen:
            raise ValueError(f"duplicate index key: {key}")
        seen.add(key)
        matches = by_key.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"index key must match exactly one trajectory row: {key}")
        logged = matches[0]
        index_hash = indexed.get("observation_sha256", indexed.get("sha256"))
        trajectory_hash = logged.get("observation_sha256")
        if not index_hash or index_hash != trajectory_hash:
            raise ValueError(f"observation hash mismatch for key: {key}")
        q = logged.get("q_values")
        if not isinstance(q, list) or len(q) != 4 or not np.isfinite(np.asarray(q, dtype=np.float64)).all():
            raise ValueError(f"logged Q vector must contain four finite values: {key}")
        pairs.append((indexed, logged))
    if len(index_rows) != 300:
        raise ValueError(f"expected exactly 300 index rows, got {len(index_rows)}")
    return pairs


def verify_stack_row(root: Path, indexed: dict[str, Any], stack_streams: dict[str, bytes] | None = None) -> tuple[bytes, str]:
    filename = Path(indexed["file"])
    if filename.is_absolute() or ".." in filename.parts or filename.parts[0] != "decision_stacks":
        raise ValueError(f"unsafe or unexpected stack path: {filename}")
    if filename.name not in STACK_SHA256:
        raise ValueError(f"unrecognized stack file: {filename}")
    path = root / "decision_stacks" / filename.name
    data = stack_streams[filename.name] if stack_streams is not None else path.read_bytes()
    if (indexed.get("bytes") != 28_224 or indexed.get("dtype") != "uint8"
            or indexed.get("shape") != [4, 84, 84]
            or indexed.get("byte_offset", -1) < 0
            or indexed.get("byte_offset", -1) % 28_224 != 0
            or len(data) != 1_411_200
            or indexed["byte_offset"] + indexed["bytes"] > len(data)):
        raise ValueError(f"invalid indexed stack metadata at {observation_key(indexed)}")
    stack_bytes = data[indexed["byte_offset"]:indexed["byte_offset"] + indexed["bytes"]]
    actual = sha256_bytes(stack_bytes)
    if actual != indexed.get("sha256") or actual != indexed.get("observation_sha256"):
        raise ValueError(f"indexed stack bytes fail observation hash at {observation_key(indexed)}")
    if stack_streams is None and sha256_bytes(data) != STACK_SHA256[filename.name]:
        raise ValueError(f"whole stack stream hash mismatch: {filename.name}")
    return stack_bytes, actual


def validate_capture_offsets(index_rows: list[dict[str, Any]], *, complete: bool = True) -> None:
    """Check sparse agent steps against contiguous per-file storage ordinals."""
    next_offset: dict[str, int] = {}
    for indexed in index_rows:
        filename = Path(indexed["file"]).name
        expected_offset = next_offset.get(filename, 0)
        if indexed.get("byte_offset") != expected_offset:
            raise ValueError(f"noncontiguous per-file capture offset for {filename}: expected {expected_offset}")
        next_offset[filename] = expected_offset + 28_224
    if complete and (set(next_offset) != set(STACK_SHA256)
                     or any(offset != 1_411_200 for offset in next_offset.values())):
        raise ValueError("expected 50 contiguous indexed stacks in each of six frozen files")


def prepare_model_input(stack: np.ndarray) -> np.ndarray:
    """Apply frozen Issue #50 preprocessing once: uint8 -> float32 / 255 -> NCHW."""
    if not isinstance(stack, np.ndarray) or stack.dtype != np.uint8 or stack.shape != (4, 84, 84):
        raise ValueError("stack must be uint8 [4,84,84]")
    return np.ascontiguousarray(stack[None].astype(np.float32) / np.float32(255.0))


def max_q_error(logged: Any, replayed: Any) -> float:
    a = np.asarray(logged, dtype=np.float32)
    b = np.asarray(replayed, dtype=np.float32)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("logged and replayed Q vectors must each be four finite float32 values")
    return float(np.max(np.abs(b - a)))


def classify_diagnostic(integrity_passed: bool, max_abs_error: float | None) -> str:
    if not integrity_passed or max_abs_error is None or not np.isfinite(max_abs_error):
        return "INCONCLUSIVE"
    return "PROMOTED" if max_abs_error <= 1e-6 else "REJECTED"


def remaining_budget(pre_run_seconds: float, cap_seconds: float = 20.0) -> float:
    if not np.isfinite(pre_run_seconds) or not np.isfinite(cap_seconds) or pre_run_seconds < 0.0:
        raise ValueError("elapsed validation time and cap must be finite and nonnegative")
    remaining = cap_seconds - pre_run_seconds
    if remaining <= 0.0:
        raise TimeoutError("pre-run validation exhausted the combined 20 second budget")
    return remaining


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_frozen_inputs(root: Path = ROOT) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, str], dict[str, bytes]]:
    input_hashes: dict[str, str] = {}
    issue50_files = {
        "decision_stack_index.jsonl": INDEX_SHA256,
        "trajectory.jsonl": TRAJECTORY_SHA256,
        "results.json": RESULTS_SHA256,
        "report.md": REPORT_SHA256,
        "manifest.json": ISSUE50_MANIFEST_SHA256,
    }
    for name, expected in issue50_files.items():
        path = ISSUE50_DIR / name
        actual = sha256_file(path)
        input_hashes[f"issue50/{name}"] = actual
        if actual != expected:
            raise ValueError(f"Issue #50 source hash mismatch: {name}")
    for name, expected in EXPECTED_FILES.items():
        actual = sha256_file(root / name)
        input_hashes[name] = actual
        if actual != expected:
            raise ValueError(f"frozen model/config hash mismatch: {name}")
    stack_streams = {name: (ISSUE50_DIR / "decision_stacks" / name).read_bytes() for name in STACK_SHA256}
    for name, expected in STACK_SHA256.items():
        actual = sha256_bytes(stack_streams[name])
        input_hashes[f"issue50/decision_stacks/{name}"] = actual
        if actual != expected:
            raise ValueError(f"frozen whole stack hash mismatch: {name}")
    if source_digest(root) != ISSUE50_SOURCE_DIGEST:
        raise ValueError("Issue #50 source digest mismatch")
    index_rows = load_jsonl(ISSUE50_DIR / "decision_stack_index.jsonl")
    trajectory_rows = load_jsonl(ISSUE50_DIR / "trajectory.jsonl")
    pairs = match_rows(index_rows, trajectory_rows)
    validate_capture_offsets(index_rows)
    for indexed, _ in pairs:
        verify_stack_row(ISSUE50_DIR, indexed, stack_streams)
    return pairs, input_hashes, stack_streams


def repo_clean(root: Path = ROOT) -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True)
    return not result.stdout.strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def run(output_dir: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    replay_budget_seconds = remaining_budget(PRE_RUN_VALIDATION_SECONDS)
    deadline = started + replay_budget_seconds
    if not repo_clean():
        raise RuntimeError("clean-source check failed; output directory must not yet exist")
    pairs, input_hashes, stack_streams = verify_frozen_inputs()
    if preflight_only:
        return {"status": "preflight_passed", "matched_rows": len(pairs), "input_hashes": input_hashes,
                "wall_seconds": time.perf_counter() - started, "replay_started": False,
                "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE", "native_frames": 0}
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    # Hash/row validation is repeated before the first model inference.
    import onnxruntime as ort
    if str(ort.__version__) != "1.22.1":
        raise RuntimeError(f"expected ONNX Runtime 1.22.1, got {ort.__version__}")
    spec = json.loads((ROOT / "configs/inference/inference_spec.json").read_text())
    if (spec["input"] != {"name": "observation", "dtype": "float32", "shape": ["N", 4, 84, 84],
                          "layout": "NCHW", "range": [0.0, 1.0]}
            or spec["output"] != {"name": "q_values", "dtype": "float32", "shape": ["N", 4],
                                  "meaning": "raw Q-values, not probabilities"}
            or spec["preprocessing"]["normalization_divisor"] != 255.0):
        raise ValueError("pinned inference spec does not match frozen Q replay contract")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "web/public/models/final_model/model.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    providers = tuple(session.get_providers())
    if providers != ("CPUExecutionProvider",):
        raise RuntimeError(f"expected CPUExecutionProvider only, got {providers}")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1
            or (inputs[0].name, inputs[0].type, tuple(inputs[0].shape)) !=
                ("observation", "tensor(float)", ("N", 4, 84, 84))
            or (outputs[0].name, outputs[0].type, tuple(outputs[0].shape)) !=
                ("q_values", "tensor(float)", ("N", 4))):
        raise ValueError("ONNX input/output contract mismatch")
    runtime = {"onnxruntime_version": str(ort.__version__), "available_providers": ort.get_available_providers(),
               "active_providers": list(providers), "actual_provider": providers[0],
               "intra_op_num_threads": 1, "inter_op_num_threads": 1}
    if runtime.get("actual_provider") != "CPUExecutionProvider":
        raise RuntimeError(f"unexpected runtime/provider: {runtime}")
    rows = []
    errors = []
    for indexed, logged in pairs:
        raw, stack_sha = verify_stack_row(ISSUE50_DIR, indexed, stack_streams)
        stack = np.frombuffer(raw, dtype=np.uint8).reshape(4, 84, 84)
        model_input = prepare_model_input(stack)
        replayed = np.asarray(session.run(["q_values"], {"observation": model_input})[0])[0]
        if replayed.dtype != np.float32 or replayed.shape != (4,) or not np.isfinite(replayed).all():
            raise ValueError(f"invalid replay Q vector for {observation_key(indexed)}")
        logged_q = np.asarray(logged["q_values"], dtype=np.float32)
        err = max_q_error(logged_q, replayed)
        errors.append(err)
        rows.append({"key": {field: indexed[field] for field in KEY_FIELDS},
                     "stack_sha256": stack_sha, "logged_q": [float(x) for x in logged_q],
                     "replayed_q": [float(x) for x in replayed], "max_abs_error": err})
        if time.perf_counter() > deadline:
            raise TimeoutError("combined validation, replay, and finalization time budget exhausted")
    aggregate = max(errors)
    diagnostic = classify_diagnostic(True, aggregate)
    elapsed = time.perf_counter() - started
    report = {
        "issue": 54, "diagnostic": diagnostic, "max_abs_error": aggregate,
        "observation_count": len(rows), "metric": "max_i max_a |Q_replayed[i,a] - Q_logged[i,a]|",
        "integrity_passed": True, "has_verified_clear": "NO", "phase1": "INCONCLUSIVE",
        "round_status": "CONTINUE_RESEARCH", "phase2_transition": False,
        "controller_or_champion_implication": False,
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                         capture_output=True, text=True).stdout.strip(),
        "research_base_commit": BASE_COMMIT, "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
        "replay_source_digest": replay_source_digest(),
        "logged_q_baseline": {"source": "Issue #50 trajectory.jsonl", "matched_observations": len(rows),
                              "q_values_per_observation": 4},
        "issue52_context": "Issue #52 fresh-range probe failed before environment creation/collection (N=0); no fresh-input evidence.",
        "scope_limit": "Only the 300 selected Issue #50 life-window stacks are tested; no HVC inference.",
        "recommended_next_decision": "Use this result only to decide whether replay implementation fidelity merits further investigation; do not infer controller/champion or Phase 2 status.",
        "spec_lineage_note": "Metadata names historical spec SHA 68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec; pinned current spec only changed the embedded Contract v2 digest; preprocessing, inputs, outputs and action semantics remain unchanged.",
        "runtime": runtime, "python_version": platform.python_version(), "numpy_version": np.__version__,
        "input_hashes": input_hashes, "pre_run_validation_seconds": PRE_RUN_VALIDATION_SECONDS,
        "pre_run_validation": PRE_RUN_VALIDATION,
        "wall_seconds_before_report_write": elapsed,
    }
    output_dir.mkdir(parents=True)
    _atomic_json(output_dir / "q_comparisons.json", rows)
    _atomic_json(output_dir / "results.json", report)
    (output_dir / "report.md").write_text(
        f"# Issue #54 Offline Q Replay Consistency\n\n**{diagnostic}** — replay consistency only.\n\n"
        f"Maximum absolute Q replay error: `{aggregate:.12g}` over {len(rows)} exact indexed observations.\n\n"
        "Has Verified Clear: **NO**. Phase 1: **INCONCLUSIVE**. Round status: `CONTINUE_RESEARCH`. No Phase 2 transition.\n")
    hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    source_commit = report["source_commit"]
    manifest = {"issue": 54, "source_commit": source_commit, "research_base_commit": BASE_COMMIT,
                "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
                "replay_source_digest": replay_source_digest(),
                "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
                "pre_run_validation_seconds": PRE_RUN_VALIDATION_SECONDS,
                "output_hashes": hashes, "formal_command": "timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.analysis.run_issue54_q_replay_consistency --output-dir research/issue-54-q-replay-consistency-artifacts",
                "combined_wall_limit_seconds": 20.0}
    _atomic_json(output_dir / "manifest.json", manifest)
    # The running guard includes report/results/manifest preparation and refuses a late claim.
    total = time.perf_counter() - started
    combined = PRE_RUN_VALIDATION_SECONDS + total
    if combined > 20.0 or time.perf_counter() > deadline:
        raise TimeoutError(f"combined validation, replay, and finalization exceeded 20 seconds: {combined:.3f}")
    return {**report, "status": "completed", "wall_seconds": total,
            "combined_pre_run_and_formal_seconds": combined,
            "output_hashes": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()}}


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.output_dir, preflight_only=args.preflight_only)
    print(json.dumps(result, sort_keys=True))
    return 0
