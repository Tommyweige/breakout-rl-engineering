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
FORMAL_COMMAND = "timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.analysis.run_issue54_q_replay_consistency --output-dir research/issue-54-q-replay-consistency-artifacts"
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
    "scripts/analysis/attach_issue54_capture.py",
    "scripts/analysis/run_issue54_once.py",
    "tests/test_issue54_q_replay_consistency.py",
    "breakout_rl/issue50_pixel_paddle_alignment.py",
)
PRE_RUN_VALIDATION = {
    "focused_test_command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue54_q_replay_consistency -v",
    "focused_test_count": 9,
    "focused_test_status": "passed",
    "focused_test_wall_seconds": 0.13999250099004712,
    "compile_command": "python -m py_compile breakout_rl/issue54_q_replay_consistency.py scripts/analysis/run_issue54_q_replay_consistency.py scripts/analysis/attach_issue54_capture.py scripts/analysis/run_issue54_once.py tests/test_issue54_q_replay_consistency.py",
    "compile_status": "passed",
    "compile_wall_seconds": 0.03140434200759046,
    "diff_check_command": "git show --check --oneline HEAD",
    "diff_check_status": "passed",
    "diff_check_wall_seconds": 0.002714288988499902,
    "preflight_command": "timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.analysis.run_issue54_q_replay_consistency --output-dir /tmp/issue54-preflight --preflight-only",
    "preflight_status": "passed",
    "preflight_wall_seconds": 0.27102403600292746,
    "preflight_matched_rows": 300,
    "preflight_replay_started": False,
    "preflight_native_frames": 0,
}
PRE_RUN_VALIDATION_SECONDS = 1.0  # conservative internal reservation; supervisor uses final measured validation JSON
FINALIZATION_MARGIN_SECONDS = 1.5
SUPERVISOR_STARTUP_ALLOWANCE_SECONDS = 0.25
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
    if filename.is_absolute() or filename.parts != ("decision_stacks", filename.name):
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


def create_cpu_session(ort: Any, model_path: Path) -> Any:
    """Use Issue #50 CpuOnnxPolicy's default SessionOptions and CPU provider."""
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def max_q_error(logged: Any, replayed: Any) -> float:
    a = np.asarray(logged, dtype=np.float32)
    b = np.asarray(replayed, dtype=np.float32)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("logged and replayed Q vectors must each be four finite float32 values")
    return float(np.max(np.abs(b - a)))


def aggregate_q_error(errors: Any) -> float:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size != 300 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("aggregate Q errors must contain 300 finite nonnegative per-row values")
    return float(np.max(values))


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
    work_deadline = deadline - FINALIZATION_MARGIN_SECONDS
    if work_deadline <= started:
        raise TimeoutError("pre-run validation left no time for replay and finalization")
    if not repo_clean():
        raise RuntimeError("clean-source check failed; output directory must not yet exist")
    integrity_passed = False
    failure_reason = None
    try:
        pairs, input_hashes, stack_streams = verify_frozen_inputs()
        integrity_passed = True
    except Exception as exc:
        pairs, input_hashes, stack_streams = [], {}, {}
        failure_reason = f"input integrity unavailable or invalid: {type(exc).__name__}: {exc}"
    if preflight_only:
        return {"status": "preflight_passed" if integrity_passed else "INCONCLUSIVE",
                "matched_rows": len(pairs), "input_hashes": input_hashes,
                "wall_seconds": time.perf_counter() - started, "replay_started": False,
                "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE", "native_frames": 0,
                "failure_reason": failure_reason}
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    # Hash/row validation is repeated before the first model inference.
    rows = []
    errors = []
    runtime = {}
    try:
      if integrity_passed:
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
        session = create_cpu_session(ort, ROOT / "web/public/models/final_model/model.onnx")
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
                   "session_options": "ONNX Runtime defaults; matches Issue #50 CpuOnnxPolicy"}
        if runtime.get("actual_provider") != "CPUExecutionProvider":
            raise RuntimeError(f"unexpected runtime/provider: {runtime}")
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
            if time.perf_counter() > work_deadline:
                raise TimeoutError("combined validation, replay, and finalization time budget exhausted")
    except Exception as exc:
        integrity_passed = False
        failure_reason = f"runtime or output unavailable or invalid: {type(exc).__name__}: {exc}"
    aggregate = aggregate_q_error(errors) if integrity_passed and len(errors) == 300 else None
    diagnostic = classify_diagnostic(integrity_passed, aggregate)
    if diagnostic == "PROMOTED":
        failure_analysis = f"All integrity gates passed; max_abs_error {aggregate:.12g} is at or below 1e-6."
    elif diagnostic == "REJECTED":
        failure_analysis = f"All integrity gates passed; max_abs_error {aggregate:.12g} exceeds 1e-6."
    else:
        failure_analysis = failure_reason or "A required input, runtime, or output failed an integrity check."
    elapsed = time.perf_counter() - started
    report = {
        "issue": 54, "diagnostic": diagnostic, "max_abs_error": aggregate,
        "diagnostic_hypothesis": "Exact Issue #50 uint8 stacks normalized once with Contract v2 reproduce logged ONNX Q vectors within 1e-6.",
        "failure_analysis": failure_analysis,
        "observation_count": len(rows), "metric": "max_i max_a |Q_replayed[i,a] - Q_logged[i,a]|",
        "integrity_passed": integrity_passed and len(rows) == 300, "matched_observation_count": len(pairs), "has_verified_clear": "NO", "phase1": "INCONCLUSIVE",
        "round_status": "CONTINUE_RESEARCH", "phase2_transition": False,
        "controller_or_champion_implication": False,
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                         capture_output=True, text=True).stdout.strip(),
        "experiment_branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                              check=True, capture_output=True, text=True).stdout.strip(),
        "research_base_commit": BASE_COMMIT, "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
        "replay_source_digest": replay_source_digest(),
        "logged_q_baseline": {"source": "Issue #50 trajectory.jsonl", "matched_observations": len(pairs),
                              "q_values_per_observation": 4},
        "issue52_context": "Issue #52 fresh-range probe failed before environment creation/collection (N=0); no fresh-input evidence.",
        "scope_limit": "Only the 300 selected Issue #50 life-window stacks are tested; no HVC inference.",
        "recommended_next_decision": "Use this result only to decide whether replay implementation fidelity merits further investigation; do not infer controller/champion or Phase 2 status.",
        "spec_lineage_note": "Metadata names historical spec SHA 68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec; pinned current spec only changed the embedded Contract v2 digest; preprocessing, inputs, outputs and action semantics remain unchanged.",
        "runtime_comparability": "CPUExecutionProvider with ONNX Runtime default SessionOptions, matching Issue #50 CpuOnnxPolicy.",
        "runtime": runtime, "python_version": platform.python_version(), "numpy_version": np.__version__,
        "formal_command": FORMAL_COMMAND,
        "input_hashes": input_hashes, "pre_run_validation_seconds": PRE_RUN_VALIDATION_SECONDS,
        "finalization_margin_seconds": FINALIZATION_MARGIN_SECONDS,
        "pre_run_validation": PRE_RUN_VALIDATION,
        "wall_seconds_before_report_write": elapsed,
    }
    aggregate_display = "unavailable" if aggregate is None else f"{aggregate:.12g}"
    output_dir.mkdir(parents=True)
    _atomic_json(output_dir / "q_comparisons.json", rows)
    _atomic_json(output_dir / "results.json", report)
    (output_dir / "report.md").write_text(
        f"# Issue #54 Offline Q Replay Consistency\n\n**{diagnostic}** — replay consistency diagnostic only.\n\n"
        "## Hypothesis and result\n\n"
        "The exact Issue #50 uint8 observation stacks, normalized once with frozen Contract v2 preprocessing, reproduce the logged ONNX Q vectors on CPU ONNX Runtime 1.22.1 within a maximum absolute error of `1e-6`.\n\n"
        f"The sole primary metric was `max_i max_a |Q_replayed[i,a] - Q_logged[i,a]|`: **`{aggregate_display}`** across **{len(rows)}** replayed observations ({len(pairs)} exact input joins). {failure_analysis}\n\n"
        "## Integrity and provenance\n\n"
        f"{'All pinned input hashes passed. The exact Issue #50 index, trajectory, and six stack streams were checked; 300 rows joined uniquely on seed, arm, episode, step, and emulator frame, with matching observation hashes and four finite logged Q values each.' if integrity_passed else 'Integrity validation did not pass; no valid Q discrepancy is claimed. See `results.json` for the failure reason.'} Experiment branch: `{report['experiment_branch']}`; source commit: `{report['source_commit']}`; research base: `{BASE_COMMIT}`; Issue #50 source digest: `{ISSUE50_SOURCE_DIGEST}`. Input hashes are recorded in `results.json`.\n\n"
        "The metadata retains historical inference-spec SHA `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`; the pinned current spec only updates the embedded Contract v2 digest, with model input, preprocessing, output, and action semantics unchanged.\n\n"
        "## Limits and decision\n\n"
        "Issue #52's fresh-range probe failed before environment creation or collection (`N=0`), so it provides no fresh-input evidence. This replay covers only the selected 300 Issue #50 life-window stacks. It does not test a controller or establish object identity or a Breakout clear.\n\n"
        "Has Verified Clear: **NO**. Phase 1: **INCONCLUSIVE**. Round status: `CONTINUE_RESEARCH`. No champion promotion or Phase 2 transition is implied. Continue research based only on replay consistency; do not infer controller performance from this audit.\n\n"
        "`MODEL_ROUTING_VERIFICATION: UNAVAILABLE`\n\n"
        "## Pre-run validation\n\n"
        f"Focused tests ({PRE_RUN_VALIDATION['focused_test_count']}): `{PRE_RUN_VALIDATION['focused_test_command']}` — passed in {PRE_RUN_VALIDATION['focused_test_wall_seconds']:.3f}s.\n\n"
        f"Compile: `{PRE_RUN_VALIDATION['compile_command']}` — passed in {PRE_RUN_VALIDATION['compile_wall_seconds']:.3f}s.\n\n"
        f"Diff check: `{PRE_RUN_VALIDATION['diff_check_command']}` — passed in {PRE_RUN_VALIDATION['diff_check_wall_seconds']:.3f}s.\n\n"
        f"Zero-inference preflight: `{PRE_RUN_VALIDATION['preflight_command']}` — passed; 300 rows; {PRE_RUN_VALIDATION['preflight_wall_seconds']:.3f}s.\n\n"
        f"Combined pre-run validation: {PRE_RUN_VALIDATION_SECONDS:.3f}s. Reserve {FINALIZATION_MARGIN_SECONDS:.1f}s for stdout serialization and post-run log/hash attachment. Formal command (run once): `{FORMAL_COMMAND}`.\n")
    hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    source_commit = report["source_commit"]
    manifest = {"issue": 54, "source_commit": source_commit,
                "experiment_branch": report["experiment_branch"], "research_base_commit": BASE_COMMIT,
                "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
                "replay_source_digest": replay_source_digest(),
                "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
                "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
                "formal_capture_protocol": {"stdout_path": "/tmp/issue54-formal.stdout",
                                             "stderr_path": "/tmp/issue54-formal.stderr",
                                             "hashes_attached_after_command_exit": True},
                "input_hashes": input_hashes,
                "pre_run_validation": PRE_RUN_VALIDATION,
                "pre_run_validation_seconds": PRE_RUN_VALIDATION_SECONDS,
                "finalization_margin_seconds": FINALIZATION_MARGIN_SECONDS,
                "output_hashes": hashes, "formal_command": FORMAL_COMMAND,
                "combined_wall_limit_seconds": 20.0}
    _atomic_json(output_dir / "manifest.json", manifest)
    # The running guard includes report/results/manifest preparation and refuses a late claim.
    total = time.perf_counter() - started
    combined = PRE_RUN_VALIDATION_SECONDS + total
    if combined + FINALIZATION_MARGIN_SECONDS > 20.0 or time.perf_counter() > work_deadline:
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


def attach_log_capture_hashes(output_dir: Path, stdout_path: Path, stderr_path: Path,
                              validation_record_path: Path | None = None,
                              formal_command_wall_seconds: float | None = None,
                              failure_bundle_wall_seconds: float = 0.0,
                              formal_cli_timeout_seconds: float | None = None,
                              formal_cli_timed_out: bool = False,
                              supervisor_startup_allowance_seconds: float = SUPERVISOR_STARTUP_ALLOWANCE_SECONDS) -> dict[str, Any]:
    """Attach hashes of the one formal command's externally captured streams."""
    attachment_started = time.perf_counter()
    stdout_artifact = output_dir / "formal.stdout"
    stderr_artifact = output_dir / "formal.stderr"
    stdout_artifact.write_bytes(stdout_path.read_bytes())
    stderr_artifact.write_bytes(stderr_path.read_bytes())
    captures = {
        "stdout_path": str(stdout_path), "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path), "stderr_sha256": sha256_file(stderr_path),
        "stdout_artifact": stdout_artifact.name, "stderr_artifact": stderr_artifact.name,
    }
    results_path, report_path, manifest_path = (output_dir / name for name in
                                                ("results.json", "report.md", "manifest.json"))
    results = json.loads(results_path.read_text())
    results["formal_capture_hashes"] = captures
    validation = None
    if validation_record_path is not None:
        validation_target = output_dir / "pre_run_validation.json"
        validation_target.write_bytes(validation_record_path.read_bytes())
        validation = json.loads(validation_target.read_text())
        results["pre_run_validation"] = validation
        results["pre_run_validation_seconds"] = validation["combined_validation_wall_seconds"]
    if formal_command_wall_seconds is not None:
        results["formal_command_wall_seconds"] = formal_command_wall_seconds
        results["failure_bundle_wall_seconds"] = failure_bundle_wall_seconds
        results["formal_cli_timeout_seconds"] = formal_cli_timeout_seconds
        results["formal_cli_timed_out"] = formal_cli_timed_out
        results["supervisor_startup_allowance_seconds"] = supervisor_startup_allowance_seconds
        if validation is not None:
            results["combined_pre_run_and_formal_seconds"] = (
                validation["combined_validation_wall_seconds"] + formal_command_wall_seconds + failure_bundle_wall_seconds
            )
    _atomic_json(results_path, results)
    report = report_path.read_text().rstrip()
    if validation is not None:
        report = report.partition("\n## Pre-run validation\n")[0]
        validation_lines = ["## Pre-run validation", ""]
        for name in ("focused_test", "compile", "diff_check", "preflight"):
            entry = validation[name]
            status = entry.get("status", entry.get("result", "passed"))
            count = f" ({entry['count']} tests)" if name == "focused_test" else ""
            validation_lines.append(f"{name.replace('_', ' ')}{count}: `{entry['command']}` — {status}, {entry['wall_seconds']:.6f}s.")
            validation_lines.append("")
        validation_lines.append(f"Combined measured validation: {validation['combined_validation_wall_seconds']:.6f}s.")
        if formal_command_wall_seconds is not None:
            validation_lines.append(f"Formal command wall: {formal_command_wall_seconds:.6f}s; failure-bundle finalization: {failure_bundle_wall_seconds:.6f}s.")
            validation_lines.append(f"The outer timeout is capped at remaining budget minus a {supervisor_startup_allowance_seconds:.2f}s startup allowance; it includes guarded-run startup and final metadata writes. The reported measured-component total excludes pre-launch supervisor startup and final metadata refresh.")
        report += "\n\n" + "\n".join(validation_lines)
    report = report.partition("\n## Formal command capture\n")[0]
    report += ("\n\n## Formal command capture\n\n"
               f"Stdout: `{captures['stdout_path']}` (preserved as `{stdout_artifact.name}`) SHA-256 `{captures['stdout_sha256']}`.\n\n"
               f"Stderr: `{captures['stderr_path']}` (preserved as `{stderr_artifact.name}`) SHA-256 `{captures['stderr_sha256']}`.\n")
    report_path.write_text(report)
    manifest = json.loads(manifest_path.read_text())
    manifest["formal_capture_hashes"] = captures
    manifest["output_hashes"] = {
        name: sha256_file(output_dir / name)
        for name in ("q_comparisons.json", "results.json", "report.md", "formal.stdout", "formal.stderr")
    }
    if validation is not None:
        manifest["pre_run_validation"] = validation
        manifest["pre_run_validation_hash"] = sha256_file(output_dir / "pre_run_validation.json")
        manifest["output_hashes"]["pre_run_validation.json"] = manifest["pre_run_validation_hash"]
    if formal_command_wall_seconds is not None:
        manifest["formal_command_wall_seconds"] = formal_command_wall_seconds
        manifest["failure_bundle_wall_seconds"] = failure_bundle_wall_seconds
        manifest["formal_cli_timeout_seconds"] = formal_cli_timeout_seconds
        manifest["formal_cli_timed_out"] = formal_cli_timed_out
        manifest["supervisor_startup_allowance_seconds"] = supervisor_startup_allowance_seconds
        if validation is not None:
            manifest["combined_pre_run_and_formal_seconds"] = (
                validation["combined_validation_wall_seconds"] + formal_command_wall_seconds + failure_bundle_wall_seconds
            )
    manifest["manifest_self_hash_excluded"] = True
    _atomic_json(manifest_path, manifest)
    attachment_wall = time.perf_counter() - attachment_started
    captures["post_run_capture_attachment_wall_seconds"] = attachment_wall
    if validation is not None and formal_command_wall_seconds is not None:
        results["post_run_capture_attachment_wall_seconds"] = attachment_wall
        results["combined_pre_run_and_formal_seconds"] = (
            validation["combined_validation_wall_seconds"] + formal_command_wall_seconds + failure_bundle_wall_seconds + attachment_wall
        )
        results["measured_combined_elapsed_scope"] = (
            "Pre-run validation + frozen CLI + failure-bundle finalization + measured capture/hash attachment; "
            "excludes pre-launch supervisor startup and final metadata refresh. The outer timeout is reduced by a 0.25s startup allowance and enforces the inclusive 20-second cap."
        )
        _atomic_json(results_path, results)
        report = report_path.read_text().rstrip()
        report += (f"\n\nPost-run capture/hash attachment: {attachment_wall:.6f}s. Combined measured-component elapsed (excluding supervisor startup and final metadata refresh): "
                   f"{results['combined_pre_run_and_formal_seconds']:.6f}s (including failure-bundle and capture/hash finalization).\n")
        report_path.write_text(report)
        manifest = json.loads(manifest_path.read_text())
        manifest["post_run_capture_attachment_wall_seconds"] = attachment_wall
        manifest["combined_pre_run_and_formal_seconds"] = results["combined_pre_run_and_formal_seconds"]
        manifest["measured_combined_elapsed_scope"] = results["measured_combined_elapsed_scope"]
        for name in ("q_comparisons.json", "results.json", "report.md", "formal.stdout", "formal.stderr", "pre_run_validation.json"):
            if (output_dir / name).exists():
                manifest["output_hashes"][name] = sha256_file(output_dir / name)
        _atomic_json(manifest_path, manifest)
    captures["manifest_sha256"] = sha256_file(manifest_path)
    return captures
