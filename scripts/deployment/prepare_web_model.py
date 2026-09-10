"""Prepare the verified Day 22 FP32 ONNX model and fixtures for Day 27 browser use."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_MODEL_METADATA = Path("assets/day22/models/final_model/model.onnx.metadata.json")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PARITY = Path("configs/inference/parity_validation.json")
DEFAULT_PROBES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_WEB_PUBLIC = Path("web/public")


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "inference" / "inference_spec.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def choose_array(
    archive: np.lib.npyio.NpzFile,
    names: tuple[str, ...],
    predicate: Callable[[np.ndarray], bool],
    *,
    description: str,
) -> np.ndarray:
    for name in names:
        if name in archive.files:
            array = np.asarray(archive[name])
            if predicate(array):
                return array
    matches = [np.asarray(archive[name]) for name in archive.files if predicate(np.asarray(archive[name]))]
    if len(matches) == 1:
        return matches[0]
    available = {name: list(np.asarray(archive[name]).shape) for name in archive.files}
    raise ValueError(f"unable to identify {description}; arrays={available}")


def prepare(root: Path, public_dir: Path) -> dict[str, Any]:
    model = root / DEFAULT_MODEL
    metadata_path = root / DEFAULT_MODEL_METADATA
    spec_path = root / DEFAULT_SPEC
    parity_path = root / DEFAULT_PARITY
    probes_path = root / DEFAULT_PROBES
    reference_path = root / DEFAULT_REFERENCE

    for path in (model, metadata_path, spec_path, parity_path, probes_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_metadata = load_json(metadata_path)
    spec = load_json(spec_path)
    parity = load_json(parity_path)

    observed_model_sha = sha256_file(model)
    declared_model_sha = model_metadata.get("model_sha256")
    if declared_model_sha != observed_model_sha:
        raise ValueError(
            f"Day 22 ONNX SHA mismatch: declared={declared_model_sha!r}, observed={observed_model_sha}"
        )

    with np.load(probes_path, allow_pickle=False) as archive:
        observations = choose_array(
            archive,
            ("observations", "probe_states", "states", "raw_observations"),
            lambda value: value.ndim == 4 and tuple(value.shape[1:]) == (4, 84, 84),
            description="probe observations",
        )

    if observations.dtype != np.uint8:
        raise ValueError(
            f"browser packaging expects raw uint8 probe observations, got {observations.dtype}"
        )
    observations = np.ascontiguousarray(observations)
    sample_count = int(observations.shape[0])

    with np.load(reference_path, allow_pickle=False) as archive:
        q_values = choose_array(
            archive,
            ("q_values", "reference_q_values", "outputs"),
            lambda value: value.ndim == 2 and value.shape == (sample_count, 4) and np.issubdtype(value.dtype, np.floating),
            description="reference Q-values",
        ).astype(np.float32, copy=False)
        greedy_actions = choose_array(
            archive,
            ("greedy_actions", "actions", "action_indices"),
            lambda value: value.ndim == 1 and value.shape == (sample_count,) and np.issubdtype(value.dtype, np.integer),
            description="reference greedy actions",
        ).astype(np.int64, copy=False)

    expected_actions = list(spec.get("actions", {}).get("meanings", []))
    if expected_actions != ["NOOP", "FIRE", "RIGHT", "LEFT"]:
        raise ValueError(f"unexpected action mapping: {expected_actions}")

    thresholds = parity.get("thresholds", {}).get("cpu")
    if not isinstance(thresholds, dict):
        raise ValueError("parity_validation.json is missing CPU thresholds")

    model_dir = public_dir / "models" / "final_model"
    fixture_dir = public_dir / "fixtures"
    model_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    packaged_model = model_dir / "model.onnx"
    packaged_metadata = model_dir / "model.onnx.metadata.json"
    packaged_spec = public_dir / "inference_spec.json"
    observation_output = fixture_dir / "day22-probe-observations.u8"
    reference_output = fixture_dir / "day22-reference.json"
    manifest_output = public_dir / "web-model-manifest.json"

    shutil.copy2(model, packaged_model)
    shutil.copy2(metadata_path, packaged_metadata)
    shutil.copy2(spec_path, packaged_spec)
    observation_output.write_bytes(observations.tobytes(order="C"))

    reference_payload = {
        "schema_version": 1,
        "artifact_type": "day27_browser_fixture_reference",
        "sample_count": sample_count,
        "observation_dtype": "uint8",
        "observation_shape": [4, 84, 84],
        "q_values": q_values.tolist(),
        "greedy_actions": greedy_actions.tolist(),
        "action_meanings": expected_actions,
        "thresholds": {
            "max_absolute_error": float(thresholds["max_absolute_error"]),
            "mean_absolute_error": float(thresholds["mean_absolute_error"]),
            "action_agreement_rate": float(thresholds["action_agreement_rate"]),
        },
        "source": {
            "probe_states": DEFAULT_PROBES.as_posix(),
            "probe_states_sha256": sha256_file(probes_path),
            "pytorch_reference": DEFAULT_REFERENCE.as_posix(),
            "pytorch_reference_sha256": sha256_file(reference_path),
        },
    }
    reference_output.write_text(
        json.dumps(reference_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "day27_web_model_manifest",
        "model": {
            "url": "/models/final_model/model.onnx",
            "source_path": DEFAULT_MODEL.as_posix(),
            "sha256": observed_model_sha,
            "size_bytes": packaged_model.stat().st_size,
        },
        "inference_spec": {
            "url": "/inference_spec.json",
            "source_path": DEFAULT_SPEC.as_posix(),
            "sha256": sha256_file(spec_path),
        },
        "fixtures": {
            "observations_url": "/fixtures/day22-probe-observations.u8",
            "reference_url": "/fixtures/day22-reference.json",
            "sample_count": sample_count,
            "observations_sha256": sha256_file(observation_output),
            "reference_sha256": sha256_file(reference_output),
        },
        "requested_backend": "wasm",
        "browser_runtime": "onnxruntime-web",
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if sha256_file(packaged_model) != observed_model_sha:
        raise RuntimeError("packaged ONNX hash changed during copy")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_WEB_PUBLIC)
    args = parser.parse_args()

    root = repository_root()
    public_dir = (root / args.public_dir).resolve() if not args.public_dir.is_absolute() else args.public_dir
    manifest = prepare(root, public_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
