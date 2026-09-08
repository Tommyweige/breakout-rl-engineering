"""Create a separate internal-FP16 ONNX artifact from the Day 22 FP32 graph."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.inference import load_inference_spec
from breakout_rl.onnx_artifacts import inspect_onnx_model


DEFAULT_CONFIG = Path("configs/inference/precision_validation.json")
DEFAULT_SOURCE = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_SOURCE_METADATA = Path(
    "assets/day22/models/final_model/model.onnx.metadata.json"
)
DEFAULT_OUTPUT = Path("assets/day25/models/model.fp16.onnx")
DEFAULT_OUTPUT_METADATA = Path(
    "assets/day25/models/model.fp16.onnx.metadata.json"
)


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _topologically_sort_graph(model: Any) -> None:
    """Repair converter-inserted Cast ordering before checker/runtime use."""

    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)
    pending = list(graph.node)
    ordered: list[Any] = []
    while pending:
        ready = [
            node
            for node in pending
            if all(not name or name in available for name in node.input)
        ]
        if not ready:
            unresolved = [node.name or node.op_type for node in pending]
            raise ValueError(
                "converted ONNX graph cannot be topologically sorted: "
                + ", ".join(unresolved)
            )
        for node in ready:
            pending.remove(node)
            ordered.append(node)
            available.update(name for name in node.output if name)
    del graph.node[:]
    graph.node.extend(ordered)


def _config_paths(config: Mapping[str, Any], *, root: Path) -> tuple[Path, Path]:
    fp16_artifact = config.get("fp16_artifact")
    lineage = config.get("lineage")
    if not isinstance(fp16_artifact, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("precision validation config is missing artifact or lineage")
    configured_source = lineage.get("onnx_model_path")
    configured_output = fp16_artifact.get("model_path")
    if not isinstance(configured_source, str) or not isinstance(configured_output, str):
        raise ValueError("precision validation config has invalid ONNX paths")
    return (root / configured_source).resolve(), (root / configured_output).resolve()


def _verify_source_lineage(
    *,
    root: Path,
    source: Path,
    source_metadata_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = config.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("precision validation config is missing lineage")
    source_metadata = _json_object(source_metadata_path)
    source_hash = sha256_file(source)
    if source_hash != lineage.get("onnx_model_sha256"):
        raise ValueError("Day 22 ONNX hash does not match precision config")
    if source_metadata.get("model_sha256") != source_hash:
        raise ValueError("Day 22 ONNX hash does not match its metadata")
    if source_metadata.get("source_model_sha256") != lineage.get(
        "canonical_source_model_sha256"
    ):
        raise ValueError("Day 22 ONNX is not derived from the Day 21 canonical model")
    if source_metadata.get("source_checkpoint_sha256") != lineage.get(
        "source_checkpoint_sha256"
    ):
        raise ValueError("Day 22 ONNX checkpoint provenance does not match Day 21")
    source_graph = inspect_onnx_model(source, check=True)
    if source_graph["inputs"][0]["dtype"] != "float32":
        raise ValueError("Day 22 ONNX input must remain float32")
    if source_graph["outputs"][0]["dtype"] != "float32":
        raise ValueError("Day 22 ONNX output must remain float32")
    spec_path = root / str(lineage.get("inference_spec_path"))
    spec = load_inference_spec(spec_path)
    input_info = source_graph["inputs"][0]
    output_info = source_graph["outputs"][0]
    if input_info["name"] != spec.input_name or output_info["name"] != spec.output_name:
        raise ValueError("Day 22 ONNX names do not match the inference contract")
    if sha256_file(spec_path, normalize_text=True) != lineage.get(
        "inference_spec_sha256"
    ):
        raise ValueError("inference spec hash does not match precision config")
    return {
        "onnx_model": {
            "path": _relative(root, source),
            "sha256": source_hash,
            "metadata_path": _relative(root, source_metadata_path),
            "metadata_sha256": sha256_file(source_metadata_path),
        },
        "source_model_sha256": lineage.get("canonical_source_model_sha256"),
        "source_checkpoint_sha256": lineage.get("source_checkpoint_sha256"),
        "source_checkpoint_step": lineage.get("source_checkpoint_step"),
        "inference_spec": {
            "path": _relative(root, spec_path),
            "sha256": lineage.get("inference_spec_sha256"),
            "contract_id": spec.contract_id,
        },
    }


def convert_onnx_model(
    source: str | Path,
    output: str | Path,
    *,
    keep_io_types: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Convert one ONNX graph while keeping the FP32 source untouched."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("FP16 output must be a separate artifact from the FP32 source")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"FP16 artifact already exists: {output_path}; use --force to replace it"
        )
    try:
        import onnx
        import onnxruntime
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as error:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "Day 25 FP16 conversion requires the pinned onnx and onnxruntime packages"
        ) from error

    source_model = onnx.load(str(source_path))
    converted = convert_float_to_float16(
        source_model,
        keep_io_types=keep_io_types,
    )
    _topologically_sort_graph(converted)
    onnx.checker.check_model(converted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(output_path))
    graph = inspect_onnx_model(output_path, check=True)
    return {
        "source_model_sha256": sha256_file(source_path),
        "model_sha256": sha256_file(output_path),
        "model_size_bytes": output_path.stat().st_size,
        "source_size_bytes": source_path.stat().st_size,
        "converter": {
            "package": "onnxruntime",
            "module": "onnxruntime.transformers.float16",
            "function": "convert_float_to_float16",
            "onnxruntime_version": str(onnxruntime.__version__),
            "onnx_version": str(onnx.__version__),
            "keep_io_types": bool(keep_io_types),
            "python_version": platform.python_version(),
        },
        "graph": graph,
    }


def convert_artifact(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    source: str | Path = DEFAULT_SOURCE,
    source_metadata: str | Path = DEFAULT_SOURCE_METADATA,
    output: str | Path = DEFAULT_OUTPUT,
    output_metadata: str | Path = DEFAULT_OUTPUT_METADATA,
    force: bool = False,
) -> dict[str, Any]:
    """Convert the configured Day 22 artifact and write auditable metadata."""

    root = repository_root()
    config_file = _resolve(root, Path(config_path))
    config = _json_object(config_file)
    if config.get("artifact_type") != "day25_precision_validation":
        raise ValueError("precision config has an unexpected artifact_type")
    configured_source, configured_output = _config_paths(config, root=root)
    source_path = _resolve(root, Path(source))
    source_metadata_path = _resolve(root, Path(source_metadata))
    output_path = (Path(output) if Path(output).is_absolute() else root / Path(output)).resolve()
    if source_path != configured_source:
        raise ValueError("conversion source must be the configured Day 22 ONNX artifact")
    if output_path != configured_output:
        raise ValueError("conversion output must be the configured Day 25 FP16 artifact")
    fp16_artifact = config.get("fp16_artifact")
    if not isinstance(fp16_artifact, Mapping):
        raise ValueError("precision config is missing fp16_artifact")
    keep_io_types = bool(fp16_artifact.get("conversion_keep_io_types"))
    lineage = _verify_source_lineage(
        root=root,
        source=source_path,
        source_metadata_path=source_metadata_path,
        config=config,
    )
    conversion = convert_onnx_model(
        source_path,
        output_path,
        keep_io_types=keep_io_types,
        force=force,
    )
    graph = conversion["graph"]
    expected_input_dtype = fp16_artifact.get("input_dtype")
    expected_output_dtype = fp16_artifact.get("output_dtype")
    if graph["inputs"][0]["dtype"] != expected_input_dtype:
        raise ValueError("converted FP16 input dtype does not match configured I/O policy")
    if graph["outputs"][0]["dtype"] != expected_output_dtype:
        raise ValueError("converted FP16 output dtype does not match configured I/O policy")
    metadata = {
        "schema_version": 1,
        "artifact_type": "day25_fp16_onnx_model",
        "model_path": _relative(root, output_path),
        "model_sha256": conversion["model_sha256"],
        "model_size_bytes": conversion["model_size_bytes"],
        "source": lineage,
        "precision": {
            "internal": fp16_artifact.get("internal_precision"),
            "input_dtype": graph["inputs"][0]["dtype"],
            "output_dtype": graph["outputs"][0]["dtype"],
            "io_policy": "float32 input / float32 output with internal float16 graph",
            "browser_baseline_unchanged": True,
        },
        "conversion": conversion["converter"],
        "graph": graph,
        "lineage_config": {
            "path": _relative(root, config_file),
            "sha256": sha256_file(config_file, normalize_text=True),
        },
    }
    metadata_path = (
        Path(output_metadata)
        if Path(output_metadata).is_absolute()
        else root / Path(output_metadata)
    ).resolve()
    if metadata_path.exists() and not force:
        raise FileExistsError(
            f"FP16 metadata already exists: {metadata_path}; use --force to replace it"
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "model": output_path,
        "metadata": metadata_path,
        "metadata_payload": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-metadata", type=Path, default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = convert_artifact(
            config_path=args.config,
            source=args.source,
            source_metadata=args.source_metadata,
            output=args.output,
            output_metadata=args.output_metadata,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Day 25 FP16 conversion failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model": str(result["model"]),
                "metadata": str(result["metadata"]),
                "model_sha256": result["metadata_payload"]["model_sha256"],
                "model_size_bytes": result["metadata_payload"]["model_size_bytes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
