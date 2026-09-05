"""Visualize the real Day 22 ONNX graph contract and operator inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.onnx_artifacts import inspect_onnx_model


DEFAULT_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_METADATA = Path("assets/day22/models/final_model/model.onnx.metadata.json")
DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_OUTPUT = Path("assets/day22/onnx-graph-summary.png")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _repository_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "eval" / "breakout_contract_v2.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate the repository root")


def _relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "visualization metadata only permits repository-relative paths: " f"{path}"
        ) from error


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unavailable"
    return f"{size_bytes / (1024 * 1024):.2f} MiB ({size_bytes:,} bytes)"


def _shape_text(shape: Sequence[Any]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def collect_graph_summary(
    model_path: str | Path,
    *,
    metadata_path: str | Path | None = DEFAULT_METADATA,
    source_model_path: str | Path | None = DEFAULT_SOURCE_MODEL,
    root: Path | None = None,
) -> dict[str, Any]:
    """Inspect ONNX itself and attach available export/file provenance."""

    repository_root = (root or _repository_root()).resolve()
    model = Path(model_path).resolve()
    graph = inspect_onnx_model(
        model,
        check=True,
        display_path=_relative_path(model, root=repository_root),
    )
    metadata: dict[str, Any] = {}
    metadata_file: Path | None = None
    if metadata_path is not None:
        metadata_file = Path(metadata_path).resolve()
        metadata = _json_object(metadata_file)
        expected_model_hash = metadata.get("model_sha256")
        observed_model_hash = _sha256_file(model)
        if (
            expected_model_hash is not None
            and expected_model_hash != observed_model_hash
        ):
            raise ValueError(
                "ONNX model hash does not match export metadata: "
                f"declared={expected_model_hash}, observed={observed_model_hash}"
            )
    source_model: Path | None = None
    source_model_size: int | None = None
    if source_model_path is not None:
        candidate = Path(source_model_path).resolve()
        if candidate.is_file():
            source_model = candidate
            source_model_size = candidate.stat().st_size

    result = {
        "technical_question": (
            "What input/output contract and serialized operator inventory did the "
            "real ONNX export produce?"
        ),
        "graph": graph,
        "files": {
            "onnx": {
                "path": _relative_path(model, root=repository_root),
                "sha256": _sha256_file(model),
                "size_bytes": model.stat().st_size,
            },
            "pytorch": (
                {
                    "path": _relative_path(source_model, root=repository_root),
                    "sha256": _sha256_file(source_model),
                    "size_bytes": source_model_size,
                }
                if source_model is not None
                else None
            ),
        },
        "export_metadata": (
            {
                "path": _relative_path(metadata_file, root=repository_root),
                "sha256": _sha256_file(metadata_file),
                "source_model_sha256": metadata.get("source_model_sha256"),
                "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
                "opset_version": (
                    metadata.get("onnx_export", {}).get("opset_version")
                    if isinstance(metadata.get("onnx_export"), Mapping)
                    else None
                ),
            }
            if metadata_file is not None
            else None
        ),
    }
    return result


def render_graph_summary(summary: Mapping[str, Any], output: str | Path) -> Path:
    """Render a readable evidence figure from an inspected graph summary."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = summary.get("graph")
    files = summary.get("files")
    if not isinstance(graph, Mapping) or not isinstance(files, Mapping):
        raise ValueError("graph summary is missing graph or file data")
    inputs = graph.get("inputs")
    outputs = graph.get("outputs")
    operator_counts = graph.get("operator_type_counts")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not isinstance(operator_counts, Mapping)
    ):
        raise ValueError("graph summary is missing ONNX input/output/operator data")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13.4, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.05, 1.45), hspace=0.28, wspace=0.25
    )
    figure.suptitle(
        "Day 22 — ONNX graph summary from the exported final policy",
        fontsize=17,
        fontweight="bold",
    )

    contract_axis = figure.add_subplot(grid[0, 0])
    contract_axis.axis("off")
    contract_axis.set_title("Serialized input/output contract", loc="left", fontsize=12)
    table_rows = []
    for label, values in (
        ("input", inputs[0] if inputs else {}),
        ("output", outputs[0] if outputs else {}),
    ):
        if not isinstance(values, Mapping):
            raise ValueError("ONNX value-info entry must be an object")
        row = [str(values.get("name", label)), str(values.get("dtype", "?")), ""]
        shape = values.get("shape", [])
        row[2] = _shape_text(shape if isinstance(shape, Sequence) else [])
        table_rows.append(row)
    contract_table = contract_axis.table(
        cellText=table_rows,
        colLabels=["name", "dtype", "shape"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=(0.28, 0.22, 0.42),
    )
    contract_table.auto_set_font_size(False)
    contract_table.set_fontsize(10)
    contract_table.scale(1.0, 1.8)
    contract_axis.text(
        0.0,
        0.13,
        "N is the dynamic batch dimension; C/H/W and action count are fixed.",
        transform=contract_axis.transAxes,
        fontsize=9.5,
        color="#444444",
    )

    facts_axis = figure.add_subplot(grid[0, 1])
    facts_axis.axis("off")
    facts_axis.set_title("What the graph contains", loc="left", fontsize=12)
    onnx_file = files.get("onnx")
    pytorch_file = files.get("pytorch")
    export_metadata = summary.get("export_metadata")
    opset = (
        export_metadata.get("opset_version")
        if isinstance(export_metadata, Mapping)
        else None
    )
    facts = (
        ("nodes", graph.get("node_count")),
        ("initializers", graph.get("initializer_count")),
        ("ONNX opset", opset if opset is not None else graph.get("opset_imports")),
        (
            "ONNX file",
            _format_size(
                onnx_file.get("size_bytes") if isinstance(onnx_file, Mapping) else None
            ),
        ),
        (
            "PyTorch .pt",
            _format_size(
                pytorch_file.get("size_bytes")
                if isinstance(pytorch_file, Mapping)
                else None
            ),
        ),
    )
    facts_text = "\n".join(f"{label:<14} {value}" for label, value in facts)
    facts_axis.text(
        0.02,
        0.88,
        facts_text,
        transform=facts_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        linespacing=1.7,
    )
    facts_axis.text(
        0.02,
        0.08,
        "Counts and sizes are read from the exported model and local artifacts.",
        transform=facts_axis.transAxes,
        fontsize=9.5,
        color="#444444",
    )

    operator_axis = figure.add_subplot(grid[1, :])
    labels = [str(label) for label in operator_counts.keys()]
    counts = [int(value) for value in operator_counts.values()]
    order = sorted(
        range(len(labels)), key=lambda index: (-counts[index], labels[index])
    )
    labels = [labels[index] for index in order]
    counts = [counts[index] for index in order]
    bars = operator_axis.barh(labels[::-1], counts[::-1], color="#2f6f9f")
    operator_axis.set_title(
        "Operator type counts in the actual ONNX graph", loc="left", fontsize=12
    )
    operator_axis.set_xlabel("node count")
    operator_axis.grid(axis="x", alpha=0.25)
    operator_axis.bar_label(bars, padding=3, fmt="%d")
    operator_axis.spines[["top", "right"]].set_visible(False)

    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)
    return output_path


def _write_sidecar(
    output: Path,
    *,
    summary: Mapping[str, Any],
    model_path: Path,
    metadata_path: Path | None,
    source_model_path: Path | None,
    root: Path,
) -> Path:
    sidecar = output.with_suffix(".json")
    payload = {
        "schema_version": 1,
        "artifact_type": "day22_onnx_graph_summary_visualization",
        "technical_question": summary["technical_question"],
        "output": _relative_path(output, root=root),
        "source_artifacts": {
            "onnx_model": _relative_path(model_path, root=root),
            "onnx_metadata": (
                _relative_path(metadata_path, root=root)
                if metadata_path is not None
                else None
            ),
            "pytorch_model": (
                _relative_path(source_model_path, root=root)
                if source_model_path is not None and source_model_path.is_file()
                else None
            ),
        },
        "command": (
            "python -m scripts.visualization.visualize_onnx_graph_summary "
            f"--model {_relative_path(model_path, root=root)}"
        ),
        "summary": dict(summary),
    }
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def render_from_files(
    *,
    model_path: str | Path = DEFAULT_MODEL,
    metadata_path: str | Path | None = DEFAULT_METADATA,
    source_model_path: str | Path | None = DEFAULT_SOURCE_MODEL,
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path]:
    """Inspect the model, render the PNG, and preserve machine-readable provenance."""

    root = _repository_root()
    model = Path(model_path).resolve()
    metadata = Path(metadata_path).resolve() if metadata_path is not None else None
    source_model = (
        Path(source_model_path).resolve() if source_model_path is not None else None
    )
    destination = Path(output).resolve()
    summary = collect_graph_summary(
        model,
        metadata_path=metadata,
        source_model_path=source_model,
        root=root,
    )
    figure = render_graph_summary(summary, destination)
    sidecar = _write_sidecar(
        figure,
        summary=summary,
        model_path=model,
        metadata_path=metadata,
        source_model_path=source_model,
        root=root,
    )
    return figure, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the actual Day 22 ONNX graph and export provenance."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        figure, sidecar = render_from_files(
            model_path=args.model,
            metadata_path=args.metadata,
            source_model_path=args.source_model,
            output=args.output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 22 ONNX graph visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"figure": figure.as_posix(), "metadata": sidecar.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
