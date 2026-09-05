"""Optional ONNX graph inspection shared by export reports and visualizations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _load_onnx():
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "ONNX inspection requires the pinned 'onnx' package; "
            "install the repository environment first"
        ) from error
    return onnx


def _dimension_value(dimension: Any) -> int | str:
    if dimension.HasField("dim_param"):
        return str(dimension.dim_param)
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    return "?"


def _value_info_summary(value_info: Any, onnx: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    onnx_dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    dtype_names = {
        "FLOAT": "float32",
        "FLOAT16": "float16",
        "DOUBLE": "float64",
        "INT64": "int64",
        "INT32": "int32",
        "UINT8": "uint8",
    }
    shape = (
        [_dimension_value(dimension) for dimension in tensor_type.shape.dim]
        if tensor_type.HasField("shape")
        else []
    )
    return {
        "name": str(value_info.name),
        "dtype": dtype_names.get(onnx_dtype, onnx_dtype.lower()),
        "onnx_dtype": onnx_dtype,
        "shape": shape,
    }


def inspect_onnx_model(
    path: str | Path,
    *,
    check: bool = False,
    display_path: str | None = None,
) -> dict[str, Any]:
    """Read graph structure and optionally run the host-side ONNX checker."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    onnx = _load_onnx()
    model = onnx.load(str(source), load_external_data=True)
    if check:
        onnx.checker.check_model(model)
    graph = model.graph
    operator_counts = Counter(str(node.op_type) for node in graph.node)
    opset_imports = {
        str(opset.domain) if opset.domain else "ai.onnx": int(opset.version)
        for opset in model.opset_import
    }
    return {
        "model_path": display_path or source.as_posix(),
        "ir_version": int(model.ir_version),
        "opset_imports": dict(sorted(opset_imports.items())),
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "operator_type_counts": dict(sorted(operator_counts.items())),
        "inputs": [_value_info_summary(value, onnx) for value in graph.input],
        "outputs": [_value_info_summary(value, onnx) for value in graph.output],
        "checker_passed": bool(check),
    }


__all__ = ["inspect_onnx_model"]
