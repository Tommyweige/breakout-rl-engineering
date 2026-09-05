"""Generate the source-backed Day 22 ONNX export report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_METADATA = Path("assets/day22/models/final_model/model.onnx.metadata.json")
DEFAULT_OUTPUT = Path("reports/day22-onnx-export.md")


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


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unavailable"
    return f"{int(value):,} bytes ({int(value) / (1024 * 1024):.2f} MiB)"


def _shape(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "?"
    return "[" + ", ".join(str(item) for item in value) + "]"


def _inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_day22_report(metadata: Mapping[str, Any]) -> str:
    """Render a concise report from export metadata and graph inspection data."""

    if not isinstance(metadata, Mapping):
        raise TypeError("Day 22 metadata must be a mapping")
    graph_validation = metadata.get("host_graph_validation")
    source_validation = metadata.get("source_model_validation")
    export = metadata.get("onnx_export")
    dynamic_batch = metadata.get("dynamic_batch_validation")
    source_model = metadata.get("source_model")
    graph = (
        graph_validation.get("graph") if isinstance(graph_validation, Mapping) else None
    )
    runtime = (
        source_validation.get("runtime")
        if isinstance(source_validation, Mapping)
        else None
    )
    if not isinstance(graph, Mapping):
        raise ValueError("Day 22 metadata is missing host graph inspection")
    if not isinstance(source_validation, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("Day 22 metadata is missing CUDA source validation")
    if not isinstance(export, Mapping) or not isinstance(dynamic_batch, Mapping):
        raise ValueError("Day 22 metadata is missing export contract metadata")
    inputs = graph.get("inputs")
    outputs = graph.get("outputs")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not inputs
        or not outputs
    ):
        raise ValueError("Day 22 graph metadata has no input/output contract")
    input_info = inputs[0]
    output_info = outputs[0]
    if not isinstance(input_info, Mapping) or not isinstance(output_info, Mapping):
        raise ValueError("Day 22 graph input/output metadata is malformed")
    operator_counts = graph.get("operator_type_counts")
    if not isinstance(operator_counts, Mapping):
        raise ValueError("Day 22 graph metadata has no operator counts")

    source_label = (
        f"{source_model.get('algorithm')} / {source_model.get('architecture')}"
        if isinstance(source_model, Mapping)
        else "unavailable"
    )
    pytorch_reference = metadata.get("pytorch_reference")
    inference_spec = metadata.get("inference_spec")
    probe_states = metadata.get("probe_states")
    return (
        "\n".join(
            [
                "# Day 22 ONNX export report",
                "",
                "這份 report 將兩條驗證路徑分開記錄：正式 deployment candidate 的 PyTorch source-model reference 在 CUDA 上產生；ONNX graph 的 checker 與 shape inspection 在 host CPU 上完成。後者不會取代前者，也尚未執行 ONNX Runtime policy parity。",
                "",
                "## Frozen source and contract",
                "",
                f"- status: `completed`",
                f"- source model: `{source_label}`",
                f"- source model SHA256: `{metadata.get('source_model_sha256')}`",
                f"- source checkpoint SHA256: `{metadata.get('source_checkpoint_sha256')}`",
                f"- source checkpoint step: `{metadata.get('source_checkpoint_step')}`",
                f"- inference spec: `{inference_spec.get('path') if isinstance(inference_spec, Mapping) else 'unavailable'}`",
                f"- probe states: `{probe_states.get('path') if isinstance(probe_states, Mapping) else 'unavailable'}` ({probe_states.get('count') if isinstance(probe_states, Mapping) else '?' } states)",
                "",
                "## CUDA source-model validation",
                "",
                f"- framework: `{runtime.get('framework')}` `{runtime.get('pytorch_version')}`",
                f"- device: `{runtime.get('device')}` / `{runtime.get('gpu_model')}`",
                f"- CUDA: `{runtime.get('cuda_version')}`, device index `{runtime.get('cuda_device_index')}`, available `{runtime.get('cuda_available')}`",
                f"- probe batch shape: `{_shape(source_validation.get('before_export', {}).get('input_shape'))}`",
                f"- Q-value output shape: `{_shape(source_validation.get('before_export', {}).get('q_values_shape'))}`",
                f"- source output max absolute difference before vs after export: `{source_validation.get('max_abs_diff_before_vs_after')}`",
                f"- greedy actions unchanged: `{source_validation.get('actions_match')}`",
                "",
                "這裡的 `0.0` 只表示匯出前後仍使用同一份 CUDA 載入的 source model 時，reference 沒有被流程改寫；它不是 ONNX Runtime parity 結果。",
                "",
                "## Host-side ONNX graph validation",
                "",
                f"- checker: `{graph_validation.get('checker')}` → `{graph_validation.get('checker_passed')}`",
                f"- input: `{input_info.get('name')}` / `{input_info.get('dtype')}` / `{_shape(input_info.get('shape'))}`",
                f"- output: `{output_info.get('name')}` / `{output_info.get('dtype')}` / `{_shape(output_info.get('shape'))}`",
                f"- opset: `{export.get('opset_version')}`; IR version: `{graph.get('ir_version')}`",
                f"- nodes: `{graph.get('node_count')}`; initializers: `{graph.get('initializer_count')}`",
                f"- operator counts: `{_inline(operator_counts)}`",
                f"- exported ONNX: `{metadata.get('model_path')}` ({_format_bytes(metadata.get('model_size_bytes'))})",
                "",
                "ONNX checker 驗證 graph 的格式與內部結構是否符合 ONNX 規則；它不能證明模型和 PyTorch 每個輸出相同，也不能證明瀏覽器 provider、外部 preprocessing 或遊戲表現正確。",
                "",
                "## Dynamic batch declaration",
                "",
                f"- dimension: `{dynamic_batch.get('dimension')}`",
                f"- declared input: `{_shape(dynamic_batch.get('declared_input_shape'))}`",
                f"- declared output: `{_shape(dynamic_batch.get('declared_output_shape'))}`",
                f"- source-model shape checks: `{dynamic_batch.get('tested_batch_sizes')}`",
                f"- ONNX Runtime execution: `{dynamic_batch.get('onnx_runtime_execution')}`",
                "",
                "## Artifacts",
                "",
                f"- ONNX model: `{metadata.get('model_path')}`",
                f"- ONNX metadata: `assets/day22/models/final_model/model.onnx.metadata.json`",
                f"- CUDA PyTorch reference: `{pytorch_reference.get('path') if isinstance(pytorch_reference, Mapping) else 'unavailable'}`",
                "- fixed probe states: `assets/day22/inference/probe_states.npz`",
                "- fixture manifest: `assets/day22/inference/metadata.json`",
                "- graph summary figure: `assets/day22/onnx-graph-summary.png`",
                "",
                "## Reproduction",
                "",
                "```powershell",
                "python -m scripts.analysis.generate_probe_states --contract configs/eval/breakout_contract_v2.json --output assets/day22/inference/probe_states.npz --states-per-seed 4 --stride 32 --max-steps 256",
                "python -m scripts.deployment.export_onnx",
                "python -m scripts.visualization.visualize_onnx_graph_summary",
                "python -m scripts.analysis.generate_day22_report",
                "```",
                "",
                "正式 export 要求 NVIDIA CUDA；CUDA unavailable 時 pipeline 應停止，而不是產生 CPU 偽 reference。",
            ]
        )
        + "\n"
    )


def generate_report(
    metadata_path: str | Path = DEFAULT_METADATA,
    output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    source = Path(metadata_path)
    destination = Path(output)
    metadata = _json_object(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_day22_report(metadata), encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Day 22 ONNX export report."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = generate_report(args.metadata, args.output)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 22 report generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"report": output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
