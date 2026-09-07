"""Generate the source-backed Day 24 inference benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.artifacts import repository_relative_path, repository_root
from breakout_rl.benchmarking import load_benchmark_artifacts


DEFAULT_INPUT_ROOT = Path("assets/day24/benchmarks")
DEFAULT_OUTPUT = Path("reports/day24-inference-benchmark.md")


def _results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = payload.get("results")
    if not isinstance(values, list) or not all(isinstance(value, Mapping) for value in values):
        raise ValueError("benchmark summary has no valid results list")
    return list(values)


def _find(
    results: Sequence[Mapping[str, Any]],
    *,
    runtime: str,
    provider: str,
    scope: str,
    batch_size: int = 1,
    thread_setting: str = "default",
) -> Mapping[str, Any]:
    matches = [
        result
        for result in results
        if result.get("runtime") == runtime
        and result.get("requested_provider") == provider
        and result.get("scope") == scope
        and result.get("batch_size") == batch_size
        and result.get("thread_setting") == thread_setting
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one result for "
            f"{runtime}/{provider}/{scope}/batch{batch_size}/threads-{thread_setting}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _latency_row(result: Mapping[str, Any]) -> str:
    latency = result["latency"]
    return (
        f"| {result['runtime']} | {result['requested_provider'].upper()} | "
        f"{result['actual_provider']} | {result['scope']} | {result['batch_size']} | "
        f"{result['thread_setting']} | {result['sample_count']} | "
        f"{_number(latency['p50_ms'])} | {_number(latency['p95_ms'])} | "
        f"{_number(latency['mean_ms'])} | {_number(latency['std_ms'])} | "
        f"{_number(latency['throughput_per_second'])} |"
    )


def _breakdown_value(result: Mapping[str, Any], field: str, percentile: str) -> str:
    breakdown = result.get("breakdown")
    if not isinstance(breakdown, Mapping) or field not in breakdown:
        return "included in runtime API" if field == "output_materialization_ns" else "unavailable"
    value = breakdown[field]
    if not isinstance(value, Mapping):
        return "unavailable"
    return _number(value.get(percentile.replace("p", "p") + "_ms"))


def _breakdown_pair(result: Mapping[str, Any], field: str) -> str:
    if (
        field == "output_materialization_ns"
        and result.get("output_materialization_scope") != "separate"
    ):
        return "included in runtime API"
    return (
        f"{_breakdown_value(result, field, 'p50')}/"
        f"{_breakdown_value(result, field, 'p95')}"
    )


def _write_report(
    output: Path,
    *,
    payload: Mapping[str, Any],
    benchmark_dir: Path,
    root: Path,
) -> Path:
    results = _results(payload)
    primary = [
        _find(results, runtime=runtime, provider=provider, scope="end_to_end")
        for runtime, provider in (
            ("PyTorch", "cpu"),
            ("PyTorch", "cuda"),
            ("ONNX Runtime", "cpu"),
            ("ONNX Runtime", "cuda"),
        )
    ]
    primary_model_only = [
        _find(results, runtime=runtime, provider=provider, scope="model_only")
        for runtime, provider in (
            ("PyTorch", "cpu"),
            ("PyTorch", "cuda"),
            ("ONNX Runtime", "cpu"),
            ("ONNX Runtime", "cuda"),
        )
    ]
    configuration = payload.get("configuration", {})
    host = payload.get("host", {})
    lineage = payload.get("lineage", {})
    source_observations = payload.get("source_observations", {})
    decision_budget_ms = 4 * 1000.0 / 60.0

    lines = [
        "# Day 24｜一次 Batch=1 決策到底需要多久？",
        "",
        "Day 23 已經確認同一個 Day 21 canonical Final Model 在 PyTorch 與 ONNX Runtime 之間仍能做出相同決策。接下來才有資格問速度：未來網頁右側的 RL Agent 每拿到一個 state，從輸入到 action 的時間是否可能拖慢即時 gameplay？",
        "",
        "這份 report 只回答 native inference 的問題。它不包含 ALE `env.step()`、瀏覽器繪圖或 ONNX Runtime Web；那些是後續 Browser workload 需要重新量測的邊界。",
        "",
        f"- benchmark id: `{payload['benchmark_id']}`",
        f"- generated at: `{payload['generated_at_utc']}`",
        f"- warm-up iterations: `{configuration.get('warmup_iterations')}`",
        f"- measured iterations per result: `{configuration.get('iterations')}`",
        f"- batch sizes: `{configuration.get('batch_sizes')}`; primary: `1`",
        f"- model-only timing semantics: `{configuration.get('model_only_semantics')}`; end-to-end timing semantics: `{configuration.get('end_to_end_semantics')}`",
        f"- input fixture: `{source_observations.get('path')}` ({source_observations.get('count')} observations)",
        "",
        "## 先看產品真正關心的 end-to-end decision",
        "",
        "`end-to-end` 代表固定的 `uint8 (4,84,84)` observation 經過既有 inference adapter 的 normalization、tensor/provider transfer、model call、輸出取回與 `argmax`。下面的 P50 是一半樣本不超過的延遲；P95 則是把較慢的尾端也算進來，因此不能只看平均值或最快一次。",
        "",
        "| runtime | requested | actual provider | scope | batch | threads | samples | P50 (ms) | P95 (ms) | mean (ms) | std (ms) | decisions/s |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_latency_row(result) for result in primary)
    lines.extend(
        [
            "",
            "圖表把同一批 raw samples 重新畫成 P50/P95 與分布；它的用途是讓讀者看到 runtime/provider 的差異，以及 tail latency 是否被平均值藏起來。",
            "",
            "[![Day 24 batch=1 P50/P95 latency](../assets/day24/batch1-latency.png)](../assets/day24/batch1-latency.png)",
            "",
            "[![Day 24 batch=1 latency distributions](../assets/day24/latency-distribution.png)](../assets/day24/latency-distribution.png)",
            "",
            "## Model-only 和 end-to-end 為什麼要分開",
            "",
            "model-only 先在計時外準備並驗證好 `float32` model input，再只量 prevalidated runtime/model call；它回答的是神經網路與 runtime API 本身的成本。end-to-end 才包含一次右側 Agent decision 實際會付出的 input preparation、policy validation、transfer、輸出取回與 action selection。兩者不能拿不同 scope 的數字直接混成一個排行榜。",
            "",
            "| runtime | requested | actual provider | scope | batch | threads | samples | P50 (ms) | P95 (ms) | mean (ms) | std (ms) | decisions/s |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_latency_row(result) for result in primary_model_only)
    lines.extend(
        [
            "",
            "這個界線也解釋了為什麼 GPU model-only 很快，不代表完整 decision 一定同樣快：小模型的輸入搬移、同步與輸出 materialization 可能和真正的神經網路計算一樣重要。",
            "",
            "## 一次 decision 的時間拆解",
            "",
            "PyTorch CUDA 的 end-to-end path 另外保留四段 timing：input preparation（包含這個 contract adapter 的 normalization 與 host→device 路徑）、runtime inference、device→host output materialization，以及 `argmax`。每一段都在必要處同步 CUDA，避免只量到非同步 kernel launch。",
            "",
            "| runtime | provider | input prepare P50/P95 (ms) | runtime P50/P95 (ms) | output P50/P95 (ms) | argmax P50/P95 (ms) | output scope |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in primary:
        lines.append(
            f"| {result['runtime']} | {result['actual_provider']} | "
            f"{_breakdown_pair(result, 'input_prepare_ns')} | "
            f"{_breakdown_pair(result, 'runtime_inference_ns')} | "
            f"{_breakdown_pair(result, 'output_materialization_ns')} | "
            f"{_breakdown_pair(result, 'argmax_ns')} | "
            f"{result['output_materialization_scope']} |"
        )
    lines.extend(
        [
            "",
            "ONNX Runtime 的一般 NumPy API 會在 runtime call 中完成 provider-side transfer 與輸出回到 host，因此 output materialization 標示為 `included_in_runtime_api`，不是假裝它可以和 PyTorch 的獨立段落直接比較。",
            "",
            "## CPU thread sweep：default 不一定是答案",
            "",
            "CPU 另外記錄 `1 / 2 / 4 / default`。這裡的 thread count 是 runtime 的設定；PyTorch 使用 `torch.set_num_threads`，ONNX Runtime 使用 `SessionOptions.intra_op_num_threads`。因此表格可以回答「這個小模型是否因 thread oversubscription 反而變慢」，而不是預先假設 thread 越多越好。",
            "",
            "| runtime | actual provider | thread setting | actual configured threads | P50 (ms) | P95 (ms) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        if (
            result.get("scope") == "end_to_end"
            and result.get("batch_size") == 1
            and result.get("runtime") in {"PyTorch", "ONNX Runtime"}
            and result.get("requested_provider") == "cpu"
        ):
            lines.append(
                f"| {result['runtime']} | {result['actual_provider']} | "
                f"{result['thread_setting']} | {result['cpu_threads']} | "
                f"{_number(result['latency']['p50_ms'])} | {_number(result['latency']['p95_ms'])} |"
            )
    lines.extend(
        [
            "",
            "## 這些數字和 60 Hz gameplay budget 的關係",
            "",
            f"Contract v2 的 frame skip 是 4；若用約 60 Hz render 作直覺參考，一次 action 覆蓋四個畫面，名義上的四幀時間約為 `{decision_budget_ms:.3f} ms`。這不是硬編進測試的 unit-test gate，而是把 benchmark 放回產品問題的尺度：比較主要 end-to-end P95 是否明顯小於這個 budget。",
            "",
            "| runtime | actual provider | end-to-end P95 (ms) | nominal 4-frame budget (ms) | native inference is below budget? |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for result in primary:
        p95 = float(result["latency"]["p95_ms"])
        lines.append(
            f"| {result['runtime']} | {result['actual_provider']} | {_number(p95)} | "
            f"{decision_budget_ms:.3f} | `{p95 < decision_budget_ms}` |"
        )
    lines.extend(
        [
            "",
            "這個結論只表示 native model decision 在本次機器與 sample count 下沒有吃掉四幀 budget；它不能取代 Day 28/29 在 `*.pages.dev` 上對 ONNX Runtime Web WASM/WebGPU 的實測，也不能推出完整遊戲 loop 的 FPS。",
            "",
            "## Provenance and reproducibility",
            "",
            f"- Day 21 source model: `{lineage['canonical_source_model']['path']}`; SHA256 `{lineage['canonical_source_model']['sha256']}`",
            f"- Day 21 source checkpoint: SHA256 `{lineage['canonical_source_model']['source_checkpoint_sha256']}` at step `{lineage['canonical_source_model']['source_checkpoint_step']}`",
            f"- Day 22 ONNX: `{lineage['onnx_model']['path']}`; SHA256 `{lineage['onnx_model']['sha256']}`",
            f"- Day 22 inference spec: `{lineage['inference_spec']['path']}`; normalized SHA256 `{lineage['inference_spec']['sha256']}`",
            f"- Day 23 parity config: `{lineage['parity_validation']['path']}`",
            f"- actual providers and graph assignment are stored in each result metadata; host GPU: `{host.get('gpu_model')}`",
            f"- PyTorch CUDA peak allocator usage during benchmark: `{host.get('pytorch_cuda_memory', {}).get('peak_allocated_bytes')}` allocated bytes / `{host.get('pytorch_cuda_memory', {}).get('peak_reserved_bytes')}` reserved bytes",
            "",
            "Raw samples are stored separately from the summary so P50/P95 can be rebuilt without trusting a copied table:",
            "",
            f"- summary: `{repository_relative_path(benchmark_dir / 'summary.json', root=root)}`",
            f"- raw samples: `{repository_relative_path(benchmark_dir / 'raw-samples.json', root=root)}`",
            "",
            "The next question is not whether native CUDA is fast enough. It is whether the browser runtime, browser provider, rendering loop, and model decision can preserve the same budget under the actual deployment contract.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def generate_report(
    *,
    benchmark_id: str,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    root = repository_root()
    benchmark_dir = (root / input_root / benchmark_id).resolve()
    payload = load_benchmark_artifacts(benchmark_dir)
    if payload.get("benchmark_id") != benchmark_id:
        raise ValueError("benchmark directory and artifact benchmark_id do not match")
    return _write_report(
        (root / output).resolve(),
        payload=payload,
        benchmark_dir=benchmark_dir,
        root=root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = generate_report(
            benchmark_id=args.benchmark_id,
            input_root=args.input_root,
            output=args.output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 24 report generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
