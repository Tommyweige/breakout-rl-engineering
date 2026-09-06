"""Plot real Day 23 PyTorch-versus-ONNX Runtime parity measurements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.artifacts import (
    repository_relative_path as _relative_path,
    repository_root as _repository_root,
    sha256_file as _sha256_file,
)


DEFAULT_INPUT = Path("assets/day23/onnx-runtime-parity.json")
DEFAULT_OUTPUT = Path("assets/day23/pytorch-vs-onnx-parity.png")


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


def _provider_batch(provider: Mapping[str, Any]) -> Mapping[str, Any]:
    batches = provider.get("batches")
    if not isinstance(batches, Mapping) or not batches:
        raise ValueError("provider result has no batch measurements")
    value = batches.get("4") or next(iter(batches.values()))
    if not isinstance(value, Mapping):
        raise ValueError("provider batch measurement is malformed")
    return value


def _metric_array(metrics: Mapping[str, Any], key: str) -> np.ndarray:
    value = metrics.get(key)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"comparison metric {key} must be a finite one-dimensional array")
    return array


def render_parity_figure(result: Mapping[str, Any], output: str | Path) -> Path:
    """Render a figure whose marks are read from the saved comparison artifact."""

    if result.get("artifact_type") != "day23_onnx_runtime_parity_result":
        raise ValueError("comparison artifact is not a Day 23 parity result")
    providers = result.get("providers")
    if not isinstance(providers, Mapping) or not providers:
        raise ValueError("comparison artifact has no provider results")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"cpu": "#2f6f9f", "cuda": "#c55a11"}
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    figure.suptitle(
        "Day 23 — PyTorch CUDA reference vs native ONNX Runtime",
        fontsize=16,
        fontweight="bold",
    )

    scatter_axis, distribution_axis, per_state_axis, action_axis = axes.ravel()
    all_q_values: list[float] = []
    reference_actions_plotted = False
    for key, provider_value in providers.items():
        if not isinstance(provider_value, Mapping):
            raise ValueError(f"provider result {key} is malformed")
        batch = _provider_batch(provider_value)
        metrics = batch.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"provider result {key} has no metrics")
        reference_q = np.asarray(metrics.get("reference_q_values"), dtype=np.float64)
        candidate_q = np.asarray(metrics.get("candidate_q_values"), dtype=np.float64)
        if (
            reference_q.ndim != 2
            or candidate_q.shape != reference_q.shape
            or not np.isfinite(reference_q).all()
            or not np.isfinite(candidate_q).all()
        ):
            raise ValueError(f"provider result {key} has invalid Q-value arrays")
        all_q_values.extend(reference_q.ravel().tolist())
        all_q_values.extend(candidate_q.ravel().tolist())
        color = palette.get(str(key), "#555555")
        scatter_axis.scatter(
            reference_q.ravel(),
            candidate_q.ravel(),
            s=14,
            alpha=0.52,
            color=color,
            label=str(key).upper(),
            edgecolors="none",
        )

        absolute_errors = _metric_array(metrics, "absolute_error_values")
        distribution_axis.hist(
            absolute_errors,
            bins=24,
            alpha=0.52,
            color=color,
            label=str(key).upper(),
        )
        sample_errors = _metric_array(metrics, "per_sample_max_absolute_error")
        sample_ids = np.arange(sample_errors.size)
        per_state_axis.plot(
            sample_ids,
            sample_errors,
            color=color,
            linewidth=1.4,
            alpha=0.8,
            label=f"{str(key).upper()} observed max",
        )
        thresholds = batch.get("thresholds")
        if isinstance(thresholds, Mapping):
            checks = thresholds.get("checks")
            if isinstance(checks, Mapping) and isinstance(
                checks.get("max_absolute_error"), Mapping
            ):
                limit = float(checks["max_absolute_error"]["limit"])
                per_state_axis.axhline(
                    limit,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.65,
                    label=f"{str(key).upper()} limit",
                )

        reference_actions = np.asarray(metrics.get("reference_actions"), dtype=np.int64)
        candidate_actions = np.asarray(metrics.get("candidate_actions"), dtype=np.int64)
        if reference_actions.shape != candidate_actions.shape:
            raise ValueError(f"provider result {key} has mismatched action arrays")
        if not reference_actions_plotted:
            action_axis.scatter(
                sample_ids,
                reference_actions,
                s=28,
                color="#222222",
                marker="x",
                label="PyTorch reference",
            )
            reference_actions_plotted = True
        action_axis.scatter(
            sample_ids,
            candidate_actions,
            s=22,
            color=color,
            alpha=0.78,
            label=f"{str(key).upper()} ORT",
        )

    low = min(all_q_values)
    high = max(all_q_values)
    padding = max((high - low) * 0.04, 1e-6)
    limits = (low - padding, high + padding)
    scatter_axis.plot(limits, limits, color="#555555", linewidth=1.0, linestyle="--")
    scatter_axis.set_xlim(limits)
    scatter_axis.set_ylim(limits)
    scatter_axis.set_xlabel("PyTorch CUDA reference Q-value")
    scatter_axis.set_ylabel("ORT Q-value")
    scatter_axis.set_title("Q-value scatter (batch N=4)")
    scatter_axis.legend(frameon=False, loc="upper left")
    scatter_axis.grid(alpha=0.22)

    distribution_axis.set_xlabel("absolute error")
    distribution_axis.set_ylabel("element count")
    distribution_axis.set_title("Distribution of all Q-value errors")
    distribution_axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    distribution_axis.legend(frameon=False)
    distribution_axis.grid(alpha=0.22)

    per_state_axis.set_xlabel("probe sample id")
    per_state_axis.set_ylabel("per-state maximum absolute error")
    per_state_axis.set_title("Per-state error against configured limit")
    per_state_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    per_state_axis.legend(frameon=False, fontsize=8)
    per_state_axis.grid(alpha=0.22)

    action_axis.set_xlabel("probe sample id")
    action_axis.set_ylabel("greedy action index")
    action_axis.set_title("Greedy action agreement")
    action_axis.set_yticks(range(4), ["0 NOOP", "1 FIRE", "2 RIGHT", "3 LEFT"])
    action_axis.set_ylim(-0.5, 3.5)
    action_axis.legend(frameon=False, loc="upper right")
    action_axis.grid(axis="y", alpha=0.22)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def render_from_files(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path]:
    root = _repository_root()
    source = Path(input_path).resolve()
    destination = Path(output).resolve()
    result = _json_object(source)
    figure = render_parity_figure(result, destination)
    sidecar = figure.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "day23_onnx_runtime_parity_visualization",
                "technical_question": (
                    "Do Q-value deviations stay close and preserve greedy action "
                    "choices across native ORT providers?"
                ),
                "source_comparison": {
                    "path": _relative_path(source, root=root),
                    "sha256": _sha256_file(source),
                    "lf_normalized_sha256": _sha256_file(
                        source,
                        normalize_text=True,
                    ),
                },
                "selected_batch": 4,
                "output": _relative_path(figure, root=root),
                "command": (
                    "python -m scripts.visualization.visualize_pytorch_onnx_parity "
                    f"--input {_relative_path(source, root=root)} "
                    f"--output {_relative_path(figure, root=root)}"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return figure, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        figure, sidecar = render_from_files(
            input_path=args.input,
            output=args.output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 23 parity visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"figure": str(figure), "metadata": str(sidecar)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
