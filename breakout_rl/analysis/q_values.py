"""Fixed-probe Q-value inference and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from breakout_rl.inference import (
    EXPECTED_ACTION_MEANINGS,
    prepare_model_input,
)
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES

if TYPE_CHECKING:
    import torch
    from torch import nn


PROBE_OBSERVATION_SHAPE = (4, 84, 84)


def validate_probe_states(states: Any) -> np.ndarray:
    """Validate the compact uint8 batch used at the model boundary."""

    array = np.asarray(states)
    if array.dtype != np.uint8:
        raise TypeError("probe states must have dtype uint8")
    if array.ndim != 4 or tuple(array.shape[1:]) != PROBE_OBSERVATION_SHAPE:
        raise ValueError(
            "probe states must have shape (N, 4, 84, 84); "
            f"received {tuple(array.shape)}"
        )
    if int(array.shape[0]) < 1:
        raise ValueError("probe states must contain at least one observation")
    return np.ascontiguousarray(array)


def save_probe_states(
    path: str | Path,
    states: Any,
    metadata: Mapping[str, Any],
) -> Path:
    """Save observations and JSON metadata in a portable NPZ artifact."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    observations = validate_probe_states(states)
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        destination,
        observations=observations,
        metadata_json=np.array(metadata_json),
    )
    return destination


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def load_probe_states(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate a probe NPZ, accepting ``states`` as a legacy key."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if "observations" in archive:
            states = archive["observations"]
        elif "states" in archive:
            states = archive["states"]
        else:
            raise ValueError("probe NPZ must contain an observations array")
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive:
            raw_metadata = _metadata_value(archive["metadata_json"])
            if not isinstance(raw_metadata, str):
                raise ValueError("probe metadata_json must contain a JSON string")
            try:
                parsed = json.loads(raw_metadata)
            except json.JSONDecodeError as error:
                raise ValueError("probe metadata_json is invalid JSON") from error
            if not isinstance(parsed, Mapping):
                raise ValueError("probe metadata_json must describe an object")
            metadata = dict(parsed)
    return validate_probe_states(states), metadata


def infer_q_values(
    model: nn.Module,
    probe_states: Any,
    *,
    device: torch.device | str,
) -> np.ndarray:
    """Run no-grad inference on validated uint8 probe states."""

    import torch
    from torch import nn

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    observations = validate_probe_states(probe_states)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    model = model.to(resolved_device)
    was_training = model.training
    model.eval()
    try:
        inputs = prepare_model_input(observations, device=resolved_device)
        with torch.no_grad():
            outputs = model(inputs)
    finally:
        model.train(was_training)
    if not isinstance(outputs, torch.Tensor):
        raise TypeError("model must return a torch.Tensor")
    expected_shape = (int(observations.shape[0]), len(EXPECTED_ACTION_MEANINGS))
    if tuple(outputs.shape) != expected_shape:
        raise ValueError(f"model output must have shape {expected_shape}")
    if outputs.dtype != torch.float32:
        raise TypeError("model output must have dtype torch.float32")
    if not torch.isfinite(outputs).all().item():
        raise ValueError("model output contains non-finite Q-values")
    return np.ascontiguousarray(outputs.detach().cpu().numpy())


def summarize_q_values(
    q_values: Any,
    *,
    action_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Aggregate Q statistics and the greedy action distribution."""

    array = np.asarray(q_values)
    if array.ndim != 2 or int(array.shape[0]) < 1 or int(array.shape[1]) < 1:
        raise ValueError("q_values must have shape (N, action_count)")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("q_values must contain real numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("q_values must contain only finite values")
    selected_actions = np.argmax(array, axis=1)
    counts = np.bincount(selected_actions, minlength=array.shape[1])
    names = action_names or ATARI_ACTION_NAMES
    distribution: dict[str, int] = {}
    for index, count in enumerate(counts):
        if count:
            label = names.get(index, f"ACTION_{index}")
            distribution[label] = int(count)
    max_values = array.max(axis=1)
    return {
        "probe_count": int(array.shape[0]),
        "action_count": int(array.shape[1]),
        "q_mean": float(array.mean()),
        "q_std": float(array.std()),
        "q_min": float(array.min()),
        "q_max": float(array.max()),
        "max_q_mean": float(max_values.mean()),
        "max_q_std": float(max_values.std()),
        "selected_action_distribution": distribution,
        "selected_action_indices": [int(index) for index in selected_actions],
    }


def analyze_q_values(
    model: nn.Module,
    probe_states: Any,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Infer and summarize Q-values for one fixed probe set."""

    observations = validate_probe_states(probe_states)
    q_values = infer_q_values(model, observations, device=device)
    return {
        **summarize_q_values(q_values),
        "observation_shape": list(PROBE_OBSERVATION_SHAPE),
        "observation_dtype": str(observations.dtype),
        "q_values": q_values.tolist(),
    }


def plot_q_probe_summary(q_values: Any, output: str | Path) -> Path:
    """Plot actual per-action Q distributions and selected action counts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(q_values, dtype=np.float64)
    summary = summarize_q_values(array)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        ATARI_ACTION_NAMES.get(index, f"ACTION_{index}")
        for index in range(array.shape[1])
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].boxplot(
        [array[:, index] for index in range(array.shape[1])],
        tick_labels=labels,
    )
    axes[0].set_title("Q-value distribution on fixed probes")
    axes[0].set_ylabel("Q-value")
    axes[0].grid(axis="y", alpha=0.25)
    counts = [summary["selected_action_distribution"].get(label, 0) for label in labels]
    axes[1].bar(labels, counts)
    axes[1].set_title("Greedy action per probe")
    axes[1].set_ylabel("probe count")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


__all__ = [
    "PROBE_OBSERVATION_SHAPE",
    "analyze_q_values",
    "infer_q_values",
    "load_probe_states",
    "plot_q_probe_summary",
    "save_probe_states",
    "summarize_q_values",
    "validate_probe_states",
]
