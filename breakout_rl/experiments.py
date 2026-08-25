"""Reusable config, manifest, comparison, and artifact helpers for Day 14."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.training.config import DQNConfig


CONFIG_FIELD_NAMES: tuple[str, ...] = tuple(field.name for field in fields(DQNConfig))
DEFAULT_RECENT_WINDOW = 20
DEFAULT_ROLLING_WINDOW = 20
EXPERIMENT_BUDGETS: dict[str, tuple[int, int]] = {
    "smoke": (1_000, 10_000),
    "development": (10_000, 50_000),
    "pilot": (100_000, 1_000_000),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and reject an accidentally supplied JSON array."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def write_json_object(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return destination


def _validate_override_fields(values: Mapping[str, Any], *, source: Path) -> None:
    unknown = sorted(set(values) - set(CONFIG_FIELD_NAMES))
    if unknown:
        raise ValueError(
            f"{source} contains unknown config field(s): {', '.join(unknown)}"
        )


def _config_values(payload: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    allowed_metadata = {
        "name",
        "label",
        "description",
        "base_config",
        "overrides",
        "config",
        "budget_level",
    }
    unknown_top_level = sorted(
        set(payload) - set(CONFIG_FIELD_NAMES) - allowed_metadata
    )
    if unknown_top_level:
        raise ValueError(
            f"{source} contains unknown config field(s): {', '.join(unknown_top_level)}"
        )
    nested = payload.get("config")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError(f"{source}: config must be a JSON object")
        values = dict(nested)
    else:
        values = {name: payload[name] for name in CONFIG_FIELD_NAMES if name in payload}
    _validate_override_fields(values, source=source)
    return values


def _inferred_budget_level(total_steps: int) -> str:
    for level, (minimum, maximum) in EXPERIMENT_BUDGETS.items():
        if minimum <= total_steps <= maximum:
            return level
    return "custom"


def _resolve_budget_level(
    payload: Mapping[str, Any],
    *,
    total_steps: int,
    inherited: str | None,
    source: Path,
) -> str:
    raw_level = payload.get("budget_level")
    if raw_level is None and inherited not in {None, "custom"}:
        raw_level = inherited
    if raw_level is None:
        return _inferred_budget_level(total_steps)
    if not isinstance(raw_level, str) or not raw_level.strip():
        raise TypeError(f"{source}: budget_level must be a non-empty string")
    level = raw_level.strip().lower()
    if level not in EXPERIMENT_BUDGETS:
        raise ValueError(
            f"{source}: budget_level must be one of "
            f"{', '.join(EXPERIMENT_BUDGETS)}"
        )
    minimum, maximum = EXPERIMENT_BUDGETS[level]
    if not minimum <= total_steps <= maximum:
        raise ValueError(
            f"{source}: total_steps={total_steps} is outside the {level} budget "
            f"range [{minimum}, {maximum}]"
        )
    return level


def _label_for(payload: Mapping[str, Any], source: Path) -> str:
    raw_label = payload.get("label", payload.get("name", source.stem))
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError(f"{source}: label must be a non-empty string")
    return raw_label.strip()


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully resolved DQN config plus its source/override provenance."""

    label: str
    source_path: Path
    config: DQNConfig
    base_config_path: Path | None
    overrides: dict[str, Any]
    budget_level: str

    @property
    def values(self) -> dict[str, Any]:
        return self.config.to_dict()


def load_experiment_config(
    path: str | Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> ExperimentConfig:
    """Load a full config or a ``base_config`` plus validated overrides."""

    source = Path(path).resolve()
    if source in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, source))
        raise ValueError(f"experiment config inheritance cycle: {cycle}")
    payload = read_json_object(source)
    base_path: Path | None = None
    base_values: dict[str, Any] = {}
    inherited_budget_level: str | None = None
    raw_base = payload.get("base_config")
    if raw_base is not None:
        if not isinstance(raw_base, str) or not raw_base.strip():
            raise TypeError(f"{source}: base_config must be a non-empty path")
        base_path = (source.parent / raw_base).resolve()
        base = load_experiment_config(base_path, _stack=(*_stack, source))
        base_values.update(base.values)
        inherited_budget_level = base.budget_level

    explicit_values = _config_values(payload, source=source)
    raw_overrides = payload.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise TypeError(f"{source}: overrides must be a JSON object")
    overrides = dict(raw_overrides)
    _validate_override_fields(overrides, source=source)
    if base_path is None and overrides:
        raise ValueError(f"{source}: overrides require base_config")

    values = {**base_values, **overrides, **explicit_values}
    config = DQNConfig.from_dict(values)
    budget_level = _resolve_budget_level(
        payload,
        total_steps=config.total_steps,
        inherited=inherited_budget_level,
        source=source,
    )
    return ExperimentConfig(
        label=_label_for(payload, source),
        source_path=source,
        config=config,
        base_config_path=base_path,
        overrides={**overrides, **explicit_values} if base_path is not None else {},
        budget_level=budget_level,
    )


def load_experiment_configs(paths: Iterable[str | Path]) -> list[ExperimentConfig]:
    configs = [load_experiment_config(path) for path in paths]
    if not configs:
        raise ValueError("at least one experiment config is required")
    return configs


def config_diff(
    base_values: Mapping[str, Any],
    variant_values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return only changed DQN fields while retaining both values."""

    diff: dict[str, dict[str, Any]] = {}
    for name in CONFIG_FIELD_NAMES:
        base_value = base_values.get(name)
        variant_value = variant_values.get(name)
        if base_value != variant_value:
            diff[name] = {"base": base_value, "variant": variant_value}
    return diff


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    normalized = normalized.strip("-._")
    return normalized or "run"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative_path(path: str | Path, *, start: str | Path) -> str:
    source = Path(path).resolve()
    base = Path(start).resolve()
    return os.path.relpath(source, base).replace(os.sep, "/")


def build_manifest(
    *,
    experiment_id: str,
    configs: Sequence[ExperimentConfig],
    manifest_path: str | Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create the durable manifest before any run starts."""

    if not configs:
        raise ValueError("configs must not be empty")
    base = configs[0]
    manifest_parent = Path(manifest_path).resolve().parent
    variants = []
    for config in configs:
        variants.append(
            {
                "label": config.label,
                "config_path": relative_path(config.source_path, start=manifest_parent),
                "config_values": config.values,
                "overrides": config.overrides,
                "changed_fields": sorted(config_diff(base.values, config.values)),
                "status": "pending",
                "run_dir": None,
                "run_id": None,
                "requested_device": config.config.requested_device,
                "resolved_device": None,
                "seed": config.config.seed,
                "step_budget": config.config.total_steps,
                "budget_level": config.budget_level,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "sequential": True,
        "base_config": {
            "label": base.label,
            "config_path": relative_path(base.source_path, start=manifest_parent),
            "values": base.values,
        },
        "variants": variants,
        "seeds": sorted({config.config.seed for config in configs}),
        "step_budgets": sorted({config.config.total_steps for config in configs}),
        "budget_levels": sorted({config.budget_level for config in configs}),
        "status": "running",
        "command": list(command) if command is not None else None,
    }


def update_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    payload = dict(manifest)
    payload["updated_at_utc"] = utc_timestamp()
    return write_json_object(path, payload)


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_metrics(run_dir: str | Path) -> list[dict[str, str]]:
    path = Path(run_dir) / "metrics.csv"
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _series(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _parse_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _episode_returns(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row in rows:
        step = _parse_float(row.get("global_step"))
        episode_return = _parse_float(row.get("raw_episode_return"))
        if step is not None and episode_return is not None:
            result.append((step, episode_return))
    return result


def _rolling_means(values: Sequence[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("rolling window must be greater than zero")
    if len(values) < window:
        return []
    return [float(mean(values[index - window : index])) for index in range(window, len(values) + 1)]


def load_run_report(
    run_dir: str | Path,
    *,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    """Read one run, including incomplete/failed runs without hiding them."""

    if recent_window < 1 or rolling_window < 1:
        raise ValueError("aggregate windows must be greater than zero")
    path = Path(run_dir).resolve()
    config = read_json_object(path / "config.json") if (path / "config.json").is_file() else {}
    summary = read_json_object(path / "summary.json") if (path / "summary.json").is_file() else {}
    failure = read_json_object(path / "failure.json") if (path / "failure.json").is_file() else {}
    rows = read_metrics(path)
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    requested_device = str(
        runtime.get("requested_device", config.get("device", "unavailable"))
    )
    resolved_device = str(runtime.get("resolved_device", "unavailable"))
    expected_steps = _parse_float(config.get("total_steps"))
    step_values = _series(rows, "global_step")
    completed_steps = int(max(step_values)) if step_values else int(summary.get("total_steps", 0) or 0)
    episode_returns = _episode_returns(rows)
    returns = [value for _, value in episode_returns]
    recent = returns[-recent_window:]
    rolling = _rolling_means(returns, rolling_window)
    status = str(summary.get("status", failure.get("status", "incomplete")))
    if status == "completed" and expected_steps is not None and completed_steps < int(expected_steps):
        status = "incomplete"
    if failure and status == "incomplete":
        status = str(failure.get("status", "failed"))
    errors = summary.get("error", failure.get("error"))
    q_fields = ("q_mean", "q_max", "q_min", "target_mean", "target_max")
    q_summary = {field: _stats(_series(rows, field)) for field in q_fields}
    runtime_sps = _parse_float(runtime.get("steps_per_second"))
    if runtime_sps is None:
        runtime_sps = _parse_float(summary.get("steps_per_second"))
    sps_values = _series(rows, "sps") or _series(rows, "steps_per_second")
    wall_clock = _parse_float(runtime.get("wall_clock_seconds"))
    if wall_clock is None:
        wall_clock = _parse_float(summary.get("wall_clock_seconds"))
    return {
        "run_id": str(config.get("run_id", path.name)),
        "run_dir": str(path),
        "label": path.name,
        "status": status,
        "error": errors,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "precision": runtime.get("precision", config.get("precision", "unavailable")),
        "gpu_name": runtime.get("gpu_name", runtime.get("cuda_device_name")),
        "cuda_device_index": runtime.get("cuda_device_index"),
        "pytorch_version": runtime.get("pytorch_version"),
        "torch_cuda_version": runtime.get("torch_cuda_version"),
        "expected_steps": int(expected_steps) if expected_steps is not None else None,
        "completed_steps": completed_steps,
        "episodes": int(summary.get("episodes", len(returns)) or 0),
        "recent_window": recent_window,
        "rolling_window": rolling_window,
        "recent_episode_return": _stats(recent),
        "mean_recent_episode_return": float(mean(recent)) if recent else None,
        "median_recent_episode_return": float(median(recent)) if recent else None,
        "best_rolling_return": float(max(rolling)) if rolling else None,
        "rolling_return_count": len(rolling),
        "loss_summary": _stats(_series(rows, "loss")),
        "q_value_summary": q_summary,
        "sps": {
            **_stats(sps_values),
            "runtime": runtime_sps,
        },
        "wall_clock_seconds": wall_clock,
        "gpu_memory": {
            "allocated_bytes": runtime.get("cuda_allocated_bytes"),
            "peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
            "reserved_bytes": runtime.get("cuda_reserved_bytes"),
            "peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
        },
        "config": {name: config.get(name) for name in CONFIG_FIELD_NAMES if name in config},
        "runtime": dict(runtime),
        "summary": summary,
        "metrics": rows,
    }


def _manifest_run_path(
    manifest_path: Path,
    raw_path: Any,
    *,
    allow_missing: bool,
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        if allow_missing:
            return None
        raise ValueError("manifest variant is missing run_dir")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_manifest_run_paths(
    manifest_path: str | Path,
    *,
    allow_missing: bool = False,
) -> list[tuple[dict[str, Any], Path | None]]:
    manifest_source = Path(manifest_path).resolve()
    manifest = read_json_object(manifest_source)
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{manifest_source}: variants must be a non-empty array")
    result: list[tuple[dict[str, Any], Path | None]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise TypeError(f"{manifest_source}: each variant must be an object")
        result.append(
            (
                variant,
                _manifest_run_path(
                    manifest_source,
                    variant.get("run_dir"),
                    allow_missing=allow_missing,
                ),
            )
        )
    return result


def _comparison_conditions(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested_devices = [str(report["requested_device"]) for report in reports]
    resolved_devices = [str(report["resolved_device"]) for report in reports]
    expected_steps = [report["expected_steps"] for report in reports]
    statuses = [str(report["status"]) for report in reports]
    formal_requested = all(
        device == "cuda" or device.startswith("cuda:") for device in requested_devices
    )
    formal_resolved = all(device.startswith("cuda:") for device in resolved_devices)
    return {
        "same_requested_device": len(set(requested_devices)) == 1,
        "same_resolved_device": len(set(resolved_devices)) == 1,
        "same_step_budget": len(set(expected_steps)) == 1,
        "requested_devices": requested_devices,
        "resolved_devices": resolved_devices,
        "step_budgets": expected_steps,
        "formal_cuda_eligible": (
            all(status == "completed" for status in statuses)
            and formal_requested
            and formal_resolved
            and len(set(requested_devices)) == 1
            and len(set(resolved_devices)) == 1
            and len(set(expected_steps)) == 1
        ),
        "quality_and_throughput_are_separate": True,
    }


def _not_started_report(
    entry: Mapping[str, Any],
    *,
    recent_window: int,
    rolling_window: int,
) -> dict[str, Any]:
    raw_status = str(entry.get("status", "not_started"))
    status = "not_started" if raw_status in {"pending", "planned"} else raw_status
    config_values = entry.get("config_values", {})
    if not isinstance(config_values, Mapping):
        config_values = {}
    expected_steps = entry.get("step_budget")
    return {
        "run_id": entry.get("run_id"),
        "run_dir": None,
        "label": str(entry.get("label", "not-started")),
        "status": status,
        "error": entry.get("error"),
        "requested_device": str(entry.get("requested_device", "unavailable")),
        "resolved_device": str(entry.get("resolved_device", "unavailable")),
        "precision": config_values.get("precision", "unavailable"),
        "gpu_name": None,
        "cuda_device_index": None,
        "pytorch_version": None,
        "torch_cuda_version": None,
        "expected_steps": expected_steps,
        "completed_steps": 0,
        "episodes": 0,
        "recent_window": recent_window,
        "rolling_window": rolling_window,
        "recent_episode_return": _stats([]),
        "mean_recent_episode_return": None,
        "median_recent_episode_return": None,
        "best_rolling_return": None,
        "rolling_return_count": 0,
        "loss_summary": _stats([]),
        "q_value_summary": {field: _stats([]) for field in ("q_mean", "q_max", "q_min", "target_mean", "target_max")},
        "sps": {**_stats([]), "runtime": None},
        "wall_clock_seconds": None,
        "gpu_memory": {
            "allocated_bytes": None,
            "peak_allocated_bytes": None,
            "reserved_bytes": None,
            "peak_reserved_bytes": None,
        },
        "config": dict(config_values),
        "runtime": {},
        "summary": {},
    }


def compare_run_dirs(
    run_dirs: Sequence[str | Path],
    *,
    base_values: Mapping[str, Any] | None = None,
    labels: Sequence[str] | None = None,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    reports = [
        load_run_report(
            run_dir,
            recent_window=recent_window,
            rolling_window=rolling_window,
        )
        for run_dir in run_dirs
    ]
    if base_values is None:
        base_values = reports[0]["config"]
    for index, report in enumerate(reports):
        report["label"] = labels[index] if labels and index < len(labels) else report["run_id"]
        report["config_diff"] = config_diff(base_values, report["config"])
        report.pop("metrics", None)
    return {
        "schema_version": 1,
        "aggregate_windows": {
            "recent_episode_returns": recent_window,
            "rolling_episode_returns": rolling_window,
        },
        "comparison_conditions": _comparison_conditions(reports),
        "runs": reports,
    }


def compare_manifest(
    manifest_path: str | Path,
    *,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = read_json_object(source)
    entries = load_manifest_run_paths(source, allow_missing=True)
    base = manifest.get("base_config", {})
    base_values = base.get("values") if isinstance(base, Mapping) else None
    if not isinstance(base_values, Mapping):
        base_values = None
    available = [(entry, path) for entry, path in entries if path is not None]
    if available:
        available_report = compare_run_dirs(
            [path for _, path in available if path is not None],
            base_values=base_values,
            labels=[str(entry.get("label", path.name)) for entry, path in available],
            recent_window=recent_window,
            rolling_window=rolling_window,
        )
        reports_by_path = {
            str(report["run_dir"]): report for report in available_report["runs"]
        }
        report = {
            **available_report,
            "runs": [],
        }
    else:
        report = {
            "schema_version": 1,
            "aggregate_windows": {
                "recent_episode_returns": recent_window,
                "rolling_episode_returns": rolling_window,
            },
            "runs": [],
        }
        reports_by_path = {}
    for entry, path in entries:
        if path is None:
            missing = _not_started_report(
                entry,
                recent_window=recent_window,
                rolling_window=rolling_window,
            )
            missing["config_diff"] = config_diff(base_values or {}, missing["config"])
            report["runs"].append(missing)
        else:
            report["runs"].append(reports_by_path[str(path)])
    report["comparison_conditions"] = _comparison_conditions(report["runs"])
    report["experiment_id"] = manifest.get("experiment_id", source.parent.name)
    report["manifest"] = str(source)
    report["manifest_status"] = manifest.get("status")
    report["sequential"] = bool(manifest.get("sequential", True))
    return report


__all__ = [
    "CONFIG_FIELD_NAMES",
    "DEFAULT_RECENT_WINDOW",
    "DEFAULT_ROLLING_WINDOW",
    "EXPERIMENT_BUDGETS",
    "ExperimentConfig",
    "build_manifest",
    "compare_manifest",
    "compare_run_dirs",
    "config_diff",
    "load_experiment_config",
    "load_experiment_configs",
    "load_manifest_run_paths",
    "load_run_report",
    "read_json_object",
    "read_metrics",
    "relative_path",
    "slugify",
    "update_manifest",
    "utc_timestamp",
    "write_json_object",
]
