"""CSV and JSON artifacts emitted by a DQN training run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from breakout_rl.training.config import DQNConfig
from breakout_rl.training.diagnostics import collect_runtime_metadata


METRIC_FIELDS: tuple[str, ...] = (
    "global_step",
    "episode",
    "raw_episode_return",
    "episode_length",
    "current_raw_episode_return",
    "current_training_episode_return",
    "epsilon",
    "loss",
    "q_mean",
    "q_max",
    "q_min",
    "target_mean",
    "target_max",
    "td_error_mean_abs",
    "td_error_max_abs",
    "gradient_norm",
    "replay_size",
    "steps_per_second",
    "sps",
    "optimizer_updates",
    "optimizer_updated",
    "target_sync_count",
    "last_target_sync_step",
    "raw_reward",
    "training_reward",
    "action",
    "action_name",
    "action_source",
    "noop_count",
    "fire_count",
    "right_count",
    "left_count",
    "random_decision_count",
    "greedy_decision_count",
    "random_decision_ratio",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


class MetricsLogger:
    """Append one structured row per environment step and write run metadata."""

    def __init__(
        self,
        run_dir: str | Path,
        config: DQNConfig,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(config, DQNConfig):
            raise TypeError("config must be a DQNConfig")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.run_dir / "config.json"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        if not self.config_path.exists():
            runtime_metadata = collect_runtime_metadata(
                seed=config.seed,
                device=config.device,
                run_dir=self.run_dir,
                extra=metadata,
            )
            config_payload = {
                "run_id": self.run_dir.name,
                **config.to_dict(),
                "runtime": runtime_metadata,
            }
            self.config_path.write_text(
                json.dumps(config_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        is_empty = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        self._file = self.metrics_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=list(METRIC_FIELDS),
            extrasaction="ignore",
        )
        if is_empty:
            self._writer.writeheader()
            self._file.flush()
        self._closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("metrics logger is closed")
        self._writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in METRIC_FIELDS
            }
        )
        self._file.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(dict(summary), indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["METRIC_FIELDS", "MetricsLogger"]
