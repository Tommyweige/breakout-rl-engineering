"""Validation and aggregation helpers for the Day 18 DQN comparison.

The comparison is intentionally represented as a manifest of stage artifacts.
Training and evaluation are separate records so an interrupted run remains
visible without being converted into a zero-valued result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summary_from_episode_rows,
    validate_embedded_summary,
    validate_episode_rows,
)
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.backend_manifest import load_day16_backend_manifest
from breakout_rl.training.config import DQNConfig


DAY18_SCHEMA_VERSION = 1
DAY18_ALGORITHMS: tuple[str, str] = ("dqn", "double_dqn")
DAY18_TRAINING_SEEDS: tuple[int, int, int] = (11, 22, 33)
DAY18_MILESTONES: dict[str, int] = {
    "screening": 100_000,
    "pilot": 250_000,
    "main": 500_000,
}
DAY18_FORMAL_STAGE = "main"
DAY18_VARIABLE_CONFIG_FIELDS = frozenset(
    {"algorithm", "seed", "total_steps", "checkpoint_interval"}
)
DAY18_REQUIRED_TRAINING_FIELDS: tuple[str, ...] = (
    "raw_episode_return",
    "loss",
    "q_mean",
    "q_max",
    "target_mean",
    "td_error_mean_abs",
    "gradient_norm",
    "sps",
)
DAY18_STAGE_COUNTER_FIELDS: tuple[str, ...] = (
    "vector_iterations",
    "optimizer_updates",
    "action_inference_batches",
    "action_inference_transitions",
    "replay_insertion_calls",
    "replay_insertion_transitions",
)
DAY18_STAGE_RATE_FIELDS: tuple[str, ...] = (
    "vector_iterations_per_second",
    "action_inference_batches_per_second",
    "action_inference_transitions_per_second",
    "replay_insertion_calls_per_second",
    "replay_insertion_transitions_per_second",
    "optimizer_updates_per_second",
    "training_samples_per_second",
)
DAY18_PROVENANCE_SOURCE_PATHS: tuple[str, ...] = (
    "configs/eval/breakout_contract_v2.json",
    "configs/training/day16-canonical-backend.json",
    "configs/experiments/day18-dqn-vs-double.json",
    "breakout_rl/training/vectorized.py",
    "breakout_rl/day18_comparison.py",
    "breakout_rl/evaluation.py",
    "scripts/training/train_vectorized_dqn.py",
    "scripts/training/run_day18_comparison.py",
    "scripts/analysis/analyze_q_values.py",
    "scripts/analysis/export_day18_evidence.py",
    "scripts/analysis/generate_day18_comparison_report.py",
    "scripts/visualization/visualize_day18_comparison.py",
    "scripts/analysis/rebuild_day18_derived_evidence.py",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: str | Path) -> str:
    """Return a portable path beginning at a repository-owned directory."""

    resolved = Path(path).resolve()
    markers = {
        "assets",
        "configs",
        "evaluations",
        "experiments",
        "reports",
        "runs",
    }
    for index, part in enumerate(resolved.parts):
        if part.lower() in markers:
            return Path(*resolved.parts[index:]).as_posix()
    return resolved.as_posix()


def relative_path(path: str | Path, *, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve()).replace(
        os.sep,
        "/",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: expected a JSON object")
    return dict(payload)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _resolve_reference(value: Any, *, source: Path, repository_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    for option in (repository_root / candidate, source.parent / candidate):
        if option.is_file():
            return option.resolve()
    return (repository_root / candidate).resolve()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _unique_ints(values: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    parsed = tuple(_positive_int(value, name=name) for value in values)
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique values")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _artifact_reference(path: Path, *, start: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative_path(path, start=start),
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class Day18ExperimentConfig:
    """Validated Day 18 source-of-truth configuration and dependencies."""

    source_path: Path
    repository_root: Path
    experiment_id: str
    backend_manifest_path: Path
    contract_path: Path
    evaluation_config_path: Path
    probe_states_path: Path
    algorithms: tuple[str, ...]
    training_seeds: tuple[int, ...]
    milestones: Mapping[str, int]
    formal_quality_horizon: str
    sequential: bool
    require_cuda: bool
    cuda_headroom_bytes: int
    resume_policy: str
    backend_manifest: Mapping[str, Any]
    contract: BreakoutEvaluationContractV2
    evaluation_config: Mapping[str, Any]
    backend_config: DQNConfig
    raw: Mapping[str, Any]

    @property
    def formal_steps(self) -> int:
        return int(self.milestones[self.formal_quality_horizon])

    @property
    def contract_provenance(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract.contract_id,
            "contract_path": relative_path(
                self.contract_path,
                start=self.repository_root,
            ),
            "contract_sha256": sha256_file(self.contract_path),
            "semantics": self.contract.to_dict(),
        }

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.evaluation_config["seeds"])

    @property
    def episodes_per_seed(self) -> int:
        return int(self.evaluation_config["episodes_per_seed"])

    def stage_steps(self, stage: str) -> int:
        if stage not in self.milestones:
            raise ValueError(f"unknown Day 18 stage: {stage}")
        return int(self.milestones[stage])

    def stages_through(self, stage: str) -> tuple[str, ...]:
        names = tuple(self.milestones)
        if stage not in names:
            raise ValueError(f"unknown Day 18 stage: {stage}")
        return names[: names.index(stage) + 1]

    def training_config(
        self,
        *,
        algorithm: str,
        seed: int,
        stage: str,
    ) -> DQNConfig:
        if algorithm not in self.algorithms:
            raise ValueError(f"unsupported Day 18 algorithm: {algorithm}")
        return self.backend_config.with_overrides(
            algorithm=algorithm,
            seed=seed,
            total_steps=self.stage_steps(stage),
            checkpoint_interval=self.stage_steps(stage),
        )


def load_day18_config(
    path: str | Path = "configs/experiments/day18-dqn-vs-double.json",
    *,
    repository_root: str | Path | None = None,
    require_probe_states: bool = False,
) -> Day18ExperimentConfig:
    source = Path(path).resolve()
    payload = _read_json(source)
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else source.parent.parent.parent
    )
    if payload.get("schema_version") != DAY18_SCHEMA_VERSION:
        raise ValueError("Day 18 config has an unsupported schema_version")
    if payload.get("day") != 18:
        raise ValueError("Day 18 config must have day=18")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Day 18 config experiment_id must be a non-empty string")

    raw_algorithms = payload.get("algorithms")
    if tuple(raw_algorithms or ()) != DAY18_ALGORITHMS:
        raise ValueError(
            "Day 18 algorithms must be exactly ['dqn', 'double_dqn']"
        )
    algorithms = tuple(str(value) for value in raw_algorithms)
    training_seeds = _unique_ints(
        payload.get("training_seeds"),
        name="training_seeds",
    )
    if training_seeds != DAY18_TRAINING_SEEDS:
        raise ValueError("Day 18 training_seeds must be [11, 22, 33]")

    raw_milestones = payload.get("milestones")
    if not isinstance(raw_milestones, Mapping):
        raise ValueError("Day 18 milestones must be an object")
    milestones = {
        stage: _positive_int(raw_milestones.get(stage), name=f"milestones.{stage}")
        for stage in DAY18_MILESTONES
    }
    if milestones != DAY18_MILESTONES:
        raise ValueError(
            "Day 18 milestones must be screening=100000, pilot=250000, main=500000"
        )
    horizon = payload.get("formal_quality_horizon")
    if horizon != DAY18_FORMAL_STAGE:
        raise ValueError("formal_quality_horizon must be 'main'")

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Day 18 execution must be an object")
    sequential = execution.get("sequential")
    require_cuda = execution.get("require_cuda")
    if sequential is not True:
        raise ValueError("Day 18 comparison execution must be sequential")
    if require_cuda is not True:
        raise ValueError("Day 18 formal comparison must require CUDA")
    resume_policy = execution.get("resume_policy")
    if not isinstance(resume_policy, str) or not resume_policy.strip():
        raise ValueError("Day 18 execution.resume_policy must be documented")

    backend_path = _resolve_reference(
        payload.get("backend_manifest"),
        source=source,
        repository_root=root,
    )
    contract_path = _resolve_reference(
        payload.get("contract"),
        source=source,
        repository_root=root,
    )
    evaluation_path = _resolve_reference(
        payload.get("evaluation_config"),
        source=source,
        repository_root=root,
    )
    probe_path = _resolve_reference(
        payload.get("probe_states"),
        source=source,
        repository_root=root,
    )
    if require_probe_states and not probe_path.is_file():
        raise FileNotFoundError(probe_path)

    backend_manifest = load_day16_backend_manifest(
        backend_path,
        repository_root=root,
        verify_evidence_files=False,
    )
    trainer = backend_manifest.get("trainer")
    if not isinstance(trainer, Mapping):
        raise ValueError("Day 16 backend manifest is missing trainer")
    backend_values = trainer.get("config")
    if not isinstance(backend_values, Mapping):
        raise ValueError("Day 16 backend manifest is missing trainer.config")
    backend_config = DQNConfig.from_dict(backend_values)
    if backend_config.algorithm != "dqn":
        raise ValueError("Day 16 backend control config must start from dqn")
    if backend_config.num_envs != 2 or backend_config.replay_backend != "gpu":
        raise ValueError("Day 18 must reuse the Day 16 N=2 GPU Replay backend")
    if backend_config.device != "cuda" or backend_config.precision != "float32":
        raise ValueError("Day 18 formal comparison requires CUDA float32")

    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    evaluation_config = _read_json(evaluation_path)
    raw_eval_seeds = _unique_ints(
        evaluation_config.get("seeds"),
        name="evaluation_config.seeds",
    )
    episodes_per_seed = _positive_int(
        evaluation_config.get("episodes_per_seed"),
        name="evaluation_config.episodes_per_seed",
    )
    epsilon = _finite_float(
        evaluation_config.get("epsilon"),
        name="evaluation_config.epsilon",
    )
    if raw_eval_seeds != tuple(
        int(seed) for seed in expand_concrete_episode_seeds(
            raw_eval_seeds,
            episodes_per_seed=episodes_per_seed,
        )[::episodes_per_seed]
    ):
        raise ValueError("evaluation config seed expansion is invalid")
    if contract.environment_id != evaluation_config.get("environment_id"):
        raise ValueError("Day 18 evaluation config and Contract v2 disagree on environment")
    if contract.concrete_episode_seeds != expand_concrete_episode_seeds(
        raw_eval_seeds,
        episodes_per_seed=episodes_per_seed,
    ):
        raise ValueError("Day 18 evaluation seeds do not match Contract v2")
    if epsilon != contract.evaluation_epsilon or epsilon != 0.0:
        raise ValueError("Day 18 formal evaluation requires Contract v2 epsilon=0")

    try:
        headroom = _positive_int(
            execution.get("cuda_headroom_bytes"),
            name="execution.cuda_headroom_bytes",
        )
    except ValueError as error:
        raise ValueError("Day 18 CUDA headroom must be positive") from error

    return Day18ExperimentConfig(
        source_path=source,
        repository_root=root,
        experiment_id=experiment_id.strip(),
        backend_manifest_path=backend_path,
        contract_path=contract_path,
        evaluation_config_path=evaluation_path,
        probe_states_path=probe_path,
        algorithms=algorithms,
        training_seeds=training_seeds,
        milestones=milestones,
        formal_quality_horizon=horizon,
        sequential=True,
        require_cuda=True,
        cuda_headroom_bytes=headroom,
        resume_policy=resume_policy.strip(),
        backend_manifest=backend_manifest,
        contract=contract,
        evaluation_config=evaluation_config,
        backend_config=backend_config,
        raw=payload,
    )


def config_diff(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    ignored_fields: Iterable[str] = DAY18_VARIABLE_CONFIG_FIELDS,
) -> dict[str, dict[str, Any]]:
    ignored = set(ignored_fields)
    fields = sorted(set(first) | set(second))
    return {
        field: {"first": first.get(field), "second": second.get(field)}
        for field in fields
        if field not in ignored and first.get(field) != second.get(field)
    }


def _stage_label(steps: int) -> str:
    for name, target in DAY18_MILESTONES.items():
        if target == steps:
            return name
    return f"{steps}"


def build_day18_manifest(
    config: Day18ExperimentConfig,
    *,
    manifest_path: str | Path,
    runs_root: str | Path = "runs",
    evaluations_root: str | Path = "evaluations",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    destination = Path(manifest_path).resolve()
    manifest_parent = destination.parent
    runs_base = Path(runs_root).resolve() / config.experiment_id
    eval_base = Path(evaluations_root).resolve() / config.experiment_id
    base_values = config.backend_config.to_dict()
    entries: list[dict[str, Any]] = []
    for stage, target in config.milestones.items():
        for seed in config.training_seeds:
            for algorithm in config.algorithms:
                stage_config = config.training_config(
                    algorithm=algorithm,
                    seed=seed,
                    stage=stage,
                )
                run_dir = runs_base / f"{algorithm}-seed{seed}" / f"stage-{target // 1000}k"
                evaluation_dir = (
                    eval_base
                    / f"{algorithm}-seed{seed}"
                    / f"step-{target:08d}"
                )
                entries.append(
                    {
                        "run_id": f"{config.experiment_id}-{algorithm}-seed{seed}-{stage}",
                        "pair_id": f"training-seed-{seed}",
                        "algorithm": algorithm,
                        "training_seed": seed,
                        "stage": stage,
                        "target_transitions": target,
                        "status": "pending",
                        "run_dir": relative_path(run_dir, start=manifest_parent),
                        "resume_from": None,
                        "checkpoint": None,
                        "evaluation": {
                            "directory": relative_path(
                                evaluation_dir,
                                start=manifest_parent,
                            ),
                            "results": None,
                            "episodes": None,
                        },
                        "q_probe": None,
                        "training_config": stage_config.to_dict(),
                        "summary": None,
                        "error": None,
                    }
                )

    return {
        "schema_version": DAY18_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "status": "planned",
        "sequential": True,
        "source_of_truth": {
            "comparison_config": _artifact_reference(
                config.source_path,
                start=config.repository_root,
            ),
            "backend_manifest": {
                **_artifact_reference(
                    config.backend_manifest_path,
                    start=config.repository_root,
                ),
                "backend_id": config.backend_manifest.get("backend_id"),
            },
            "contract": {
                **_artifact_reference(
                    config.contract_path,
                    start=config.repository_root,
                ),
                **config.contract_provenance,
            },
            "evaluation_config": _artifact_reference(
                config.evaluation_config_path,
                start=config.repository_root,
            ),
            "probe_states": {
                "path": relative_path(
                    config.probe_states_path,
                    start=config.repository_root,
                ),
                "sha256": (
                    sha256_file(config.probe_states_path)
                    if config.probe_states_path.is_file()
                    else None
                ),
            },
        },
        "protocol": {
            "algorithms": list(config.algorithms),
            "training_seeds": list(config.training_seeds),
            "paired_seed_rule": "for each training seed, DQN and Double DQN share the same seed, backend, stage budget, and evaluation protocol",
            "milestones": dict(config.milestones),
            "formal_quality_horizon": config.formal_quality_horizon,
            "formal_quality_transitions": config.formal_steps,
            "screening_is_not_final_selection": True,
            "evaluation_seeds": list(config.evaluation_seeds),
            "episodes_per_evaluation_seed": config.episodes_per_seed,
            "evaluation_epsilon": 0.0,
            "raw_reward": True,
            "requires_cuda": True,
            "sequential": True,
            "cuda_headroom_bytes": config.cuda_headroom_bytes,
            "resume_policy": config.resume_policy,
            "actual_transition_definition": "global_step is accepted environment transitions, not vector iterations, optimizer updates, or raw Atari frames",
        },
        "base_backend": {
            "backend_id": config.backend_manifest.get("backend_id"),
            "trainer": config.backend_manifest.get("trainer"),
            "control_values": base_values,
        },
        "provenance": {
            "source_hashes": day18_source_hashes(config.repository_root),
            "historical_run_worktree_provenance": {
                "git_dirty_at_run": "not_applicable_before_runs_exist",
                "git_diff_sha256_at_run": "not_applicable_before_runs_exist",
                "limitation": (
                    "Future runs record git_dirty and git_diff_sha256 in runtime metadata; "
                    "this planned manifest has no historical run state."
                ),
            },
        },
        "runs": entries,
        "random_baseline_results": relative_path(
            config.repository_root
            / "evaluations/day16-contract-v2-random/results.json",
            start=manifest_parent,
        ),
        "command": list(command) if command is not None else None,
    }


def resolve_manifest_reference(manifest_path: str | Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(manifest_path).resolve().parent / candidate).resolve()


def read_day18_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != DAY18_SCHEMA_VERSION:
        raise ValueError("unsupported Day 18 manifest schema_version")
    if not isinstance(payload.get("runs"), list) or not payload["runs"]:
        raise ValueError("Day 18 manifest runs must be a non-empty array")
    if payload.get("sequential") is not True:
        raise ValueError("Day 18 manifest must record sequential execution")
    for index, entry in enumerate(payload["runs"]):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Day 18 manifest runs[{index}] must be an object")
        for field in ("algorithm", "training_seed", "stage", "target_transitions"):
            if field not in entry:
                raise ValueError(f"Day 18 manifest runs[{index}] is missing {field}")
        if entry["algorithm"] not in DAY18_ALGORITHMS:
            raise ValueError(f"unsupported Day 18 algorithm: {entry['algorithm']}")
        if entry["stage"] not in DAY18_MILESTONES:
            raise ValueError(f"unsupported Day 18 stage: {entry['stage']}")
        if int(entry["target_transitions"]) != DAY18_MILESTONES[entry["stage"]]:
            raise ValueError(
                f"run {index} target_transitions does not match its stage"
            )
    return payload


def _read_metrics(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer_value(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _counter_from_report(
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    name: str,
) -> int | None:
    value = summary.get(name)
    if value is None:
        value = runtime.get(name)
    return _integer_value(value)


def _stage_start_from_report(
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, int] | None:
    raw = runtime.get("stage_start_counters", summary.get("stage_start_counters"))
    if not isinstance(raw, Mapping):
        return None
    result: dict[str, int] = {}
    for name in ("global_step", "physical_environment_steps", *DAY18_STAGE_COUNTER_FIELDS):
        value = _integer_value(raw.get(name))
        if value is None or value < 0:
            return None
        result[name] = value
    return result


def _current_stage_counters(
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, int] | None:
    result: dict[str, int] = {}
    for name in DAY18_STAGE_COUNTER_FIELDS:
        value = _counter_from_report(summary, runtime, name)
        if value is None or value < 0:
            return None
        result[name] = value
    return result


def _historical_stage_accounting(
    report: dict[str, Any],
    *,
    previous_report: Mapping[str, Any] | None,
) -> None:
    """Rebuild stage-local rates without changing cumulative run semantics.

    Day 18 artifacts created before stage counter snapshots existed contain the
    cumulative counters and stage wall-clock needed for an unambiguous rebuild,
    provided the previous stage for the same algorithm/seed is available.
    """

    summary_raw = report.get("summary")
    if not isinstance(summary_raw, Mapping) or not summary_raw:
        return
    summary = dict(summary_raw)
    runtime_raw = summary.get("runtime")
    runtime = dict(runtime_raw) if isinstance(runtime_raw, Mapping) else {}
    report_runtime = report.get("runtime")
    if isinstance(report_runtime, Mapping):
        runtime.update(dict(report_runtime))
    current = _current_stage_counters(summary, runtime)
    elapsed = _number(runtime.get("wall_clock_seconds"))
    total_steps = _integer_value(
        summary.get("total_transitions", summary.get("total_steps", runtime.get("training_steps")))
    )
    stage_start_step = _integer_value(
        runtime.get("stage_start_step", summary.get("stage_start_step"))
    )
    current_physical = _integer_value(
        runtime.get("physical_environment_steps", summary.get("physical_environment_steps"))
    )
    if current_physical is None and total_steps is not None:
        current_physical = total_steps
    if current is None or elapsed is None or elapsed <= 0 or total_steps is None:
        return

    stage_start = _stage_start_from_report(summary, runtime)
    source = "runtime_stage_start_snapshot"
    if stage_start is None:
        previous_summary = (
            previous_report.get("summary") if previous_report is not None else None
        )
        previous_runtime = (
            previous_report.get("runtime")
            if previous_report is not None
            else None
        )
        previous_summary = (
            previous_summary if isinstance(previous_summary, Mapping) else {}
        )
        previous_runtime = (
            previous_runtime if isinstance(previous_runtime, Mapping) else {}
        )
        previous_counters = _current_stage_counters(
            previous_summary,
            previous_runtime,
        )
        previous_steps = _integer_value(
            previous_summary.get(
                "total_transitions",
                previous_summary.get("total_steps", previous_runtime.get("training_steps")),
            )
        )
        previous_physical = _integer_value(
            previous_runtime.get(
                "physical_environment_steps",
                previous_summary.get("physical_environment_steps", previous_steps),
            )
        )
        if previous_counters is not None and previous_steps is not None:
            if previous_physical is None:
                previous_physical = previous_steps
            stage_start = {
                "global_step": previous_steps,
                "physical_environment_steps": previous_physical,
                **previous_counters,
            }
            source = "previous_stage_cumulative_counters"
        elif stage_start_step == 0:
            stage_start = {
                "global_step": 0,
                "physical_environment_steps": 0,
                **{name: 0 for name in DAY18_STAGE_COUNTER_FIELDS},
            }
            source = "fresh_run_zero_baseline"

    if stage_start is None or stage_start_step is None or current_physical is None:
        return
    if stage_start["global_step"] != stage_start_step:
        return
    if stage_start["global_step"] > total_steps or stage_start["physical_environment_steps"] > current_physical:
        return
    if any(
        current[name] < stage_start[name] for name in DAY18_STAGE_COUNTER_FIELDS
    ):
        return

    stage_steps = total_steps - stage_start["global_step"]
    stage_physical_steps = current_physical - stage_start["physical_environment_steps"]
    stage_counters = {
        name: current[name] - stage_start[name]
        for name in DAY18_STAGE_COUNTER_FIELDS
    }
    config = report.get("config")
    if not isinstance(config, Mapping):
        config = report.get("training_config")
    if not isinstance(config, Mapping):
        config = {}
    rates = {
        "steps_per_second": float(stage_steps / elapsed),
        "environment_transitions_per_second": float(stage_steps / elapsed),
        "physical_environment_steps_per_second": float(
            stage_physical_steps / elapsed
        ),
        "vector_iterations_per_second": float(
            stage_counters["vector_iterations"] / elapsed
        ),
        "action_inference_batches_per_second": float(
            stage_counters["action_inference_batches"] / elapsed
        ),
        "action_inference_transitions_per_second": float(
            stage_counters["action_inference_transitions"] / elapsed
        ),
        "replay_insertion_calls_per_second": float(
            stage_counters["replay_insertion_calls"] / elapsed
        ),
        "replay_insertion_transitions_per_second": float(
            stage_counters["replay_insertion_transitions"] / elapsed
        ),
        "optimizer_updates_per_second": float(
            stage_counters["optimizer_updates"]
            / elapsed
        ),
        "training_samples_per_second": float(
            stage_counters["optimizer_updates"]
            * int(config.get("batch_size", 0) or 0)
            / elapsed
        ),
    }
    accounting = {
        "schema_version": 1,
        "counter_semantics": "stage-local counter delta divided by stage-local wall-clock elapsed",
        "stage_start_source": source,
        "stage_start_counters": stage_start,
        "stage_counters": stage_counters,
        "stage_elapsed_seconds": float(elapsed),
        "reconstructed_from_existing_artifacts": True,
    }
    summary["stage_start_counters"] = stage_start
    summary["stage_counters"] = stage_counters
    summary["stage_rates"] = rates
    summary["throughput_accounting"] = accounting
    for field, value in rates.items():
        summary[field] = value
    runtime["stage_start_counters"] = stage_start
    runtime["stage_counters"] = stage_counters
    runtime["stage_rates"] = rates
    runtime["throughput_accounting"] = accounting
    for field, value in rates.items():
        runtime[field] = value
    summary["runtime"] = runtime
    report["summary"] = summary
    report["runtime"] = runtime


def normalize_training_stage_accounting(
    reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize vectorized stage rates while preserving cumulative counters."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(
            (str(report.get("algorithm")), int(report.get("training_seed", -1))),
            [],
        ).append(report)
    stage_order = {stage: index for index, stage in enumerate(DAY18_MILESTONES)}
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                stage_order.get(str(item.get("stage")), len(stage_order)),
                int(item.get("target_transitions", 0)),
            ),
        )
        previous: dict[str, Any] | None = None
        for report in ordered:
            _historical_stage_accounting(report, previous_report=previous)
            if isinstance(report.get("summary"), Mapping) and report.get("summary"):
                previous = report
    return list(reports)


def compact_training_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep manifest summaries compact while retaining corrected runtime data."""

    fields = (
        "status",
        "total_transitions",
        "episodes",
        "optimizer_updates",
        "steps_per_second",
        "stage_start_counters",
        "stage_counters",
        "stage_rates",
        "throughput_accounting",
        "runtime",
        "resume_provenance",
    )
    return {field: summary[field] for field in fields if field in summary}


def day18_source_hashes(repository_root: str | Path) -> dict[str, str | None]:
    """Hash the source/config files that define or interpret Day 18 evidence."""

    root = Path(repository_root).resolve()
    return {
        path: sha256_file(root / path) if (root / path).is_file() else None
        for path in DAY18_PROVENANCE_SOURCE_PATHS
    }


def historical_run_provenance(
    training: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe what can and cannot be recovered from historical run metadata."""

    commits = sorted(
        {
            str(entry.get("runtime", {}).get("git_commit_sha"))
            for entry in training
            if isinstance(entry.get("runtime"), Mapping)
            and entry.get("runtime", {}).get("git_commit_sha")
            not in (None, "", "unavailable")
        }
    )
    return {
        "run_base_git_commit": commits[0] if len(commits) == 1 else None,
        "run_base_git_commits": commits,
        "git_dirty_at_run": "unknown",
        "git_diff_sha256_at_run": "unavailable",
        "limitation": (
            "Historical Day 18 runs recorded git_commit_sha but not working-tree "
            "cleanliness or tracked diff bytes; the uncommitted source state at "
            "run time cannot be recovered without fabricating provenance."
        ),
    }


def _summary_for_entry(
    manifest_path: Path,
    entry: Mapping[str, Any],
    *,
    include_metrics: bool = True,
    require_checkpoint: bool = True,
) -> dict[str, Any]:
    raw_run_dir = entry.get("run_dir")
    run_dir = (
        resolve_manifest_reference(manifest_path, raw_run_dir)
        if raw_run_dir
        else None
    )
    summary: dict[str, Any] = {}
    config: dict[str, Any] = dict(entry.get("training_config", {}))
    runtime: dict[str, Any] = {}
    failure: dict[str, Any] = {}
    metrics: list[dict[str, str]] = []
    if run_dir is not None:
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        failure_path = run_dir / "failure.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
        if config_path.is_file():
            run_config = _read_json(config_path)
            planned_config = entry.get("training_config", {})
            if isinstance(planned_config, Mapping):
                config = {
                    str(name): run_config.get(name)
                    for name in planned_config
                    if name in run_config
                }
            raw_runtime = run_config.get("runtime")
            if isinstance(raw_runtime, Mapping):
                runtime = dict(raw_runtime)
        if isinstance(summary.get("runtime"), Mapping):
            runtime.update(dict(summary["runtime"]))
        if failure_path.is_file():
            failure = _read_json(failure_path)
        if include_metrics:
            metrics = _read_metrics(run_dir)
    status = str(entry.get("status", "pending"))
    summary_status = str(summary.get("status", failure.get("status", status)))
    target = int(entry["target_transitions"])
    completed_steps = int(
        max(
            (_number(row.get("global_step")) or 0.0 for row in metrics),
            default=_number(summary.get("total_transitions")) or 0.0,
        )
    )
    if summary.get("total_transitions") is not None:
        completed_steps = max(completed_steps, int(summary["total_transitions"]))
    actual_completed = (
        summary_status == "completed"
        and completed_steps == target
        and bool(summary.get("total_transitions", summary.get("total_steps")) == target)
    )
    contract = runtime.get("environment_contract")
    if contract is None and isinstance(summary.get("environment_contract"), Mapping):
        contract = summary["environment_contract"]
    reasons: list[str] = []
    if summary_status != "completed":
        reasons.append(f"summary status is {summary_status}")
    if completed_steps != target:
        reasons.append(f"completed transitions {completed_steps} != target {target}")
    if not bool(summary.get("total_transitions", summary.get("total_steps")) == target):
        reasons.append("summary does not prove the target transition budget")
    if run_dir is None or not (run_dir / "metrics.csv").is_file():
        reasons.append("metrics.csv is missing")
    checkpoint_path: Path | None = None
    raw_checkpoint = entry.get("checkpoint")
    if isinstance(raw_checkpoint, Mapping) and raw_checkpoint.get("path"):
        checkpoint_path = resolve_manifest_reference(manifest_path, raw_checkpoint["path"])
    elif run_dir is not None:
        candidates = sorted((run_dir / "checkpoints").glob("step-*.pt"))
        if candidates:
            checkpoint_path = candidates[-1]
    if require_checkpoint and (checkpoint_path is None or not checkpoint_path.is_file()):
        reasons.append("checkpoint is missing")
    if not isinstance(contract, Mapping):
        reasons.append("environment contract provenance is missing")
    elif not all(contract.get(field) for field in ("contract_id", "contract_path", "contract_sha256")):
        reasons.append("environment contract identity is incomplete")
    eligible = actual_completed and not reasons
    return {
        "run_id": entry.get("run_id"),
        "pair_id": entry.get("pair_id"),
        "algorithm": entry["algorithm"],
        "training_seed": int(entry["training_seed"]),
        "stage": entry["stage"],
        "target_transitions": target,
        "status": "completed" if eligible else (summary_status or status),
        "eligible": eligible,
        "eligibility_reasons": reasons,
        "run_dir": (
            relative_path(run_dir, start=manifest_path.parent)
            if run_dir is not None
            else None
        ),
        "checkpoint": (
            {
                "path": relative_path(checkpoint_path, start=manifest_path.parent),
                "sha256": sha256_file(checkpoint_path),
            }
            if checkpoint_path is not None and checkpoint_path.is_file()
            else None
        ),
        "config": config,
        "summary": summary,
        "runtime": runtime,
        "environment_contract": dict(contract) if isinstance(contract, Mapping) else None,
        "metrics": metrics,
        "failure": failure,
    }


def load_training_entries(
    manifest_path: str | Path,
    *,
    include_metrics: bool = True,
    require_checkpoint: bool = True,
) -> list[dict[str, Any]]:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    reports = [
        _summary_for_entry(
            source,
            entry,
            include_metrics=include_metrics,
            require_checkpoint=require_checkpoint,
        )
        for entry in manifest["runs"]
    ]
    normalize_training_stage_accounting(reports)
    if not include_metrics:
        for report in reports:
            report.pop("metrics", None)
    return reports


def _contract_matches(
    value: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(value.get(field) == expected.get(field) for field in (
        "contract_id",
        "contract_path",
        "contract_sha256",
    ))


def _eval_summary(
    manifest_path: Path,
    training_entry: Mapping[str, Any],
    evaluation_entry: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
    episodes_per_seed: int,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw_results = evaluation_entry.get("results")
    results_path = (
        resolve_manifest_reference(manifest_path, raw_results)
        if isinstance(raw_results, str)
        else None
    )
    reasons: list[str] = []
    payload: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if results_path is None or not results_path.is_file():
        reasons.append("evaluation results.json is missing")
    else:
        try:
            payload = read_evaluation_results(results_path)
            rows = validate_episode_rows(
                payload,
                source=results_path,
                expected_seeds=expected_seeds,
                expected_episodes_per_seed=episodes_per_seed,
                require_complete=True,
            )
            computed = summary_from_episode_rows(rows)
            validate_embedded_summary(payload, computed, source=results_path)
        except (TypeError, ValueError, FileNotFoundError) as error:
            reasons.append(str(error))
    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_contract = metadata.get(
        "evaluation_contract_provenance",
        metadata.get("evaluation_contract"),
    )
    if not _contract_matches(raw_contract, expected_contract):
        reasons.append("evaluation Contract v2 provenance does not match manifest")
    if payload.get("evaluation_epsilon") != 0.0:
        reasons.append("evaluation epsilon is not zero")
    if payload.get("requested_device") not in {"cuda", "cuda:0"}:
        reasons.append("evaluation did not request CUDA")
    if not str(payload.get("resolved_device", "")).startswith("cuda"):
        reasons.append("evaluation did not resolve to CUDA")
    training = payload.get("training", {})
    try:
        evaluation_training_seed = int(training.get("training_seed", -1))
    except (AttributeError, TypeError, ValueError):
        evaluation_training_seed = -1
    if not isinstance(training, Mapping) or evaluation_training_seed != int(
        training_entry["training_seed"]
    ):
        reasons.append("evaluation training seed does not match training entry")
    checkpoint = payload.get("checkpoint", {})
    try:
        evaluation_checkpoint_step = int(checkpoint.get("step", -1))
    except (AttributeError, TypeError, ValueError):
        evaluation_checkpoint_step = -1
    if not isinstance(checkpoint, Mapping) or evaluation_checkpoint_step != int(
        training_entry["target_transitions"]
    ):
        reasons.append("evaluation checkpoint step does not match training milestone")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["evaluation_seed"]), []).append(
            float(row["episode_return"])
        )
    seed_groups = {
        str(seed): {
            "count": len(values),
            "mean_return": float(fmean(values)),
            "median_return": float(median(values)),
            "std_return": float(pstdev(values)),
        }
        for seed, values in sorted(grouped.items())
    }
    returns = [float(row["episode_return"]) for row in rows]
    eligible = bool(rows) and not reasons
    return {
        "run_id": training_entry.get("run_id"),
        "algorithm": training_entry["algorithm"],
        "training_seed": int(training_entry["training_seed"]),
        "stage": training_entry["stage"],
        "target_transitions": int(training_entry["target_transitions"]),
        "eligible": eligible,
        "eligibility_reasons": reasons,
        "results": (
            {
                "path": relative_path(results_path, start=manifest_path.parent),
                "sha256": sha256_file(results_path),
            }
            if results_path is not None and results_path.is_file()
            else None
        ),
        "evaluation_seeds": list(expected_seeds),
        "episodes_per_seed": episodes_per_seed,
        "seed_groups": seed_groups,
        "summary": (
            {
                "count": len(returns),
                "mean_return": float(fmean(returns)),
                "median_return": float(median(returns)),
                "std_return": float(pstdev(returns)),
                "min_return": float(min(returns)),
                "max_return": float(max(returns)),
            }
            if returns
            else None
        ),
        "payload": payload,
    }


def load_evaluation_entries(
    manifest_path: str | Path,
    training_entries: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    protocol = manifest.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("Day 18 manifest protocol is missing")
    expected_seeds = tuple(int(seed) for seed in protocol.get("evaluation_seeds", ()))
    episodes_per_seed = int(protocol.get("episodes_per_evaluation_seed", 0))
    contract_ref = manifest.get("source_of_truth", {}).get("contract", {})
    expected_contract = contract_ref if isinstance(contract_ref, Mapping) else {}
    expected_contract = {
        "contract_id": expected_contract.get("contract_id"),
        "contract_path": expected_contract.get("contract_path"),
        "contract_sha256": expected_contract.get("sha256")
        or expected_contract.get("contract_sha256"),
    }
    trainings = list(training_entries or load_training_entries(source, include_metrics=False))
    by_key = {
        (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"])): entry
        for entry in trainings
    }
    reports: list[dict[str, Any]] = []
    for raw_entry in manifest["runs"]:
        training_entry = by_key[
            (
                str(raw_entry["algorithm"]),
                int(raw_entry["training_seed"]),
                str(raw_entry["stage"]),
            )
        ]
        evaluation = raw_entry.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            evaluation = {}
        reports.append(
            _eval_summary(
                source,
                training_entry,
                evaluation,
                expected_seeds=expected_seeds,
                episodes_per_seed=episodes_per_seed,
                expected_contract=expected_contract,
            )
        )
    return reports


def _q_summary(
    manifest_path: Path,
    training_entry: Mapping[str, Any],
    raw_path: Any,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = (
        resolve_manifest_reference(manifest_path, raw_path)
        if isinstance(raw_path, str)
        else None
    )
    reasons: list[str] = []
    payload: dict[str, Any] = {}
    values: list[list[float]] = []
    if path is None or not path.is_file():
        reasons.append("Q probe artifact is missing")
    else:
        payload = _read_json(path)
        analysis = payload.get("analysis")
        if not isinstance(analysis, Mapping) or not isinstance(analysis.get("q_values"), list):
            reasons.append("Q probe artifact is missing analysis.q_values")
        else:
            try:
                values = [
                    [_finite_float(value, name="q_value") for value in row]
                    for row in analysis["q_values"]
                ]
            except ValueError as error:
                reasons.append(str(error))
    checkpoint_payload = payload.get("checkpoint")
    contract = (
        checkpoint_payload.get("environment_contract")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    if not _contract_matches(contract, expected_contract):
        reasons.append("Q probe Contract v2 provenance does not match manifest")
    checkpoint_step = (
        checkpoint_payload.get("training_steps")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    if checkpoint_step != training_entry["target_transitions"]:
        reasons.append("Q probe checkpoint step does not match training milestone")
    if not values:
        stats = None
    else:
        flat = [value for row in values for value in row]
        max_values = [max(row) for row in values if row]
        stats = {
            "probe_count": len(values),
            "action_count": len(values[0]) if values[0] else 0,
            "q_mean": float(fmean(flat)),
            "q_std": float(pstdev(flat)) if len(flat) > 1 else 0.0,
            "q_min": min(flat),
            "q_max": max(flat),
            "max_q_mean": float(fmean(max_values)),
            "max_q_std": float(pstdev(max_values)) if len(max_values) > 1 else 0.0,
        }
    return {
        "run_id": training_entry.get("run_id"),
        "algorithm": training_entry["algorithm"],
        "training_seed": training_entry["training_seed"],
        "stage": training_entry["stage"],
        "target_transitions": training_entry["target_transitions"],
        "eligible": not reasons and stats is not None,
        "eligibility_reasons": reasons,
        "path": (
            {
                "path": relative_path(path, start=manifest_path.parent),
                "sha256": sha256_file(path),
            }
            if path is not None and path.is_file()
            else None
        ),
        "summary": stats,
        "q_values": values,
        "payload": payload,
    }


def load_q_probe_entries(
    manifest_path: str | Path,
    training_entries: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    contract_ref = manifest.get("source_of_truth", {}).get("contract", {})
    expected_contract = contract_ref if isinstance(contract_ref, Mapping) else {}
    expected_contract = {
        "contract_id": expected_contract.get("contract_id"),
        "contract_path": expected_contract.get("contract_path"),
        "contract_sha256": expected_contract.get("sha256")
        or expected_contract.get("contract_sha256"),
    }
    trainings = list(training_entries or load_training_entries(source, include_metrics=False))
    by_key = {
        (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"])): entry
        for entry in trainings
    }
    reports: list[dict[str, Any]] = []
    for raw_entry in manifest["runs"]:
        key = (
            str(raw_entry["algorithm"]),
            int(raw_entry["training_seed"]),
            str(raw_entry["stage"]),
        )
        reports.append(
            _q_summary(
                source,
                by_key[key],
                raw_entry.get("q_probe"),
                expected_contract=expected_contract,
            )
        )
    return reports


def _pair_control_conditions(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = [entry for entry in entries if entry.get("eligible")]
    requested = [
        str(entry.get("runtime", {}).get("requested_device", entry.get("config", {}).get("device", "unavailable")))
        for entry in completed
    ]
    resolved = [str(entry.get("runtime", {}).get("resolved_device", "unavailable")) for entry in completed]
    precisions = [str(entry.get("runtime", {}).get("precision", entry.get("config", {}).get("precision", "unavailable"))) for entry in completed]
    contracts = [entry.get("environment_contract") for entry in completed]
    config_diffs: list[dict[str, Any]] = []
    by_pair_stage: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for entry in entries:
        by_pair_stage.setdefault(
            (int(entry["training_seed"]), str(entry["stage"])),
            [],
        ).append(entry)
    for key, pair_entries in sorted(by_pair_stage.items()):
        if len(pair_entries) != 2:
            config_diffs.append({"pair": key, "error": "missing algorithm pair"})
            continue
        first, second = pair_entries
        diff = config_diff(first.get("config", {}), second.get("config", {}))
        if diff:
            config_diffs.append({"pair": key, "diff": diff})
    return {
        "completed_run_count": len(completed),
        "same_requested_device": bool(requested) and len(set(requested)) == 1,
        "same_resolved_device": bool(resolved) and len(set(resolved)) == 1,
        "same_precision": bool(precisions) and len(set(precisions)) == 1,
        "requested_devices": requested,
        "resolved_devices": resolved,
        "precisions": precisions,
        "same_contract": bool(contracts) and all(
            contract == contracts[0] for contract in contracts[1:]
        ),
        "pair_config_diffs": config_diffs,
        "only_algorithm_varies_within_pair": not config_diffs,
    }


def _completed_keys(
    entries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> set[tuple[str, int]]:
    return {
        (str(entry["algorithm"]), int(entry["training_seed"]))
        for entry in entries
        if entry.get("stage") == stage and entry.get("eligible")
    }


def _paired_evaluation_rows(
    evaluation_entries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [
        entry
        for entry in evaluation_entries
        if entry.get("stage") == stage and entry.get("eligible")
    ]
    by_key = {
        (str(entry["algorithm"]), int(entry["training_seed"])): entry
        for entry in selected
    }
    rows: list[dict[str, Any]] = []
    for seed in DAY18_TRAINING_SEEDS:
        dqn = by_key.get(("dqn", seed))
        double = by_key.get(("double_dqn", seed))
        if dqn is None or double is None:
            continue
        dqn_mean = float(dqn["summary"]["mean_return"])
        double_mean = float(double["summary"]["mean_return"])
        rows.append(
            {
                "training_seed": seed,
                "dqn_mean_return": dqn_mean,
                "double_dqn_mean_return": double_mean,
                "dqn_minus_double_dqn": dqn_mean - double_mean,
            }
        )
    return rows, selected


def _conclusion(
    *,
    main_complete: bool,
    stability_failure: bool,
    paired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if stability_failure:
        return {
            "code": "E",
            "label": "correctness_or_stability_failure",
            "statement": "至少一個正式 run 發生 correctness/stability failure，不能做演算法 ranking。",
            "evidence_strength": "failed",
        }
    if not main_complete or len(paired_rows) < len(DAY18_TRAINING_SEEDS):
        return {
            "code": "D",
            "label": "500k_insufficient_or_incomplete",
            "statement": "目前沒有完整的三組 500K paired CUDA evidence，尚不足以可靠分辨兩個演算法。",
            "evidence_strength": "incomplete",
        }
    differences = [float(row["dqn_minus_double_dqn"]) for row in paired_rows]
    mean_difference = float(fmean(differences))
    spread = float(pstdev(differences)) if len(differences) > 1 else 0.0
    if all(value > 0.0 for value in differences):
        code = "A"
        label = "dqn_stronger_at_500k"
        statement = "500K 下 DQN 在三個 paired training seeds 的 evaluation mean 都高於 Double DQN。"
    elif all(value < 0.0 for value in differences):
        code = "B"
        label = "double_dqn_stronger_at_500k"
        statement = "500K 下 Double DQN 在三個 paired training seeds 的 evaluation mean 都高於 DQN。"
    elif abs(mean_difference) <= spread:
        code = "C"
        label = "seed_variance_covers_effect"
        statement = "兩個演算法的差異小於或等於 paired seed differences 的 spread，seed variance 蓋過目前的 algorithm effect。"
    else:
        code = "D"
        label = "500k_insufficient_to_distinguish"
        statement = "500K 下仍沒有跨 seed 一致的方向，尚不足以可靠分辨兩個演算法。"
    return {
        "code": code,
        "label": label,
        "statement": statement,
        "evidence_strength": "three_paired_cuda_seeds_at_500k",
        "paired_mean_difference_dqn_minus_double_dqn": mean_difference,
        "paired_difference_population_std": spread,
        "decision_rule": "A/B require all three paired differences to share a strict sign; C applies when abs(mean difference) <= population std; otherwise D.",
    }


def _random_baseline(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw_path = manifest.get("random_baseline_results")
    if raw_path is None:
        return None
    path = resolve_manifest_reference(manifest_path, raw_path)
    if not path.is_file():
        return {
            "path": relative_path(path, start=manifest_path.parent),
            "available": False,
            "reason": "random baseline artifact is missing",
        }
    payload = read_evaluation_results(path)
    if payload.get("policy_type") != "random":
        raise ValueError("Day 18 random baseline must have policy_type=random")
    protocol = manifest.get("protocol", {})
    rows = validate_episode_rows(
        payload,
        source=path,
        expected_seeds=protocol.get("evaluation_seeds"),
        expected_episodes_per_seed=int(protocol.get("episodes_per_evaluation_seed", 0)),
        require_complete=True,
    )
    computed = summary_from_episode_rows(rows)
    validate_embedded_summary(payload, computed, source=path)
    source_contract = manifest.get("source_of_truth", {}).get("contract", {})
    metadata = payload.get("metadata", {})
    baseline_contract = (
        metadata.get("evaluation_contract")
        if isinstance(metadata, Mapping)
        else None
    )
    expected_contract_id = (
        source_contract.get("contract_id")
        if isinstance(source_contract, Mapping)
        else None
    )
    if not isinstance(baseline_contract, Mapping) or baseline_contract.get(
        "contract_id"
    ) != expected_contract_id:
        raise ValueError("Day 18 random baseline is not the Contract v2 baseline")
    return {
        "path": relative_path(path, start=manifest_path.parent),
        "sha256": sha256_file(path),
        "policy_type": payload.get("policy_type"),
        "summary": computed,
        "contract": baseline_contract,
    }


def build_day18_report(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    training = load_training_entries(source, include_metrics=False)
    evaluations = load_evaluation_entries(source, training)
    q_probe = load_q_probe_entries(source, training)
    formal_stage = str(
        manifest.get("protocol", {}).get("formal_quality_horizon", DAY18_FORMAL_STAGE)
    )
    main_training_keys = _completed_keys(training, stage=formal_stage)
    expected_main_keys = {
        (algorithm, seed)
        for algorithm in DAY18_ALGORITHMS
        for seed in DAY18_TRAINING_SEEDS
    }
    paired_rows, main_evaluations = _paired_evaluation_rows(
        evaluations,
        stage=formal_stage,
    )
    paired_rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage in DAY18_MILESTONES:
        stage_rows, _stage_evaluations = _paired_evaluation_rows(
            evaluations,
            stage=stage,
        )
        paired_rows_by_stage[stage] = stage_rows
    main_evaluation_keys = {
        (str(entry["algorithm"]), int(entry["training_seed"]))
        for entry in main_evaluations
    }
    main_q_keys = {
        (str(entry["algorithm"]), int(entry["training_seed"]))
        for entry in q_probe
        if entry.get("stage") == formal_stage and entry.get("eligible")
    }
    stability_failure = any(
        entry.get("summary", {}).get("status") == "failed_non_finite"
        or str(entry.get("status")) in {"failed_non_finite", "failed"}
        for entry in training
    )
    conditions = _pair_control_conditions(training)
    formal_cuda = (
        main_training_keys == expected_main_keys
        and main_evaluation_keys == expected_main_keys
        and main_q_keys == expected_main_keys
        and conditions["same_requested_device"]
        and conditions["same_resolved_device"]
        and conditions["same_precision"]
        and conditions["same_contract"]
        and conditions["only_algorithm_varies_within_pair"]
        and all(device.startswith("cuda") for device in conditions["resolved_devices"])
        and all(device in {"cuda", "cuda:0"} for device in conditions["requested_devices"])
    )
    conditions.update(
        {
            "formal_cuda_eligible": formal_cuda,
            "main_training_complete": main_training_keys == expected_main_keys,
            "main_evaluation_complete": main_evaluation_keys == expected_main_keys,
            "main_q_probe_complete": main_q_keys == expected_main_keys,
            "formal_quality_eligible": formal_cuda and not stability_failure,
            "quality_budget": "actual accepted environment transitions",
            "screening_and_pilot_are_not_final_selection": True,
            "quality_and_engineering_cost_are_separate": True,
        }
    )
    engineering_runs = [
        {
            "algorithm": entry["algorithm"],
            "training_seed": entry["training_seed"],
            "stage": entry["stage"],
            "eligible": entry["eligible"],
            "stage_start_counters": entry.get("summary", {}).get(
                "stage_start_counters"
            ),
            "stage_counters": entry.get("summary", {}).get("stage_counters"),
            "stage_rates": {
                field: entry.get("runtime", {}).get(field)
                for field in (
                    "steps_per_second",
                    "environment_transitions_per_second",
                    "physical_environment_steps_per_second",
                    *DAY18_STAGE_RATE_FIELDS,
                )
            },
            "throughput_accounting": entry.get("summary", {}).get(
                "throughput_accounting"
            ),
            "steps_per_second": entry.get("runtime", {}).get(
                "steps_per_second",
                entry.get("summary", {}).get("steps_per_second"),
            ),
            "wall_clock_seconds": entry.get("runtime", {}).get("wall_clock_seconds"),
            "peak_allocated_vram_bytes": entry.get("runtime", {}).get(
                "cuda_peak_allocated_bytes"
            ),
            "peak_reserved_vram_bytes": entry.get("runtime", {}).get(
                "cuda_peak_reserved_bytes"
            ),
            "gpu_utilization_percent": entry.get("runtime", {}).get(
                "gpu_utilization_percent"
            ),
            "gpu_model": entry.get("runtime", {}).get("gpu_model"),
            "resolved_device": entry.get("runtime", {}).get("resolved_device"),
        }
        for entry in training
        if entry.get("stage") == formal_stage
    ]
    report = {
        "schema_version": DAY18_SCHEMA_VERSION,
        "generated_at_utc": utc_timestamp(),
        "question": "在相同 Day 16 vectorized backend、Contract v2、actual transition budget、paired training seeds 與 evaluation protocol 下，DQN 與 Double DQN 到 250K/500K 是否出現跨 seed 可重複差異？",
        "manifest": relative_path(source, start=Path.cwd()),
        "manifest_sha256": sha256_file(source),
        "manifest_status": manifest.get("status"),
        "protocol": manifest.get("protocol"),
        "source_of_truth": manifest.get("source_of_truth"),
        "provenance": {
            **historical_run_provenance(training),
            "source_hashes": day18_source_hashes(source.parent.parent.parent),
        },
        "training": {
            "entries": training,
            "completed_entry_count": sum(bool(entry.get("eligible")) for entry in training),
        },
        "evaluation": {
            "entries": evaluations,
            "completed_entry_count": sum(bool(entry.get("eligible")) for entry in evaluations),
            "paired_seed_rows": paired_rows,
            "paired_seed_rows_by_stage": paired_rows_by_stage,
        },
        "q_probe": {
            "entries": q_probe,
            "completed_entry_count": sum(bool(entry.get("eligible")) for entry in q_probe),
        },
        "engineering_cost": {
            "stage": formal_stage,
            "runs": engineering_runs,
            "measurement_rule": "SPS and wall-clock are reported from each run runtime; VRAM/utilization remain null when the runtime could not measure them reliably.",
        },
        "random_baseline": _random_baseline(source, manifest),
        "comparison_conditions": conditions,
        "conclusion": _conclusion(
            main_complete=formal_cuda,
            stability_failure=stability_failure,
            paired_rows=paired_rows,
        ),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


__all__ = [
    "DAY18_ALGORITHMS",
    "DAY18_FORMAL_STAGE",
    "DAY18_MILESTONES",
    "DAY18_SCHEMA_VERSION",
    "DAY18_TRAINING_SEEDS",
    "DAY18_PROVENANCE_SOURCE_PATHS",
    "DAY18_STAGE_COUNTER_FIELDS",
    "DAY18_STAGE_RATE_FIELDS",
    "Day18ExperimentConfig",
    "build_day18_manifest",
    "build_day18_report",
    "compact_training_summary",
    "config_diff",
    "load_day18_config",
    "load_evaluation_entries",
    "load_q_probe_entries",
    "load_training_entries",
    "day18_source_hashes",
    "historical_run_provenance",
    "normalize_training_stage_accounting",
    "read_day18_manifest",
    "relative_path",
    "repository_path",
    "resolve_manifest_reference",
    "sha256_file",
    "utc_timestamp",
    "write_json",
]
