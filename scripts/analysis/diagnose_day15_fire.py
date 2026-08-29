"""Trace FIRE/TimeLimit behavior and run Day 15 diagnostic ablations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from breakout_env import make_breakout_env
from breakout_rl.day15_diagnostics import (
    EpisodeSpec,
    run_diagnostic_evaluation,
    write_diagnostic_artifacts,
)
from breakout_rl.evaluation import (
    load_day14_provenance,
    load_dqn_checkpoint,
    load_evaluation_config,
    validate_checkpoint_provenance,
)
from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summary_from_episode_rows,
    validate_episode_rows,
)
from breakout_rl.evaluation_contract import expand_concrete_episode_seeds


DEFAULT_OUTPUT_ROOT = Path("evaluations/day15-diagnostics")
DEFAULT_TRACE_SEED_COUNT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace FIRE/TimeLimit behavior without changing Day 15 v1 artifacts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda",
        help="explicit torch device; real diagnostic DQN runs require CUDA",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-day14-manifest", type=Path, default=None)
    parser.add_argument("--source-day14-profiling-report", type=Path, default=None)
    parser.add_argument(
        "--trace-seeds",
        type=int,
        nargs="+",
        default=None,
        help="concrete v1 episode seeds; default selects two timeouts and one termination",
    )
    return parser


def _is_cuda_request(device: str) -> bool:
    normalized = device.strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def _resolve_reference(config_value: str | None, config_path: Path) -> Path | None:
    if config_value is None:
        return None
    configured = Path(config_value)
    for candidate in (configured, Path.cwd() / configured, config_path.parent / configured):
        if candidate.is_file():
            return candidate
    return configured


def _episode_specs(config: Any) -> tuple[EpisodeSpec, ...]:
    concrete = expand_concrete_episode_seeds(
        config.seeds,
        episodes_per_seed=config.episodes_per_seed,
    )
    specs: list[EpisodeSpec] = []
    position = 0
    for evaluation_seed in config.seeds:
        for episode_index in range(1, config.episodes_per_seed + 1):
            specs.append(
                EpisodeSpec(
                    evaluation_seed=int(evaluation_seed),
                    episode_index=episode_index,
                    episode_seed=concrete[position],
                )
            )
            position += 1
    return tuple(specs)


def _select_trace_specs(
    specs: Sequence[EpisodeSpec],
    v1_results_path: Path,
    explicit_seeds: Sequence[int] | None,
) -> tuple[EpisodeSpec, ...]:
    by_concrete_seed = {spec.episode_seed: spec for spec in specs}
    if explicit_seeds is not None:
        if len(set(explicit_seeds)) != len(explicit_seeds):
            raise ValueError("--trace-seeds must be unique")
        try:
            selected = tuple(by_concrete_seed[int(seed)] for seed in explicit_seeds)
        except KeyError as error:
            raise ValueError(f"unknown --trace-seeds value: {error.args[0]}") from error
        return selected
    payload = read_evaluation_results(v1_results_path)
    rows = validate_episode_rows(payload, source=v1_results_path)
    timeout_rows = [row for row in rows if row["truncated"]]
    terminated_rows = [row for row in rows if row["terminated"]]
    if len(timeout_rows) < 2 or not terminated_rows:
        raise ValueError("v1 results do not contain two timeouts and one terminated episode")
    selected_seeds = [
        int(timeout_rows[0]["episode_seed"]),
        int(terminated_rows[0]["episode_seed"]),
        int(timeout_rows[1]["episode_seed"]),
    ]
    return tuple(by_concrete_seed[seed] for seed in selected_seeds)


def _v1_marker(policy_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": "day15-evaluation-v1",
        "policy_type": policy_type,
        "policy_responsible_fire": True,
        "fire_reset": False,
        "terminal_on_life_loss": False,
        "evaluation_epsilon": 0.0,
        "raw_reward_rule": "sum environment rewards without clipping",
        "results_artifact": "results.json",
        "legacy_reason": "preserved before FIRE/TimeLimit root-cause audit",
    }


def _write_v1_marker(directory: Path, policy_type: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract-v1.json"
    path.write_text(
        json.dumps(_v1_marker(policy_type), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _write_v1_time_limit_summary(
    source_path: Path,
    *,
    output_path: Path,
    trace_path: Path,
    agent_step_limit: int,
) -> Path:
    payload = read_evaluation_results(source_path)
    rows = validate_episode_rows(payload, source=source_path)
    for row in rows:
        row["time_limit"] = bool(row["truncated"])
        row["stop_reason"] = "time_limit" if row["time_limit"] else row["stop_reason"]
    summary = summary_from_episode_rows(rows)
    output = {
        "schema_version": 2,
        "contract_id": "day15-evaluation-v1-time-limit-summary",
        "legacy_contract": "day15-evaluation-v1",
        "source_results": source_path.as_posix(),
        "time_limit_detection": {
            "rule": "classify legacy truncated rows as ALE TimeLimit after the root trace observed ale.game_truncated",
            "source_trace": trace_path.as_posix(),
            "agent_step_limit": agent_step_limit,
            "truncated_rows_classified_as_time_limit": True,
        },
        "concrete_episode_seeds": [int(row["episode_seed"]) for row in rows],
        "summary": summary,
        "per_episode": rows,
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    if not _is_cuda_request(args.device):
        raise ValueError("real Day 15 DQN diagnostics require an explicit CUDA device")
    config = load_evaluation_config(args.config)
    manifest_path = args.source_day14_manifest or _resolve_reference(
        config.source_day14_manifest,
        args.config,
    )
    if manifest_path is None:
        raise ValueError("Day 14 manifest is required")
    profiling_path = args.source_day14_profiling_report or _resolve_reference(
        config.source_day14_profiling_report,
        args.config,
    )
    provenance = load_day14_provenance(
        manifest_path,
        profiling_report_path=profiling_path,
    )
    gate = provenance.get("day14_gate", {})
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        raise ValueError(f"Day 14 Gate A is not passed: {gate}")
    loaded = load_dqn_checkpoint(
        args.checkpoint,
        device=args.device,
        source_day14_manifest=manifest_path,
    )
    validate_checkpoint_provenance(
        loaded.checkpoint_metadata,
        loaded.training_metadata,
        provenance,
    )
    training = {
        **dict(loaded.training_metadata),
        "source_of_truth": provenance["source_of_truth"],
        "source_day14_manifest": provenance["manifest_path"],
        "config_reference": provenance.get("config_reference"),
        "selection_rule": provenance["selection_rule"],
        "selection_rationale": provenance.get("selection_rationale"),
        "day14_experiment_id": provenance.get("experiment_id"),
        "day14_run_artifact_dir": provenance.get("run_dir"),
        "trainer_runtime": provenance.get("runtime", {}),
        "gpu_profiling_summary": provenance.get("gpu_profiling_summary", {}),
        "source_day14_profiling_report": provenance.get(
            "source_day14_profiling_report"
        ),
        "day14_gate": gate,
    }
    specs = _episode_specs(config)
    v1_results_path = Path("evaluations/day15-dqn-cuda/results.json")
    trace_specs = _select_trace_specs(specs, v1_results_path, args.trace_seeds)
    trace_seeds = [spec.episode_seed for spec in trace_specs]
    common_metadata = {
        "evaluation_config_path": args.config.as_posix(),
        "v1_results_path": v1_results_path.as_posix(),
        "trace_selection": [spec.__dict__ for spec in trace_specs],
        "source_day14_manifest": provenance["manifest_path"],
        "source_day14_profiling_report": provenance.get(
            "source_day14_profiling_report"
        ),
    }
    output_root = args.output_root
    root_payload = run_diagnostic_evaluation(
        loaded.model,
        env_factory=make_breakout_env,
        device=args.device,
        episode_specs=trace_specs,
        mode="v1",
        trace_seeds=trace_seeds,
        checkpoint=loaded.checkpoint_metadata,
        training=training,
        metadata={**common_metadata, "trace_purpose": "FIRE/TimeLimit root-cause trace"},
    )
    fire_payload = run_diagnostic_evaluation(
        loaded.model,
        env_factory=lambda: make_breakout_env(fire_reset=True),
        device=args.device,
        episode_specs=specs,
        mode="fire_assist",
        trace_seeds=trace_seeds,
        environment_fire_assist=True,
        checkpoint=loaded.checkpoint_metadata,
        training=training,
        metadata=common_metadata,
    )
    epsilon_payload = run_diagnostic_evaluation(
        loaded.model,
        env_factory=make_breakout_env,
        device=args.device,
        episode_specs=specs,
        mode="epsilon005",
        trace_seeds=trace_seeds,
        checkpoint=loaded.checkpoint_metadata,
        training=training,
        metadata=common_metadata,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    root_paths = write_diagnostic_artifacts(root_payload, output_root / "root-cause")
    fire_paths = write_diagnostic_artifacts(fire_payload, output_root / "fire-assist")
    epsilon_paths = write_diagnostic_artifacts(
        epsilon_payload,
        output_root / "epsilon005",
    )
    marker_paths = [
        _write_v1_marker(Path("evaluations/day15-random-baseline"), "random"),
        _write_v1_marker(Path("evaluations/day15-dqn-cuda"), "dqn"),
    ]
    agent_step_limit = int(
        next(
            int(item["time_limit_agent_step_limit"])
            for item in root_payload["per_episode"]
            if item.get("time_limit_agent_step_limit") is not None
        )
    )
    v1_summary_paths = [
        _write_v1_time_limit_summary(
            Path("evaluations/day15-random-baseline/results.json"),
            output_path=Path("evaluations/day15-random-baseline/time-limit-summary.json"),
            trace_path=root_paths[2],
            agent_step_limit=agent_step_limit,
        ),
        _write_v1_time_limit_summary(
            Path("evaluations/day15-dqn-cuda/results.json"),
            output_path=Path("evaluations/day15-dqn-cuda/time-limit-summary.json"),
            trace_path=root_paths[2],
            agent_step_limit=agent_step_limit,
        ),
    ]
    manifest = {
        "diagnostic_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_v1_artifacts": {
            "random_results": "evaluations/day15-random-baseline/results.json",
            "dqn_results": "evaluations/day15-dqn-cuda/results.json",
            "contract_markers": [path.as_posix() for path in marker_paths],
            "time_limit_summaries": [path.as_posix() for path in v1_summary_paths],
        },
        "concrete_episode_seeds": [spec.episode_seed for spec in specs],
        "trace_episode_seeds": trace_seeds,
        "modes": {
            "root_cause": [path.as_posix() for path in root_paths],
            "fire_assist": [path.as_posix() for path in fire_paths],
            "epsilon005": [path.as_posix() for path in epsilon_paths],
        },
        "checkpoint": loaded.checkpoint_metadata,
        "training": training,
        "day14_gate": gate,
    }
    manifest_path_out = output_root / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_diagnostics(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as error:
        print(f"Day 15 diagnostics failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
