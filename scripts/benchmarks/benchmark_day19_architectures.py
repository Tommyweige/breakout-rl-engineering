"""Compare standard and Dueling Double DQN CUDA smoke runtime costs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_env import make_breakout_vector_env
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import resolve_device
from breakout_rl.training.vectorized import VectorizedDQNTrainer


DEFAULT_CONTRACT = Path("configs/eval/breakout_contract_v2.json")
DEFAULT_OUTPUT = Path("assets/day19/dueling-smoke-runtime.json")
DEFAULT_REPORT = Path("reports/day19-dueling-smoke.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-root", type=Path, default=Path("runs/day19-architecture-smoke"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _run_one(
    architecture: str,
    *,
    args: argparse.Namespace,
    contract_id: str,
    contract_path: Path,
    environment_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = args.run_root / f"{architecture}-seed{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory already contains artifacts: {run_dir}; choose a new --run-root"
        )
    config = DQNConfig.day17_smoke(
        total_steps=args.total_steps,
        device=args.device,
        algorithm="double_dqn",
        architecture=architecture,
    ).with_overrides(
        contract_id=contract_id,
        contract_path=contract_path.as_posix(),
    )
    env = make_breakout_vector_env(
        config.num_envs,
        **dict(environment_kwargs),
    )
    started_at = time.perf_counter()
    try:
        trainer = VectorizedDQNTrainer(env, config, run_dir=run_dir)
        summary = trainer.train()
    finally:
        env.close()
    checkpoint_path = Path(summary["last_checkpoint"])
    reload_run_dir = args.run_root / f"{architecture}-seed{args.seed}-reload"
    reload_env = make_breakout_vector_env(
        config.num_envs,
        **dict(environment_kwargs),
    )
    reload_trainer: VectorizedDQNTrainer | None = None
    try:
        reload_trainer = VectorizedDQNTrainer(
            reload_env,
            config,
            run_dir=reload_run_dir,
            resume_from=checkpoint_path,
        )
        checkpoint_load = {
            "status": "passed",
            "checkpoint": checkpoint_path.as_posix(),
            "global_step": reload_trainer.global_step,
            "architecture": getattr(
                reload_trainer.online_network,
                "architecture",
                None,
            ),
        }
    finally:
        if reload_trainer is not None:
            reload_trainer.metrics.close()
        reload_env.close()
    runtime = summary.get("runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    return {
        "architecture": architecture,
        "run_dir": run_dir.as_posix(),
        "config": config.to_dict(),
        "summary": summary,
        "checkpoint_load": checkpoint_load,
        "runtime_metrics": {
            "parameter_count": summary["model_config"]["parameter_count"],
            "end_to_end_environment_sps": summary["environment_transitions_per_second"],
            "optimizer_updates_per_second": summary["optimizer_updates_per_second"],
            "training_samples_per_second": summary["training_samples_per_second"],
            "wall_clock_seconds": runtime.get("wall_clock_seconds"),
            "peak_allocated_vram_bytes": runtime.get("cuda_peak_allocated_bytes"),
            "peak_reserved_vram_bytes": runtime.get("cuda_peak_reserved_bytes"),
            "cuda_device_name": runtime.get("cuda_device_name"),
            "cuda_device_index": runtime.get("cuda_device_index"),
            "pytorch_version": runtime.get("pytorch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
        },
        "wall_clock_seconds_from_runner": time.perf_counter() - started_at,
    }


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _comparison(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_architecture = {str(result["architecture"]): result for result in results}
    standard = by_architecture["standard"]
    dueling = by_architecture["dueling"]
    standard_config = standard["config"]
    dueling_config = dueling["config"]
    comparable_fields = tuple(
        field
        for field in standard_config
        if field != "architecture"
    )
    same_training_config = all(
        standard_config.get(field) == dueling_config.get(field)
        for field in comparable_fields
    )
    standard_metrics = standard["runtime_metrics"]
    dueling_metrics = dueling["runtime_metrics"]
    standard_sps = _numeric(standard_metrics.get("end_to_end_environment_sps"))
    dueling_sps = _numeric(dueling_metrics.get("end_to_end_environment_sps"))
    return {
        "same_algorithm": standard_config.get("algorithm") == "double_dqn"
        and dueling_config.get("algorithm") == "double_dqn",
        "same_seed": standard_config.get("seed") == dueling_config.get("seed"),
        "same_transition_budget": standard_config.get("total_steps")
        == dueling_config.get("total_steps"),
        "same_training_config_except_architecture": same_training_config,
        "parameter_count_delta": (
            dueling_metrics.get("parameter_count", 0)
            - standard_metrics.get("parameter_count", 0)
        ),
        "dueling_to_standard_environment_sps_ratio": (
            dueling_sps / standard_sps
            if standard_sps and dueling_sps
            else None
        ),
        "both_completed": all(
            result.get("summary", {}).get("status") == "completed"
            for result in results
        ),
        "both_checkpoint_save_load_passed": all(
            result.get("checkpoint_load", {}).get("status") == "passed"
            for result in results
        ),
        "interpretation": (
            "This is an infrastructure and hot-path regression check. It does not "
            "rank policy quality or select a Day 20 winner."
        ),
    }


def _write_report(
    path: Path,
    *,
    payload: Mapping[str, Any],
) -> None:
    results = payload["results"]
    comparison = payload["comparison"]
    lines = [
        "# Day 19 Dueling Double DQN CUDA smoke",
        "",
        "這份 report 的問題是：加入 Dueling heads 後，既有的 CUDA/vectorized training hot path 是否仍能完成，且工程成本是否可量測？它不是 model-quality ranking。",
        "",
        f"- generated at: `{payload['generated_at_utc']}`",
        f"- device: `{payload['device']}`",
        f"- contract: `{payload['contract_path']}` (`{payload['contract']['contract_id']}`)",
        f"- seed: `{payload['protocol']['seed']}`",
        f"- transitions per run: `{payload['protocol']['total_steps']}`",
        "",
        "## Observed runtime",
        "",
        "| architecture | parameters | environment SPS | optimizer updates/s | training samples/s | wall-clock (s) | peak allocated VRAM | peak reserved VRAM | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        metrics = result["runtime_metrics"]
        summary = result["summary"]
        def display(value: Any) -> str:
            return "unavailable" if value is None else str(value)
        lines.append(
            "| {architecture} | {parameters} | {sps} | {updates} | {samples} | {wall} | {allocated} | {reserved} | {status} |".format(
                architecture=result["architecture"],
                parameters=display(metrics.get("parameter_count")),
                sps=display(metrics.get("end_to_end_environment_sps")),
                updates=display(metrics.get("optimizer_updates_per_second")),
                samples=display(metrics.get("training_samples_per_second")),
                wall=display(metrics.get("wall_clock_seconds")),
                allocated=display(metrics.get("peak_allocated_vram_bytes")),
                reserved=display(metrics.get("peak_reserved_vram_bytes")),
                status=summary.get("status"),
            )
        )
    lines.extend(
        [
            "",
            "## Comparison interpretation",
            "",
            f"- both runs completed: `{comparison['both_completed']}`",
            f"- checkpoint save/load passed for both runs: `{comparison['both_checkpoint_save_load_passed']}`",
            f"- same Double DQN algorithm: `{comparison['same_algorithm']}`",
            f"- same seed and transition budget: `{comparison['same_seed'] and comparison['same_transition_budget']}`",
            f"- same training settings except architecture: `{comparison['same_training_config_except_architecture']}`",
            f"- parameter-count delta (dueling − standard): `{comparison['parameter_count_delta']}`",
            f"- Dueling/standard environment-SPS ratio: `{comparison['dueling_to_standard_environment_sps_ratio']}`",
            "",
            comparison["interpretation"],
            "",
            "Raw values and complete provenance are in `assets/day19/dueling-smoke-runtime.json`; the run directories are reproducible inputs for later inspection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_steps < 1:
        raise ValueError("total steps must be positive")
    if args.total_steps < 1_000:
        raise ValueError("Day 19 smoke must include the canonical 1,000-step warmup")
    if args.total_steps % 2 != 0:
        raise ValueError("total steps must be divisible by the canonical num_envs=2")

    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Day 19 formal architecture smoke requires an explicit NVIDIA CUDA device"
        )
    environment_kwargs = breakout_environment_kwargs(contract)
    results = [
        _run_one(
            architecture,
            args=args,
            contract_id=contract.contract_id,
            contract_path=args.contract,
            environment_kwargs=environment_kwargs,
        )
        for architecture in ("standard", "dueling")
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "contract_path": args.contract.as_posix(),
        "contract": contract.to_dict(),
        "protocol": {
            "algorithm": "double_dqn",
            "architectures": ["standard", "dueling"],
            "seed": args.seed,
            "total_steps": args.total_steps,
            "num_envs": 2,
            "batch_size": 32,
            "learning_starts": 1_000,
            "train_frequency": 4,
            "target_update_interval": 500,
            "replay_backend": "gpu",
            "precision": "float32",
            "strict_action_selection_parity": True,
            "checkpoint_save_load_required": True,
            "fire_reset": contract.fire_reset,
            "terminal_on_life_loss": contract.terminal_on_life_loss,
            "frame_skip": contract.frame_skip,
            "frame_stack": contract.frame_stack,
            "sticky_action_probability": contract.sticky_action_probability,
            "generation_command": " ".join(str(value) for value in ([sys.executable, *sys.argv])),
        },
        "results": results,
    }
    payload["comparison"] = _comparison(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(args.report, payload=payload)
    print(json.dumps(payload["comparison"], indent=2, ensure_ascii=False))
    print(f"Runtime evidence written to: {args.output}")
    print(f"Runtime report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
