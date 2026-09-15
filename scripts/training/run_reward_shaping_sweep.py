"""Run the Issue #9 Stage 2 250k penalty sweep without changing budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.analysis.analyze_training_run import analyze_run
from breakout_rl.training.config import DQNConfig


DEFAULT_CONFIGS = (
    Path("configs/issue9_reward_shaping_250k_baseline.json"),
    Path("configs/issue9_reward_shaping_250k_penalty_minus025.json"),
    Path("configs/issue9_reward_shaping_250k_penalty_minus05.json"),
    Path("configs/issue9_reward_shaping_250k_penalty_minus1.json"),
)


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: config must be a JSON object")
    return payload


def _resolved_config(path: Path) -> DQNConfig:
    payload = _load_payload(path)
    raw_config = payload.get("training_config", payload)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{path}: training_config must be a JSON object")
    config = DQNConfig.from_dict(raw_config)
    if (
        config.total_steps != 250_000
        or config.seed != 2022
        or config.algorithm != "double_dqn"
        or config.architecture != "dueling"
    ):
        raise ValueError(
            f"{path}: Stage 2 requires total_steps=250000, seed=2022, "
            "algorithm=double_dqn, architecture=dueling"
        )
    return config


def _label(path: Path, config: DQNConfig) -> str:
    payload = _load_payload(path)
    raw_label = payload.get("label")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()
    return f"penalty-{config.life_loss_penalty:g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-condition Issue #9 250k penalty sweep."
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        default=None,
        help="repeat for explicit configs; defaults to the four Stage 2 configs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/issue-9-reward-shaping/stage2-250k"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="run variants concurrently; GPU memory must be sufficient for all workers",
    )
    return parser


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    config_paths = tuple(args.configs or DEFAULT_CONFIGS)
    if not config_paths:
        raise ValueError("at least one sweep config is required")
    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = _resolved_config(config_path)
        variants.append(
            {
                "label": _label(config_path, config),
                "config_path": config_path.as_posix(),
                "config": config.to_dict(),
                "run_dir": str(runs_dir / _label(config_path, config)),
                "status": "pending",
                "command": None,
            }
        )

    manifest_path = output_dir / "training-sweep.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "issue9_reward_shaping_250k_training_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "training_seed": 2022,
        "training_transitions": 250_000,
        "algorithm": "double_dqn",
        "architecture": "dueling",
        "selection_status": "candidate screening; not final model promotion",
        "variants": variants,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.dry_run:
        return manifest

    def command_for(variant: Mapping[str, Any]) -> list[str]:
        return [
            sys.executable,
            "-m",
            "scripts.training.train_vectorized_dqn",
            "--config",
            variant["config_path"],
            "--device",
            args.device,
            "--run-dir",
            str(runs_dir),
            "--run-id",
            variant["label"],
        ]

    def write_manifest() -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def finalize_variant(variant: dict[str, Any]) -> None:
        """Persist diagnostics and plots as part of the sweep artifact."""

        run_dir = Path(str(variant["run_dir"]))
        report = analyze_run(run_dir)
        report_path = run_dir / "analysis-report.json"
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        variant["analysis_report"] = str(report_path)
        variant["plots_dir"] = str(run_dir / "plots")

    if args.parallel:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
        for variant in variants:
            command = command_for(variant)
            variant["command"] = command
            variant["status"] = "running"
            log_stream = (logs_dir / f"{variant['label']}.log").open(
                "w",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            active.append((variant, process, log_stream))
        write_manifest()
        while active:
            remaining: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
            for variant, process, log_stream in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((variant, process, log_stream))
                    continue
                log_stream.close()
                if returncode != 0:
                    variant["status"] = "failed"
                    variant["returncode"] = returncode
                    for other_variant, other_process, other_log in remaining:
                        other_process.terminate()
                        other_log.close()
                        other_variant["status"] = "cancelled"
                    manifest["status"] = "failed"
                    write_manifest()
                    raise subprocess.CalledProcessError(returncode, variant["command"])
                variant["status"] = "completed"
                variant["final_checkpoint"] = str(
                    Path(variant["run_dir"])
                    / "checkpoints"
                    / "step-00250000.pt"
                )
                finalize_variant(variant)
            active = remaining
            write_manifest()
            if active:
                time.sleep(2)
        manifest["status"] = "completed"
        write_manifest()
        return manifest

    for variant in variants:
        command = command_for(variant)
        variant["command"] = command
        variant["status"] = "running"
        write_manifest()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            variant["status"] = "failed"
            variant["returncode"] = error.returncode
            manifest["status"] = "failed"
            write_manifest()
            raise
        variant["status"] = "completed"
        variant["final_checkpoint"] = str(
            Path(variant["run_dir"])
            / "checkpoints"
            / "step-00250000.pt"
        )
        finalize_variant(variant)
        write_manifest()
    manifest["status"] = "completed"
    write_manifest()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_sweep(args)
    except (FileNotFoundError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Reward-shaping sweep failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
