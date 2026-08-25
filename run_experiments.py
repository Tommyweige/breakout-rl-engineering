"""Run a controlled Day 14 experiment batch sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from breakout_env import make_breakout_env
from breakout_rl.experiments import (
    build_manifest,
    load_experiment_configs,
    relative_path,
    slugify,
    update_manifest,
    utc_timestamp,
    write_json_object,
)
from breakout_rl.training.dqn_trainer import DQNTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DQN configs sequentially and record a Day 14 manifest."
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="stable output id; an existing experiment directory is never overwritten",
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="require every config to request cuda or cuda:<index>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configs and write a planned manifest without training",
    )
    return parser


def _new_experiment_id() -> str:
    return f"day14-{utc_timestamp().replace(':', '').replace('-', '').replace('.', '')}"


def _failure_status(config_device: str, error: BaseException) -> str:
    message = str(error).lower()
    if config_device.startswith("cuda") and any(
        marker in message
        for marker in (
            "cuda was requested",
            "cuda is not available",
            "cuda device index",
            "refusing to fall back",
        )
    ):
        return "blocked"
    return "failed"


def _write_failure(
    run_dir: Path,
    *,
    status: str,
    error: BaseException | str,
    requested_device: str,
) -> None:
    message = str(error)
    write_json_object(
        run_dir / "failure.json",
        {
            "status": status,
            "error": message,
            "error_type": type(error).__name__ if isinstance(error, BaseException) else "Error",
            "requested_device": requested_device,
        },
    )


def _update_entry(
    manifest: dict[str, Any],
    manifest_path: Path,
    index: int,
    *,
    run_dir: Path,
    status: str,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    entry = manifest["variants"][index]
    entry["run_dir"] = relative_path(run_dir, start=manifest_path.parent)
    entry["run_id"] = run_dir.name
    entry["status"] = status
    if summary is not None:
        runtime = summary.get("runtime", {})
        entry["resolved_device"] = (
            runtime.get("resolved_device") if isinstance(runtime, dict) else None
        )
        entry["summary"] = {
            "status": summary.get("status"),
            "total_steps": summary.get("total_steps"),
            "episodes": summary.get("episodes"),
            "optimizer_updates": summary.get("optimizer_updates"),
            "steps_per_second": summary.get("steps_per_second"),
            "wall_clock_seconds": (
                runtime.get("wall_clock_seconds") if isinstance(runtime, dict) else None
            ),
        }
    if error is not None:
        entry["error"] = error


def run_batch(args: argparse.Namespace) -> tuple[int, Path, dict[str, Any]]:
    configs = load_experiment_configs(args.configs)
    run_keys = [(slugify(config.label), config.config.seed) for config in configs]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError(
            "config labels/seed pairs must be unique so run directories cannot collide"
        )
    if args.require_cuda:
        non_cuda = [
            config.label
            for config in configs
            if not (
                config.config.requested_device == "cuda"
                or config.config.requested_device.startswith("cuda:")
            )
        ]
        if non_cuda:
            raise ValueError(
                "--require-cuda rejected config(s): " + ", ".join(non_cuda)
            )
        requested_devices = [config.config.requested_device for config in configs]
        if len(set(requested_devices)) != 1:
            raise ValueError(
                "--require-cuda requires every variant to request the same CUDA device"
            )

    experiment_id = slugify(args.experiment_id or _new_experiment_id())
    experiments_root = args.experiments_root.resolve()
    runs_root = args.runs_root.resolve()
    experiment_dir = experiments_root / experiment_id
    if experiment_dir.exists():
        raise FileExistsError(
            f"experiment output already exists: {experiment_dir}; refusing to overwrite"
        )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    run_parent = runs_root / experiment_id
    run_parent.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment_dir / "manifest.json"
    manifest = build_manifest(
        experiment_id=experiment_id,
        configs=configs,
        manifest_path=manifest_path,
        command=[str(value) for value in sys.argv],
    )
    update_manifest(manifest_path, manifest)

    if args.dry_run:
        for index, config in enumerate(configs):
            run_dir = run_parent / f"{slugify(config.label)}-seed{config.config.seed}"
            _update_entry(
                manifest,
                manifest_path,
                index,
                run_dir=run_dir,
                status="planned",
            )
        manifest["status"] = "planned"
        update_manifest(manifest_path, manifest)
        return 0, manifest_path, manifest

    interrupted = False
    for index, config in enumerate(configs):
        run_dir = run_parent / f"{slugify(config.label)}-seed{config.config.seed}"
        if run_dir.exists():
            raise FileExistsError(
                f"run output already exists: {run_dir}; refusing to overwrite"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            env = make_breakout_env()
            try:
                trainer = DQNTrainer(
                    env,
                    config.config,
                    run_dir=run_dir,
                )
                summary = trainer.train()
            finally:
                env.close()
            status = str(summary.get("status", "completed"))
            _update_entry(
                manifest,
                manifest_path,
                index,
                run_dir=run_dir,
                status=status,
                summary=summary,
            )
        except KeyboardInterrupt as error:
            interrupted = True
            _write_failure(
                run_dir,
                status="interrupted",
                error=error,
                requested_device=config.config.requested_device,
            )
            _update_entry(
                manifest,
                manifest_path,
                index,
                run_dir=run_dir,
                status="interrupted",
                error="keyboard interrupt",
            )
            update_manifest(manifest_path, manifest)
            break
        except Exception as error:
            status = _failure_status(config.config.requested_device, error)
            _write_failure(
                run_dir,
                status=status,
                error=error,
                requested_device=config.config.requested_device,
            )
            _update_entry(
                manifest,
                manifest_path,
                index,
                run_dir=run_dir,
                status=status,
                error=str(error),
            )
        update_manifest(manifest_path, manifest)

    statuses = [str(entry.get("status")) for entry in manifest["variants"]]
    if interrupted or "interrupted" in statuses:
        manifest["status"] = "interrupted"
    elif all(status == "completed" for status in statuses):
        manifest["status"] = "completed"
    elif any(status == "blocked" for status in statuses):
        manifest["status"] = "blocked"
    else:
        manifest["status"] = "failed"
    update_manifest(manifest_path, manifest)
    exit_code = 0 if manifest["status"] == "completed" else 1
    return exit_code, manifest_path, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, manifest_path, manifest = run_batch(args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(manifest_path),
                "variants": [
                    {
                        "label": entry["label"],
                        "status": entry["status"],
                        "run_dir": entry["run_dir"],
                    }
                    for entry in manifest["variants"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
