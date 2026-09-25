"""Audit whether the Day 20 Uniform Replay runs fit the frozen PER protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_TRAINING_FIELDS = (
    "seed",
    "algorithm",
    "architecture",
    "total_steps",
    "num_envs",
    "replay_backend",
    "replay_capacity",
    "batch_size",
    "learning_rate",
    "gamma",
    "learning_starts",
    "train_frequency",
    "target_update_interval",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "reward_clip",
    "precision",
    "cpu_threads",
    "life_loss_penalty",
)


def _json_from_git(commit: str, path: str) -> dict[str, Any]:
    output = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        text=True,
        encoding="utf-8",
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ValueError(f"{commit}:{path} must contain a JSON object")
    return value


def _text_from_git(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        text=True,
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(
    checks: list[dict[str, Any]],
    name: str,
    expected: Any,
    observed: Any,
    *,
    normalize_missing_zero: bool = False,
) -> bool:
    if normalize_missing_zero and observed is None:
        observed = 0.0
    passed = expected == observed
    checks.append(
        {
            "name": name,
            "passed": passed,
            "expected": expected,
            "observed": observed,
        }
    )
    return passed


def _load_local_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def audit_baseline(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    experiment = _load_local_config(config_path)
    baseline = experiment["baseline"]
    expected_training = experiment["training_config"]
    expected_seeds = [int(seed) for seed in experiment["training_seeds"]]
    expected_milestones = [int(step) for step in experiment["milestones"]]
    evaluation = experiment["evaluation"]
    source_commit = str(baseline["source_commit"])
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("baseline.source_commit must be a full 40-character SHA")

    checks: list[dict[str, Any]] = []
    manifest = _json_from_git(source_commit, str(baseline["manifest_path"]))
    protocol = manifest["protocol"]
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    _check(
        checks,
        "pinned Day 20 source commit is in current history",
        True,
        source_is_ancestor,
    )
    backend_source = manifest["source_of_truth"]["backend_manifest"]
    backend_contents = subprocess.check_output(
        ["git", "show", f"{source_commit}:{backend_source['path']}"]
    )
    crlf_bytes = bytes((13, 10))
    lf_bytes = bytes((10,))
    backend_lf = backend_contents.replace(crlf_bytes, lf_bytes)
    backend_crlf = backend_lf.replace(lf_bytes, crlf_bytes)
    backend_git_sha256 = hashlib.sha256(backend_lf).hexdigest()
    backend_windows_sha256 = hashlib.sha256(backend_crlf).hexdigest()
    _check(
        checks,
        "Day 16 canonical backend manifest hash (LF or Windows CRLF)",
        True,
        backend_source["sha256"]
        in {backend_git_sha256, backend_windows_sha256},
    )
    _check(
        checks,
        "Day 16 canonical backend identity",
        baseline["backend_id"],
        manifest.get("base_backend", {}).get("backend_id"),
    )
    contract_path = Path(str(evaluation["contract"]))
    evaluation_path = Path(str(evaluation["config"]))
    if not contract_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("current Contract v2 or evaluation config is missing")

    _check(checks, "baseline manifest completed", "completed", manifest.get("status"))
    _check(checks, "Day 20 seed list", expected_seeds, protocol.get("training_seeds"))
    observed_milestones = [
        protocol.get("milestones", {}).get("screening"),
        protocol.get("milestones", {}).get("pilot"),
        protocol.get("milestones", {}).get("main"),
    ]
    _check(checks, "Day 20 transition milestones", expected_milestones, observed_milestones)
    _check(
        checks,
        "Day 20 formal transition budget",
        expected_milestones[-1],
        protocol.get("formal_quality_transitions"),
    )
    _check(checks, "Day 20 evaluation seeds", evaluation["seeds"], protocol.get("evaluation_seeds"))
    _check(
        checks,
        "Day 20 episodes per evaluation seed",
        evaluation["episodes_per_seed"],
        protocol.get("episodes_per_evaluation_seed"),
    )
    _check(checks, "Day 20 evaluation epsilon", evaluation["epsilon"], protocol.get("evaluation_epsilon"))
    _check(checks, "Day 20 raw Atari reward", evaluation["raw_reward"], protocol.get("raw_reward"))

    source_truth = manifest["source_of_truth"]
    frozen_evaluation = experiment["evaluation"]
    for label, local_path, frozen_key in (
        ("contract", contract_path, "contract_sha256"),
        ("evaluation_config", evaluation_path, "config_sha256"),
    ):
        _check(
            checks,
            f"experiment manifest frozen {label} hash",
            frozen_evaluation[frozen_key],
            _sha256(local_path),
        )
    for label, local_path in (
        ("contract", contract_path),
        ("evaluation_config", evaluation_path),
    ):
        source_entry = source_truth[label]
        _check(
            checks,
            f"{label} source hash",
            source_entry["sha256"],
            _sha256(local_path),
        )
    baseline_contract = source_truth["contract"]
    current_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _check(
        checks,
        "Contract v2 identifier",
        baseline_contract.get("contract_id"),
        current_contract.get("contract_id"),
    )

    uniform_source = _text_from_git(source_commit, "breakout_rl/replay_gpu.py")
    _check(
        checks,
        "historical GPU replay implementation is uniform",
        True,
        "sample_indices" in uniform_source
        and "torch.randperm" in uniform_source
        and "sample_prioritized" not in uniform_source,
    )

    artifact_records: list[dict[str, Any]] = []
    source_provenance: dict[int, dict[str, Any]] = {}
    for seed in expected_seeds:
        previous_milestone = 0
        for milestone in expected_milestones:
            artifact_path = str(baseline["evaluation_template"]).format(
                seed=seed,
                step=milestone,
            )
            payload = _json_from_git(source_commit, artifact_path)
            training = payload.get("training")
            if not isinstance(training, Mapping):
                raise ValueError(f"{artifact_path}: missing training metadata")
            observed_config = training.get("training_config")
            if not isinstance(observed_config, Mapping):
                raise ValueError(f"{artifact_path}: missing training_config")
            for field in REQUIRED_TRAINING_FIELDS:
                observed = observed_config.get(field)
                expected = (
                    milestone
                    if field == "total_steps"
                    else seed
                    if field == "seed"
                    else expected_training.get(field)
                )
                if field == "life_loss_penalty":
                    _check(
                        checks,
                        f"seed {seed} step {milestone} {field}",
                        expected,
                        observed,
                        normalize_missing_zero=True,
                    )
                else:
                    _check(
                        checks,
                        f"seed {seed} step {milestone} {field}",
                        expected,
                        observed,
                    )

            _check(
                checks,
                f"seed {seed} step {milestone} Uniform Replay mode",
                "uniform",
                baseline.get("replay_sampling"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} evaluation schema",
                2,
                payload.get("schema_version"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} evaluation seeds",
                evaluation["seeds"],
                payload.get("evaluation_seeds"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} episodes per seed",
                evaluation["episodes_per_seed"],
                payload.get("episodes_per_seed"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} evaluation epsilon",
                evaluation["epsilon"],
                payload.get("evaluation_epsilon"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} total episodes",
                len(evaluation["seeds"]) * evaluation["episodes_per_seed"],
                payload.get("total_episodes"),
            )
            checkpoint = payload.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                checkpoint = {}
            checkpoint_sha = checkpoint.get("sha256")
            valid_checkpoint_sha = isinstance(checkpoint_sha, str) and bool(
                re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha)
            )
            _check(
                checks,
                f"seed {seed} step {milestone} checkpoint SHA-256",
                True,
                valid_checkpoint_sha,
            )
            _check(
                checks,
                f"seed {seed} step {milestone} checkpoint schema",
                2,
                checkpoint.get("format_version"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} checkpoint transition count",
                milestone,
                checkpoint.get("training_steps", checkpoint.get("step")),
            )
            environment_contract = training.get("environment_contract", {})
            _check(
                checks,
                f"seed {seed} step {milestone} Contract v2 hash",
                baseline_contract.get("contract_sha256"),
                environment_contract.get("contract_sha256"),
            )
            per_episode = payload.get("per_episode")
            valid_episode_records = (
                isinstance(per_episode, list)
                and len(per_episode)
                == len(evaluation["seeds"]) * evaluation["episodes_per_seed"]
                and all(
                    isinstance(item, Mapping)
                    and isinstance(item.get("episode_return"), (int, float))
                    for item in per_episode
                )
            )
            _check(
                checks,
                f"seed {seed} step {milestone} per-episode raw returns",
                True,
                valid_episode_records,
            )

            runtime = training.get("trainer_runtime")
            if not isinstance(runtime, Mapping):
                runtime = {}
            expected_stage_transitions = milestone - previous_milestone
            _check(
                checks,
                f"seed {seed} step {milestone} runtime stage start",
                previous_milestone,
                runtime.get("stage_start_step"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} runtime stage transition count",
                expected_stage_transitions,
                runtime.get("stage_training_steps"),
            )
            runtime_wall_seconds = runtime.get("wall_clock_seconds")
            observed_runtime_rate = runtime.get("steps_per_second")
            expected_runtime_rate = (
                expected_stage_transitions / runtime_wall_seconds
                if isinstance(runtime_wall_seconds, (int, float))
                and float(runtime_wall_seconds) > 0.0
                else None
            )
            rate_matches_stage = (
                isinstance(observed_runtime_rate, (int, float))
                and expected_runtime_rate is not None
                and abs(float(observed_runtime_rate) - expected_runtime_rate)
                <= max(1e-6, expected_runtime_rate * 1e-4)
            )
            _check(
                checks,
                f"seed {seed} step {milestone} runtime rate matches its stage",
                True,
                rate_matches_stage,
            )
            _check(
                checks,
                f"seed {seed} step {milestone} CUDA runtime",
                True,
                isinstance(runtime.get("resolved_device"), str)
                and runtime.get("resolved_device").startswith("cuda"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} runtime CPU thread count",
                expected_training["cpu_threads"],
                runtime.get("cpu_thread_count"),
            )
            _check(
                checks,
                f"seed {seed} step {milestone} runtime precision",
                expected_training["precision"],
                runtime.get("precision"),
            )
            provenance = {
                "training_seed": seed,
                "git_commit_sha": runtime.get("git_commit_sha"),
                "git_dirty": runtime.get("git_dirty"),
                "git_diff_sha256": runtime.get("git_diff_sha256"),
                "wall_clock_seconds": runtime.get("wall_clock_seconds"),
                "steps_per_second": runtime.get("steps_per_second"),
            }
            source_provenance[seed] = provenance
            artifact_records.append(
                {
                    "training_seed": seed,
                    "transitions": milestone,
                    "evaluation_artifact": artifact_path,
                    "evaluation_id": payload.get("evaluation_id"),
                    "checkpoint_sha256": checkpoint_sha,
                    "mean_raw_return": payload.get("summary", {}).get("mean_return"),
                    "median_raw_return": payload.get("summary", {}).get("median_return"),
                    "evaluation_schema_version": payload.get("schema_version"),
                    "training_stage_start_step": runtime.get("stage_start_step"),
                    "training_stage_transitions": runtime.get("stage_training_steps"),
                    "training_stage_wall_clock_seconds": runtime_wall_seconds,
                }
            )
            previous_milestone = milestone

    provenance_values = list(source_provenance.values())
    provenance_complete = all(
        isinstance(item.get("git_commit_sha"), str)
        and len(item["git_commit_sha"]) == 40
        and (
            item.get("git_dirty") is not True
            or (
                isinstance(item.get("git_diff_sha256"), str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", item["git_diff_sha256"]))
            )
        )
        for item in provenance_values
    )
    _check(
        checks,
        "baseline training source provenance recorded",
        True,
        provenance_complete,
    )
    unique_source_commits = sorted(
        {str(item["git_commit_sha"]) for item in provenance_values if item.get("git_commit_sha")}
    )
    unique_diff_hashes = sorted(
        {str(item["git_diff_sha256"]) for item in provenance_values if item.get("git_diff_sha256")}
    )

    passed = all(check["passed"] for check in checks)
    missing_penalty_fields = [
        check["name"]
        for check in checks
        if check["name"].endswith("life_loss_penalty")
        and check["observed"] is None
    ]
    return {
        "schema_version": 1,
        "status": "compatible" if passed else "incompatible",
        "reuse_allowed": passed,
        "decision": "reuse_day20_uniform" if passed else "rerun_uniform_baseline",
        "experiment_id": experiment["experiment_id"],
        "baseline": {
            "experiment_id": baseline["experiment_id"],
            "family_id": baseline["family_id"],
            "backend_id": baseline["backend_id"],
            "backend_manifest_sha256": backend_source["sha256"],
            "backend_manifest_git_blob_sha256": backend_git_sha256,
            "backend_manifest_windows_sha256": backend_windows_sha256,
            "replay_sampling": baseline["replay_sampling"],
            "source_commit": source_commit,
            "training_source_commits": unique_source_commits,
            "training_source_diff_hashes": unique_diff_hashes,
            "training_source_provenance": provenance_values,
            "historical_life_loss_penalty_missing_fields_normalized_to_zero": missing_penalty_fields,
        },
        "protocol": {
            "training_seeds": expected_seeds,
            "milestones": expected_milestones,
            "evaluation_seeds": list(evaluation["seeds"]),
            "episodes_per_evaluation_seed": evaluation["episodes_per_seed"],
            "evaluation_epsilon": evaluation["epsilon"],
            "raw_reward": evaluation["raw_reward"],
            "contract_sha256": _sha256(contract_path),
            "evaluation_config_sha256": _sha256(evaluation_path),
        },
        "checkpoints": artifact_records,
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed compatibility audit for the Day 20 Uniform Replay baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/per_experiment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/per/baseline-compatibility.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_baseline(args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PER baseline audit failed closed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": audit["status"],
                "reuse_allowed": audit["reuse_allowed"],
                "decision": audit["decision"],
                "checks": len(audit["checks"]),
                "failed_checks": len(audit["failed_checks"]),
                "output": args.output.as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if audit["reuse_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
