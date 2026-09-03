"""Tests for the Day 20 family protocol, reuse audit, and selection gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from breakout_rl.day20_comparison import (
    DAY20_FAMILIES,
    DAY20_FAMILY_IDS,
    DAY20_EXTENSION_STAGE,
    DAY20_EXTENSION_TARGET,
    DAY20_MILESTONES,
    DAY20_TRAINING_SEEDS,
    audit_day18_evidence_reuse,
    build_day20_manifest,
    build_day20_report,
    config_diff,
    load_day20_config,
    read_day20_manifest,
    select_top_candidates,
    sha256_file,
    validate_day20_manifest,
    write_json,
)
from scripts.training.run_dqn_family_comparison import (
    _invalidate_extension_entries,
    _invalidate_reused_entries,
)


class Day20ComparisonTests(unittest.TestCase):
    def test_config_freezes_three_families_and_cuda_backend(self) -> None:
        config = load_day20_config(require_probe_states=True)

        self.assertEqual(config.families, DAY20_FAMILIES)
        self.assertEqual(config.family_ids, DAY20_FAMILY_IDS)
        self.assertEqual(config.training_seeds, DAY20_TRAINING_SEEDS)
        self.assertEqual(dict(config.milestones), DAY20_MILESTONES)
        self.assertEqual(config.backend_config.num_envs, 2)
        self.assertEqual(config.backend_config.replay_backend, "gpu")
        self.assertEqual(config.requested_device, "cuda")
        self.assertEqual(config.precision, "float32")

    def test_manifest_has_one_entry_per_family_seed_and_milestone(self) -> None:
        config = load_day20_config(require_probe_states=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "experiments" / "day20" / "manifest.json"
            manifest = build_day20_manifest(
                config,
                manifest_path=manifest_path,
                runs_root=root / "runs",
                evaluations_root=root / "evaluations",
            )
            write_json(manifest_path, manifest)
            restored = read_day20_manifest(manifest_path)

        self.assertEqual(
            len(restored["runs"]),
            len(DAY20_FAMILIES) * len(DAY20_TRAINING_SEEDS) * len(DAY20_MILESTONES),
        )
        keys = {
            (entry["family_id"], entry["training_seed"], entry["stage"])
            for entry in restored["runs"]
        }
        self.assertEqual(len(keys), len(restored["runs"]))
        self.assertTrue(all(entry["status"] == "pending" for entry in restored["runs"]))

    def test_day18_evidence_is_accepted_only_when_shared_conditions_match(self) -> None:
        config = load_day20_config(require_probe_states=True)
        audit = audit_day18_evidence_reuse(config)

        self.assertEqual(audit["status"], "compatible")
        self.assertTrue(audit["reuse_allowed"])
        self.assertEqual(audit["reusable_entry_count"], 18)
        self.assertEqual(audit["incompatibilities"], [])

    def test_config_diff_reports_changed_non_family_control(self) -> None:
        first = {"batch_size": 32, "gamma": 0.99}
        second = {"batch_size": 64, "gamma": 0.99}

        self.assertEqual(
            config_diff(first, second),
            {"batch_size": {"first": 32, "second": 64}},
        )

    def test_top_candidates_use_aggregate_quality_not_single_seed(self) -> None:
        aggregates = [
            {
                "family_id": "dqn",
                "quality_mean": 20.0,
                "quality_median": 20.0,
                "quality_seed_spread": 8.0,
            },
            {
                "family_id": "double_dqn",
                "quality_mean": 22.0,
                "quality_median": 22.0,
                "quality_seed_spread": 2.0,
            },
            {
                "family_id": "dueling_double_dqn",
                "quality_mean": None,
                "quality_median": None,
                "quality_seed_spread": None,
            },
        ]

        self.assertEqual(select_top_candidates(aggregates), ["double_dqn", "dqn"])

    def test_report_uses_complete_extension_for_final_selection(self) -> None:
        config = load_day20_config(require_probe_states=False)
        entries = []
        base_values = {
            "dqn": (10.0, 10.0, 10.0),
            "double_dqn": (12.0, 12.0, 12.0),
            "dueling_double_dqn": (13.0, 11.0, 13.0),
        }
        extension_values = {
            "double_dqn": (18.0, 18.0, 18.0),
            "dueling_double_dqn": (20.0, 20.0, 20.0),
        }
        for family in DAY20_FAMILIES:
            for index, seed in enumerate(config.training_seeds):
                entries.append(
                    self._complete_entry(
                        family.family_id,
                        family.algorithm,
                        family.architecture,
                        seed,
                        "main",
                        DAY20_MILESTONES["main"],
                        base_values[family.family_id][index],
                    )
                )
                if family.family_id in extension_values:
                    entries.append(
                        self._complete_entry(
                            family.family_id,
                            family.algorithm,
                            family.architecture,
                            seed,
                            DAY20_EXTENSION_STAGE,
                            DAY20_EXTENSION_TARGET,
                            extension_values[family.family_id][index],
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._attach_artifacts(root, entries, config)
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                {"schema_version": 1, "sequential": True, "runs": entries},
            )
            report = build_day20_report(manifest_path, config=config)

        self.assertEqual(report["selection"]["base_500k"]["final_training_family"], "dueling_double_dqn")
        self.assertEqual(report["selection"]["final_training_family"], "dueling_double_dqn")
        self.assertTrue(report["selection"]["extension"]["applied"])
        self.assertEqual(report["extension"]["status"], "complete")
        self.assertEqual(report["extension"]["completed_entry_count"], 6)
        self.assertIsNone(report["selection"]["deployment_candidate"])

    def test_incomplete_extension_is_not_treated_as_zero_or_final_selection(self) -> None:
        config = load_day20_config(require_probe_states=False)
        entries = []
        for family in DAY20_FAMILIES:
            for seed in config.training_seeds:
                entries.append(
                    self._complete_entry(
                        family.family_id,
                        family.algorithm,
                        family.architecture,
                        seed,
                        "main",
                        DAY20_MILESTONES["main"],
                        {
                            "dqn": (10.0, 10.0, 10.0),
                            "double_dqn": (12.0, 12.0, 12.0),
                            "dueling_double_dqn": (13.0, 11.0, 13.0),
                        }[family.family_id][config.training_seeds.index(seed)],
                    )
                )
        for family_id, value in (("double_dqn", 18.0), ("dueling_double_dqn", 20.0)):
            family = config.family(family_id)
            for seed in config.training_seeds:
                if family_id == "dueling_double_dqn" and seed == 33:
                    continue
                entries.append(
                    self._complete_entry(
                        family_id,
                        family.algorithm,
                        family.architecture,
                        seed,
                        DAY20_EXTENSION_STAGE,
                        DAY20_EXTENSION_TARGET,
                        value,
                    )
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._attach_artifacts(root, entries, config)
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                {"schema_version": 1, "sequential": True, "runs": entries},
            )
            report = build_day20_report(manifest_path, config=config)

        self.assertEqual(report["selection"]["final_training_family"], "dueling_double_dqn")
        self.assertFalse(report["selection"]["extension"]["applied"])
        self.assertEqual(report["extension"]["status"], "pending")
        self.assertIsNone(report["extension"]["selection"]["final_training_family"])

    def test_incompatible_reuse_invalidates_old_entries_instead_of_mixing(self) -> None:
        manifest = {
            "runs": [
                {
                    "family_id": "dqn",
                    "status": "reused",
                    "eligible": True,
                    "source": {"kind": "day18_evidence_reuse"},
                    "run_dir": "../../runs/day20/dqn/seed11/stage-500k",
                    "evaluation": {"directory": "../../evaluations/day20/dqn"},
                },
                {"family_id": "dueling_double_dqn", "status": "completed"},
            ]
        }

        self.assertEqual(
            _invalidate_reused_entries(manifest, reason="protocol mismatch"),
            1,
        )
        entry = manifest["runs"][0]
        self.assertEqual(entry["status"], "pending")
        self.assertFalse(entry["eligible"])
        self.assertIsNone(entry["checkpoint"])
        self.assertIsNone(entry["source"])
        self.assertEqual(entry["error"], "protocol mismatch")

    def test_extension_evidence_is_invalidated_when_base_reuse_changes(self) -> None:
        manifest = {
            "runs": [
                {
                    "stage": DAY20_EXTENSION_STAGE,
                    "status": "completed",
                    "eligible": True,
                    "run_dir": "../../runs/day20/dueling/seed11/stage-1000k",
                    "evaluation": {"directory": "../../evaluations/day20/dueling"},
                }
            ]
        }

        self.assertEqual(
            _invalidate_extension_entries(manifest, reason="base provenance changed"),
            1,
        )
        self.assertEqual(manifest["runs"][0]["status"], "pending")
        self.assertFalse(manifest["runs"][0]["eligible"])
        self.assertEqual(manifest["runs"][0]["error"], "base provenance changed")

    def test_resume_manifest_identity_and_hashes_are_validated(self) -> None:
        config = load_day20_config(require_probe_states=False)
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            manifest = build_day20_manifest(
                config,
                manifest_path=manifest_path,
                runs_root=Path(temporary_directory) / "runs",
                evaluations_root=Path(temporary_directory) / "evaluations",
            )
            manifest["experiment_id"] = "stale-experiment"
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "different experiment"):
                validate_day20_manifest(manifest_path, config=config)

    @staticmethod
    def _attach_artifacts(
        root: Path,
        entries: list[dict],
        config,
    ) -> None:
        metric_fields = (
            "global_step",
            "raw_episode_return",
            "loss",
            "q_mean",
            "q_max",
            "target_mean",
            "td_error_mean_abs",
            "gradient_norm",
            "sps",
        )
        for entry in entries:
            token = entry["run_id"]
            metrics_path = root / "metrics" / f"{token}.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                ",".join(metric_fields)
                + "\n"
                + ",".join(
                    [
                        str(entry["target_transitions"]),
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            evaluation_dir = root / "evaluations" / token
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            mean_return = float(entry["evaluation"]["summary"]["mean_return"])
            rows = [
                {
                    "evaluation_seed": evaluation_seed,
                    "episode_index": episode_index,
                    "episode_seed": evaluation_seed + episode_index - 1,
                    "episode_return": mean_return,
                    "episode_length": 1,
                    "terminated": True,
                    "truncated": False,
                    "time_limit": False,
                    "complete": True,
                    "stop_reason": "terminated",
                }
                for evaluation_seed in config.evaluation_seeds
                for episode_index in range(1, config.episodes_per_seed + 1)
            ]
            summary = {
                "count": len(rows),
                "mean_return": mean_return,
                "median_return": mean_return,
                "std_return": 0.0,
                "min_return": mean_return,
                "max_return": mean_return,
                "mean_episode_length": 1.0,
                "complete_episodes": len(rows),
                "finished_episode_count": len(rows),
                "terminated_count": len(rows),
                "truncated_count": 0,
                "time_limit_truncated_count": 0,
                "truncation_rate": 0.0,
                "mean_return_terminated": mean_return,
                "mean_return_truncated": None,
                "mean_length_terminated": 1.0,
                "mean_length_truncated": None,
            }
            write_json(
                evaluation_dir / "results.json",
                {
                    "schema_version": 1,
                    "environment_id": config.contract.environment_id,
                    "evaluation_seeds": list(config.evaluation_seeds),
                    "episodes_per_seed": config.episodes_per_seed,
                    "evaluation_epsilon": 0.0,
                    "requested_device": "cuda",
                    "resolved_device": "cuda:0",
                    "runtime": {"cuda_available": True},
                    "per_episode": rows,
                    "summary": summary,
                },
            )
            (evaluation_dir / "episodes.csv").write_text("fixture\n", encoding="utf-8")
            q_path = root / "q" / f"{token}.json"
            write_json(
                q_path,
                {
                    "schema_version": 1,
                    "probe_states": {
                        "path": str(config.probe_states_path),
                        "sha256": sha256_file(config.probe_states_path),
                    },
                    "analysis": {
                        "probe_count": 1,
                        "action_count": 4,
                        "q_values": [[0.0, 0.0, 0.0, 0.0]],
                    },
                },
            )
            entry["training"] = {"metrics_path": str(metrics_path.relative_to(root))}
            entry["evaluation"] = {
                "directory": str(evaluation_dir.relative_to(root)),
                "results": str((evaluation_dir / "results.json").relative_to(root)),
                "episodes": str((evaluation_dir / "episodes.csv").relative_to(root)),
                "summary": summary,
                "status": "completed",
            }
            entry["q_probe"] = {
                "path": str(q_path.relative_to(root)),
                "summary": {"probe_count": 1, "action_count": 4},
                "status": "completed",
            }
            checkpoint_path = root / "checkpoints" / f"{token}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fixture-checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            entry["checkpoint"] = {
                "path": str(checkpoint_path.relative_to(root)),
                "sha256": checkpoint_sha,
                "step": entry["target_transitions"],
            }
            evaluation_path = evaluation_dir / "results.json"
            evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation_payload["checkpoint"] = {
                "sha256": checkpoint_sha,
                "step": entry["target_transitions"],
            }
            evaluation_payload["training"] = {
                "algorithm": entry["algorithm"],
                "architecture": entry["architecture"],
                "training_seed": entry["training_seed"],
                "training_budget": entry["target_transitions"],
            }
            evaluation_payload["metadata"] = {
                "raw_reward": True,
                "evaluation_contract_provenance": {
                    "contract_id": config.contract.contract_id,
                    "contract_sha256": sha256_file(config.contract_path),
                },
            }
            write_json(evaluation_path, evaluation_payload)

    @staticmethod
    def _complete_entry(
        family_id: str,
        algorithm: str,
        architecture: str,
        seed: int,
        stage: str,
        target: int,
        mean_return: float,
    ) -> dict:
        return {
            "run_id": f"{family_id}-seed{seed}-{stage}",
            "family_id": family_id,
            "algorithm": algorithm,
            "architecture": architecture,
            "training_seed": seed,
            "stage": stage,
            "target_transitions": target,
            "status": "completed",
            "eligible": True,
            "summary": {
                "status": "completed",
                "total_transitions": target,
                "training_steps": target,
                "physical_environment_steps": target,
            },
            "runtime": {
                "requested_device": "cuda",
                "resolved_device": "cuda:0",
                "precision": "float32",
                "cuda_available": True,
                "cuda_device_index": 0,
                "cuda_device_name": "fixture-cuda",
                "pytorch_version": "fixture-pytorch",
                "torch_cuda_version": "fixture-cuda-runtime",
                "steps_per_second": 300.0,
                "wall_clock_seconds": 100.0,
                "cuda_peak_allocated_bytes": 600_000_000,
                "gpu_memory_total_bytes": 8_000_000_000,
                "training_steps": target,
                "physical_environment_steps": target,
                "action_inference_transitions": target,
                "replay_insertion_transitions": target,
            },
            "evaluation": {
                "status": "completed",
                "summary": {
                    "count": 15,
                    "mean_return": mean_return,
                    "median_return": mean_return,
                    "std_return": 0.0,
                    "min_return": mean_return,
                    "max_return": mean_return,
                    "mean_episode_length": 1.0,
                    "complete_episodes": 15,
                    "finished_episode_count": 15,
                    "terminated_count": 15,
                    "truncated_count": 0,
                    "time_limit_truncated_count": 0,
                    "truncation_rate": 0.0,
                    "mean_return_terminated": mean_return,
                    "mean_return_truncated": None,
                    "mean_length_terminated": 1.0,
                    "mean_length_truncated": None,
                },
            },
            "q_probe": {"status": "completed"},
        }


if __name__ == "__main__":
    unittest.main()
