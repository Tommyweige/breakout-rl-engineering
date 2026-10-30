"""Schema and provenance checks for the issue #73 closeout evidence pack."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs" / "final-results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
INVENTORY_PATH = RESULTS_DIR / "evidence-inventory.csv"
ARTIFACT_MANIFEST_PATH = RESULTS_DIR / "artifact-manifest.json"

ALLOWED_STATUSES = {
    "verified",
    "historical_reported",
    "missing_raw_artifacts",
    "not_yet_evaluated",
}
ALLOWED_RAW_ARTIFACT_STATUSES = {
    "in_repo",
    "external_branch",
    "missing",
    "not_applicable",
}
ALLOWED_ARTIFACT_STATUSES = {
    "in_repo",
    "external_branch",
    "missing",
    "not_created",
}
REQUIRED_METRIC_FIELDS = (
    "id",
    "label",
    "value",
    "unit",
    "contract",
    "seed_count",
    "source",
    "ref",
    "verification_status",
)


def _load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


class FinalResultsSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = _load_summary()
        cls.metrics = {metric["id"]: metric for metric in cls.summary["metrics"]}

    def test_required_top_level_sections_are_present(self) -> None:
        for key in (
            "schema_version",
            "frozen_at",
            "project_status",
            "research_question",
            "claim_policy",
            "contracts",
            "headline",
            "metrics",
            "table_metric_ids",
            "comparisons",
            "experiments",
            "non_claims",
            "links",
        ):
            self.assertIn(key, self.summary)

        self.assertEqual(self.summary["schema_version"], 1)
        self.assertEqual(self.summary["project_status"], "frozen_evidence_closeout")
        headline = self.summary["headline"]
        self.assertEqual(headline["canonical_clear_detections"], 4)
        self.assertEqual(headline["canonical_clear_episodes"], 15)
        self.assertIn("a8445df", headline["protocol"])

    def test_every_metric_has_the_required_provenance_fields(self) -> None:
        for metric in self.summary["metrics"]:
            for field in REQUIRED_METRIC_FIELDS:
                self.assertIn(field, metric, metric.get("id"))
            self.assertIn(metric["verification_status"], ALLOWED_STATUSES, metric["id"])
            if "raw_artifact_status" in metric:
                self.assertIn(
                    metric["raw_artifact_status"],
                    ALLOWED_RAW_ARTIFACT_STATUSES,
                    metric["id"],
                )
            self.assertIsInstance(metric["seed_count"], int, metric["id"])
            self.assertGreaterEqual(metric["seed_count"], 0, metric["id"])
            contract = metric["contract"]
            self.assertIsInstance(contract, dict, metric["id"])
            self.assertIn("id", contract, metric["id"])
            self.assertIn("path", contract, metric["id"])
            self.assertIn("frame_skip", contract, metric["id"])

    def test_metric_ids_are_unique_and_table_ids_resolve(self) -> None:
        metric_ids = [metric["id"] for metric in self.summary["metrics"]]
        self.assertEqual(len(metric_ids), len(set(metric_ids)))
        for metric_id in self.summary["table_metric_ids"]:
            self.assertIn(metric_id, self.metrics)

    def test_contract_hashes_match_the_versioned_files(self) -> None:
        for contract_key in ("v2", "v3"):
            contract = self.summary["contracts"][contract_key]
            path = ROOT / contract["path"]
            self.assertTrue(path.is_file(), contract["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, contract["sha256"], contract["path"])

        v2 = self.summary["contracts"]["v2"]
        v3 = self.summary["contracts"]["v3"]
        self.assertEqual(v2["frame_skip"], 4)
        self.assertEqual(v2["agent_step_limit"], 27000)
        self.assertEqual(v3["frame_skip"], 1)
        self.assertEqual(v3["agent_step_limit"], 108000)
        self.assertEqual(v2["max_num_frames_per_episode"], v3["max_num_frames_per_episode"])
        self.assertEqual(v2["sticky_action_probability"], v3["sticky_action_probability"])
        self.assertEqual(v2["evaluation_seeds"], v3["evaluation_seeds"])

    def test_metric_contract_paths_exist_and_known_contracts_match(self) -> None:
        known = {
            contract["contract_id"]: contract["sha256"]
            for contract in self.summary["contracts"].values()
        }
        for metric in self.summary["metrics"]:
            contract = metric["contract"]
            self.assertTrue((ROOT / contract["path"]).is_file(), contract["path"])
            if contract["id"] in known:
                actual = hashlib.sha256((ROOT / contract["path"]).read_bytes()).hexdigest()
                self.assertEqual(actual, known[contract["id"]], metric["id"])

    def test_cross_contract_comparison_is_not_presented_as_a_win(self) -> None:
        comparisons = {item["id"]: item for item in self.summary["comparisons"]}
        self.assertEqual(
            comparisons["dqn_vs_rgb_controller"]["comparability"],
            "not_directly_comparable",
        )
        self.assertEqual(comparisons["v2_vs_v3"]["comparability"], "not_directly_comparable")
        self.assertEqual(
            comparisons["dqn_v3_vs_rgb_controller_v3"]["comparability"],
            "same_contract_descriptive",
        )

    def test_controller_results_are_tied_to_their_control_law(self) -> None:
        current = self.metrics["rgb_controller_a8445df_canonical_clear_detections"]
        self.assertEqual((current["value"], current["seed_count"]), (4, 15))
        self.assertEqual(self.metrics["rgb_controller_a8445df_mean_score"]["value"], 721.8)
        self.assertEqual(self.metrics["rgb_controller_pr26_mean_score"]["value"], 137.47)
        self.assertEqual(self.metrics["rgb_controller_pr26_canonical_clear_detections"]["value"], 0)
        # 0 clears is only a correct measurement because the old law never finished one wall.
        self.assertLess(self.metrics["rgb_controller_pr26_max_score"]["value"], 432)
        for metric in self.summary["metrics"]:
            if metric["id"].startswith(("rgb_controller_pr26_", "rgb_controller_a8445df_")):
                self.assertEqual(metric["contract"]["id"], "breakout-evaluation-v3-frame-skip-1")
        dqn_v3 = self.metrics["dqn_v3_2m_mean_score"]
        self.assertEqual(dqn_v3["contract"]["id"], "breakout-evaluation-v3-frame-skip-1")
        self.assertEqual(dqn_v3["value"], 6.93)
        self.assertEqual(self.metrics["dqn_v3_2m_clears"]["value"], 0)

    def test_clear_wording_stays_within_canonical_detection(self) -> None:
        texts = [
            json.dumps(self.summary),
            INVENTORY_PATH.read_text(encoding="utf-8"),
            (RESULTS_DIR / "README.md").read_text(encoding="utf-8"),
            (RESULTS_DIR / "handoff.md").read_text(encoding="utf-8"),
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "README.zh-TW.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "clear-semantics.md").read_text(encoding="utf-8"),
        ]
        for text in texts:
            self.assertIsNone(re.search(r"verified[ _]?(full )?clears?", text, re.I))

    def test_local_references_exist(self) -> None:
        prefixes = ("docs/", "configs/", "web/", "scripts/", "tests/", "audits/")
        for metric in self.summary["metrics"]:
            ref = metric["ref"]
            if ref.startswith(prefixes) and "://" not in ref:
                path = ROOT / ref.split("#", 1)[0]
                self.assertTrue(path.exists(), f"{metric['id']}: {ref}")

        for link in (
            self.summary["links"]["results_index"],
            self.summary["links"]["handoff"],
            self.summary["links"]["evidence_inventory"],
            self.summary["links"]["artifact_manifest"],
        ):
            self.assertTrue((ROOT / link).is_file(), link)

    def test_inventory_csv_matches_the_experiment_ledger(self) -> None:
        with INVENTORY_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertTrue(rows)
        expected_header = {
            "experiment_id",
            "title",
            "method",
            "contract",
            "seed_summary",
            "headline_result",
            "verification_status",
            "raw_artifact_status",
            "source",
            "notes",
        }
        self.assertEqual(set(rows[0]), expected_header)
        self.assertEqual(
            {row["experiment_id"] for row in rows},
            {experiment["id"] for experiment in self.summary["experiments"]},
        )
        for row in rows:
            self.assertIn(row["verification_status"], ALLOWED_STATUSES, row["experiment_id"])
            self.assertIn(
                row["raw_artifact_status"],
                ALLOWED_RAW_ARTIFACT_STATUSES,
                row["experiment_id"],
            )

    def test_markdown_local_links_exist(self) -> None:
        for markdown_path in (
            ROOT / "README.md",
            ROOT / "README.zh-TW.md",
            RESULTS_DIR / "README.md",
            RESULTS_DIR / "handoff.md",
            RESULTS_DIR / "screenshots" / "README.md",
        ):
            text = markdown_path.read_text(encoding="utf-8")
            for match in re.finditer(r"\]\(([^)\s]+)\)", text):
                link = match.group(1)
                if link.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = (markdown_path.parent / link.split("#", 1)[0]).resolve()
                self.assertTrue(target.exists(), f"{markdown_path.name}: broken link {link}")

    def test_artifact_manifest_matches_the_repository(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        artifacts = manifest["artifacts"]
        self.assertEqual(len({artifact["id"] for artifact in artifacts}), len(artifacts))

        for artifact in artifacts:
            for field in ("id", "kind", "path", "producer", "status", "sha256", "source_record"):
                self.assertIn(field, artifact, artifact.get("id"))
            self.assertIn(artifact["status"], ALLOWED_ARTIFACT_STATUSES, artifact["id"])
            path = ROOT / artifact["path"]
            if artifact["status"] == "in_repo":
                self.assertTrue(path.exists(), artifact["id"])
                if artifact["sha256"] is not None:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    self.assertEqual(actual, artifact["sha256"], artifact["id"])
            else:
                self.assertFalse(
                    path.exists(),
                    f"{artifact['id']} is marked {artifact['status']} but {artifact['path']} exists",
                )

            source = artifact["source_record"]
            if not source.startswith("http") and "://" not in source:
                self.assertTrue((ROOT / source.split("#", 1)[0]).exists(), artifact["id"])

    def test_case_study_page_and_real_media_exist(self) -> None:
        page = ROOT / "web" / "case-study.html"
        self.assertTrue(page.is_file())
        html = page.read_text(encoding="utf-8")
        self.assertIn('lang="en"', html)
        self.assertIn('/src/case-study/main.ts', html)
        self.assertIn('rel="canonical"', html)
        self.assertIn(
            'rel="canonical" href="https://breakout.tommypan.com/case-study"',
            html,
        )

        for asset in (
            ROOT / "web" / "public" / "case-study" / "tracker.mp4",
            ROOT / "web" / "public" / "case-study" / "tracker-poster.png",
            ROOT / "web" / "public" / "case-study" / "gameplay-poster.png",
        ):
            self.assertTrue(asset.is_file(), asset)
            self.assertGreater(asset.stat().st_size, 0, asset)

    def test_case_study_routes_do_not_loop_on_cloudflare_pages(self) -> None:
        redirects = (ROOT / "web" / "public" / "_redirects").read_text(encoding="utf-8")
        self.assertNotRegex(redirects, r"(?m)^/case-study\s+/case-study\.html\b")
        self.assertNotRegex(redirects, r"(?m)^/results\s+/case-study\.html\b")
        self.assertIn("/results /case-study 301", redirects)
        self.assertIn("case-study.zh-TW", redirects)

        view = (ROOT / "web" / "src" / "app" / "view.ts").read_text(encoding="utf-8")
        self.assertIn('href="/case-study"', view)
        self.assertNotIn('href="/case-study.html"', view)
        self.assertTrue(
            self.summary["links"]["case_study"].endswith("/case-study"),
            self.summary["links"]["case_study"],
        )
        self.assertNotIn("/case-study.html", self.summary["links"]["case_study"])

    def test_case_study_does_not_hard_code_headline_results(self) -> None:
        renderer = ROOT / "web" / "src" / "case-study" / "renderCaseStudy.ts"
        self.assertTrue(renderer.is_file())
        source = renderer.read_text(encoding="utf-8")
        for literal in ("137.47", "30.933", "51.400", "72.56", "2.12"):
            self.assertNotIn(
                literal,
                source,
                f"headline value {literal} must come from summary.json, not the renderer",
            )
        self.assertIsNone(
            re.search(r"\bworld record\b", self.summary["headline"]["statement"], re.I)
        )


if __name__ == "__main__":
    unittest.main()
