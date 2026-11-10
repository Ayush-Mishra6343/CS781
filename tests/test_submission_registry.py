"""Unit tests for the 47-submission registry and run artifact integrity."""

import csv
import json
import os
import unittest


class TestSubmissionRegistry(unittest.TestCase):
    """Verifies that the competition submission registry is consistent with on-disk artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.registry_path = os.path.join(cls.repo_root, "submissions", "submission_registry.csv")

    def test_registry_exists_and_has_47_entries(self):
        """Verifies exactly 47 submissions across all 7 phases are catalogued."""
        self.assertTrue(os.path.exists(self.registry_path), f"Registry missing: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 47, f"Expected 47 submissions, got {len(reader)}")

    def test_notebook_references_exist(self):
        """Verifies every notebook_reference in the registry points to an actual file."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        missing = []
        for row in reader:
            nb = row["notebook_reference"]
            full_path = os.path.join(self.repo_root, nb)
            if not os.path.exists(full_path):
                missing.append((row["submission_id"], nb))

        self.assertEqual(len(missing), 0, f"Missing notebook files: {missing}")

    def test_run_manifests_and_scores_match(self):
        """Verifies every run has a valid manifest and its score matches the registry."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        mismatches = []
        for row in reader:
            p = int(row["phase"])
            sid = row["submission_id"]
            reg_score = float(row["portal_score"])

            run_dir = os.path.join(self.repo_root, f"submissions/phase_{p:02d}/runs/{sid}")
            manifest_file = os.path.join(run_dir, "manifest.json")
            if not os.path.exists(manifest_file):
                mismatches.append((sid, "manifest_missing"))
                continue

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            manifest_score = float(manifest.get("portal_score", manifest.get("score", -1)))
            if abs(manifest_score - reg_score) > 1e-4:
                mismatches.append((sid, f"score mismatch: reg={reg_score} vs manifest={manifest_score}"))

        self.assertEqual(len(mismatches), 0, f"Run manifest mismatches: {mismatches}")

    def test_run_prediction_manifests_and_artifacts_exist(self):
        """Verifies every run directory contains required prediction and metric files."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        missing_artifacts = []
        for row in reader:
            p = int(row["phase"])
            sid = row["submission_id"]
            run_dir = os.path.join(self.repo_root, f"submissions/phase_{p:02d}/runs/{sid}")

            for req in ["manifest.json", "metrics.json", "prediction_manifest.json"]:
                p_file = os.path.join(run_dir, req)
                if not os.path.exists(p_file):
                    missing_artifacts.append((sid, req))

        self.assertEqual(len(missing_artifacts), 0, f"Missing run artifacts: {missing_artifacts}")


if __name__ == "__main__":
    unittest.main()
