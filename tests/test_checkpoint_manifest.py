"""Unit tests for the checkpoint manifest integrity and cryptographic verification."""

import csv
import hashlib
import os
import unittest


class TestCheckpointManifest(unittest.TestCase):
    """Verifies the checkpoint manifest format and cryptographic hashes."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.manifest_path = os.path.join(cls.repo_root, "checkpoints", "CHECKPOINT_MANIFEST.csv")

    def test_manifest_exists_and_populated(self):
        """Verifies the manifest exists and contains at least 128 entries."""
        self.assertTrue(os.path.exists(self.manifest_path), f"Manifest missing at {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertGreaterEqual(len(reader), 128, f"Expected at least 128 entries, got {len(reader)}")

    def test_required_columns_present(self):
        """Verifies that all required column headers exist."""
        required = {"checkpoint_name", "phase", "file_size_mb", "storage_status", "archive_location", "sha256_checksum"}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
        self.assertTrue(required.issubset(fieldnames), f"Missing fields: {required - fieldnames}")

    def test_no_synthetic_hashes(self):
        """Verifies no artificial ON_DISK_RECORD or placeholder hashes remain."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        invalid_hashes = []
        for r in reader:
            sha = r.get("sha256_checksum", "")
            name = r.get("checkpoint_name", "")
            if sha.startswith("ON_DISK_RECORD_"):
                invalid_hashes.append((name, sha))
            elif sha and (len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha.lower())):
                invalid_hashes.append((name, sha))

        self.assertEqual(len(invalid_hashes), 0, f"Found invalid or synthetic hashes: {invalid_hashes}")

    def test_repo_included_assets_cryptographic_match(self):
        """Verifies that any checkpoint marked 'Included in Repository' matches its real hash."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        for r in reader:
            if r.get("storage_status") == "Included in Repository":
                loc = r.get("archive_location", "")
                parts = loc.replace("\\", "/").split("/")
                full_path = os.path.join(self.repo_root, *parts)
                self.assertTrue(os.path.exists(full_path), f"File marked 'Included in Repository' not found: {full_path}")

                hasher = hashlib.sha256()
                with open(full_path, "rb") as fb:
                    while chunk := fb.read(1024 * 1024):
                        hasher.update(chunk)
                computed = hasher.hexdigest().lower()
                expected = r.get("sha256_checksum", "").lower()
                self.assertEqual(computed, expected, f"Hash mismatch for {loc}: computed={computed} vs expected={expected}")


if __name__ == "__main__":
    unittest.main()
