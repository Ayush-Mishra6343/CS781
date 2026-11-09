import unittest
import os
import sys
import json
import tempfile
import zipfile
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase_01.model import ModelConfig, HybridLegalLM
from src.common.metrics import compute_macro_f1, compute_pcr_f1_at_k
from src.common.legal_losses import AsymmetricFocalLoss
from src.common.codabench_packager import package_submission
from src.phase_07.model import FastVectorizedBM25

class TestCommonModules(unittest.TestCase):
    def test_model_config_export(self):
        """Verify ModelConfig dataclass is exported and has correct defaults."""
        cfg = ModelConfig()
        self.assertEqual(cfg.vocab_size, 16000)
        self.assertEqual(cfg.n_embd, 768)
        self.assertEqual(cfg.n_layer, 12)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.n_kv_head, 4)
        self.assertEqual(cfg.max_seq_len, 1024)

    def test_macro_f1_computation(self):
        """Verify compute_macro_f1 computes exact harmonic mean across classes."""
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        pred_labels = np.array([0, 1, 2, 0, 1, 2])
        f1 = compute_macro_f1(true_labels, pred_labels, num_classes=3)
        self.assertAlmostEqual(f1, 1.0, places=4)

        # Imperfect predictions
        pred_labels_err = np.array([0, 1, 1, 0, 1, 2])
        f1_err = compute_macro_f1(true_labels, pred_labels_err, num_classes=3)
        self.assertLess(f1_err, 1.0)
        self.assertGreater(f1_err, 0.0)

    def test_asymmetric_focal_loss_forward(self):
        """Verify AsymmetricFocalLoss forward pass returns valid scalar tensor."""
        loss_fn = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0)
        logits = torch.randn(4, 7, requires_grad=True)
        targets = torch.randint(0, 2, (4, 7)).float()
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_codabench_packager(self):
        """Verify package_submission creates valid ZIP archive containing prediction.json."""
        data = {"case_001": [1, 2, 3], "case_002": [4, 5, 6]}
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test_submission.zip")
            package_submission(data, zip_path, filename_inside_zip="prediction.json")
            self.assertTrue(os.path.exists(zip_path))
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("prediction.json", zf.namelist())
                read_data = json.loads(zf.read("prediction.json").decode("utf-8"))
                self.assertEqual(read_data, data)

    def test_fast_bm25_search(self):
        """Verify FastVectorizedBM25 indexes candidate corpus and returns ranked results."""
        corpus = {
            "cand_1": "murder under section 302 indian penal code",
            "cand_2": "cheating and dishonestly inducing delivery of property section 420",
            "cand_3": "culpable homicide not amounting to murder section 304"
        }
        engine = FastVectorizedBM25(k1=1.5, b=0.75)
        engine.fit(corpus)
        hits = engine.search("murder section 302", top_k=2)
        self.assertEqual(len(hits), 2)
        top_cand_id, top_score = hits[0]
        self.assertEqual(top_cand_id, "cand_1")
        self.assertGreater(top_score, 0.0)

if __name__ == '__main__':
    unittest.main()
