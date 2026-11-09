import unittest
import torch
import torch.nn as nn
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestModelShapes(unittest.TestCase):
    def test_phase1_hybrid_transformer_shapes(self):
        """Verify Phase 1 HybridLegalLM forward pass tensor shapes."""
        from src.phase_01.model import HybridLegalLM
        model = HybridLegalLM(vocab_size=25000, n_embd=768, n_layer=2, n_head=12, n_kv_head=4, max_seq_len=128)
        x = torch.randint(0, 25000, (2, 32))
        logits, _ = model(x)
        self.assertEqual(logits.shape, (2, 32, 25000))

    def test_phase4_feedforward_chunk_shapes(self):
        """Verify Phase 4 FeedForward / SwiGLU chunk shapes."""
        from src.phase_04.model import FeedForward
        ffn = FeedForward(n_embd=256, multiplier=4, dropout=0.0)
        x = torch.randn(2, 16, 256)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 16, 256))

if __name__ == '__main__':
    unittest.main()
