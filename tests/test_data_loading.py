import unittest
import os
import sys
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDataLoading(unittest.TestCase):
    def test_config_files_exist(self):
        """Verify that all 7 phase configurations exist and parse as valid JSON."""
        import json
        for p in range(1, 8):
            cfg_p = os.path.join(os.path.dirname(__file__), '..', 'configs', f'phase_{p:02d}', 'config.json')
            self.assertTrue(os.path.exists(cfg_p), f"Missing config for Phase {p}")
            with open(cfg_p, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.assertIsInstance(cfg, dict)

if __name__ == '__main__':
    unittest.main()
