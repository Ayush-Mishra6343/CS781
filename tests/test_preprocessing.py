import unittest
import os
import sys
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestPreprocessing(unittest.TestCase):
    def test_results_csv_integrity(self):
        """Verify that official_submissions.csv contains all 7 phases with valid scores."""
        import csv
        csv_p = os.path.join(os.path.dirname(__file__), '..', 'results', 'official_submissions.csv')
        self.assertTrue(os.path.exists(csv_p))
        with open(csv_p, 'r', encoding='utf-8') as f:
            r = list(csv.reader(f))
        self.assertEqual(len(r), 8) # Header + 7 rows
        for row in r[1:]:
            score = float(row[2])
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
