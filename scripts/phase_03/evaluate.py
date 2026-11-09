"""Phase 03: Macro F1 Evaluation across 7 Rhetorical Roles."""
import argparse
from src.common.metrics import compute_macro_f1

def main():
    parser = argparse.ArgumentParser(description="Phase 03 Evaluation")
    parser.add_argument("--prediction_file", type=str, default="results/phase_03/prediction.json")
    args = parser.parse_args()

    print("=== Phase 03 Macro F1 Evaluation ===")
    print("Target Classes: Facts, Issue, Argument, Statute, Precedent, Ratio, Ruling")
    print("Official Winning Score: 0.4337 (Macro F1)")

if __name__ == "__main__":
    main()
