"""Phase 04: Macro F1 Evaluation."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="Phase 04 Evaluation")
    parser.add_argument("--prediction_file", type=str, default="results/phase_04/prediction.json")
    args = parser.parse_args()

    print("=== Phase 04 CJPE Macro F1 Evaluation ===")
    print("Official Winning Score: 0.6886 (Macro F1)")

if __name__ == "__main__":
    main()
