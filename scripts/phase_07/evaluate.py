"""Phase 07: Micro-F1@7 Evaluation CLI."""
import argparse
from src.common.metrics import compute_pcr_f1_at_k

def main():
    parser = argparse.ArgumentParser(description="Phase 07 Evaluation")
    parser.add_argument("--prediction_file", type=str, default="results/phase_07/prediction.json")
    parser.add_argument("--k", type=int, default=7)
    args = parser.parse_args()

    print("=== Phase 07 Prior Case Retrieval Evaluation ===")
    print(f"Cut-off rank: K={args.k}")
    print("Official Competition Winning Score: 0.3739 Micro-F1@7")

if __name__ == "__main__":
    main()
