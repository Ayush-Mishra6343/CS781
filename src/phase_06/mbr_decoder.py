"""Phase 6: Minimum Bayes Risk (MBR) Consensus Decoding for Legal Summarization.

Computes pair-wise BERTScore/token-F1 similarity matrix across candidate generations
and selects the candidate maximizing expected utility under the model posterior:

    y_MBR = argmax_{y_i \in C} \sum_{j=1}^K u(y_i, y_j)
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def pairwise_f1_similarity(text_a: str, text_b: str) -> float:
    """Computes exact token-level F1 similarity between two texts."""
    tokens_a = text_a.strip().lower().split()
    tokens_b = text_b.strip().lower().split()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    counts_a: Dict[str, int] = {}
    for t in tokens_a:
        counts_a[t] = counts_a.get(t, 0) + 1

    counts_b: Dict[str, int] = {}
    for t in tokens_b:
        counts_b[t] = counts_b.get(t, 0) + 1

    common = 0
    for t, count in counts_a.items():
        if t in counts_b:
            common += min(count, counts_b[t])

    if common == 0:
        return 0.0

    precision = common / len(tokens_a)
    recall = common / len(tokens_b)
    return (2 * precision * recall) / (precision + recall)


def compute_mbr_consensus(
    candidate_summaries: List[str],
    similarity_matrix_fn: Optional[Callable[[str, str], float]] = None,
) -> Tuple[str, int, np.ndarray]:
    """Computes pair-wise similarity across candidate generations and selects the consensus centroid."""
    valid_candidates = [c.strip() for c in candidate_summaries if c.strip()]
    if not valid_candidates:
        return "", 0, np.zeros(0)

    sim_fn = similarity_matrix_fn if similarity_matrix_fn is not None else pairwise_f1_similarity

    N = len(valid_candidates)
    sim_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            s = float(sim_fn(valid_candidates[i], valid_candidates[j]))
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    expected_utility = sim_matrix.mean(axis=1)
    best_idx = int(np.argmax(expected_utility))
    return valid_candidates[best_idx], best_idx, expected_utility


class BERTScoreMBR:
    """BERTScore-based MBR Consensus Centroid Selector."""

    def __init__(self, model_type: str = "roberta-large", num_layers: int = 17, device: str = "cpu"):
        self.model_type = model_type
        self.num_layers = num_layers
        self.device = device
        self._has_bert_score = False
        try:
            import bert_score
            self._has_bert_score = True
        except ImportError:
            self._has_bert_score = False

    def select_centroid(self, candidates: List[str]) -> Tuple[str, int, Dict[str, float]]:
        valid_cands = [c.strip() for c in candidates if c.strip()]
        if not valid_cands:
            return "", 0, {}

        k = len(valid_cands)
        if k == 1:
            return valid_cands[0], 0, {"expected_utility": 1.0}

        if self._has_bert_score:
            import bert_score
            cands_repeated = []
            refs_repeated = []
            for i in range(k):
                for j in range(k):
                    cands_repeated.append(valid_cands[i])
                    refs_repeated.append(valid_cands[j])

            P, R, F1 = bert_score.score(
                cands_repeated,
                refs_repeated,
                model_type=self.model_type,
                num_layers=self.num_layers,
                device=self.device,
                verbose=False,
                batch_size=32,
            )
            sim_matrix = F1.view(k, k).cpu().numpy()
            expected_utility = sim_matrix.mean(axis=1)
            best_idx = int(np.argmax(expected_utility))
            return valid_cands[best_idx], best_idx, {"expected_utility": float(expected_utility[best_idx])}

        # Fallback to token-F1 consensus
        best_cand, best_idx, expected_utility = compute_mbr_consensus(valid_cands)
        return best_cand, best_idx, {"expected_utility": float(expected_utility[best_idx])}
