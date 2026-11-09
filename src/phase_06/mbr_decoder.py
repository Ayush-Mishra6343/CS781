"""
Phase 6: Minimum Bayes Risk (MBR) Consensus Decoding for Legal Summarization.
Computes pair-wise BERTScore/ROUGE similarity matrix across candidate generations
and selects the candidate maximizing expected utility under the model posterior.
"""
import numpy as np

def compute_mbr_consensus(candidate_summaries, similarity_matrix_fn):
    N = len(candidate_summaries)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = similarity_matrix_fn(candidate_summaries[i], candidate_summaries[j])
    expected_utility = sim_matrix.sum(axis=1)
    best_idx = np.argmax(expected_utility)
    return candidate_summaries[best_idx], best_idx, expected_utility
