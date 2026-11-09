import time
import math
from collections import Counter
import numpy as np

class FastVectorizedBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_id_to_idx = {}
        self.N = 0
        self.index = {}  # term -> (np.array(doc_indices), np.array(term_scores))

    def fit(self, doc_tokens_dict):
        t0 = time.time()
        self.doc_ids = list(doc_tokens_dict.keys())
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.N = len(self.doc_ids)

        doc_lens = np.array([len(doc_tokens_dict[doc_id]) for doc_id in self.doc_ids], dtype=np.float32)
        avg_doc_len = float(np.mean(doc_lens)) if self.N > 0 else 1.0

        # Build term postings
        term_postings = {}
        for doc_idx, doc_id in enumerate(self.doc_ids):
            tf = Counter(doc_tokens_dict[doc_id])
            for term, count in tf.items():
                if term not in term_postings:
                    term_postings[term] = []
                term_postings[term].append((doc_idx, count))

        # Vectorize each term
        self.index = {}
        for term, postings in term_postings.items():
            df = len(postings)
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue

            doc_indices = np.array([p[0] for p in postings], dtype=np.int32)
            counts = np.array([p[1] for p in postings], dtype=np.float32)
            dls = doc_lens[doc_indices]

            # Vectorized BM25 formula
            numerator = counts * (self.k1 + 1.0)
            denominator = counts + self.k1 * (1.0 - self.b + self.b * (dls / avg_doc_len))
            scores = (idf * (numerator / denominator)).astype(np.float32)

            self.index[term] = (doc_indices, scores)

        print(f"FastVectorizedBM25 fit {self.N} docs with {len(self.index):,} unique terms in {time.time()-t0:.2f}s")

    def search(self, query_tokens, top_k=60, exclude_id=None):
        scores = np.zeros(self.N, dtype=np.float32)
        q_tf = Counter(query_tokens)

        for term, q_cnt in q_tf.items():
            if term in self.index:
                doc_idx, term_scores = self.index[term]
                # Fast numpy in-place accumulation
                scores[doc_idx] += term_scores

        if exclude_id is not None and exclude_id in self.doc_id_to_idx:
            scores[self.doc_id_to_idx[exclude_id]] = -1e9

        top_indices = np.argpartition(-scores, min(top_k, self.N - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > -1e8]
