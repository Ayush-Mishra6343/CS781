"""
Comprehensive Legal NLP Evaluation Suite:
- Macro & Micro F1 across arbitrary classes
- Precision@K & Micro-F1@K for Prior Case Retrieval
- Exact span overlap and boundary evaluation
"""
import numpy as np

def compute_macro_f1(true_labels, pred_labels, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((true_labels == c) & (pred_labels == c))
        fp = np.sum((true_labels != c) & (pred_labels == c))
        fn = np.sum((true_labels == c) & (pred_labels != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def compute_pcr_f1_at_k(ground_truth_dict, predictions_dict, k=7):
    # ground_truth_dict: {qid: set([doc_ids])}
    # predictions_dict: {qid: [top_doc_ids_ranked]}
    total_tp = 0
    total_pred = 0
    total_gold = 0
    for qid, gold_set in ground_truth_dict.items():
        preds = predictions_dict.get(qid, [])[:k]
        tp = len(set(preds) & set(gold_set))
        total_tp += tp
        total_pred += len(preds)
        total_gold += len(gold_set)
    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    return micro_f1, micro_prec, micro_rec
