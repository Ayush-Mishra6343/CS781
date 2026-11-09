"""Evaluation, Metric Calculation, and Threshold Calibration for Phase 5 LSI.

Implements:
1. Exact official metrics: Macro-F1, Micro-F1, and Subset Accuracy.
2. Fixed 100-label metric calculation (preventing label truncation on local splits).
3. Empirical threshold calibration (global sweep + support-regularized per-label shrinkage).
4. Prediction formatter and validator matching Codabench submission requirements.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_lsi_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    num_labels: int = 100,
) -> Dict[str, Any]:
    """Computes Macro-F1, Micro-F1, and Subset Accuracy at a given threshold.

    y_true: [N, 100] binary {0, 1}
    y_pred_probs: [N, 100] float probabilities [0, 1]
    """
    y_pred = (y_pred_probs >= threshold).astype(np.uint8)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)

    # Per-label metrics
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    label_support = y_true.sum(axis=0)

    # Cardinality metrics
    pred_counts = y_pred.sum(axis=1)
    true_counts = y_true.sum(axis=1)
    empty_pred_rate = float(np.mean(pred_counts == 0))

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "subset_accuracy": float(subset_acc),
        "threshold": float(threshold),
        "mean_predicted_cardinality": float(np.mean(pred_counts)),
        "median_predicted_cardinality": float(np.median(pred_counts)),
        "mean_true_cardinality": float(np.mean(true_counts)),
        "empty_prediction_rate": empty_pred_rate,
        "per_label_f1": per_label_f1.tolist(),
        "per_label_support": label_support.tolist(),
    }


def find_best_global_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold_grid: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Sweeps threshold values to find the global threshold maximizing Macro-F1 on held-out data."""
    if threshold_grid is None:
        threshold_grid = [round(x, 2) for x in np.linspace(0.05, 0.90, 35)]

    best_thresh = 0.5
    best_macro = -1.0
    best_metrics = None

    for thresh in threshold_grid:
        m = compute_lsi_metrics(y_true, y_pred_probs, threshold=thresh)
        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]
            best_thresh = thresh
            best_metrics = m

    return best_thresh, best_macro, best_metrics


def calibrate_regularized_per_label_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    base_global_threshold: float,
    min_support: int = 50,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Performs shrinkage per-label threshold search for frequent classes.

    Rare classes (support < min_support) remain strictly anchored to base_global_threshold
    to prevent severe overfitting on sparse validation labels.
    """
    N, num_labels = y_true.shape
    label_supports = y_true.sum(axis=0)
    per_label_thresh = np.full(num_labels, base_global_threshold, dtype=np.float32)

    delta_grid = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    for label_idx in range(num_labels):
        if label_supports[label_idx] >= min_support:
            y_t = y_true[:, label_idx]
            y_p = y_pred_probs[:, label_idx]
            best_f1 = -1.0
            best_t = base_global_threshold
            for delta in delta_grid:
                candidate_t = float(np.clip(base_global_threshold + delta, 0.05, 0.95))
                pred_binary = (y_p >= candidate_t).astype(np.uint8)
                score = f1_score(y_t, pred_binary, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = candidate_t
            per_label_thresh[label_idx] = best_t

    # Compute overall metrics with per-label thresholds
    y_pred = (y_pred_probs >= per_label_thresh).astype(np.uint8)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    subset_acc = float(accuracy_score(y_true, y_pred))

    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_acc,
        "per_label_thresholds": per_label_thresh.tolist(),
    }
    return per_label_thresh, macro_f1, metrics


def format_predictions(
    ids: List[str],
    pred_probs: np.ndarray,
    thresholds: Any,
) -> List[Dict[str, Any]]:
    """Formats predictions into [{"id": "...", "pred": [0, 2, ...]}, ...]."""
    if isinstance(thresholds, (int, float)):
        binary_preds = (pred_probs >= thresholds).astype(np.uint8)
    else:
        thresh_arr = np.array(thresholds)
        binary_preds = (pred_probs >= thresh_arr).astype(np.uint8)

    records = []
    for doc_id, row in zip(ids, binary_preds):
        active_labels = [int(i) for i in np.where(row == 1)[0]]
        records.append({"id": str(doc_id), "pred": active_labels})
    return records
