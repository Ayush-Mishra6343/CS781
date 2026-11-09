"""
Inference, window aggregation, development threshold calibration, and submission packaging.
Preserves the exact notebook aggregation and output serialization semantics.
"""

import csv
import json
import math
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.model import CaseClassifier


def scores(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Computes precision, recall, f1 per class, plus unweighted macro-F1 and accuracy."""
    out = {}
    f1s = []
    for c in (0, 1):
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[str(c)] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": sum(a == c for a in y_true),
        }
        f1s.append(f)
    out["macro_f1"] = sum(f1s) / 2.0
    out["accuracy"] = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    return out


@torch.no_grad()
def predict_windows(
    model: CaseClassifier,
    rows: List[Dict[str, Any]],
    windows: List[np.ndarray],
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 16,
    mode: str = "all",
    threshold: float = 0.5,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    progress_fn: Optional[Callable[[str], None]] = None,
    progress_every: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Evaluates documents by aggregating logits over selected document windows.
    Modes:
      - 'all': all extracted windows
      - 'begin': first window only
      - 'middle': middle window only
      - 'end': last window only
    """
    model.eval()
    selected = []
    owners = []

    for i, ws in enumerate(windows):
        if mode == "begin":
            chosen = ws[:1]
        elif mode == "middle":
            chosen = ws[(len(ws) - 1) // 2 : (len(ws) - 1) // 2 + 1]
        elif mode == "end":
            chosen = ws[-1:]
        else:
            chosen = ws
        selected.extend(chosen)
        owners.extend([i] * len(chosen))

    flat = np.stack(selected, axis=0)
    if flat.ndim != 2:
        raise ValueError(f"Expected prediction windows with shape [windows, tokens], got {flat.shape}")

    all_logits = []
    infer_started = time.time()
    infer_total = math.ceil(len(flat) / batch_size)

    for batch_idx, j in enumerate(range(0, len(flat), batch_size), start=1):
        ids = torch.as_tensor(flat[j : j + batch_size], dtype=torch.long, device=device)
        mask = ids.ne(pad_token_id)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            z = model(ids, mask)
        all_logits.append(z.float().cpu().numpy())

        if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == infer_total:
            elapsed = time.time() - infer_started
            rate = batch_idx / max(elapsed, 1e-6)
            eta = (infer_total - batch_idx) / max(rate, 1e-6)
            msg = f"Predict {mode}: batch {batch_idx}/{infer_total} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg, flush=True)

    window_logits = np.concatenate(all_logits, axis=0)
    logits_sum = np.zeros((len(rows), 2), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int64)

    np.add.at(logits_sum, np.asarray(owners), window_logits)
    np.add.at(counts, np.asarray(owners), 1)

    logits = (logits_sum / counts[:, None]).astype(np.float32)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= threshold).astype(int).tolist()
    return logits, probs, preds


def calibrate_threshold(
    dev_y: List[int],
    dev_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any], List[Tuple[float, float, Dict[str, Any]]]]:
    """
    Sweeps thresholds over development probabilities to find the threshold maximizing macro-F1.
    Ties broken by proximity to 0.5.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.701, 0.01)

    threshold_results = []
    for t in thresholds:
        m = scores(dev_y, (dev_prob >= t).astype(int).tolist())
        threshold_results.append((m["macro_f1"], float(t), m))

    _, best_threshold, best_calibrated = max(threshold_results, key=lambda x: (x[0], -abs(x[1] - 0.5)))
    return best_threshold, best_calibrated, threshold_results


def run_context_ablations(
    model: CaseClassifier,
    dev_rows: List[Dict[str, Any]],
    dev_windows: List[np.ndarray],
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 16,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Runs context ablations: 'begin', 'middle', 'end', and 'all' on development set."""
    dev_y = [r["label"] for r in dev_rows]
    ablations = {}
    for mode in ("begin", "middle", "end", "all"):
        _, prob, pred = predict_windows(
            model=model,
            rows=dev_rows,
            windows=dev_windows,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
            mode=mode,
            threshold=0.5,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            progress_fn=progress_fn,
        )
        ablations[mode] = scores(dev_y, pred)
        msg = f"Context ablation [{mode}]: macro-F1={ablations[mode]['macro_f1']:.4f}, acc={ablations[mode]['accuracy']:.4f}"
        if progress_fn:
            progress_fn(msg)
        else:
            print(msg, flush=True)
    return ablations


def package_submission(
    test_rows: List[Dict[str, Any]],
    test_pred: List[int],
    output_dir: str,
) -> Tuple[Path, Path]:
    """
    Creates prediction.json and phase4_v4_lora_submission.zip according to the competition contract.
    Ensures strict validation of schema, ID alignment, and single-file root zip contents.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(test_rows) != len(test_pred):
        raise ValueError(f"Length mismatch: {len(test_rows)} rows vs {len(test_pred)} predictions")

    prediction = [{"id": str(row["id"]), "pred": int(pred)} for row, pred in zip(test_rows, test_pred)]

    # Validate schema
    for item in prediction:
        if set(item.keys()) != {"id", "pred"}:
            raise ValueError(f"Invalid keys in prediction item: {list(item.keys())}")
        if type(item["pred"]) is not int or item["pred"] not in (0, 1):
            raise ValueError(f"Invalid prediction value: {item['pred']}")

    pred_path = out_dir / "prediction.json"
    pred_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = out_dir / "phase4_v4_lora_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")

    # Roundtrip validation
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        if names != ["prediction.json"]:
            raise ValueError(f"ZIP must contain exactly one root file 'prediction.json', found: {names}")
        if z.testzip() is not None:
            raise ValueError("ZIP archive corrupted")
        roundtrip = json.loads(z.read("prediction.json").decode("utf-8"))

    if roundtrip != prediction:
        raise ValueError("ZIP roundtrip verification failed: unpacked content does not match original")

    return pred_path, zip_path


def save_test_probabilities(
    test_rows: List[Dict[str, Any]],
    test_probs: np.ndarray,
    test_preds: List[int],
    save_path: str,
):
    """Saves per-test-case probability distribution and discrete prediction to CSV."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prob_0", "prob_1", "prob", "pred"])
        for row, prob, pred in zip(test_rows, test_probs, test_preds):
            writer.writerow([row["id"], f"{1.0 - prob:.6f}", f"{prob:.6f}", f"{prob:.6f}", pred])
