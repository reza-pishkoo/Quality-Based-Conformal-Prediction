# src/metrics.py
import numpy as np

def coverage(y_true, pred_sets):
    """
    y_true: (n,) int labels in {0,1}
    pred_sets: list of sets or array of booleans shape (n, n_classes)
               If array, pred_sets[i, k] = 1 iff class k is included.
    Returns: float coverage in [0,1]
    """
    if isinstance(pred_sets, list):
        hits = np.array([y in S for y, S in zip(y_true, pred_sets)], dtype=float)
    else:
        # boolean array, y_true are indices
        hits = pred_sets[np.arange(len(y_true)), y_true].astype(float)
    return float(hits.mean())

def avg_set_size(pred_sets):
    """
    pred_sets: list of sets or boolean array (n, n_classes)
    """
    if isinstance(pred_sets, list):
        sizes = np.array([len(S) for S in pred_sets], dtype=float)
    else:
        sizes = pred_sets.sum(axis=1).astype(float)
    return float(sizes.mean())

def mask_stats(y_true, pred_sets, mask):
    """Compute coverage and avg |S| on a boolean mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"coverage": np.nan, "avg_size": np.nan, "n": 0}
    if isinstance(pred_sets, list):
        sub_sets = [pred_sets[i] for i in idx]
    else:
        sub_sets = pred_sets[idx]
    return {
        "coverage": coverage(y_true[idx], sub_sets),
        "avg_size": avg_set_size(sub_sets),
        "n": int(len(idx)),
    }
