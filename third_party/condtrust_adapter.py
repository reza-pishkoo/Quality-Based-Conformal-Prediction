# third_party/condtrust_adapter.py
import os
import sys
import numpy as np

# --- make submodules importable without installing ---
TP_DIR = os.path.dirname(__file__)
CCT_DIR = os.path.join(TP_DIR, "conditional-conformal-trust")
TS_DIR  = os.path.join(TP_DIR, "TrustScore")
for p in (CCT_DIR, TS_DIR):
    if p not in sys.path:
        sys.path.append(p)

# imports from submodules
from trustscore.trustscore import TrustScore
from core.conformal import (
    compute_conformity_score_softmax,
    compute_sets_split,
    compute_sets_cond,
)

def _as_float_col(x):
    """Return a clean (n,1) float64 column with NaN/Inf handled."""
    a = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)

def _to_bool_sets(pred_sets_01):
    """Ensure (n, K) int/boolean array."""
    return pred_sets_01.astype(int)

def run_condtrust_for_classifier(clf, X_cal, y_cal, X_te, y_te, alpha=0.1):
    """
    Run Conditional-Conformal-Trust baseline using a given classifier (old or new).
    Returns:
        sets_cond, sets_split : (n, K) int arrays (1 means class included)
    Notes:
      - Uses calibration data to fit TrustScore (as a proxy for train).
      - If you prefer, pass a real training split to TrustScore instead.
    """
    # logits = class probabilities
    calib_logits = clf.predict_proba(X_cal)
    test_logits  = clf.predict_proba(X_te)

    # Fit TrustScore
    ts = TrustScore()
    ts.fit(X_cal, y_cal)

    # Build features phi = [trust, confidence] as strict float64
    y_pred_cal = clf.predict(X_cal)
    trust_cal  = _as_float_col(ts.get_score(X_cal, y_pred_cal))
    conf_cal   = _as_float_col(np.max(calib_logits, axis=1))
    phi_cal    = np.hstack([trust_cal, conf_cal]).astype(np.float64, copy=False)

    y_pred_te = clf.predict(X_te)
    trust_te  = _as_float_col(ts.get_score(X_te, y_pred_te))
    conf_te   = _as_float_col(np.max(test_logits, axis=1))
    phi_te    = np.hstack([trust_te, conf_te]).astype(np.float64, copy=False)

    # Conformity scores
    calib_scores, test_scores, test_scores_all = compute_conformity_score_softmax(
        calib_logits, test_logits, y_cal, y_te, temp_scaling=False
    )

    # Conditional CP sets (their method)
    _, pred_sets_cond, _ = compute_sets_cond(
        phi_cal, calib_scores, phi_te, test_scores, test_scores_all, alpha, rand=True
    )

    # Split CP sets (their baseline)
    _, pred_sets_split, _ = compute_sets_split(
        calib_scores, test_scores, test_scores_all, alpha
    )

    return _to_bool_sets(pred_sets_cond), _to_bool_sets(pred_sets_split)
