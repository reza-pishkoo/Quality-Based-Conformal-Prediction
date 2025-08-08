# src/conformal/cp.py
import numpy as np

def _quantile(scores, q):
    # conservative quantile for split CP
    n = len(scores)
    k = int(np.ceil((n + 1) * q))
    k = min(max(k, 1), n)
    return np.partition(scores, k - 1)[k - 1]

class SplitCPClassifier:
    """
    Simple split conformal classifier for 2 classes (works for any K though).
    Calibrate on (probs_cal, y_cal) then form sets on probs_eval.
    """

    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        self.qhat = None

    def fit(self, probs_cal, y_cal):
        """
        probs_cal: (n_cal, K) predicted probabilities (from refined estimator)
        y_cal: (n_cal,) true labels
        """
        p_true = probs_cal[np.arange(len(y_cal)), y_cal]
        scores = 1.0 - p_true  # nonconformity
        # quantile level per split CP theory
        self.qhat = _quantile(scores, q=1.0 - self.alpha)
        return self

    def predict_sets(self, probs_eval):
        """
        probs_eval: (n, K) predicted probabilities
        Returns boolean array (n, K) with 1 iff class k is included in set.
        """
        if self.qhat is None:
            raise RuntimeError("Call fit() before predict_sets().")
        return (probs_eval >= (1.0 - self.qhat)).astype(int)
