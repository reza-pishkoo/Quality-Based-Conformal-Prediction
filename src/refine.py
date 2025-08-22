# src/refine.py
import numpy as np

def _two_col_proba(model, X):
    """
    Return probabilities as (n, 2) for classes [0, 1], even if the model was
    trained on a single class (so its predict_proba(X) is (n, 1)).
    If predict_proba is unavailable, fall back to a degenerate distribution
    from hard predictions.
    """
    # Try predict_proba
    P = None
    classes = None
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X)
        classes = getattr(model, "classes_", None)

    if P is None:
        # Fallback: hard predictions -> degenerate probs
        yhat = model.predict(X)
        n = len(yhat)
        out = np.zeros((n, 2), dtype=float)
        out[np.arange(n), (yhat != 0).astype(int)] = 1.0
        return out

    n = P.shape[0]
    out = np.zeros((n, 2), dtype=float)

    if classes is None:
        # Heuristic fallback: assume columns are [p0, p1] if K=2; if K=1 assume it's p1
        if P.ndim == 2 and P.shape[1] == 2:
            return np.clip(P, 0.0, 1.0)
        if P.ndim == 2 and P.shape[1] == 1:
            out[:, 1] = P[:, 0]
            out[:, 0] = 1.0 - out[:, 1]
            return np.clip(out, 0.0, 1.0)
        raise ValueError(f"Unsupported probability shape: {P.shape}")

    classes = np.asarray(classes)
    if classes.size == 2:
        # Map columns by class label to [0,1]
        idx0 = int(np.where(classes == 0)[0][0]) if (classes == 0).any() else None
        idx1 = int(np.where(classes == 1)[0][0]) if (classes == 1).any() else None
        if idx0 is not None:
            out[:, 0] = P[:, idx0]
        if idx1 is not None:
            out[:, 1] = P[:, idx1]
        # If one is missing (shouldn't happen with size==2), fill complement
        if idx0 is None and idx1 is not None:
            out[:, 0] = 1.0 - out[:, 1]
        if idx1 is None and idx0 is not None:
            out[:, 1] = 1.0 - out[:, 0]
        return np.clip(out, 0.0, 1.0)

    if classes.size == 1:
        # Degenerate model: all mass on the single seen class
        c = int(classes[0])
        if c == 0:
            out[:, 0] = P[:, 0]
            out[:, 1] = 0.0
        else:
            out[:, 1] = P[:, 0]
            out[:, 0] = 0.0
        return out

    # K>2 not supported by this project
    raise ValueError(f"Unexpected number of classes: {classes.size}")

class RefinedEstimator:
    """
    Combines old and new model predictions using a quality model,
    returning refined probabilities for conformal prediction.
    """

    def __init__(self, old_model, new_model, quality_model):
        self.old_model = old_model
        self.new_model = new_model
        self.quality_model = quality_model

    def p_new_prime(self, X):
        """
        Modify the new model's predicted probabilities for the positive class:
        p_new_prime_class1 = 0.5 * p_new_class1 + 0.25
        Then rebuild the full probability vector.
        """
        p_new = _two_col_proba(self.new_model, X)       # (n, 2)
        p1 = p_new[:, 1]
        p1_prime = 0.5 * p1 + 0.25
        p1_prime = np.clip(p1_prime, 0.0, 1.0)
        return np.c_[1.0 - p1_prime, p1_prime]

    def predict_proba(self, X):
        """
        p_refined_class1 = gamma * p_new_prime_class1 + (1 - gamma) * p_old_class1
        Returns a full probability vector for binary classification.
        """
        p_old = _two_col_proba(self.old_model, X)       # (n, 2)
        p_newp = self.p_new_prime(X)                    # (n, 2)
        # gamma from quality model (also robust to single-class training)
        g = _two_col_proba(self.quality_model, X)       # (n, 2)
        gamma = g[:, 1]                                 # P(mismatch=1 | x)

        p_ref1 = gamma * p_newp[:, 1] + (1.0 - gamma) * p_old[:, 1]
        p_ref1 = np.clip(p_ref1, 0.0, 1.0)
        return np.c_[1.0 - p_ref1, p_ref1]
