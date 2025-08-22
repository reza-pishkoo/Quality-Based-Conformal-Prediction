# src/refine.py
import numpy as np

class RefinedEstimator:
    def __init__(self, old_model, new_model, quality_model, lam=0.5, beta=0.25):
        self.old_model = old_model
        self.new_model = new_model
        self.quality_model = quality_model
        self.lam = lam
        self.beta = beta

    def p_new_prime(self, X):
        p_new = self.new_model.predict_proba(X)[:, 1]
        p1 = np.clip(self.lam * p_new + self.beta, 0.0, 1.0)
        return np.column_stack([1 - p1, p1])

    def predict_proba(self, X):
        p_old = self.old_model.predict_proba(X)  # (n,2)
        p_new_p = self.p_new_prime(X)            # (n,2)
        try:
            gamma = self.quality_model.predict_proba(X)[:, 1]
        except AttributeError:
            gamma = self.quality_model.predict(X)
        gamma = np.asarray(gamma).reshape(-1, 1)
        p_ref1 = gamma[:, 0] * p_new_p[:, 1] + (1 - gamma[:, 0]) * p_old[:, 1]
        return np.column_stack([1 - p_ref1, p_ref1])
