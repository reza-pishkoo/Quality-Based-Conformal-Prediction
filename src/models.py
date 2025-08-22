# src/models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

def train_old_model(X_old, y_old):
    """Train a model on the old dataset."""
    model = RandomForestClassifier(n_estimators=200, random_state=2025)
    model.fit(X_old, y_old)
    return model

def train_new_model(X_new, y_new):
    """Train a model on the new dataset."""
    model = RandomForestClassifier(n_estimators=200, random_state=2025)
    model.fit(X_new, y_new)
    return model

def train_quality_model(old_model, X_new, y_new, seed=2025, mc_samples=1, balanced=False):
    """
    Coin-flip mismatch quality model.
      Draw 𝑌̃ ~ Bernoulli(p_old(x)), set target = 1[y != 𝑌̃].
    If mc_samples > 1, use the mean of multiple flips as a soft target
    and regress to it, then expose a predict_proba-like interface.

    Args:
        old_model: classifier with predict_proba
        X_new, y_new: NEW-TRAIN split
        seed: RNG seed for reproducibility
        mc_samples: number of flips per x (>=1). 1 = classifier on noisy labels
        balanced: if True and mc_samples==1, use class_weight='balanced'
    """
    rng = np.random.default_rng(seed)
    p_old = old_model.predict_proba(X_new)[:, 1]

    if mc_samples == 1:
        y_tilde = (rng.random(len(p_old)) < p_old).astype(int)
        mismatch = (y_new != y_tilde).astype(int)

        kwargs = dict(n_estimators=200, random_state=seed)
        if balanced:
            kwargs["class_weight"] = "balanced"

        q = RandomForestClassifier(**kwargs)
        q.fit(X_new, mismatch)
        return q

    # mc_samples > 1: soft target in [0,1]
    flips = rng.random((len(p_old), mc_samples)) < p_old[:, None]
    mism = (y_new[:, None] != flips).astype(float)
    target = mism.mean(axis=1)  # ≈ E[ mismatch | x ]

    reg = GradientBoostingRegressor(random_state=seed)
    reg.fit(X_new, target)

    class QualityWrapper:
        def __init__(self, reg):
            self.reg = reg
        def predict_proba(self, X):
            g1 = np.clip(self.reg.predict(X), 0.0, 1.0)
            return np.c_[1.0 - g1, g1]
    return QualityWrapper(reg)
