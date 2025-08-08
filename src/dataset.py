# src/dataset.py
from dataclasses import dataclass
import numpy as np
from sklearn.utils import check_random_state

@dataclass
class DatasetConfig:
    # sizes
    size_old: int = 10000
    size_new: int = 2000
    # class priors
    p_old: float = 0.5
    p_new: float = 0.5
    # linear logit params (old vs new distribution)
    w_old: tuple = (1.0, 0.0)
    b_old: float = 0.0
    w_new: tuple = (0.6, 0.8)
    b_new: float = -0.2
    # feature noise (Gaussian)
    sigma_x: float = 1.0
    # label noise (extra flip prob)
    eta_old: float = 0.0
    eta_new: float = 0.0
    # easy/hard threshold on |logit|
    tau_easy: float = 1.0
    seed: int = 2025

def _make_split(n, p, w, b, sigma_x, eta, tau, rng):
    """
    Generate X in R^2, labels by Bernoulli(sigmoid(w·x + b)), optional extra flip eta.
    easy = |w·x + b| >= tau ; hard = otherwise
    """
    X = rng.normal(0.0, sigma_x, size=(n, 2))
    # allow slight class prior shift via intercept tweak
    # (we also keep p available if you want to extend)
    logit = X @ np.asarray(w) + b
    prob = 1 / (1 + np.exp(-logit))           # true P(Y=1|X)
    y = (rng.random(n) < prob).astype(int)
    # optional extra symmetric label noise
    if eta > 0:
        flips = rng.random(n) < eta
        y = np.where(flips, 1 - y, y)
    easy = np.abs(logit) >= tau
    hard = ~easy
    return X, y, prob, easy, hard

def generate_datasets(cfg: DatasetConfig):
    """
    Returns:
      X_old, y_old, X_new, y_new,
      masks: dict with 'easy_old','hard_old','easy_new','hard_new'
      prob_true: dict with true probs for analysis (old/new)
    """
    rng = check_random_state(cfg.seed)

    X_old, y_old, p_old_true, easy_old, hard_old = _make_split(
        cfg.size_old, cfg.p_old, cfg.w_old, cfg.b_old, cfg.sigma_x, cfg.eta_old, cfg.tau_easy, rng
    )
    X_new, y_new, p_new_true, easy_new, hard_new = _make_split(
        cfg.size_new, cfg.p_new, cfg.w_new, cfg.b_new, cfg.sigma_x, cfg.eta_new, cfg.tau_easy, rng
    )

    masks = {
        "easy_old": easy_old, "hard_old": hard_old,
        "easy_new": easy_new, "hard_new": hard_new,
    }
    prob_true = {"old": p_old_true, "new": p_new_true}
    return X_old, y_old, X_new, y_new, masks, prob_true
