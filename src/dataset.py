# src/dataset.py
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

@dataclass
class DatasetConfig:
    # sizes
    size_old: int
    size_new: int            # for sweeps you'll pass a scalar per run

    # mixture weights
    # old ~ c_old[0]*P0 + c_old[1]*P1
    # new ~ c_new[0]*P0 + c_new[1]*P2
    c_old: Tuple[float, float]      # (w_P0, w_P1)
    c_new: Tuple[float, float]      # (w_P0, w_P2)

    # P2 label Bernoulli prob
    rho2: float                     # P(Y=1|X~P2)

    # means (R^d; we assume 2D here but code generalizes to d=len(mu0))
    mu0: Tuple[float, ...]
    mu1: Tuple[float, ...]
    mu2: Tuple[float, ...]

    # (isotropic variance per component) and a small off-diagonal coupling
    # If d>2, we build var*I and add cov_off to all off-diagonals.
    var0: float
    var1: float
    var2: float
    cov_off: float

    # random seed
    seed: int

def _cov_matrix(d: int, var: float, cov_off: float) -> np.ndarray:
    C = np.eye(d) * var
    if cov_off != 0.0:
        off = np.ones((d, d)) - np.eye(d)
        C += cov_off * off
    return C

def _sample_gaussian(mean, var, cov_off, n, rng: np.random.Generator) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    d = mean.shape[0]
    cov = _cov_matrix(d, var, cov_off)
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)

def _sample_mixture(n: int, weights: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Return component ids for a 2-component mixture with probs 'weights' (sum to 1)."""
    w = np.asarray(weights, dtype=float)
    assert np.isclose(w.sum(), 1.0), "Mixture weights must sum to 1."
    return (rng.random(n) >= w[0]).astype(int)  # 0 with prob w[0], else 1

def generate_datasets(cfg: DatasetConfig):
    """
    Generates:
      OLD: mixture of P0 and P1 with weights c_old
      NEW: mixture of P0 and P2 with weights c_new

    Labels:
      P0 -> y=0
      P1 -> y=1
      P2 -> y ~ Bernoulli(rho2)

    Returns:
      X_old, y_old, X_new, y_new, masks, meta
    where
      masks['easy_new'] = (origin_new != 2)
      masks['hard_new'] = (origin_new == 2)
      meta['origin_new'] in {0:P0, 1:P1(not present in NEW), 2:P2}
    """
    rng = np.random.default_rng(cfg.seed)

    # ----- OLD -----
    comp_old = _sample_mixture(cfg.size_old, cfg.c_old, rng)  # 0 -> P0, 1 -> P1
    n0_old = int((comp_old == 0).sum())
    n1_old = cfg.size_old - n0_old

    X0_old = _sample_gaussian(cfg.mu0, cfg.var0, cfg.cov_off, n0_old, rng)
    X1_old = _sample_gaussian(cfg.mu1, cfg.var1, cfg.cov_off, n1_old, rng)
    X_old = np.vstack([X0_old, X1_old])

    # labels: P0 -> 0, P1 -> 1
    y_old = np.concatenate([np.zeros(n0_old, dtype=int),
                            np.ones(n1_old, dtype=int)])

    # shuffle OLD to remove any ordering
    idx_old = rng.permutation(cfg.size_old)
    X_old, y_old = X_old[idx_old], y_old[idx_old]

    # ----- NEW -----
    comp_new = _sample_mixture(cfg.size_new, cfg.c_new, rng)  # 0 -> P0, 1 -> P2
    n0_new = int((comp_new == 0).sum())
    n2_new = cfg.size_new - n0_new

    X0_new = _sample_gaussian(cfg.mu0, cfg.var0, cfg.cov_off, n0_new, rng)
    X2_new = _sample_gaussian(cfg.mu2, cfg.var2, cfg.cov_off, n2_new, rng)
    X_new = np.vstack([X0_new, X2_new])

    # labels: P0 -> 0, P2 -> Bernoulli(rho2)
    y0_new = np.zeros(n0_new, dtype=int)
    y2_new = rng.binomial(n=1, p=cfg.rho2, size=n2_new).astype(int)
    y_new = np.concatenate([y0_new, y2_new])

    # origin ids for NEW (0=P0, 2=P2)
    origin_new = np.concatenate([np.zeros(n0_new, dtype=int),
                                 2*np.ones(n2_new, dtype=int)])

    # shuffle NEW
    idx_new = rng.permutation(cfg.size_new)
    X_new, y_new, origin_new = X_new[idx_new], y_new[idx_new], origin_new[idx_new]

    masks = {
        "easy_new": (origin_new != 2),  # everything not P2
        "hard_new": (origin_new == 2),  # P2 samples
    }
    meta: Dict[str, Any] = {"origin_new": origin_new}

    return X_old, y_old, X_new, y_new, masks, meta
