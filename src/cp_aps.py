# src/cp_aps.py
import numpy as np

class ProbabilityAccumulator:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(self.K)
        self.prob_sort = -np.sort(-prob, axis=1)
        self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        if alpha > 0:
            L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()
        else:
            L = (self.Z.shape[1] - 1) * np.ones((self.n,), dtype=int)

        if epsilon is not None:
            Z_excess = self.Z[np.arange(self.n), L] - (1.0 - alpha)
            p_remove = Z_excess / self.prob_sort[np.arange(self.n), L]
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                L[i] = max(0, L[i]-1) if not allow_empty else L[i]-1

        S = [self.order[i, np.arange(0, L[i] + 1)] for i in range(self.n)]
        return S

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        ranks = self.ranks[np.arange(len(Y)), Y]
        prob_cum = self.Z[np.arange(len(Y)), ranks]
        prob = self.prob_sort[np.arange(len(Y)), ranks]
        alpha_max = 1.0 - prob_cum
        alpha_max += (prob * (epsilon if epsilon is not None else 1.0))
        alpha_max = np.minimum(alpha_max, 1.0)
        return alpha_max

class ConformalPredictorAPS:
    """
    APS split conformal using cumulative probability + randomization.
    Expects predictor with .predict_proba(X) -> (n,K) probabilities.
    """
    def __init__(self, predictor, random_state=2025):
        self.predictor = predictor
        self.rng = np.random.default_rng(random_state)
        self.q = None

    def _proba(self, X):
        P = self.predictor.predict_proba(X)
        if P.ndim == 1:  # binary vector -> expand
            P = np.vstack([1 - P, P]).T
        return P

    def fit(self, X_cal, y_cal, alpha):
        pi_cal = self._proba(X_cal)
        pa_cal = ProbabilityAccumulator(pi_cal)
        eps = self.rng.uniform(0.0, 1.0, size=len(y_cal))
        alpha_max = pa_cal.calibrate_scores(y_cal.astype(int), epsilon=eps)
        nonconformity = 1.0 - alpha_max
        self.q = np.quantile(nonconformity, 1 - alpha)
        return self

    def predict_sets(self, X, allow_empty=True):
        pi_test = self._proba(X)
        pa_test = ProbabilityAccumulator(pi_test)
        eps = self.rng.uniform(0.0, 1.0, size=pi_test.shape[0])
        # APS threshold uses (1 - q)
        pred_sets = pa_test.predict_sets(1 - self.q, epsilon=eps, allow_empty=allow_empty)
        return pred_sets

    def predict_sets_bool(self, X, allow_empty=True):
        S = self.predict_sets(X, allow_empty=allow_empty)
        K = self._proba(X).shape[1]
        out = np.zeros((len(S), K), dtype=bool)
        for i, s in enumerate(S):
            out[i, np.asarray(s)] = True
        return out
