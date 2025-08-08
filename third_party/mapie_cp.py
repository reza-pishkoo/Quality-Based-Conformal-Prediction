# third_party/mapie_cp.py
import numpy as np
from mapie.classification import MapieClassifier

class SkRefEstimator:
    """
    Wrap your RefinedEstimator to look like a scikit-learn estimator.
    Must provide: fit, predict, predict_proba, classes_.
    """
    def __init__(self, refined_estimator):
        self.ref = refined_estimator
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = None  # set in fit if X provided

    def fit(self, X=None, y=None):
        # Models are already trained; we just set shapes for sklearn compatibility.
        if X is not None:
            # try to record feature count for downstream checks
            self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        return self.ref.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class MapieCPClassifier:
    """
    Adapter exposing fit/predict_sets using MAPIE.
    For binary classification, use method='score' or 'lac'.
    """
    def __init__(self, refined_estimator, alpha=0.1, method="score"):
        self.alpha = float(alpha)
        self._skref = SkRefEstimator(refined_estimator)
        self._mapie = MapieClassifier(
            estimator=self._skref,
            method=method,   # 'score' or 'lac' for binary; 'cumulated_score' for multiclass
            cv="prefit"
        )

    def fit(self, X_cal, y_cal):
        # estimator is 'prefit', but MAPIE still expects a fitted estimator object
        # so we call fit() (no training, just sets shapes).
        self._skref.fit(X_cal, y_cal)
        self._mapie.fit(X_cal, y_cal)
        return self

    def predict_sets(self, X):
        # y_ps shape: (n, K, n_alpha)
        _, y_ps = self._mapie.predict(
            X,
            alpha=[self.alpha],
            include_last_label=True
        )
        return y_ps[:, :, 0].astype(int)
