# third_party/mapie_cp.py
import numpy as np
from mapie.classification import MapieClassifier

class SkRefEstimator:
    """
    Wrap your RefinedEstimator to look like an sklearn estimator.
    """
    def __init__(self, refined_estimator):
        self.ref = refined_estimator
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y=None):
        # already 'prefit' via your old/new/quality models
        return self

    def predict_proba(self, X):
        return self.ref.predict_proba(X)


class MapieCPClassifier:
    """
    Adapter exposing the same fit/predict_sets API we used before.
    Uses MAPIE for split conformal classification sets.
    """
    def __init__(self, refined_estimator, alpha=0.1, method="cumulated_score"):
        # method options include: "score", "cumulated_score"
        self.alpha = float(alpha)
        self._skref = SkRefEstimator(refined_estimator)
        # cv="prefit" => estimator is already fit; y_cal is used for calibration
        self._mapie = MapieClassifier(
            estimator=self._skref,
            method=method,
            cv="prefit"
        )

    def fit(self, X_cal, y_cal):
        self._mapie.fit(X_cal, y_cal)
        return self

    def predict_sets(self, X):
        # y_pred is ignored; y_ps is boolean inclusion mask for each class
        # include_last_label=True ensures the conservative tie handling.
        _, y_ps = self._mapie.predict(
            X,
            alpha=[self.alpha],
            include_last_label=True
        )
        # y_ps shape: (n, K, len(alpha)) -> take [:,:,0]
        sets_bool = y_ps[:, :, 0].astype(int)
        return sets_bool
