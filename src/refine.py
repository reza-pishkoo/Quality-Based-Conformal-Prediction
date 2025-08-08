# src/refined_estimator.py
import numpy as np

class RefinedEstimator:
    """
    Combines old and new model predictions using a quality model,
    returning refined probabilities for conformal prediction.
    """

    def __init__(self, old_model, new_model, quality_model):
        """
        Parameters
        ----------
        old_model : object
            Model with predict_proba(X) method trained on the old dataset.
        new_model : object
            Model with predict_proba(X) method trained on the new dataset.
        quality_model : object
            Model that outputs gamma(x) in [0, 1], representing the mismatch
            of the old model. Can use predict() or predict_proba().
        """
        self.old_model = old_model
        self.new_model = new_model
        self.quality_model = quality_model

    def p_new_prime(self, X):
        """
        Modify the new model's predicted probabilities for the positive class:
        p_new_prime_class1 = 0.5 * p_new_class1 + 0.25
        Then rebuild the full probability vector.
        """
        p_new = self.new_model.predict_proba(X)  # shape: (n_samples, 2)
        p_new_class1 = p_new[:, 1]  # positive class (fraud)
        p_new_prime_class1 = 0.5 * p_new_class1 + 0.25
        p_new_prime = np.column_stack([
            1 - p_new_prime_class1,
            p_new_prime_class1
        ])
        return p_new_prime

    def predict_proba(self, X):
        """
        Compute refined probabilities:
        p_refined_class1 = gamma * p_new_prime_class1 + (1 - gamma) * p_old_class1
        Returns a full probability vector for binary classification.
        """
        p_old = self.old_model.predict_proba(X)  # shape: (n_samples, 2)
        p_new_p = self.p_new_prime(X)            # modified new probs

        # Get gamma(x) from quality model
        try:
            gamma = self.quality_model.predict_proba(X)[:, 1]
        except AttributeError:
            gamma = self.quality_model.predict(X)

        gamma = np.array(gamma).reshape(-1, 1)  # ensure column vector

        # Compute refined positive-class probabilities
        p_ref_class1 = gamma[:, 0] * p_new_p[:, 1] + (1 - gamma[:, 0]) * p_old[:, 1]

        # Return full probability vector
        p_ref = np.column_stack([1 - p_ref_class1, p_ref_class1])
        return p_ref
