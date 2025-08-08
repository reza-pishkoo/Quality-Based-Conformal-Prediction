# src/models.py
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

def train_quality_model(old_model, X_new, y_new):
    """
    Train a quality model to predict mismatch probability of the old model.
    Uses probabilistic sampling from the old model's predicted probabilities.
    """
    # Predict probability of positive class
    pi_base = old_model.predict_proba(X_new)[:, 1]

    # Sample binary labels from probabilities
    hat_Y = (np.random.rand(len(pi_base)) < pi_base).astype(int)

    # Define mismatch label
    mismatch = (y_new != hat_Y).astype(int)

    # Train quality model
    quality_model = RandomForestClassifier(n_estimators=200, random_state=2025)
    quality_model.fit(X_new, mismatch)

    return quality_model

