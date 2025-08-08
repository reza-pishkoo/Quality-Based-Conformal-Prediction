# src/main.py
from dataset import DatasetConfig, generate_datasets
from models import train_old_model, train_new_model, train_quality_model
from refine import RefinedEstimator
import numpy as np

def main():
    cfg = DatasetConfig(size_old=5000, size_new=1000, seed=2025)
    X_old, y_old, X_new, y_new, masks, _ = generate_datasets(cfg)

    f_old = train_old_model(X_old, y_old)
    f_new = train_new_model(X_new, y_new)
    q_old = train_quality_model(f_old, X_new, y_new)  # gamma(x) mismatch of old on NEW

    ref = RefinedEstimator(f_old, f_new, q_old)
    p_ref = ref.predict_proba(X_new)
    print("Refined probs shape:", p_ref.shape, " | mean positive prob:", np.mean(p_ref[:,1]))

if __name__ == "__main__":
    main()
