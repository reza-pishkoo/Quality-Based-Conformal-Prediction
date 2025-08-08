# experiments/run_experiment.py
import argparse, os, yaml
import numpy as np

# If you prefer not to export PYTHONPATH, uncomment:
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import DatasetConfig, generate_datasets
from src.models import train_old_model, train_new_model, train_quality_model
from src.refine import RefinedEstimator
from src.conformal.cp import SplitCPClassifier
from src.metrics import coverage, avg_set_size, mask_stats
from src.utils import append_rows_csv, now_iso


def deep_set(d, dotted_key, value):
    """Set cfg['a']['b']['c'] = value for 'a.b.c'."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def run(cfg):
    # --- dataset config
    dcfg = DatasetConfig(
        size_old = cfg["data"]["size_old"],
        size_new = cfg["data"]["size_new"],
        p_old    = cfg["data"]["p_old"],
        p_new    = cfg["data"]["p_new"],
        w_old    = tuple(cfg["data"]["w_old"]),
        b_old    = cfg["data"]["b_old"],
        w_new    = tuple(cfg["data"]["w_new"]),
        b_new    = cfg["data"]["b_new"],
        sigma_x  = cfg["data"]["sigma_x"],
        eta_old  = cfg["data"]["eta_old"],
        eta_new  = cfg["data"]["eta_new"],
        tau_easy = cfg["data"]["tau_easy"],
        seed     = cfg["seed"],
    )

    X_old, y_old, X_new, y_new, masks, _ = generate_datasets(dcfg)

    # --- train models
    f_old = train_old_model(X_old, y_old)
    f_new = train_new_model(X_new, y_new)
    q_old = train_quality_model(f_old, X_new, y_new)  # gamma(x) on NEW

    # --- refined estimator
    ref = RefinedEstimator(f_old, f_new, q_old)

    # --- split new data into calibration/test
    rng = np.random.default_rng(cfg["seed"])
    n_new = len(y_new)
    idx = rng.permutation(n_new)
    n_cal = int(cfg["split"]["cal_frac"] * n_new)
    cal_idx, te_idx = idx[:n_cal], idx[n_cal:]

    X_cal, y_cal = X_new[cal_idx], y_new[cal_idx]
    X_te,  y_te  = X_new[te_idx],  y_new[te_idx]

    # --- get refined probabilities
    p_cal = ref.predict_proba(X_cal)
    p_te  = ref.predict_proba(X_te)

    # --- conformal calibration
    cp = SplitCPClassifier(alpha=cfg["cp"]["alpha"]).fit(p_cal, y_cal)
    sets_te = cp.predict_sets(p_te)  # boolean (n, K)

    # --- evaluation
    overall = {
        "coverage": coverage(y_te, sets_te),
        "avg_size": avg_set_size(sets_te),
        "n": int(len(y_te)),
    }
    # easy/hard based on NEW masks
    easy_mask = masks["easy_new"][te_idx]
    hard_mask = masks["hard_new"][te_idx]
    easy_stats = mask_stats(y_te, sets_te, easy_mask)
    hard_stats = mask_stats(y_te, sets_te, hard_mask)

    # --- evaluation rows (long format)
    common = {
        "timestamp": now_iso(),
        "experiment": cfg["experiment_name"],
        "seed": cfg["seed"],
        "alpha": cfg["cp"]["alpha"],
        "size_old": dcfg.size_old,
        "size_new": dcfg.size_new,
        "sigma_x": dcfg.sigma_x,
        "eta_old": dcfg.eta_old,
        "eta_new": dcfg.eta_new,
        "tau_easy": dcfg.tau_easy,
        "p_old": dcfg.p_old,
        "p_new": dcfg.p_new,
        "w_old_0": dcfg.w_old[0], "w_old_1": dcfg.w_old[1],
        "b_old": dcfg.b_old,
        "w_new_0": dcfg.w_new[0], "w_new_1": dcfg.w_new[1],
        "b_new": dcfg.b_new,
        "cal_frac": cfg["split"]["cal_frac"],
    }

    rows = [
        {"subset": "overall",  "coverage": overall["coverage"],  "avg_size": overall["avg_size"],  "n": overall["n"],  **common},
        {"subset": "easy_new", "coverage": easy_stats["coverage"], "avg_size": easy_stats["avg_size"], "n": easy_stats["n"], **common},
        {"subset": "hard_new", "coverage": hard_stats["coverage"], "avg_size": hard_stats["avg_size"], "n": hard_stats["n"], **common},
    ]

    header = list(rows[0].keys())  # stable ordering
    csv_path = cfg["output_csv"]
    append_rows_csv(csv_path, rows, header)

    print("Appended 3 rows to:", csv_path)
    print("Overall:", overall)
    print("Easy:", easy_stats)
    print("Hard:", hard_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override as section.key=value")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # apply seed override
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    # apply key=value overrides (e.g., --set data.size_new=2000 --set cp.alpha=0.05)
    for item in args.set:
        k, v = item.split("=", 1)
        vv = v
        if v.lower() in {"true", "false"}:
            vv = (v.lower() == "true")
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    pass
        deep_set(cfg, k, vv)

    run(cfg)
