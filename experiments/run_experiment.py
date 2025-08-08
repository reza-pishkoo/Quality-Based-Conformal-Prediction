# experiments/run_experiment.py
import argparse, os, yaml
import numpy as np

from src.dataset import DatasetConfig, generate_datasets
from src.models import train_old_model, train_new_model, train_quality_model
from src.refine import RefinedEstimator
from src.metrics import coverage, avg_set_size, mask_stats
from src.utils import append_rows_csv, now_iso
from third_party.mapie_cp import MapieCPClassifier
from third_party.condtrust_adapter import run_condtrust_for_classifier

def deep_set(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def run(cfg):
    # --- dataset config
    dcfg = DatasetConfig(
        size_old=cfg["data"]["size_old"],
        size_new=cfg["data"]["size_new"],
        p_old=cfg["data"]["p_old"],
        p_new=cfg["data"]["p_new"],
        w_old=tuple(cfg["data"]["w_old"]),
        b_old=cfg["data"]["b_old"],
        w_new=tuple(cfg["data"]["w_new"]),
        b_new=cfg["data"]["b_new"],
        sigma_x=cfg["data"]["sigma_x"],
        eta_old=cfg["data"]["eta_old"],
        eta_new=cfg["data"]["eta_new"],
        tau_easy=cfg["data"]["tau_easy"],
        seed=cfg["seed"],
    )

    X_old, y_old, X_new, y_new, masks, _ = generate_datasets(dcfg)

    # --- train models
    f_old = train_old_model(X_old, y_old)
    f_new = train_new_model(X_new, y_new)
    q_old = train_quality_model(f_old, X_new, y_new)  # gamma(x) on NEW

    # --- refined estimator (ours)
    ref = RefinedEstimator(f_old, f_new, q_old)

    # --- split new data into calibration/test
    rng = np.random.default_rng(cfg["seed"])
    n_new = len(y_new)
    idx = rng.permutation(n_new)
    n_cal = int(cfg["split"]["cal_frac"] * n_new)
    cal_idx, te_idx = idx[:n_cal], idx[n_cal:]
    X_cal, y_cal = X_new[cal_idx], y_new[cal_idx]
    X_te,  y_te  = X_new[te_idx],  y_new[te_idx]
    easy_mask = masks["easy_new"][te_idx]
    hard_mask = masks["hard_new"][te_idx]

    # --- CP on our refined estimator (MAPIE)
    cp_method = cfg["cp"].get("method", "score")   # "score" for binary; "cumulated_score" for multiclass
    cp = MapieCPClassifier(ref, alpha=cfg["cp"]["alpha"], method=cp_method)
    cp.fit(X_cal, y_cal)
    sets_te = cp.predict_sets(X_te)  # (n, K) boolean array

    # --- evaluation for our method
    overall = {"coverage": coverage(y_te, sets_te), "avg_size": avg_set_size(sets_te), "n": int(len(y_te))}
    easy_stats = mask_stats(y_te, sets_te, easy_mask)
    hard_stats = mask_stats(y_te, sets_te, hard_mask)

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
        "cp_method": cp_method,
    }

    rows = [
        {"subset":"overall","method":"ours_refined",  **overall,    **common},
        {"subset":"easy_new","method":"ours_refined", **easy_stats, **common},
        {"subset":"hard_new","method":"ours_refined", **hard_stats, **common},
    ]

    # --- optional third-party baselines (conditional-conformal-trust)
    cmp_cfg = cfg.get("comparison", {})
    if cmp_cfg.get("enable", False) and cmp_cfg.get("impl", "condtrust") == "condtrust":
        which = cmp_cfg.get("classifiers", ["old","new"])

        def eval_sets(sets_bool):
            return {
                "overall": {"coverage": coverage(y_te, sets_bool), "avg_size": avg_set_size(sets_bool), "n": int(len(y_te))},
                "easy_new": mask_stats(y_te, sets_bool, easy_mask),
                "hard_new": mask_stats(y_te, sets_bool, hard_mask),
            }

        if "old" in which:
            sets_cond_old, sets_split_old = run_condtrust_for_classifier(f_old, X_cal, y_cal, X_te, y_te, alpha=cfg["cp"]["alpha"])
            for tag, sets in [("condtrust_old", sets_cond_old), ("split_old", sets_split_old)]:
                stats = eval_sets(sets)
                rows += [{ "subset":k, "method":tag, **v, **common } for k,v in stats.items()]

        if "new" in which:
            sets_cond_new, sets_split_new = run_condtrust_for_classifier(f_new, X_cal, y_cal, X_te, y_te, alpha=cfg["cp"]["alpha"])
            for tag, sets in [("condtrust_new", sets_cond_new), ("split_new", sets_split_new)]:
                stats = eval_sets(sets)
                rows += [{ "subset":k, "method":tag, **v, **common } for k,v in stats.items()]

    # --- append all rows to CSV
    header = list(rows[0].keys())
    csv_path = cfg["output_csv"]
    append_rows_csv(csv_path, rows, header)
    print(f"Appended {len(rows)} rows to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override as section.key=value")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    for item in args.set:
        k, v = item.split("=", 1)
        vv = v
        if v.lower() in {"true","false"}:
            vv = (v.lower() == "true")
        else:
            try: vv = int(v)
            except ValueError:
                try: vv = float(v)
                except ValueError:
                    pass
        deep_set(cfg, k, vv)

    run(cfg)
