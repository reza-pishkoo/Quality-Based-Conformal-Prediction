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


def _build_dataset_config(cfg):
    """
    Build DatasetConfig from the *mixture* schema only.
    Required fields in cfg['data']:
      size_old, size_new, c_old, c_new, rho2,
      mu0, mu1, mu2, var0, var1, var2, cov_off
    """
    data = cfg["data"]
    required = [
        "size_old", "size_new",
        "c_old", "c_new",
        "rho2",
        "mu0", "mu1", "mu2",
        "var0", "var1", "var2",
        "cov_off",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required data keys for mixture schema: {missing}")

    return DatasetConfig(
        size_old=int(data["size_old"]),
        size_new=int(data["size_new"]),
        c_old=tuple(data["c_old"]),
        c_new=tuple(data["c_new"]),
        rho2=float(data["rho2"]),
        mu0=tuple(data["mu0"]),
        mu1=tuple(data["mu1"]),
        mu2=tuple(data["mu2"]),
        var0=float(data["var0"]),
        var1=float(data["var1"]),
        var2=float(data["var2"]),
        cov_off=float(data["cov_off"]),
        seed=int(cfg["seed"]),
    )


def run(cfg):
    # --- dataset
    dcfg = _build_dataset_config(cfg)
    X_old, y_old, X_new, y_new, masks, meta = generate_datasets(dcfg)

    # --- split NEW into train / cal / test
    rng = np.random.default_rng(cfg["seed"])
    n_new = len(y_new)
    idx = rng.permutation(n_new)

    train_frac = cfg.get("split", {}).get("train_frac", 0.4)
    cal_frac   = cfg.get("split", {}).get("cal_frac",   0.3)
    assert 0 < train_frac < 1 and 0 < cal_frac < 1 and train_frac + cal_frac < 1, \
        "Invalid split fractions (need 0<train,cal<1 and train+cal<1)."

    n_train = int(train_frac * n_new)
    n_cal   = int(cal_frac   * n_new)

    tr_idx  = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    te_idx  = idx[n_train + n_cal:]

    X_tr,  y_tr  = X_new[tr_idx],  y_new[tr_idx]
    X_cal, y_cal = X_new[cal_idx], y_new[cal_idx]
    X_te,  y_te  = X_new[te_idx],  y_new[te_idx]
    easy_mask = masks["easy_new"][te_idx]
    hard_mask = masks["hard_new"][te_idx]

    # --- train models
    f_old = train_old_model(X_old, y_old)  # OLD only
    f_new = train_new_model(X_tr, y_tr)    # NEW-TRAIN

    q_kwargs = cfg.get("quality", {})
    q_old = train_quality_model(
        f_old, X_tr, y_tr,
        seed=cfg["seed"],
        mc_samples=q_kwargs.get("mc_samples", 1),
        balanced=q_kwargs.get("balanced", False),
    )

    # --- refined estimator (ours)
    lam  = cfg.get("refine", {}).get("lam", 0.5)
    beta = cfg.get("refine", {}).get("beta", 0.25)
    try:
        ref = RefinedEstimator(f_old, f_new, q_old, lam=lam, beta=beta)
    except TypeError:
        # in case src/refine.py hasn't been updated yet
        ref = RefinedEstimator(f_old, f_new, q_old)

    # --- CP on our refined estimator (MAPIE)
    cp_method = cfg["cp"].get("method", "lac")   # "score" or "cumulated_score"
    cp = MapieCPClassifier(ref, alpha=cfg["cp"]["alpha"], method=cp_method)
    cp.fit(X_cal, y_cal)
    sets_te = cp.predict_sets(X_te)

    # --- evaluation for our method
    overall = {"coverage": coverage(y_te, sets_te), "avg_size": avg_set_size(sets_te), "n": int(len(y_te))}
    easy_stats = mask_stats(y_te, sets_te, easy_mask)
    hard_stats = mask_stats(y_te, sets_te, hard_mask)

    # --- metadata (mixture schema only)
    common = {
        "timestamp": now_iso(),
        "experiment": cfg["experiment_name"],
        "seed": cfg["seed"],
        "alpha": cfg["cp"]["alpha"],
        "cp_method": cp_method,
        "train_frac": train_frac,
        "cal_frac": cal_frac,
        "lam": lam, "beta": beta,
        "quality_mc": q_kwargs.get("mc_samples", 1),
        "quality_balanced": q_kwargs.get("balanced", False),

        "size_old": dcfg.size_old,
        "size_new": dcfg.size_new,
        "c_old_0": dcfg.c_old[0], "c_old_1": dcfg.c_old[1],
        "c_new_0": dcfg.c_new[0], "c_new_1": dcfg.c_new[1],
        "rho2": dcfg.rho2,
        "mu0": tuple(dcfg.mu0),
        "mu1": tuple(dcfg.mu1),
        "mu2": tuple(dcfg.mu2),
        "var0": dcfg.var0, "var1": dcfg.var1, "var2": dcfg.var2,
        "cov_off": dcfg.cov_off,
    }

    rows = [
        {"subset": "overall",  "method": "ours_refined",  **overall,    **common},
        {"subset": "easy_new", "method": "ours_refined",  **easy_stats, **common},
        {"subset": "hard_new", "method": "ours_refined",  **hard_stats, **common},
    ]

    # --- optional baselines (conditional-conformal-trust)
    cmp_cfg = cfg.get("comparison", {})
    baseline_rows_before = len(rows)

    def _ensure_bool_sets(sets, n, k_expected=None):
        import numpy as np
        arr = np.asarray(sets)
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"Baseline sets have wrong shape: {arr.shape}, expected (n_test, K)")
        # allow ints/floats; convert to boolean membership
        if arr.dtype != np.bool_:
            arr = (arr > 0).astype(bool)
        if k_expected is not None and arr.shape[1] != k_expected:
            raise ValueError(f"Baseline sets have K={arr.shape[1]} classes, expected {k_expected}")
        return arr

    if cmp_cfg.get("enable", False) and cmp_cfg.get("impl", "condtrust") == "condtrust":
        which = cmp_cfg.get("classifiers", ["old", "new"])

        def eval_sets(sets_bool):
            return {
                "overall":  {"coverage": coverage(y_te, sets_bool), "avg_size": avg_set_size(sets_bool), "n": int(len(y_te))},
                "easy_new": mask_stats(y_te, sets_bool, easy_mask),
                "hard_new": mask_stats(y_te, sets_bool, hard_mask),
            }

        K = 2  # binary in your setup
        if "old" in which:
            sets_cond_old, sets_split_old = run_condtrust_for_classifier(
                f_old, X_cal, y_cal, X_te, y_te, alpha=cfg["cp"]["alpha"]
            )
            sets_cond_old  = _ensure_bool_sets(sets_cond_old,  len(y_te), K)
            sets_split_old = _ensure_bool_sets(sets_split_old, len(y_te), K)
            for tag, sets in [("condtrust_old", sets_cond_old), ("split_old", sets_split_old)]:
                stats = eval_sets(sets)
                rows += [{"subset": k, "method": tag, **v, **common} for k, v in stats.items()]

        if "new" in which:
            sets_cond_new, sets_split_new = run_condtrust_for_classifier(
                f_new, X_cal, y_cal, X_te, y_te, alpha=cfg["cp"]["alpha"]
            )
            sets_cond_new  = _ensure_bool_sets(sets_cond_new,  len(y_te), K)
            sets_split_new = _ensure_bool_sets(sets_split_new, len(y_te), K)
            for tag, sets in [("condtrust_new", sets_cond_new), ("split_new", sets_split_new)]:
                stats = eval_sets(sets)
                rows += [{"subset": k, "method": tag, **v, **common} for k, v in stats.items()]

    print(f"Baselines appended: {len(rows) - baseline_rows_before}")

    # --- append CSV
    header = list(rows[0].keys())
    csv_path = cfg["output_csv"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    append_rows_csv(csv_path, rows, header)
    print(f"Appended {len(rows)} rows to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
