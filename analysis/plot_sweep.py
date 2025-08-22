# analysis/plot_agg.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def detect_varying_key(df, candidates):
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1:
            return c
    # fallback: any varying numeric column not in excluded set
    excluded = {"coverage","avg_size","n","alpha","seed"}
    for c in df.columns:
        if c in excluded: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and s.nunique(dropna=True) > 1:
            return c
    return None

def agg_stats(df, xkey):
    g = (
        df.groupby(["subset", "method", xkey], dropna=True)
          .agg(coverage_mean=("coverage", "mean"),
               coverage_std =("coverage", "std"),
               size_mean    =("avg_size", "mean"),
               size_std     =("avg_size", "std"),
               count        =("coverage", "size"))
          .reset_index()
    )
    g["coverage_sem"] = g["coverage_std"] / np.sqrt(g["count"].clip(lower=1))
    g["size_sem"]     = g["size_std"] / np.sqrt(g["count"].clip(lower=1))
    return g.sort_values(["subset","method", xkey])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to merged sweep CSV")
    ap.add_argument("--xkey", default=None, help="Column to put on x-axis (auto-detect if omitted)")
    ap.add_argument("--alpha", type=float, default=None, help="Optional: filter by alpha value")
    ap.add_argument("--subset-order", nargs="*", default=["easy_new","overall","hard_new"])
    ap.add_argument("--out", default="figure.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # numeric coercion (keep this list broad; script is tolerant to missing cols)
    numeric_candidates = [
        "coverage","avg_size","n","alpha","seed",
        "size_old","size_new","sigma_x","eta_old","eta_new","tau_easy",
        "p_old","p_new","w_old_0","w_old_1","w_new_0","w_new_1","b_old","b_new","cal_frac",
        "rho2","lam","beta",
    ]
    df = coerce_numeric(df, [c for c in numeric_candidates if c in df.columns])

    if args.alpha is not None and "alpha" in df.columns:
        df = df[np.isclose(df["alpha"], args.alpha)]
        if df.empty:
            print("No rows after filtering by alpha.")
            return

    print("Rows:", len(df))
    print("Subsets:", sorted(df["subset"].unique()))
    print("Methods:", sorted(df["method"].unique()))

    xkey = args.xkey or detect_varying_key(df, ["size_new","rho2","p_old","p_new","cal_frac","lam","beta"])
    if xkey is None or xkey not in df.columns:
        print("Could not detect a varying x-axis column. Pass --xkey explicitly.")
        return

    df = df.dropna(subset=["subset","method","coverage","avg_size", xkey])
    if df.empty:
        print("No data to plot after dropping NaNs.")
        return

    G = agg_stats(df, xkey)

    # label map (use same as plot_sweep)
    label_map = {
        "ours_refined":   "Quality_Cond",
        "condtrust_old":  "Trust_Cond_old",
        "condtrust_new":  "Trust_Cond_new",
        "split_old":      "Std_CP_old",
        "split_new":      "Std_CP_new",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    methods = list(G["method"].unique()); methods.sort()
    color_for = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(methods)}
    label_for = defaultdict(lambda: None, label_map)

    subsets = args.subset_order
    ncols = len(subsets)
    fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 8), sharex=True)
    if ncols == 1:
        axes = np.array([[axes[0]],[axes[1]]])

    titles = {"easy_new":"Easy", "overall":"Full", "hard_new":"Hard"}
    # target coverage line if alpha present
    cov_target = None
    if "alpha" in df.columns and df["alpha"].notna().any():
        a = pd.to_numeric(df["alpha"], errors="coerce").dropna().unique()
        if len(a) == 1:
            cov_target = 1.0 - float(a[0])

    for j, sub in enumerate(subsets):
        ax_cov = axes[0, j]; ax_sz = axes[1, j]
        Gj = G[G["subset"] == sub]

        for m in methods:
            gm = Gj[Gj["method"] == m]
            if gm.empty: 
                continue
            x = gm[xkey].values
            y_cov = gm["coverage_mean"].values
            y_sz  = gm["size_mean"].values
            order = np.argsort(x)
            x, y_cov, y_sz = x[order], y_cov[order], y_sz[order]
            lbl = label_for[m] if label_for[m] else m
            ax_cov.plot(x, y_cov, marker="o", label=lbl, color=color_for[m])
            ax_sz.plot(x, y_sz, marker="o", label=lbl, color=color_for[m])

        ax_cov.set_title(titles.get(sub, sub))
        ax_cov.set_ylabel("Coverage")
        if cov_target is not None:
            ax_cov.axhline(cov_target, ls="--", lw=1, color="gray", alpha=0.7)
        ax_sz.set_ylabel("Set size")
        ax_sz.set_xlabel(xkey)
        ax_cov.grid(alpha=0.2)
        ax_sz.grid(alpha=0.2)
        if j == ncols - 1:
            ax_cov.legend(loc="best", fontsize=9)

    fig.suptitle(f"Conformal Comparison vs {xkey}", y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
