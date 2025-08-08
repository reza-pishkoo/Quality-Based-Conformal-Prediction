# analysis/plot_sweep.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SUBSET_ORDER = ["easy_new", "overall", "hard_new"]
SUBSET_TITLES = {"easy_new": "Easy", "overall": "Full", "hard_new": "Hard"}

# Choose a consistent order & label for methods (edit as you like)
METHOD_ORDER = [
    "ours_refine",
    "condtrust_old", "split_old",
    "condtrust_new", "split_new",
]
METHOD_LABEL = {
    "ours_refine":   "Quality_Cond",
    "condtrust_old": "Trust_Cond_old",
    "split_old":     "Std_CP_old",
    "condtrust_new": "Trust_Cond_new",
    "split_new":     "Std_CP_new",
}

def aggregate(df: pd.DataFrame, x_col: str):
    """
    Group by (subset, method, x_col) across seeds and compute mean & stderr.
    Returns a tidy DataFrame with columns:
        subset, method, x, coverage_mean, coverage_se, avg_size_mean, avg_size_se, count
    """
    # normalize column names
    df = df.rename(columns={x_col: "x"})
    # keep only columns we need
    need = ["subset", "method", "x", "coverage", "avg_size", "seed"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    grp = df.groupby(["subset", "method", "x"], as_index=False)
    def agg_fn(g):
        n = g["seed"].nunique() if "seed" in g else len(g)
        return pd.Series({
            "coverage_mean": g["coverage"].mean(),
            "coverage_se":   g["coverage"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            "avg_size_mean": g["avg_size"].mean(),
            "avg_size_se":   g["avg_size"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            "count":         n,
        })
    out = grp.apply(agg_fn).reset_index(drop=True)
    return out

def plot_panels(summary: pd.DataFrame, x_label: str, out_png: Path, title: str | None):
    # ensure ordering
    summary["subset"] = pd.Categorical(summary["subset"], categories=SUBSET_ORDER, ordered=True)
    summary["method"] = pd.Categorical(summary["method"],
                                       categories=[m for m in METHOD_ORDER if m in summary["method"].unique()],
                                       ordered=True)
    summary = summary.sort_values(["subset", "method", "x"])

    # set up 2×3 grid (Coverage top; Set size bottom)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "-", "-", "-", "-", "-", "-"]

    for col_i, subset in enumerate(SUBSET_ORDER):
        sub = summary[summary["subset"] == subset]
        if sub.empty:
            # leave blank if subset not present
            axes[0, col_i].set_visible(False)
            axes[1, col_i].set_visible(False)
            continue

        # coverage
        ax_cov = axes[0, col_i]
        for j, method in enumerate(sub["method"].cat.categories):
            d = sub[sub["method"] == method]
            if d.empty: 
                continue
            ax_cov.errorbar(d["x"], d["coverage_mean"], yerr=d["coverage_se"],
                            marker=markers[j % len(markers)], linestyle=linestyles[j % len(linestyles)],
                            label=METHOD_LABEL.get(method, method), capsize=3)
        ax_cov.set_title(SUBSET_TITLES.get(subset, subset))
        ax_cov.set_ylabel("Coverage")
        ax_cov.grid(True, alpha=0.25)
        ax_cov.axhline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.6)  # target alpha line

        # avg set size
        ax_sz = axes[1, col_i]
        for j, method in enumerate(sub["method"].cat.categories):
            d = sub[sub["method"] == method]
            if d.empty:
                continue
            ax_sz.errorbar(d["x"], d["avg_size_mean"], yerr=d["avg_size_se"],
                           marker=markers[j % len(markers)], linestyle=linestyles[j % len(linestyles)],
                           label=METHOD_LABEL.get(method, method), capsize=3)
        ax_sz.set_xlabel(x_label)
        ax_sz.set_ylabel("Set size")
        ax_sz.grid(True, alpha=0.25)

    # one legend for all
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the big sweep CSV (long format).")
    ap.add_argument("--x-col", required=True, help="Column to sweep on (e.g. size_new, p_old, p_new, etc.)")
    ap.add_argument("--out", default="figures/sweep.png", help="Output PNG path.")
    ap.add_argument("--title", default=None, help="Optional figure title.")
    ap.add_argument("--save-agg", default=None, help="Optional path to save aggregated CSV.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # be robust to stringy numerics in x-col
    if args.x_col in df.columns:
        try:
            df[args.x_col] = pd.to_numeric(df[args.x_col])
        except Exception:
            pass

    # filter to methods we know, but keep whatever exists
    # (no filtering necessary unless you want to trim)
    summary = aggregate(df, args.x_col)
    if args.save_agg:
        p = Path(args.save_agg)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(p, index=False)
        print(f"[saved] {p}")

    plot_panels(summary, x_label=args.x_col, out_png=Path(args.out), title=args.title)

if __name__ == "__main__":
    main()
