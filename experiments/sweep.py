# experiments/sweep.py
import argparse, os, sys, yaml, itertools, subprocess

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_EXP = os.path.join(PROJ_ROOT, "experiments", "run_experiment.py")

GRID_KEYS = [
    ("data", "size_new"),
    ("data", "p_old"),
    ("data", "p_new"),
    # add more keys if you want: ("cp","alpha"), ("data","eta_new"), ...
]

def as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--parallel-safe",
        action="store_true",
        help="Use per-run CSV parts rather than a single append file."
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        base = yaml.safe_load(f)

    # seeds list (fallback to single 'seed' if 'seeds' missing)
    seeds = base.get("seeds")
    seeds = as_list(seeds) if seeds is not None else [base.get("seed", 2025)]

    # collect sweep dimensions
    dim_lists = []
    for sect, key in GRID_KEYS:
        v = base.get(sect, {}).get(key, None)
        dim_lists.append(as_list(v) if v is not None else [None])

    combos = list(itertools.product(*dim_lists, seeds))
    print(f"[sweep] total runs: {len(combos)}")

    for *vals, seed in combos:
        # build overrides for run_experiment.py
        sets = []
        for (sect, key), val in zip(GRID_KEYS, vals):
            if val is not None:
                sets.append(f"{sect}.{key}={val}")
        sets.append(f"seed={seed}")

        # choose output file policy
        if args.parallel_safe:
            # write to per-run CSV part (unique file per combo)
            exp = base.get("experiment_name", "exp")
            size_new = next((v for (s,k), v in zip(GRID_KEYS, vals) if (s,k)==("data","size_new")), "NA")
            p_old    = next((v for (s,k), v in zip(GRID_KEYS, vals) if (s,k)==("data","p_old")), "NA")
            p_new    = next((v for (s,k), v in zip(GRID_KEYS, vals) if (s,k)==("data","p_new")), "NA")
            part = f"outputs/parts/{exp}/part_seed{seed}_sizeNew{size_new}_pOld{p_old}_pNew{p_new}.csv"
            sets.append(f"output_csv={part}")
            os.makedirs(os.path.dirname(part), exist_ok=True)
        # else: use output_csv from YAML (single accumulating file)

        cmd = [sys.executable, RUN_EXP, "--config", args.config]
        for s in sets:
            cmd.extend(["--set", s])

        print("[run]", " ".join(cmd))
        env = os.environ.copy()
        # If needed locally: env["PYTHONPATH"] = PROJ_ROOT
        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
