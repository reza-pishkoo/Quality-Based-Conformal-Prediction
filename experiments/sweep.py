# experiments/sweep.py
import argparse, os, sys, yaml, itertools, subprocess

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_EXP = os.path.join(PROJ_ROOT, "experiments", "run_experiment.py")

def as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]

def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        dk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, dk))
        else:
            out[dk] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--parallel-safe", action="store_true")
    ap.add_argument("--seed-from-array", action="store_true")
    ap.add_argument("--only-keys", default=None,
                    help="Comma-separated dotted keys to sweep (e.g. 'data.p_old,data.size_new'). "
                         "If omitted, all list-valued keys in YAML are swept.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        base = yaml.safe_load(f)

    exp_name = base.get("experiment_name", "exp")

    # seeds
    if args.seed_from_array:
        seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        seeds = [seed]
    else:
        seeds = list(base.get("seeds") or [])
        if not seeds:
            seeds = [base.get("seed", 2025)]

    flat = flatten_dict(base)
    # choose which keys to sweep
    if args.only_keys:
        keys_to_sweep = [k.strip() for k in args.only_keys.split(",") if k.strip()]
    else:
        keys_to_sweep = [k for k, v in flat.items() if isinstance(v, (list, tuple))]

    val_lists = [as_list(flat[k]) for k in keys_to_sweep] or [[]]
    combos = list(itertools.product(*val_lists, seeds))
    print(f"[sweep] sweeping keys: {keys_to_sweep}  | total runs: {len(combos)}")

    for combo in combos:
        seed = combo[-1]
        vals = combo[:-1]

        sets = [f"seed={seed}"]
        for k, v in zip(keys_to_sweep, vals):
            sets.append(f"{k}={v}")

        if args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
            if args.parallel_safe:
                tag = "_".join([f"{k.split('.')[-1]}{v}" for k, v in zip(keys_to_sweep, vals)]) or "base"
                part = os.path.join(args.results_dir, "parts", exp_name, f"part_seed{seed}_{tag}.csv")
                os.makedirs(os.path.dirname(part), exist_ok=True)
                sets.append(f"output_csv={part}")
            else:
                sets.append(f"output_csv={os.path.join(args.results_dir, f'{exp_name}.csv')}")

        cmd = [sys.executable, RUN_EXP, "--config", args.config]
        for s in sets:
            cmd.extend(["--set", s])
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
