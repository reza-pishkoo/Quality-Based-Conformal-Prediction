#!/usr/bin/env bash
# hpc/exp_ours.sh  — clean, stable env + paths that match merge_results.sh

# ---------------- Slurm env hygiene ----------------
set -euo pipefail
# If you keep SBATCH lines in a submitter script, fine; otherwise you can add:
# #SBATCH --partition=main
# #SBATCH --time=12:00:00
# #SBATCH --mem=16G
# #SBATCH --cpus-per-task=4
# #SBATCH --export=NONE

# ---------------- Load Python and venv (ONLY ONCE) ----------------
module purge
module load python/3.10.16
source /home1/pishkoo/Quality-Based-Conformal-Prediction/.venv/bin/activate

# Make 'src' importable
cd /home1/pishkoo/Quality-Based-Conformal-Prediction
export PYTHONPATH="$PWD"

# Thread sanity (avoid oversubscription)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---------------- Args from submit_experiment.sh ----------------
EXP_NAME="${1:?EXP_NAME}"
CONFIG="${2:?CONFIG}"
RHO2="${3:?rho2}"
SIZE_OLD="${4:?size_old}"
SIZE_NEW="${5:?size_new}"
ALPHA="${6:?alpha}"
SEED="${7:?seed}"

# Optional priors as bracket strings, e.g. [0.7,0.3]
COLD_STR="${8:-}"   # optional data.c_old
CNEW_STR="${9:-}"   # optional data.c_new

# ---------------- RESULTS PATH (matches merge_results.sh) ----------------
# Write under $HOME so hpc/merge_results.sh finds them at /home1/pishkoo/qbcp_runs/...
RESULTS_DIR="$HOME/qbcp_runs/$EXP_NAME"
PART_DIR="$RESULTS_DIR/parts/$EXP_NAME"
mkdir -p "$PART_DIR"

OUT="$PART_DIR/part_seed${SEED}_rho2${RHO2}_sizeold${SIZE_OLD}_sizenew${SIZE_NEW}_alpha${ALPHA}.csv"

# ---------------- CP method switch (optional) ----------------
# Choose one: aps_custom (your APS) or lac (MAPIE LAC). Default to aps_custom.
CP_METHOD="${CP_METHOD:-aps_custom}"

EXTRA=(
  --set "cp.alpha=$ALPHA"
  --set "data.rho2=$RHO2"
  --set "data.size_old=$SIZE_OLD"
  --set "data.size_new=$SIZE_NEW"
  --set "output_csv=$OUT"
  --set "cp.method=$CP_METHOD"
)
[[ -n "$COLD_STR" ]] && EXTRA+=(--set "data.c_old=${COLD_STR}")
[[ -n "$CNEW_STR" ]] && EXTRA+=(--set "data.c_new=${CNEW_STR}")

# ---------------- Debug prints (appear in .out) ----------------
echo "==== DEBUG ENV ===="
echo "HOST: $(hostname)"
echo "PWD : $PWD"
echo "PY  : $(which python)"
python --version
python - <<'PY'
import sys, os
print("sys.path[0]:", sys.path[0])
print("MAPIE?      :", end=" ")
try:
    import mapie, importlib.metadata as im
    print("OK", im.version('mapie'))
except Exception as e:
    print("N/A", e)
PY
echo "RESULT OUT  : $OUT"
echo "CP_METHOD   : $CP_METHOD"
echo "==============="

# ---------------- Run ----------------
python experiments/run_experiment.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  "${EXTRA[@]}"
