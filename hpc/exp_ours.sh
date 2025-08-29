#!/usr/bin/env bash
set -euo pipefail

# Args from submit_experiment.sh
EXP_NAME="${1:?EXP_NAME}"
CONFIG="${2:?CONFIG}"
RHO2="${3:?rho2}"
SIZE_OLD="${4:?size_old}"
SIZE_NEW="${5:?size_new}"
ALPHA="${6:?alpha}"
SEED="${7:?seed}"

# NEW (optional): priors as bracket strings, e.g. [0.7,0.3]
COLD_STR="${8:-}"   # optional data.c_old
CNEW_STR="${9:-}"   # optional data.c_new

module load python/3.11
source "$HOME/qbcp_env/bin/activate"

cd "$HOME/Quality-Based-Conformal-Prediction"
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Writable base
SCRATCH_BASE="${SCRATCH:-/scratch2/$USER}"
[[ -d "$SCRATCH_BASE" ]] || SCRATCH_BASE="$HOME"

RESULTS_DIR="$SCRATCH_BASE/qbcp_runs/$EXP_NAME"
PART_DIR="$RESULTS_DIR/parts/$EXP_NAME"
mkdir -p "$PART_DIR"

OUT="$PART_DIR/part_seed${SEED}_rho2${RHO2}_sizeold${SIZE_OLD}_sizenew${SIZE_NEW}_alpha${ALPHA}.csv"

EXTRA=()
[[ -n "$COLD_STR" ]] && EXTRA+=(--set "data.c_old=${COLD_STR}")
[[ -n "$CNEW_STR" ]] && EXTRA+=(--set "data.c_new=${CNEW_STR}")

python experiments/run_experiment.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --set cp.alpha="$ALPHA" \
  --set data.rho2="$RHO2" \
  --set data.size_old="$SIZE_OLD" \
  --set data.size_new="$SIZE_NEW" \
  --set output_csv="$OUT" \
  "${EXTRA[@]}"
