#!/usr/bin/env bash
#SBATCH -J qbcp_sweep
#SBATCH -p compute
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-${SEEDS_N}
#SBATCH -o ${RESULTS_DIR}/logs/%x_%A_%a.out
#SBATCH -e ${RESULTS_DIR}/logs/%x_%A_%a.err

set -euo pipefail

module load python/3.11
source ~/qbcp_env/bin/activate

cd ~/Quality-Based-Conformal-Prediction
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Make result dirs
mkdir -p "${RESULTS_DIR}/logs"

# One CSV per config lives at: ${RESULTS_DIR}/${EXP_NAME}.csv
# sweep.py will set output_csv to this when --results-dir is given and we do NOT use --parallel-safe
python experiments/sweep.py \
  --config "${CFG}" \
  --results-dir "${RESULTS_DIR}" \
  --seed-from-array \
  --only-keys "${ONLY_KEYS}"
