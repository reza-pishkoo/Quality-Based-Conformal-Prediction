#!/usr/bin/env bash
#SBATCH -p main
#SBATCH -t 04:00:00      # adjust as needed, up to 2-00:00:00 on main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

module load python/3.11
source ~/qbcp_env/bin/activate

cd ~/Quality-Based-Conformal-Prediction
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "${RESULTS_DIR}/logs"

python experiments/sweep.py \
  --config "${CFG}" \
  --results-dir "${RESULTS_DIR}" \
  --seed-from-array \
  --only-keys "${ONLY_KEYS}"
