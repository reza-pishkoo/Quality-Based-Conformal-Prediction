#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 --sweep_pold | --sweep_sizenew"
  exit 1
fi

NAME="$1"
case "$NAME" in
  --sweep_pold)
    CONFIG="configs/sweep_pold.yaml"
    ONLY_KEYS="data.p_old"        # restrict sweep to this key (seeds come from array)
    ;;
  --sweep_sizenew)
    CONFIG="configs/sweep_sizenew.yaml"
    ONLY_KEYS="data.size_new"
    ;;
  --sweep_rho2)
    CONFIG="configs/sweep_rho2.yaml"
    ONLY_KEYS="data.rho2"
    ;;
  *)
    echo "Unknown sweep name: $NAME"
    exit 1
    ;;
esac

RESULTS_DIR="/scratch/$USER/qbcp_runs/${NAME#--}"
SEEDS_N=5                   # how many seeds
PARTITION="debug"           # or debug/main
TIME="00:30:00"
CPUS=4
MEM="8G"

mkdir -p logs

sbatch --job-name="${NAME#--}" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --array=1-"$SEEDS_N" \
  --output="logs/%x_%A_%a.out" \
  --wrap "
module load python/3.11
source ~/qbcp_env/bin/activate
cd ~/Quality-Based-Conformal-Prediction
export PYTHONPATH=\$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python experiments/sweep.py \
  --config $CONFIG \
  --results-dir $RESULTS_DIR \
  --parallel-safe \
  --seed-from-array \
  --only-keys $ONLY_KEYS
"
