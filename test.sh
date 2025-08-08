CFG=configs/sweep_rho2.yaml
EXP_NAME=sweep_rho2
ONLY_KEYS=data.rho2
SEEDS_N=50
RESULTS_DIR=$HOME/qbcp_runs/$EXP_NAME
mkdir -p "$RESULTS_DIR/logs"

sbatch -p main --array=1-${SEEDS_N}   -o ${RESULTS_DIR}/logs/%x_%A_%a.out   -e ${RESULTS_DIR}/logs/%x_%A_%a.err   --export=ALL,CFG=${CFG},ONLY_KEYS=${ONLY_KEYS},SEEDS_N=${SEEDS_N},RESULTS_DIR=${RESULTS_DIR},EXP_NAME=${EXP_NAME}   slurm/sweep_array.sbatch
