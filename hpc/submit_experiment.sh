#!/usr/bin/env bash
set -euo pipefail

############################################
# Choose which configuration to run
############################################
CONF=1   # 1: sweep rho2, 2: sweep size_old  (add more cases as you go)

############################################
# Slurm defaults (adjust if you need more)
############################################
PARTITION=${PARTITION:-main}
TIME=${TIME:-12:00:00}
MEM=${MEM:-16G}
CPUS=${CPUS:-4}

############################################
# Fixed paths
############################################
REPO="$HOME/Quality-Based-Conformal-Prediction"
CONFIG="configs/base_synthetic.yaml"

mkdir -p logs

############################################
# Grids (professor-style lists)
############################################
if [[ $CONF == 1 ]]; then
  # Exp 1: sweep rho2 at fixed sizes (your sweep_rho2)
  EXP_NAME="exp1_rho2"
  RHO2_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
  SIZEOLD_LIST=(10000)
  SIZENEW_LIST=(1000)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)     # ← 100 seeds
elif [[ $CONF == 2 ]]; then
  # Exp 2: sweep size_old at fixed rho2 (your sweep_sizeold)
  EXP_NAME="exp2_sizeold"
  RHO2_LIST=(0.6)
  SIZEOLD_LIST=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
  SIZENEW_LIST=(1000)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)     # ← 100 seeds
else
  echo "Unknown CONF=$CONF"; exit 1
fi

############################################
# Submit jobs (one sbatch per combo)
############################################
ORDP="sbatch --partition=$PARTITION --time=$TIME --mem=$MEM --cpus-per-task=$CPUS"

LOGDIR="logs/$EXP_NAME"
mkdir -p "$LOGDIR"

for SEED in $SEED_LIST; do
  for RHO2 in "${RHO2_LIST[@]}"; do
    for SO in "${SIZEOLD_LIST[@]}"; do
      for SN in "${SIZENEW_LIST[@]}"; do
        for A in "${ALPHA_LIST[@]}"; do

          JOBN="${EXP_NAME}_r${RHO2}_so${SO}_sn${SN}_a${A}_s${SEED}"
          OUTF="$LOGDIR/${JOBN}.out"
          ERRF="$LOGDIR/${JOBN}.err"

          CMD="hpc/exp_ours.sh $EXP_NAME $CONFIG $RHO2 $SO $SN $A $SEED"
          echo "$ORDP -J $JOBN -o $OUTF -e $ERRF $CMD"
          $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" $CMD

        done
      done
    done
  done
done
