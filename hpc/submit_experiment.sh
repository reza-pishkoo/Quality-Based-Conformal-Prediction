#!/usr/bin/env bash
set -euo pipefail

############################################
# Choose which configuration to run
############################################
CONF=${1:-1}   # 1: rho2 | 2: size_old | 3: size_new | 4: c_old | 5: c_new

############################################
# Slurm defaults
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
# Grids
############################################
if [[ $CONF == 1 ]]; then
  EXP_NAME="exp1_rho2"
  RHO2_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
  SIZEOLD_LIST=(10000)
  SIZENEW_LIST=(1000)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)
elif [[ $CONF == 2 ]]; then
  EXP_NAME="exp2_sizeold"
  RHO2_LIST=(0.6)
  SIZEOLD_LIST=(5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
  SIZENEW_LIST=(1000)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)
elif [[ $CONF == 3 ]]; then
  EXP_NAME="exp3_sizenew"
  RHO2_LIST=(0.6)
  SIZEOLD_LIST=(10000)
  SIZENEW_LIST=(100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)
elif [[ $CONF == 4 ]]; then
  EXP_NAME="exp4_cold"
  RHO2_LIST=(0.6)
  SIZEOLD_LIST=(10000)
  SIZENEW_LIST=(1000)
  COLD_P_LIST=(0.5 0.6 0.7 0.8 0.9)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)
elif [[ $CONF == 5 ]]; then
  EXP_NAME="exp5_cnew"
  RHO2_LIST=(0.6)
  SIZEOLD_LIST=(10000)
  SIZENEW_LIST=(1000)
  CNEW_P_LIST=(0.5 0.6 0.7 0.8 0.9)
  ALPHA_LIST=(0.1)
  SEED_LIST=$(seq 1 100)
else
  echo "Unknown CONF=$CONF"; exit 1
fi

# <<< NEW: choose CP method for this sweep; default aps_custom (your APS), set to lac for standard CP
CP_METHOD="${CP_METHOD:-aps_custom}"

ORDP="sbatch --partition=$PARTITION --time=$TIME --mem=$MEM --cpus-per-task=$CPUS"

############################################
# Submit jobs
############################################
LOGDIR="logs/$EXP_NAME"
mkdir -p "$LOGDIR"

if [[ $CONF -le 3 ]]; then
  for SEED in $SEED_LIST; do
    for RHO2 in "${RHO2_LIST[@]}"; do
      for SO in "${SIZEOLD_LIST[@]}"; do
        for SN in "${SIZENEW_LIST[@]}"; do
          for A in "${ALPHA_LIST[@]}"; do
            JOBN="${EXP_NAME}_r${RHO2}_so${SO}_sn${SN}_a${A}_s${SEED}"
            OUTF="$LOGDIR/${JOBN}.out"
            ERRF="$LOGDIR/${JOBN}.err"
            CMD="hpc/exp_ours.sh $EXP_NAME $CONFIG $RHO2 $SO $SN $A $SEED"
            echo "$ORDP -J $JOBN -o $OUTF -e $ERRF --export=ALL,CP_METHOD=$CP_METHOD $CMD"   # <<< NEW
            $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" --export=ALL,CP_METHOD="$CP_METHOD" $CMD   # <<< NEW
          done
        done
      done
    done
  done

elif [[ $CONF == 4 ]]; then
  for SEED in $SEED_LIST; do
    for RHO2 in "${RHO2_LIST[@]}"; do
      for SO in "${SIZEOLD_LIST[@]}"; do
        for SN in "${SIZENEW_LIST[@]}"; do
          for A in "${ALPHA_LIST[@]}"; do
            for P in "${COLD_P_LIST[@]}"; do
              Q=$(awk "BEGIN{printf \"%.3f\", 1-$P}")
              COLD_STR="[$P,$Q]"
              JOBN="${EXP_NAME}_cold${P}_r${RHO2}_so${SO}_sn${SN}_a${A}_s${SEED}"
              OUTF="$LOGDIR/${JOBN}.out"; ERRF="$LOGDIR/${JOBN}.err"
              CMD="hpc/exp_ours.sh $EXP_NAME $CONFIG $RHO2 $SO $SN $A $SEED $COLD_STR"
              echo "$ORDP -J $JOBN -o $OUTF -e $ERRF --export=ALL,CP_METHOD=$CP_METHOD $CMD"   # <<< NEW
              $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" --export=ALL,CP_METHOD="$CP_METHOD" $CMD   # <<< NEW
            done
          done
        done
      done
    done
  done

elif [[ $CONF == 5 ]]; then
  for SEED in $SEED_LIST; do
    for RHO2 in "${RHO2_LIST[@]}"; do
      for SO in "${SIZEOLD_LIST[@]}"; do
        for SN in "${SIZENEW_LIST[@]}"; do
          for A in "${ALPHA_LIST[@]}"; do
            for P in "${CNEW_P_LIST[@]}"; do
              Q=$(awk "BEGIN{printf \"%.3f\", 1-$P}")
              CNEW_STR="[$P,$Q]"
              JOBN="${EXP_NAME}_cnew${P}_r${RHO2}_so${SO}_sn${SN}_a${A}_s${SEED}"
              OUTF="$LOGDIR/${JOBN}.out"; ERRF="$LOGDIR/${JOBN}.err"
              CMD="hpc/exp_ours.sh $EXP_NAME $CONFIG $RHO2 $SO $SN $A $SEED '' $CNEW_STR"
              echo "$ORDP -J $JOBN -o $OUTF -e $ERRF --export=ALL,CP_METHOD=$CP_METHOD $CMD"   # <<< NEW
              $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" --export=ALL,CP_METHOD="$CP_METHOD" $CMD   # <<< NEW
            done
          done
        done
      done
    done
  done
fi
