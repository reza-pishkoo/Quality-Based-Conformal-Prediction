#!/usr/bin/env bash
# Common environment for CARC jobs

module purge
module load python/3.11
source ~/qbcp_env/bin/activate

# Be nice to shared BLAS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Writable results base
SCRATCH_BASE="${SCRATCH:-/scratch2/$USER}"
[ -d "$SCRATCH_BASE" ] || SCRATCH_BASE="$HOME"
export SCRATCH_BASE
export PYTHONPATH=$PWD
