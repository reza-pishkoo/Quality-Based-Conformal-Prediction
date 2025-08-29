#!/usr/bin/env bash
# Run this on YOUR LAPTOP (not on CARC)
# Usage: bash hpc/download_results.sh rho2 pishkoo@discovery1
set -euo pipefail
EXP="${1:-rho2}"
REMOTE="${2:-pishkoo@discovery1}"

REMOTE_PATH="/home1/$USER/qbcp_runs/sweep_${EXP}/sweep_${EXP}_merged.csv"
scp "${REMOTE}:${REMOTE_PATH}" .
echo "[downloaded] ./sweep_${EXP}_merged.csv"
