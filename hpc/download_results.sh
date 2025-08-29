#!/usr/bin/env bash
# Run this on YOUR LAPTOP (not on CARC)
# Usage: bash hpc/download_results.sh exp1_rho2 pishkoo@discovery1 [local_dir]
set -euo pipefail

EXP="${1:?exp dir name, e.g. exp1_rho2}"   # e.g., exp1_rho2
REMOTE="${2:-pishkoo@discovery1}"          # user@host you ssh to
LOCAL_DIR="${3:-results_hpc/$EXP}"         # where to save locally

REMOTE_FILE="~/qbcp_runs/${EXP}/${EXP}_merged.csv"

mkdir -p "$LOCAL_DIR"
scp "${REMOTE}:${REMOTE_FILE}" "${LOCAL_DIR}/"
echo "[downloaded] ${LOCAL_DIR}/${EXP}_merged.csv"

# If you ever want the whole folder (parts + merged), use rsync instead:
# rsync -avP "${REMOTE}:~/qbcp_runs/${EXP}/" "${LOCAL_DIR}/"
