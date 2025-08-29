#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${1:?Usage: merge_results.sh EXP_NAME  (e.g., exp1_rho2)}"

# Base (same logic as exp_ours.sh)
SCRATCH_BASE="${SCRATCH:-/scratch2/$USER}"
[[ -d "$SCRATCH_BASE" ]] || SCRATCH_BASE="$HOME"

BASE="$SCRATCH_BASE/qbcp_runs/$EXP_NAME"
PART_DIR="$BASE/parts/$EXP_NAME"
OUT="$BASE/${EXP_NAME}_merged.csv"

shopt -s nullglob
files=("$PART_DIR"/part_seed*.csv)
if (( ${#files[@]} == 0 )); then
  echo "No part files under $PART_DIR"; exit 1
fi

# header from first file + all bodies
( head -n1 "${files[0]}"; for f in "${files[@]}"; do tail -n +2 "$f"; done ) > "$OUT"
echo "[merged] $OUT  lines: $(wc -l < "$OUT")"
