#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-exp1_rho2}"
XKEY="${2:-rho2}"
ALPHA="${3:-0.1}"

CSV="results_hpc/${EXP}/${EXP}_merged.csv"
OUT="figs/${EXP}.png"

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found at: $CSV"
  echo "Hint: ensure you downloaded to results_hpc/${EXP}/${EXP}_merged.csv"
  exit 1
fi

mkdir -p figs
python analysis/plot_agg.py --csv "$CSV" --xkey "$XKEY" --alpha "$ALPHA" --out "$OUT"
echo "Saved: $OUT"
