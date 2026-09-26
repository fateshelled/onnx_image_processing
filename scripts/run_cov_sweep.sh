#!/usr/bin/env bash
# Sweep sigma_px (full model) and the fixed depth-to-lateral ratio.
# Writes one JSON per condition to notes/20260923-covariant-ablation/.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT=notes/20260923-covariant-ablation
mkdir -p "$OUT"
fail=0
for sp in 0.5 2 5; do
  echo "=== full sigma_px=$sp ==="
  if ! .venv/bin/python scripts/diag_covariant_sim3.py --weight-mode full \
    --sigma-px "$sp" --output "$OUT/sigma_px_$sp.json"; then
    echo "FAIL sigma_px=$sp"
    fail=1
  fi
done
for r in 10 31.6 100 316 1000; do
  echo "=== fixed_depth ratio=$r ==="
  if ! .venv/bin/python scripts/diag_covariant_sim3.py --weight-mode fixed_depth \
    --depth-ratio "$r" --output "$OUT/fixed_depth_$r.json"; then
    echo "FAIL ratio=$r"
    fail=1
  fi
done
echo "SWEEP_DONE fail=$fail"
exit "$fail"
