#!/usr/bin/env bash
# Sweep the Geman-McClure scale parameter for the covariance refit.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT=notes/20260923-covariant-ablation
mkdir -p "$OUT"
fail=0
for d in 1 2 3 5 10; do
  echo "=== gm delta=$d ==="
  if ! .venv/bin/python scripts/diag_covariant_sim3.py --weight-mode full \
      --kernel gm --kernel-delta "$d" --output "$OUT/gm_d$d.json"; then
    echo "FAIL gm delta=$d"
    fail=1
  fi
done
echo "SWEEP3_DONE fail=$fail"
exit "$fail"
