#!/usr/bin/env bash
# Robust kernels (Cauchy/Tukey) and the parallax-depth clamp at ratio=31.6.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT=notes/20260923-covariant-ablation
mkdir -p "$OUT"
fail=0
run() {
  local name="$1"; shift
  echo "=== $name ==="
  if ! .venv/bin/python scripts/diag_covariant_sim3.py "$@" \
      --output "$OUT/$name.json"; then
    echo "FAIL $name"
    fail=1
  fi
}
run kernel_cauchy --weight-mode full --kernel cauchy
run kernel_tukey --weight-mode full --kernel tukey
run clamp_full_r31.6 --weight-mode full --max-depth-ratio 31.6
run clamp_fixed_lateral_r31.6 --weight-mode fixed_lateral --max-depth-ratio 31.6
echo "SWEEP2_DONE fail=$fail"
exit "$fail"
