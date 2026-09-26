#!/bin/bash
# Holdout validation of the Phase-1 (odometry/RPE) tuning on unseen fr2/fr3/fr1
# sequences (loop disabled). Writes eval/results/phase1_holdout.json incrementally.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ulimit -v 20971520
export TORCH_THREADS=4
exec .venv/bin/python scripts/verify_phase1_holdout.py
