"""Analyse the room_kf Rustuna study: what do the low-ATE trials share?

Reads eval/results/rustuna_room_kf.db (optuna-compatible SQLite), builds a
parameter table for every COMPLETE trial, and compares the good trials
(value <= threshold) with the rest.

Usage: .venv/bin/python scripts/analyze_room_tune.py [threshold]
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "eval/results/rustuna_room_kf.db"
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30

FIXED_DEFAULTS = {"keyframe_decim": 15, "kf_trans_thresh": 6.0,
                  "kf_rot_thresh": 10.0}

c = sqlite3.connect(DB)
rows = c.execute("""
    select t.trial_id, t.number, v.value
    from trials t join trial_values v on v.trial_id = t.trial_id
    where t.state = 'COMPLETE' and v.value is not null
    order by t.number
""").fetchall()

params_of = {}
for tid, name, val in c.execute(
        "select trial_id, param_name, param_value from trial_params"):
    params_of.setdefault(tid, {})[name] = val

records = []
for tid, number, value in rows:
    p = dict(FIXED_DEFAULTS)
    p.update(params_of.get(tid, {}))
    records.append({"number": number, "value": value, "params": p})

print(f"complete trials: {len(records)}")
for th in (0.28, 0.29, 0.30, 0.32, 0.35):
    n = sum(1 for r in records if r["value"] <= th)
    print(f"  value<={th:.2f}: {n} ({100*n/len(records):.1f}%)")
print(f"best = {min(r['value'] for r in records):.5f} "
      f"(trial {min(records, key=lambda r: r['value'])['number']})")

good = [r for r in records if r["value"] <= THRESHOLD]
bad = [r for r in records if r["value"] > THRESHOLD]
print(f"\ngood (<= {THRESHOLD}) = {len(good)}, bad (> {THRESHOLD}) = {len(bad)}")

numeric = ["kf_trans_thresh", "kf_rot_thresh", "loop_window", "loop_min_gap",
           "loop_min_inlier", "loop_temporal_k", "scale_prior_sigma",
           "step_scale_t", "loop_sigma_scale"]

all_vals = np.array([r["value"] for r in records])
print(f"\n{'param':18} {'good_med':>9} {'bad_med':>9} {'good_q1..q3':>20} "
      f"{'bad_q1..q3':>20} {'spearman':>9} {'p':>9}")
for name in numeric:
    g = np.array([r["params"].get(name, np.nan) for r in good], float)
    b = np.array([r["params"].get(name, np.nan) for r in bad], float)
    a = np.array([r["params"].get(name, np.nan) for r in records], float)
    g = g[~np.isnan(g)]
    b = b[~np.isnan(b)]
    mask = ~np.isnan(a)
    rho, p = spearmanr(a[mask], all_vals[mask])
    gq = np.percentile(g, [25, 75]) if len(g) else [np.nan, np.nan]
    bq = np.percentile(b, [25, 75]) if len(b) else [np.nan, np.nan]
    print(f"{name:18} {np.median(g):9.3f} {np.median(b):9.3f} "
          f"{gq[0]:9.3f}..{gq[1]:<9.3f} {bq[0]:9.3f}..{bq[1]:<9.3f} "
          f"{rho:9.3f} {p:9.2e}")

print("\nloop_temporal_k counts  good:", dict(sorted(Counter(
    int(r["params"]["loop_temporal_k"]) for r in good).items())))
print("loop_temporal_k counts  bad :", dict(sorted(Counter(
    int(r["params"]["loop_temporal_k"]) for r in bad).items())))

print("\ntop 20 trials:")
print(f"{'#':>4} {'value':>8} {'kf_t':>6} {'kf_r':>6} {'lw':>4} {'gap':>4} "
      f"{'inl':>5} {'tk':>3} {'prior':>6} {'step':>6} {'sig':>6}")
for r in sorted(records, key=lambda r: r["value"])[:20]:
    p = r["params"]
    print(f"{r['number']:>4} {r['value']:8.4f} {p.get('kf_trans_thresh', float('nan')):6.2f} "
          f"{p.get('kf_rot_thresh', float('nan')):6.1f} {int(p['loop_window']):>4} "
          f"{int(p['loop_min_gap']):>4} {p['loop_min_inlier']:5.3f} "
          f"{int(p['loop_temporal_k']):>3} {p['scale_prior_sigma']:6.3f} "
          f"{p['step_scale_t']:6.3f} {p['loop_sigma_scale']:6.3f}")

# Write the good-trial parameter table for further inspection.
out = [{"number": r["number"], "value": r["value"],
        "params": {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in r["params"].items()}} for r in good]
(REPO / "eval/results/room_tune_good.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote eval/results/room_tune_good.json ({len(out)} trials)")
