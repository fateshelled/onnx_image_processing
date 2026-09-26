"""Summarise the completed Phase1 odometry study (tune_odom.db / odom_v1).

Reads every completed trial's value (mean RPE_trans_2m_median), worst-case and
per-sequence metrics, then reports:

* value distribution
* best trial by mean, and robust re-ranking by mean + lambda * worst
* plateau width (how many trials/unique parameter sets sit within x% of best)
* per-dimension sensitivity around the best cell (median value per category)
* comparison with the reference parameter sets shipped in tune_odom.json

Read-only: writes nothing to the study.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--storage", default=str(REPO / "eval/results/tune_odom.db"))
ap.add_argument("--study-name", default="odom_v1")
ap.add_argument("--json-out", default=str(REPO / "eval/results/tune_odom_summary.json"))
args = ap.parse_args()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name=args.study_name,
                          storage=f"sqlite:///{args.storage}")
trials = [t for t in study.trials if t.value is not None]
n_all = len(study.trials)
print(f"trials total={n_all} completed={len(trials)} "
      f"failed={sum(1 for t in study.trials if t.state.name == 'FAIL')}")

vals = sorted(t.value for t in trials)
n = len(vals)


def pct(p):
    return vals[min(n - 1, int(round(p / 100.0 * (n - 1))))]


stats = {
    "n_completed": n,
    "min": vals[0],
    "p10": pct(10),
    "p25": pct(25),
    "p50": pct(50),
    "p75": pct(75),
    "p90": pct(90),
    "max": vals[-1],
}
print("value distribution:", {k: round(v, 5) for k, v in stats.items()})

best_mean = min(trials, key=lambda t: t.value)
print(f"\nbest mean: trial {best_mean.number} value={best_mean.value:.5f} "
      f"worst={best_mean.user_attrs.get('worst')}")


def per(t):
    return json.loads(t.user_attrs.get("per_seq", "{}"))


def rpe2m(ps, seq):
    try:
        return ps[seq]["RPE_trans_2m_median"]
    except Exception:
        return float("nan")


result = {
    "n_total": n_all,
    "n_completed": n,
    "value_distribution": stats,
    "by_lambda": {},
    "top20_mean_plus_worst": [],
    "plateau": {},
    "sensitivity": {},
    "comparison": {},
}

for lam in (0.0, 0.5, 1.0, 2.0):
    def score(t):
        return t.value + lam * float(t.user_attrs.get("worst", 0.0) or 0.0)
    b = min(trials, key=score)
    ps = per(b)
    result["by_lambda"][str(lam)] = {
        "trial": b.number,
        "mean": b.value,
        "worst": float(b.user_attrs.get("worst", float("nan")) or 0.0),
        "score": score(b),
        "per_seq_rpe2m": {s: rpe2m(ps, s) for s in ("desk", "desk2", "room")},
        "per_seq_ate": {s: ps.get(s, {}).get("ATE_median") for s in ("desk", "desk2", "room")},
        "params": json.loads(b.user_attrs.get("eval_params", "{}")),
    }
    print(f"lambda={lam}: trial {b.number} mean={b.value:.5f} "
          f"worst={float(b.user_attrs.get('worst', float('nan')) or 0.0):.5f} "
          f"score={score(b):.5f}")

# plateau: within 1%/2%/5% of best, count trials and unique eval-param groups.
bestv = best_mean.value
plat = {}
for tol in (0.01, 0.02, 0.05):
    sel = [t for t in trials if t.value <= bestv * (1 + tol)]
    groups = defaultdict(list)
    for t in sel:
        key = json.dumps(json.loads(t.user_attrs.get("eval_params", "{}")), sort_keys=True)
        groups[key].append(t.number)
    plat[f"{int(tol*100)}pct"] = {
        "n_trials": len(sel),
        "n_unique_param_sets": len(groups),
    }
    print(f"plateau <= +{tol*100:.0f}%: {len(sel)} trials, "
          f"{len(groups)} unique param sets")
result["plateau"] = plat

# top20 by mean+worst
top = sorted(trials, key=lambda t: t.value + float(t.user_attrs.get("worst", 0.0) or 0.0))[:20]
for t in top:
    ps = per(t)
    result["top20_mean_plus_worst"].append({
        "trial": t.number,
        "mean": t.value,
        "worst": float(t.user_attrs.get("worst", float("nan")) or 0.0),
        "rpe2m": {s: rpe2m(ps, s) for s in ("desk", "desk2", "room")},
        "ate": {s: ps.get(s, {}).get("ATE_median") for s in ("desk", "desk2", "room")},
        "params": json.loads(t.user_attrs.get("eval_params", "{}")),
    })

# sensitivity around the robust best (lambda=1): median value per category level.
robust = result["by_lambda"]["1.0"]
sens = defaultdict(lambda: defaultdict(list))
for t in trials:
    p = json.loads(t.user_attrs.get("eval_params", "{}"))
    for k in ("step_scale_t", "scale_prior_sigma", "kf_trans_thresh",
              "kf_rot_thresh", "kf_max_gap", "max_keyframes",
              "nl_reg_c", "nl_reg_tau", "nl_reg_length", "loop_iterations"):
        if k in p:
            sens[k][repr(p[k])].append(t.value)
for k, lv in sens.items():
    result["sensitivity"][k] = {}
    for level, vv in sorted(lv.items(), key=lambda kv: -len(kv[1])):
        vv2 = sorted(vv)
        med = vv2[len(vv2) // 2]
        result["sensitivity"][k][level] = {
            "n": len(vv2), "median": med, "min": vv2[0]}
    print(f"\n[{k}]")
    for level, d in result["sensitivity"][k].items():
        print(f"  {level:>6} n={d['n']:>3} median={d['median']:.4f} min={d['min']:.4f}")

json.dump(result, open(args.json_out, "w"), indent=2)
print("\nwrote", args.json_out)
