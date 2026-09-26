"""Robust selection for the seq_opt1 study: minimise mean + lambda * worst.

The running tuner optimises the mean ATE over the training sequences, but the
final parameter set should not overfit one dataset. Each trial already stores
its per-sequence ATEs and its worst-case, so we re-rank the completed trials by
a robust score ``mean + lambda * worst`` and report the winners for several
lambdas. Holdout sequences are NOT used here (they stay for validation).
"""

import argparse
import json
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--storage",
                default=str(REPO / "eval/results/tune_seq_opt1_optuna.db"))
ap.add_argument("--study-name", default="seq_opt1")
ap.add_argument("--out", default=str(REPO / "eval/results/tune_seq_opt1_robust.json"))
ap.add_argument("--lambdas", default="0,0.5,1.0,2.0")
args = ap.parse_args()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name=args.study_name,
                          storage=f"sqlite:///{args.storage}")
completed = [t for t in study.trials if t.value is not None]
print(f"completed trials: {len(completed)}")
if not completed:
    raise SystemExit("no completed trials yet")

result = {"n_trials": len(completed), "by_lambda": {}}
for lam in [float(x) for x in args.lambdas.split(",")]:
    def score(t):
        return t.value + lam * float(t.user_attrs.get("worst", 0.0))
    best = min(completed, key=score)
    per = json.loads(best.user_attrs.get("per_seq", "{}"))
    result["by_lambda"][str(lam)] = {
        "trial": best.number,
        "mean": best.value,
        "worst": float(best.user_attrs.get("worst", float("nan"))),
        "robust_score": score(best),
        "per_seq": per,
        "params": json.loads(best.user_attrs.get("eval_params", "{}")),
    }
    print(f"lambda={lam}: trial {best.number} mean={best.value:.5f} "
          f"worst={float(best.user_attrs.get('worst', float('nan'))):.5f} "
          f"score={score(best):.5f} per_seq={per}")

# Top-10 by mean+worst for inspection.
lam1 = min(completed, key=lambda t: t.value + float(t.user_attrs.get("worst", 0.0)))
top = sorted(completed, key=lambda t: t.value + float(t.user_attrs.get("worst", 0.0)))[:10]
result["top10_mean_plus_worst"] = [
    {"trial": t.number, "mean": t.value,
     "worst": float(t.user_attrs.get("worst", float("nan"))),
     "per_seq": json.loads(t.user_attrs.get("per_seq", "{}"))} for t in top]

json.dump(result, open(args.out, "w"), indent=2)
print("wrote", args.out)
