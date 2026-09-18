"""Empirical test: can a Rustuna/optuna SQLite study be resumed with a
different search space (some params removed)?

Usage: .venv/bin/python scripts/test_resume_paramspace.py
"""

import os
from pathlib import Path

import rustuna

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "eval/results/_test_paramspace.db"
if DB.exists():
    DB.unlink()

storage = rustuna.storages.SQLite3Storage(str(DB))

print("=== phase 1: suggest {a, b, decim} ===")


def obj1(trial):
    a = trial.suggest_float("a", 0.0, 1.0)
    b = trial.suggest_float("b", 0.0, 1.0)
    d = trial.suggest_int("decim", 4, 16)
    return a + b + 0.0 * d


s1 = rustuna.create_study(direction="minimize", study_name="t",
                          storage=storage, sampler=rustuna.samplers.TPESampler(seed=1))
s1.optimize(obj1, 6)
print("phase1 best:", s1.best_trial.params, s1.best_trial.value)

print("=== phase 2: resume, suggest {a, b} only (no decim) ===")


def obj2(trial):
    a = trial.suggest_float("a", 0.0, 1.0)
    b = trial.suggest_float("b", 0.0, 1.0)
    return a + b


s2 = rustuna.create_study(direction="minimize", study_name="t",
                          storage=storage, load_if_exists=True,
                          sampler=rustuna.samplers.TPESampler(seed=1))
print("loaded n_trials:", len(s2.trials))
s2.optimize(obj2, 3)
print("phase2 best:", s2.best_trial.params, s2.best_trial.value)
print("RESUME_OK n_trials=", len(s2.trials))
DB.unlink()
