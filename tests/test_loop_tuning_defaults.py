"""Persistence must distinguish solver defaults and freeze trial replay settings."""
import json

from eval import rustuna_tune_loop as tuning


def test_study_key_separates_legacy_and_ridge_with_same_cli():
    arguments = {"seq": "desk,desk2", "objective": "mean", "n_trials": 10,
                 "out": "first.json", "storage": "study.db"}
    ridge = dict(tuning.SEQ_OPT1_DEFAULTS)
    legacy = {**ridge, "marginalization_backend": "legacy",
              "canonical_regularization": "legacy", "nl_reg": True, "loop_iterations": 10}
    assert tuning.evaluation_settings_key(arguments, ridge) != tuning.evaluation_settings_key(arguments, legacy)
    resumed = {**arguments, "n_trials": 30, "out": "resumed.json", "report_only": True}
    assert tuning.evaluation_settings_key(arguments, ridge) == tuning.evaluation_settings_key(resumed, ridge)


def test_saved_effective_trial_replays_after_defaults_change(monkeypatch):
    saved = json.loads(json.dumps(tuning.effective_eval_params({"kf_max_gap": 5})))
    changed_defaults = {**tuning.SEQ_OPT1_DEFAULTS, "relative_reg_alpha": .1,
                        "loop_iterations": 2, "nl_reg": True,
                        "marginalization_backend": "legacy", "canonical_regularization": "legacy"}
    monkeypatch.setattr(tuning, "SEQ_OPT1_DEFAULTS", changed_defaults)
    assert tuning.effective_eval_params(saved) == saved
    assert saved["relative_reg_alpha"] == 10.
    assert saved["marginalization_backend"] == "canonical_marginal"
    assert saved["nl_reg"] is False
