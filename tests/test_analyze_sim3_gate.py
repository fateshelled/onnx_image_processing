"""Tests for the gate/LOSO re-aggregation of Sim(3) diagnostic reports."""

import json
import sys

import pytest

from scripts import analyze_sim3_gate as gate


def _row(label, n_inliers, fit_ok=True):
    row = {"label": label, "n_inliers": n_inliers}
    if fit_ok:
        row["fit_ok"] = True
    return row


def _report(name, rows):
    return {"sequence": name, "rows": rows}


def test_wilson_matches_known_values():
    lo, hi = gate.wilson(1, 250)
    assert lo == pytest.approx(0.0007, abs=1e-4)
    assert hi == pytest.approx(0.0223, abs=1e-4)
    lo, hi = gate.wilson(0, 5)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.4345, abs=1e-3)
    assert gate.wilson(0, 0) == (None, None)


def test_accepted_requires_fit_and_gate():
    assert gate.accepted({"fit_ok": True, "n_inliers": 10}, 10)
    assert not gate.accepted({"fit_ok": True, "n_inliers": 9}, 10)
    assert not gate.accepted({"n_inliers": 12}, 10)


def test_accept_counts_split_by_label():
    report = _report("s", [_row(True, 12), _row(True, 8), _row(False, 12),
                           _row(False, 3), _row(True, 0, fit_ok=False)])
    assert gate.true_accepts(report, 10) == 1
    assert gate.false_accepts(report, 10) == 1
    assert gate.false_accepts(report, 13) == 0


def test_select_loso_gate_uses_max_false_fit_plus_one():
    rest = [_report("a", [_row(False, 7), _row(True, 30)]),
            _report("b", [_row(False, 10, fit_ok=False), _row(False, 4)])]
    assert gate.select_loso_gate(rest) == 8
    assert gate.select_loso_gate([_report("c", [_row(True, 20)])]) is None


def test_main_handles_reports_without_false_candidates(monkeypatch, tmp_path,
                                                       capsys):
    report = _report("only_true", [_row(True, 20), _row(True, 6)])
    path = tmp_path / "reports.json"
    path.write_text(json.dumps([report]))
    monkeypatch.setattr(sys, "argv",
                        ["analyze_sim3_gate.py", str(path),
                         "--gates", "8", "--locked-gate", "12"])
    gate.main()
    out = capsys.readouterr().out
    assert "no false candidates" in out
    assert "no_information" in out


def test_main_reports_loso_failure(monkeypatch, tmp_path, capsys):
    reports = [_report("train", [_row(False, 5), _row(True, 20)]),
               _report("hold", [_row(False, 7), _row(True, 20)])]
    path = tmp_path / "reports.json"
    path.write_text(json.dumps(reports))
    monkeypatch.setattr(sys, "argv",
                        ["analyze_sim3_gate.py", str(path),
                         "--gates", "8", "--locked-gate", "12"])
    gate.main()
    out = capsys.readouterr().out
    assert "selected gate= 6" in out
    assert "[FAIL]" in out
    assert "informative folds failing: 1/2" in out


def test_load_sequences_rejects_duplicates(tmp_path):
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps([_report("dup", [])]))
    with pytest.raises(SystemExit):
        gate.load_sequences([tmp_path / "a.json", tmp_path / "b.json"])


def test_load_sequences_skips_error_reports(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps([{"sequence": "bad", "error": "boom"},
                                _report("good", [])]))
    sequences = gate.load_sequences([path])
    assert [sequence["sequence"] for sequence in sequences] == ["good"]


def test_validate_fit_floor_rejects_legacy_report_below_floor():
    report = {**_report("legacy", []), "options": {"min_inliers": 8}}
    with pytest.raises(SystemExit, match="fit floor 8 exceeds requested gate 6"):
        gate.validate_fit_floor([report], [6, 8])

    current = {**_report("current", []),
               "options": {"min_inliers": 8, "min_tracks": 5}}
    gate.validate_fit_floor([current], [6, 8])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
