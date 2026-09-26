"""Unit tests for the offline Sim(3) loop-separation diagnostic helpers."""

import json
import math
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import diag_loop_sim3 as diag
from scripts.diag_loop_sim3 import (
    _auc,
    _candidate_seed,
    _compose_odom,
    _error_report,
    _finite_or_none,
    _local_cloud,
    _point_key,
)


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_compose_odom_matches_sequential_transforms():
    odom = [
        {"ok": True, "R": _rot_z(0.1), "t": np.array([1.0, 0.0, 0.0])},
        {"ok": True, "R": _rot_z(0.2), "t": np.array([0.0, 1.0, 0.0])},
        {"ok": True, "R": np.eye(3), "t": np.array([0.5, 0.0, 0.0])},
    ]
    rotation, translation = _compose_odom(odom, 0, 3)
    point = np.array([0.3, -0.2, 0.7])
    expected = odom[2]["R"] @ (
        odom[1]["R"] @ (odom[0]["R"] @ point + odom[0]["t"])
        + odom[1]["t"]) + odom[2]["t"]
    np.testing.assert_allclose(rotation @ point + translation, expected)
    assert _compose_odom(odom, 0, 4) is None
    assert _compose_odom([{"ok": False, "R": None, "t": None}], 0, 1) is None
    identity, zero = _compose_odom(odom, 2, 2)
    np.testing.assert_allclose(identity, np.eye(3))
    np.testing.assert_allclose(zero, np.zeros(3))


def test_auc_orders_scores_and_counts_ties():
    assert _auc([True, False], [1.0, 0.0]) == 1.0
    assert _auc([True, False], [0.0, 1.0]) == 0.0
    assert _auc([True, False], [1.0, 1.0]) == 0.5
    assert _auc([True, True, False, False],
                [0.4, 0.2, 0.3, 0.0]) == 0.75
    assert math.isnan(_auc([True], [1.0]))
    assert math.isnan(_auc([False], [1.0]))


def test_local_cloud_selects_odom_slot_before_the_endpoint(monkeypatch):
    stride = 10
    odom = [{"ok": True, "R": np.eye(3),
             "t": np.array([float(slot + 1), 0.0, 0.0])}
            for slot in range(5)]
    # Slot 2 is invalid so only the backward window is usable here.
    odom[2] = {"ok": False, "R": None, "t": None}
    cache = {"feat": {10: (None, None), 20: (None, None)}, "odom": odom,
             "stride": stride}
    points_previous = np.array([[100.0, 200.0], [120.0, 220.0]])
    points_endpoint = np.array([[110.0, 210.0], [130.0, 230.0]])
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (points_previous,
                                                 points_endpoint, None))
    monkeypatch.setattr(
        diag, "triangulate_local",
        lambda *args, **kwargs: SimpleNamespace(
            points=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
            valid=np.array([True, True]),
            parallax_deg=np.array([5.0, 5.0])))
    cam = SimpleNamespace(K=np.eye(3))
    cloud = _local_cloud(cache, None, cam, 20, stride, 1, None)
    assert set(cloud) == {_point_key(points_endpoint[0]),
                          _point_key(points_endpoint[1])}
    # endpoint 20 -> previous 10 -> odom slot 1 -> translation [2, 0, 0].
    np.testing.assert_allclose(cloud[_point_key(points_endpoint[0])],
                               [2.0, 0.0, 1.0])
    np.testing.assert_allclose(cloud[_point_key(points_endpoint[1])],
                               [2.0, 0.0, 2.0])


def test_local_cloud_composes_the_requested_window(monkeypatch):
    stride = 10
    odom = [{"ok": True, "R": np.eye(3),
             "t": np.array([float(slot + 1), 0.0, 0.0])}
            for slot in range(5)]
    cache = {"feat": {0: (None, None), 20: (None, None)}, "odom": odom,
             "stride": stride}
    points_previous = np.array([[100.0, 200.0]])
    points_endpoint = np.array([[110.0, 210.0]])
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (points_previous,
                                                 points_endpoint, None))
    monkeypatch.setattr(
        diag, "triangulate_local",
        lambda *args, **kwargs: SimpleNamespace(
            points=np.array([[0.0, 0.0, 1.0]]),
            valid=np.array([True]),
            parallax_deg=np.array([5.0])))
    cam = SimpleNamespace(K=np.eye(3))
    # Two stride pairs: slots 0 and 1 -> translation [1, 0, 0] + [2, 0, 0].
    cloud = _local_cloud(cache, None, cam, 20, stride, 2, None)
    np.testing.assert_allclose(cloud[_point_key(points_endpoint[0])],
                               [3.0, 0.0, 1.0])


def test_local_cloud_falls_back_to_a_forward_window(monkeypatch):
    stride = 10
    odom = [{"ok": False, "R": None, "t": None},
            {"ok": True, "R": np.eye(3), "t": np.array([1.0, 0.0, 0.0])}]
    cache = {"feat": {10: (None, None), 20: (None, None)}, "odom": odom,
             "stride": stride}
    points_endpoint = np.array([[110.0, 210.0]])
    points_next = np.array([[115.0, 215.0]])
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (points_endpoint,
                                                 points_next, None))
    monkeypatch.setattr(
        diag, "triangulate_local",
        lambda *args, **kwargs: SimpleNamespace(
            points=np.array([[0.0, 0.0, 4.0]]),
            valid=np.array([True]),
            parallax_deg=np.array([5.0])))
    cam = SimpleNamespace(K=np.eye(3))
    cloud = _local_cloud(cache, None, cam, 10, stride, 1, None)
    assert set(cloud) == {_point_key(points_endpoint[0])}
    # Forward triangulation is already in the endpoint camera: no transform.
    np.testing.assert_allclose(cloud[_point_key(points_endpoint[0])],
                               [0.0, 0.0, 4.0])


def test_local_cloud_returns_none_without_usable_odometry(monkeypatch):
    odom = [{"ok": False, "R": None, "t": None}]
    cache = {"feat": {}, "odom": odom, "stride": 10}
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (np.zeros((1, 2)),
                                                 np.zeros((1, 2)), None))
    cam = SimpleNamespace(K=np.eye(3))
    assert _local_cloud(cache, None, cam, 10, 10, 1, None) is None


def test_local_cloud_falls_back_when_the_longest_window_is_empty(monkeypatch):
    stride = 10
    odom = [{"ok": True, "R": np.eye(3),
             "t": np.array([float(slot + 1), 0.0, 0.0])}
            for slot in range(5)]
    cache = {"feat": {0: (None, None), 20: (None, None)}, "odom": odom,
             "stride": stride}
    points_previous = np.array([[100.0, 200.0]])
    points_endpoint = np.array([[110.0, 210.0]])
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (points_previous,
                                                 points_endpoint, None))
    calls = {"n": 0}

    def fake_triangulate(*args, **kwargs):
        calls["n"] += 1
        valid = np.array([False]) if calls["n"] == 1 else np.array([True])
        return SimpleNamespace(points=np.array([[0.0, 0.0, 1.0]]),
                               valid=valid, parallax_deg=np.array([5.0]))

    monkeypatch.setattr(diag, "triangulate_local", fake_triangulate)
    cam = SimpleNamespace(K=np.eye(3))
    cloud = _local_cloud(cache, None, cam, 20, stride, 2, None)
    # All windows are scanned; the empty longest one is skipped and the
    # densest non-empty window (forward, already in the endpoint camera) wins.
    assert calls["n"] == 4
    assert set(cloud) == {_point_key(points_previous[0])}
    np.testing.assert_allclose(cloud[_point_key(points_previous[0])],
                               [0.0, 0.0, 1.0])


def test_local_cloud_prefers_the_densest_window(monkeypatch):
    stride = 10
    odom = [{"ok": True, "R": np.eye(3),
             "t": np.array([float(slot + 1), 0.0, 0.0])}
            for slot in range(5)]
    cache = {"feat": {0: (None, None), 20: (None, None)}, "odom": odom,
             "stride": stride}
    points_previous = np.array([[100.0, 200.0], [101.0, 201.0],
                                [102.0, 202.0]])
    points_endpoint = np.array([[110.0, 210.0], [111.0, 211.0],
                                [112.0, 212.0]])
    monkeypatch.setattr(diag, "_details",
                        lambda *args, **kwargs: (points_previous,
                                                 points_endpoint, None))
    calls = {"n": 0}

    def fake_triangulate(*args, **kwargs):
        calls["n"] += 1
        valid = (np.array([True, False, False]) if calls["n"] == 1
                 else np.array([True, True, True]))
        return SimpleNamespace(points=np.ones((3, 3)), valid=valid,
                               parallax_deg=np.full(3, 5.0))

    monkeypatch.setattr(diag, "triangulate_local", fake_triangulate)
    cam = SimpleNamespace(K=np.eye(3))
    cloud = _local_cloud(cache, None, cam, 20, stride, 2, None)
    # The sparse longest window must not hide the denser forward window, and
    # every window is scanned before the densest one is chosen.
    assert calls["n"] == 4
    assert len(cloud) == 3
    assert set(cloud) == {_point_key(key) for key in points_previous}


def _synthetic_opts(tmp_path, **overrides):
    values = dict(
        seq="syn", cache_dir=str(tmp_path), dataset_root="unused",
        max_candidates=10, min_gap=20, appearance_min=0.2, local_strides=1,
        backward_only=True,
        min_parallax_deg=1.0, min_condition_ratio=1e-3, true_distance=0.5,
        false_distance=1.0, min_inliers=8, min_tracks=5,
        residual_fraction=0.1,
        iterations=200, seed=0, fx=525.0, fy=525.0, cx=320.0, cy=240.0,
        width=640, height=480, output=None)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("cache_name", ["match_cache_syn_torch.pkl",
                                        "match_cache_syn_numpy.pkl"])
def test_evaluate_sequence_records_rows_and_gate(monkeypatch, tmp_path,
                                                 cache_name):
    cache = {
        "feat": {10: (None, None), 40: (None, None), 50: (None, None),
                 60: (None, None)},
        "odom": [{"ok": True, "R": np.eye(3), "t": np.array([1.0, 0.0, 0.0])}],
        "gt_pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1],
                            [0.0, 0.0, 0.0], [0.0, 0.0, 10.0],
                            [0.0, 0.0, 0.1], [0.0, 0.0, 10.0],
                            [0.0, 0.0, 10.0]]),
        "stride": 10,
    }
    monkeypatch.setattr(diag, "load_cache", lambda cache_dir, seq: cache)
    monkeypatch.setattr(diag, "intrinsics_for",
                        lambda root, seq, default: default)
    match_cache = {
        (10, 40): {"ok": True, "inlier_ratio": 0.5},  # true, 0.1 m apart
        (10, 50): {"ok": True, "inlier_ratio": 0.5},  # false, 10 m apart
        (10, 60): {"ok": True, "inlier_ratio": 0.5},  # false, no pose
    }
    with (tmp_path / cache_name).open("wb") as handle:
        pickle.dump(match_cache, handle)

    keys_a = np.column_stack([np.arange(6) + 100.0, np.arange(6) + 200.0])
    keys_b = keys_a + 1.0
    false_keys = np.column_stack([np.arange(6) + 400.0,
                                  np.arange(6) + 500.0])
    structure = np.array([[float(index % 3), float(index // 3),
                           float(index) * 0.1 + 1.0] for index in range(6)])
    cloud_a = {diag._point_key(key): point
               for key, point in zip(keys_a, structure)}
    cloud_b = {diag._point_key(key): point
               for key, point in zip(keys_b, structure)}
    rng = np.random.default_rng(3)
    cloud_false = {diag._point_key(key): point
                    for key, point in zip(false_keys, rng.normal(size=(6, 3)))}
    clouds = {10: cloud_a, 40: cloud_b, 50: cloud_false}

    def fake_details(c, matcher, cam, a, b, args, require_pose=True):
        if (a, b) == (10, 40):
            return keys_a, keys_b, None
        if (a, b) == (10, 50):
            return keys_a, false_keys, None
        return None

    monkeypatch.setattr(diag, "_details", fake_details)
    backward_flags = []

    def fake_local_cloud(c, matcher, cam, endpoint, stride, window_strides,
                         args, min_parallax_deg=1.0, backward_only=False):
        backward_flags.append(backward_only)
        return clouds.get(int(endpoint))

    monkeypatch.setattr(diag, "_local_cloud", fake_local_cloud)
    opts = _synthetic_opts(tmp_path, min_inliers=6)
    report = diag.evaluate_sequence("syn", str(tmp_path), "unused", object(),
                                    opts)
    assert report["evaluated"] == 3
    assert report["true_accepted"] == 1
    assert report["false_accepted"] == 0
    assert report["accept_precision"] == 1.0
    assert backward_flags and all(backward_flags)
    # Six-track candidates remain fitted so locked gate 6 can be re-aggregated.
    assert report["auc_inlier_count"] >= 0.9
    reasons = [row.get("reason") for row in report["rows"]]
    assert "no_pose" in reasons


def test_evaluate_sequence_rejects_non_aligned_keys(monkeypatch, tmp_path):
    cache = {"feat": {}, "odom": [], "gt_pos": np.zeros((2, 3)), "stride": 10}
    monkeypatch.setattr(diag, "load_cache", lambda cache_dir, seq: cache)
    monkeypatch.setattr(diag, "intrinsics_for",
                        lambda root, seq, default: default)
    with (tmp_path / "match_cache_syn_torch.pkl").open("wb") as handle:
        pickle.dump({(0, 15): {"ok": True, "inlier_ratio": 0.9}}, handle)
    opts = _synthetic_opts(tmp_path)
    with pytest.raises(ValueError, match="stride-aligned"):
        diag.evaluate_sequence("syn", str(tmp_path), "unused", object(), opts)


def test_finite_or_none_sanitizes_non_finite_values():
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(np.nan) is None
    assert _finite_or_none(np.inf) is None
    assert _finite_or_none(-np.inf) is None


def test_candidate_seed_is_deterministic_and_pair_specific():
    assert _candidate_seed(0, 10, 40) == _candidate_seed(0, 10, 40)
    assert _candidate_seed(0, 10, 40) != _candidate_seed(0, 10, 41)
    assert 0 <= _candidate_seed(3, 5, 7) < 2 ** 31 - 1


def test_error_report_matches_success_schema(tmp_path):
    report = _error_report("syn", RuntimeError("boom"),
                           _synthetic_opts(tmp_path))
    assert report["error"].startswith("RuntimeError: boom")
    assert report["rows"] == []
    for key in ("match_cache", "elapsed_sec", "candidate_pool", "eligible",
                "selected", "evaluated", "fitted", "true_fitted",
                "false_fitted", "accepted", "true_accepted", "false_accepted",
                "true_rejected", "false_rejected", "accept_precision",
                "auc_inlier_count", "auc_inlier_count_fitted",
                "auc_inlier_ratio", "auc_inlier_ratio_fitted",
                "auc_negative_residual_fitted"):
        assert key in report


def test_main_records_sequence_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(diag, "TorchSinkhornMatcher", lambda **kwargs: object())
    monkeypatch.setattr(sys, "argv",
                        ["diag_loop_sim3.py", "--seq", "missing",
                         "--cache-dir", str(tmp_path)])
    diag.main()
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 1
    assert reports[0]["sequence"] == "missing"
    assert "error" in reports[0]
    assert reports[0]["rows"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
