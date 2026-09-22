"""Tests for the P0a local-track diagnostic helpers."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import diag_local_tracks as diag
from scripts.diag_local_tracks import frame_pairs, histogram


def test_frame_pairs_respects_local_radius():
    assert frame_pairs([0, 2, 4, 6], 2) == [
        (0, 2), (0, 4), (2, 4), (2, 6), (4, 6),
    ]
    with pytest.raises(ValueError, match="pair_radius"):
        frame_pairs([0, 2], 0)


def test_histogram_uses_json_keys():
    assert histogram([3, 2, 3]) == {"2": 1, "3": 2}
    assert histogram([]) == {}


def test_percentile_summary_filters_nonfinite_values_for_strict_json():
    assert diag.percentile_summary([np.inf, np.nan]) == {
        "median": None, "p90": None,
    }
    summary = diag.percentile_summary([1.0, np.inf, 3.0])
    assert summary == {"median": 2.0, "p90": 2.8}
    json.dumps(summary, allow_nan=False)


def test_anchor_poses_scales_multistep_pair_bridge_to_equal_step_gauge():
    invalid = {"ok": False, "R": None, "t": None}
    valid = {"ok": True, "R": np.eye(3), "t": [-1.0, 0.0, 0.0]}
    cache = {"stride": 2, "odom": [valid, invalid, valid]}
    poses, bridges = diag.anchor_poses(
        cache, [0, 2, 6],
        [(2, 6, np.eye(3), np.array([-1.0, 0.0, 0.0]))],
        return_bridges=True)

    np.testing.assert_allclose(poses[0][1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(poses[2][1], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(poses[6][1], [-3.0, 0.0, 0.0])
    assert bridges == [[2, 6]]


def test_anchor_poses_handles_reverse_rotated_bridge():
    angle = np.deg2rad(20.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    cache = {"stride": 2, "odom": [
        {"ok": True, "R": np.eye(3), "t": [-1.0, 0.0, 0.0]},
        {"ok": False, "R": None, "t": None},
    ]}
    poses = diag.anchor_poses(
        cache, [0, 2, 4],
        [(4, 2, rotation, np.array([1.0, 0.0, 0.0]))])
    inverse_rotation, inverse_translation = diag._inverse_pose(
        rotation, np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(poses[4][0], inverse_rotation)
    np.testing.assert_allclose(
        poses[4][1], inverse_rotation @ np.array([-1.0, 0.0, 0.0])
        + inverse_translation)


def test_anchor_poses_prefers_complete_odometry_and_rejects_disconnection():
    valid = {"ok": True, "R": np.eye(3), "t": [-1.0, 0.0, 0.0]}
    poses, bridges = diag.anchor_poses(
        {"stride": 2, "odom": [valid, valid]}, [0, 2, 4],
        [(0, 4, np.eye(3), np.array([1.0, 0.0, 0.0]))],
        return_bridges=True)
    np.testing.assert_allclose(poses[4][1], [-2.0, 0.0, 0.0])
    assert bridges == []

    with pytest.raises(RuntimeError, match="no pose path"):
        diag.anchor_poses(
            {"stride": 2, "odom": []}, [0, 2], pair_poses=[])


def _opts(**overrides):
    values = dict(
        cache_dir="unused", dataset_root="unused", max_frames=3,
        pair_radius=2, min_track_length=3, min_shared_tracks=1,
        max_neighbors=2, fx=525.0, fy=525.0, cx=320.0, cy=240.0,
        width=640, height=480,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_evaluate_sequence_uses_inliers_and_track_length_for_covisibility(
        monkeypatch):
    keypoints = np.zeros((1, 2, 2), dtype=float)
    descriptors = np.zeros((1, 2, 4), dtype=float)
    cache = {"feat": {frame: (keypoints, descriptors)
                      for frame in (0, 2, 4)},
             "stride": 2,
             "odom": [{"ok": True, "R": np.eye(3), "t": [-1, 0, 0]},
                      {"ok": True, "R": np.eye(3), "t": [-1, 0, 0]}]}
    monkeypatch.setattr(diag, "load_cache", lambda *_args: cache)
    monkeypatch.setattr(diag, "intrinsics_for",
                        lambda *_args: (525.0, 525.0, 320.0, 240.0))
    monkeypatch.setattr(
        diag, "extract_match_indices",
        lambda *_args: (np.array([0, 1]), np.array([0, 1]),
                        np.array([0.9, 0.8])))
    calls = {"n": 0}

    def fake_pose(*_args):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"ok": False}
        if calls["n"] == 3:
            return {"ok": True, "mask": np.array([True, False])}
        return {"ok": True, "mask": np.array([True, True])}

    monkeypatch.setattr(diag, "estimate_pose_from_matches", fake_pose)

    class Matcher:
        def match_probs(self, *_args):
            return np.ones((3, 3))

    captured_pairs = []
    real_build = diag.build_feature_tracks

    def capture_build(pairs):
        captured_pairs.extend(pairs)
        return real_build(pairs)

    monkeypatch.setattr(diag, "build_feature_tracks", capture_build)
    monkeypatch.setattr(diag, "triangulate_feature_track",
                        lambda *_args: None)
    report = diag.evaluate_sequence("syn", _opts(), Matcher())
    assert report["n_pairs"] == 3
    assert report["pose_failures"] == 1
    assert report["inlier_matches"] == 3
    assert report["tracks"] == 2
    assert report["selected_tracks"] == 1
    assert report["covisibility_pairs"] == 3
    assert captured_pairs[2].feature_i == (0, 1)
    assert captured_pairs[2].inlier_mask == (True, False)
    assert report["neighbors"]["0"] == [
        {"frame": 2, "shared_tracks": 1},
        {"frame": 4, "shared_tracks": 1},
    ]
    json.dumps(report, allow_nan=False)


def test_evaluate_sequence_rejects_pose_mask_mismatch(monkeypatch):
    keypoints = np.zeros((1, 2, 2), dtype=float)
    descriptors = np.zeros((1, 2, 4), dtype=float)
    cache = {"feat": {frame: (keypoints, descriptors) for frame in (0, 2)},
             "stride": 2, "odom": []}
    monkeypatch.setattr(diag, "load_cache", lambda *_args: cache)
    monkeypatch.setattr(diag, "intrinsics_for",
                        lambda *_args: (525.0, 525.0, 320.0, 240.0))
    monkeypatch.setattr(
        diag, "extract_match_indices",
        lambda *_args: (np.array([0, 1]), np.array([0, 1]),
                        np.array([0.9, 0.8])))
    monkeypatch.setattr(diag, "estimate_pose_from_matches",
                        lambda *_args: {"ok": True,
                                        "mask": np.array([True])})

    class Matcher:
        def match_probs(self, *_args):
            return np.ones((3, 3))

    with pytest.raises(RuntimeError, match="mask length mismatch"):
        diag.evaluate_sequence("syn", _opts(max_frames=2), Matcher())


def test_main_returns_nonzero_after_writing_failure_json(
        monkeypatch, tmp_path, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr(diag, "TorchSinkhornMatcher",
                        lambda **_kwargs: object())
    monkeypatch.setattr(diag, "evaluate_sequence",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        sys, "argv", ["diag_local_tracks.py", "--seq", "bad",
                      "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        diag.main()
    assert exc.value.code == 1
    report = json.loads(output.read_text())
    assert report == [{"sequence": "bad", "error": "RuntimeError: boom"}]
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    "args", [["--seq", ""], ["--pair-radius", "0"],
             ["--min-shared-tracks", "0"], ["--max-neighbors", "-1"],
             ["--fx", "nan"], ["--width", "0"]])
def test_main_rejects_invalid_cli_values(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["diag_local_tracks.py", *args])
    with pytest.raises(SystemExit) as exc:
        diag.main()
    assert exc.value.code == 2
