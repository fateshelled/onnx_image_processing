"""Causal input and fair frozen-pose A/B checks for the BA overlay."""

from types import SimpleNamespace

import numpy as np
import pytest

from eval.rustuna_tune_loop import _eval_online_kf_prior, SEQ_OPT1_DEFAULTS
from scripts import ab_stream_local_ba as stream
from vo.local_tracks import PairMatches
from vo.se3 import se3_exp


def _pose(x, y=0.):
    T = np.eye(4)
    T[:2, 3] = [x, y]
    return T


def _opts():
    return SimpleNamespace(ba_window_size=3, pair_radius=2,
                           ba_max_iterations=5, ba_huber=3.,
                           ba_max_rotation_deg=2.,
                           ba_max_translation_ratio=1.,
                           window_mode="consecutive")


def test_overlay_receives_graph_revisions_before_old_pose_is_frozen(monkeypatch):
    frames = (0, 2, 4, 6, 8)
    locations = {0: _pose(0), 2: _pose(1), 4: _pose(2), 6: _pose(3), 8: _pose(4)}
    cache = {"stride": 2, "odom": [object()] * 4,
             "gt_pos": np.array([[0., .5, 0.], [1., .2, 0.],
                                 [2., 0., 0.], [3., 0., 0.], [4., 0., 0.]]),
             "feat": {frame: (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
                      for frame in frames}}
    seen_pairs = []

    class Graph:
        def pose(self, frame):
            return locations[frame].copy()

    def fake_eval(_cache, _params, _cam, _matcher, _mc, *, on_frame):
        graph = Graph()
        for frame in frames:
            if frame == 6:
                locations[0] = _pose(0, .5)
                locations[2] = _pose(1, .2)
            on_frame(frame, graph)
        return {"ATE_median": .1, "n_loop": 0}

    def fake_solve(active, pairs, _poses, _keypoints, _cam, _opts):
        seen_pairs.append((active[-1], tuple((p.frame_i, p.frame_j) for p in pairs)))
        return {}, {"accepted": False, "reason": "test skip"}

    monkeypatch.setattr(stream, "eval_seq", fake_eval)
    monkeypatch.setattr(stream, "_pair", lambda _c, a, b, _m, _cam:
                        PairMatches(a, b, (), (), (), ()))
    monkeypatch.setattr(stream, "_solve_window", fake_solve)
    report = stream.evaluate_stream(cache, {}, object(), object(), _opts())

    assert report["frames"] == 5
    assert report["accepted"] == 0
    assert report["baseline_frozen_ate"]["rmse"] == pytest.approx(0., abs=1e-12)
    assert report["ba_frozen_ate"]["rmse"] == pytest.approx(0., abs=1e-12)
    assert seen_pairs[0] == (4, ((0, 2), (0, 4), (2, 4)))
    assert seen_pairs[-1] == (8, ((4, 6), (4, 8), (6, 8)))


def test_graph_callback_observes_only_frames_through_current_step():
    odom = {"ok": True, "R": np.eye(3), "t": np.array([-1., 0., 0.]),
            "inlier": 1.}
    cache = {"stride": 2, "odom": [odom, odom],
             "gt_pos": np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])}
    params = {**SEQ_OPT1_DEFAULTS, "loop_verifier": "none"}
    seen = []

    def callback(frame, graph):
        seen.append((frame, graph.pose(frame)[:3, 3].copy(),
                     tuple(sorted(graph._T))))

    offline = _eval_online_kf_prior(
        cache, params, object(), lambda *_args: {"ok": False})
    callback_run = _eval_online_kf_prior(
        cache, params, object(), lambda *_args: {"ok": False},
        on_frame=callback)
    assert [row[0] for row in seen] == [0, 2, 4]
    assert [row[2] for row in seen] == [(0,), (0, 2), (0, 2, 4)]
    for i, row in enumerate(seen):
        np.testing.assert_allclose(row[1], [i, 0., 0.])
    assert callback_run["ATE_median"] == pytest.approx(offline["ATE_median"])


def test_accepted_ba_update_survives_rotated_graph_revision_and_freeze(monkeypatch):
    frames = (0, 2, 4, 6, 8)
    base = {f: _pose(f / 2) for f in frames}
    rotation = np.eye(4)
    rotation[:3, :3] = np.array([[0., -1., 0.],
                                [1., 0., 0.], [0., 0., 1.]])
    graph_poses = {f: pose.copy() for f, pose in base.items()}
    cache = {"stride": 2, "odom": [object()] * 4,
             "gt_pos": np.array([[0., 0., 0.], [0., 1., 0.],
                                 [0., 2., 0.], [0., 3., 0.], [0., 4., 0.]]),
             "feat": {f: (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
                      for f in frames}}
    proposal = base[4] @ se3_exp([0., 0., .1, 0., .15, 0.])
    captured = {}
    frozen = []

    class Graph:
        def pose(self, frame):
            return graph_poses[frame].copy()

    def fake_eval(_cache, _params, _cam, _matcher, _mc, *, on_frame):
        graph = Graph()
        for frame in frames:
            if frame == 6:
                for old in frames:
                    graph_poses[old] = rotation @ graph_poses[old]
            on_frame(frame, graph)
        return {"ATE_median": 0., "n_loop": 1}

    def fake_solve(active, _pairs, poses, _keypoints, _cam, _opts):
        if active[-1] == 4:
            output = {4: proposal.copy()}
        else:
            output = {}
        if active[-1] == 6:
            captured.update({f: poses[f].copy() for f in active})
        return output, {"accepted": bool(output), "reason": ""}

    real_ate = stream._trajectory_ate

    def capture_frozen(poses, *args):
        frozen.append({f: pose.copy() for f, pose in poses.items()})
        return real_ate(poses, *args)

    monkeypatch.setattr(stream, "eval_seq", fake_eval)
    monkeypatch.setattr(stream, "_pair", lambda _c, a, b, _m, _cam:
                        PairMatches(a, b, (), (), (), ()))
    monkeypatch.setattr(stream, "_solve_window", fake_solve)
    monkeypatch.setattr(stream, "_trajectory_ate", capture_frozen)
    result = stream.evaluate_stream(cache, {}, object(), object(), _opts())

    assert result["accepted"] == 1
    np.testing.assert_allclose(captured[4], rotation @ proposal)
    expected_new = (rotation @ proposal
                    @ np.linalg.inv(rotation @ base[4]) @ (rotation @ base[6]))
    np.testing.assert_allclose(captured[6], expected_new)
    np.testing.assert_allclose(frozen[1][4], rotation @ proposal)
    np.testing.assert_allclose(frozen[0][4], rotation @ base[4])
    assert result["ba_frozen_ate"]["rmse"] > result["baseline_frozen_ate"]["rmse"]


def test_keyframe_window_uses_previous_keyframes_and_current(monkeypatch):
    frames = (0, 2, 4, 6, 8, 10)
    keyframes_at = {
        0: (0,), 2: (0,), 4: (0, 4), 6: (0, 4),
        8: (0, 4, 8), 10: (0, 4, 8),
    }
    poses = {frame: _pose(frame / 2) for frame in frames}
    cache = {"stride": 2, "odom": [object()] * (len(frames) - 1),
             "gt_pos": np.array([[i, 0., 0.] for i in range(len(frames))]),
             "feat": {frame: (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
                      for frame in frames}}
    solved = []

    class Graph:
        keyframes = ()

        def pose(self, frame):
            return poses[frame].copy()

    def fake_eval(_cache, _params, _cam, _matcher, _mc, *, on_frame):
        graph = Graph()
        for frame in frames:
            graph.keyframes = keyframes_at[frame]
            on_frame(frame, graph)
        return {"ATE_median": 0., "n_loop": 0}

    def fake_solve(window, pairs, _poses, _keypoints, _cam, _opts):
        solved.append((tuple(window), tuple((p.frame_i, p.frame_j) for p in pairs)))
        return {}, {"accepted": False, "reason": "test skip"}

    opts = _opts()
    opts.window_mode = "keyframe"
    monkeypatch.setattr(stream, "eval_seq", fake_eval)
    monkeypatch.setattr(stream, "_pair", lambda _c, a, b, _m, _cam:
                        PairMatches(a, b, (), (), (), ()))
    monkeypatch.setattr(stream, "_solve_window", fake_solve)

    report = stream.evaluate_stream(cache, {}, object(), object(), opts)

    assert report["window_mode"] == "keyframe"
    assert [window for window, _pairs in solved] == [
        (0, 4, 6), (0, 4, 8), (4, 8, 10),
    ]
    assert solved[-1][1] == ((4, 8), (4, 10), (8, 10))
    assert set(report["windows"][-1]["window_frames"]) == {4, 8, 10}


def test_keyframe_update_moves_following_non_keyframes_consistently(monkeypatch):
    frames = (0, 2, 4, 6, 8)
    poses = {frame: _pose(frame / 2) for frame in frames}
    cache = {"stride": 2, "odom": [object()] * (len(frames) - 1),
             "gt_pos": np.array([[i, 0., 0.] for i in range(len(frames))]),
             "feat": {frame: (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
                      for frame in frames}}
    outputs = []

    class Graph:
        keyframes = ()

        def pose(self, frame):
            return poses[frame].copy()

    def fake_eval(_cache, _params, _cam, _matcher, _mc, *, on_frame):
        graph = Graph()
        for frame in frames:
            graph.keyframes = tuple(f for f in (0, 4, 8) if f <= frame)
            on_frame(frame, graph)
        return {"ATE_median": 0., "n_loop": 0}

    def fake_solve(window, _pairs, current, _keypoints, _cam, _opts):
        proposal = {}
        if window[-1] == 8:
            proposal[4] = _pose(2, .5)
        return proposal, {"accepted": bool(proposal), "reason": ""}

    opts = _opts()
    opts.window_mode = "keyframe"
    monkeypatch.setattr(stream, "eval_seq", fake_eval)
    monkeypatch.setattr(stream, "_pair", lambda _c, a, b, _m, _cam:
                        PairMatches(a, b, (), (), (), ()))
    monkeypatch.setattr(stream, "_solve_window", fake_solve)
    monkeypatch.setattr(stream, "_trajectory_ate",
                        lambda values, *_args: outputs.append(values.copy()) or {"rmse": 0.})

    report = stream.evaluate_stream(cache, {}, object(), object(), opts)

    assert report["accepted"] == 1
    baseline, ba = outputs
    np.testing.assert_allclose(ba[2], baseline[2])
    np.testing.assert_allclose(ba[4][:2, 3], [2., .5])
    np.testing.assert_allclose(ba[6][:2, 3], [3., .5])
    np.testing.assert_allclose(ba[8][:2, 3], [4., 0.])


def test_keyframe_update_stops_at_unoptimized_keyframe_boundary():
    overlay = {frame: _pose(frame) for frame in range(7)}
    proposal = {0: _pose(0, .1), 2: _pose(2, .2), 6: _pose(6, .6)}

    stream._apply_keyframe_proposal(proposal, overlay, (0, 2, 4, 6))

    np.testing.assert_allclose(overlay[1][:2, 3], [1., .1])
    np.testing.assert_allclose(overlay[3][:2, 3], [3., .2])
    np.testing.assert_allclose(overlay[4][:2, 3], [4., 0.])
    np.testing.assert_allclose(overlay[5][:2, 3], [5., 0.])
    np.testing.assert_allclose(overlay[6][:2, 3], [6., .6])


def test_keyframe_update_preserves_rotation_and_rejected_proposal(monkeypatch):
    frames = (0, 2, 4, 6, 8)
    poses = {frame: _pose(frame / 2) for frame in frames}
    graph_poses = {frame: pose.copy() for frame, pose in poses.items()}
    keyframes_at = {0: (0,), 2: (0,), 4: (0, 4), 6: (0, 4), 8: (0, 4, 8)}
    cache = {"stride": 2, "odom": [object()] * 4,
             "gt_pos": np.array([[i, 0., 0.] for i in range(5)]),
             "feat": {frame: (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
                      for frame in frames}}
    angle = np.deg2rad(90.)
    graph_rotation = np.eye(4)
    graph_rotation[:3, :3] = np.array(
        [[np.cos(angle), -np.sin(angle), 0.],
         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    proposal = graph_rotation @ _pose(2., .5)
    calls = []
    frozen = []

    class Graph:
        keyframes = ()

        def pose(self, frame):
            return graph_poses[frame].copy()

    def fake_eval(_cache, _params, _cam, _matcher, _mc, *, on_frame):
        graph = Graph()
        for frame in frames:
            graph.keyframes = keyframes_at[frame]
            if frame == 6:
                for old in frames:
                    graph_poses[old] = graph_rotation @ graph_poses[old]
            on_frame(frame, graph)
        return {"ATE_median": 0., "n_loop": 0}

    def fake_solve(window, _pairs, _current, _keypoints, _cam, _opts):
        calls.append(window[-1])
        if window[-1] == 6:
            return {4: proposal.copy()}, {"accepted": True, "reason": ""}
        if window[-1] == 8:
            return {}, {"accepted": False, "reason": "pose jump gate"}
        return {}, {"accepted": False, "reason": "test skip"}

    monkeypatch.setattr(stream, "eval_seq", fake_eval)
    monkeypatch.setattr(stream, "_pair", lambda _c, a, b, _m, _cam:
                        PairMatches(a, b, (), (), (), ()))
    monkeypatch.setattr(stream, "_solve_window", fake_solve)
    real_ate = stream._trajectory_ate

    def capture_frozen(values, *args):
        frozen.append({frame: pose.copy() for frame, pose in values.items()})
        return real_ate(values, *args)

    monkeypatch.setattr(stream, "_trajectory_ate", capture_frozen)
    opts = _opts()
    opts.window_mode = "keyframe"
    result = stream.evaluate_stream(cache, {}, object(), object(), opts)

    assert calls == [6, 8]
    assert result["accepted"] == 1
    # The rejected proposal at frame 6 must not alter the accepted segment.
    np.testing.assert_allclose(frozen[1][4], proposal)
    expected_6 = proposal @ np.linalg.inv(graph_rotation @ _pose(2.)) \
        @ (graph_rotation @ _pose(3.))
    expected_8 = proposal @ np.linalg.inv(graph_rotation @ _pose(2.)) \
        @ (graph_rotation @ _pose(4.))
    np.testing.assert_allclose(frozen[1][6], expected_6)
    np.testing.assert_allclose(frozen[1][8], expected_8)
