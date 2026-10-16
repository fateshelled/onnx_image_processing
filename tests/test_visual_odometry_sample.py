"""Integration tests for the visual-odometry sample wiring."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sample"))

from sample import visual_odometry as sample  # noqa: E402


class _Io:
    def __init__(self, name):
        self.name = name


class _Session:
    def get_inputs(self):
        return [_Io("image1"), _Io("image2")]

    def get_outputs(self):
        return [_Io("keypoints1"), _Io("keypoints2"), _Io("matching_probs")]


class _Reader:
    is_camera = False

    def __init__(self, count):
        self.remaining = count

    def read(self):
        if self.remaining == 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((2, 2), dtype=np.uint8)


class _Matcher:
    def __init__(self, *_args, **_kwargs):
        self.debug_display = False
        self.last = None

    def match(self, *_args):
        return {"ok": False, "n_matches": 0, "inlier_ratio": 0.0}


class _CorrectingGraph:
    def __init__(self, *_args, **_kwargs):
        self.poses = {}
        self.match_keyframes = []
        self.last_stats = {"ok": True, "n_matches": 0}
        self.n_kf = 0
        self.n_loop = 0

    def add_frame(self, frame_id):
        pose = np.eye(4)
        pose[0, 3] = float(frame_id)
        self.poses[frame_id] = pose
        if frame_id == 1:
            # Simulate a graph optimization revising an earlier pose.
            self.poses[0] = self.poses[0].copy()
            self.poses[0][0, 3] = 10.0
            self.poses[1][0, 3] = 11.0
        return self.poses[frame_id].copy()

    def pose(self, frame_id):
        return self.poses[frame_id].copy()


def test_run_returns_one_final_graph_pose_per_frame(monkeypatch):
    monkeypatch.setattr(sample, "OnnxSessionMatcher", _Matcher)
    monkeypatch.setattr(sample, "OnlinePoseGraph", _CorrectingGraph)

    trajectory, _graph = sample.run_visual_odometry(
        _Session(), _Reader(2), camera_intrinsics=None,
        model_height=2, model_width=2, verbose=False)

    assert len(trajectory) == 2
    assert np.allclose(trajectory.get_positions_array(),
                       [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])


def test_run_with_no_frames_returns_empty_trajectory(monkeypatch):
    monkeypatch.setattr(sample, "OnnxSessionMatcher", _Matcher)
    monkeypatch.setattr(sample, "OnlinePoseGraph", _CorrectingGraph)

    trajectory, _graph = sample.run_visual_odometry(
        _Session(), _Reader(0), camera_intrinsics=None,
        model_height=2, model_width=2, verbose=False)

    assert len(trajectory) == 0
