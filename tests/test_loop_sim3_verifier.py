"""Unit tests for the shared Sim(3) loop verifier used online."""

import math

import numpy as np
import pytest

from vo.loop_sim3_verifier import (
    Sim3LoopVerifier,
    candidate_seed,
    compose_odom,
    point_key,
    translation_angle_deg,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
STRIDE = 10
STRUCTURE = np.array([
    [0.3, 0.1, 2.0], [0.6, -0.2, 2.5], [0.1, 0.3, 3.0], [0.5, 0.4, 3.5],
    [0.2, -0.3, 4.0], [0.7, 0.2, 4.5], [0.4, -0.1, 5.0], [0.0, 0.2, 2.2],
    [0.8, -0.4, 2.8], [0.3, 0.5, 3.2], [0.6, -0.5, 4.2], [0.1, -0.1, 4.8],
])


def _odom(count):
    return [{"ok": True, "R": np.eye(3), "t": np.array([1.0, 0.0, 0.0])}
            for _ in range(count)]


def _keypoints(frame):
    # Cameras recede along -x, so the chain measurement is (I, [1, 0, 0]).
    camera = STRUCTURE + np.array([frame / STRIDE, 0.0, 0.0])
    uv = (K @ camera.T).T
    xy = uv[:, :2] / uv[:, 2:]
    return xy[:, ::-1]


def _match_fn(a, b):
    return _keypoints(a), _keypoints(b)


def _permuted_match_fn(a, b):
    keypoints_a, keypoints_b = _keypoints(a), _keypoints(b)
    if (a, b) == (10, 40):
        order = np.random.default_rng(0).permutation(len(keypoints_b))
        keypoints_b = keypoints_b[order]
    return keypoints_a, keypoints_b


def test_shared_helpers_compose_and_keys():
    odom = _odom(3)
    rotation, translation = compose_odom(odom, 0, 3)
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(translation, [3.0, 0.0, 0.0])
    assert compose_odom(odom, 0, 4) is None
    assert point_key([1.23456, -2.0]) == (1.2346, -2.0)
    assert candidate_seed(0, 10, 40) == candidate_seed(0, 10, 40)
    assert candidate_seed(0, 10, 40) != candidate_seed(0, 10, 41)


def test_translation_angle_handles_direction_and_degenerate_input():
    assert translation_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert translation_angle_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert translation_angle_deg([0, 0, 0], [1, 0, 0]) is None


def test_verifier_accepts_consistent_loop():
    verifier = Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                gate=8, window_strides=2)
    assert verifier(10, 40) is True
    assert verifier.n_accept == 1
    assert verifier.n_reject == 0
    assert verifier.n_abstain == 0


def test_verifier_rejects_inconsistent_correspondence():
    verifier = Sim3LoopVerifier(_odom(5), _permuted_match_fn, K, STRIDE,
                                gate=8, window_strides=2)
    assert verifier(10, 40) is False
    assert verifier.n_reject == 1


def test_verifier_rejects_inconsistent_translation_direction():
    def wrong_pose(a, b):
        return {"ok": True, "t": np.array([-3.0, 0.0, 0.0])}

    verifier = Sim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=wrong_pose, max_translation_angle_deg=30.0)
    assert verifier(10, 40) is False
    assert verifier.n_direction_rejected == 1
    assert verifier.last_translation_angle_deg == pytest.approx(180.0)


def test_verifier_accepts_consistent_translation_direction():
    def matching_pose(a, b):
        return {"ok": True, "t": np.array([3.0, 0.0, 0.0])}

    verifier = Sim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=matching_pose, max_translation_angle_deg=30.0)
    assert verifier(10, 40) is True
    assert verifier.last_translation_angle_deg == pytest.approx(0.0)


@pytest.mark.parametrize("pose", [None, {"ok": False}, {"ok": True, "t": None}])
def test_verifier_direction_gate_abstains_when_pose_is_unavailable(pose):
    verifier = Sim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=lambda a, b: pose, max_translation_angle_deg=30.0)
    assert verifier(10, 40) is False
    assert verifier.n_abstain_no_pose == 1
    assert verifier.n_abstain_no_match == 0


def test_verifier_direction_gate_accept_policy_keeps_unavailable_pose():
    verifier = Sim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=lambda a, b: None, max_translation_angle_deg=30.0,
        abstain_policy="accept")
    assert verifier(10, 40) is True
    assert verifier.n_abstain_no_pose == 1


def test_direction_accept_policy_cannot_bypass_inlier_gate():
    verifier = Sim3LoopVerifier(
        _odom(5), _truncated_match_fn, K, STRIDE, gate=6,
        window_strides=2, min_tracks=5, pose_fn=lambda a, b: None,
        max_translation_angle_deg=30.0, abstain_policy="accept")
    assert verifier(10, 40) is False
    assert verifier.n_abstain_no_pose == 0


def test_verifier_diagnostic_state_is_per_candidate():
    verifier = Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                gate=8, window_strides=2)
    assert verifier(10, 40) is True
    assert verifier.last_scale is not None
    assert verifier(0, 40) is False
    assert verifier.last_scale is None
    assert verifier.last_translation_angle_deg is None
    assert sum(verifier.scale_ratio_hist) == 1


@pytest.mark.parametrize("direction", [[0.0, 0.0, 0.0],
                                         [float("nan"), 0.0, 0.0]])
def test_degenerate_candidate_direction_abstains(direction):
    verifier = Sim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=lambda a, b: {"ok": True, "t": direction},
        max_translation_angle_deg=30.0, abstain_policy="accept")
    assert verifier(10, 40) is True
    assert verifier.n_abstain_no_pose == 1
    assert verifier.n_direction_rejected == 0
    assert sum(verifier.scale_ratio_hist) == 1


def _none_for_loop_pair(a, b):
    if (a, b) == (10, 40):
        return None
    return _match_fn(a, b)


def test_verifier_abstain_policy_for_missing_matches():
    reject = Sim3LoopVerifier(_odom(5), _none_for_loop_pair, K, STRIDE,
                              gate=8, window_strides=2)
    assert reject(10, 40) is False
    assert reject.n_abstain == 1
    assert reject.n_abstain_no_match == 1
    accept = Sim3LoopVerifier(_odom(5), _none_for_loop_pair, K, STRIDE,
                              gate=8, window_strides=2,
                              abstain_policy="accept")
    assert accept(10, 40) is True
    assert accept.n_abstain_no_match == 1


def test_verifier_abstain_policy_without_backward_support():
    verifier = Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                gate=8, window_strides=2)
    # Frame 0 has no preceding frames, so no causal window exists.
    assert verifier(0, 40) is False
    assert verifier.n_abstain_no_cloud == 1
    relaxed = Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                               gate=8, window_strides=2, backward_only=False)
    assert relaxed(0, 40) is True
    assert relaxed.n_abstain == 0


def _truncated_match_fn(a, b):
    keypoints_a, keypoints_b = _keypoints(a), _keypoints(b)
    if (a, b) == (10, 40):
        return keypoints_a[:5], keypoints_b[:5]
    return keypoints_a, keypoints_b


def test_verifier_min_tracks_and_gate_boundaries():
    at_floor = Sim3LoopVerifier(_odom(5), _truncated_match_fn, K, STRIDE,
                                gate=5, window_strides=2, min_tracks=5)
    assert at_floor(10, 40) is True
    above_gate = Sim3LoopVerifier(_odom(5), _truncated_match_fn, K, STRIDE,
                                  gate=6, window_strides=2, min_tracks=5)
    assert above_gate(10, 40) is False
    below_floor = Sim3LoopVerifier(_odom(5), _truncated_match_fn, K, STRIDE,
                                   gate=5, window_strides=2, min_tracks=6)
    assert below_floor(10, 40) is False
    assert below_floor.n_abstain_few_tracks == 1


def test_verifier_cloud_cache_is_bounded():
    verifier = Sim3LoopVerifier(_odom(8), _match_fn, K, STRIDE,
                                gate=8, window_strides=2, cache_size=1)
    verifier(10, 30)
    verifier(20, 40)
    assert len(verifier._clouds) == 1


def _keyframe_window_fn(endpoint):
    if endpoint == 40:
        return [(10, 40, np.eye(3), np.array([3.0, 0.0, 0.0]), False)]
    return []


def test_verifier_uses_injected_keyframe_windows():
    # The injected pose matches the synthetic chain: x_cam(40) = x_cam(10)
    # + [3, 0, 0], so the keyframe-baseline cloud is well formed even though
    # the odometry cache has no window for frame 40.
    verifier = Sim3LoopVerifier(
        _odom(3), _match_fn, K, STRIDE, gate=8, window_strides=2,
        window_fn=_keyframe_window_fn)
    assert verifier(10, 40) is True
    assert verifier.n_accept == 1


def test_verifier_validates_parameters():
    with pytest.raises(ValueError):
        Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE, gate=0)
    with pytest.raises(ValueError):
        Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE, window_strides=0)
    with pytest.raises(ValueError):
        Sim3LoopVerifier(_odom(5), _match_fn, K, STRIDE, abstain_policy="maybe")


def _translated_pose(offset):
    pose = np.eye(4)
    pose[0, 3] = offset
    return pose


def test_keyframe_windows_use_nearest_previous_keyframes():
    rtl = pytest.importorskip("eval.rustuna_tune_loop")

    class Graph:
        keyframes = [0, 10, 20, 30]
        _T = {0: _translated_pose(0.0), 10: _translated_pose(1.0),
              20: _translated_pose(2.0), 30: _translated_pose(3.0)}

    windows = rtl._keyframe_windows(Graph(), 30)
    assert [window[0] for window in windows] == [10, 20]
    # Camera-to-world chain: the pose from frame 10 to 30 is x_30 = x_10 - 2.
    np.testing.assert_allclose(windows[0][3], [-2.0, 0.0, 0.0])
    assert rtl._keyframe_windows(Graph(), 0) == []

    class EndpointOnly:
        keyframes = [30]
        _T = {30: _translated_pose(3.0)}

    assert rtl._keyframe_windows(EndpointOnly(), 30) == []


def test_keyframe_windows_match_composed_odometry():
    rtl = pytest.importorskip("eval.rustuna_tune_loop")

    stride = 10
    odom = [{"ok": True, "R": np.eye(3), "t": np.array([-1.0, 0.0, 0.0])}
            for _ in range(3)]

    class Graph:
        keyframes = [0, 10, 20]
        _T = {}

    for slot in range(3):
        pose = np.eye(4)
        pose[0, 3] = float(slot)
        Graph._T[slot * stride] = pose

    windows = rtl._keyframe_windows(Graph(), 20)
    assert [window[0] for window in windows] == [0, 10]
    for window, (first_slot, last_slot) in zip(windows, [(0, 2), (1, 2)]):
        expected_rotation, expected_translation = compose_odom(
            odom, first_slot, last_slot)
        np.testing.assert_allclose(window[2], expected_rotation)
        np.testing.assert_allclose(window[3], expected_translation)


def test_keyframe_windows_match_composed_odometry_with_rotation():
    rtl = pytest.importorskip("eval.rustuna_tune_loop")

    def rot_z(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    stride = 10
    odom = [{"ok": True, "R": rot_z(0.1 * (slot + 1)),
             "t": np.array([-1.0, 0.0, 0.0])} for slot in range(2)]

    class Graph:
        keyframes = [0, 10, 20]
        _T = {}

    pose = np.eye(4)
    Graph._T[0] = pose.copy()
    for slot, entry in enumerate(odom):
        relative = np.eye(4)
        relative[:3, :3] = entry["R"]
        relative[:3, 3] = entry["t"]
        pose = pose @ np.linalg.inv(relative)
        Graph._T[stride * (slot + 1)] = pose.copy()

    windows = rtl._keyframe_windows(Graph(), 20)
    expected_rotation, expected_translation = compose_odom(odom, 0, 2)
    np.testing.assert_allclose(windows[0][2], expected_rotation)
    np.testing.assert_allclose(windows[0][3], expected_translation)


def test_eval_online_kf_prior_verifier_wiring(monkeypatch):
    rtl = pytest.importorskip("eval.rustuna_tune_loop")
    from vo.pose_estimation import CameraIntrinsics

    captured = {}

    class StubGraph:
        def __init__(self, params, cam, match_fn):
            captured["graph"] = self
            self.loop_verifier = None
            self.n_loop = 0
            self.n_kf = 1
            self.n_cycle_rejected = 0
            self.n_verifier_rejected = 0
            self.n_verifier_abstained = 0
            self.n_robust_downweighted = 0
            self.n_robust_rot_downweighted = 0
            self.n_robust_dir_downweighted = 0
            self.loop_rot_scale = 0.1
            self.loop_dir_scale = 0.4
            self._hist_rot = {}

        def add_frame(self, idx, odom=None):
            return np.eye(4)

        def pose(self, idx):
            pose = np.eye(4)
            pose[0, 3] = float(idx) * 0.1
            return pose

    monkeypatch.setattr(rtl, "OnlinePoseGraph", StubGraph)
    cache = {
        "stride": 10,
        "odom": [{"ok": True, "R": np.eye(3),
                  "t": np.array([1.0, 0.0, 0.0])} for _ in range(4)],
        "gt_pos": np.array([[float(i), 0.0, 0.0] for i in range(5)]),
        "feat": {},
    }
    cam = CameraIntrinsics(525.0, 525.0, 320.0, 240.0, 640, 480)
    match_frames = lambda a, b: None
    raw_match_frames = lambda a, b: (np.zeros((0, 2)), np.zeros((0, 2)))
    assert rtl.SEQ_OPT1_DEFAULTS["loop_verifier"] == "sim3"
    assert rtl.SEQ_OPT1_DEFAULTS["loop_verifier_keyframes"] is True
    assert rtl.SEQ_OPT1_DEFAULTS["loop_verifier_abstain"] == "accept"
    params = {**rtl.SEQ_OPT1_DEFAULTS, "loop_verifier": "sim3",
              "loop_verifier_abstain": "reject"}
    result = rtl._eval_online_kf_prior(cache, params, cam, match_frames,
                                       raw_match_frames=raw_match_frames)
    verifier = captured["graph"].loop_verifier
    assert isinstance(verifier, Sim3LoopVerifier)
    assert verifier.gate == 6
    assert verifier.window_strides == 4
    assert verifier.abstain_policy == "reject"
    assert result["verifier_accept"] == 0

    accept_params = {**rtl.SEQ_OPT1_DEFAULTS, "loop_verifier": "sim3",
                     "loop_verifier_abstain": "accept"}
    rtl._eval_online_kf_prior(cache, accept_params, cam, match_frames,
                              raw_match_frames=raw_match_frames)
    assert captured["graph"].loop_verifier.abstain_policy == "accept"

    kf_params = {**rtl.SEQ_OPT1_DEFAULTS, "loop_verifier": "sim3",
                 "loop_verifier_keyframes": True}
    rtl._eval_online_kf_prior(cache, kf_params, cam, match_frames,
                              raw_match_frames=raw_match_frames)
    assert captured["graph"].loop_verifier.window_fn is not None

    with pytest.raises(ValueError):
        rtl._eval_online_kf_prior(cache, params, cam, match_frames)

    # The shared defaults now enable the verifier; the explicit opt-out keeps
    # the legacy path byte-identical.
    default_params = {**rtl.SEQ_OPT1_DEFAULTS, "loop_verifier": "none"}
    plain = rtl._eval_online_kf_prior(cache, default_params, cam, match_frames,
                                      raw_match_frames=raw_match_frames)
    assert plain["n_verifier_rejected"] == 0
    assert "verifier_accept" not in plain


def test_ab_configs_have_real_baseline_and_keyframe_toggle():
    from scripts.ab_loop_verifier import _configs

    plain = dict(_configs([6], ["accept"], keyframes=False))
    assert plain["baseline"]["loop_verifier"] == "none"
    assert plain["sim3_g6_accept"]["loop_verifier_keyframes"] is False
    keyframe = dict(_configs([6], ["accept"], keyframes=True))
    assert keyframe["sim3k_g6_accept"]["loop_verifier_keyframes"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
