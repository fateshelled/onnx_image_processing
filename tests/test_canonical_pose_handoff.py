import numpy as np
import pytest

from vo.joint_prior import JointWindowOptimizer
from vo.online_graph import OnlinePoseGraph
from vo.se3 import se3_exp


@pytest.mark.parametrize("mode", ["canonical_raw", "canonical_conditional",
                                  "canonical_marginal", "canonical_retained",
                                  "canonical_shared"])
@pytest.mark.parametrize("tracking_failure", [False, True])
def test_new_segment_inherits_large_boundary_correction(monkeypatch, mode, tracking_failure):
    g = OnlinePoseGraph(dict(marginalization_backend=mode, kf_max_gap=4,
                            kf_trans_thresh=100, kf_rot_thresh=180,
                            loop_enable=False, nl_reg=False),
                        None, lambda a, b: {"ok": False})
    correction = se3_exp(np.array([.3, -.4, 2.3, 3., -2., 1.]))
    calls = []
    first_raw = {}

    def controlled_solve(window, verbose=False):
        calls.append(1)
        if len(calls) == 1:
            # Model a large common world-frame correction returned by solve1.
            first_raw.update({k: v.copy() for k, v in g._T.items()})
            for k in window.pose_ids:
                window.T[k] = correction @ window.T[k]
        else:
            assert len(calls) == 2
            for k in window.pose_ids:
                np.testing.assert_allclose(window.T[k], correction @ g._T[k], atol=1e-12)
            # New-new and boundary-new odometry remain in the same frame.
            for i, j in ((4, 5), (5, 6), (6, 8)):
                np.testing.assert_allclose(np.linalg.inv(window.T[j]) @ window.T[i],
                                           np.linalg.inv(g._T[j]) @ g._T[i], atol=1e-12)
            for k, raw in first_raw.items():
                np.testing.assert_array_equal(g._T[k], raw)
            # Window construction must not rewrite the canonical prior.
            for prior in g._canonical.priors:
                assert any(prior is current for current in window.joint_priors)
        return 0.

    monkeypatch.setattr(JointWindowOptimizer, "optimize", controlled_solve)
    for i in range(9):
        odom = ((None, None) if tracking_failure and i == 6
                else (se3_exp(np.array([.01, .02, -.03, 0, 0, 0]))[:3, :3],
                      np.array([-1., .1, .2])))
        g.add_frame(i, odom=odom)
    assert len(calls) == 2
    g._canonical.assert_ownership()
