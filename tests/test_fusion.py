import cv2
import numpy as np
import pytest

from lunar_registration.fusion import assess_pwift_quality, fuse_pwift_neural, miho_plus_gcp
from lunar_registration.matching import MatchResult


def test_bad_pwift_does_not_veto_good_neural():
    pw = MatchResult(
        method="pwift",
        pts_src=np.zeros((3, 2), dtype=np.float32),
        pts_dst=np.zeros((3, 2), dtype=np.float32),
        scores=np.ones(3, dtype=np.float32),
    )
    rng = np.random.default_rng(42)
    ro = MatchResult(
        method="roma2",
        pts_src=rng.uniform(10, 500, size=(200, 2)).astype(np.float32),
        pts_dst=rng.uniform(10, 500, size=(200, 2)).astype(np.float32),
        scores=np.ones(200, dtype=np.float32),
    )

    fused = fuse_pwift_neural(pw, ro, gsd_ref=0.5, alpha=2.0, beta=1.0)
    assert fused.provenance == "pwift_rejected"
    assert len(fused.pts_src) == 200
    assert fused.method == "hybrid_pwift_roma2"


def test_good_pwift_fuses_with_neural():
    rng = np.random.default_rng(42)
    pw_src = rng.uniform(50, 450, size=(24, 2)).astype(np.float32)
    pw_dst = pw_src + np.array([8.0, -4.0], dtype=np.float32)
    pw = MatchResult(
        method="pwift",
        pts_src=pw_src,
        pts_dst=pw_dst,
        scores=np.ones(24, dtype=np.float32),
    )

    ro_src = rng.uniform(50, 450, size=(60, 2)).astype(np.float32)
    ro_dst = ro_src + np.array([8.0, -4.0], dtype=np.float32)
    ro = MatchResult(
        method="roma2",
        pts_src=ro_src,
        pts_dst=ro_dst,
        scores=np.ones(60, dtype=np.float32),
    )

    fused = fuse_pwift_neural(pw, ro, gsd_ref=0.5, alpha=5.0, beta=2.0)
    assert fused.provenance == "fused_pwift_roma2"
    assert len(fused.pts_src) >= 24


def test_miho_plus_gcp_optimizer():
    rng = np.random.default_rng(123)
    pts0 = rng.uniform(10, 500, size=(120, 2)).astype(np.float32)
    pts1 = pts0 + np.array([5.0, -3.0], dtype=np.float32)
    H_coarse = np.eye(3, dtype=np.float64)
    H_coarse[0, 2] = 5.0
    H_coarse[1, 2] = -3.0

    match_res = MatchResult(
        method="roma2",
        pts_src=pts0,
        pts_dst=pts1,
        scores=np.ones(120, dtype=np.float32),
    )
    res = miho_plus_gcp(H_coarse, match_res, target_gcps=35)
    assert "Hs_local" in res
    assert len(res["Hs_local"]) == 4
    assert "gcps" in res
    assert 0 < len(res["gcps"]) <= 35
    assert res["coverage"] > 0.0


def test_soft_candidate_pooling_preserves_parallax():
    """Verify that neural candidates with 3D relief parallax (>3px from planar fit)
    are retained rather than discarded by hard planar gating."""
    pw_src = np.array([
        [50, 50], [450, 50], [50, 450], [450, 450],
        [250, 250], [100, 200], [200, 100], [300, 400],
        [150, 350], [350, 150], [200, 300], [300, 200],
    ], dtype=np.float32)
    pw_dst = pw_src + np.array([5.0, -3.0], dtype=np.float32)
    pw = MatchResult(method="pwift", pts_src=pw_src, pts_dst=pw_dst, scores=np.ones(len(pw_src), dtype=np.float32))

    # Neural matches with crater relief parallax (e.g. 6px deviation from planar translation)
    ro_src = np.array([[120, 120], [380, 380], [180, 280], [320, 160]], dtype=np.float32)
    ro_dst = ro_src + np.array([11.0, -3.0], dtype=np.float32)
    ro = MatchResult(method="roma2", pts_src=ro_src, pts_dst=ro_dst, scores=np.ones(len(ro_src), dtype=np.float32))

    fused = fuse_pwift_neural(pw, ro, gsd_ref=1.0, beta=1.0)
    # Soft candidate pooling pools all neural candidates not within beta/GSD
    assert len(fused.pts_src) == len(pw_src) + len(ro_src)
    assert fused.provenance == "fused_pwift_roma2"

