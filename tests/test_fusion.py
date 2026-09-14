import cv2
import numpy as np
import pytest

from lunar_registration.fusion import (
    assess_pwift_quality,
    fuse_pwift_neural,
    miho_plus_gcp,
    verify_photometric_structural_consistency,
)
from lunar_registration.matching import MatchResult
from lunar_registration.pwift import PWIFTMaps


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


def test_photometric_structural_verification_consistent():
    """Verify high agreement and preserved confidence for true structural correspondences."""
    y, x = np.ogrid[:128, :128]
    crater = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 12.0**2)).astype(np.float32)

    pts = np.array([[64.0, 64.0]], dtype=np.float32)
    scores = np.array([0.95], dtype=np.float32)

    v_src, v_dst, v_scores = verify_photometric_structural_consistency(
        pts, pts, scores=scores,
        src_illum=crater, ref_illum=crater,
        patch_radius=4, min_energy_thresh=0.02, reject_thresh=0.1
    )

    assert len(v_src) == 1
    assert v_scores[0] >= 0.90


def test_photometric_structural_verification_abstain_maria():
    """Verify abstain behavior (omega = 1.0) on uniform low-contrast maria terrain."""
    flat_src = np.full((128, 128), 0.1, dtype=np.float32)
    flat_ref = np.full((128, 128), 0.1, dtype=np.float32)

    pts = np.array([[64.0, 64.0]], dtype=np.float32)
    scores = np.array([0.85], dtype=np.float32)

    v_src, v_dst, v_scores = verify_photometric_structural_consistency(
        pts, pts, scores=scores,
        src_illum=flat_src, ref_illum=flat_ref,
        patch_radius=4, min_energy_thresh=0.02, reject_thresh=0.1
    )

    assert len(v_src) == 1
    # Neural confidence must be preserved exactly by the abstain rule
    assert v_scores[0] == pytest.approx(0.85)


def test_photometric_structural_verification_shadow_pruning():
    """Verify rejection of candidates located in invalid illumination masks or deep shadow."""
    m_pw = np.ones((128, 128), dtype=np.float32)
    w_valid = np.ones((128, 128), dtype=np.float32)
    mask_valid = np.ones((128, 128), dtype=bool)

    w_shadow = np.ones((128, 128), dtype=np.float32)
    w_shadow[64, 64] = 0.002  # deep shadow, < 0.01

    mask_shadow = np.ones((128, 128), dtype=bool)
    mask_shadow[64, 64] = False

    maps_shadow = PWIFTMaps(
        M_PW=m_pw, m_PW=m_pw * 0.5, MIM=np.zeros((128, 128), dtype=np.int32),
        w=w_shadow, w_soft=w_shadow, mask=mask_shadow
    )
    maps_ref = PWIFTMaps(
        M_PW=m_pw, m_PW=m_pw * 0.5, MIM=np.zeros((128, 128), dtype=np.int32),
        w=w_valid, w_soft=w_valid, mask=mask_valid
    )

    pts = np.array([[64.0, 64.0]], dtype=np.float32)
    v_src, v_dst, v_scores = verify_photometric_structural_consistency(
        pts, pts, scores=np.array([0.9], dtype=np.float32),
        src_illum=maps_shadow, ref_illum=maps_ref,
        patch_radius=4, reject_thresh=0.1
    )
    # Shadow point must be pruned
    assert len(v_src) == 0

    # Also test contradictory polarity (inverted patch, rho = -1.0)
    y, x = np.ogrid[:128, :128]
    bump = (np.sin(x * 0.15) * np.cos(y * 0.15)).astype(np.float32)
    inverted = (-bump).astype(np.float32)

    v_src2, v_dst2, v_scores2 = verify_photometric_structural_consistency(
        pts, pts, scores=np.array([0.9], dtype=np.float32),
        src_illum=bump, ref_illum=inverted,
        patch_radius=4, min_energy_thresh=0.02, reject_thresh=0.1
    )
    # Blatant structural contradiction must be pruned
    assert len(v_src2) == 0


def test_fuse_pwift_neural_with_photometric_verification():
    """Verify that fuse_pwift_neural applies photometric verification to neural candidates."""
    y, x = np.ogrid[:128, :128]
    crater = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 12.0**2)).astype(np.float32)

    # Valid PWIFT anchors
    pw_src = np.array([
        [20, 20], [100, 20], [20, 100], [100, 100],
        [40, 60], [60, 40], [80, 60], [60, 80],
    ], dtype=np.float32)
    pw_dst = pw_src.copy()
    pw = MatchResult(method="pwift", pts_src=pw_src, pts_dst=pw_dst, scores=np.ones(len(pw_src), dtype=np.float32))

    # Neural candidates: one valid, one in deep shadow
    ro_src = np.array([[64.0, 64.0], [10.0, 10.0]], dtype=np.float32)
    ro_dst = np.array([[64.0, 64.0], [10.0, 10.0]], dtype=np.float32)
    ro = MatchResult(method="roma2", pts_src=ro_src, pts_dst=ro_dst, scores=np.array([0.9, 0.9], dtype=np.float32))

    w_src = np.ones((128, 128), dtype=np.float32)
    w_src[10, 10] = 0.0  # shadow at (10, 10)
    mask_src = np.ones((128, 128), dtype=bool)
    mask_src[10, 10] = False

    maps_src = PWIFTMaps(
        M_PW=crater, m_PW=crater * 0.5, MIM=np.zeros((128, 128), dtype=np.int32),
        w=w_src, w_soft=w_src, mask=mask_src
    )
    maps_ref = PWIFTMaps(
        M_PW=crater, m_PW=crater * 0.5, MIM=np.zeros((128, 128), dtype=np.int32),
        w=np.ones((128, 128), dtype=np.float32), w_soft=np.ones((128, 128), dtype=np.float32),
        mask=np.ones((128, 128), dtype=bool)
    )

    fused = fuse_pwift_neural(
        pw, ro, gsd_ref=1.0, beta=1.0,
        src_illum=maps_src, ref_illum=maps_ref,
        verify_photometric=True,
    )

    assert fused.provenance == "fused_pwift_roma2"
    # Should include 8 PWIFT points + 1 verified neural point (the shadow point was pruned)
    assert len(fused.pts_src) == 9

