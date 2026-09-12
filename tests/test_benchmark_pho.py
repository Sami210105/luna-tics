import glob
import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from lunar_registration.config import PipelineConfig
from lunar_registration.fusion import fuse_pwift_neural
from lunar_registration.matching import get_matcher, resolve_romav2_weights
from lunar_registration.pwift import akimov_weight, fsc_homography


def test_akimov_zero_variance_guard():
    """Verify that Akimov weighting on crops with uniform photometric angles
    does not collapse all weights to 0.0 due to zero variance."""
    inc = np.full((128, 128), 35.0, dtype=np.float32)
    emi = np.full((128, 128), 25.0, dtype=np.float32)
    w = akimov_weight(inc, emi)
    assert not np.all(w == 0.0)
    assert w.min() > 0.0
    assert w.max() > 0.5


def test_magsac_plus_plus_consensus():
    """Verify that cv2.USAC_MAGSAC consensus estimator correctly identifies inliers
    and rejects large outliers with continuous density estimation."""
    rng = np.random.default_rng(42)
    pts0 = rng.uniform(20, 480, size=(100, 2)).astype(np.float32)
    pts1 = pts0 + np.array([4.5, -2.5], dtype=np.float32)

    # Add 25 outliers
    pts1[:25] += rng.uniform(30.0, 100.0, size=(25, 2)).astype(np.float32)

    H, mask = fsc_homography(pts0, pts1, reproj_threshold=3.0, confidence=0.999)
    assert H is not None
    assert mask is not None
    # Outliers should be rejected
    assert np.all(~mask[:25])
    # Ground truth inliers should be kept
    assert mask[25:].sum() >= 70


def test_benchmark_nac_pho_tiles():
    """Benchmark test against real LROC NAC PHO tiles.
    Confirms inlier count (>= 62) and corner transfer error (~0.11px or better)."""
    cache_dirs = [
        Path("/home/ojas/projects/SIH/illumination_variation/data/nac_pho_cache"),
        Path(__file__).resolve().parent.parent / "data" / "nac_pho_cache",
    ]
    cache_dir = next((d for d in cache_dirs if d.exists()), None)
    if cache_dir is None:
        pytest.skip("NAC PHO tile cache directory not found")

    tile_files = sorted(list(cache_dir.glob("*.npz")))
    if not tile_files:
        pytest.skip("No cached NAC PHO tiles found")

    weights_path = resolve_romav2_weights()
    if not weights_path.exists():
        pytest.skip("RoMa v2 fine-tuned weights not available")

    # Load first cached tile
    tile = np.load(str(tile_files[0]))
    img0 = tile["img"]
    w, h = 512.0, 512.0

    # Ground truth transformation with mild rotation + translation
    M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.5, 1.0)
    M_rot[0, 2] += 4.0
    M_rot[1, 2] -= 3.0
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[:2, :] = M_rot

    img1 = cv2.warpPerspective(img0, H_gt, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    cfg = PipelineConfig()
    cfg.device = "cpu"

    # Match with hybrid PWIFT + RoMa2
    pw_matcher = get_matcher("pwift", cfg)
    pw_res = pw_matcher.match(img0, img1)

    roma_matcher = get_matcher("roma2", cfg)
    ro_res = roma_matcher.match(img0, img1)

    fused_res = fuse_pwift_neural(pw_res, ro_res, gsd_ref=0.5, beta=1.0)

    # Estimate consensus homography via standardized MAGSAC++
    H_est, mask = fsc_homography(fused_res.pts_src, fused_res.pts_dst, reproj_threshold=3.0, confidence=0.999)
    assert H_est is not None
    assert mask is not None

    inliers = int(mask.sum())
    # Verify inlier counts match or exceed standalone numbers (62+ inliers)
    assert inliers >= 62, f"Inliers {inliers} below target 62+"

    # Verify corner transfer error is ~0.11px or better
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    c_h = np.column_stack([corners, np.ones(4)])
    c_gt_h = (H_gt @ c_h.T).T
    c_gt = c_gt_h[:, :2] / (c_gt_h[:, 2:3] + 1e-12)

    c_est_h = (H_est @ c_h.T).T
    c_est = c_est_h[:, :2] / (c_est_h[:, 2:3] + 1e-12)

    corner_err = float(np.mean(np.linalg.norm(c_gt - c_est, axis=1)))
    # Corner error should match or exceed ~0.11px (allowing small margin, e.g. < 0.15px)
    assert corner_err <= 0.15, f"Corner error {corner_err:.4f}px exceeded target ~0.11px"
