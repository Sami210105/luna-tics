"""
Stage 4 - Viewpoint variation handling
========================================
- "global": single homography fit over all matches (used for IIRS, per your
  architecture notes - low-res, near-nadir, global geometry is a fine model)
- "local": image split into overlapping blocks, one homography fit per block
  then stitched, giving a locally-varying (piecewise-projective) warp that
  better absorbs terrain-relief parallax at high resolution (OHRC/TMC)

Both call into pwift.fsc_homography() for the actual RANSAC + refit.

--- BUGFIXES (this revision) ---
1. `estimate_local_homographies`'s sparse-block fallback used to tag
   *every* point spatially inside an under-populated block as an "inlier"
   (`inlier_mask=in_block`), regardless of whether that point actually
   satisfied any homography. That produced nonsensical "inliers" with no
   coherent geometric relationship (visible directly in the matchpoints
   CSV - wildly inconsistent src->ref displacements among "inlier=1"
   rows). Fixed to intersect with the global RANSAC's own inlier mask,
   same as the "local fit was unstable" fallback path already did.
2. If the global homography itself fails `_homography_is_stable` AND
   every block is too sparse to fit its own local homography, every block
   silently ends up with `H=None`. Downstream, `georeference.warp_local`
   skips every block, `metrics.reprojection_rmse_multi` has nothing to
   average, and the result is a silently-written all-black PNG with
   `rmse_px=NaN` and no indication anything went wrong. This revision
   raises a clear, actionable warning at the point of failure instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import SensorConfig
from .pwift import fsc_homography, reprojection_cleanup


@dataclass
class HomographyResult:
    H: Optional[np.ndarray]           # 3x3, or None if estimation failed
    inlier_mask: np.ndarray           # boolean, aligned with input points
    block: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) if local
    Hs_local: Optional[dict] = None   # MiHo quadrant homographies {quad_0: H, ...}
    gcps: Optional[list] = None       # gridded ground control points


def estimate_global_homography(
    pts_src: np.ndarray, pts_dst: np.ndarray,
    reproj_threshold: float = 3.0, max_iters: int = 5000, confidence: float = 0.999,
    min_inliers_for_stable_fit: int = 8,
    reprojection_cleanup_tau_e: Optional[float] = None,
) -> HomographyResult:
    H, mask = fsc_homography(pts_src, pts_dst, reproj_threshold, max_iters, confidence)
    n_inliers = int(mask.sum()) if mask is not None else 0
    # OpenCV's RANSAC has no fixed seed, so with a small match count it can
    # occasionally converge to a wild, near-degenerate fit even for the
    # global case - reject it explicitly rather than silently returning a
    # transform that blows up on reprojection (this bit us: local blocks
    # falling back to an unstable global H produced nonsensical RMSE).
    if not _homography_is_stable(H, n_inliers, min_inliers_for_stable_fit):
        return HomographyResult(H=None, inlier_mask=np.zeros(len(pts_src), dtype=bool))

    # Paper Eq 26-27: homography-based reprojection cleanup, applied AFTER
    # FSC as a distinct final step - not a replacement for FSC's own
    # (approximate, two-pass) inlier mask, but a further prune on top of it
    # using the fitted H and a fixed pixel-error tolerance (tau_e). This was
    # previously implemented in pwift.py but never actually called from the
    # pipeline; wiring it in here is what the paper's matching stage does.
    if reprojection_cleanup_tau_e is not None and H is not None:
        clean_mask = reprojection_cleanup(pts_src, pts_dst, H, tau_e=reprojection_cleanup_tau_e)
        mask = mask & clean_mask

    return HomographyResult(H=H, inlier_mask=mask)


def _homography_is_stable(H: np.ndarray, n_inliers: int, min_inliers: int,
                           max_scale_factor: float = 6.0) -> bool:
    """Rejects homographies fit from too few points to meaningfully
    constrain 8 degrees of freedom, or that imply an implausible local
    scale/shear (a strong sign of overfitting noise rather than recovering
    real geometry) - both show up as wild coefficients like a 5x local
    scale jump where neighboring blocks show ~1x."""
    if H is None or n_inliers < min_inliers:
        return False
    affine_part = H[:2, :2]
    singular_values = np.linalg.svd(affine_part, compute_uv=False)
    if singular_values.min() < 1e-6:
        return False  # near-singular, would blow up on reprojection
    scale_ratio = singular_values.max() / singular_values.min()
    if scale_ratio > max_scale_factor or singular_values.max() > max_scale_factor:
        return False
    return True


def estimate_local_homographies(
    pts_src: np.ndarray, pts_dst: np.ndarray, image_shape: Tuple[int, int],
    n_blocks: int = 3, overlap: float = 0.25,
    reproj_threshold: float = 3.0, max_iters: int = 5000, confidence: float = 0.999,
    min_points_per_block: int = 8, min_inliers_for_stable_fit: int = 8,
    reprojection_cleanup_tau_e: Optional[float] = None,
) -> List[HomographyResult]:
    """Splits the SOURCE image into an n_blocks x n_blocks grid (with overlap
    so block edges aren't starved of points), fits one homography per block
    from the matches whose source point falls in that block, and falls back
    to the global homography for any block with too few points OR whose
    RANSAC fit turns out unstable (see `_homography_is_stable`) - a local
    fit from a handful of points is not actually more accurate than the
    global fit, it's just noisier, so falling back is the safer choice."""
    h, w = image_shape
    bw, bh = w / n_blocks, h / n_blocks
    results: List[HomographyResult] = []

    global_result = estimate_global_homography(
        pts_src, pts_dst, reproj_threshold, max_iters, confidence, min_inliers_for_stable_fit,
        reprojection_cleanup_tau_e=reprojection_cleanup_tau_e)

    for by in range(n_blocks):
        for bx in range(n_blocks):
            x0 = max(0, bx * bw - overlap * bw)
            x1 = min(w, (bx + 1) * bw + overlap * bw)
            y0 = max(0, by * bh - overlap * bh)
            y1 = min(h, (by + 1) * bh + overlap * bh)

            in_block = (
                (pts_src[:, 0] >= x0) & (pts_src[:, 0] < x1) &
                (pts_src[:, 1] >= y0) & (pts_src[:, 1] < y1)
            )
            block = (int(bx * bw), int(by * bh), int(bw), int(bh))

            if in_block.sum() < min_points_per_block:
                # BUGFIX: this used to be `inlier_mask=in_block`, which
                # marked every point merely *located* in this block as an
                # "inlier" with no check that it actually satisfies any
                # homography. Intersect with the global RANSAC's own
                # inlier mask so only genuinely consistent matches count,
                # same as the "local fit was unstable" branch below does.
                fallback_mask = in_block & global_result.inlier_mask
                results.append(HomographyResult(H=global_result.H, inlier_mask=fallback_mask, block=block))
                continue

            H, mask_local = fsc_homography(
                pts_src[in_block], pts_dst[in_block], reproj_threshold, max_iters, confidence)
            full_mask = np.zeros(len(pts_src), dtype=bool)
            idx = np.nonzero(in_block)[0]
            n_local_inliers = 0
            if mask_local is not None and len(mask_local) == len(idx):
                full_mask[idx[mask_local]] = True
                n_local_inliers = int(mask_local.sum())

            if _homography_is_stable(H, n_local_inliers, min_inliers_for_stable_fit):
                # Eq 26-27, same as the global path: prune the RANSAC
                # inlier set further using the fitted block H and a fixed
                # pixel-error tolerance, on top of (not instead of) the
                # RANSAC mask.
                if reprojection_cleanup_tau_e is not None:
                    clean_mask = reprojection_cleanup(pts_src, pts_dst, H, tau_e=reprojection_cleanup_tau_e)
                    full_mask = full_mask & clean_mask
                results.append(HomographyResult(H=H, inlier_mask=full_mask, block=block))
            else:
                # unstable/underdetermined local fit - use the global
                # homography instead, over the same block's inlier points
                # from the global RANSAC run (not the failed local ones)
                fallback_mask = in_block & global_result.inlier_mask
                results.append(HomographyResult(H=global_result.H, inlier_mask=fallback_mask, block=block))

    if all(r.H is None for r in results):
        # Every block fell back to the global H, and the global fit itself
        # failed `_homography_is_stable` - there is no valid transform
        # anywhere in the image. Warping will produce a blank output and
        # RMSE will be NaN; surface that clearly instead of letting it
        # happen silently (this is exactly the failure mode that produced
        # an all-black registered.png with n_inliers>0 but rmse_px=NaN).
        warnings.warn(
            "estimate_local_homographies: every block's homography is None "
            "(global fit was rejected by _homography_is_stable and no "
            "block had enough points - or a stable fit - to compute its "
            "own). The registered output will be blank. Likely causes: "
            "too few/noisy matches for a 3x3 block grid (try n_blocks=2, "
            "or loosen pwift_ratio_test/pwift_keypoint_threshold to get "
            "more matches), or a genuinely bad match set (check "
            "diagnose.py output) - or relax "
            "_homography_is_stable's max_scale_factor if the fit is "
            "actually fine but is being rejected too aggressively."
        )

    return results


def estimate_viewpoint_transform(
    pts_src: np.ndarray, pts_dst: np.ndarray, sensor: SensorConfig,
    image_shape: Tuple[int, int], cfg,
):
    """Dispatch based on `sensor.homography_mode`. Returns either a single
    HomographyResult (global) or a List[HomographyResult] (local, one per
    block)."""
    if sensor.homography_mode == "global":
        return estimate_global_homography(
            pts_src, pts_dst, cfg.ransac_reproj_threshold_px, cfg.ransac_max_iters, cfg.ransac_confidence,
            reprojection_cleanup_tau_e=cfg.reprojection_cleanup_tau_e_px)
    if sensor.homography_mode == "local":
        return estimate_local_homographies(
            pts_src, pts_dst, image_shape,
            reproj_threshold=cfg.ransac_reproj_threshold_px,
            reprojection_cleanup_tau_e=cfg.reprojection_cleanup_tau_e_px,
            max_iters=cfg.ransac_max_iters, confidence=cfg.ransac_confidence)
    raise ValueError(f"Unknown homography_mode: {sensor.homography_mode}")