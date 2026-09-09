"""
Evaluation metrics: RMSE, inlier count/ratio, spatial uniformity of matches
across the image (per PS deliverable: "sub-pixel accuracy ... maintaining
uniform distribution across the images").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MatchMetrics:
    method: str
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    rmse_px: float
    uniformity_score: float  # 0-1, 1 = perfectly even coverage across grid cells


def reprojection_rmse(pts_src: np.ndarray, pts_dst: np.ndarray, H: np.ndarray) -> float:
    if H is None or len(pts_src) == 0:
        return float("nan")
    ones = np.ones((len(pts_src), 1), dtype=np.float32)
    homog = np.hstack([pts_src, ones])
    proj = (H @ homog.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
    errors = np.linalg.norm(proj - pts_dst, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def reprojection_rmse_multi(pts_src: np.ndarray, pts_dst: np.ndarray, hom_results) -> float:
    """RMSE across a *local* (piecewise) homography fit: each block's points
    are reprojected through that block's own H, never through a
    neighboring block's H. Reprojecting pooled points through a single
    arbitrarily-chosen block's H (what a naive single-H RMSE would do here)
    produces nonsensical, huge errors for every point outside that one
    block's footprint - this is the fix for exactly that bug."""
    all_errors = []
    for r in hom_results:
        if r.H is None or not np.any(r.inlier_mask):
            continue
        idx = np.nonzero(r.inlier_mask)[0]
        ones = np.ones((len(idx), 1), dtype=np.float32)
        homog = np.hstack([pts_src[idx], ones])
        proj = (r.H @ homog.T).T
        proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
        errors = np.linalg.norm(proj - pts_dst[idx], axis=1)
        all_errors.append(errors)
    if not all_errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(all_errors) ** 2)))


def spatial_uniformity(
    pts: np.ndarray, image_shape: Tuple[int, int], grid: int = 8,
) -> float:
    """Fraction-of-cells-occupied style uniformity score: divides the image
    into a grid x grid grid, counts matches per cell, and scores 1.0 for a
    perfectly flat distribution across all occupied cells (low coefficient
    of variation) and lower for matches clustered in a few cells."""
    if len(pts) == 0:
        return 0.0
    h, w = image_shape
    cell_h, cell_w = h / grid, w / grid
    counts = np.zeros((grid, grid), dtype=np.int32)
    for x, y in pts:
        cx = min(grid - 1, max(0, int(x // cell_w)))
        cy = min(grid - 1, max(0, int(y // cell_h)))
        counts[cy, cx] += 1

    occupied = counts[counts > 0]
    if len(occupied) == 0:
        return 0.0
    coverage = len(occupied) / (grid * grid)          # how much of the image has any match
    cv = occupied.std() / (occupied.mean() + 1e-8)      # spread among occupied cells
    evenness = 1.0 / (1.0 + cv)
    return float(coverage * evenness)


def compute_metrics(
    method: str, pts_src: np.ndarray, pts_dst: np.ndarray,
    inlier_mask: np.ndarray, H_or_local_results, image_shape: Tuple[int, int], grid: int = 8,
) -> MatchMetrics:
    """`H_or_local_results` is either a single 3x3 homography (global mode)
    or a list of HomographyResult (local/piecewise mode, from
    viewpoint.estimate_local_homographies) - pass whichever
    `estimate_viewpoint_transform` returned, unpacked as-is."""
    n_matches = len(pts_src)
    n_inliers = int(inlier_mask.sum()) if n_matches else 0
    inlier_ratio = n_inliers / n_matches if n_matches else 0.0
    if n_inliers == 0:
        rmse = float("nan")
    elif isinstance(H_or_local_results, list):
        rmse = reprojection_rmse_multi(pts_src, pts_dst, H_or_local_results)
    else:
        rmse = reprojection_rmse(pts_src[inlier_mask], pts_dst[inlier_mask], H_or_local_results)
    uniformity = spatial_uniformity(pts_src[inlier_mask], image_shape, grid) if n_inliers else 0.0
    return MatchMetrics(
        method=method, n_matches=n_matches, n_inliers=n_inliers,
        inlier_ratio=inlier_ratio, rmse_px=rmse, uniformity_score=uniformity,
    )