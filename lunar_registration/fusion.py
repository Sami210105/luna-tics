"""
PWIFT-quality-gated fusion with deep neural matchers (RoMa2 / ELoFTR),
MiHo piecewise planar homographies, and gridded GCP selection (§2).
=====================================================================
- Prevents degraded/ambiguous PWIFT from vetoing good deep neural matches.
- Filters in ground units: prior radius alpha * GSD_ref, deduplication beta * GSD_ref.
- MiHo fits local piecewise quadrant transformations to absorb lunar relief parallax.
- Gridded GCP optimizer yields up to 35 well-distributed high-utility ground control points.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .matching import MatchResult


def assess_pwift_quality(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    scores: Optional[np.ndarray] = None,
    n_min: int = 8,
    c_min: int = 3,
    r_min: float = 0.2,
    rmse_max_px: float = 15.0,
) -> Tuple[bool, str]:
    """Evaluate PWIFT match quality before allowing it to constrain downstream fusion."""
    n = len(pts_src)
    if n < n_min or len(pts_dst) < n_min:
        return False, f"insufficient_points: {n} < {n_min}"

    # Occupancy check across 4x4 spatial cells
    x_min, x_max = float(pts_src[:, 0].min()), float(pts_src[:, 0].max())
    y_min, y_max = float(pts_src[:, 1].min()), float(pts_src[:, 1].max())

    if (x_max - x_min) < 1e-3 or (y_max - y_min) < 1e-3:
        return False, "degenerate_spatial_distribution"

    cell_x = np.clip(((pts_src[:, 0] - x_min) / (x_max - x_min + 1e-6) * 4).astype(int), 0, 3)
    cell_y = np.clip(((pts_src[:, 1] - y_min) / (y_max - y_min + 1e-6) * 4).astype(int), 0, 3)
    occupied_cells = len(set(zip(cell_x, cell_y)))

    if occupied_cells < c_min:
        return False, f"low_cell_occupancy: {occupied_cells} < {c_min}"

    # Robust RANSAC check
    ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, inliers = cv2.findHomography(
        pts_src.astype(np.float32),
        pts_dst.astype(np.float32),
        method=ransac_method,
        ransacReprojThreshold=3.0,
        maxIters=1000,
        confidence=0.999,
    )

    if H is None or inliers is None:
        return False, "ransac_homography_failed"

    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(1, n)

    if inlier_ratio < r_min:
        return False, f"low_inlier_ratio: {inlier_ratio:.2f} < {r_min:.2f}"

    # Reprojection RMSE of inliers
    m = inliers.reshape(-1).astype(bool)
    proj = cv2.perspectiveTransform(pts_src[m].reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)).reshape(-1, 2)
    err = np.linalg.norm(proj - pts_dst[m], axis=1)
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) > 0 else float("inf")

    if rmse > rmse_max_px:
        return False, f"high_pwift_rmse: {rmse:.2f}px > {rmse_max_px:.2f}px"

    return True, "pwift_quality_passed"


def fuse_pwift_neural(
    pwift_result: MatchResult,
    neural_result: MatchResult,
    gsd_ref: float = 1.0,
    alpha: float = 2.0,
    beta: float = 1.0,
    pwift_n_min: int = 8,
    pwift_c_min: int = 3,
    pwift_r_min: float = 0.2,
    pwift_rmse_max_px: float = 15.0,
) -> MatchResult:
    """Fuse PWIFT and neural matcher correspondences with quality gating and ground-unit deduplication (§2)."""
    p_pts0 = np.asarray(pwift_result.pts_src, dtype=np.float32)
    p_pts1 = np.asarray(pwift_result.pts_dst, dtype=np.float32)
    p_scores = (
        np.asarray(pwift_result.scores, dtype=np.float32)
        if pwift_result.scores is not None
        else np.ones(len(p_pts0), dtype=np.float32)
    )

    r_pts0 = np.asarray(neural_result.pts_src, dtype=np.float32)
    r_pts1 = np.asarray(neural_result.pts_dst, dtype=np.float32)
    r_scores = (
        np.asarray(neural_result.scores, dtype=np.float32)
        if neural_result.scores is not None
        else np.ones(len(r_pts0), dtype=np.float32)
    )

    hybrid_method_name = f"hybrid_pwift_{neural_result.method}"

    # Quality Gate for PWIFT: prevent degraded PWIFT from corrupting good neural matches
    pw_ok, reason = assess_pwift_quality(
        p_pts0, p_pts1, p_scores,
        n_min=pwift_n_min, c_min=pwift_c_min, r_min=pwift_r_min, rmse_max_px=pwift_rmse_max_px
    )
    if not pw_ok:
        return MatchResult(
            method=hybrid_method_name,
            pts_src=r_pts0,
            pts_dst=r_pts1,
            scores=r_scores,
            provenance="pwift_rejected",
            latency_ms=pwift_result.latency_ms + neural_result.latency_ms,
        )

    if len(r_pts0) == 0:
        return MatchResult(
            method=hybrid_method_name,
            pts_src=p_pts0,
            pts_dst=p_pts1,
            scores=p_scores,
            provenance="pwift_only",
            latency_ms=pwift_result.latency_ms + neural_result.latency_ms,
        )

    # Ground-unit deduplication without single-homography pruning
    scale = max(float(gsd_ref), 1e-4)
    dedupe_thresh_px = float(beta) / scale

    # Soft candidate pooling: deduplicate neural points within dedupe_thresh_px of PWIFT
    d0 = np.linalg.norm(r_pts0[:, None, :] - p_pts0[None, :, :], axis=2)
    d1 = np.linalg.norm(r_pts1[:, None, :] - p_pts1[None, :, :], axis=2)
    dup = np.any((d0 <= dedupe_thresh_px) & (d1 <= dedupe_thresh_px), axis=1)
    kept_r = np.flatnonzero(~dup)

    fused_pts0 = np.concatenate([p_pts0, r_pts0[kept_r]], axis=0).astype(np.float32)
    fused_pts1 = np.concatenate([p_pts1, r_pts1[kept_r]], axis=0).astype(np.float32)
    fused_scores = np.concatenate([p_scores, r_scores[kept_r]], axis=0).astype(np.float32)

    return MatchResult(
        method=hybrid_method_name,
        pts_src=fused_pts0,
        pts_dst=fused_pts1,
        scores=fused_scores,
        provenance=f"fused_pwift_{neural_result.method}",
        latency_ms=pwift_result.latency_ms + neural_result.latency_ms,
    )


def miho_plus_gcp(
    H_coarse: Optional[np.ndarray],
    fused: Union[MatchResult, Dict[str, Any]],
    grid_size: int = 6,
    target_gcps: int = 35,
) -> Dict[str, Any]:
    """MiHo piecewise planar homography clustering + 6x6 gridded GCP selection.

    Produces local homographies Hs_local (2x2 quadrants) and up to target_gcps
    well-distributed Ground Control Points with U-utility.
    """
    if isinstance(fused, MatchResult):
        pts0 = np.asarray(fused.pts_src, dtype=np.float32)
        pts1 = np.asarray(fused.pts_dst, dtype=np.float32)
        scores = (
            np.asarray(fused.scores, dtype=np.float32)
            if fused.scores is not None
            else np.ones(len(pts0), dtype=np.float32)
        )
    else:
        pts0 = np.asarray(fused.get("pts_src", []), dtype=np.float32)
        pts1 = np.asarray(fused.get("pts_dst", []), dtype=np.float32)
        scores = np.asarray(fused.get("scores", np.ones(len(pts0))), dtype=np.float32)

    n = len(pts0)
    if n == 0 or H_coarse is None:
        return {
            "Hs_local": {},
            "gcps": [],
            "coverage": 0.0,
        }

    # Bounding box of correspondences in source coordinates
    x_min, x_max = float(pts0[:, 0].min()), float(pts0[:, 0].max())
    y_min, y_max = float(pts0[:, 1].min()), float(pts0[:, 1].max())
    w_span = max(x_max - x_min, 1.0)
    h_span = max(y_max - y_min, 1.0)

    # 1. MiHo 2x2 quadrant piecewise homographies
    quad_x = (pts0[:, 0] >= (x_min + w_span / 2)).astype(int)
    quad_y = (pts0[:, 1] >= (y_min + h_span / 2)).astype(int)
    quad_ids = quad_y * 2 + quad_x

    ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    Hs_local: Dict[str, np.ndarray] = {}
    for q in range(4):
        mask_q = (quad_ids == q)
        if np.count_nonzero(mask_q) >= 4:
            H_q, in_q = cv2.findHomography(
                pts0[mask_q].astype(np.float32), pts1[mask_q].astype(np.float32),
                method=ransac_method, ransacReprojThreshold=3.0,
                maxIters=1500, confidence=0.999,
            )
            Hs_local[f"quad_{q}"] = H_q if H_q is not None else H_coarse
        else:
            Hs_local[f"quad_{q}"] = H_coarse

    # 2. 6x6 Gridded GCP Selection with U-utility
    proj = cv2.perspectiveTransform(
        pts0.reshape(-1, 1, 2).astype(np.float32), H_coarse.astype(np.float32)
    ).reshape(-1, 2)
    resids = np.linalg.norm(proj - pts1, axis=1)

    # Utility U = score / (1.0 + residual)
    utilities = scores / (1.0 + np.clip(resids, 0.0, 50.0))

    cell_x = np.clip(((pts0[:, 0] - x_min) / w_span * grid_size).astype(int), 0, grid_size - 1)
    cell_y = np.clip(((pts0[:, 1] - y_min) / h_span * grid_size).astype(int), 0, grid_size - 1)
    cell_keys = cell_y * grid_size + cell_x

    best_in_cell: Dict[int, Dict[str, Any]] = {}
    for idx, key in enumerate(cell_keys):
        u = float(utilities[idx])
        if key not in best_in_cell or u > best_in_cell[key]["utility"]:
            best_in_cell[key] = {
                "pt_src": pts0[idx].tolist(),
                "pt_dst": pts1[idx].tolist(),
                "utility": u,
                "residual": float(resids[idx]),
                "cell": (int(cell_x[idx]), int(cell_y[idx])),
            }

    selected_gcps = list(best_in_cell.values())
    selected_gcps.sort(key=lambda x: x["utility"], reverse=True)
    final_gcps = selected_gcps[:target_gcps]

    coverage = float(len(best_in_cell) / (grid_size * grid_size))

    return {
        "Hs_local": Hs_local,
        "gcps": final_gcps,
        "coverage": coverage,
    }
