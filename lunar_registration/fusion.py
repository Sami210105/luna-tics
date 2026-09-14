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
from .pwift import PWIFTMaps


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


def _extract_structural_and_validity(
    illum: Optional[Union[PWIFTMaps, np.ndarray, Dict[str, Any]]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract structural map (M_PW), Akimov reliability weights (w), and valid mask."""
    if illum is None:
        return None, None, None
    if isinstance(illum, np.ndarray):
        arr = illum.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if arr.size > 0 and float(arr.max()) > 1.0:
            arr = arr / 255.0
        return arr, None, None
    if hasattr(illum, "M_PW"):
        m_pw = np.asarray(illum.M_PW, dtype=np.float32)
        w = np.asarray(illum.w, dtype=np.float32) if getattr(illum, "w", None) is not None else None
        mask = np.asarray(illum.mask) if getattr(illum, "mask", None) is not None else None
        return m_pw, w, mask
    elif isinstance(illum, dict):
        m_pw = np.asarray(illum["M_PW"], dtype=np.float32) if "M_PW" in illum else None
        w = np.asarray(illum["w"], dtype=np.float32) if "w" in illum else None
        mask = np.asarray(illum["mask"]) if "mask" in illum else None
        return m_pw, w, mask
    return None, None, None


def verify_photometric_structural_consistency(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    scores: Optional[np.ndarray] = None,
    src_illum: Optional[Union[PWIFTMaps, np.ndarray, Dict[str, Any]]] = None,
    ref_illum: Optional[Union[PWIFTMaps, np.ndarray, Dict[str, Any]]] = None,
    patch_radius: int = 4,
    min_energy_thresh: float = 0.02,
    reject_thresh: float = 0.1,
    return_weights: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Photometric and structural verification for proposed neural correspondences (§3).

    Evaluates neural candidate correspondences against PWIFT's illumination-invariant
    phase congruency representations (M_PW) and Akimov photometric reliability weights (w):
    1. Photometric Validity: Penalize/reject points falling into unilluminated shadow regions (w < 0.01 or invalid mask).
    2. Energy Assessment: Measure local phase congruency std. If both patches have std < min_energy_thresh,
       the verifier abstains (omega = 1.0) to preserve neural semantic matches across low-contrast lunar maria.
    3. Structural Agreement: Computes normalized cross-correlation (rho) on M_PW patches to verify true
       topological alignment: omega = clip(0.5 + 0.5 * rho, 0.0, 1.0).
    4. Score Modulation & Filtering: Modulates scores s' = s * omega and discards blatant contradictions (omega < reject_thresh).
    """
    pts_s = np.asarray(pts_src, dtype=np.float32)
    pts_d = np.asarray(pts_dst, dtype=np.float32)
    n = len(pts_s)
    s = (
        np.asarray(scores, dtype=np.float32)
        if scores is not None
        else np.ones(n, dtype=np.float32)
    )

    if n == 0 or src_illum is None or ref_illum is None:
        if return_weights:
            return pts_s, pts_d, s, np.ones(n, dtype=np.float32)
        return pts_s, pts_d, s

    map_src, w_src, mask_src = _extract_structural_and_validity(src_illum)
    map_ref, w_ref, mask_ref = _extract_structural_and_validity(ref_illum)

    if map_src is None or map_ref is None:
        if return_weights:
            return pts_s, pts_d, s, np.ones(n, dtype=np.float32)
        return pts_s, pts_d, s

    h_s, w_s = map_src.shape[:2]
    h_r, w_r = map_ref.shape[:2]
    r = max(1, int(patch_radius))

    # Pad maps to safely extract (2r+1, 2r+1) patches for any in-bound keypoint
    padded_src = np.pad(map_src, pad_width=r, mode="reflect")
    padded_ref = np.pad(map_ref, pad_width=r, mode="reflect")

    omegas = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x_s, y_s = float(pts_s[i, 0]), float(pts_s[i, 1])
        x_r, y_r = float(pts_d[i, 0]), float(pts_d[i, 1])

        ix_s, iy_s = int(round(x_s)), int(round(y_s))
        ix_r, iy_r = int(round(x_r)), int(round(y_r))

        # 1. Bounds check
        if ix_s < 0 or ix_s >= w_s or iy_s < 0 or iy_s >= h_s:
            omegas[i] = 0.0
            continue
        if ix_r < 0 or ix_r >= w_r or iy_r < 0 or iy_r >= h_r:
            omegas[i] = 0.0
            continue

        # Photometric validity check (Akimov reliability and shadow mask)
        if w_src is not None and float(w_src[iy_s, ix_s]) < 0.01:
            omegas[i] = 0.0
            continue
        if w_ref is not None and float(w_ref[iy_r, ix_r]) < 0.01:
            omegas[i] = 0.0
            continue
        if mask_src is not None and not bool(mask_src[iy_s, ix_s]):
            omegas[i] = 0.0
            continue
        if mask_ref is not None and not bool(mask_ref[iy_r, ix_r]):
            omegas[i] = 0.0
            continue

        # Extract (2r+1, 2r+1) patches from padded maps
        patch_s = padded_src[iy_s : iy_s + 2 * r + 1, ix_s : ix_s + 2 * r + 1]
        patch_r = padded_ref[iy_r : iy_r + 2 * r + 1, ix_r : ix_r + 2 * r + 1]

        # 2. Energy Assessment (std of phase congruency map)
        std_s = float(np.std(patch_s))
        std_r = float(np.std(patch_r))

        if std_s < min_energy_thresh and std_r < min_energy_thresh:
            # Abstain rule: smooth featureless terrain (maria), preserve neural confidence
            omegas[i] = 1.0
            continue

        # 3. Structural Agreement via Local Patch NCC
        if std_s < 1e-7 or std_r < 1e-7:
            # One side flat, other textured -> structural mismatch
            rho = 0.0
        else:
            norm_s = patch_s - np.mean(patch_s)
            norm_r = patch_r - np.mean(patch_r)
            denom = float(patch_s.size * std_s * std_r)
            rho = float(np.sum(norm_s * norm_r) / (denom + 1e-12))
            rho = float(np.clip(rho, -1.0, 1.0))

        omega = float(np.clip(0.5 + 0.5 * rho, 0.0, 1.0))
        omegas[i] = omega

    # 4. Score Modulation & Filtering
    verified_scores = s * omegas
    keep = omegas >= float(reject_thresh)

    v_pts_s = pts_s[keep]
    v_pts_d = pts_d[keep]
    v_scores = verified_scores[keep]
    v_omegas = omegas[keep]

    if return_weights:
        return v_pts_s, v_pts_d, v_scores, v_omegas
    return v_pts_s, v_pts_d, v_scores


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
    src_illum: Optional[Union[PWIFTMaps, np.ndarray, Dict[str, Any]]] = None,
    ref_illum: Optional[Union[PWIFTMaps, np.ndarray, Dict[str, Any]]] = None,
    verify_photometric: bool = True,
    patch_radius: int = 4,
    min_energy_thresh: float = 0.02,
    reject_thresh: float = 0.1,
) -> MatchResult:
    """Fuse PWIFT and neural matcher correspondences with quality gating,
    photometric-structural verification, and ground-unit deduplication (§2)."""
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

    # Photometric & Structural Verification on neural proposals
    if verify_photometric and (src_illum is not None and ref_illum is not None) and len(r_pts0) > 0:
        r_pts0, r_pts1, r_scores = verify_photometric_structural_consistency(
            r_pts0, r_pts1, r_scores,
            src_illum=src_illum,
            ref_illum=ref_illum,
            patch_radius=patch_radius,
            min_energy_thresh=min_energy_thresh,
            reject_thresh=reject_thresh,
        )

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
    if len(p_pts0) > 0 and len(r_pts0) > 0:
        d0 = np.linalg.norm(r_pts0[:, None, :] - p_pts0[None, :, :], axis=2)
        d1 = np.linalg.norm(r_pts1[:, None, :] - p_pts1[None, :, :], axis=2)
        dup = np.any((d0 <= dedupe_thresh_px) & (d1 <= dedupe_thresh_px), axis=1)
        kept_r = np.flatnonzero(~dup)
    else:
        kept_r = np.arange(len(r_pts0))

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
