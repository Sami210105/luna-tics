"""
Orthogonal Cheap-Gate and Rigid-Only Hypothesis Competition (§§1, 4).
======================================================================
- Rejects confidently-wrong periodic crater shifts via rigid similarity projection
  and margin competition.
- Enforces orthogonal escalation: 256px phase-correlation agreement, structural-NCC
  on gradient maps, scale prior consistency, and coverage.
- Time-bounded verification budget (fails open to robust path if exceeded).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def compute_structural_ncc(
    img0: np.ndarray,
    img1: np.ndarray,
    H: np.ndarray,
    max_patches: int = 200,
) -> float:
    """Compute structural normalized cross-correlation on gradient maps under homography H.

    Sobel gradient magnitudes are robust to inverted shadow orientations under changing
    sun angles on the lunar surface.
    """
    if H is None:
        return 0.0

    h, w = img1.shape[:2]

    def _grad_mag(im: np.ndarray) -> np.ndarray:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    g0 = _grad_mag(img0)
    g1 = _grad_mag(img1)

    warped_g0 = cv2.warpPerspective(g0, H.astype(np.float64), (w, h))
    mask = cv2.warpPerspective(
        np.ones_like(g0, dtype=np.uint8) * 255, H.astype(np.float64), (w, h)
    ) > 0

    if float(mask.mean()) < 0.02:
        return 0.0

    x = warped_g0[mask].ravel().astype(np.float64)
    y = g1[mask].ravel().astype(np.float64)

    # Subsample if large to keep verification fast
    max_pts = max_patches * 64
    if len(x) > max_pts:
        step = len(x) // max_pts
        x = x[::step]
        y = y[::step]

    xz = x - x.mean()
    yz = y - y.mean()
    denom = math.sqrt(float((xz**2).sum() * (yz**2).sum()))
    if denom <= 1e-8:
        return 0.0

    return float(np.clip((xz * yz).sum() / denom, -1.0, 1.0))


def _coarse_phase_align_256(img0: np.ndarray, img1: np.ndarray) -> Dict[str, float]:
    """Lightweight 256px phase correlation coarse shift and peak sharpness."""
    def _to_gray_256(im: np.ndarray) -> np.ndarray:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(im.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
        if resized.max() > 1.0:
            resized = resized / 255.0
        return resized

    g0 = _to_gray_256(img0)
    g1 = _to_gray_256(img1)

    hann = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    (dx_256, dy_256), response = cv2.phaseCorrelate(g0, g1, window=hann)

    # Scale back translation to img1 dimensions
    h1, w1 = img1.shape[:2]
    dx = float(dx_256 * (w1 / 256.0))
    dy = float(dy_256 * (h1 / 256.0))

    return {"dx": dx, "dy": dy, "peak_sharpness": float(response)}


def gate_cheap(
    warp_candidate: Optional[np.ndarray],
    img0: np.ndarray,
    img1: np.ndarray,
    gsd_src: float = 1.0,
    gsd_ref: float = 1.0,
    t_struct: float = 0.40,
    tau_agree: float = 24.0,
    k_sigma: float = 3.0,
    budget_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Orthogonal escalation gate (§1).

    Checks:
    1. 256px phase correlation agreement with warp translation within tau_agree
    2. Structural-NCC on gradient maps >= t_struct
    3. Implied scale within k_sigma * sigma_logscale of GSD prior
    Budget capped at budget_ms; over-budget fails open to robust path.
    """
    t0 = time.perf_counter()

    if warp_candidate is None:
        cost = (time.perf_counter() - t0) * 1000.0
        return {"pass": False, "reason": "no_warp_emitted", "cost_ms": cost}

    # 1. 256px Phase correlation agreement
    coarse = _coarse_phase_align_256(img0, img1)
    tx = float(warp_candidate[0, 2])
    ty = float(warp_candidate[1, 2])
    err_agree = math.hypot(tx - coarse["dx"], ty - coarse["dy"])

    if err_agree > tau_agree:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"phase_correlation_disagreement: {err_agree:.1f}px > {tau_agree:.1f}px",
            "cost_ms": cost,
        }

    # 2. Implied scale vs GSD prior
    s_prior = max(float(gsd_ref) / max(float(gsd_src), 1e-4), 1e-4)
    det = abs(warp_candidate[0, 0] * warp_candidate[1, 1] - warp_candidate[0, 1] * warp_candidate[1, 0])
    implied_scale = math.sqrt(max(det, 1e-6))
    log_err = abs(math.log(max(implied_scale, 1e-4) / s_prior))
    sigma_logscale = max(0.1, 0.5 * (1.0 - min(coarse["peak_sharpness"], 1.0)))

    if log_err > k_sigma * sigma_logscale:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"implied_scale_violation: log_err {log_err:.2f} > {k_sigma * sigma_logscale:.2f}",
            "cost_ms": cost,
        }

    # 3. Structural-NCC
    struct_ncc = compute_structural_ncc(img0, img1, warp_candidate)
    if struct_ncc < t_struct:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"structural_ncc_low: {struct_ncc:.3f} < {t_struct:.3f}",
            "cost_ms": cost,
        }

    cost = (time.perf_counter() - t0) * 1000.0
    if budget_ms is not None and cost > budget_ms:
        return {
            "pass": False,
            "reason": f"verification_budget_exceeded ({cost:.1f}ms > {budget_ms:.1f}ms) -> fail open",
            "cost_ms": cost,
        }

    return {"pass": True, "reason": "all_checks_passed", "cost_ms": cost, "struct_ncc": struct_ncc}


def compete_rigid(
    candidates: List[Dict[str, Any]],
    terrain_spacing_px: float = 24.0,
    lambda_dof: float = 0.1,
    margin_min: float = 0.05,
) -> Dict[str, Any]:
    """Rigid-only hypothesis competition (§4).

    Resolves repeating crater false-matching:
    1. Scores: score = fit + coverage - lambda_dof * dof_resid
    2. Identifies competing modes separated by > terrain_spacing_px.
    3. If margin Δ = best - second < margin_min, flags ambiguous=True.
    """
    if not candidates:
        return {
            "winner": None,
            "margin": 0.0,
            "ambiguous": True,
            "reason": "no_candidates",
        }

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        warp = np.asarray(c.get("warp", np.eye(3)), dtype=np.float64)
        fit = float(c.get("fit", 0.0))
        cov = float(c.get("coverage", 0.0))
        dof_resid = float(c.get("dof_resid", 0.0))
        score = fit + cov - (lambda_dof * dof_resid)

        tx = float(warp[0, 2])
        ty = float(warp[1, 2])

        entry = dict(c)
        entry["score"] = score
        entry["tx"] = tx
        entry["ty"] = ty
        scored.append(entry)

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    winner = scored[0]

    if len(scored) == 1:
        return {
            "winner": winner,
            "margin": float("inf"),
            "ambiguous": False,
            "all_scored": scored,
        }

    # Find competing hypotheses separated by > terrain_spacing_px
    best_tx, best_ty = winner["tx"], winner["ty"]
    competing_runners = [
        c for c in scored[1:]
        if math.hypot(c["tx"] - best_tx, c["ty"] - best_ty) >= terrain_spacing_px
    ]

    if not competing_runners:
        margin = float(winner["score"] - scored[1]["score"])
        return {
            "winner": winner,
            "margin": margin,
            "ambiguous": False,
            "all_scored": scored,
        }

    second_best = competing_runners[0]
    margin = float(winner["score"] - second_best["score"])
    ambiguous = bool(margin < margin_min)

    return {
        "winner": winner,
        "runner_up": second_best,
        "margin": margin,
        "ambiguous": ambiguous,
        "all_scored": scored,
    }
