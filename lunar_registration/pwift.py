"""
PWIFT-style illumination-invariant keypoints & descriptors.

--- WHAT'S PAPER-EXACT NOW (PWIFT.pdf, Yan/Guo/Zeng, Aerosp. Sci. Technol. 179
    (2026) 113462) AND WHAT ISN'T ---

Implemented to the equations in the paper (function -> equation mapping):
    akimov_weight                          -> Eq 1-2
    compute_valid_mask, soft_weight         -> Eq 3-6
    photometric_weighted_structural_maps    -> Eq 7-8 (M_PW, m_PW, MIM)
    detect_keypoints_pw                     -> Eq 9, Sec 3.4 candidate/NMS/retention logic
    dominant_orientation_pw                 -> Eq 10-11
    compute_bichannel_descriptors           -> Eq 12-18
    swap_aware_match                        -> Eq 19-21
    coarse_to_fine_rotation_scale           -> Eq 22-25
    reprojection_cleanup                    -> Eq 26-27

STILL A SWAP POINT (the paper's extracted text doesn't give these in closed
form / as a number):
  - D_Aki(alpha, beta, gamma) itself: the paper cites this to ref [45]
    (Schroder et al. 2018, Ceres opposition-effect disk function) and only
    gives you its *contract* (Eq 1-2: zero on invalid pixels, min-max
    normalized over the valid region). `akimov_weight()` below satisfies
    that contract using a Lommel-Seeliger-type limb/terminator-attenuation
    disk function with an optional phase-darkening term - it is NOT a
    transcription of [45]. Swap in the real Akimov photometric-coordinate
    formulation if you need paper-exact fidelity for a report.
  - w_soft_lo/hi (Eq 6) and gamma_pc (Eq 7): described qualitatively, no
    numeric values given. See config.py's PipelineConfig for the current
    defaults - tune against your own imagery.
  - The "adaptive quality screening" rule and the m_PW minimum retention
    ratio rho (Sec 3.4): described qualitatively only ("excluded when...
    low-score... too close to stronger candidates"; "up to rho*N keypoints
    ... rho not given numerically"). Implemented here as a score-percentile
    threshold + the stated minimum-distance NMS + a configurable rho.
  - The illumination mask L(x) (Sec 3.2/3.3): the paper assumes a mission
    product ("illumination mask S") that isn't available in this pipeline's
    inputs. `photometric_weighted_structural_maps` uses the normalized
    image intensity itself as a stand-in (near-zero intensity ~= not
    illuminated) - swap in a real per-pixel illumination-validity product
    if you have one.

The ORIGINAL generic implementation below (phase_congruency,
detect_keypoints_dual_channel, compute_descriptors,
match_descriptors_swap_aware, photometric_weighted_pc,
coarse_to_fine_rotation_scale_ncc) is kept as-is and still used by
matching.py for TMC/IIRS, which have no per-pixel photometric-geometry
product for the paper's full method to key off of in the first place.

--- PERF FIX (this revision): the coarse-to-fine search was ~8-25x slower
    than it needed to be ---
`coarse_to_fine_rotation_scale` evaluates ~24 (scale x rotation x
refinement) hypotheses. The old code had two real inefficiencies, not just
"the paper's method is inherently expensive":
  1. `_evaluate_rs_candidate` recomputed the REFERENCE image's full
     structural maps from scratch on every one of those ~24 calls, even
     though the reference side is fixed for the whole search. Now computed
     once in `coarse_to_fine_rotation_scale` and passed through.
  2. The "lightweight" search stage used the exact same n_scales/n_orient
     (4x12=48 filter/FFT passes) as the final full-resolution match - only
     image size and keypoint budget shrank. `run_pwift_stage`/
     `_evaluate_rs_candidate` now take explicit overrides, and the search
     uses `cfg.pwift_search_scales`/`cfg.pwift_search_orientations`
     (see config.py) instead.
See also scale.py's `select_best_scale` fix for a related issue: sensors
with a fixed scale ratio (scale_range lo==hi, e.g. LROC-vs-LROC) used to
still generate `pyramid_levels` (default 4) identical scale "candidates"
and evaluate each one at full cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift


# ----------------------------------------------------------------------
# 1. Multi-scale, multi-orientation phase congruency (log-Gabor filters)
#    - shared by both the old generic path and the new paper-faithful path
# ----------------------------------------------------------------------

def _log_gabor_filter(shape, scale_wavelength, orientation_rad,
                       sigma_onf=0.55, d_theta_sigma=1.2, n_orient=6):
    rows, cols = shape
    half_r, half_c = rows // 2, cols // 2
    # NB: `-rows // 2` floor-divides toward -inf and gives an off-by-one vs
    # `-(rows // 2)` for odd sizes (e.g. rows=627 -> -314 vs -313), which
    # silently produced a filter one pixel larger than the image on odd-sized
    # arrays. Compute the half-size once and negate it explicitly instead.
    yy, xx = np.mgrid[-half_r : rows - half_r, -half_c : cols - half_c]
    yy, xx = yy / rows, xx / cols
    radius = np.sqrt(xx ** 2 + yy ** 2)
    radius[half_r, half_c] = 1.0  # avoid log(0)
    theta = np.arctan2(-yy, xx)

    fo = 1.0 / scale_wavelength
    log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2))
    log_gabor[rows // 2, cols // 2] = 0.0

    d_theta = np.pi / n_orient
    theta_diff = theta - orientation_rad
    theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi
    spread = np.exp(-(theta_diff ** 2) / (2 * (d_theta * d_theta_sigma) ** 2))

    return fftshift(log_gabor * spread)


def phase_congruency(
    img: np.ndarray, n_scales: int = 4, n_orient: int = 6,
    min_wavelength: float = 3.0, mult: float = 2.1, noise_k: float = 2.0,
) -> np.ndarray:
    """Kovesi-style phase congruency map, illumination/contrast invariant by
    construction (depends on local phase alignment, not absolute intensity -
    this is *why* PC-based features are a good base for illumination-robust
    lunar matching). GENERIC path (TMC/IIRS) - see module docstring."""
    rows, cols = img.shape
    IMG = fft2(img)
    pc_total = np.zeros((rows, cols), dtype=np.float64)

    for o in range(n_orient):
        orientation = o * np.pi / n_orient
        sum_an, sum_re, sum_im = (
            np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
        )
        an_arrays = []
        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            filt = _log_gabor_filter((rows, cols), wavelength, orientation, n_orient=n_orient)
            resp = ifft2(IMG * filt)
            an = np.abs(resp)
            an_arrays.append(an)
            sum_an += an
            sum_re += resp.real
            sum_im += resp.imag

        mean_amp = np.mean(an_arrays[0])
        tau = mean_amp / np.sqrt(np.log(4)) if mean_amp > 0 else 1e-6
        total_energy = np.sqrt(sum_re ** 2 + sum_im ** 2)
        noise_est = tau * np.sqrt(np.pi / 2) * noise_k
        energy = np.maximum(total_energy - noise_est, 0.0)
        pc_o = energy / (sum_an + 1e-6)
        pc_total += pc_o

    return (pc_total / n_orient).astype(np.float32)


# ----------------------------------------------------------------------
# 2a. Photometric reliability map - Eq 1-2 (paper-faithful)
# ----------------------------------------------------------------------

def akimov_weight(incidence_deg: np.ndarray, emission_deg: np.ndarray,
                   phase_deg: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Eq 1-2: photometric reliability map w(p) in [0,1].

    w0(p) = D_Aki(...) for valid p, else 0 (Eq 1)
    w(p) = (w0 - min_valid(w0)) / (max_valid(w0) - min_valid(w0) + eps) (Eq 2)

    SWAP POINT: see module docstring - D_Aki's real closed form isn't in the
    paper's extracted text. This stand-in downweights near-terminator/near-
    limb pixels (incidence or emission >= 90 deg is treated as invalid, per
    Eq 1's "invalid pixels -> w0=0") and optionally applies phase-angle
    darkening if a phase map is available."""
    valid = (
        (incidence_deg >= 0.0) & (incidence_deg < 90.0) &
        (emission_deg >= 0.0) & (emission_deg < 90.0) &
        np.isfinite(incidence_deg) & np.isfinite(emission_deg)
    )
    if phase_deg is not None:
        valid = valid & (phase_deg >= 0.0) & (phase_deg <= 180.0) & np.isfinite(phase_deg)

    mu0 = np.cos(np.radians(np.where(valid, incidence_deg, 0.0)))
    mu = np.cos(np.radians(np.where(valid, emission_deg, 0.0)))

    mu0_c = np.clip(mu0, eps, 1.0)
    mu_c = np.clip(mu, eps, 1.0)
    w0 = (2 * mu0_c) / (mu0_c + mu_c)
    if phase_deg is not None:
        w0 = w0 * np.cos(np.radians(np.clip(np.where(valid, phase_deg, 0.0), 0, 180)) / 2.0) ** 2

    w0 = np.where(valid, w0, 0.0)

    if not np.any(valid):
        return np.zeros_like(w0, dtype=np.float32)

    lo = float(w0[valid].min())
    hi = float(w0[valid].max())
    if (hi - lo) < 1e-4:
        w = w0
    else:
        w = (w0 - lo) / (hi - lo + eps)
    w = np.where(valid, np.clip(w, 0.0, 1.0), 0.0)
    return w.astype(np.float32)


def photometric_weighted_pc(
    img: np.ndarray, incidence_deg: Optional[np.ndarray] = None,
    emission_deg: Optional[np.ndarray] = None, n_scales: int = 4, n_orient: int = 6,
) -> np.ndarray:
    """GENERIC single-map illumination-weighted PC - kept for the old
    detect_keypoints_dual_channel/compute_descriptors path (TMC/IIRS)."""
    pc = phase_congruency(img, n_scales=n_scales, n_orient=n_orient)
    if incidence_deg is not None and emission_deg is not None:
        w = akimov_weight(incidence_deg, emission_deg)
        pc = pc * (0.5 + 0.5 * w)  # soft weighting - never fully zero out a region
    return pc


# ----------------------------------------------------------------------
# 2b. Multi-scale photometric-weighted structural maps - Eq 3-8
# ----------------------------------------------------------------------

def compute_valid_mask(W_norm: np.ndarray, L: np.ndarray, w_bg: float, on_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Eq 3-5. Returns (mW, mask): mW is the background-threshold mask alone
    (needed separately for Eq 7's hard zeroing), mask is the union mW | mL
    used to constrain PC computation to reliable-or-geometrically-valid
    regions."""
    mW = W_norm >= w_bg
    mL = L > on_thr
    return mW, (mW | mL)


def soft_weight(W_norm: np.ndarray, w_lo: float, w_hi: float) -> np.ndarray:
    """Eq 6: piecewise-linear soft weight - smooth transition in low-weight
    regions, hard zeroing only applied to the background via mW (Eq 7)."""
    return np.clip((W_norm - w_lo) / max(w_hi - w_lo, 1e-6), 0.0, 1.0)


def photometric_weighted_structural_maps(
    img: np.ndarray,
    incidence_deg: Optional[np.ndarray] = None, emission_deg: Optional[np.ndarray] = None,
    phase_deg: Optional[np.ndarray] = None,
    n_scales: int = 4, n_orient: int = 12,
    gamma_pc: float = 1.0, w_bg: float = 0.15, on_thr: float = 0.05,
    w_soft_lo: float = 0.10, w_soft_hi: float = 0.60,
    min_wavelength: float = 3.0, mult: float = 2.1, noise_k: float = 2.0,
) -> dict:
    """Section 3.3, Eq 3-8: multi-scale photometric-weighted phase
    congruency, producing M_PW (max moment), m_PW (min moment), and MIM
    (dominant-orientation index map).

    If no angle maps are given (e.g. the reference image, which usually has
    no photometric-geometry product of its own), W falls back to all-ones -
    Eq 1-2 with no invalid pixels, i.e. fully reliable everywhere.

    Returns a dict: {"M_PW", "m_PW", "MIM", "w", "w_soft", "mask"}.
    """
    rows, cols = img.shape
    if incidence_deg is not None and emission_deg is not None:
        W_norm = akimov_weight(incidence_deg, emission_deg, phase_deg)
    else:
        W_norm = np.ones_like(img, dtype=np.float32)

    # L(x): illumination-mask stand-in - see module docstring SWAP POINT.
    L = img
    mW, mask = compute_valid_mask(W_norm, L, w_bg, on_thr)
    w_soft = soft_weight(W_norm, w_soft_lo, w_soft_hi)

    IMG = fft2(img)
    covx2 = np.zeros((rows, cols), dtype=np.float64)
    covy2 = np.zeros((rows, cols), dtype=np.float64)
    covxy = np.zeros((rows, cols), dtype=np.float64)
    cs_stack = np.zeros((n_orient, rows, cols), dtype=np.float64)

    for o in range(n_orient):
        orientation = o * np.pi / n_orient
        sum_an, sum_re, sum_im = (
            np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
        )
        an_arrays = []
        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            filt = _log_gabor_filter((rows, cols), wavelength, orientation, n_orient=n_orient)
            resp = ifft2(IMG * filt)
            an = np.abs(resp)
            an_arrays.append(an)
            sum_an += an
            sum_re += resp.real
            sum_im += resp.imag

        mean_amp = np.mean(an_arrays[0])
        tau = mean_amp / np.sqrt(np.log(4)) if mean_amp > 0 else 1e-6
        total_energy = np.sqrt(sum_re ** 2 + sum_im ** 2)
        noise_est = tau * np.sqrt(np.pi / 2) * noise_k
        energy = np.maximum(total_energy - noise_est, 0.0)
        pc_o = energy / (sum_an + 1e-6)

        # Eq 7: photometrically weighted orientation PC map (soft weighting,
        # hard zero only on the background via mW, NOT the full mask - a
        # direct pixel-wise weighting by W itself would uniformly reduce PC
        # on the shadowed side per the paper's discussion in Sec 3.3)
        pc_pw_o = np.where(mW, pc_o * (w_soft ** gamma_pc), 0.0)

        cs_stack[o] = sum_an  # CS_o(x): accumulated orientation magnitude, Eq 8
        covx2 += pc_pw_o * np.cos(orientation) ** 2
        covy2 += pc_pw_o * np.sin(orientation) ** 2
        covxy += pc_pw_o * np.sin(orientation) * np.cos(orientation)

    # Standard phase-congruency covariance-moment decomposition (Kovesi
    # feature-type formulation), applied here to the photometrically
    # weighted PC^PW_o accumulated above, giving the dominant/subordinate
    # structural-energy moments the paper calls M_PW and m_PW.
    denom = np.sqrt(np.clip((covx2 - covy2) ** 2 + 4 * covxy ** 2, 0, None))
    M_PW = (0.5 * (covx2 + covy2 + denom)).astype(np.float32)
    m_PW = (0.5 * (covx2 + covy2 - denom)).astype(np.float32)
    MIM = np.argmax(cs_stack, axis=0).astype(np.int32)  # Eq 8

    return {"M_PW": M_PW, "m_PW": m_PW, "MIM": MIM, "w": W_norm, "w_soft": w_soft, "mask": mask}


@dataclass
class PWIFTMaps:
    """Thin typed wrapper around photometric_weighted_structural_maps()'s
    dict, so matching.py can dispatch on `isinstance(x, PWIFTMaps)` instead
    of duck-typing a dict."""
    M_PW: np.ndarray
    m_PW: np.ndarray
    MIM: np.ndarray
    w: np.ndarray
    w_soft: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_dict(cls, d: dict) -> "PWIFTMaps":
        return cls(**d)


# ----------------------------------------------------------------------
# 3. Dual bright/dark channel keypoint detection
# ----------------------------------------------------------------------

@dataclass
class Keypoint:
    x: float
    y: float
    channel: str        # "bright"/"dark" (generic path) or "M_PW"/"m_PW" (paper path)
    response: float
    orientation: float = 0.0  # radians, filled in by the descriptor/orientation stage
    mim_bin: int = 0          # k_p, Eq 11 - paper path only


def detect_keypoints_dual_channel(
    pc_map: np.ndarray, threshold: float = 0.15, min_distance: int = 8,
    max_keypoints: int = 800,
) -> List[Keypoint]:
    """GENERIC path (TMC/IIRS): local maxima of the phase-congruency map
    (bright channel) and of its complement (dark channel)."""
    keypoints: List[Keypoint] = []
    for channel_name, resp_map in (("bright", pc_map), ("dark", 1.0 - pc_map)):
        mx = ndimage.maximum_filter(resp_map, size=min_distance)
        mask = (resp_map == mx) & (resp_map > threshold)
        ys, xs = np.nonzero(mask)
        vals = resp_map[ys, xs]
        order = np.argsort(-vals)[: max_keypoints // 2]
        for i in order:
            keypoints.append(Keypoint(x=float(xs[i]), y=float(ys[i]),
                                       channel=channel_name, response=float(vals[i])))
    return keypoints


def detect_keypoints_pw(
    M_PW: np.ndarray, m_PW: np.ndarray, w_soft: np.ndarray, mask: np.ndarray,
    min_distance: int = 8, max_keypoints: int = 800, min_retention_ratio: float = 0.3,
    score_percentile: float = 60.0,
) -> List[Keypoint]:
    """Eq 9, Sec 3.4: unified score Sc(x) = Rc(x)*w_soft(x) for c in
    {M_PW, m_PW}. Dual-channel candidate extraction, adaptive quality
    screening (score-percentile threshold; see module docstring - the paper
    gives this qualitatively, not as a formula), a minimum-distance NMS, and
    a minimum retention ratio for the m_PW channel so it isn't crowded out
    by M_PW's typically stronger responses (paper Sec 3.4, penultimate
    paragraph)."""
    candidates = []  # (score, x, y, channel_name)
    for channel_name, resp in (("M_PW", M_PW), ("m_PW", m_PW)):
        score_map = np.where(mask, resp * w_soft, 0.0)
        mx = ndimage.maximum_filter(score_map, size=min_distance)
        is_local_max = (score_map == mx) & (score_map > 0)
        if not np.any(is_local_max):
            continue
        thresh = np.percentile(score_map[is_local_max], score_percentile)
        keep = is_local_max & (score_map >= thresh)
        ys, xs = np.nonzero(keep)
        for x, y in zip(xs, ys):
            candidates.append((float(score_map[y, x]), float(x), float(y), channel_name))

    candidates.sort(key=lambda c: c[0], reverse=True)

    def _select(pool, budget, already_taken_xy):
        kept = []
        for score, x, y, ch in pool:
            if len(kept) >= budget:
                break
            too_close = False
            for kx, ky in already_taken_xy:
                if (x - kx) ** 2 + (y - ky) ** 2 < min_distance ** 2:
                    too_close = True
                    break
            if too_close:
                continue
            for _, kx, ky, _ in kept:
                if (x - kx) ** 2 + (y - ky) ** 2 < min_distance ** 2:
                    too_close = True
                    break
            if not too_close:
                kept.append((score, x, y, ch))
        return kept

    m_pw_pool = [c for c in candidates if c[3] == "m_PW"]
    n_from_m_pw = int(min_retention_ratio * max_keypoints)
    kept = _select(m_pw_pool, n_from_m_pw, [])
    taken_xy = [(k[1], k[2]) for k in kept]
    remaining_budget = max(0, max_keypoints - len(kept))
    kept += _select(candidates, remaining_budget, taken_xy)

    return [Keypoint(x=x, y=y, channel=ch, response=score) for score, x, y, ch in kept]


# ----------------------------------------------------------------------
# 4a. GENERIC rotation-normalized bi-channel descriptor (TMC/IIRS path)
# ----------------------------------------------------------------------

@dataclass
class BiChannelDescriptor:
    keypoint: Keypoint
    vector: np.ndarray  # unit-normalized histogram, fixed length


def _dominant_orientation(patch: np.ndarray) -> float:
    gy, gx = np.gradient(patch)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    hist, edges = np.histogram(ang, bins=36, range=(-np.pi, np.pi), weights=mag)
    return edges[int(np.argmax(hist))]


def compute_descriptors(
    img: np.ndarray, pc_map: np.ndarray, keypoints: List[Keypoint],
    patch_size: int = 32, n_bins_ori: int = 8, n_cells: int = 4,
) -> List[BiChannelDescriptor]:
    """GENERIC path (TMC/IIRS) - unchanged from before."""
    half = patch_size // 2
    h, w = img.shape
    descriptors: List[BiChannelDescriptor] = []

    for kp in keypoints:
        xi, yi = int(round(kp.x)), int(round(kp.y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        img_patch = img[yi - half : yi + half, xi - half : xi + half]
        pc_patch = pc_map[yi - half : yi + half, xi - half : xi + half]

        theta = _dominant_orientation(img_patch)
        kp.orientation = theta
        img_patch = ndimage.rotate(img_patch, -np.degrees(theta), reshape=False, order=1)
        pc_patch = ndimage.rotate(pc_patch, -np.degrees(theta), reshape=False, order=1)

        cell = patch_size // n_cells
        feat = []
        for channel_patch in (img_patch, pc_patch):
            gy, gx = np.gradient(channel_patch)
            mag = np.hypot(gx, gy)
            ang = np.arctan2(gy, gx)
            for cy in range(n_cells):
                for cx in range(n_cells):
                    sub_ang = ang[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell]
                    sub_mag = mag[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell]
                    hist, _ = np.histogram(sub_ang, bins=n_bins_ori,
                                            range=(-np.pi, np.pi), weights=sub_mag)
                    feat.append(hist)
        vec = np.concatenate(feat).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        descriptors.append(BiChannelDescriptor(keypoint=kp, vector=vec))

    return descriptors


@dataclass
class Match:
    src_idx: int
    dst_idx: int
    distance: float


def match_descriptors_swap_aware(
    desc_a: List[BiChannelDescriptor], desc_b: List[BiChannelDescriptor],
    ratio_test: float = 0.85,
) -> List[Match]:
    """GENERIC path (TMC/IIRS) - unchanged from before. NOTE: despite the
    name, this is NOT the paper's swap-aware distance (Eq 19-21) - see
    `swap_aware_match` below for that. Kept under its original name so
    nothing calling it from the TMC/IIRS branch breaks."""
    if not desc_a or not desc_b:
        return []

    from scipy.spatial.distance import cdist

    A = np.stack([d.vector for d in desc_a])
    B = np.stack([d.vector for d in desc_b])
    dists = cdist(A, B, metric="euclidean")

    matches: List[Match] = []
    for i in range(dists.shape[0]):
        order = np.argsort(dists[i])
        if len(order) < 2:
            continue
        best, second = order[0], order[1]
        if dists[i, best] < ratio_test * dists[i, second]:
            back_order = np.argsort(dists[:, best])
            if back_order[0] == i:
                matches.append(Match(src_idx=i, dst_idx=int(best), distance=float(dists[i, best])))
    return matches


# ----------------------------------------------------------------------
# 4b. Paper-faithful rotation normalization + bi-channel descriptor
#     - Eq 10-18
# ----------------------------------------------------------------------

def dominant_orientation_pw(kp: Keypoint, MIM: np.ndarray, M_PW: np.ndarray, w_soft: np.ndarray,
                             K: int = 12, radius: int = 16, sigma: Optional[float] = None) -> None:
    """Eq 10-11: MIM-based orientation histogram in a neighborhood of `kp`,
    voting-weighted by structural strength (M_PW), photometric soft weight,
    and a Gaussian spatial weight. Sets kp.orientation (radians, bin center
    of k_p) and kp.mim_bin (k_p) IN PLACE."""
    sigma = sigma or radius / 2.0
    h, w = MIM.shape
    xi, yi = int(round(kp.x)), int(round(kp.y))
    y0, y1 = max(0, yi - radius), min(h, yi + radius)
    x0, x1 = max(0, xi - radius), min(w, xi + radius)
    if y1 <= y0 or x1 <= x0:
        kp.orientation, kp.mim_bin = 0.0, 0
        return

    mim_patch = MIM[y0:y1, x0:x1]
    mpw_patch = M_PW[y0:y1, x0:x1]
    wsoft_patch = w_soft[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist2 = (xs - xi) ** 2 + (ys - yi) ** 2
    gauss = np.exp(-dist2 / (2 * sigma ** 2))

    weight = mpw_patch * wsoft_patch * gauss
    hist = np.bincount(mim_patch.ravel().astype(int), weights=weight.ravel(), minlength=K)[:K]
    k_p = int(np.argmax(hist))
    kp.mim_bin = k_p
    kp.orientation = k_p * np.pi / K


def compute_bichannel_descriptors(
    keypoints: List[Keypoint], MIM: np.ndarray, M_PW: np.ndarray, w_soft: np.ndarray, W: np.ndarray,
    patch_size: int = 32, no: int = 4, nbins: int = 12, t: float = 0.2, sigma: Optional[float] = None,
) -> List[BiChannelDescriptor]:
    """Eq 12-18: rotation-normalized bright-dark bi-channel descriptor.

    Rotation is applied to the *offset coordinates* (Eq 12) and the MIM bin
    (Eq 13), not to the patch pixels themselves - MIM is a categorical
    orientation-index map, and interpolating it (as naive patch rotation
    would) is meaningless.

    Returned vector layout: (no*no cells, row-major) x (2 channels: bright
    then dark) x (nbins), flattened - `swap_aware_match` depends on this
    exact layout to split bright/dark back out."""
    half = patch_size // 2
    sigma = sigma or patch_size / 4.0
    h, w = MIM.shape
    cell = patch_size / no
    descriptors: List[BiChannelDescriptor] = []

    ys, xs = np.mgrid[-half:half, -half:half]
    dx, dy = xs.astype(np.float32), ys.astype(np.float32)
    r2 = dx ** 2 + dy ** 2

    for kp in keypoints:
        xi, yi = int(round(kp.x)), int(round(kp.y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue

        mim_patch = MIM[yi - half:yi + half, xi - half:xi + half]
        mpw_patch = M_PW[yi - half:yi + half, xi - half:xi + half]
        wsoft_patch = w_soft[yi - half:yi + half, xi - half:xi + half]
        wcol_patch = W[yi - half:yi + half, xi - half:xi + half]

        theta_p = kp.orientation
        # Eq 12: rotate the offset coordinates into the canonical frame
        xr = np.cos(theta_p) * dx - np.sin(theta_p) * dy
        yr = np.sin(theta_p) * dx + np.cos(theta_p) * dy
        # Eq 13: shift MIM into a relative orientation bin
        b_norm = (mim_patch - kp.mim_bin + nbins) % nbins

        cx = np.clip(((xr + half) // cell).astype(int), 0, no - 1)
        cy = np.clip(((yr + half) // cell).astype(int), 0, no - 1)
        cell_idx = cy * no + cx

        # Eq 14: voting weight
        weight = mpw_patch * wsoft_patch * np.exp(-r2 / (2 * sigma ** 2))
        bright = wcol_patch >= t

        desc = np.zeros((no * no, 2, nbins), dtype=np.float32)
        flat_cell = cell_idx.ravel()
        flat_bin = b_norm.ravel().astype(int)
        flat_w = weight.ravel()
        flat_bright = bright.ravel()

        # Eq 16-17: bright/dark channel histogram accumulation per cell
        np.add.at(desc, (flat_cell[flat_bright], 0, flat_bin[flat_bright]), flat_w[flat_bright])
        np.add.at(desc, (flat_cell[~flat_bright], 1, flat_bin[~flat_bright]), flat_w[~flat_bright])

        vec = desc.reshape(-1).astype(np.float32)
        # L2 normalize -> clip -> L2 normalize again (end of Sec 3.5)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        vec = np.clip(vec, 0, 0.2)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        descriptors.append(BiChannelDescriptor(keypoint=kp, vector=vec))

    return descriptors


# ----------------------------------------------------------------------
# 4c. Contextual (surrounding-terrain) descriptor - NOT in the paper.
#     Addresses the "two craters with near-identical local appearance but
#     different surroundings" ambiguity by encoding what's AROUND the
#     keypoint's own patch, not just inside it.
# ----------------------------------------------------------------------

def compute_context_descriptor(
    keypoints: List[Keypoint], M_PW: np.ndarray, mask: np.ndarray,
    n_rings: int = 3, n_sectors: int = 8, ring_spacing_px: int = 24,
) -> np.ndarray:
    """For each keypoint, samples mean M_PW structural energy in a set of
    concentric rings x angular sectors extending OUTWARD from the
    keypoint - i.e. "is there a ridge to the upper-left, a second crater
    further out, open flat terrain" - rather than what the local
    descriptor patch already covers. Sampled relative to the keypoint's
    own dominant orientation (same rotation-normalization convention as
    Eq 12), so it stays comparable across images at different rotations.

    Returns an (N, n_rings*n_sectors) float32 array, L2-normalized per row.
    Independent of `compute_bichannel_descriptors` - meant to be combined
    with its output at matching time (see `swap_aware_match_with_context`),
    not concatenated into the same 384-dim vector, since it has a
    different physical meaning (surroundings, not local appearance) and
    mixing it into one L2 norm would let one dominate the other
    unpredictably depending on relative magnitude."""
    h, w = M_PW.shape
    n_cells = n_rings * n_sectors
    out = np.zeros((len(keypoints), n_cells), dtype=np.float32)
    max_radius = n_rings * ring_spacing_px
    sector_width = 2 * np.pi / n_sectors

    for i, kp in enumerate(keypoints):
        xi, yi = int(round(kp.x)), int(round(kp.y))
        y0, y1 = max(0, yi - max_radius), min(h, yi + max_radius)
        x0, x1 = max(0, xi - max_radius), min(w, xi + max_radius)
        if y1 <= y0 or x1 <= x0:
            continue

        patch = M_PW[y0:y1, x0:x1]
        pm = mask[y0:y1, x0:x1]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = (xs - xi).astype(np.float32)
        dy = (ys - yi).astype(np.float32)

        # rotate into the keypoint's own canonical frame - same convention
        # as Eq 12's xr/yr, so "upper-left" means the same thing relative
        # to local structure in both images being compared
        theta = kp.orientation
        xr = np.cos(theta) * dx - np.sin(theta) * dy
        yr = np.sin(theta) * dx + np.cos(theta) * dy

        r = np.sqrt(xr ** 2 + yr ** 2)
        ang = (np.arctan2(yr, xr) + 2 * np.pi) % (2 * np.pi)
        ring_idx = np.clip((r // ring_spacing_px).astype(int), 0, n_rings - 1)
        sector_idx = np.clip((ang // sector_width).astype(int), 0, n_sectors - 1)
        cell_idx = (ring_idx * n_sectors + sector_idx)

        valid = pm & (r > 0) & (r < max_radius)
        vec = np.zeros(n_cells, dtype=np.float32)
        counts = np.zeros(n_cells, dtype=np.float32)
        np.add.at(vec, cell_idx[valid], patch[valid])
        np.add.at(counts, cell_idx[valid], 1.0)
        vec = vec / np.maximum(counts, 1.0)  # mean energy per cell

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        out[i] = vec

    return out


def swap_aware_match_with_context(
    desc_a: List[BiChannelDescriptor], desc_b: List[BiChannelDescriptor],
    context_a: np.ndarray, context_b: np.ndarray,
    no: int = 4, nbins: int = 12, ratio_test: float = 0.85, context_weight: float = 0.3,
) -> List[Match]:
    """Same swap-aware distance as `swap_aware_match` (Eq 19-21), plus a
    weighted contextual-distance term. Both terms come from L2-normalized
    vectors, so both are on a comparable [0, ~1.4] scale and a linear blend
    is meaningful without extra rescaling. `context_weight=0` reduces
    exactly to `swap_aware_match`; start low (0.2-0.3) and increase only if
    it measurably improves inlier ratio - a large weight will start
    penalizing genuinely correct matches near two similar-looking features
    just because their surroundings were sampled slightly differently."""
    if not desc_a or not desc_b:
        return []
    from scipy.spatial.distance import cdist

    n_cells = no * no
    A = np.stack([d.vector for d in desc_a]).reshape(len(desc_a), n_cells, 2, nbins)
    B = np.stack([d.vector for d in desc_b]).reshape(len(desc_b), n_cells, 2, nbins)
    A_bright = A[:, :, 0, :].reshape(len(desc_a), -1)
    A_dark = A[:, :, 1, :].reshape(len(desc_a), -1)
    B_bright = B[:, :, 0, :].reshape(len(desc_b), -1)
    B_dark = B[:, :, 1, :].reshape(len(desc_b), -1)

    d_bb = cdist(A_bright, B_bright, metric="sqeuclidean")
    d_dd = cdist(A_dark, B_dark, metric="sqeuclidean")
    d_bd = cdist(A_bright, B_dark, metric="sqeuclidean")
    d_db = cdist(A_dark, B_bright, metric="sqeuclidean")
    d_ori = np.sqrt(d_bb + d_dd)
    d_ex = np.sqrt(d_bd + d_db)
    appearance_dist = np.minimum(d_ori, d_ex)

    context_dist = cdist(context_a, context_b, metric="euclidean")
    dists = (1.0 - context_weight) * appearance_dist + context_weight * context_dist

    matches: List[Match] = []
    for i in range(dists.shape[0]):
        order = np.argsort(dists[i])
        if len(order) < 2:
            continue
        best, second = order[0], order[1]
        if dists[i, best] < ratio_test * dists[i, second]:
            back_order = np.argsort(dists[:, best])
            if back_order[0] == i:
                matches.append(Match(src_idx=i, dst_idx=int(best), distance=float(dists[i, best])))
    return matches


def swap_aware_match(desc_a: List[BiChannelDescriptor], desc_b: List[BiChannelDescriptor],
                      no: int = 4, nbins: int = 12, ratio_test: float = 0.85) -> List[Match]:
    """Eq 19-21: swap-aware descriptor distance = min(d_ori, d_exchange),
    mutual nearest-neighbor + Lowe ratio test. Splits each descriptor
    vector back into its bright/dark halves (see
    `compute_bichannel_descriptors`'s layout)."""
    if not desc_a or not desc_b:
        return []
    from scipy.spatial.distance import cdist

    n_cells = no * no
    A = np.stack([d.vector for d in desc_a]).reshape(len(desc_a), n_cells, 2, nbins)
    B = np.stack([d.vector for d in desc_b]).reshape(len(desc_b), n_cells, 2, nbins)
    A_bright = A[:, :, 0, :].reshape(len(desc_a), -1)
    A_dark = A[:, :, 1, :].reshape(len(desc_a), -1)
    B_bright = B[:, :, 0, :].reshape(len(desc_b), -1)
    B_dark = B[:, :, 1, :].reshape(len(desc_b), -1)

    d_bb = cdist(A_bright, B_bright, metric="sqeuclidean")
    d_dd = cdist(A_dark, B_dark, metric="sqeuclidean")
    d_bd = cdist(A_bright, B_dark, metric="sqeuclidean")
    d_db = cdist(A_dark, B_bright, metric="sqeuclidean")

    d_ori = np.sqrt(d_bb + d_dd)  # Eq 20
    d_ex = np.sqrt(d_bd + d_db)   # Eq 21
    dists = np.minimum(d_ori, d_ex)  # Eq 19

    matches: List[Match] = []
    for i in range(dists.shape[0]):
        order = np.argsort(dists[i])
        if len(order) < 2:
            continue
        best, second = order[0], order[1]
        if dists[i, best] < ratio_test * dists[i, second]:
            back_order = np.argsort(dists[:, best])
            if back_order[0] == i:
                matches.append(Match(src_idx=i, dst_idx=int(best), distance=float(dists[i, best])))
    return matches


# ----------------------------------------------------------------------
# 5. Single-call PWIFT stage (detect -> orient -> describe), reused by both
#    the full matching stage (matching.py) and the lightweight evaluation
#    inside the coarse-to-fine search below, exactly as the paper describes
#    ("full matching stage uses the complete... construction described in
#    Sections 3.2-3.5"; the lightweight stage is the same thing "at a
#    reduced image resolution with a limited number of keypoints").
# ----------------------------------------------------------------------

def run_pwift_stage(img: np.ndarray, incidence_deg=None, emission_deg=None, phase_deg=None,
                     cfg=None, max_keypoints: Optional[int] = None,
                     n_scales_override: Optional[int] = None, n_orient_override: Optional[int] = None):
    """PERF: `n_scales_override`/`n_orient_override` let the coarse-to-fine
    search (below) run the phase-congruency filter bank at reduced cost
    (fewer scales/orientations) than the final full-resolution matching
    pass. Previously this always used `cfg.pwift_scales`/`cfg.pwift_orientations`
    (4 x 12 = 48 log-Gabor filter/FFT passes) even during the ~24-hypothesis
    search, where only image resolution and keypoint budget were actually
    reduced - the single most expensive part of the computation wasn't."""
    max_keypoints = max_keypoints if max_keypoints is not None else cfg.pwift_max_keypoints
    n_scales = n_scales_override if n_scales_override is not None else cfg.pwift_scales
    n_orient = n_orient_override if n_orient_override is not None else cfg.pwift_orientations
    maps = photometric_weighted_structural_maps(
        img, incidence_deg, emission_deg, phase_deg,
        n_scales=n_scales, n_orient=n_orient,
        gamma_pc=cfg.pwift_gamma_pc, w_bg=cfg.pwift_bg_threshold, on_thr=cfg.pwift_illum_threshold,
        w_soft_lo=cfg.pwift_soft_weight_lo, w_soft_hi=cfg.pwift_soft_weight_hi,
    )
    keypoints = detect_keypoints_pw(
        maps["M_PW"], maps["m_PW"], maps["w_soft"], maps["mask"],
        min_distance=cfg.pwift_min_keypoint_distance, max_keypoints=max_keypoints,
        min_retention_ratio=cfg.pwift_min_retention_ratio,
        score_percentile=cfg.pwift_keypoint_score_percentile,
    )
    for kp in keypoints:
        dominant_orientation_pw(kp, maps["MIM"], maps["M_PW"], maps["w_soft"], K=n_orient)
    descriptors = compute_bichannel_descriptors(
        keypoints, maps["MIM"], maps["M_PW"], maps["w_soft"], maps["w"],
        patch_size=cfg.pwift_descriptor_patch, no=cfg.pwift_descriptor_cells,
        nbins=n_orient, t=cfg.pwift_bright_dark_threshold,
    )
    return maps, keypoints, descriptors


def match_pwift(desc_a: List[BiChannelDescriptor], desc_b: List[BiChannelDescriptor], cfg,
                 nbins_override: Optional[int] = None) -> List[Match]:
    nbins = nbins_override if nbins_override is not None else cfg.pwift_orientations
    return swap_aware_match(desc_a, desc_b, no=cfg.pwift_descriptor_cells,
                             nbins=nbins, ratio_test=cfg.pwift_ratio_test)


# ----------------------------------------------------------------------
# 6a. GENERIC coarse-to-fine rotation-scale search (NCC-based, TMC/IIRS)
# ----------------------------------------------------------------------

def coarse_to_fine_rotation_scale_ncc(
    img_a: np.ndarray, img_b: np.ndarray,
    scale_candidates: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    rotation_candidates_deg: Tuple[float, ...] = tuple(range(0, 360, 30)),
    coarse_size: int = 256,
) -> Tuple[float, float]:
    """GENERIC path - cheap NCC-on-gradient search, not the paper's method.
    Kept under this name for callers that want it explicitly (e.g. as a
    fallback when no PWIFTMaps/angle data is available for the lightweight
    lexicographic search below to key off of)."""
    from PIL import Image as _PILImage

    def resize(a, size):
        return np.array(
            _PILImage.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((size, size))
        ).astype(np.float32) / 255.0

    def edge_proxy(a):
        gy, gx = np.gradient(a)
        return np.hypot(gx, gy)

    small_a = edge_proxy(resize(img_a, coarse_size))
    best_score, best_scale, best_rot = -np.inf, 1.0, 0.0

    for scale in scale_candidates:
        rescaled_b_full = ndimage.zoom(img_b, scale, order=1)
        for rot in rotation_candidates_deg:
            rotated = ndimage.rotate(rescaled_b_full, rot, reshape=False, order=1)
            if rotated.shape[0] < 8 or rotated.shape[1] < 8:
                continue
            small_b = edge_proxy(resize(rotated, coarse_size))
            a_n = (small_a - small_a.mean()) / (small_a.std() + 1e-6)
            b_n = (small_b - small_b.mean()) / (small_b.std() + 1e-6)
            score = float(np.mean(a_n * b_n))
            if score > best_score:
                best_score, best_scale, best_rot = score, scale, rot

    return best_scale, best_rot


# ----------------------------------------------------------------------
# 6b. Paper-faithful coarse-to-fine rotation-scale search - Eq 22-25
# ----------------------------------------------------------------------

def _lexicographic_key(n_inliers: int, inlier_ratio: float, rmse: float, n_matches: int) -> tuple:
    """Eq 25's ranking criterion Gamma(C): inliers desc, then inlier ratio
    desc, then reprojection RMSE asc, then raw match count desc. Encoded as
    a tuple to sort descending on (so RMSE is negated, and NaN RMSE - no
    inliers to compute it from - sorts last)."""
    rmse_key = -rmse if np.isfinite(rmse) else -1e9
    return (n_inliers, inlier_ratio, rmse_key, n_matches)


def _evaluate_rs_candidate(src_img: np.ndarray, desc_r: List[BiChannelDescriptor], cfg,
                            max_keypoints: int, n_scales_search: int, n_orient_search: int) -> tuple:
    """Runs the lightweight PWIFT stage on the (transformed) SOURCE side
    only and matches against the caller's already-computed reference
    descriptors, returning the lexicographic_key (Eq 25).

    PERF FIX: this used to also recompute `run_pwift_stage` on the
    reference image every single call - but the reference side (`small_ref`
    in `coarse_to_fine_rotation_scale`) never changes across hypotheses, so
    that was ~24 redundant, identical, full phase-congruency computations
    of the exact same data. The caller now computes `desc_r` ONCE and
    passes it in. `n_scales_search`/`n_orient_search` let the search use a
    cheaper filter bank than the final full-resolution match (see
    `run_pwift_stage`'s PERF note) - consistent with the paper's own
    framing of this as a "lightweight" stage distinct from the "full
    matching stage" that follows once C* is chosen."""
    from .metrics import reprojection_rmse

    _, _, desc_s = run_pwift_stage(
        src_img, cfg=cfg, max_keypoints=max_keypoints,
        n_scales_override=n_scales_search, n_orient_override=n_orient_search)
    matches = match_pwift(desc_s, desc_r, cfg, nbins_override=n_orient_search)
    if len(matches) < 4:
        return _lexicographic_key(0, 0.0, float("nan"), len(matches))

    pts_a = np.array([[desc_s[m.src_idx].keypoint.x, desc_s[m.src_idx].keypoint.y] for m in matches], np.float32)
    pts_b = np.array([[desc_r[m.dst_idx].keypoint.x, desc_r[m.dst_idx].keypoint.y] for m in matches], np.float32)
    H, mask = fsc_homography(pts_a, pts_b, cfg.ransac_reproj_threshold_px, cfg.ransac_max_iters, cfg.ransac_confidence)
    n_inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = n_inliers / len(matches) if matches else 0.0
    rmse = reprojection_rmse(pts_a[mask], pts_b[mask], H) if n_inliers else float("nan")
    return _lexicographic_key(n_inliers, inlier_ratio, rmse, len(matches))


def coarse_to_fine_rotation_scale(
    src_img: np.ndarray, ref_img: np.ndarray, cfg,
    scale_candidates: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, float]:
    """Eq 22-25 (Sec 3.6): coarse-to-fine rotation-scale hypothesis search.

    1) Evaluate every scale candidate in Srs (Eq 22) with a lightweight
       PWIFT match; keep the top `cfg.pwift_topk_scale` by lexicographic
       ranking (Eq 25).
    2) For each retained scale, coarse-search rotation over Theta_c
       (Eq 23); pool all evaluated (scale, rotation) pairs and keep the top
       `cfg.pwift_topk_rotation` overall.
    3) Locally refine each kept coarse angle by Delta_Theta (Eq 24).
    4) Return the best (scale, rotation) = C* = argmax_C Gamma(C) (Eq 25)
       across everything evaluated.

    `scale_candidates` overrides `cfg.pwift_scale_candidates` (used by
    `scale.select_best_scale` to substitute a sensor-specific range while
    keeping the paper's coarse-to-fine *procedure*)."""
    from PIL import Image as _PILImage

    scale_candidates = scale_candidates or cfg.pwift_scale_candidates
    coarse_size = cfg.pwift_search_coarse_size
    max_kp = cfg.pwift_search_max_keypoints
    # PERF: use a cheaper filter bank for the search-only stages than the
    # final full-resolution match (falls back to the full cfg values if
    # these fields aren't present, so older PipelineConfig instances still
    # work - just without the speedup).
    n_scales_search = getattr(cfg, "pwift_search_scales", cfg.pwift_scales)
    n_orient_search = getattr(cfg, "pwift_search_orientations", cfg.pwift_orientations)

    def resize(a, size):
        return np.array(
            _PILImage.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((size, size))
        ).astype(np.float32) / 255.0

    small_ref = resize(ref_img, coarse_size)
    # PERF FIX: compute the reference side's descriptors ONCE, up front.
    # `small_ref` is fixed for the entire search - every hypothesis below
    # only transforms the SOURCE image - so recomputing it per-hypothesis
    # (the old behavior, buried inside `_evaluate_rs_candidate`) was pure
    # duplicated work: roughly half of the search's total FFT cost was
    # spent recomputing identical reference data.
    _, _, desc_ref = run_pwift_stage(
        small_ref, cfg=cfg, max_keypoints=max_kp,
        n_scales_override=n_scales_search, n_orient_override=n_orient_search)

    # ---- stage 1: scale search (Eq 22) ----
    scored_scales = []
    for s in scale_candidates:
        zoomed = ndimage.zoom(src_img, s, order=1)
        if zoomed.size == 0 or min(zoomed.shape) < 8:
            continue
        small_src = resize(zoomed, coarse_size)
        key = _evaluate_rs_candidate(small_src, desc_ref, cfg, max_kp, n_scales_search, n_orient_search)
        scored_scales.append((key, s))
    if not scored_scales:
        return 1.0, 0.0
    scored_scales.sort(key=lambda t: t[0], reverse=True)
    top_scales = [s for _, s in scored_scales[: cfg.pwift_topk_scale]]

    # ---- stage 2: coarse rotation search (Eq 23) ----
    rot_scored = []
    for s in top_scales:
        zoomed = ndimage.zoom(src_img, s, order=1)
        for rot in cfg.pwift_coarse_rotation_candidates_deg:
            rotated = ndimage.rotate(zoomed, rot, reshape=False, order=1)
            if min(rotated.shape) < 8:
                continue
            small_src = resize(rotated, coarse_size)
            key = _evaluate_rs_candidate(small_src, desc_ref, cfg, max_kp, n_scales_search, n_orient_search)
            rot_scored.append((key, s, rot))
    if not rot_scored:
        return top_scales[0], 0.0
    rot_scored.sort(key=lambda t: t[0], reverse=True)
    top_coarse = rot_scored[: cfg.pwift_topk_rotation]

    # ---- stage 3: local rotation refinement (Eq 24) ----
    refine_scored = list(top_coarse)
    for key, s, rot in top_coarse:
        zoomed = ndimage.zoom(src_img, s, order=1)
        for d in cfg.pwift_rotation_refine_offsets_deg:
            if d == 0.0:
                continue  # already evaluated as the coarse candidate itself
            rr = rot + d
            rotated = ndimage.rotate(zoomed, rr, reshape=False, order=1)
            if min(rotated.shape) < 8:
                continue
            small_src = resize(rotated, coarse_size)
            k2 = _evaluate_rs_candidate(small_src, desc_ref, cfg, max_kp, n_scales_search, n_orient_search)
            refine_scored.append((k2, s, rr))

    # ---- stage 4: C* = argmax Gamma(C) (Eq 25) ----
    refine_scored.sort(key=lambda t: t[0], reverse=True)
    _, best_scale, best_rot = refine_scored[0]
    return best_scale, best_rot


# ----------------------------------------------------------------------
# 7. FSC (fast sample consensus) + homography reprojection cleanup
#    - Eq 26-27
# ----------------------------------------------------------------------

def fsc_homography(
    pts_a: np.ndarray, pts_b: np.ndarray,
    reproj_threshold: float = 3.0, max_iters: int = 2000, confidence: float = 0.999,
):
    """Homography estimation via MAGSAC++ (cv2.USAC_MAGSAC) with continuous
    residual density estimation. Requires OpenCV."""
    import cv2

    if len(pts_a) < 4 or len(pts_b) < 4:
        return None, np.zeros(len(pts_a), dtype=bool)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(
        pts_a.astype(np.float32),
        pts_b.astype(np.float32),
        method=method,
        ransacReprojThreshold=reproj_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts_a), dtype=bool)
    return H, mask


def reprojection_cleanup(pts_src: np.ndarray, pts_dst: np.ndarray, H: Optional[np.ndarray],
                          tau_e: float = 5.0) -> np.ndarray:
    """Eq 26-27: explicit homography-based reprojection cleanup, named and
    parameterized (tau_e) exactly as in the paper - applied AFTER FSC as a
    distinct final step, on top of (not instead of) the FSC inlier mask.
    Returns a boolean mask the same length as pts_src."""
    if H is None or len(pts_src) == 0:
        return np.zeros(len(pts_src), dtype=bool)
    ones = np.ones((len(pts_src), 1), dtype=np.float32)
    homog = np.hstack([pts_src.astype(np.float32), ones])
    proj = (H @ homog.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
    errors = np.linalg.norm(proj - pts_dst, axis=1)  # e_est,i, Eq 26
    return errors <= tau_e  # Eq 27