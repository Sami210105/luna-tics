"""
Stage 2 - Illumination variation handling, branched by sensor
================================================================
- OHRC / LROC : paper-faithful PWIFT photometric-weighted structural
                representation (pwift.photometric_weighted_structural_maps,
                Sec 3.2-3.3 of PWIFT.pdf) -> returns a PWIFTMaps bundle
                (M_PW/m_PW/MIM/w/w_soft/mask), not a single PC array.
- TMC         : histogram matching + shadow normalization + log transform
- IIRS        : CLAHE + inversion + dilation fallback (per architecture notes)

matching.py dispatches on the return type of apply_illumination_correction:
a PWIFTMaps -> the full paper-faithful PWIFT matcher (matching.py's new
run_pwift_matching_pw); a plain ndarray -> the original generic PWIFT
matcher (matching.run_pwift_matching), since TMC/IIRS have no per-pixel
photometric-geometry product for the paper's method to key off of.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import ndimage

from .config import SensorConfig
from .pwift import photometric_weighted_structural_maps, PWIFTMaps


def correct_ohrc_lroc(
    img: np.ndarray,
    incidence_deg: Optional[np.ndarray] = None,
    emission_deg: Optional[np.ndarray] = None,
    phase_deg: Optional[np.ndarray] = None,
    n_scales: int = 4, n_orient: int = 12,
    cfg=None,
) -> PWIFTMaps:
    """PWIFT branch (Sec 3.2-3.3). Returns the full photometric-weighted
    structural-map bundle (M_PW/m_PW/MIM/w/w_soft/mask), which is what the
    paper's keypoint detection, descriptor construction, and orientation
    normalization stages all consume - NOT a single flattened PC map (that
    was the previous, non-paper-faithful behaviour)."""
    kwargs = {}
    if cfg is not None:
        kwargs.update(
            gamma_pc=cfg.pwift_gamma_pc, w_bg=cfg.pwift_bg_threshold, on_thr=cfg.pwift_illum_threshold,
            w_soft_lo=cfg.pwift_soft_weight_lo, w_soft_hi=cfg.pwift_soft_weight_hi,
        )
    maps = photometric_weighted_structural_maps(
        img, incidence_deg=incidence_deg, emission_deg=emission_deg, phase_deg=phase_deg,
        n_scales=n_scales, n_orient=n_orient, **kwargs,
    )
    return PWIFTMaps.from_dict(maps)


def _match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src_vals, src_idx, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_vals, ref_counts = np.unique(reference.ravel(), return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    interp_vals = np.interp(src_cdf, ref_cdf, ref_vals)
    return interp_vals[src_idx].reshape(source.shape)


def correct_tmc(
    img: np.ndarray, reference: Optional[np.ndarray] = None,
    shadow_percentile: float = 5.0, log_eps: float = 1e-3,
) -> np.ndarray:
    """TMC branch: (1) shadow normalization - lift very dark (shadowed)
    pixels toward the local non-shadow mean so shadow boundaries don't
    dominate feature detection, (2) log transform to compress the wide
    dynamic range typical of TMC-2 wide-swath imagery, (3) optional histogram
    matching against the reference image so both source and reference sit on
    a comparable intensity scale before matching."""
    shadow_thresh = np.percentile(img, shadow_percentile)
    non_shadow_mean = img[img > shadow_thresh].mean() if np.any(img > shadow_thresh) else img.mean()
    shadow_mask = img <= shadow_thresh
    corrected = img.copy()
    corrected[shadow_mask] = 0.5 * img[shadow_mask] + 0.5 * non_shadow_mean

    log_img = np.log1p(corrected / log_eps) / np.log1p(1.0 / log_eps)
    log_img = np.clip(log_img, 0.0, 1.0)

    if reference is not None:
        log_img = _match_histogram(log_img, reference)
        log_img = np.clip(log_img, 0.0, 1.0)

    return log_img.astype(np.float32)


def correct_iirs(
    img: np.ndarray, clip_limit: float = 0.02, tile_grid: int = 8,
    invert: bool = False, dilate_iters: int = 1,
) -> np.ndarray:
    """IIRS fallback branch: CLAHE (contrast-limited adaptive histogram
    equalization) for the very low native contrast typical of IIRS imagery,
    optional inversion if the surface-vs-background convention differs from
    the reference, then a light grayscale dilation to bridge small gaps left
    by CLAHE noise amplification."""
    try:
        import cv2
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit * 100, tileGridSize=(tile_grid, tile_grid))
        eq = clahe.apply(img_u8).astype(np.float32) / 255.0
    except ImportError:
        eq = _simple_tiled_equalize(img, tile_grid)

    if invert:
        eq = 1.0 - eq

    if dilate_iters > 0:
        eq = ndimage.grey_dilation(eq, size=(3, 3))
        for _ in range(dilate_iters - 1):
            eq = ndimage.grey_dilation(eq, size=(3, 3))

    return np.clip(eq, 0.0, 1.0).astype(np.float32)


def _simple_tiled_equalize(img: np.ndarray, tile_grid: int) -> np.ndarray:
    h, w = img.shape
    th, tw = h // tile_grid, w // tile_grid
    out = np.zeros_like(img)
    for i in range(tile_grid):
        for j in range(tile_grid):
            y0, y1 = i * th, (i + 1) * th if i < tile_grid - 1 else h
            x0, x1 = j * tw, (j + 1) * tw if j < tile_grid - 1 else w
            tile = img[y0:y1, x0:x1]
            lo, hi = tile.min(), tile.max()
            out[y0:y1, x0:x1] = (tile - lo) / max(hi - lo, 1e-6)
    return out


def apply_illumination_correction(
    img: np.ndarray, sensor: SensorConfig,
    incidence_deg: Optional[np.ndarray] = None,
    emission_deg: Optional[np.ndarray] = None,
    phase_deg: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    n_scales: int = 4, n_orient: int = 12,
    cfg=None,
) -> Union[PWIFTMaps, np.ndarray]:
    """Dispatch to the correct branch based on `sensor.illumination_method`.
    Returns a PWIFTMaps bundle for the pwift_akimov branch (OHRC/LROC), or a
    plain ndarray for the other branches - matching.py dispatches on this."""
    if sensor.illumination_method == "pwift_akimov":
        return correct_ohrc_lroc(img, incidence_deg=incidence_deg, emission_deg=emission_deg,
                                  phase_deg=phase_deg, n_scales=n_scales, n_orient=n_orient, cfg=cfg)
    if sensor.illumination_method == "hist_shadow_log":
        return correct_tmc(img, reference=reference)
    if sensor.illumination_method == "clahe_invert_dilate":
        return correct_iirs(img)
    raise ValueError(f"Unknown illumination method: {sensor.illumination_method}")