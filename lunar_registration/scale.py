"""
Stage 5 - Scale variation handling
=====================================
Builds a multi-scale pyramid of the SOURCE image spanning the sensor's
configured scale_range (see config.py - e.g. OHRC 0.5x-3x vs the reference),
runs the coarse-to-fine PWIFT rotation/scale search (pwift.py) at each level
to pick the best-scoring level, and returns that resampled image + the scale
factor actually used - which then gets folded into the final homography
in georeference.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import SensorConfig, PipelineConfig
from .pwift import coarse_to_fine_rotation_scale


@dataclass
class PyramidLevel:
    scale: float
    image: np.ndarray


def build_pyramid(img: np.ndarray, sensor: SensorConfig, cfg: PipelineConfig) -> List[PyramidLevel]:
    lo, hi = sensor.scale_range
    if cfg.pyramid_levels <= 1 or lo == hi:
        return [PyramidLevel(scale=1.0, image=img)]
    scales = np.geomspace(lo, hi, cfg.pyramid_levels)
    levels = []
    for s in scales:
        resized = ndimage.zoom(img, s, order=1)
        if resized.size > 0:
            levels.append(PyramidLevel(scale=float(s), image=resized))
    return levels


def select_best_scale(
    src_img: np.ndarray, dst_img: np.ndarray, sensor: SensorConfig, cfg: PipelineConfig,
    prior_scale: Optional[float] = None, prior_band_frac: float = 0.15,
) -> Tuple[float, float]:
    """Runs the cheap coarse-to-fine correlation search (pwift.py) over a
    set of scale candidates to pick a starting (scale, rotation) estimate
    before the full matching stage runs. Returns (best_scale,
    best_rotation_deg).

    `prior_scale` (Stage 1.5, preprocessing.estimate_gsd_scale_prior): when
    given, the search is narrowed to a tight geomspace band of
    `cfg.pyramid_levels` candidates spanning
    `prior_scale * (1 +/- prior_band_frac)` (clipped to the sensor's
    configured `scale_range` so it never searches outside physically
    plausible bounds for this sensor) instead of the sensor's full range.
    This is strictly a search-space narrowing, not a hard override - the
    coarse-to-fine search still picks whichever candidate actually scores
    best, so a wrong/stale GSD prior costs you search coverage, not
    correctness, and a bad hit here just means falling back to the wider
    unconstrained behavior next run. Pass `prior_scale=None` (default) to
    get the previous, unconstrained behavior unchanged.

    PERF FIX (earlier revision, still applies to the unconstrained path):
    when `sensor.scale_range` is a fixed ratio (lo == hi - e.g. LROC-vs-
    LROC, scale_range=(1.0, 1.0)), skip straight to a single-scale
    rotation-only search rather than evaluating several identical copies
    of the same scale value."""
    lo, hi = sensor.scale_range

    if prior_scale is not None and prior_scale > 0 and lo != hi:
        band_lo = max(lo, prior_scale * (1.0 - prior_band_frac))
        band_hi = min(hi, prior_scale * (1.0 + prior_band_frac))
        if band_lo < band_hi:
            n = max(3, cfg.pyramid_levels)
            scale_candidates = tuple(np.geomspace(band_lo, band_hi, n))
            return coarse_to_fine_rotation_scale(src_img, dst_img, cfg, scale_candidates=scale_candidates)
        import warnings
        warnings.warn(
            f"select_best_scale: GSD prior scale={prior_scale:.4f} falls "
            f"outside sensor scale_range={sensor.scale_range} - ignoring "
            "the prior and falling back to the unconstrained search over "
            "the full configured range."
        )

    if lo == hi:
        return coarse_to_fine_rotation_scale(src_img, dst_img, cfg, scale_candidates=(lo,))
    n = max(3, cfg.pyramid_levels)
    scale_candidates = tuple(np.geomspace(lo, hi, n))
    return coarse_to_fine_rotation_scale(src_img, dst_img, cfg, scale_candidates=scale_candidates)


def apply_scale(img: np.ndarray, scale: float) -> np.ndarray:
    return ndimage.zoom(img, scale, order=1).astype(np.float32)


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg while preserving the original image size."""
    return ndimage.rotate(
        img,
        angle_deg,
        reshape=False,
        order=1,
        mode="nearest",
    ).astype(np.float32)