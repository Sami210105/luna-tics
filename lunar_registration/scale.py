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
from typing import List, Tuple

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
) -> Tuple[float, float]:
    """Runs the cheap coarse-to-fine correlation search (pwift.py) over the
    sensor's configured scale range to pick a starting (scale, rotation)
    estimate before the full matching stage runs. Returns (best_scale,
    best_rotation_deg).

    PERF FIX (this revision): when `sensor.scale_range` is a fixed ratio
    (lo == hi - e.g. LROC-vs-LROC, scale_range=(1.0, 1.0)), the old code
    still called `np.geomspace(lo, hi, n)` with `n = max(3, cfg.pyramid_levels)`
    (4 by default), which for lo==hi produces FOUR IDENTICAL copies of that
    same scale value. Each one triggered a full, expensive PWIFT coarse-
    search evaluation of the exact same data - pure wasted work for zero
    information gain. `build_pyramid` above already had this exact
    `lo == hi` short-circuit; `select_best_scale` just didn't. Now it does:
    skip straight to a single-scale rotation-only search."""
    lo, hi = sensor.scale_range
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