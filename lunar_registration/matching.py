"""
Stage 3 - Feature detection and matching: PWIFT vs EfficientLoFTR
===================================================================
Both matchers are run and reported independently (with their own inlier
count / ratio / RMSE via metrics.py) so you can show the comparison table
your PS explicitly asks for ("best match selected"); `run_matching()` then
picks whichever gave more RANSAC inliers as the one that feeds stage 4.

--- PWIFT dispatch ---
`run_pwift_matching` receives whatever illumination.apply_illumination_correction
returned. If it's a `pwift.PWIFTMaps` bundle (OHRC/LROC), it runs the
paper-faithful pipeline from pwift.py (Eq 9-21). Otherwise (TMC/IIRS, a
plain ndarray) it falls back to the original generic dual-channel matcher,
which was already a best-effort approximation for sensors without per-pixel
photometric geometry.

--- SWAP POINT ---
EfficientLoFTR here uses the HuggingFace `transformers` pipeline
("zju-community/efficientloftr"), which is trained on terrestrial imagery
(MegaDepth/ScanNet). For lunar cross-domain use, validate zero-shot quality
first; fine-tune on lunar pairs (e.g. via MoonAnything/LunarPhoto renders)
if zero-shot inlier ratio is too low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from . import pwift as pwift_core
from .config import PipelineConfig
from .pwift import PWIFTMaps


@dataclass
class MatchResult:
    method: str                      # "pwift" or "eloftr"
    pts_src: np.ndarray              # (N, 2) float32, (x, y) in source image
    pts_dst: np.ndarray              # (N, 2) float32, (x, y) in reference image
    scores: Optional[np.ndarray] = None  # per-match confidence, if available


def filter_by_displacement_prior(
    pts_src: np.ndarray, pts_dst: np.ndarray, scores: Optional[np.ndarray],
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Pre-RANSAC outlier rejection using a robust displacement prior.

    On repetitive terrain (craters), a chunk of descriptor matches pass the
    ratio test but are simply matched to the wrong, similar-looking crater
    - these show up as long, wildly-angled lines in the match visualization
    and dilute RANSAC's inlier pool. Since src/dst are the same sensor,
    scale, and framing here, the *correct* matches' displacement vectors
    (dst - src) cluster tightly; this keeps only matches whose displacement
    is close to that cluster (median + MAD), plus an optional hard cap.

    Skips filtering (returns input unchanged) if there are too few matches
    to compute robust statistics from - cfg.pwift_displacement_min_matches.
    """
    n = len(pts_src)
    if n < cfg.pwift_displacement_min_matches:
        return pts_src, pts_dst, scores

    disp = pts_dst - pts_src  # (N, 2)

    keep = np.ones(n, dtype=bool)
    if cfg.pwift_max_displacement_px is not None:
        mag = np.linalg.norm(disp, axis=1)
        keep &= mag <= cfg.pwift_max_displacement_px

    median_disp = np.median(disp[keep], axis=0)
    dev = np.linalg.norm(disp - median_disp, axis=1)
    mad = np.median(np.abs(dev - np.median(dev)))
    # MAD can be 0 (e.g. if most matches are near-identical) - guard against
    # a degenerate zero-width threshold that would reject everything.
    scale = mad if mad > 1e-6 else max(np.median(dev), 1.0)
    threshold = np.median(dev) + cfg.pwift_displacement_mad_k * scale
    keep &= dev <= threshold

    if keep.sum() < cfg.pwift_displacement_min_matches:
        # Filter was too aggressive (or the data's genuinely too scattered
        # to have a dominant displacement cluster) - fall back to
        # unfiltered rather than starving RANSAC entirely.
        import warnings
        warnings.warn(
            "filter_by_displacement_prior: robust filter would leave only "
            f"{int(keep.sum())} matches (< "
            f"{cfg.pwift_displacement_min_matches}); skipping the filter "
            "for this pair and passing all raw matches to RANSAC instead."
        )
        return pts_src, pts_dst, scores

    filtered_scores = scores[keep] if scores is not None else None
    return pts_src[keep], pts_dst[keep], filtered_scores


# ----------------------------------------------------------------------
# PWIFT matcher - GENERIC path (TMC/IIRS)
# ----------------------------------------------------------------------

def run_pwift_matching_generic(
    src_pc: np.ndarray, src_img: np.ndarray,
    dst_pc: np.ndarray, dst_img: np.ndarray,
    cfg: PipelineConfig,
) -> MatchResult:
    kp_src = pwift_core.detect_keypoints_dual_channel(
        src_pc, threshold=cfg.pwift_keypoint_threshold if hasattr(cfg, "pwift_keypoint_threshold") else 0.08)
    kp_dst = pwift_core.detect_keypoints_dual_channel(
        dst_pc, threshold=cfg.pwift_keypoint_threshold if hasattr(cfg, "pwift_keypoint_threshold") else 0.08)

    desc_src = pwift_core.compute_descriptors(
        src_img, src_pc, kp_src, patch_size=cfg.pwift_descriptor_patch)
    desc_dst = pwift_core.compute_descriptors(
        dst_img, dst_pc, kp_dst, patch_size=cfg.pwift_descriptor_patch)

    matches = pwift_core.match_descriptors_swap_aware(
        desc_src, desc_dst, ratio_test=cfg.pwift_ratio_test)

    if not matches:
        return MatchResult("pwift", np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))

    pts_src = np.array([[desc_src[m.src_idx].keypoint.x, desc_src[m.src_idx].keypoint.y]
                         for m in matches], dtype=np.float32)
    pts_dst = np.array([[desc_dst[m.dst_idx].keypoint.x, desc_dst[m.dst_idx].keypoint.y]
                         for m in matches], dtype=np.float32)
    scores = np.array([1.0 / (1.0 + m.distance) for m in matches], dtype=np.float32)
    pts_src, pts_dst, scores = filter_by_displacement_prior(pts_src, pts_dst, scores, cfg)
    return MatchResult("pwift", pts_src, pts_dst, scores)


# ----------------------------------------------------------------------
# PWIFT matcher - paper-faithful path (OHRC/LROC), Eq 9-21
# ----------------------------------------------------------------------

def run_pwift_matching_pw(
    src_maps: PWIFTMaps, dst_maps: PWIFTMaps, cfg: PipelineConfig,
) -> MatchResult:
    """Full PWIFT matching stage (Sec 3.4-3.5, Eq 9-21) using the
    photometric-weighted structural maps already computed by
    illumination.apply_illumination_correction. This is the "full matching
    process" the paper runs under the rotation-scale hypothesis C* chosen
    by pwift.coarse_to_fine_rotation_scale (called from scale.py, upstream
    of illumination/matching in pipeline.py)."""
    keypoints_src = pwift_core.detect_keypoints_pw(
        src_maps.M_PW, src_maps.m_PW, src_maps.w_soft, src_maps.mask,
        min_distance=cfg.pwift_min_keypoint_distance, max_keypoints=cfg.pwift_max_keypoints,
        min_retention_ratio=cfg.pwift_min_retention_ratio,
        score_percentile=cfg.pwift_keypoint_score_percentile,
    )
    keypoints_dst = pwift_core.detect_keypoints_pw(
        dst_maps.M_PW, dst_maps.m_PW, dst_maps.w_soft, dst_maps.mask,
        min_distance=cfg.pwift_min_keypoint_distance, max_keypoints=cfg.pwift_max_keypoints,
        min_retention_ratio=cfg.pwift_min_retention_ratio,
        score_percentile=cfg.pwift_keypoint_score_percentile,
    )
    for kp in keypoints_src:
        pwift_core.dominant_orientation_pw(kp, src_maps.MIM, src_maps.M_PW, src_maps.w_soft, K=cfg.pwift_orientations)
    for kp in keypoints_dst:
        pwift_core.dominant_orientation_pw(kp, dst_maps.MIM, dst_maps.M_PW, dst_maps.w_soft, K=cfg.pwift_orientations)

    desc_src = pwift_core.compute_bichannel_descriptors(
        keypoints_src, src_maps.MIM, src_maps.M_PW, src_maps.w_soft, src_maps.w,
        patch_size=cfg.pwift_descriptor_patch, no=cfg.pwift_descriptor_cells,
        nbins=cfg.pwift_orientations, t=cfg.pwift_bright_dark_threshold,
    )
    desc_dst = pwift_core.compute_bichannel_descriptors(
        keypoints_dst, dst_maps.MIM, dst_maps.M_PW, dst_maps.w_soft, dst_maps.w,
        patch_size=cfg.pwift_descriptor_patch, no=cfg.pwift_descriptor_cells,
        nbins=cfg.pwift_orientations, t=cfg.pwift_bright_dark_threshold,
    )

    matches = pwift_core.swap_aware_match(
        desc_src, desc_dst, no=cfg.pwift_descriptor_cells, nbins=cfg.pwift_orientations,
        ratio_test=cfg.pwift_ratio_test,
    )

    if not matches:
        return MatchResult("pwift", np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))

    pts_src = np.array([[desc_src[m.src_idx].keypoint.x, desc_src[m.src_idx].keypoint.y]
                         for m in matches], dtype=np.float32)
    pts_dst = np.array([[desc_dst[m.dst_idx].keypoint.x, desc_dst[m.dst_idx].keypoint.y]
                         for m in matches], dtype=np.float32)
    scores = np.array([1.0 / (1.0 + m.distance) for m in matches], dtype=np.float32)
    pts_src, pts_dst, scores = filter_by_displacement_prior(pts_src, pts_dst, scores, cfg)
    return MatchResult("pwift", pts_src, pts_dst, scores)


def run_pwift_matching(
    src_illum: Union[PWIFTMaps, np.ndarray], src_img: np.ndarray,
    dst_illum: Union[PWIFTMaps, np.ndarray], dst_img: np.ndarray,
    cfg: PipelineConfig,
) -> MatchResult:
    """Dispatches on the type of `src_illum`/`dst_illum` (whatever
    illumination.apply_illumination_correction returned): a PWIFTMaps pair
    -> the paper-faithful path; anything else -> the generic path."""
    if isinstance(src_illum, PWIFTMaps) and isinstance(dst_illum, PWIFTMaps):
        return run_pwift_matching_pw(src_illum, dst_illum, cfg)
    return run_pwift_matching_generic(src_illum, src_img, dst_illum, dst_img, cfg)


# ----------------------------------------------------------------------
# EfficientLoFTR matcher (HuggingFace transformers) - unchanged
# ----------------------------------------------------------------------

_ELOFTR_PIPE = None


def _get_eloftr_pipeline(model_id: str, device: str):
    global _ELOFTR_PIPE
    if _ELOFTR_PIPE is None:
        from transformers import pipeline as hf_pipeline
        _ELOFTR_PIPE = hf_pipeline(task="keypoint-matching", model=model_id, device=device)
    return _ELOFTR_PIPE


def run_eloftr_matching(
    src_img: np.ndarray, dst_img: np.ndarray, cfg: PipelineConfig,
) -> MatchResult:
    from PIL import Image as PILImage

    def to_pil(a):
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return PILImage.fromarray(u8).convert("RGB")

    pipe = _get_eloftr_pipeline(cfg.eloftr_model_id, cfg.eloftr_device)
    result = pipe([to_pil(src_img), to_pil(dst_img)], threshold=cfg.eloftr_confidence_threshold)

    entries = result[0] if isinstance(result, list) and len(result) == 1 else result
    pts_src, pts_dst, scores = [], [], []
    for m in entries:
        kp0 = m.get("keypoint_image_0", m.get("keypoints0"))
        kp1 = m.get("keypoint_image_1", m.get("keypoints1"))
        if kp0 is None or kp1 is None:
            continue
        pts_src.append([kp0["x"], kp0["y"]])
        pts_dst.append([kp1["x"], kp1["y"]])
        scores.append(float(m.get("score", 1.0)))

    return MatchResult(
        "eloftr",
        np.array(pts_src, dtype=np.float32) if pts_src else np.zeros((0, 2), np.float32),
        np.array(pts_dst, dtype=np.float32) if pts_dst else np.zeros((0, 2), np.float32),
        np.array(scores, dtype=np.float32) if scores else None,
    )


# ----------------------------------------------------------------------
# Combined driver
# ----------------------------------------------------------------------

def run_matching(
    src_illum: Union[PWIFTMaps, np.ndarray], src_img: np.ndarray,
    dst_illum: Union[PWIFTMaps, np.ndarray], dst_img: np.ndarray,
    cfg: PipelineConfig, use_eloftr: bool = True,
) -> Tuple[MatchResult, Optional[MatchResult]]:
    """Runs PWIFT always (dispatched by run_pwift_matching); runs
    EfficientLoFTR if `use_eloftr` and `transformers`/`torch` are importable
    (falls back to PWIFT-only with a warning otherwise)."""
    pwift_result = run_pwift_matching(src_illum, src_img, dst_illum, dst_img, cfg)

    eloftr_result = None
    if use_eloftr:
        try:
            eloftr_result = run_eloftr_matching(src_img, dst_img, cfg)
        except ImportError as e:
            import warnings
            warnings.warn(
                f"EfficientLoFTR unavailable ({e}); install with "
                "`pip install transformers torch`. Continuing with PWIFT only."
            )

    return pwift_result, eloftr_result