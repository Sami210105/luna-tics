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

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from . import pwift as pwift_core
from .config import PipelineConfig
from .pwift import PWIFTMaps


@dataclass
class MatchResult:
    method: str                              # "pwift", "roma2", "eloftr", "hybrid_pwift_roma2", etc.
    pts_src: np.ndarray                      # (N, 2) float32, (x, y) in source image
    pts_dst: np.ndarray                      # (N, 2) float32, (x, y) in reference image
    scores: Optional[np.ndarray] = None      # (N,) float32 confidence / certainty
    inlier_mask: Optional[np.ndarray] = None # (N,) bool inlier mask
    H: Optional[np.ndarray] = None           # (3, 3) coarse homography if computed
    provenance: str = "direct"               # "direct", "fused_pwift_roma", "pwift_rejected", etc.
    latency_ms: float = 0.0


class MissingWeights(FileNotFoundError):
    """Raised when deep matcher weights cannot be resolved."""
    pass


class BaseMatcher(ABC):
    """Abstract base class for lunar registration matchers."""

    @abstractmethod
    def match(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        ref_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        **kwargs,
    ) -> MatchResult:
        """Run matching on source and reference images."""
        pass


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

    if cfg.pwift_use_context_descriptor:
        context_src = pwift_core.compute_context_descriptor(
            [d.keypoint for d in desc_src], src_maps.M_PW, src_maps.mask,
            n_rings=cfg.pwift_context_rings, n_sectors=cfg.pwift_context_sectors,
            ring_spacing_px=cfg.pwift_context_ring_spacing_px,
        )
        context_dst = pwift_core.compute_context_descriptor(
            [d.keypoint for d in desc_dst], dst_maps.M_PW, dst_maps.mask,
            n_rings=cfg.pwift_context_rings, n_sectors=cfg.pwift_context_sectors,
            ring_spacing_px=cfg.pwift_context_ring_spacing_px,
        )
        matches = pwift_core.swap_aware_match_with_context(
            desc_src, desc_dst, context_src, context_dst,
            no=cfg.pwift_descriptor_cells, nbins=cfg.pwift_orientations,
            ratio_test=cfg.pwift_ratio_test, context_weight=cfg.pwift_context_weight,
        )
    else:
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


# ----------------------------------------------------------------------
# PWIFT Matcher
# ----------------------------------------------------------------------

class PWIFTMatcher(BaseMatcher):
    """Photometric-weighted structural feature matcher."""

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()

    def match(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        ref_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        **kwargs,
    ) -> MatchResult:
        t0 = time.perf_counter()

        if isinstance(src_illum, PWIFTMaps) and isinstance(ref_illum, PWIFTMaps):
            res = run_pwift_matching_pw(src_illum, ref_illum, self.cfg)
        else:
            s_map = src_illum if isinstance(src_illum, np.ndarray) else src_img
            r_map = ref_illum if isinstance(ref_illum, np.ndarray) else ref_img
            res = run_pwift_matching_generic(s_map, src_img, r_map, ref_img, self.cfg)

        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res


def run_pwift_matching(
    src_illum: Union[PWIFTMaps, np.ndarray], src_img: np.ndarray,
    dst_illum: Union[PWIFTMaps, np.ndarray], dst_img: np.ndarray,
    cfg: PipelineConfig,
) -> MatchResult:
    matcher = PWIFTMatcher(cfg)
    return matcher.match(src_img, dst_img, src_illum=src_illum, ref_illum=dst_illum)


# ----------------------------------------------------------------------
# RoMa v2 Matcher (Fine-tuned Lunar Model)
# ----------------------------------------------------------------------

def resolve_romav2_weights(weights_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve RoMa v2 fine-tuned model weights path portably.

    Checks:
    1. Explicit weights_path argument if provided.
    2. ROMA2_WEIGHTS_PATH environment variable.
    3. LUNATICS_MODELS_DIR / "romav2_stereolunar_finetuned.pt".
    4. Repo-relative paths: <repo_root>/models/romav2_stereolunar_finetuned.pt, <repo_root>/weights/...
    5. Local fallback paths if available on the current host.
    """
    if weights_path:
        p = Path(weights_path)
        if p.exists():
            return p
        return p

    env_val = os.environ.get("ROMA2_WEIGHTS_PATH")
    if env_val and Path(env_val).exists():
        return Path(env_val)

    models_dir = os.environ.get("LUNATICS_MODELS_DIR")
    if models_dir:
        p = Path(models_dir) / "romav2_stereolunar_finetuned.pt"
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "models" / "romav2_stereolunar_finetuned.pt",
        repo_root / "weights" / "romav2_stereolunar_finetuned.pt",
        Path("/home/ojas/projects/SIH/illumination_variation/models/romav2_stereolunar_finetuned.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return repo_root / "models" / "romav2_stereolunar_finetuned.pt"


def _ensure_romav2_path():
    """Locate and add RoMaV2 source directory to sys.path portably."""
    try:
        import romav2
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    candidates: List[Path] = []
    env_src = os.environ.get("ROMA2_SRC_DIR")
    if env_src:
        candidates.append(Path(env_src))

    try:
        import vismatch
        candidates.append(Path(vismatch.__file__).resolve().parent / "third_party" / "RoMaV2" / "src")
    except ImportError:
        pass

    candidates.extend([
        repo_root / "vismatch" / "third_party" / "RoMaV2" / "src",
        repo_root / "third_party" / "RoMaV2" / "src",
        Path("/home/ojas/projects/SIH/illumination_variation/.venv/lib/python3.13/site-packages/vismatch/third_party/RoMaV2/src"),
    ])
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            return


class _AutocastDisabledTorch:
    """Disables autocast contexts on non-CUDA devices."""
    def __getattr__(self, name):
        import torch
        return getattr(torch, name)

    @staticmethod
    def autocast(*args, **kwargs):
        import torch
        device_type = kwargs.get("device_type", args[0] if args else "cuda")
        if device_type != "cuda":
            kwargs["enabled"] = False
        return torch.autocast(*args, **kwargs)


class Roma2Matcher(BaseMatcher):
    """Fine-tuned RoMa v2 matcher with dense certainty and tiling support."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_keypoints: int = 2048,
        cfg_setting: str = "fast",
        tile_size: int = 800,
    ):
        import torch

        self.max_keypoints = max_keypoints
        self.cfg_setting = cfg_setting
        self.tile_size = tile_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve weights portably
        self.weights_path = resolve_romav2_weights(weights_path)
        if not self.weights_path.exists():
            raise MissingWeights(
                f"RoMa v2 weights file not found: {self.weights_path}. "
                "Specify via weights_path, ROMA2_WEIGHTS_PATH env var, or place at models/romav2_stereolunar_finetuned.pt"
            )

        _ensure_romav2_path()

        if "cuda" not in str(self.device):
            for name, module in list(sys.modules.items()):
                if (name == "romav2" or name.startswith("romav2.")) and getattr(module, "torch", None) is torch:
                    module.torch = _AutocastDisabledTorch()
        else:
            for name, module in list(sys.modules.items()):
                if (name == "romav2" or name.startswith("romav2.")) and isinstance(getattr(module, "torch", None), _AutocastDisabledTorch):
                    module.torch = torch

        from romav2 import RoMaV2
        try:
            from vismatch.utils import set_device_globals
            set_device_globals("romav2", str(self.device))
        except Exception:
            pass

        # Adaptive VRAM check: if CUDA free memory is under 3GB, switch to 'turbo' setting
        if "cuda" in str(self.device) and self.cfg_setting == "fast":
            try:
                free_b, _ = torch.cuda.mem_get_info()
                if free_b < 3.0 * (1024**3):
                    self.cfg_setting = "turbo"
            except Exception:
                pass

        cfg = RoMaV2.Cfg(compile=False, setting=self.cfg_setting)
        self.model = RoMaV2(cfg=cfg)

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict, strict=False)

        if "cuda" not in str(self.device):
            self.model = self.model.float()

        try:
            self.model.eval().to(self.device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                import warnings
                warnings.warn(f"CUDA OOM when loading RoMa2 ({e}); falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.float().to(self.device)
            else:
                raise

    def _prepare_tensor(self, img: np.ndarray) -> "torch.Tensor":
        import torch
        t = torch.from_numpy(np.asarray(img))
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3:
            if t.shape[2] == 3 or t.shape[2] == 1:
                if t.shape[2] == 1:
                    t = t.repeat(1, 1, 3)
                t = t.permute(2, 0, 1)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        return t.unsqueeze(0).to(self.device)

    def match(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        ref_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        **kwargs,
    ) -> MatchResult:
        import torch

        t0 = time.perf_counter()
        h0, w0 = src_img.shape[:2]
        h1, w1 = ref_img.shape[:2]

        max_dim = max(h0, w0, h1, w1)
        if max_dim <= self.tile_size:
            tiles_src = [{"tile": src_img, "bbox": (0, 0, w0, h0)}]
            tiles_ref = [{"tile": ref_img, "bbox": (0, 0, w1, h1)}]
        else:
            tiles_src = self._generate_tiles(src_img, self.tile_size, overlap=0.15)
            tiles_ref = self._generate_tiles(ref_img, self.tile_size, overlap=0.15)

        all_pts0 = []
        all_pts1 = []
        all_conf = []

        n_pairs = min(len(tiles_src), len(tiles_ref))
        kpts_per_tile = max(128, self.max_keypoints // max(1, n_pairs))

        with torch.inference_mode():
            for i in range(n_pairs):
                t_s = tiles_src[i]
                t_r = tiles_ref[i]
                im0 = t_s["tile"]
                im1 = t_r["tile"]
                bb0 = t_s["bbox"]
                bb1 = t_r["bbox"]

                ten0 = self._prepare_tensor(im0)
                ten1 = self._prepare_tensor(im1)
                th0, tw0 = ten0.shape[-2:]
                th1, tw1 = ten1.shape[-2:]

                try:
                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()
                        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            preds = self.model.match(ten0, ten1)
                    else:
                        preds = self.model.match(ten0, ten1)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    try:
                        self.model.apply_setting("turbo")
                        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            preds = self.model.match(ten0, ten1)
                    except torch.cuda.OutOfMemoryError:
                        import warnings
                        warnings.warn("RoMa2 CUDA OOM during inference; falling back to CPU.")
                        torch.cuda.empty_cache()
                        self.device = torch.device("cpu")
                        self.model = self.model.float().to(self.device)
                        ten0 = self._prepare_tensor(im0)
                        ten1 = self._prepare_tensor(im1)
                        preds = self.model.match(ten0, ten1)

                if isinstance(preds, dict):
                    preds = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in preds.items()}

                matches, confidence, _, _ = self.model.sample(preds, kpts_per_tile)
                mkpts0, mkpts1 = self.model.to_pixel_coordinates(matches, th0, tw0, th1, tw1)

                p0 = mkpts0.detach().cpu().numpy().reshape(-1, 2)
                p1 = mkpts1.detach().cpu().numpy().reshape(-1, 2)
                conf = confidence.detach().cpu().numpy().reshape(-1)

                p0[:, 0] += bb0[0]
                p0[:, 1] += bb0[1]
                p1[:, 0] += bb1[0]
                p1[:, 1] += bb1[1]

                all_pts0.append(p0)
                all_pts1.append(p1)
                all_conf.append(conf)

        if all_pts0 and sum(len(p) for p in all_pts0) > 0:
            pts0 = np.concatenate(all_pts0, axis=0).astype(np.float32)
            pts1 = np.concatenate(all_pts1, axis=0).astype(np.float32)
            scores = np.concatenate(all_conf, axis=0).astype(np.float32)
        else:
            pts0 = np.zeros((0, 2), dtype=np.float32)
            pts1 = np.zeros((0, 2), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return MatchResult(
            method="roma2",
            pts_src=pts0,
            pts_dst=pts1,
            scores=scores,
            provenance="direct",
            latency_ms=latency_ms,
        )

    def _generate_tiles(self, img: np.ndarray, tile_size: int, overlap: float = 0.15) -> List[Dict[str, Any]]:
        h, w = img.shape[:2]
        step = int(tile_size * (1.0 - overlap))
        tiles = []
        for y in range(0, max(1, h - tile_size + step), step):
            for x in range(0, max(1, w - tile_size + step), step):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = img[y:y_end, x:x_end]
                tiles.append({"tile": tile, "bbox": (x, y, x_end - x, y_end - y)})
        return tiles


# ----------------------------------------------------------------------
# EfficientLoFTR Matcher (Fine-tuned Lunar Model)
# ----------------------------------------------------------------------

def resolve_eloftr_checkpoint(checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve EfficientLoFTR fine-tuned checkpoint path portably.

    Checks:
    1. Explicit checkpoint_path argument if provided.
    2. ELOFT_CHECKPOINT_PATH / ELOFT_CKPT_PATH environment variable.
    3. LUNATICS_MODELS_DIR / "eloftr_lunar.ckpt".
    4. Repo-relative paths: <repo_root>/models/eloftr_lunar.ckpt, <repo_root>/finetune/...
    5. Local fallback paths if available on the current host.
    """
    if checkpoint_path:
        p = Path(checkpoint_path)
        if p.exists():
            return p
        return p

    for env_k in ("ELOFT_CHECKPOINT_PATH", "ELOFT_CKPT_PATH", "ELOFR_CHECKPOINT_PATH"):
        env_val = os.environ.get(env_k)
        if env_val and Path(env_val).exists():
            return Path(env_val)

    models_dir = os.environ.get("LUNATICS_MODELS_DIR")
    if models_dir:
        p = Path(models_dir) / "eloftr_lunar.ckpt"
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "models" / "eloftr_lunar.ckpt",
        repo_root / "weights" / "eloftr_lunar.ckpt",
        repo_root / "finetune" / "EfficientLoFTR" / "logs" / "tb_logs" / "lunar_full_finetune" / "version_1" / "checkpoints" / "epoch=0-auc@5=0.788-auc@10=0.853-auc@20=0.899.ckpt",
        Path("/home/ojas/projects/SIH/illumination_variation/finetune/EfficientLoFTR/logs/tb_logs/lunar_full_finetune/version_1/checkpoints/epoch=0-auc@5=0.788-auc@10=0.853-auc@20=0.899.ckpt"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return repo_root / "models" / "eloftr_lunar.ckpt"


def resolve_eloftr_config(config_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve EfficientLoFTR config path portably."""
    if config_path:
        p = Path(config_path)
        if p.exists():
            return p
        return p

    for env_k in ("ELOFT_CONFIG_PATH", "ELOFR_CONFIG_PATH"):
        env_val = os.environ.get(env_k)
        if env_val and Path(env_val).exists():
            return Path(env_val)

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "finetune" / "EfficientLoFTR" / "configs" / "loftr" / "eloftr_lunar.py",
        repo_root / "configs" / "loftr" / "eloftr_lunar.py",
        Path("/home/ojas/projects/SIH/illumination_variation/finetune/EfficientLoFTR/configs/loftr/eloftr_lunar.py"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return repo_root / "configs" / "loftr" / "eloftr_lunar.py"


def _ensure_eloftr_path():
    """Locate and add EfficientLoFTR source directory to sys.path portably."""
    try:
        import src.loftr
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    candidates: List[Path] = []
    env_src = os.environ.get("ELOFT_SRC_DIR")
    if env_src:
        candidates.append(Path(env_src))

    candidates.extend([
        repo_root / "finetune" / "EfficientLoFTR",
        repo_root / "third_party" / "EfficientLoFTR",
        Path("/home/ojas/projects/SIH/illumination_variation/finetune/EfficientLoFTR"),
    ])
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            return


class EloftrMatcher(BaseMatcher):
    """Fine-tuned EfficientLoFTR matcher."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        import torch

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve paths portably
        self.checkpoint_path = resolve_eloftr_checkpoint(checkpoint_path)
        self.config_path = resolve_eloftr_config(config_path)

        if not self.checkpoint_path.exists():
            raise MissingWeights(
                f"EfficientLoFTR checkpoint not found: {self.checkpoint_path}. "
                "Specify via checkpoint_path, ELOFT_CHECKPOINT_PATH env var, or place at models/eloftr_lunar.ckpt"
            )

        _ensure_eloftr_path()

        from src.config.default import get_cfg_defaults
        from src.loftr import LoFTR, reparameter
        from src.utils.misc import lower_config

        cfg = get_cfg_defaults()
        if self.config_path.exists():
            cfg.merge_from_file(str(self.config_path))

        self.model = LoFTR(config=lower_config(cfg)["loftr"])
        sd = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        raw_sd = sd.get("state_dict", sd)
        clean_sd = {k.replace("matcher.", ""): v for k, v in raw_sd.items()}
        self.model.load_state_dict(clean_sd, strict=False)
        try:
            self.model = reparameter(self.model).eval().to(self.device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                import warnings
                warnings.warn(f"CUDA OOM when loading EfficientLoFTR ({e}); falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = reparameter(self.model).eval().to(self.device)
            else:
                raise
        for p in self.model.parameters():
            p.requires_grad = False

    def _prepare_gray_tensor(self, img: np.ndarray) -> Tuple["torch.Tensor", Tuple[int, int]]:
        import torch
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            if a.shape[2] == 1:
                a = a[..., 0]
            else:
                a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        orig_h, orig_w = a.shape[:2]
        pad_h = (32 - (orig_h % 32)) % 32
        pad_w = (32 - (orig_w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="reflect")
        if a.max() > 1.0:
            a = a / 255.0
        t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return t, (orig_h, orig_w)

    def match(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        ref_illum: Optional[Union[PWIFTMaps, np.ndarray]] = None,
        **kwargs,
    ) -> MatchResult:
        import torch

        t0 = time.perf_counter()
        t_src, (h0, w0) = self._prepare_gray_tensor(src_img)
        t_ref, (h1, w1) = self._prepare_gray_tensor(ref_img)

        b = {"image0": t_src, "image1": t_ref}

        with torch.inference_mode():
            try:
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        self.model(b)
                    torch.cuda.synchronize()
                else:
                    self.model(b)
            except torch.cuda.OutOfMemoryError:
                import warnings
                warnings.warn("EfficientLoFTR CUDA OOM during inference; falling back to CPU.")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                t_src, (h0, w0) = self._prepare_gray_tensor(src_img)
                t_ref, (h1, w1) = self._prepare_gray_tensor(ref_img)
                b = {"image0": t_src, "image1": t_ref}
                self.model(b)

        pts0 = b["mkpts0_f"].detach().cpu().numpy().reshape(-1, 2).astype(np.float32)
        pts1 = b["mkpts1_f"].detach().cpu().numpy().reshape(-1, 2).astype(np.float32)
        conf = b.get("mconf")
        scores = conf.detach().cpu().numpy().reshape(-1).astype(np.float32) if conf is not None else np.ones(len(pts0), dtype=np.float32)

        # Filter out points in padded border regions
        if len(pts0) > 0:
            valid = (pts0[:, 0] < w0) & (pts0[:, 1] < h0) & (pts1[:, 0] < w1) & (pts1[:, 1] < h1)
            pts0 = pts0[valid]
            pts1 = pts1[valid]
            scores = scores[valid]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return MatchResult(
            method="eloftr",
            pts_src=pts0,
            pts_dst=pts1,
            scores=scores,
            provenance="direct",
            latency_ms=latency_ms,
        )


# ----------------------------------------------------------------------
# Matcher Factory
# ----------------------------------------------------------------------

def get_matcher(
    name: str,
    cfg: Optional[PipelineConfig] = None,
    device: Optional[str] = None,
) -> BaseMatcher:
    """Instantiate a modular matcher by name."""
    cfg = cfg or PipelineConfig()
    dev = device or cfg.device
    clean_name = name.strip().lower()

    if clean_name in ("pwift", "pwift_akimov"):
        return PWIFTMatcher(cfg)
    elif clean_name in ("roma", "roma2", "romav2"):
        return Roma2Matcher(
            weights_path=cfg.roma2_weights_path,
            device=dev,
            max_keypoints=cfg.roma2_max_keypoints,
            cfg_setting=cfg.roma2_cfg_setting,
            tile_size=cfg.roma2_tile_size,
        )
    elif clean_name in ("eloftr", "efficientloftr", "loftr"):
        return EloftrMatcher(
            checkpoint_path=cfg.eloftr_checkpoint_path,
            config_path=cfg.eloftr_config_path,
            device=dev,
        )
    else:
        raise ValueError(
            f"Unknown matcher '{name}'. Available: 'pwift', 'roma2', 'eloftr'."
        )


# ----------------------------------------------------------------------
# Combined driver (backward-compatible)
# ----------------------------------------------------------------------

def run_matching(
    src_illum: Union[PWIFTMaps, np.ndarray], src_img: np.ndarray,
    dst_illum: Union[PWIFTMaps, np.ndarray], dst_img: np.ndarray,
    cfg: PipelineConfig, use_eloftr: bool = True,
) -> Tuple[MatchResult, Optional[MatchResult]]:
    """Runs PWIFT always; runs EfficientLoFTR if requested."""
    pwift_result = run_pwift_matching(src_illum, src_img, dst_illum, dst_img, cfg)

    eloftr_result = None
    if use_eloftr:
        try:
            matcher = get_matcher("eloftr", cfg)
            eloftr_result = matcher.match(src_img, dst_img)
        except Exception as e:
            import warnings
            warnings.warn(f"EfficientLoFTR match failed ({e}); continuing with PWIFT only.")

    return pwift_result, eloftr_result