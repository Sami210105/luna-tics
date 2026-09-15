"""
End-to-end orchestrator. Run as:

    python -m lunar_registration.pipeline \\
        --source path/to/ohrc_image.img --source-sensor OHRC \\
        --reference path/to/lroc_nac.cub \\
        --out-dir outputs/run1 \\
        [--source-nac-pho path/to/NAC_PHO_..._source.cub] \\
        [--reference-nac-pho path/to/NAC_PHO_..._reference.cub] \\
        [--angles-from-label | --source-incidence inc.tif --source-emission emi.tif] \\
        [--window 2243,298,512,512] \\
        [--no-eloftr]

`--source-nac-pho`/`--reference-nac-pho` read real pixel-wise incidence/
emission/phase angle maps straight out of an LROC NAC_PHO photometry cube's
angle bands (Band 2 = Phase, Band 3 = Local Emission, Band 4 = Local
Incidence) - see preprocessing.py's `load_angles_from_nac_pho`. This is the
preferred angle source whenever you have the NAC_PHO product for an image,
since PWIFT's photometric weighting (paper Sec 3.2) is defined per-pixel;
it takes priority over all the scalar shortcuts below. Both the source and
the reference get their own independent photometric weighting when
supplied - the paper applies this to both images in a pair, not just one.

`--angles-from-label` reads incidence/emission/phase straight from the
source image's PDS3 label (fast, no ISIS) instead of requiring phocube
angle-map .tif files - see preprocessing.py's `load_angles_from_label` for
what it does and when it falls back. Explicit `--source-incidence`/
`--source-emission` paths, if given, always take priority over the label
shortcut. Neither applies to the reference image, which only supports
`--reference-nac-pho` for now.

See README.md for full setup (dependencies, ISIS pre-processing needed for
angle maps, etc).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Optimize PyTorch CUDA allocator to prevent OOM fragmentation on constrained GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import cv2

from .config import PipelineConfig, get_sensor_config
from .preprocessing import load_image, load_angle_maps, LoadedImage, estimate_gsd_scale_prior
from .illumination import apply_illumination_correction
from .scale import select_best_scale, apply_scale, apply_rotation
from .matching import get_matcher, run_pwift_matching, MatchResult, BaseMatcher
from .fusion import fuse_pwift_neural, miho_plus_gcp, assess_pwift_quality
from .verify import compute_structural_ncc, compete_rigid, gate_cheap
from .refine import refine_tile, choose_refiner, compute_texture_energy
from .viewpoint import estimate_viewpoint_transform, HomographyResult
from .georeference import register_image, write_outputs
from .metrics import compute_metrics
from .pwift import reprojection_cleanup


def _parse_window(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def determine_adaptive_matcher(
    requested_matcher: Optional[str],
    incidence_deg: Optional[float],
    real_time: bool,
    cfg: PipelineConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Determines the matcher and operational regime based on physical & operational conditions.

    Returns:
        (resolved_matcher_name, routing_metadata)
    """
    clean = (requested_matcher or "auto").strip().lower()

    if clean != "auto":
        return clean, {
            "mode": "manual_override",
            "regime": "user_specified",
            "incidence_deg": incidence_deg,
            "real_time_requested": real_time,
            "resolved_matcher": clean,
            "reason": f"Explicit user override: {clean}",
        }

    inc = incidence_deg if incidence_deg is not None else 30.0

    # Condition 1: Extreme Polar / Grazing Sun (i >= 70°)
    if inc >= cfg.polar_incidence_threshold_deg:
        chosen = "hybrid_pwift_eloftr" if real_time else "hybrid_pwift_roma2"
        return chosen, {
            "mode": "adaptive",
            "regime": "polar_grazing",
            "incidence_deg": inc,
            "real_time_requested": real_time,
            "resolved_matcher": chosen,
            "reason": (
                f"Incidence {inc:.1f}° >= {cfg.polar_incidence_threshold_deg:.1f}°: "
                "PWIFT harmonic Akimov masking suppresses migrating shadow boundaries."
            ),
        }

    # Condition 2: Real-Time Operational Constraint (Descent TRN)
    if real_time or cfg.real_time_mode:
        return "eloftr", {
            "mode": "adaptive",
            "regime": "real_time_trn",
            "incidence_deg": inc,
            "real_time_requested": True,
            "resolved_matcher": "eloftr",
            "reason": (
                "Real-time descent navigation constraint requested: "
                "EfficientLoFTR selected for ~88 ms latency and high inlier density."
            ),
        }

    # Condition 3: Sub-Pixel Surface Cartography (Nominal)
    return "roma2", {
        "mode": "adaptive",
        "regime": "subpixel_cartography",
        "incidence_deg": inc,
        "real_time_requested": False,
        "resolved_matcher": "roma2",
        "reason": (
            "Nominal orbital mapping: RoMa v2 selected for sub-pixel precision "
            "(79.9% MMA@1px, 0.10 px corner error)."
        ),
    }


def run_pipeline(
    source_path: str, reference_path: str, out_dir: str,
    source_sensor: Optional[str] = None,
    source_incidence_path: Optional[str] = None,
    source_emission_path: Optional[str] = None,
    source_phase_path: Optional[str] = None,
    source_nac_pho_path: Optional[str] = None,
    reference_nac_pho_path: Optional[str] = None,
    nac_pho_band_phase: int = 2,
    nac_pho_band_emission: int = 3,
    nac_pho_band_incidence: int = 4,
    angles_from_label: bool = False,
    fetch_angles_online: bool = False,
    manual_incidence_deg: Optional[float] = None,
    manual_emission_deg: Optional[float] = None,
    manual_phase_deg: Optional[float] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
    source_window: Optional[Tuple[int, int, int, int]] = None,
    reference_window: Optional[Tuple[int, int, int, int]] = None,
    matcher: Optional[str] = None,
    roma2_weights: Optional[str] = None,
    eloftr_checkpoint: Optional[str] = None,
    eloftr_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    use_eloftr: bool = True,
    fuse_pwift_eloftr: bool = False,
    real_time: bool = False,
    enable_orthogonal_gate: Optional[bool] = None,
    polar_incidence_threshold_deg: Optional[float] = None,
    export_gcl_gcps_csv: Optional[bool] = None,
    cfg: Optional[PipelineConfig] = None,
) -> dict:
    cfg = cfg or PipelineConfig()

    if eloftr_ckpt:
        eloftr_checkpoint = eloftr_ckpt
    if roma2_weights:
        cfg.roma2_weights_path = roma2_weights
    if eloftr_checkpoint:
        cfg.eloftr_checkpoint_path = eloftr_checkpoint
    if device:
        cfg.device = device
    if real_time:
        cfg.real_time_mode = True
    if enable_orthogonal_gate is not None:
        cfg.enable_orthogonal_gate = enable_orthogonal_gate
    if polar_incidence_threshold_deg is not None:
        cfg.polar_incidence_threshold_deg = polar_incidence_threshold_deg
    if export_gcl_gcps_csv is not None:
        cfg.export_gcl_gcps_csv = export_gcl_gcps_csv

    if matcher is None:
        if fuse_pwift_eloftr:
            matcher = "hybrid_pwift_eloftr"
        elif not use_eloftr:
            matcher = "pwift"
        else:
            matcher = getattr(cfg, "matcher_type", "auto")

    if source_window is None:
        source_window = window
    if reference_window is None:
        reference_window = window

    # ---- Stage 1: preprocessing ----
    src: LoadedImage = load_image(
        source_path, sensor_hint=source_sensor, window=source_window,
        angles_from_label=angles_from_label, fetch_angles_online=fetch_angles_online,
        manual_incidence_deg=manual_incidence_deg, manual_emission_deg=manual_emission_deg,
        manual_phase_deg=manual_phase_deg,
        nac_pho_path=source_nac_pho_path,
        nac_pho_band_phase=nac_pho_band_phase, nac_pho_band_emission=nac_pho_band_emission,
        nac_pho_band_incidence=nac_pho_band_incidence,
    )
    ref: LoadedImage = load_image(
        reference_path, sensor_hint="LROC", window=reference_window,
        nac_pho_path=reference_nac_pho_path,
        nac_pho_band_phase=nac_pho_band_phase, nac_pho_band_emission=nac_pho_band_emission,
        nac_pho_band_incidence=nac_pho_band_incidence,
    )

    _MAX_SAFE_PIXELS = 4_000_000  # ~2000x2000
    if window is None:
        for _tag, _loaded in (("source", src), ("reference", ref)):
            if _loaded.data.size > _MAX_SAFE_PIXELS:
                _h, _w = _loaded.data.shape
                raise RuntimeError(
                    f"{_tag} image '{_loaded.path}' is {_w}x{_h} "
                    f"({_loaded.data.size:,} px) and no crop window was given. "
                    "Running the full illumination/matching stage at this "
                    "resolution will very likely exhaust memory. Pass "
                    "--source-window x,y,w,h and --reference-window x,y,w,h "
                    "to crop the common geographic overlap region."
                )

    src_incidence, src_emission, src_phase = src.incidence_deg, src.emission_deg, src.phase_deg
    if source_incidence_path and source_emission_path:
        src_incidence, src_emission, src_phase = load_angle_maps(
            source_incidence_path, source_emission_path,
            source_phase_path or source_emission_path, window=source_window,
        )
    ref_incidence, ref_emission, ref_phase = ref.incidence_deg, ref.emission_deg, ref.phase_deg

    # Resolve adaptive matcher based on physical and operational conditions
    inc_for_routing = None
    if manual_incidence_deg is not None:
        inc_for_routing = float(manual_incidence_deg)
    elif src_incidence is not None:
        inc_for_routing = float(np.nanmean(src_incidence))
    elif src.incidence_deg is not None:
        inc_for_routing = float(np.nanmean(src.incidence_deg))

    resolved_matcher, routing_info = determine_adaptive_matcher(
        requested_matcher=matcher,
        incidence_deg=inc_for_routing,
        real_time=real_time or cfg.real_time_mode,
        cfg=cfg,
    )

    # ---- Stage 1.5: GSD-aware scale prior, then coarse-to-fine search ----
    gsd_scale_prior = estimate_gsd_scale_prior(src, ref)
    best_scale, best_rot = select_best_scale(
        src.data, ref.data, src.sensor, cfg, prior_scale=gsd_scale_prior,
    )
    src_scaled = apply_scale(src.data, best_scale)
    src_scaled = apply_rotation(src_scaled, best_rot)
    src_incidence_scaled = apply_scale(src_incidence, best_scale) if src_incidence is not None else None
    src_incidence_scaled = apply_rotation(src_incidence_scaled, best_rot) if src_incidence_scaled is not None else None
    src_emission_scaled = apply_scale(src_emission, best_scale) if src_emission is not None else None
    src_emission_scaled = apply_rotation(src_emission_scaled, best_rot) if src_emission_scaled is not None else None
    src_phase_scaled = apply_scale(src_phase, best_scale) if src_phase is not None else None
    src_phase_scaled = apply_rotation(src_phase_scaled, best_rot) if src_phase_scaled is not None else None

    # ---- Stage 2: illumination correction (per-sensor branch) ----
    src_illum = apply_illumination_correction(
        src_scaled, src.sensor, incidence_deg=src_incidence_scaled,
        emission_deg=src_emission_scaled, phase_deg=src_phase_scaled,
        reference=ref.data, n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations, cfg=cfg,
    )
    ref_illum = apply_illumination_correction(
        ref.data, ref.sensor, incidence_deg=ref_incidence,
        emission_deg=ref_emission, phase_deg=ref_phase, reference=None,
        n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations, cfg=cfg,
    )

    # ---- Stage 3: matching & fusion (Modular, Swappable with Contingency Fallback) ----
    results_to_evaluate: List[MatchResult] = []
    contingency_fallback = {
        "triggered": False,
        "original_matcher": resolved_matcher,
        "fallback_matcher": None,
        "reason": None,
    }

    if resolved_matcher.startswith("hybrid_pwift_"):
        neural_name = resolved_matcher.replace("hybrid_pwift_", "")
        pw_matcher = get_matcher("pwift", cfg)
        try:
            pw_res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
        finally:
            del pw_matcher

        neural_res = None
        n_matcher = None
        try:
            n_matcher = get_matcher(neural_name, cfg)
            neural_res = n_matcher.match(src_scaled, ref.data)
        except Exception as e:
            warnings.warn(f"Neural matcher '{neural_name}' unavailable ({e}); falling back to PWIFT only.")
            contingency_fallback = {
                "triggered": True,
                "original_matcher": resolved_matcher,
                "fallback_matcher": "pwift",
                "reason": str(e),
            }
        finally:
            if n_matcher is not None:
                del n_matcher
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        if neural_res is not None and len(neural_res.pts_src) >= 4:
            fused_res = fuse_pwift_neural(
                pw_res, neural_res,
                gsd_ref=ref.gsd_m or 1.0,
                alpha=cfg.fusion_alpha,
                beta=cfg.fusion_beta,
                pwift_n_min=cfg.pwift_min_quality_inliers,
                pwift_c_min=cfg.pwift_min_quality_cells,
                pwift_r_min=cfg.pwift_min_quality_ratio,
                pwift_rmse_max_px=cfg.pwift_max_quality_rmse_px,
                src_illum=src_illum,
                ref_illum=ref_illum,
                verify_photometric=cfg.fusion_verify_photometric,
                patch_radius=cfg.fusion_verification_patch_r,
                min_energy_thresh=cfg.fusion_verification_min_energy,
                reject_thresh=cfg.fusion_verification_reject_thresh,
            )
            results_to_evaluate.extend([fused_res, neural_res, pw_res])
        else:
            if neural_res is not None and len(neural_res.pts_src) < 4 and not contingency_fallback["triggered"]:
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": f"Neural matcher '{neural_name}' yielded < 4 matches; degraded to PWIFT only.",
                }
            results_to_evaluate.append(pw_res)
    else:
        m = None
        try:
            m = get_matcher(resolved_matcher, cfg)
            res = m.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
            if (res is None or len(res.pts_src) < 4) and resolved_matcher != "pwift":
                warnings.warn(
                    f"Matcher '{resolved_matcher}' returned insufficient matches "
                    f"({len(res.pts_src) if res is not None else 0} < 4); "
                    "triggering contingency fallback to PWIFT standalone."
                )
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": f"Insufficient matches from {resolved_matcher} (< 4 points)",
                }
                pw_matcher = get_matcher("pwift", cfg)
                try:
                    res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
                finally:
                    del pw_matcher
            if res is not None:
                results_to_evaluate.append(res)
        except Exception as e:
            if resolved_matcher != "pwift":
                warnings.warn(
                    f"Matcher '{resolved_matcher}' failed ({e}); triggering contingency fallback to PWIFT standalone."
                )
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": str(e),
                }
                pw_matcher = get_matcher("pwift", cfg)
                try:
                    res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
                    results_to_evaluate.append(res)
                finally:
                    del pw_matcher
            else:
                raise
        finally:
            if m is not None:
                del m
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

    results_by_method = {}
    competition_pool = []

    for match_result in results_to_evaluate:
        if match_result is None or len(match_result.pts_src) < 4:
            continue

        # ---- Stage 4: viewpoint (homography + RANSAC) ----
        hom_result = estimate_viewpoint_transform(
            match_result.pts_src, match_result.pts_dst, src.sensor,
            image_shape=src_scaled.shape, cfg=cfg,
        )
        H_or_local_results = hom_result if isinstance(hom_result, list) else hom_result.H
        inlier_mask = hom_result.inlier_mask if not isinstance(hom_result, list) else np.zeros(
            len(match_result.pts_src), dtype=bool)
        if isinstance(hom_result, list):
            for r in hom_result:
                inlier_mask |= r.inlier_mask

        # ---- Eq 26-27: explicit homography-based reprojection cleanup ----
        tau_e = cfg.reprojection_cleanup_tau_e_px
        if isinstance(hom_result, list):
            clean_mask = np.zeros(len(match_result.pts_src), dtype=bool)
            for r in hom_result:
                if r.H is None or not np.any(r.inlier_mask):
                    continue
                idx = np.nonzero(r.inlier_mask)[0]
                clean = reprojection_cleanup(match_result.pts_src[idx], match_result.pts_dst[idx], r.H, tau_e)
                clean_mask[idx[clean]] = True
            inlier_mask = inlier_mask & clean_mask
        elif hom_result.H is not None:
            clean = reprojection_cleanup(match_result.pts_src, match_result.pts_dst, hom_result.H, tau_e)
            inlier_mask = inlier_mask & clean

        primary_H = hom_result.H if not isinstance(hom_result, list) else (hom_result[0].H if hom_result else None)

        metrics = compute_metrics(
            match_result.method, match_result.pts_src, match_result.pts_dst,
            inlier_mask, H_or_local_results, image_shape=src_scaled.shape, grid=cfg.uniformity_grid,
        )

        # Structural fit and non-rigid DOF residual for rigid competition
        s_ncc = compute_structural_ncc(src_scaled, ref.data, primary_H) if primary_H is not None else 0.0
        coverage = float(metrics.uniformity_score)
        dof_resid = 0.0
        if primary_H is not None and len(match_result.pts_src) > 0:
            proj = cv2.perspectiveTransform(
                match_result.pts_src.reshape(-1, 1, 2).astype(np.float32), primary_H.astype(np.float32)
            ).reshape(-1, 2)
            dof_resid = float(np.median(np.abs(proj - match_result.pts_dst)))

        candidate_entry = {
            "method": match_result.method,
            "match_result": match_result,
            "hom_result": hom_result,
            "inlier_mask": inlier_mask,
            "metrics": metrics,
            "warp": primary_H if primary_H is not None else np.eye(3),
            "fit": s_ncc,
            "coverage": coverage,
            "dof_resid": dof_resid,
        }
        results_by_method[match_result.method] = candidate_entry
        competition_pool.append(candidate_entry)

    if not results_by_method:
        raise RuntimeError(
            f"No usable matches from {resolved_matcher} (requested: {matcher}). Check input images / thresholds."
        )

    # Rigid-only hypothesis competition (§4)
    comp_res = compete_rigid(
        competition_pool,
        terrain_spacing_px=cfg.rigid_terrain_spacing_px,
        lambda_dof=cfg.rigid_lambda_dof,
        margin_min=cfg.rigid_margin_min,
    )
    best_method = comp_res["winner"]["method"]
    best = results_by_method[best_method]

    # ---- Orthogonal Verification Gate (gate_cheap) ----
    primary_H = best["hom_result"].H if not isinstance(best["hom_result"], list) else (
        best["hom_result"][0].H if best["hom_result"] else None
    )
    ortho_eval = None
    if cfg.enable_orthogonal_gate and primary_H is not None:
        ortho_eval = gate_cheap(
            primary_H,
            src_scaled,
            ref.data,
            gsd_src=src.gsd_m or 1.0,
            gsd_ref=ref.gsd_m or 1.0,
            t_struct=cfg.orthogonal_gate_t_struct,
            tau_agree=cfg.orthogonal_gate_tau_agree,
            k_sigma=cfg.orthogonal_gate_k_sigma,
        )
        if not ortho_eval.get("pass", False):
            warnings.warn(
                f"Orthogonal verification gate flagged winning transform: {ortho_eval.get('reason')}. "
                "Proceeding with flagged confidence."
            )

    # ---- Stage 5: MiHo Piecewise Geometry + 6x6 Gridded GCP Optimizer (§2) ----
    miho_out = miho_plus_gcp(primary_H, best["match_result"], grid_size=cfg.miho_grid_size, target_gcps=cfg.miho_target_gcps)

    if not isinstance(best["hom_result"], list):
        best["hom_result"].Hs_local = miho_out.get("Hs_local")
        best["hom_result"].gcps = miho_out.get("gcps")

    # ---- Stage 5.5: Cause-Branched Subpixel Refinement (§5) ----
    src_inc_mean = float(np.nanmean(src_incidence_scaled)) if src_incidence_scaled is not None else 0.0
    ref_inc_mean = float(np.nanmean(ref_incidence)) if ref_incidence is not None else 0.0
    illum_delta_deg = abs(src_inc_mean - ref_inc_mean)
    subpixel_refine_out = refine_tile(src_scaled, ref.data, illum_delta_deg=illum_delta_deg)

    # ---- Stage 6: georeferencing and output ----
    registered = register_image(src_scaled, ref.data.shape, best["hom_result"])
    outputs = write_outputs(
        out_dir, tag=os.path.splitext(os.path.basename(source_path))[0],
        registered_img=registered,
        pts_src=best["match_result"].pts_src, pts_dst=best["match_result"].pts_dst,
        inlier_mask=best["inlier_mask"], method=best_method,
        ref_geotransform=ref.geotransform, ref_crs=ref.crs,
        src_img=src_scaled, ref_img=ref.data,
        gcps=miho_out.get("gcps", []),
        export_gcl_gcps_csv=cfg.export_gcl_gcps_csv,
    )

    summary = {
        "source": source_path, "reference": reference_path,
        "sensor": src.sensor.name, "matcher": matcher, "resolved_matcher": resolved_matcher,
        "best_method": best_method,
        "condition_routing": routing_info,
        "contingency_fallback": contingency_fallback,
        "orthogonal_gate": {
            "enabled": cfg.enable_orthogonal_gate,
            "passed": ortho_eval.get("pass", False) if ortho_eval is not None else True,
            "reason": ortho_eval.get("reason", "disabled") if ortho_eval is not None else "disabled",
            "cost_ms": ortho_eval.get("cost_ms", 0.0) if ortho_eval is not None else 0.0,
            "struct_ncc": ortho_eval.get("struct_ncc", 0.0) if ortho_eval is not None else 0.0,
        },
        "provenance": getattr(best["match_result"], "provenance", "direct"),
        "chosen_scale": best_scale, "chosen_rotation_deg": best_rot,
        "gsd_scale_prior": gsd_scale_prior,
        "rigid_competition": {
            "margin": comp_res.get("margin", 0.0),
            "ambiguous": comp_res.get("ambiguous", False),
        },
        "miho_gcps": {
            "count": len(miho_out.get("gcps", [])),
            "coverage": miho_out.get("coverage", 0.0),
            "csv_path": outputs.get("gcl_gcps_csv"),
            "gcps": miho_out.get("gcps", [])[:5],
        },
        "subpixel_refine": {
            "method": subpixel_refine_out.get("method"),
            "dx": subpixel_refine_out.get("dx"),
            "dy": subpixel_refine_out.get("dy"),
            "low_precision": subpixel_refine_out.get("low_precision", False),
        },
        "metrics": {
            m: {
                "n_matches": r["metrics"].n_matches, "n_inliers": r["metrics"].n_inliers,
                "inlier_ratio": r["metrics"].inlier_ratio, "rmse_px": r["metrics"].rmse_px,
                "uniformity_score": r["metrics"].uniformity_score,
            }
            for m, r in results_by_method.items()
        },
        "outputs": outputs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="CH2 <-> LROC lunar image registration pipeline")
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-sensor", default=None, choices=["OHRC", "TMC", "IIRS", "LROC"])
    parser.add_argument(
        "--matcher", default="auto",
        choices=["auto", "hybrid_pwift_roma2", "hybrid_pwift_eloftr", "roma2", "eloftr", "pwift"],
        help="Matcher model to use. Defaults to 'auto' for condition-based adaptive dispatch.",
    )
    parser.add_argument(
        "--real-time", "--descent-trn", action="store_true", dest="real_time",
        help="Enable real-time descent TRN mode (routes to EfficientLoFTR).",
    )
    parser.add_argument(
        "--polar-threshold-deg", type=float, default=70.0,
        help="Solar incidence angle threshold (deg) for extreme polar grazing illumination routing.",
    )
    parser.add_argument(
        "--disable-orthogonal-gate", action="store_true",
        help="Disable the orthogonal verification gate (gate_cheap).",
    )
    parser.add_argument("--roma2-weights", default=None, help="Path to fine-tuned RoMa v2 weights")
    parser.add_argument("--eloftr-ckpt", default=None, help="Path to fine-tuned EfficientLoFTR checkpoint")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--source-incidence", default=None)
    parser.add_argument("--source-emission", default=None)
    parser.add_argument("--source-phase", default=None)
    parser.add_argument("--source-nac-pho", default=None,
                         help="Path to the source image's LROC NAC_PHO photometry cube.")
    parser.add_argument("--reference-nac-pho", default=None,
                         help="Path to the reference image's LROC NAC_PHO photometry cube.")
    parser.add_argument("--nac-pho-band-phase", type=int, default=2)
    parser.add_argument("--nac-pho-band-emission", type=int, default=3)
    parser.add_argument("--nac-pho-band-incidence", type=int, default=4)
    parser.add_argument("--angles-from-label", action="store_true")
    parser.add_argument("--fetch-lroc-angles", action="store_true",
                         help="Auto-fetch incidence/emission/phase from the LROC ODE page.")
    parser.add_argument("--incidence-deg", type=float, default=None,
                         help="Manually supply the source image's incidence angle in degrees.")
    parser.add_argument("--emission-deg", type=float, default=None,
                         help="Manually supply the source image's emission angle in degrees.")
    parser.add_argument("--phase-deg", type=float, default=None,
                         help="Manually supply the source image's phase angle in degrees.")
    parser.add_argument(
        "--window", default=None,
        help="Backward-compatible shared x,y,w,h crop.",
    )
    parser.add_argument(
        "--source-window", default=None,
        help="Source-image crop x,y,w,h.",
    )
    parser.add_argument(
        "--reference-window", default=None,
        help="Reference-image crop x,y,w,h.",
    )
    parser.add_argument("--no-eloftr", action="store_true", help="Shorthand to run PWIFT only")
    parser.add_argument(
        "--fuse-pwift-eloftr",
        action="store_true",
        help="Shorthand for --matcher hybrid_pwift_eloftr",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        source_path=args.source, reference_path=args.reference, out_dir=args.out_dir,
        source_sensor=args.source_sensor,
        matcher=args.matcher,
        real_time=args.real_time,
        enable_orthogonal_gate=not args.disable_orthogonal_gate,
        polar_incidence_threshold_deg=args.polar_threshold_deg,
        roma2_weights=args.roma2_weights,
        eloftr_checkpoint=args.eloftr_ckpt,
        device=args.device,
        source_incidence_path=args.source_incidence, source_emission_path=args.source_emission,
        source_phase_path=args.source_phase,
        source_nac_pho_path=args.source_nac_pho, reference_nac_pho_path=args.reference_nac_pho,
        nac_pho_band_phase=args.nac_pho_band_phase, nac_pho_band_emission=args.nac_pho_band_emission,
        nac_pho_band_incidence=args.nac_pho_band_incidence,
        angles_from_label=args.angles_from_label,
        fetch_angles_online=args.fetch_lroc_angles,
        manual_incidence_deg=args.incidence_deg, manual_emission_deg=args.emission_deg,
        manual_phase_deg=args.phase_deg,
        window=_parse_window(args.window),
        source_window=_parse_window(args.source_window),
        reference_window=_parse_window(args.reference_window),
        use_eloftr=not args.no_eloftr,
        fuse_pwift_eloftr=args.fuse_pwift_eloftr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()