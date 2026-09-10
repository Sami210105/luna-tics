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
import json
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import cv2

from .config import PipelineConfig, get_sensor_config
from .preprocessing import load_image, load_angle_maps, LoadedImage
from .illumination import apply_illumination_correction
from .scale import select_best_scale, apply_scale, apply_rotation
from .matching import run_matching
from .viewpoint import estimate_viewpoint_transform, HomographyResult
from .georeference import register_image, write_outputs
from .metrics import compute_metrics
from .pwift import reprojection_cleanup


def _parse_window(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def filter_eloftr_with_pwift(pwift_result, eloftr_result, cfg: PipelineConfig):
    """Approach A: PWIFT-only geometry filters EfficientLoFTR, then fuse."""
    if pwift_result is None or eloftr_result is None:
        return None

    if len(pwift_result.pts_src) < 4 or len(eloftr_result.pts_src) == 0:
        return None

    prior_threshold = float(
        getattr(cfg, "pwift_eloftr_prior_threshold_px", 5.0)
    )
    dedupe_radius = float(
        getattr(cfg, "pwift_eloftr_dedupe_radius_px", 2.0)
    )

    # Preliminary model comes ONLY from PWIFT.
    H_pwift, pwift_inliers = cv2.findHomography(
        pwift_result.pts_src.astype(np.float32),
        pwift_result.pts_dst.astype(np.float32),
        cv2.RANSAC,
        float(cfg.ransac_reproj_threshold_px),
        maxIters=int(cfg.ransac_max_iters),
        confidence=float(cfg.ransac_confidence),
    )

    if H_pwift is None or pwift_inliers is None:
        warnings.warn("PWIFT -> LoFTR fusion skipped: no PWIFT homography.")
        return None

    if int(np.count_nonzero(pwift_inliers)) < 4:
        warnings.warn("PWIFT -> LoFTR fusion skipped: fewer than 4 PWIFT inliers.")
        return None

    src_l = eloftr_result.pts_src.astype(np.float32)
    dst_l = eloftr_result.pts_dst.astype(np.float32)

    predicted = cv2.perspectiveTransform(
        src_l.reshape(-1, 1, 2), H_pwift
    ).reshape(-1, 2)

    error = np.linalg.norm(predicted - dst_l, axis=1)
    keep = np.isfinite(error) & (error <= prior_threshold)

    n_prior = int(keep.sum())

    # Remove LoFTR correspondences already represented by PWIFT anchors.
    candidate_idx = np.flatnonzero(keep)
    if len(candidate_idx):
        src_delta = (
            src_l[candidate_idx, None, :]
            - pwift_result.pts_src[None, :, :]
        )
        dst_delta = (
            dst_l[candidate_idx, None, :]
            - pwift_result.pts_dst[None, :, :]
        )

        duplicate = (
            np.linalg.norm(src_delta, axis=2) <= dedupe_radius
        ) & (
            np.linalg.norm(dst_delta, axis=2) <= dedupe_radius
        )
        keep[candidate_idx] &= ~np.any(duplicate, axis=1)

    kept = np.flatnonzero(keep)

    fused_src = np.concatenate(
        [pwift_result.pts_src, src_l[kept]], axis=0
    ).astype(np.float32)
    fused_dst = np.concatenate(
        [pwift_result.pts_dst, dst_l[kept]], axis=0
    ).astype(np.float32)

    pwift_scores = (
        pwift_result.scores
        if pwift_result.scores is not None
        else np.ones(len(pwift_result.pts_src), dtype=np.float32)
    )
    loftr_scores = (
        eloftr_result.scores[kept]
        if eloftr_result.scores is not None
        else np.ones(len(kept), dtype=np.float32)
    )
    fused_scores = np.concatenate(
        [pwift_scores, loftr_scores], axis=0
    ).astype(np.float32)

    from .matching import MatchResult

    print(
        f"[FUSION] PWIFT anchors: {len(pwift_result.pts_src)} | "
        f"LoFTR candidates: {len(eloftr_result.pts_src)} | "
        f"within PWIFT prior: {n_prior} | "
        f"after dedupe: {len(kept)} | "
        f"fused: {len(fused_src)}"
    )

    return MatchResult(
        "pwift_eloftr_fused",
        fused_src,
        fused_dst,
        fused_scores,
    )


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
    use_eloftr: bool = True,
    fuse_pwift_eloftr: bool = False,
    cfg: Optional[PipelineConfig] = None,
) -> dict:
    cfg = cfg or PipelineConfig()

    # `--window` remains as a backward-compatible shorthand for a shared
    # crop. For two independently acquired LROC NAC observations, use
    # source_window/reference_window: the same lunar terrain generally has
    # different pixel coordinates in the two images.
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

    # ---- BUGFIX (this revision): fail fast and clearly on oversized images ----
    # pwift.py's photometric_weighted_structural_maps (the PWIFT branch used
    # for OHRC/LROC) allocates several full-resolution float64/complex128
    # buffers per (scale, orientation) pass - up to
    # cfg.pwift_scales * cfg.pwift_orientations of them. Running this on a
    # raw, uncropped LROC NAC strip (tens of millions of pixels) can require
    # 60-100+ GB of RAM and previously crashed with a bare
    # numpy.core._exceptions._ArrayMemoryError several calls deep inside
    # pwift.py, with no indication of *why* or what to do about it. The
    # paper's own benchmark only ever runs PWIFT on 512x512 patches (see
    # PWIFT.pdf Sec 4.1) - this is a --window problem, not a bug in the
    # matching code, so surface it as one clearly instead of letting the
    # allocation itself be the first sign anything is wrong.
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
                    "to crop the common geographic overlap "
                    "overlap region (roughly 512x512 up to ~2048x2048, "
                    "depending on available RAM) before matching - see "
                    "preview_overlap.py to find good coordinates for your "
                    "specific image pair."
                )

    src_incidence, src_emission, src_phase = src.incidence_deg, src.emission_deg, src.phase_deg
    if source_incidence_path and source_emission_path:
        src_incidence, src_emission, src_phase = load_angle_maps(
            source_incidence_path, source_emission_path,
            source_phase_path or source_emission_path, window=source_window,
        )
    # The reference image (LROC, illumination_method="pwift_akimov" same as
    # OHRC) needs its own photometric weighting too - PWIFT's structural
    # maps (Sec 3.2-3.3) are built independently per image. Previously only
    # the source ever received incidence/emission/phase, so the reference
    # silently ran unweighted PWIFT even when `--reference-nac-pho` (or any
    # other angle source) was available for it.
    ref_incidence, ref_emission, ref_phase = ref.incidence_deg, ref.emission_deg, ref.phase_deg

    # ---- Stage 5a: pick a starting scale/rotation before matching ----
    best_scale, best_rot = select_best_scale(src.data, ref.data, src.sensor, cfg)
    src_scaled = apply_scale(src.data, best_scale)
    src_scaled = apply_rotation(src_scaled, best_rot)
    src_incidence_scaled = apply_scale(src_incidence, best_scale) if src_incidence is not None else None
    src_incidence_scaled = apply_rotation(src_incidence_scaled, best_rot) if src_incidence_scaled is not None else None
    src_emission_scaled = apply_scale(src_emission, best_scale) if src_emission is not None else None
    src_emission_scaled = apply_rotation(src_emission_scaled, best_rot) if src_emission_scaled is not None else None
    src_phase_scaled = apply_scale(src_phase, best_scale) if src_phase is not None else None
    src_phase_scaled = apply_rotation(src_phase_scaled, best_rot) if src_phase_scaled is not None else None

    # ---- Stage 2: illumination correction (per-sensor branch) ----
    # Returns a pwift.PWIFTMaps bundle for OHRC/LROC (Sec 3.2-3.3 of
    # PWIFT.pdf), or a plain ndarray for TMC/IIRS - matching.py dispatches
    # on which one it got.
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

    # ---- Stage 3: matching (PWIFT + EfficientLoFTR) ----
    pwift_result, eloftr_result = run_matching(
        src_illum, src_scaled, ref_illum, ref.data, cfg, use_eloftr=use_eloftr,
    )

    # Approach A: use PWIFT geometry to filter LoFTR, then fuse.
    fused_result = None
    if fuse_pwift_eloftr and eloftr_result is not None:
        fused_result = filter_eloftr_with_pwift(
            pwift_result, eloftr_result, cfg
        )

    results_by_method = {}
    for match_result in [pwift_result, eloftr_result, fused_result]:
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
        # Tightens (never loosens) the FSC inlier mask above - a match must
        # be both an FSC inlier AND within tau_e of its own block's/the
        # global H's reprojection to survive.
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

        metrics = compute_metrics(
            match_result.method, match_result.pts_src, match_result.pts_dst,
            inlier_mask, H_or_local_results, image_shape=src_scaled.shape, grid=cfg.uniformity_grid,
        )
        results_by_method[match_result.method] = {
            "match_result": match_result, "hom_result": hom_result,
            "inlier_mask": inlier_mask, "metrics": metrics,
        }

    if not results_by_method:
        raise RuntimeError(
            "No usable matches from either PWIFT or EfficientLoFTR. Check "
            "input images / thresholds (config.py: pwift_bg_threshold, "
            "eloftr_confidence_threshold)."
        )

    best_method = max(results_by_method, key=lambda m: results_by_method[m]["metrics"].n_inliers)
    best = results_by_method[best_method]

    # ---- Stage 6: georeferencing and output ----
    registered = register_image(src_scaled, ref.data.shape, best["hom_result"])
    outputs = write_outputs(
        out_dir, tag=os.path.splitext(os.path.basename(source_path))[0],
        registered_img=registered,
        pts_src=best["match_result"].pts_src, pts_dst=best["match_result"].pts_dst,
        inlier_mask=best["inlier_mask"], method=best_method,
        ref_geotransform=ref.geotransform, ref_crs=ref.crs,
        src_img=src_scaled, ref_img=ref.data,
    )

    summary = {
        "source": source_path, "reference": reference_path,
        "sensor": src.sensor.name, "chosen_scale": best_scale, "chosen_rotation_deg": best_rot,
        "best_method": best_method,
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
    parser.add_argument("--source-incidence", default=None)
    parser.add_argument("--source-emission", default=None)
    parser.add_argument("--source-phase", default=None)
    parser.add_argument("--source-nac-pho", default=None,
                         help="Path to the source image's LROC NAC_PHO photometry "
                              "cube (or a GeoTIFF exported from one). Reads real "
                              "pixel-wise incidence/emission/phase from its angle "
                              "bands - see preprocessing.load_angles_from_nac_pho. "
                              "Takes priority over --incidence-deg/--angles-from-label.")
    parser.add_argument("--reference-nac-pho", default=None,
                         help="Same as --source-nac-pho, but for the reference "
                              "(LROC) image. The reference also uses PWIFT's "
                              "photometric weighting, so it benefits from its own "
                              "NAC_PHO angle maps just like the source does.")
    parser.add_argument("--nac-pho-band-phase", type=int, default=2)
    parser.add_argument("--nac-pho-band-emission", type=int, default=3)
    parser.add_argument("--nac-pho-band-incidence", type=int, default=4)
    parser.add_argument("--angles-from-label", action="store_true")
    parser.add_argument("--fetch-lroc-angles", action="store_true",
                         help="Auto-fetch incidence/emission/phase from the LROC ODE "
                              "product page (data.lroc.im-ldi.com) by product ID derived "
                              "from --source's filename. Requires `pip install requests` "
                              "and internet access. Overridden by --incidence-deg/etc if given.")
    parser.add_argument("--incidence-deg", type=float, default=None,
                         help="Manually supply the source image's incidence angle in "
                              "degrees (e.g. from the LROC ODE page or a paper's Table 1). "
                              "Takes priority over --fetch-lroc-angles/--angles-from-label.")
    parser.add_argument("--emission-deg", type=float, default=None,
                         help="Manually supply the source image's emission angle in degrees.")
    parser.add_argument("--phase-deg", type=float, default=None,
                         help="Manually supply the source image's phase angle in degrees (optional).")
    parser.add_argument(
        "--window", default=None,
        help="Backward-compatible shared x,y,w,h crop. Prefer separate "
             "--source-window and --reference-window for LROC/LROC pairs.",
    )
    parser.add_argument(
        "--source-window", default=None,
        help="Source-image crop x,y,w,h (recommended for geometry-selected LROC overlap).",
    )
    parser.add_argument(
        "--reference-window", default=None,
        help="Reference-image crop x,y,w,h (recommended for geometry-selected LROC overlap).",
    )
    parser.add_argument("--no-eloftr", action="store_true")
    parser.add_argument(
        "--fuse-pwift-eloftr",
        action="store_true",
        help="Approach A: filter EfficientLoFTR using a PWIFT-only "
             "preliminary homography, then fuse surviving LoFTR matches "
             "with PWIFT anchors before final RANSAC.",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        source_path=args.source, reference_path=args.reference, out_dir=args.out_dir,
        source_sensor=args.source_sensor,
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