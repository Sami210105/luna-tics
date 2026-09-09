"""
Diagnostic: runs each pipeline stage on your real images and prints stats at
every step, so we can see exactly where it goes wrong instead of just getting
"No usable matches" from the full pipeline.

Run: python diagnose.py --source data\\M129133239RE.IMG --source-sensor LROC \
    --reference data\\M150368601RE.IMG --angles-from-label \
    --window 2243,298,512,512

Paste the full printed output back.
"""

import argparse

import numpy as np

from lunar_registration.preprocessing import load_image
from lunar_registration.illumination import apply_illumination_correction
from lunar_registration.scale import select_best_scale, apply_scale
from lunar_registration.pwift import detect_keypoints_dual_channel, compute_descriptors, match_descriptors_swap_aware
from lunar_registration.config import PipelineConfig


def stats(name, arr):
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--source-sensor", default=None)
    p.add_argument("--angles-from-label", action="store_true")
    p.add_argument("--window", default=None, help="x,y,w,h")
    p.add_argument("--ratio-test", type=float, default=None,
                   help="Override PipelineConfig.pwift_ratio_test (default 0.85). "
                        "Higher = looser = more matches, more false positives.")
    p.add_argument("--keypoint-threshold", type=float, default=None,
                   help="Override PipelineConfig.pwift_keypoint_threshold (default 0.15). "
                        "Lower = more keypoints detected.")
    args = p.parse_args()

    window = None
    if args.window:
        x, y, w, h = (int(v) for v in args.window.split(","))
        window = (x, y, w, h)

    cfg = PipelineConfig()
    if args.ratio_test is not None:
        cfg.pwift_ratio_test = args.ratio_test
    if args.keypoint_threshold is not None:
        cfg.pwift_keypoint_threshold = args.keypoint_threshold
    print(f"Using pwift_ratio_test={cfg.pwift_ratio_test}, "
          f"pwift_keypoint_threshold={cfg.pwift_keypoint_threshold}")

    print("=== Loading images ===")
    src = load_image(args.source, sensor_hint=args.source_sensor, window=window,
                      angles_from_label=args.angles_from_label)
    ref = load_image(args.reference, sensor_hint="LROC", window=window)
    stats("src.data (normalized)", src.data)
    stats("ref.data (normalized)", ref.data)
    print(f"src sensor: {src.sensor.name}, ref sensor: {ref.sensor.name}")
    if src.incidence_deg is not None:
        print(f"src incidence_deg: {src.incidence_deg[0,0]:.2f}, "
              f"emission_deg: {src.emission_deg[0,0]:.2f}")
    else:
        print("src incidence/emission: None (label didn't have them, or --angles-from-label not set)")

    print("\n=== Stage 5a: coarse scale/rotation search ===")
    best_scale, best_rot = select_best_scale(src.data, ref.data, src.sensor, cfg)
    print(f"chosen_scale={best_scale:.4f}, chosen_rotation_deg={best_rot}")
    src_scaled = apply_scale(src.data, best_scale)
    stats("src_scaled", src_scaled)

    print("\n=== Stage 2: illumination correction (phase congruency) ===")
    src_illum = apply_illumination_correction(
        src_scaled, src.sensor, incidence_deg=None, emission_deg=None,
        n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations,
    )
    ref_illum = apply_illumination_correction(
        ref.data, ref.sensor, n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations,
    )
    stats("src_illum (PC map)", src_illum)
    stats("ref_illum (PC map)", ref_illum)

    print("\n=== Stage 3a: keypoint detection ===")
    kp_src = detect_keypoints_dual_channel(src_illum, threshold=cfg.pwift_keypoint_threshold)
    kp_ref = detect_keypoints_dual_channel(ref_illum, threshold=cfg.pwift_keypoint_threshold)
    print(f"n_keypoints src={len(kp_src)}, ref={len(kp_ref)} "
          f"(threshold={cfg.pwift_keypoint_threshold})")
    if kp_src:
        resp_src = [k.response for k in kp_src]
        print(f"  src response range: {min(resp_src):.4f} - {max(resp_src):.4f}")
    if kp_ref:
        resp_ref = [k.response for k in kp_ref]
        print(f"  ref response range: {min(resp_ref):.4f} - {max(resp_ref):.4f}")

    print("\n=== Stage 3b: descriptors ===")
    desc_src = compute_descriptors(src_scaled, src_illum, kp_src, patch_size=cfg.pwift_descriptor_patch)
    desc_ref = compute_descriptors(ref.data, ref_illum, kp_ref, patch_size=cfg.pwift_descriptor_patch)
    print(f"n_descriptors src={len(desc_src)}, ref={len(desc_ref)} "
          f"(dropped if too close to image edge for patch_size={cfg.pwift_descriptor_patch})")

    print("\n=== Stage 3c: matching ===")
    matches = match_descriptors_swap_aware(desc_src, desc_ref, ratio_test=cfg.pwift_ratio_test)
    print(f"n_matches={len(matches)} (ratio_test={cfg.pwift_ratio_test})")

    print("\n=== Diagnosis ===")
    if src.data.std() < 0.02 or ref.data.std() < 0.02:
        print("!! Very low pixel variance in src or ref - likely a blank/no-data region. "
              "Try a different --window or drop --window to use the full image.")
    if len(kp_src) == 0 or len(kp_ref) == 0:
        print("!! Zero keypoints detected - pwift_keypoint_threshold in config.py is too "
              "high for this image's phase-congruency response range shown above. Try "
              "lowering PipelineConfig.pwift_keypoint_threshold (e.g. to 0.05).")
    elif len(desc_src) == 0 or len(desc_ref) == 0:
        print("!! Keypoints found but all too close to the image edge for the descriptor "
              "patch - try a larger --window or smaller pwift_descriptor_patch.")
    elif len(matches) == 0:
        print("!! Descriptors computed but none passed the ratio test - try raising "
              "PipelineConfig.pwift_ratio_test (e.g. to 0.95, looser) or check whether "
              "chosen_scale/chosen_rotation above look plausible for your actual image pair.")
    elif len(matches) < 4:
        print(f"!! Only {len(matches)} match(es) - not enough for RANSAC (needs >=4). "
              "The descriptor is finding correspondences (good sign - not zero) but the "
              "ratio test is rejecting most of them. Try: --ratio-test 0.95 (looser) "
              "and/or --keypoint-threshold 0.08 (more keypoints to match from).")
    else:
        print(f"{len(matches)} matches found - the full pipeline should work with these "
              "settings; if it still errors, the issue is downstream (homography/RANSAC).")


if __name__ == "__main__":
    main()