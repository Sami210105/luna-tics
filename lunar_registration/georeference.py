"""
Stage 6 - Georeferencing and output
======================================
Warps the source image onto the reference frame using the estimated
homography (global, or piecewise-local blended), writes:
  - the registered image (PNG always; GeoTIFF too if the reference had a
    CRS/geotransform from rasterio and `rasterio` is installed)
  - a match-points CSV (source x,y / reference x,y / inlier flag / method)
  - a match-point visualization PNG (src|ref side by side, green/red lines -
    see visualize.py), when `src_img`/`ref_img` are provided

--- SWAP POINT ---
True selenoreferencing (assigning a real Moon body-fixed CRS / IAU2000
Moon geographic CRS to the output) requires the reference image's original
map projection info from ISIS (`campt`/`maplab`). If you have that, pass it
via `ref_crs`/`ref_geotransform` and this will tag the output GeoTIFF with it
through rasterio; otherwise the output stays in plain pixel coordinates.

--- BUGFIX (this revision) ---
`write_outputs` used to silently write out an all-black PNG with no
indication anything was wrong whenever every block homography came back
`None` (see viewpoint.py's fix + warning for the root cause). This now
checks the registered image's pixel variance and warns loudly if it's
degenerate, instead of leaving that to be discovered only by opening the
file.
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

from .viewpoint import HomographyResult
from .visualize import draw_match_points


def warp_global(src_img: np.ndarray, H: np.ndarray, dst_shape: Tuple[int, int]) -> np.ndarray:
    h, w = dst_shape
    warped = cv2.warpPerspective((src_img * 255).astype(np.uint8), H, (w, h))
    return warped.astype(np.float32) / 255.0


def warp_local(
    src_img: np.ndarray, block_results: List[HomographyResult], dst_shape: Tuple[int, int],
) -> np.ndarray:
    """Warps each block with its own homography into the destination frame
    and blends overlaps by simple averaging (weighted by a distance-to-edge
    feather to reduce seams)."""
    h, w = dst_shape
    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    sh, sw = src_img.shape
    n_valid_blocks = 0
    for res in block_results:
        if res.H is None or res.block is None:
            continue
        n_valid_blocks += 1
        bx, by, bw, bh = res.block
        bx1, by1 = min(bx + bw, sw), min(by + bh, sh)
        block_img = np.zeros_like(src_img)
        block_img[by:by1, bx:bx1] = src_img[by:by1, bx:bx1]

        warped = cv2.warpPerspective((block_img * 255).astype(np.uint8), res.H, (w, h)).astype(np.float32) / 255.0
        mask = cv2.warpPerspective(
            (np.ones_like(block_img) * 255).astype(np.uint8), res.H, (w, h)
        ).astype(np.float32) / 255.0
        feather = cv2.GaussianBlur(mask, (31, 31), 0)

        accum += warped * feather
        weight += feather

    if n_valid_blocks == 0:
        warnings.warn(
            "warp_local: every block homography was None - there is no "
            "valid transform for any part of the image, so the registered "
            "output will be entirely blank. See the warning from "
            "estimate_local_homographies for likely causes (too few/noisy "
            "matches for the block grid, or an unstable global fit)."
        )

    weight[weight < 1e-6] = 1.0
    return np.clip(accum / weight, 0.0, 1.0)


def register_image(
    src_img: np.ndarray, ref_shape: Tuple[int, int],
    homography_result: Union[HomographyResult, List[HomographyResult]],
) -> np.ndarray:
    if isinstance(homography_result, list):
        return warp_local(src_img, homography_result, ref_shape)
    if homography_result.H is None:
        raise RuntimeError("Homography estimation failed - no transform to warp with.")

    # If MiHo 2x2 local quadrant homographies exist, blend them across quadrants
    if getattr(homography_result, "Hs_local", None) and len(homography_result.Hs_local) == 4:
        sh, sw = src_img.shape[:2]
        hw, hh = sw // 2, sh // 2
        quad_blocks = [
            (0, 0, hw, hh),
            (hw, 0, sw - hw, hh),
            (0, hh, hw, sh - hh),
            (hw, hh, sw - hw, sh - hh),
        ]
        miho_results = []
        for q in range(4):
            H_q = homography_result.Hs_local.get(f"quad_{q}", homography_result.H)
            miho_results.append(HomographyResult(H=H_q, inlier_mask=homography_result.inlier_mask, block=quad_blocks[q]))
        return warp_local(src_img, miho_results, ref_shape)

    return warp_global(src_img, homography_result.H, ref_shape)


def _pixel_to_geo(px: float, py: float, transform=None, crs=None) -> Tuple[float, float]:
    """Converts pixel coordinate (px, py) to geographic/selenographic coordinate (lon, lat)."""
    if transform is not None:
        try:
            if hasattr(transform, "__mul__"):
                gx, gy = transform * (px, py)
                return float(gx), float(gy)
            elif isinstance(transform, (list, tuple)) and len(transform) == 6:
                c, a, b, f, d, e = transform
                gx = c + a * px + b * py
                gy = f + d * px + e * py
                return float(gx), float(gy)
        except Exception:
            pass
    return float(px), float(py)


def write_outputs(
    out_dir: str, tag: str,
    registered_img: np.ndarray,
    pts_src: np.ndarray, pts_dst: np.ndarray, inlier_mask: np.ndarray, method: str,
    ref_geotransform=None, ref_crs=None,
    src_img: Optional[np.ndarray] = None,
    ref_img: Optional[np.ndarray] = None,
    save_visualization: bool = True,
    gcps: Optional[List[Dict[str, Any]]] = None,
    export_gcl_gcps_csv: bool = True,
) -> dict:
    """`src_img`/`ref_img`, if provided, are the pre-warp images (at the
    resolution matching was run on - e.g. pipeline.py's `src_scaled` and
    `ref.data`) used only to render the match-point visualization PNG; they
    don't affect registration itself. Pass `save_visualization=False` to
    skip it.
    
    If `gcps` is provided and `export_gcl_gcps_csv` is True, exports
    `{tag}_gcl_gcps.csv` containing the Ground Control Lattice GCPs.
    """
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{tag}_registered.png")
    csv_path = os.path.join(out_dir, f"{tag}_matchpoints.csv")

    if registered_img.size and registered_img.std() < 1e-4:
        warnings.warn(
            f"write_outputs: '{png_path}' has essentially zero pixel "
            f"variance (std={registered_img.std():.2e}) - this almost "
            "certainly means registration failed (no valid homography "
            "anywhere) rather than a genuinely blank scene. Check the "
            "warnings from estimate_local_homographies/warp_local above."
        )

    from PIL import Image
    Image.fromarray((np.clip(registered_img, 0, 1) * 255).astype(np.uint8)).save(png_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "inlier", "method"])
        for (sx, sy), (dx, dy), inl in zip(pts_src, pts_dst, inlier_mask):
            writer.writerow([sx, sy, dx, dy, int(bool(inl)), method])

    outputs = {"registered_png": png_path, "matchpoints_csv": csv_path}

    if export_gcl_gcps_csv and gcps:
        gcp_csv_path = os.path.join(out_dir, f"{tag}_gcl_gcps.csv")
        with open(gcp_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gcp_id", "src_x", "src_y", "ref_x", "ref_y",
                "gcs_lon", "gcs_lat", "inlier", "cell_x", "cell_y", "confidence"
            ])
            for i, gcp in enumerate(gcps):
                pt_src = gcp.get("pt_src", [0.0, 0.0])
                pt_dst = gcp.get("pt_dst", [0.0, 0.0])
                sx, sy = float(pt_src[0]), float(pt_src[1])
                rx, ry = float(pt_dst[0]), float(pt_dst[1])
                glon, glat = _pixel_to_geo(rx, ry, ref_geotransform, ref_crs)

                inl = 1
                if "inlier" in gcp:
                    inl = int(bool(gcp["inlier"]))
                elif "residual" in gcp:
                    inl = 1 if float(gcp["residual"]) <= 3.0 else 0

                cell = gcp.get("cell", (0, 0))
                cx, cy = int(cell[0]), int(cell[1])
                conf = float(gcp.get("utility", gcp.get("score", 1.0)))

                writer.writerow([i, sx, sy, rx, ry, glon, glat, inl, cx, cy, conf])

        outputs["gcl_gcps_csv"] = gcp_csv_path

    if save_visualization and src_img is not None and ref_img is not None and len(pts_src):
        viz_path = os.path.join(out_dir, f"{tag}_matches.png")
        draw_match_points(
            src_img, ref_img, pts_src, pts_dst, inlier_mask, viz_path,
            title=f"{tag} ({method})",
        )
        outputs["matches_png"] = viz_path

    if ref_geotransform is not None and ref_crs is not None:
        try:
            import rasterio
            tif_path = os.path.join(out_dir, f"{tag}_registered.tif")
            h, w = registered_img.shape
            with rasterio.open(
                tif_path, "w", driver="GTiff", height=h, width=w, count=1,
                dtype="uint8", crs=ref_crs, transform=ref_geotransform,
            ) as dst:
                dst.write((np.clip(registered_img, 0, 1) * 255).astype(np.uint8), 1)
            outputs["registered_tif"] = tif_path
        except ImportError:
            pass  # PNG + CSV are still written; GeoTIFF is a bonus if rasterio's available

    return outputs