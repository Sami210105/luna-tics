"""
Match-point visualization: draws src/reference images side by side with
lines connecting matched points, green for inliers and red for outliers,
plus a match-count label - the "Registration Results" style comparison
figure (see reference screenshots this was matched to).

Not a pipeline stage; call `draw_match_points()` yourself, or leave it wired
into `georeference.write_outputs()` (default `save_visualization=True`),
which writes it alongside the registered PNG/CSV as `{tag}_matches.png`.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import cv2


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def draw_match_points(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    inlier_mask: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    divider_px: int = 6,
    max_lines: Optional[int] = 400,
    line_thickness: int = 1,
    point_radius: int = 3,
    show_outliers: bool = True,
) -> str:
    """Side-by-side src|ref visualization with match lines overlaid, styled
    after the reference "Registration Results" figure: green lines/points
    for inliers, red for outliers, black vertical divider between the two
    images, and a label with the match count.

    `pts_src`/`pts_dst` are (N,2) float (x,y) arrays as produced by
    matching.py; `inlier_mask` is the boolean array from the same stage
    (or from `viewpoint`/pipeline's combined mask). Points are drawn on the
    two images at their *original* (unscaled) coordinates, so pass the
    same `src_img`/`ref_img` resolution the matches were computed on
    (typically `src_scaled`/`ref.data` from pipeline.py).
    """
    src_bgr = _to_bgr_u8(src_img)
    ref_bgr = _to_bgr_u8(ref_img)

    h = max(src_bgr.shape[0], ref_bgr.shape[0])
    w_src, w_ref = src_bgr.shape[1], ref_bgr.shape[1]

    def pad(img, target_h):
        if img.shape[0] == target_h:
            return img
        out = np.zeros((target_h, img.shape[1], 3), dtype=np.uint8)
        out[: img.shape[0]] = img
        return out

    src_bgr = pad(src_bgr, h)
    ref_bgr = pad(ref_bgr, h)

    divider = np.zeros((h, divider_px, 3), dtype=np.uint8)
    canvas = np.concatenate([src_bgr, divider, ref_bgr], axis=1)
    x_offset = w_src + divider_px

    n = len(pts_src)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    idx_inliers = np.nonzero(inlier_mask)[0]
    idx_outliers = np.nonzero(~inlier_mask)[0] if show_outliers else np.array([], dtype=int)

    # If there are a lot of matches, subsample for legibility rather than
    # drawing an unreadable tangle of lines - inliers are prioritized since
    # they're the ones worth actually looking at.
    def _subsample(idx, budget):
        if max_lines is None or len(idx) <= budget:
            return idx
        chosen = np.linspace(0, len(idx) - 1, budget).astype(int)
        return idx[chosen]

    if max_lines is not None:
        idx_inliers = _subsample(idx_inliers, max_lines)
        idx_outliers = _subsample(idx_outliers, max(0, max_lines - len(idx_inliers)) // 2)

    GREEN = (60, 200, 60)
    RED = (50, 50, 220)

    for idx_set, color in ((idx_outliers, RED), (idx_inliers, GREEN)):
        for i in idx_set:
            p1 = (int(round(pts_src[i, 0])), int(round(pts_src[i, 1])))
            p2 = (int(round(pts_dst[i, 0])) + x_offset, int(round(pts_dst[i, 1])))
            cv2.line(canvas, p1, p2, color, line_thickness, cv2.LINE_AA)
            cv2.circle(canvas, p1, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, point_radius, color, -1, cv2.LINE_AA)

    n_inliers = int(inlier_mask.sum())
    label = f"Inliers: {n_inliers}/{n}  ({(n_inliers / n * 100 if n else 0):.0f}%)"
    if title:
        label = f"{title}   {label}"

    text_h = 34
    banner = np.full((text_h, canvas.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(banner, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.concatenate([banner, canvas], axis=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)
    return out_path