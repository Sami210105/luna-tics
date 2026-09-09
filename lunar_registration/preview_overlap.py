"""
preview_overlap.py
==================

Geometry-aware overlap preview for LROC NAC_PHO PDS3 products.

The LROC NAC_PHO products are map-projected equirectangular images and their
PDS3 labels contain the geographic footprint of the raster. This script uses
that metadata instead of SIFT/feature matching to find the common geographic
area between two observations.

IMPORTANT:
    The windows printed by this script are WINDOWS IN THE NAC_PHO PRODUCT
    PIXEL SPACE. NAC_PHO is map-projected and can have a different resolution
    and extent from the raw *_RE.IMG EDR. Therefore these coordinates must NOT
    be blindly passed as --window to a raw EDR pipeline.

For the preview we use NAC_PHO Band 1 (I/F), because it is on the same
map-projected grid described by the geographic metadata.

The script does NOT load the entire multi-gigabyte NAC_PHO file into RAM.
It uses numpy.memmap and samples only the pixels needed for the preview.

Usage:
    python -m lunar_registration.preview_overlap \
        --source-nac-pho data/NAC_PHO_E010N0230_M129133239R.IMG \
        --reference-nac-pho data/NAC_PHO_E010N0230_M150368601R.IMG \
        --out overlap_preview.png

Optional:
    --margin-px 128
    --max-preview-height 1200
    --band 1
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class GeoRaster:
    path: str
    record_bytes: int
    lines: int
    samples: int
    bands: int
    dtype: np.dtype
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    map_resolution: float
    map_scale_m_per_px: float
    center_latitude: float
    center_longitude: float
    line_projection_offset: float
    sample_projection_offset: float
    data_offset: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.lines, self.samples


def _read_label(path: str) -> str:
    """Read only the ASCII PDS3 label, never the multi-GB raster."""
    with open(path, "rb") as f:
        head = f.read(100_000)

    text = head.decode("latin-1", errors="ignore")

    def get_int(name: str):
        m = re.search(rf"\b{name}\s*=\s*(\d+)", text, re.I)
        return int(m.group(1)) if m else None

    record_bytes = get_int("RECORD_BYTES")
    label_records = get_int("LABEL_RECORDS")

    if record_bytes is None or label_records is None:
        raise ValueError(
            f"{path}: could not find RECORD_BYTES/LABEL_RECORDS in PDS3 label."
        )

    label_size = record_bytes * label_records

    with open(path, "rb") as f:
        return f.read(label_size).decode("latin-1", errors="ignore")


def _get_float(text: str, name: str) -> float:
    m = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        text,
        re.I | re.M,
    )
    if not m:
        raise ValueError(f"Missing {name} in NAC_PHO label.")
    return float(m.group(1))


def _get_int(text: str, name: str) -> int:
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*(\d+)", text, re.I | re.M)
    if not m:
        raise ValueError(f"Missing {name} in NAC_PHO label.")
    return int(m.group(1))


def read_nac_pho_metadata(path: str) -> GeoRaster:
    text = _read_label(path)

    record_bytes = _get_int(text, "RECORD_BYTES")
    label_records = _get_int(text, "LABEL_RECORDS")
    lines = _get_int(text, "LINES")
    samples = _get_int(text, "LINE_SAMPLES")
    bands = _get_int(text, "BANDS")

    sample_type_m = re.search(
        r"^\s*SAMPLE_TYPE\s*=\s*([A-Za-z0-9_]+)", text, re.I | re.M
    )
    sample_type = sample_type_m.group(1).upper() if sample_type_m else ""

    if sample_type == "PC_REAL":
        dtype = np.dtype("<f4")
    elif sample_type in ("REAL", "IEEE_REAL"):
        dtype = np.dtype(">f4")
    else:
        raise ValueError(
            f"{path}: unsupported SAMPLE_TYPE={sample_type!r}. "
            "Expected a 32-bit real NAC_PHO product."
        )

    min_lat = _get_float(text, "MINIMUM_LATITUDE")
    max_lat = _get_float(text, "MAXIMUM_LATITUDE")
    west_lon = _get_float(text, "WESTERNMOST_LONGITUDE")
    east_lon = _get_float(text, "EASTERNMOST_LONGITUDE")
    map_resolution = _get_float(text, "MAP_RESOLUTION")
    map_scale = _get_float(text, "MAP_SCALE")
    center_latitude = _get_float(text, "CENTER_LATITUDE")
    center_longitude = _get_float(text, "CENTER_LONGITUDE")
    line_projection_offset = _get_float(text, "LINE_PROJECTION_OFFSET")
    sample_projection_offset = _get_float(text, "SAMPLE_PROJECTION_OFFSET")

    expected_record_bytes = samples * dtype.itemsize
    if record_bytes != expected_record_bytes:
        raise ValueError(
            f"{path}: RECORD_BYTES={record_bytes}, but "
            f"LINE_SAMPLES*4={expected_record_bytes}."
        )

    return GeoRaster(
        path=str(path),
        record_bytes=record_bytes,
        lines=lines,
        samples=samples,
        bands=bands,
        dtype=dtype,
        min_lat=min_lat,
        max_lat=max_lat,
        west_lon=west_lon,
        east_lon=east_lon,
        map_resolution=map_resolution,
        map_scale_m_per_px=map_scale,
        center_latitude=center_latitude,
        center_longitude=center_longitude,
        line_projection_offset=line_projection_offset,
        sample_projection_offset=sample_projection_offset,
        data_offset=record_bytes * label_records,
    )


def print_metadata(tag: str, g: GeoRaster) -> None:
    print(f"\n[{tag}] {g.path}")
    print(f"  shape                 : ({g.lines}, {g.samples})")
    print(f"  bands                 : {g.bands}")
    print("  map projection        : equirectangular / planetocentric")
    print(f"  latitude              : {g.min_lat:.12f} .. {g.max_lat:.12f} deg")
    print(f"  longitude             : {g.west_lon:.12f} .. {g.east_lon:.12f} deg")
    print(f"  map resolution        : {g.map_resolution:.6f} px/deg")
    print(f"  map scale             : {g.map_scale_m_per_px:.3f} m/px")
    print(f"  projection center     : ({g.center_latitude:.6f}, {g.center_longitude:.6f}) deg")
    print(f"  line/sample offsets   : ({g.line_projection_offset:.3f}, {g.sample_projection_offset:.3f}) px")


def geographic_overlap(a: GeoRaster, b: GeoRaster):
    min_lat = max(a.min_lat, b.min_lat)
    max_lat = min(a.max_lat, b.max_lat)
    west_lon = max(a.west_lon, b.west_lon)
    east_lon = min(a.east_lon, b.east_lon)

    if min_lat >= max_lat or west_lon >= east_lon:
        return None

    return min_lat, max_lat, west_lon, east_lon


def geo_to_pixel(g: GeoRaster, lat: float, lon: float) -> Tuple[float, float]:
    """Convert planetocentric lat/lon to LROC NAC_PHO pixel coordinates.

    Uses the actual LROC equirectangular projection parameters from the PDS
    label instead of linearly stretching the footprint bounding box.

    LROC convention (pixel centers are 1-based):
        line   = 1 + LINE_PROJECTION_OFFSET - lat * MAP_RESOLUTION
        sample = 1 + SAMPLE_PROJECTION_OFFSET
                 + dlon * MAP_RESOLUTION * cos(CENTER_LATITUDE)

    The returned coordinates are zero-based numpy coordinates.
    """
    lat_rad = np.deg2rad(g.center_latitude)
    dlon = (lon - g.center_longitude + 180.0) % 360.0 - 180.0

    line_one_based = (
        1.0 + g.line_projection_offset - lat * g.map_resolution
    )
    sample_one_based = (
        1.0
        + g.sample_projection_offset
        + dlon * g.map_resolution * np.cos(lat_rad)
    )

    return sample_one_based - 1.0, line_one_based - 1.0


def overlap_window(
    g: GeoRaster,
    overlap,
    margin_px: int,
) -> Tuple[int, int, int, int]:
    min_lat, max_lat, west_lon, east_lon = overlap

    x0f, y0f = geo_to_pixel(g, max_lat, west_lon)
    x1f, y1f = geo_to_pixel(g, min_lat, east_lon)

    x0 = max(0, int(np.floor(min(x0f, x1f))) - margin_px)
    y0 = max(0, int(np.floor(min(y0f, y1f))) - margin_px)
    x1 = min(g.samples, int(np.ceil(max(x0f, x1f))) + 1 + margin_px)
    y1 = min(g.lines, int(np.ceil(max(y0f, y1f))) + 1 + margin_px)

    return x0, y0, x1 - x0, y1 - y0


def _band_memmap(g: GeoRaster, band: int):
    if not 1 <= band <= g.bands:
        raise ValueError(f"{g.path}: band must be 1..{g.bands}, got {band}")

    # BAND_SEQUENTIAL:
    # band 1 occupies lines*record_bytes bytes, followed by band 2, etc.
    band_offset = g.data_offset + (band - 1) * g.lines * g.record_bytes

    return np.memmap(
        g.path,
        dtype=g.dtype,
        mode="r",
        offset=band_offset,
        shape=(g.lines, g.samples),
        order="C",
    )


def _normalise_u8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(a)
    finite &= np.abs(a) < 1e10

    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)

    # Use float64 for normalization so large finite special-pixel values do
    # not overflow float32 arithmetic.
    vals = a[finite].astype(np.float64)
    lo, hi = np.percentile(vals, (1.0, 99.0))

    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())

    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)

    out = (a.astype(np.float64) - lo) / (hi - lo)
    out[~finite] = 0.0
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def read_preview_window(
    g: GeoRaster,
    window: Tuple[int, int, int, int],
    band: int,
    max_height: int,
) -> np.ndarray:
    """
    Read a downsampled view directly from the memmapped band.

    No multi-GB array is created.
    """
    x, y, w, h = window
    step = max(1, int(np.ceil(h / max_height)))

    mm = _band_memmap(g, band)
    arr = np.asarray(mm[y:y + h:step, x:x + w:step])

    return _normalise_u8(arr)


def make_side_by_side(src: np.ndarray, ref: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(src.shape[0], ref.shape[0])
    canvas = np.zeros((h, src.shape[1] + gap + ref.shape[1]), dtype=np.uint8)
    canvas[:src.shape[0], :src.shape[1]] = src
    canvas[:ref.shape[0], src.shape[1] + gap:] = ref
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--source-nac-pho", required=True)
    ap.add_argument("--reference-nac-pho", required=True)

    ap.add_argument("--out", default="overlap_preview.png")

    ap.add_argument(
        "--margin-px",
        type=int,
        default=128,
        help="margin around the geographic overlap in each NAC_PHO pixel grid "
             "(default: 128)",
    )

    ap.add_argument(
        "--max-preview-height",
        type=int,
        default=1200,
        help="maximum preview height; default 1200",
    )

    ap.add_argument(
        "--band",
        type=int,
        default=1,
        help="NAC_PHO band to preview; default 1 = I/F",
    )

    args = ap.parse_args()

    src = read_nac_pho_metadata(args.source_nac_pho)
    ref = read_nac_pho_metadata(args.reference_nac_pho)

    print_metadata("SOURCE", src)
    print_metadata("REFERENCE", ref)

    overlap = geographic_overlap(src, ref)

    if overlap is None:
        print("\nNo geographic overlap found.")
        return

    min_lat, max_lat, west_lon, east_lon = overlap

    print("\n" + "=" * 72)
    print("GEOGRAPHIC OVERLAP")
    print("=" * 72)
    print(f"  latitude  : {min_lat:.12f} .. {max_lat:.12f} deg")
    print(f"  longitude : {west_lon:.12f} .. {east_lon:.12f} deg")

    src_window = overlap_window(src, overlap, args.margin_px)
    ref_window = overlap_window(ref, overlap, args.margin_px)

    print("\n" + "=" * 72)
    print("NAC_PHO PIXEL WINDOWS")
    print("=" * 72)
    print(f"  source    : x,y,w,h = {src_window}")
    print(f"  reference : x,y,w,h = {ref_window}")

    print(f"\nReading Band {args.band} preview samples directly from disk...")

    src_preview = read_preview_window(
        src, src_window, args.band, args.max_preview_height
    )
    ref_preview = read_preview_window(
        ref, ref_window, args.band, args.max_preview_height
    )

    canvas = make_side_by_side(src_preview, ref_preview)
    Image.fromarray(canvas, mode="L").save(args.out)

    print(f"\nSaved preview: {args.out}")
    print(
        "\nIMPORTANT: these windows are NAC_PHO pixel coordinates. "
        "NAC_PHO is map-projected at 0.4/0.5 m/px and can have a different "
        "extent/resolution from the raw *_RE.IMG EDR."
    )
    print(
        "Do not blindly pass these coordinates as raw EDR --window values. "
        "They are correct for the map-projected NAC_PHO grid."
    )


if __name__ == "__main__":
    main()
