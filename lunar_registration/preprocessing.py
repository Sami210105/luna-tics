"""
Stage 1 - Preprocessing
=======================

Supports:

- PDS3 .IMG images
- LROC NAC_PHO multi-band .IMG products
- ISIS / GeoTIFF / PNG / JPG
- Pixel-wise NAC_PHO incidence / emission / phase angle maps
- Manual / label / online angle sources
- Image normalization
- Windowed reading/cropping
- GSD resampling

NAC_PHO convention used here:

    Band 1 = calibrated image
    Band 2 = phase angle
    Band 3 = emission angle
    Band 4 = incidence angle

The important addition is that a NAC_PHO .IMG can now be used as the
ACTUAL image input, not merely as an angle source.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import SensorConfig, get_sensor_config


# ----------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ----------------------------------------------------------------------
# Loaded image container
# ----------------------------------------------------------------------

@dataclass
class LoadedImage:
    data: np.ndarray
    path: str
    sensor: SensorConfig

    incidence_deg: Optional[np.ndarray] = None
    emission_deg: Optional[np.ndarray] = None
    phase_deg: Optional[np.ndarray] = None

    gsd_m: Optional[float] = None

    geotransform: Optional[tuple] = None
    crs: Optional[object] = None


# ----------------------------------------------------------------------
# Sensor detection
# ----------------------------------------------------------------------

def detect_sensor(path: str, label_text: str = "") -> SensorConfig:
    """
    Guess sensor from filename + optional label text.
    """

    haystack = (os.path.basename(path) + " " + label_text).upper()

    for name, cfg in {
        "OHRC": get_sensor_config("OHRC"),
        "TMC": get_sensor_config("TMC"),
        "IIRS": get_sensor_config("IIRS"),
        "LROC": get_sensor_config("LROC"),
    }.items():

        for kw in cfg.label_keywords:
            if kw in haystack:
                return cfg

    warnings.warn(
        f"Could not auto-detect sensor for '{path}'. "
        "Defaulting to OHRC."
    )

    return get_sensor_config("OHRC")


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to float32 [0, 1].

    Uses robust 0.5 / 99.5 percentiles.
    """

    img = img.astype(np.float32)

    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(finite, (0.5, 99.5))

    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(
        (img - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )

    return img.astype(np.float32)


# ----------------------------------------------------------------------
# PDS3 label helpers
# ----------------------------------------------------------------------

def _find_pds3_label_size(path: str) -> int:
    """
    Find PDS3 ASCII label size using:

        RECORD_BYTES * LABEL_RECORDS
    """

    with open(path, "rb") as f:
        header = f.read(20000).decode(
            "latin-1",
            errors="ignore",
        )

    record_match = re.search(
        r"RECORD_BYTES\s*=\s*(\d+)",
        header,
        re.IGNORECASE,
    )

    label_match = re.search(
        r"LABEL_RECORDS\s*=\s*(\d+)",
        header,
        re.IGNORECASE,
    )

    if record_match and label_match:
        return (
            int(record_match.group(1))
            * int(label_match.group(1))
        )

    warnings.warn(
        f"Could not parse PDS3 label size for '{path}'. "
        "Assuming zero offset."
    )

    return 0


def _read_label_text(
    path: str,
    max_bytes: int = 200_000,
) -> str:
    """
    Read the ASCII PDS3 label.
    """

    label_size = _find_pds3_label_size(path)

    read_size = (
        label_size
        if label_size > 0
        else max_bytes
    )

    with open(path, "rb") as f:
        return f.read(read_size).decode(
            "latin-1",
            errors="ignore",
        )


def _find_pds3_image_dimensions(
    path: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse:

        LINES
        LINE_SAMPLES

    from a PDS3 label.
    """

    text = _read_label_text(path)

    lines_match = re.search(
        r"\bLINES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    samples_match = re.search(
        r"\bLINE_SAMPLES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    lines = (
        int(lines_match.group(1))
        if lines_match
        else None
    )

    samples = (
        int(samples_match.group(1))
        if samples_match
        else None
    )

    return lines, samples


# ----------------------------------------------------------------------
# Standard single-band PDS3 .IMG reader
# ----------------------------------------------------------------------

def _read_raw_img(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Read a simple single-band uint8 PDS3 .IMG.

    This is primarily for the original LROC EDR products such as:

        M129133239RE.IMG
        M150368601RE.IMG

    It is NOT used for NAC_PHO products.
    """

    # --------------------------------------------------------------
    # Try planetaryimage first
    # --------------------------------------------------------------

    try:

        from planetaryimage import PDS3Image

        img = PDS3Image.open(path)

        arr = np.asarray(img.image)

        if arr.ndim > 2:
            arr = np.squeeze(arr)

        if window is not None:
            x, y, w, h = window
            arr = arr[
                y:y + h,
                x:x + w,
            ]

        return arr

    except Exception as e:

        warnings.warn(
            f"planetaryimage failed to load '{path}' "
            f"({type(e).__name__}: {e}). "
            "Attempting raw PDS3 fallback."
        )

    # --------------------------------------------------------------
    # Raw fallback
    # --------------------------------------------------------------

    label_size = _find_pds3_label_size(path)

    lines, samples = _find_pds3_image_dimensions(path)

    if lines is None or samples is None:
        raise ValueError(
            f"Could not determine LINES/LINE_SAMPLES "
            f"for '{path}'."
        )

    with open(path, "rb") as f:

        f.seek(label_size)

        raw = np.fromfile(
            f,
            dtype=np.uint8,
            count=lines * samples,
        )

    expected = lines * samples

    if raw.size < expected:
        raise ValueError(
            f"'{path}' contains only {raw.size} image bytes "
            f"but the label requires {expected}."
        )

    arr = raw.reshape(
        lines,
        samples,
    )

    if window is not None:

        x, y, w, h = window

        arr = arr[
            y:y + h,
            x:x + w,
        ]

    return arr


# ----------------------------------------------------------------------
# NAC_PHO helpers
# ----------------------------------------------------------------------

_NAC_PHO_BAND_NAME_PATTERNS = {

    "phase_deg": re.compile(
        r"phase\s*angle",
        re.IGNORECASE,
    ),

    "emission_deg": re.compile(
        r"emission\s*angle",
        re.IGNORECASE,
    ),

    "incidence_deg": re.compile(
        r"incidence\s*angle",
        re.IGNORECASE,
    ),
}


def _is_nac_pho(path: str) -> bool:
    """
    Detect whether a file is an LROC NAC_PHO product.

    We deliberately use the filename rather than assuming every .IMG
    is NAC_PHO.
    """

    name = os.path.basename(path).upper()

    return (
        "NAC_PHO" in name
        or "_PHO_" in name
    )


def _read_nac_pho_with_rasterio(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Attempt to read NAC_PHO using rasterio/GDAL.

    IMPORTANT:
    Only the requested window is read, preventing us from loading the
    entire 4-band NAC_PHO product into RAM.

    NAC_PHO convention:
        Band 1 = calibrated image
        Band 2 = phase angle
        Band 3 = emission angle
        Band 4 = incidence angle

    Invalid/no-data angle values are converted to NaN.
    """

    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is not installed."
        )

    with rasterio.open(path) as ds:

        if ds.count < 4:
            raise ValueError(
                f"NAC_PHO product '{path}' has only "
                f"{ds.count} bands. Expected at least 4."
            )

        # ----------------------------------------------------------
        # Rasterio window
        # ----------------------------------------------------------

        rio_window = None

        if window is not None:

            x, y, w, h = window

            rio_window = rasterio.windows.Window(
                col_off=x,
                row_off=y,
                width=w,
                height=h,
            )

        # ----------------------------------------------------------
        # Read Band 1 = image
        # ----------------------------------------------------------

        image = ds.read(
            1,
            window=rio_window,
        )

        # ----------------------------------------------------------
        # Read geometry bands
        # ----------------------------------------------------------

        phase = ds.read(
            2,
            window=rio_window,
        ).astype(np.float32)

        emission = ds.read(
            3,
            window=rio_window,
        ).astype(np.float32)

        incidence = ds.read(
            4,
            window=rio_window,
        ).astype(np.float32)

        # ----------------------------------------------------------
        # Clean invalid NAC_PHO geometry values
        # ----------------------------------------------------------
        #
        # NAC_PHO uses very large negative values such as
        # -3.4028227e+38 to represent invalid/no-data pixels.
        #
        # These are NOT real angles, so convert them to NaN.
        # ----------------------------------------------------------

        for arr in (incidence, emission, phase):

            # Remove NaN/Inf values
            arr[~np.isfinite(arr)] = np.nan

            # Remove PDS3 invalid/no-data values
            arr[arr < -1e20] = np.nan

        # ----------------------------------------------------------
        # Preserve geospatial metadata
        # ----------------------------------------------------------

        geotransform = ds.transform
        crs = ds.crs

    return (
        image,
        incidence,
        emission,
        phase,
        geotransform,
        crs,
    )


def _read_nac_pho_raw(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Fallback NAC_PHO reader.

    This handles the case where GDAL/rasterio cannot open the PDS3
    NAC_PHO .IMG.

    The reader first parses the PDS3 label to determine:

        LINES
        LINE_SAMPLES
        SAMPLE_BITS
        SAMPLE_TYPE
        BANDS
        BAND_STORAGE_TYPE

    It supports the common BSQ layout used by PDS3 products.

    If the label describes another storage layout, we fail loudly
    instead of silently returning corrupted data.
    """

    text = _read_label_text(path)

    # --------------------------------------------------------------
    # Basic dimensions
    # --------------------------------------------------------------

    lines_match = re.search(
        r"\bLINES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    samples_match = re.search(
        r"\bLINE_SAMPLES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    bands_match = re.search(
        r"\bBANDS\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    if not lines_match or not samples_match:
        raise ValueError(
            f"Could not determine NAC_PHO dimensions "
            f"from '{path}'."
        )

    lines = int(lines_match.group(1))
    samples = int(samples_match.group(1))

    bands = (
        int(bands_match.group(1))
        if bands_match
        else 1
    )

    if bands < 4:
        raise ValueError(
            f"NAC_PHO product '{path}' reports "
            f"{bands} band(s), expected at least 4."
        )

    # --------------------------------------------------------------
    # Pixel format
    # --------------------------------------------------------------

    sample_bits_match = re.search(
        r"\bSAMPLE_BITS\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    sample_bits = (
        int(sample_bits_match.group(1))
        if sample_bits_match
        else 32
    )

    sample_type_match = re.search(
        r"\bSAMPLE_TYPE\s*=\s*([A-Z0-9_]+)",
        text,
        re.IGNORECASE,
    )

    sample_type = (
        sample_type_match.group(1).upper()
        if sample_type_match
        else ""
    )

    # --------------------------------------------------------------
    # Determine dtype
    # --------------------------------------------------------------

    if sample_bits == 8:

        dtype = np.uint8

    elif sample_bits == 16:

        if "MSB" in sample_type:

            dtype = np.dtype(">i2")

        elif "UNSIGNED" in sample_type:

            dtype = np.dtype(">u2")

        else:

            dtype = np.dtype(">i2")

    elif sample_bits == 32:

        if "REAL" in sample_type or "FLOAT" in sample_type:

            dtype = np.dtype(">f4")

        elif "UNSIGNED" in sample_type:

            dtype = np.dtype(">u4")

        else:

            dtype = np.dtype(">i4")

    else:

        raise ValueError(
            f"Unsupported NAC_PHO SAMPLE_BITS="
            f"{sample_bits} in '{path}'."
        )

    # --------------------------------------------------------------
    # Storage type
    # --------------------------------------------------------------

    storage_match = re.search(
        r"\bBAND_STORAGE_TYPE\s*=\s*([A-Z0-9_]+)",
        text,
        re.IGNORECASE,
    )

    storage = (
        storage_match.group(1).upper()
        if storage_match
        else "BSQ"
    )

    if storage != "BSQ":

        raise ValueError(
            f"NAC_PHO '{path}' uses "
            f"BAND_STORAGE_TYPE={storage}. "
            "The raw fallback currently supports BSQ only."
        )

    # --------------------------------------------------------------
    # Label offset
    # --------------------------------------------------------------

    label_size = _find_pds3_label_size(path)

    # --------------------------------------------------------------
    # Window
    # --------------------------------------------------------------

    if window is None:

        x = 0
        y = 0
        w = samples
        h = lines

    else:

        x, y, w, h = window

    if x < 0 or y < 0:
        raise ValueError(
            f"Invalid NAC_PHO window {window}."
        )

    if x + w > samples or y + h > lines:

        raise ValueError(
            f"NAC_PHO window {window} exceeds "
            f"image dimensions {(samples, lines)}."
        )

    # --------------------------------------------------------------
    # Bytes per pixel
    # --------------------------------------------------------------

    bytes_per_sample = sample_bits // 8

    # One complete band
    band_bytes = (
        lines
        * samples
        * bytes_per_sample
    )

    # --------------------------------------------------------------
    # Read only the requested region from each band
    # --------------------------------------------------------------

    image = np.empty(
        (h, w),
        dtype=np.float32,
    )

    phase = np.empty(
        (h, w),
        dtype=np.float32,
    )

    emission = np.empty(
        (h, w),
        dtype=np.float32,
    )

    incidence = np.empty(
        (h, w),
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # Helper for reading one band window
    # --------------------------------------------------------------

    def read_band(
        band_number: int,
    ) -> np.ndarray:

        band_offset = (
            label_size
            + (band_number - 1)
            * band_bytes
        )

        out = np.empty(
            (h, w),
            dtype=dtype,
        )

        row_bytes = (
            samples
            * bytes_per_sample
        )

        wanted_bytes = (
            w
            * bytes_per_sample
        )

        with open(path, "rb") as f:

            for row in range(h):

                source_row = y + row

                offset = (
                    band_offset
                    + source_row * row_bytes
                    + x * bytes_per_sample
                )

                f.seek(offset)

                raw = f.read(wanted_bytes)

                if len(raw) != wanted_bytes:

                    raise IOError(
                        f"Unexpected EOF while reading "
                        f"band {band_number}, row {row} "
                        f"from '{path}'."
                    )

                out[row, :] = np.frombuffer(
                    raw,
                    dtype=dtype,
                    count=w,
                )

        return out

    # --------------------------------------------------------------
    # NAC_PHO bands
    # --------------------------------------------------------------

    image[:, :] = read_band(1)

    phase[:, :] = read_band(2)

    emission[:, :] = read_band(3)

    incidence[:, :] = read_band(4)

    # --------------------------------------------------------------
    # Convert endian / dtype to native float32
    # --------------------------------------------------------------

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    phase = np.asarray(
        phase,
        dtype=np.float32,
    )

    emission = np.asarray(
        emission,
        dtype=np.float32,
    )

    incidence = np.asarray(
        incidence,
        dtype=np.float32,
    )

    return (
        image,
        incidence,
        emission,
        phase,
        None,
        None,
    )


def _read_nac_pho(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Read an LROC NAC_PHO product.

    Strategy:

        1. Try rasterio/GDAL.
        2. If rasterio cannot open the PDS3 .IMG, use our raw PDS3
           BSQ reader.

    Returns:

        image
        incidence
        emission
        phase
        geotransform
        crs
    """

    # --------------------------------------------------------------
    # First attempt: rasterio
    # --------------------------------------------------------------

    try:

        return _read_nac_pho_with_rasterio(
            path,
            window=window,
        )

    except Exception as e:

        warnings.warn(
            f"rasterio could not open NAC_PHO file "
            f"'{path}' ({type(e).__name__}: {e}). "
            "Falling back to the native PDS3 NAC_PHO reader."
        )

    # --------------------------------------------------------------
    # Second attempt: raw PDS3 reader
    # --------------------------------------------------------------

    return _read_nac_pho_raw(
        path,
        window=window,
    )


# ----------------------------------------------------------------------
# Main image loader
# ----------------------------------------------------------------------

def load_image(
    path: str,
    sensor_hint: Optional[str] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
    angles_from_label: bool = False,
    fetch_angles_online: bool = False,
    manual_incidence_deg: Optional[float] = None,
    manual_emission_deg: Optional[float] = None,
    manual_phase_deg: Optional[float] = None,
    nac_pho_path: Optional[str] = None,
    nac_pho_band_phase: int = 2,
    nac_pho_band_emission: int = 3,
    nac_pho_band_incidence: int = 4,
) -> LoadedImage:
    """
    Load an image.

    IMPORTANT NEW BEHAVIOUR:

    If `path` itself is an LROC NAC_PHO product, Band 1 becomes the
    actual image and Bands 2/3/4 become the pixel-wise geometry maps.

    Therefore this now works:

        load_image(
            "NAC_PHO_E010N0230_M129133239R.IMG",
            sensor_hint="LROC",
            window=(x, y, w, h),
        )

    and gives:

        loaded.data
        loaded.phase_deg
        loaded.emission_deg
        loaded.incidence_deg

    all on the SAME pixel grid.
    """

    ext = os.path.splitext(path)[1].lower()

    geotransform = None
    crs = None

    # ==============================================================
    # NAC_PHO DIRECT MODE
    # ==============================================================

    if ext == ".img" and _is_nac_pho(path):

        (
            arr,
            incidence,
            emission,
            phase,
            geotransform,
            crs,
        ) = _read_nac_pho(
            path,
            window=window,
        )

        sensor = (
            get_sensor_config(sensor_hint)
            if sensor_hint
            else get_sensor_config("LROC")
        )

        normalized = _normalize(arr)

        loaded = LoadedImage(
            data=normalized,
            path=path,
            sensor=sensor,
            incidence_deg=incidence,
            emission_deg=emission,
            phase_deg=phase,
            geotransform=geotransform,
            crs=crs,
        )

        # ----------------------------------------------------------
        # If another NAC_PHO path was explicitly supplied, let it
        # override the angle source.
        # ----------------------------------------------------------

        if nac_pho_path is not None:

            attach_nac_pho_angles(
                loaded,
                nac_pho_path,
                window=window,
                band_phase=nac_pho_band_phase,
                band_emission=nac_pho_band_emission,
                band_incidence=nac_pho_band_incidence,
            )

        return loaded

    # ==============================================================
    # NORMAL IMAGE MODE
    # ==============================================================

    if ext == ".img":

        arr = _read_raw_img(
            path,
            window=window,
        )

    elif (
        ext in (".cub", ".tif", ".tiff")
        and _HAS_RASTERIO
    ):

        with rasterio.open(path) as ds:

            if window is None:

                arr = ds.read(1)

            else:

                x, y, w, h = window

                rio_window = rasterio.windows.Window(
                    col_off=x,
                    row_off=y,
                    width=w,
                    height=h,
                )

                arr = ds.read(
                    1,
                    window=rio_window,
                )

            geotransform = ds.transform
            crs = ds.crs

    elif (
        ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        and _HAS_PIL
    ):

        arr = np.array(
            Image.open(path).convert("F")
        )

        if window is not None:

            x, y, w, h = window

            arr = arr[
                y:y + h,
                x:x + w,
            ]

    else:

        raise ValueError(
            f"Unsupported file type '{ext}' or missing dependency "
            f"(rasterio={_HAS_RASTERIO}, PIL={_HAS_PIL})"
        )

    # ==============================================================
    # Construct LoadedImage
    # ==============================================================

    sensor = (
        get_sensor_config(sensor_hint)
        if sensor_hint
        else detect_sensor(path)
    )

    normalized = _normalize(arr)

    loaded = LoadedImage(
        data=normalized,
        path=path,
        sensor=sensor,
        geotransform=geotransform,
        crs=crs,
    )

    # ==============================================================
    # Attach angle source
    # ==============================================================

    if nac_pho_path is not None:

        attach_nac_pho_angles(
            loaded,
            nac_pho_path,
            window=window,
            band_phase=nac_pho_band_phase,
            band_emission=nac_pho_band_emission,
            band_incidence=nac_pho_band_incidence,
        )

    elif (
        manual_incidence_deg is not None
        and manual_emission_deg is not None
    ):

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_manual(
            manual_incidence_deg,
            manual_emission_deg,
            loaded.data.shape,
            phase_deg=manual_phase_deg,
        )

    elif fetch_angles_online:

        attach_lroc_fetched_angles(loaded)

    elif angles_from_label:

        attach_label_angles(loaded)

    return loaded


# ----------------------------------------------------------------------
# Separate angle-map loader
# ----------------------------------------------------------------------

def load_angle_maps(
    incidence_path: str,
    emission_path: str,
    phase_path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load separate incidence/emission/phase rasters.
    """

    def _load(p):

        if _HAS_RASTERIO:

            with rasterio.open(p) as ds:

                if window is None:

                    a = ds.read(1)

                else:

                    x, y, w, h = window

                    rio_window = rasterio.windows.Window(
                        col_off=x,
                        row_off=y,
                        width=w,
                        height=h,
                    )

                    a = ds.read(
                        1,
                        window=rio_window,
                    )

                return a.astype(np.float32)

        elif _HAS_PIL:

            a = np.array(
                Image.open(p)
            ).astype(np.float32)

            if window is not None:

                x, y, w, h = window

                a = a[
                    y:y + h,
                    x:x + w,
                ]

            return a

        else:

            raise ImportError(
                "Need rasterio or PIL to read angle-map rasters."
            )

    return (
        _load(incidence_path),
        _load(emission_path),
        _load(phase_path),
    )


# ----------------------------------------------------------------------
# NAC_PHO angle loader
# ----------------------------------------------------------------------

def _detect_nac_pho_bands(ds) -> dict:
    """
    Best-effort detection of angle bands from raster metadata.

    Defaults remain:

        phase     = 2
        emission  = 3
        incidence = 4
    """

    found = {}

    try:

        for band_idx in range(
            1,
            ds.count + 1,
        ):

            desc = (
                ds.descriptions[band_idx - 1]
                or ""
            )

            tags = ds.tags(
                band_idx
            ) or {}

            haystack = " ".join(
                [desc]
                + [str(v) for v in tags.values()]
            )

            for key, pattern in (
                _NAC_PHO_BAND_NAME_PATTERNS.items()
            ):

                if (
                    key not in found
                    and pattern.search(haystack)
                ):

                    found[key] = band_idx

    except Exception:

        return {}

    return found


def load_angles_from_nac_pho(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
    band_phase: int = 2,
    band_emission: int = 3,
    band_incidence: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pixel-wise NAC_PHO angle maps.

    Returns:

        incidence
        emission
        phase

    in that order.
    """

    # --------------------------------------------------------------
    # If this is a NAC_PHO PDS3 IMG, use our dedicated reader.
    # --------------------------------------------------------------

    if (
        os.path.splitext(path)[1].lower() == ".img"
        and _is_nac_pho(path)
    ):

        (
            _image,
            incidence,
            emission,
            phase,
            _geotransform,
            _crs,
        ) = _read_nac_pho(
            path,
            window=window,
        )

        return (
            incidence,
            emission,
            phase,
        )

    # --------------------------------------------------------------
    # Otherwise use rasterio.
    # --------------------------------------------------------------

    if not _HAS_RASTERIO:

        raise ImportError(
            "rasterio is required to read this NAC_PHO angle source."
        )

    with rasterio.open(path) as ds:

        required_band = max(
            band_phase,
            band_emission,
            band_incidence,
        )

        if ds.count < required_band:

            raise ValueError(
                f"'{path}' has only {ds.count} band(s), "
                f"but band {required_band} is required."
            )

        detected = _detect_nac_pho_bands(ds)

        b_phase = detected.get(
            "phase_deg",
            band_phase,
        )

        b_emission = detected.get(
            "emission_deg",
            band_emission,
        )

        b_incidence = detected.get(
            "incidence_deg",
            band_incidence,
        )

        if not detected:

            warnings.warn(
                f"'{path}': could not determine band names. "
                f"Using phase={band_phase}, "
                f"emission={band_emission}, "
                f"incidence={band_incidence}."
            )

        # ----------------------------------------------------------
        # Read only requested window
        # ----------------------------------------------------------

        rio_window = None

        if window is not None:

            x, y, w, h = window

            rio_window = rasterio.windows.Window(
                col_off=x,
                row_off=y,
                width=w,
                height=h,
            )

        phase = ds.read(
            b_phase,
            window=rio_window,
        ).astype(np.float32)

        emission = ds.read(
            b_emission,
            window=rio_window,
        ).astype(np.float32)

        incidence = ds.read(
            b_incidence,
            window=rio_window,
        ).astype(np.float32)

    # --------------------------------------------------------------
    # Clean invalid/special pixels
    # --------------------------------------------------------------

    for arr in (
        incidence,
        emission,
        phase,
    ):

        arr[~np.isfinite(arr)] = np.nan

        arr[
            (arr < -1e3)
            | (arr > 1e3)
        ] = np.nan

    return (
        incidence,
        emission,
        phase,
    )


# ----------------------------------------------------------------------
# Attach NAC_PHO angles to LoadedImage
# ----------------------------------------------------------------------

def attach_nac_pho_angles(
    loaded: LoadedImage,
    nac_pho_path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
    band_phase: int = 2,
    band_emission: int = 3,
    band_incidence: int = 4,
) -> LoadedImage:
    """
    Attach NAC_PHO angle maps to an already-loaded image.
    """

    try:

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_from_nac_pho(
            nac_pho_path,
            window=window,
            band_phase=band_phase,
            band_emission=band_emission,
            band_incidence=band_incidence,
        )

        # Clean NAC_PHO no-data values
        loaded.incidence_deg = loaded.incidence_deg.astype(np.float32)
        loaded.emission_deg = loaded.emission_deg.astype(np.float32)
        loaded.phase_deg = loaded.phase_deg.astype(np.float32)

        loaded.incidence_deg[loaded.incidence_deg < -1e20] = np.nan
        loaded.emission_deg[loaded.emission_deg < -1e20] = np.nan
        loaded.phase_deg[loaded.phase_deg < -1e20] = np.nan

    except Exception as e:

        warnings.warn(
            f"attach_nac_pho_angles failed for "
            f"'{nac_pho_path}' "
            f"({type(e).__name__}: {e}) - "
            "continuing without photometric weighting."
        )

    return loaded


# ----------------------------------------------------------------------
# Label-based angle shortcut
# ----------------------------------------------------------------------

_ANGLE_FIELD_PATTERNS = {

    "incidence_deg":
        r"INCIDENCE_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "emission_deg":
        r"EMISSION_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "phase_deg":
        r"PHASE_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_azimuth_deg":
        r"SUB_SOLAR_AZIMUTH\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_latitude_deg":
        r"SUB_SOLAR_LATITUDE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_longitude_deg":
        r"SUB_SOLAR_LONGITUDE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
}


def parse_label_angles(path: str) -> dict:
    """
    Parse incidence/emission/phase angles from PDS3 label.
    """

    text = _read_label_text(path)

    values = {}
    missing = []

    for key, pattern in _ANGLE_FIELD_PATTERNS.items():

        m = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if m:

            values[key] = float(
                m.group(1)
            )

        else:

            missing.append(key)

    required = (
        "incidence_deg",
        "emission_deg",
    )

    missing_required = [
        k
        for k in required
        if k in missing
    ]

    if missing_required:

        raise ValueError(
            f"Label for '{path}' is missing "
            f"required angle field(s): "
            f"{missing_required}. "
            "Fall back to phocube or another geometry source."
        )

    if missing:

        warnings.warn(
            f"Label for '{path}' is missing "
            f"optional field(s): {missing}"
        )

    return values


def load_angles_from_label(
    path: str,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Broadcast scene-level angles to the requested image shape.
    """

    values = parse_label_angles(path)

    incidence = np.full(
        shape,
        values["incidence_deg"],
        dtype=np.float32,
    )

    emission = np.full(
        shape,
        values["emission_deg"],
        dtype=np.float32,
    )

    phase = np.full(
        shape,
        values.get(
            "phase_deg",
            np.nan,
        ),
        dtype=np.float32,
    )

    return (
        incidence,
        emission,
        phase,
    )


def attach_label_angles(
    loaded: LoadedImage,
) -> LoadedImage:
    """
    Attach label-derived angle maps.
    """

    try:

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_from_label(
            loaded.path,
            loaded.data.shape,
        )

    except ValueError as e:

        warnings.warn(str(e))

    return loaded


# ----------------------------------------------------------------------
# Manual angles
# ----------------------------------------------------------------------

def load_angles_manual(
    incidence_deg: float,
    emission_deg: float,
    shape: Tuple[int, int],
    phase_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Broadcast manually supplied angles.
    """

    incidence = np.full(
        shape,
        incidence_deg,
        dtype=np.float32,
    )

    emission = np.full(
        shape,
        emission_deg,
        dtype=np.float32,
    )

    phase = np.full(
        shape,
        (
            phase_deg
            if phase_deg is not None
            else np.nan
        ),
        dtype=np.float32,
    )

    return (
        incidence,
        emission,
        phase,
    )


# ----------------------------------------------------------------------
# LROC ODE online angle fetching
# ----------------------------------------------------------------------

def derive_lroc_product_id(
    path: str,
) -> str:
    """
    Example:

        M109080308RE.IMG
        ->
        M109080308RE
    """

    return os.path.splitext(
        os.path.basename(path)
    )[0]


_LROC_ODE_ANGLE_PATTERNS = {

    "incidence_deg":
        r"Incidence angle\s+([-\d.]+)",

    "emission_deg":
        r"Emission angle\s+([-\d.]+)",

    "phase_deg":
        r"Phase angle\s+([-\d.]+)",
}


def fetch_lroc_angles(
    product_id: str,
    dataset: str = "LRO-L-LROC-2-EDR-V1.0",
    timeout: float = 15.0,
) -> dict:
    """
    Fetch angle values from LROC ODE.
    """

    import requests

    url = (
        "https://data.lroc.im-ldi.com/"
        f"lroc/view_lroc/{dataset}/{product_id}"
    )

    resp = requests.get(
        url,
        timeout=timeout,
    )

    resp.raise_for_status()

    clean = re.sub(
        r"<[^>]+>",
        " ",
        resp.text,
    )

    clean = re.sub(
        r"[|]",
        " ",
        clean,
    )

    values = {}
    missing = []

    for key, pattern in (
        _LROC_ODE_ANGLE_PATTERNS.items()
    ):

        m = re.search(
            pattern,
            clean,
            re.IGNORECASE,
        )

        if m:

            values[key] = float(
                m.group(1)
            )

        else:

            missing.append(key)

    if (
        "incidence_deg" in missing
        or "emission_deg" in missing
    ):

        raise ValueError(
            f"Could not parse required angle "
            f"field(s) {missing} from {url}."
        )

    return values


def attach_lroc_fetched_angles(
    loaded: LoadedImage,
    dataset: str = "LRO-L-LROC-2-EDR-V1.0",
) -> LoadedImage:
    """
    Fetch LROC angle values and broadcast them to the image.
    """

    product_id = derive_lroc_product_id(
        loaded.path
    )

    try:

        values = fetch_lroc_angles(
            product_id,
            dataset=dataset,
        )

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_manual(
            values["incidence_deg"],
            values["emission_deg"],
            loaded.data.shape,
            phase_deg=values.get(
                "phase_deg"
            ),
        )

    except Exception as e:

        warnings.warn(
            f"fetch_lroc_angles failed for "
            f"'{product_id}' "
            f"({type(e).__name__}: {e}). "
            "Continuing without photometric weighting."
        )

    return loaded


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------

def resample_to_gsd(
    img: np.ndarray,
    src_gsd_m: float,
    dst_gsd_m: float,
) -> np.ndarray:
    """
    Resample image to requested GSD using PIL bilinear resize.
    """

    if not _HAS_PIL:

        raise ImportError(
            "PIL required for resample_to_gsd "
            "(pip install pillow)"
        )

    scale = (
        src_gsd_m
        / dst_gsd_m
    )

    h, w = img.shape

    new_h = max(
        1,
        int(round(h * scale)),
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    pil_img = Image.fromarray(
        (img * 255).astype(np.uint8)
    )

    resized = pil_img.resize(
        (new_w, new_h),
        Image.BILINEAR,
    )

    return (
        np.array(resized)
        .astype(np.float32)
        / 255.0
    )


# ----------------------------------------------------------------------
# GSD-based scale prior (Stage 1.5)
# ----------------------------------------------------------------------

def estimate_gsd_scale_prior(
    src: "LoadedImage",
    ref: "LoadedImage",
) -> Optional[float]:
    """Returns a prior estimate of src-vs-ref scale ratio from ground
    sampling distance metadata, so scale.select_best_scale's coarse-to-fine
    search (pwift.coarse_to_fine_rotation_scale) can be given a tight band
    around the physically-expected ratio instead of blindly searching the
    sensor's full scale_range (e.g. OHRC's 0.5x-3x) from scratch.

    Priority: each image's own measured `gsd_m` (from its label/metadata,
    when `load_image` was able to populate it) over the sensor's
    `approx_gsd_m` placeholder from config.py. If NEITHER image has a
    usable GSD, returns None - callers should fall back to the existing
    full-range search unchanged.

    Ratio convention matches scale.apply_scale / pwift's scale candidates:
    a return value of 2.0 means the source image needs to be scaled *up*
    2x to match the reference's pixel scale (i.e. src has 2x coarser GSD
    than ref).
    """
    src_gsd = src.gsd_m or getattr(src.sensor, "approx_gsd_m", None)
    ref_gsd = ref.gsd_m or getattr(ref.sensor, "approx_gsd_m", None)

    if not src_gsd or not ref_gsd or src_gsd <= 0 or ref_gsd <= 0:
        warnings.warn(
            "estimate_gsd_scale_prior: no usable GSD for source and/or "
            "reference (checked LoadedImage.gsd_m, then "
            "SensorConfig.approx_gsd_m) - falling back to an unconstrained "
            "scale search over the sensor's full scale_range."
        )
        return None

    return float(src_gsd / ref_gsd
    )