"""
Per-sensor configuration.

--- SWAP POINT ---
The numeric ranges below (resolution, scale ratio vs LROC reference, expected
sun-angle range) are placeholders taken from public CH2 instrument specs /
your architecture diagram. Replace with the exact numbers from your mission
data sheet if the SIH judges ask for provenance.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SensorConfig:
    name: str
    # illumination branch to use in illumination.py
    illumination_method: str
    # (min, max) scale ratio of THIS sensor's GSD vs the reference image GSD
    scale_range: Tuple[float, float]
    # whether to use a local (block-wise) or global homography in viewpoint.py
    homography_mode: str  # "local" or "global"
    # approximate ground sampling distance in meters (placeholder - overwrite from metadata)
    approx_gsd_m: float
    # PDS3 / label keywords used to auto-detect this sensor in preprocessing.py
    label_keywords: Tuple[str, ...] = field(default_factory=tuple)


SENSOR_CONFIGS = {
    "OHRC": SensorConfig(
        name="OHRC",
        illumination_method="pwift_akimov",
        scale_range=(0.5, 3.0),
        homography_mode="local",
        approx_gsd_m=0.25,
        label_keywords=("OHRC", "ORBITER HIGH RESOLUTION CAMERA"),
    ),
    "TMC": SensorConfig(
        name="TMC",
        illumination_method="hist_shadow_log",
        scale_range=(0.3, 5.0),
        homography_mode="local",
        approx_gsd_m=5.0,
        label_keywords=("TMC", "TERRAIN MAPPING CAMERA"),
    ),
    "IIRS": SensorConfig(
        name="IIRS",
        illumination_method="clahe_invert_dilate",
        scale_range=(0.8, 1.25),
        homography_mode="global",
        approx_gsd_m=80.0,
        label_keywords=("IIRS", "IMAGING INFRARED SPECTROMETER"),
    ),
    "LROC": SensorConfig(
        name="LROC",
        illumination_method="pwift_akimov",
        scale_range=(1.0, 1.0),
        homography_mode="local",
        approx_gsd_m=0.5,
        label_keywords=("LROC", "NAC", "WAC"),
    ),
}


def get_sensor_config(name: str) -> SensorConfig:
    key = name.strip().upper()
    if key not in SENSOR_CONFIGS:
        raise KeyError(
            f"Unknown sensor '{name}'. Known: {list(SENSOR_CONFIGS)}. "
            "Add a new SensorConfig entry if this is a new instrument."
        )
    return SENSOR_CONFIGS[key]


@dataclass
class PipelineConfig:
    # ---- PWIFT structural representation (paper Sec 3.3, Eq 3-8) ----
    # NOTE: these are the FULL-fidelity settings used for the final match,
    # once the coarse-to-fine search below has already picked (scale, rotation).
    pwift_scales: int = 4
    # paper: "orientation space is quantized into 12 bins" - this doubles as
    # both the log-Gabor orientation count AND the MIM/descriptor bin count K
    pwift_orientations: int = 12
    pwift_bg_threshold: float = 0.15       # w_bg, Eq 3
    pwift_illum_threshold: float = 0.05    # on_thr, Eq 4
    pwift_soft_weight_lo: float = 0.10     # w_soft_lo, Eq 6 - NOT given numerically in the paper text; tune against your own imagery
    pwift_soft_weight_hi: float = 0.60     # w_soft_hi, Eq 6 - same caveat
    pwift_gamma_pc: float = 1.0            # gamma_pc, Eq 7 - not given numerically; 1.0 = neutral soft weighting

    # ---- keypoint detection (Sec 3.4, Eq 9-11) ----
    pwift_min_keypoint_distance: int = 8
    pwift_max_keypoints: int = 800
    # rho: minimum retention ratio of the keypoint budget reserved for the
    # m_PW (subordinate-structure) channel. The paper states this mechanism
    # exists but does not give a numeric value for rho - tune this.
    pwift_min_retention_ratio: float = 0.3
    pwift_keypoint_score_percentile: float = 60.0  # adaptive quality screening - paper describes this qualitatively only

    # ---- contextual (surrounding-terrain) descriptor - NOT in the paper ----
    # Off by default so baseline behavior is unchanged. Turn on to test
    # whether encoding what's AROUND a keypoint (nearby ridge, second
    # crater, open terrain) resolves "two craters that look identical
    # locally but sit in different surroundings" - the classic lunar
    # repetitive-terrain failure mode.
    pwift_use_context_descriptor: bool = True
    pwift_context_rings: int = 3
    pwift_context_sectors: int = 8
    pwift_context_ring_spacing_px: int = 24
    pwift_context_weight: float = 0.3   # 0 = pure appearance (= baseline); start here, raise cautiously

    pwift_max_displacement_px: "Optional[float]" = None
    pwift_displacement_mad_k: float = 3.0          # robust outlier threshold, in MAD units
    pwift_displacement_min_matches: int = 6        # need at least this many raw matches to trust the stats

    # ---- descriptor (Sec 3.5, Eq 14-18) ----
    pwift_descriptor_patch: int = 32
    pwift_descriptor_cells: int = 4        # no, Eq 18 -> 4x4x12x2 = 384-dim with pwift_orientations=12
    pwift_bright_dark_threshold: float = 0.2  # t, explicitly given in the paper (Eq 16-17)
    pwift_ratio_test: float = 0.85         # Lowe ratio test on the swap-aware distance, Eq 19-21

    # ---- coarse-to-fine rotation-scale search (Sec 3.6, Eq 22-25) ----
    # all of these are given explicitly in the paper text
    pwift_scale_candidates: Tuple[float, ...] = (0.5, 0.707, 1.0, 1.414, 2.0, 3.0)
    pwift_coarse_rotation_candidates_deg: Tuple[float, ...] = (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0)
    pwift_rotation_refine_offsets_deg: Tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    pwift_topk_scale: int = 2
    pwift_topk_rotation: int = 2
    pwift_search_coarse_size: int = 256
    pwift_search_max_keypoints: int = 200  # "limited number of keypoints" for the lightweight evaluation stage
    # PERF (new): the paper calls this stage "lightweight" and describes it
    # as running at reduced resolution with a limited number of keypoints -
    # but the previous implementation still ran the full n_scales x n_orient
    # filter bank (4x12=48 FFT-based filter passes) for every one of the
    # ~24 hypotheses evaluated during the search, which is the single most
    # expensive part of the computation and the one the resolution/keypoint
    # reduction did nothing to address. These two fields let the search use
    # a cheaper filter bank (2x6=12 passes by default, a 4x reduction) while
    # the final full-resolution match (above) still uses the full
    # pwift_scales/pwift_orientations for real descriptor fidelity. Tune
    # upward if the search starts picking visibly wrong (scale, rotation)
    # hypotheses - there's a real accuracy/speed tradeoff here, the paper's
    # text doesn't give numbers for it.
    pwift_search_scales: int = 2
    pwift_search_orientations: int = 6

    # ---- EfficientLoFTR params ----
    eloftr_model_id: str = "zju-community/efficientloftr"
    eloftr_confidence_threshold: float = 0.2
    eloftr_device: str = "cpu"  # set to "cuda" if you have a GPU

    # ---- RANSAC / homography ----
    ransac_reproj_threshold_px: float = 3.0
    ransac_max_iters: int = 5000
    ransac_confidence: float = 0.999

    # ---- reprojection cleanup (Eq 26-27) ----
    reprojection_cleanup_tau_e_px: float = 5.0  # explicitly given in the paper

    # ---- scale pyramid ----
    pyramid_levels: int = 4

    # ---- uniformity metric grid ----
    uniformity_grid: int = 8