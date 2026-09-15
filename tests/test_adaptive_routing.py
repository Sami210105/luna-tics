import csv
import json
import os
import tempfile
import cv2
import numpy as np
import pytest

from lunar_registration.config import PipelineConfig
from lunar_registration.pipeline import determine_adaptive_matcher, run_pipeline
from lunar_registration.matching import MatchResult


def _create_synthetic_lunar_image(size=(256, 256), seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    # Add synthetic crater rims and floors
    for _ in range(15):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    return img


def test_determine_adaptive_matcher_polar_grazing():
    cfg = PipelineConfig()
    
    # Polar grazing sun (i >= 70 deg) without real-time constraint -> hybrid_pwift_roma2
    matcher, meta = determine_adaptive_matcher(
        requested_matcher="auto",
        incidence_deg=75.0,
        real_time=False,
        cfg=cfg,
    )
    assert matcher == "hybrid_pwift_roma2"
    assert meta["mode"] == "adaptive"
    assert meta["regime"] == "polar_grazing"
    assert meta["incidence_deg"] == 75.0
    assert meta["real_time_requested"] is False

    # Polar grazing sun (i >= 70 deg) WITH real-time constraint -> hybrid_pwift_eloftr
    matcher_rt, meta_rt = determine_adaptive_matcher(
        requested_matcher="auto",
        incidence_deg=80.0,
        real_time=True,
        cfg=cfg,
    )
    assert matcher_rt == "hybrid_pwift_eloftr"
    assert meta_rt["regime"] == "polar_grazing"
    assert meta_rt["real_time_requested"] is True


def test_determine_adaptive_matcher_real_time_trn():
    cfg = PipelineConfig()
    
    # Moderate sun (i < 70 deg) with real-time constraint -> eloftr
    matcher, meta = determine_adaptive_matcher(
        requested_matcher="auto",
        incidence_deg=35.0,
        real_time=True,
        cfg=cfg,
    )
    assert matcher == "eloftr"
    assert meta["mode"] == "adaptive"
    assert meta["regime"] == "real_time_trn"
    assert meta["real_time_requested"] is True


def test_determine_adaptive_matcher_subpixel_cartography():
    cfg = PipelineConfig()
    
    # Moderate sun (i < 70 deg) nominal cartography -> roma2
    matcher, meta = determine_adaptive_matcher(
        requested_matcher="auto",
        incidence_deg=30.0,
        real_time=False,
        cfg=cfg,
    )
    assert matcher == "roma2"
    assert meta["mode"] == "adaptive"
    assert meta["regime"] == "subpixel_cartography"
    assert meta["real_time_requested"] is False


def test_determine_adaptive_matcher_manual_override():
    cfg = PipelineConfig()
    
    # Explicit user override: should bypass condition routing regardless of angles
    matcher, meta = determine_adaptive_matcher(
        requested_matcher="pwift",
        incidence_deg=85.0,
        real_time=True,
        cfg=cfg,
    )
    assert matcher == "pwift"
    assert meta["mode"] == "manual_override"
    assert meta["regime"] == "user_specified"

    matcher_roma, meta_roma = determine_adaptive_matcher(
        requested_matcher="roma2",
        incidence_deg=85.0,
        real_time=True,
        cfg=cfg,
    )
    assert matcher_roma == "roma2"
    assert meta_roma["mode"] == "manual_override"


def test_pipeline_contingency_fallback_on_neural_exception(tmp_path, monkeypatch):
    """If a neural matcher raises an exception (e.g. CUDA OOM or missing weights),
    the pipeline should gracefully fall back to PWIFT standalone and complete."""
    src_img = _create_synthetic_lunar_image(seed=50)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 3.0
    M[1, 2] -= 2.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_fallback.png")
    ref_path = str(tmp_path / "ref_fallback.png")
    out_dir = str(tmp_path / "out_fallback")
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    import lunar_registration.pipeline as lr_pipe

    class FailingMatcher:
        def match(self, *args, **kwargs):
            raise RuntimeError("Simulated CUDA Out of Memory (OOM)")

    def mock_get_matcher(name, cfg):
        if name in ("roma2", "eloftr"):
            return FailingMatcher()
        return lr_pipe.BaseMatcher.__subclasses__()[0](cfg)  # PWIFT

    monkeypatch.setattr(lr_pipe, "get_matcher", mock_get_matcher)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="roma2",  # Explicitly request failing neural matcher
    )

    assert summary["contingency_fallback"]["triggered"] is True
    assert summary["contingency_fallback"]["fallback_matcher"] == "pwift"
    assert "CUDA Out of Memory" in summary["contingency_fallback"]["reason"]
    assert os.path.exists(summary["outputs"]["registered_png"])
    assert os.path.exists(summary["outputs"]["matchpoints_csv"])


def test_pipeline_gcl_gcps_csv_export(tmp_path):
    """Verifies that Ground Control Lattice (GCL) GCPs are exported to CSV with proper headers."""
    src_img = _create_synthetic_lunar_image(seed=60)
    M = cv2.getRotationMatrix2D((128, 128), 1.5, 1.0)
    M[0, 2] += 2.0
    M[1, 2] -= 1.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_gcl.png")
    ref_path = str(tmp_path / "ref_gcl.png")
    out_dir = str(tmp_path / "out_gcl")
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="pwift",
        export_gcl_gcps_csv=True,
    )

    assert "gcl_gcps_csv" in summary["outputs"]
    gcp_csv = summary["outputs"]["gcl_gcps_csv"]
    assert os.path.exists(gcp_csv)

    with open(gcp_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_header = [
            "gcp_id", "src_x", "src_y", "ref_x", "ref_y",
            "gcs_lon", "gcs_lat", "inlier", "cell_x", "cell_y", "confidence"
        ]
        assert header == expected_header
        rows = list(reader)
        assert len(rows) > 0  # Should have generated GCPs


def test_pipeline_orthogonal_gate_metadata(tmp_path):
    """Verifies that gate_cheap verification gate executes and populates summary metadata."""
    src_img = _create_synthetic_lunar_image(seed=70)
    M = cv2.getRotationMatrix2D((128, 128), 0.5, 1.0)
    M[0, 2] += 1.0
    M[1, 2] -= 1.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_gate.png")
    ref_path = str(tmp_path / "ref_gate.png")
    out_dir = str(tmp_path / "out_gate")
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="pwift",
        enable_orthogonal_gate=True,
    )

    assert "orthogonal_gate" in summary
    assert summary["orthogonal_gate"]["enabled"] is True
    assert "passed" in summary["orthogonal_gate"]
    assert "cost_ms" in summary["orthogonal_gate"]
    assert "struct_ncc" in summary["orthogonal_gate"]


def test_pipeline_e2e_adaptive_dispatch_polar(tmp_path):
    """Verifies that running the pipeline with default matcher='auto' and i=75 deg
    properly routes to polar grazing regime and executes successfully."""
    src_img = _create_synthetic_lunar_image(seed=80)
    M = cv2.getRotationMatrix2D((128, 128), 1.0, 1.0)
    M[0, 2] += 2.0
    M[1, 2] -= 1.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_polar.png")
    ref_path = str(tmp_path / "ref_polar.png")
    out_dir = str(tmp_path / "out_polar")
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        manual_incidence_deg=75.0,  # Extreme polar angle
        matcher="auto",
    )

    assert summary["condition_routing"]["regime"] == "polar_grazing"
    assert summary["condition_routing"]["resolved_matcher"] == "hybrid_pwift_roma2"
    assert "best_method" in summary
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(summary["outputs"]["registered_png"])


def test_pipeline_e2e_adaptive_dispatch_real_time(tmp_path):
    """Verifies that running the pipeline with default matcher='auto' and real_time=True
    properly routes to real_time_trn regime and executes successfully."""
    src_img = _create_synthetic_lunar_image(seed=90)
    M = cv2.getRotationMatrix2D((128, 128), 1.0, 1.0)
    M[0, 2] += 2.0
    M[1, 2] -= 1.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_rt.png")
    ref_path = str(tmp_path / "ref_rt.png")
    out_dir = str(tmp_path / "out_rt")
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        manual_incidence_deg=30.0,
        real_time=True,  # Real-time descent TRN requested
        matcher="auto",
    )

    assert summary["condition_routing"]["regime"] == "real_time_trn"
    assert summary["condition_routing"]["resolved_matcher"] == "eloftr"
    assert "best_method" in summary
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(summary["outputs"]["registered_png"])

