import json
import os
import tempfile
import cv2
import numpy as np
import pytest

from lunar_registration.pipeline import run_pipeline
from lunar_registration.matching import resolve_romav2_weights, resolve_eloftr_checkpoint


def _create_synthetic_lunar_image(size=(256, 256), seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    # Add synthetic crater circles
    for _ in range(15):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r = int(rng.integers(8, 25))
        # Outer rim
        cv2.circle(img, (cx, cy), r, 200, 2)
        # Inner shadow
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        # Sunlight floor
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    return img


def test_pipeline_e2e_pwift(tmp_path):
    src_img = _create_synthetic_lunar_image(seed=10)
    # Slightly rotated/shifted reference
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src.png")
    ref_path = str(tmp_path / "ref.png")
    out_dir = str(tmp_path / "output_pwift")

    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="pwift",
    )

    assert "best_method" in summary
    assert "metrics" in summary
    assert "rigid_competition" in summary
    assert "miho_gcps" in summary
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(summary["outputs"]["registered_png"])
    assert os.path.exists(summary["outputs"]["matchpoints_csv"])


def test_pipeline_e2e_hybrid_roma2(tmp_path):
    roma2_weights = resolve_romav2_weights()
    if not roma2_weights.exists():
        pytest.skip("RoMa v2 fine-tuned weights not available")

    src_img = _create_synthetic_lunar_image(seed=24)
    M = cv2.getRotationMatrix2D((128, 128), 1.5, 1.0)
    M[0, 2] += 3.0
    M[1, 2] -= 2.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_hybrid.png")
    ref_path = str(tmp_path / "ref_hybrid.png")
    out_dir = str(tmp_path / "output_hybrid_roma2")

    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="hybrid_pwift_roma2",
        roma2_weights=str(roma2_weights),
        device=device,
    )

    assert "best_method" in summary
    assert "provenance" in summary
    assert "miho_gcps" in summary
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(summary["outputs"]["registered_png"])
    assert os.path.exists(summary["outputs"]["matchpoints_csv"])


def test_pipeline_e2e_hybrid_eloftr(tmp_path):
    eloftr_ckpt = resolve_eloftr_checkpoint()
    if not eloftr_ckpt.exists():
        pytest.skip("EfficientLoFTR checkpoint not available")

    src_img = _create_synthetic_lunar_image(seed=30)
    M = cv2.getRotationMatrix2D((128, 128), 1.0, 1.0)
    M[0, 2] += 2.0
    M[1, 2] -= 1.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))

    src_path = str(tmp_path / "src_eloftr.png")
    ref_path = str(tmp_path / "ref_eloftr.png")
    out_dir = str(tmp_path / "output_hybrid_eloftr")

    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=out_dir,
        source_sensor="LROC",
        matcher="hybrid_pwift_eloftr",
        eloftr_ckpt=str(eloftr_ckpt),
        device="cpu",
    )

    assert "best_method" in summary
    assert "provenance" in summary
    assert "miho_gcps" in summary
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(summary["outputs"]["registered_png"])
    assert os.path.exists(summary["outputs"]["matchpoints_csv"])

