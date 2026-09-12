import numpy as np
import pytest

from lunar_registration.refine import (
    choose_refiner,
    compute_texture_energy,
    refine_tile,
    check_seam,
)


def test_choose_refiner_illum_routing():
    # Severe illumination difference should never use raw ECC
    assert choose_refiner(illum_delta_deg=120.0, texture_energy=0.05) in ("pc_ncc", "skip")
    assert choose_refiner(illum_delta_deg=45.0, texture_energy=0.02) in ("pc_ncc", "skip")

    # Moderate illumination difference with good texture uses ECC
    assert choose_refiner(illum_delta_deg=5.0, texture_energy=0.04) == "ecc"
    # Low texture uses phase correlation
    assert choose_refiner(illum_delta_deg=5.0, texture_energy=0.02) == "phase"


def test_refine_tile_phase():
    rng = np.random.default_rng(42)
    tile0 = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
    # Shifted tile
    tile1 = np.roll(np.roll(tile0, 3, axis=1), 2, axis=0)

    out = refine_tile(tile0, tile1, illum_delta_deg=5.0, method="phase")
    assert "dx" in out and "dy" in out
    assert "cov2x2" in out
    assert out["cov2x2"].shape == (2, 2)


def test_check_seam():
    warpA = np.eye(3, dtype=np.float64)
    warpB = np.eye(3, dtype=np.float64)
    # Under identical warps, seam should agree
    res = check_seam(warpA, warpB, overlap_bbox=(10, 10, 50, 50), tau_seam_meters=2.0, gsd_ref=0.5)
    assert res["agree"] is True
    assert res["tag"] == "clean"
