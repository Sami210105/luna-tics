import numpy as np
import pytest

from lunar_registration.verify import compute_structural_ncc, compete_rigid, gate_cheap


def test_structural_ncc_gradient_robust():
    rng = np.random.default_rng(1)
    im0 = rng.uniform(0, 1, size=(128, 128)).astype(np.float32)
    H_id = np.eye(3, dtype=np.float64)

    # Identical images should have near 1.0 structural NCC
    score_id = compute_structural_ncc(im0, im0, H_id)
    assert score_id > 0.95

    # Inverted intensity image still maintains structural edge gradients
    im_inv = 1.0 - im0
    score_inv = compute_structural_ncc(im0, im_inv, H_id)
    assert score_inv > 0.85


def test_compete_rigid_periodic_ambiguity():
    # Two competing hypotheses separated by 25px with close scores
    cands = [
        {"warp": np.eye(3), "fit": 0.92, "coverage": 0.80, "dof_resid": 0.0},
        {
            "warp": np.eye(3) + np.array([[0, 0, 25.0], [0, 0, 0], [0, 0, 0]]),
            "fit": 0.90,
            "coverage": 0.79,
            "dof_resid": 0.0,
        },
    ]
    res = compete_rigid(cands, terrain_spacing_px=24.0, lambda_dof=0.1, margin_min=0.05)
    assert res["ambiguous"] is True
    assert res["margin"] < 0.05


def test_compete_rigid_clear_winner():
    cands = [
        {"warp": np.eye(3), "fit": 0.95, "coverage": 0.85, "dof_resid": 0.0},
        {
            "warp": np.eye(3) + np.array([[0, 0, 30.0], [0, 0, 0], [0, 0, 0]]),
            "fit": 0.60,
            "coverage": 0.50,
            "dof_resid": 0.5,
        },
    ]
    res = compete_rigid(cands, terrain_spacing_px=24.0, lambda_dof=0.1, margin_min=0.05)
    assert res["ambiguous"] is False
    assert res["winner"]["fit"] == 0.95
