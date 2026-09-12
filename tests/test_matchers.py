from pathlib import Path
import numpy as np
import pytest

from lunar_registration.config import PipelineConfig
from lunar_registration.matching import (
    BaseMatcher,
    EloftrMatcher,
    MissingWeights,
    PWIFTMatcher,
    Roma2Matcher,
    get_matcher,
    resolve_romav2_weights,
    resolve_eloftr_checkpoint,
)


def test_matcher_factory_pwift():
    matcher = get_matcher("pwift")
    assert isinstance(matcher, BaseMatcher)
    assert isinstance(matcher, PWIFTMatcher)


def test_missing_weights_fails_loudly():
    with pytest.raises(MissingWeights):
        Roma2Matcher(weights_path="/path/to/nonexistent/weights.pt")

    with pytest.raises(MissingWeights):
        EloftrMatcher(checkpoint_path="/path/to/nonexistent/ckpt.ckpt")


def test_pwift_matcher_synthetic():
    cfg = PipelineConfig()
    matcher = get_matcher("pwift", cfg)
    im0 = np.ones((128, 128), dtype=np.float32) * 0.5
    im1 = np.ones((128, 128), dtype=np.float32) * 0.5
    res = matcher.match(im0, im1)
    assert res.method == "pwift"
    assert hasattr(res, "pts_src")
    assert hasattr(res, "pts_dst")


def test_roma2_matcher_initialization():
    weights_path = resolve_romav2_weights()
    if not weights_path.exists():
        pytest.skip("RoMa v2 fine-tuned weights not found")

    matcher = Roma2Matcher(weights_path=weights_path, device="cpu")
    assert isinstance(matcher, BaseMatcher)
    assert matcher.model is not None


def test_eloftr_matcher_initialization():
    ckpt_path = resolve_eloftr_checkpoint()
    if not ckpt_path.exists():
        pytest.skip("ELoFTR fine-tuned checkpoint not found")

    matcher = EloftrMatcher(checkpoint_path=ckpt_path, device="cpu")
    assert isinstance(matcher, BaseMatcher)
    assert matcher.model is not None
