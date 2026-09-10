"""
Fine-tuned LoFTR inference adapter for the lunar-registration pipeline.

Expected directory structure:

finetuned_loftr/
    config.yaml
    hparams.yaml
    last.ckpt
    epoch=....ckpt
    epoch=....ckpt
    ...

This module loads a previously trained LoFTR checkpoint and exposes a
simple matcher that returns point correspondences in pixel coordinates.

The checkpoint is NOT retrained here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR = PROJECT_ROOT / "finetuned_loftr"

DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "config.yaml"
DEFAULT_HPARAMS = DEFAULT_MODEL_DIR / "hparams.yaml"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "last.ckpt"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _find_checkpoint(model_dir: Path) -> Path:
    """
    Find a usable checkpoint.

    Preference:
        1. last.ckpt
        2. newest epoch=*.ckpt
    """

    last_ckpt = model_dir / "last.ckpt"

    if last_ckpt.exists():
        return last_ckpt

    checkpoints = sorted(
        model_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No LoFTR checkpoint found in '{model_dir}'. "
            f"Expected 'last.ckpt' or '*.ckpt'."
        )

    return checkpoints[0]


def _load_gray_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a numpy image into LoFTR's expected grayscale tensor.

    Output:
        [1, 1, H, W], float32, range approximately [0, 1].
    """

    image = np.asarray(image)

    if image.ndim == 3:
        # If an accidental multi-channel image is supplied,
        # convert it to grayscale.
        if image.shape[2] == 1:
            image = image[..., 0]
        else:
            image = (
                0.299 * image[..., 0]
                + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]
            )

    if image.ndim != 2:
        raise ValueError(
            f"LoFTR expects a 2-D image, got shape {image.shape}"
        )

    image = image.astype(np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        raise ValueError("Input image contains no finite pixels.")

    lo = float(np.nanpercentile(image, 1))
    hi = float(np.nanpercentile(image, 99))

    if hi > lo:
        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image[~finite] = 0.0

    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)

    return tensor


# ---------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------

class FineTunedLoFTR:
    """
    Wrapper around a fine-tuned LoFTR PyTorch-Lightning checkpoint.

    This class intentionally keeps model loading isolated from the rest
    of the registration pipeline.

    Example
    -------
    model = FineTunedLoFTR()

    pts0, pts1, scores = model.match(image0, image1)
    """

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        checkpoint: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):

        self.model_dir = (
            Path(model_dir)
            if model_dir is not None
            else DEFAULT_MODEL_DIR
        )

        self.checkpoint_path = (
            Path(checkpoint)
            if checkpoint is not None
            else _find_checkpoint(self.model_dir)
        )

        self.config_path = (
            Path(config_path)
            if config_path is not None
            else DEFAULT_CONFIG
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = None
        self.config = None

        self._load_model()

    # -----------------------------------------------------------------
    # Model initialization
    # -----------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the original LoFTR Lightning implementation.

        The fine-tuned checkpoint is expected to have been produced
        using the ZJU3DV LoFTR training framework.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"  {self.checkpoint_path}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"LoFTR config not found:\n"
                f"  {self.config_path}"
            )

        try:
            from src.config.default import get_cfg_defaults
            from src.lightning.lightning_loftr import PL_LoFTR
        except ImportError as exc:
            raise ImportError(
                "\nCould not import the original LoFTR training code.\n\n"
                "Your fine-tuned checkpoint appears to come from the "
                "ZJU3DV/PyTorch-Lightning LoFTR implementation.\n\n"
                "The LoFTR source package containing:\n"
                "    src/config/default.py\n"
                "    src/lightning/lightning_loftr.py\n"
                "must be available on PYTHONPATH.\n\n"
                "Do NOT install a random LoFTR package yet. We first "
                "need to use the exact implementation that produced "
                "your checkpoint.\n"
            ) from exc

        # -------------------------------------------------------------
        # Build configuration
        # -------------------------------------------------------------

        config = get_cfg_defaults()

        config.merge_from_file(str(self.config_path))

        self.config = config

        print()
        print("=" * 72)
        print("LOADING FINE-TUNED LOFTR")
        print("=" * 72)
        print(f"Model directory : {self.model_dir}")
        print(f"Config          : {self.config_path}")
        print(f"Checkpoint      : {self.checkpoint_path}")
        print(f"Device          : {self.device}")
        print("=" * 72)

        # -------------------------------------------------------------
        # Lightning model
        # -------------------------------------------------------------

        self.model = PL_LoFTR(
            config,
            pretrained_ckpt=str(self.checkpoint_path),
            profiler=None,
        )

        self.model.eval()
        self.model.to(self.device)

        print("Fine-tuned LoFTR loaded successfully.")
        print()

    # -----------------------------------------------------------------
    # Matching
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match two images.

        Parameters
        ----------
        image0:
            Source image, H x W.

        image1:
            Reference image, H x W.

        Returns
        -------
        pts0:
            Source coordinates, shape [N, 2].
            Coordinates are (x, y).

        pts1:
            Reference coordinates, shape [N, 2].
            Coordinates are (x, y).

        scores:
            LoFTR confidence scores, shape [N].
        """

        if self.model is None:
            raise RuntimeError("LoFTR model has not been loaded.")

        image0_t = _load_gray_tensor(
            image0,
            self.device,
        )

        image1_t = _load_gray_tensor(
            image1,
            self.device,
        )

        batch = {
            "image0": image0_t,
            "image1": image1_t,
        }

        # -------------------------------------------------------------
        # Run LoFTR
        # -------------------------------------------------------------

        output = self.model(batch)

        # -------------------------------------------------------------
        # Extract correspondences.
        #
        # Original LoFTR stores:
        #
        #   mkpts0_f
        #   mkpts1_f
        #   mconf
        #
        # These are already in image pixel coordinates.
        # -------------------------------------------------------------

        pts0 = output.get("mkpts0_f")
        pts1 = output.get("mkpts1_f")
        scores = output.get("mconf")

        if pts0 is None or pts1 is None:
            raise RuntimeError(
                "Fine-tuned LoFTR did not return 'mkpts0_f' "
                "and 'mkpts1_f'.\n"
                f"Available output keys: {list(output.keys())}"
            )

        if scores is None:
            scores = torch.ones(
                pts0.shape[0],
                device=pts0.device,
                dtype=torch.float32,
            )

        pts0 = pts0.detach().cpu().numpy().astype(np.float32)
        pts1 = pts1.detach().cpu().numpy().astype(np.float32)
        scores = scores.detach().cpu().numpy().astype(np.float32)

        # -------------------------------------------------------------
        # Safety filtering
        # -------------------------------------------------------------

        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]

        valid = (
            np.isfinite(pts0).all(axis=1)
            & np.isfinite(pts1).all(axis=1)
            & np.isfinite(scores)
            & (pts0[:, 0] >= 0)
            & (pts0[:, 0] < w0)
            & (pts0[:, 1] >= 0)
            & (pts0[:, 1] < h0)
            & (pts1[:, 0] >= 0)
            & (pts1[:, 0] < w1)
            & (pts1[:, 1] >= 0)
            & (pts1[:, 1] < h1)
        )

        pts0 = pts0[valid]
        pts1 = pts1[valid]
        scores = scores[valid]

        print(
            f"Fine-tuned LoFTR matches: {len(pts0)}"
        )

        return pts0, pts1, scores


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def match_finetuned_loftr(
    image0: np.ndarray,
    image1: np.ndarray,
    model: Optional[FineTunedLoFTR] = None,
    model_dir: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper.

    Returns:
        pts0, pts1, scores
    """

    if model is None:
        model = FineTunedLoFTR(
            model_dir=model_dir,
            checkpoint=checkpoint,
            config_path=config_path,
            device=device,
        )

    return model.match(
        image0,
        image1,
    )


# ---------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("Fine-tuned LoFTR module")
    print("-----------------------")
    print(f"Model directory : {DEFAULT_MODEL_DIR}")
    print(f"Config          : {DEFAULT_CONFIG}")
    print(f"Checkpoint      : {DEFAULT_CHECKPOINT}")
    print()

    model = FineTunedLoFTR()

    print("Model initialization successful.")