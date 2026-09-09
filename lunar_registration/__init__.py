"""
lunar_registration
===================
End-to-end pipeline for SIH PS: "Multi-modal, Sun angle and scale invariant
image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)".

Stages:
    1. preprocessing   - metadata parsing, sensor detection, normalization, resampling
    2. illumination    - per-sensor illumination-invariance handling (PWIFT/Akimov,
                          histogram/shadow/log for TMC, CLAHE/inversion/dilation for IIRS)
    3. matching         - PWIFT matcher + EfficientLoFTR, compared and merged
    4. viewpoint        - homography estimation (global/local) + RANSAC cleanup
    5. scale            - multi-scale pyramid search per sensor
    6. georeference     - warp source onto reference, write registered product + GCPs
    metrics             - RMSE, inlier count/ratio, spatial uniformity

See README.md for setup and CLI usage.
"""

__version__ = "0.1.0"
