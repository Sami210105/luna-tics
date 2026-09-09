# Lunar Image Registration Pipeline (SIH PS: CH2 OHRC/TMC/IIRS <-> LROC)

Full 6-stage pipeline for multi-modal, sun-angle- and scale-invariant image
correspondence between Chandrayaan-2 optical images (OHRC/TMC/IIRS) and a
lunar reference image (LROC NAC/WAC), producing a registered product +
match points + evaluation metrics.

## What's actually been tested here vs. what you need to verify

This was built and debugged in a sandbox **without** ISIS3, GDAL, or your
real image files, and without internet access to download PyTorch. So:

| Component | Status |
|---|---|
| Stage 1-2, 4-6 (preprocessing, illumination branches, viewpoint, scale, georef, metrics) | Ran end-to-end on a synthetic cratered surface with simulated relighting + warp; two real bugs found and fixed (a descriptor-matching memory blowup, and an off-by-one in the log-Gabor filter that silently misaligned image/filter sizes on odd dimensions) |
| PWIFT matcher | Same synthetic test - produces plausible-looking matches and a registered image, but accuracy on synthetic data was mediocre (RMSE ~250px with only 16 inliers) - **needs real-data tuning**, see below |
| EfficientLoFTR matcher | **Not runnable in this sandbox** (no disk space / network access for `torch`) - code is written against the documented HuggingFace `transformers` `keypoint-matching` pipeline API, but you must test it on your machine before trusting it |
| ISIS-dependent bits (.cub reading, angle maps) | Not testable here at all - written against your existing file layout (processed/angle_maps/*.tif) but unverified against your actual files |

**Bottom line: the plumbing works, but do not treat the accuracy numbers or
the EfficientLoFTR path as validated until you run it on your real LROC
images (you already have 3) and a real CH2 image.**

## Setup

```bash
pip install -r requirements.txt
# for the EfficientLoFTR stage specifically:
pip install torch transformers
```

`rasterio` and `planetaryimage` are optional but recommended - without them
you fall back to a minimal raw-bytes PDS3 reader (works for simple uint8
products like your existing 3 LROC images) and lose GeoTIFF output / CRS
tagging.

## Quick sanity check (no real data needed)

```bash
python test_synthetic.py
```

Generates a synthetic cratered surface, relights it under two different sun
angles, applies a rotation/scale/translation warp, and runs the full
pipeline (PWIFT only, since EfficientLoFTR needs torch) on the result.
Confirms the code runs; does **not** validate real accuracy.

## Running on your real data

You already have the 3 LROC images + ISIS-processed angle maps for the
PWIFT-only leg of this project (see your working window
`--window 2243,298,512,512`). Example:

```bash
python -m lunar_registration.pipeline \
    --source path/to/M129133239RE.cub \
    --source-sensor LROC \
    --reference path/to/M150368601RE.cub \
    --source-incidence processed/angle_maps/M129133239RE_incidence.tif \
    --source-emission processed/angle_maps/M129133239RE_emission.tif \
    --window 2243,298,512,512 \
    --out-dir outputs/lroc_pair_1 \
    --no-eloftr   # drop this flag once you've verified EfficientLoFTR locally
```

For an actual CH2 OHRC image against an LROC reference:

```bash
python -m lunar_registration.pipeline \
    --source path/to/ch2_ohrc_product.img \
    --source-sensor OHRC \
    --reference path/to/lroc_reference.cub \
    --out-dir outputs/ohrc_vs_lroc
```

Outputs land in `--out-dir`: `<name>_registered.png` (+ `.tif` if rasterio
and a georeferenced reference are available), `<name>_matchpoints.csv`, and
`summary.json` with the metrics table for whichever of PWIFT/EfficientLoFTR
won (more RANSAC inliers).

## Where to focus tuning effort

1. **`config.py` -> `PipelineConfig`**: `pwift_keypoint_threshold` and
   `pwift_ratio_test` are the first knobs to turn if you get too few/too
   many matches. `ransac_reproj_threshold_px` controls how strict "inlier"
   means for your RMSE metric.
2. **`pwift.py` -> `akimov_weight()`**: currently a Lommel-Seeliger-style
   approximation, not transcribed from PWIFT.pdf's exact equation - check
   the paper and swap in the real Akimov disk function if it differs.
3. **`pwift.py` -> `BiChannelDescriptor` / `compute_descriptors()`**: current
   descriptor is a rotation-normalized HOG-style histogram over
   intensity+phase-congruency channels - reasonable but not paper-exact;
   the paper's descriptor is likely more specialized.
4. **`matching.py` -> EfficientLoFTR**: verify zero-shot quality on a real
   lunar pair first. If poor, the MoonAnything/LunarPhoto synthetic dataset
   (multi-illumination renders) is a good fine-tuning source before you rely
   on it in the final comparison table.
5. **`scale.py` -> `select_best_scale()`**: the coarse-to-fine search now
   correlates on gradient-magnitude (illumination-robust proxy) rather than
   raw pixels - reasonable but crude; if scale estimates are consistently
   off on real data, consider running it on the actual phase-congruency maps
   instead (more expensive, more accurate).

## File map

```
lunar_registration/
  config.py          sensor configs (OHRC/TMC/IIRS/LROC) + PipelineConfig knobs
  preprocessing.py   stage 1: load .img/.cub/.tif, sensor detection, angle maps, normalize
  illumination.py    stage 2: per-sensor illumination branch dispatch
  pwift.py           PWIFT core: phase congruency, Akimov weighting, keypoints,
                      descriptors, swap-aware matching, coarse-to-fine search, FSC+homography
  matching.py         stage 3: runs PWIFT + EfficientLoFTR, returns both for comparison
  viewpoint.py        stage 4: global/local homography estimation + RANSAC
  scale.py            stage 5: multi-scale pyramid + best-scale/rotation search
  georeference.py     stage 6: warp + write registered image, match-points CSV, GeoTIFF
  metrics.py          RMSE, inlier count/ratio, spatial uniformity score
  pipeline.py          orchestrator + CLI entry point
test_synthetic.py      synthetic end-to-end sanity check
requirements.txt
```
