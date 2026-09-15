# Chandrayaan-2 (TMC-2 & IIRS) Multi-Sensor Lunar Registration Benchmark Report

**Date:** 2026-09-15 14:29:52  
**Test Set:** 2 Pairs across Authentic Chandrayaan-2 Observations (tmc_optical)  
**Sensors Evaluated:** Chandrayaan-2 TMC-2 Optical (5.0 m/px) & IIRS Hyperspectral (75 m/px)  
**Pipeline:** lunar_registration modular architecture with isolated latency reporting and USAC_MAGSAC consensus  

## Overall Leaderboard

| Model | Inliers | Inlier Ratio | MMA@1px | MMA@3px | RMSE (px) | Corner Err (px) | SSIM | PSNR (dB) | Total Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **eloftr_finetuned** | 3290.0 | 98.4% | 89.8% | 100.0% | 0.69 | 0.21 | 0.940 | 27.8 | 235.9 |
| **pwift** | 340.0 | 97.8% | 89.4% | 100.0% | 0.70 | 0.16 | 0.940 | 27.9 | 3878.7 |

## Isolated Latency Breakdown (De-Aliased)

| Model | Neural Inf (ms) | PWIFT Prep (ms) | Fusion & NCC (ms) | Total Match (ms) | End-to-End (ms) |
| --- | --- | --- | --- | --- | --- |
| **eloftr_finetuned** | 234.2 | 0.0 | 0.0 | 234.2 | 235.9 |
| **pwift** | 0.0 | 3878.3 | 0.0 | 3878.3 | 3878.7 |

## Stratified Performance by Sensor Track

### Track: `TMC_OPTICAL` (2 pairs)

| Model | Inliers | Inlier Ratio | MMA@1px | RMSE (px) | Corner Err (px) | SSIM | PSNR (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **eloftr_finetuned** | 3290.0 | 98.4% | 89.8% | 0.69 | 0.21 | 0.940 | 27.8 |
| **pwift** | 340.0 | 97.8% | 89.4% | 0.70 | 0.16 | 0.940 | 27.9 |
