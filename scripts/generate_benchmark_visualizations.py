#!/usr/bin/env python3
"""
generate_benchmark_visualizations.py - Generates comparison plots and visual artifacts
from the NAC PHO benchmark run.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark_results" / "nac_pho_benchmark"
ARTIFACT_DIR = Path("/home/ojas/.gemini/antigravity-cli/brain/70c20969-23bd-4f2a-9b72-85062d73bcd4")

def main():
    lb_path = BENCHMARK_DIR / "leaderboard.csv"
    pp_path = BENCHMARK_DIR / "per_pair_metrics.csv"
    if not lb_path.exists() or not pp_path.exists():
        print("Leaderboard or per-pair csv missing.")
        return

    df_lb = pd.read_csv(lb_path)
    df_pp = pd.read_csv(pp_path)

    # 1. Leaderboard multi-metric comparison chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = df_lb["Model"].tolist()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Inliers & Candidates
    x = np.arange(len(models))
    width = 0.35
    axes[0, 0].bar(x - width/2, df_lb["Avg Candidates"], width, label="Candidates", color="#aec7e8")
    axes[0, 0].bar(x + width/2, df_lb["Avg Inliers"], width, label="Inliers (MAGSAC++)", color="#1f77b4")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Match Count")
    axes[0, 0].set_title("Correspondence Yield: Candidates vs. Inliers", fontsize=12, fontweight="bold")
    axes[0, 0].legend()

    # Accuracy: MMA@1px vs MMA@3px
    axes[0, 1].bar(x - width/2, df_lb["MMA@1px (%)"], width, label="MMA@1px (%)", color="#2ca02c")
    axes[0, 1].bar(x + width/2, df_lb["MMA@3px (%)"], width, label="MMA@3px (%)", color="#98df8a")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Mean Matching Accuracy (MMA)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend()

    # Geometric Precision: Corner Transfer Error & RMSE
    axes[1, 0].bar(x - width/2, df_lb["Corner Err (px)"], width, label="Corner Transfer Err (px)", color="#d62728")
    axes[1, 0].bar(x + width/2, df_lb["RMSE (px)"], width, label="Reprojection RMSE (px)", color="#ff9896")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Error (pixels)")
    axes[1, 0].set_title("Geometric Homography Precision (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1, 0].legend()

    # Computational Latency
    axes[1, 1].bar(x, df_lb["Match Latency (ms)"], 0.5, color="#ff7f0e", alpha=0.85)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].set_title("GPU Feature Matching Latency", fontsize=12, fontweight="bold")
    for i, v in enumerate(df_lb["Match Latency (ms)"]):
        axes[1, 1].text(i, v + 50, f"{v:.1f} ms", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("LROC NAC PHO Benchmark - Lunar Registration Pipeline Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file1 = BENCHMARK_DIR / "benchmark_leaderboard_summary.png"
    plt.savefig(out_file1, dpi=200, bbox_inches="tight")
    if ARTIFACT_DIR.exists():
        plt.savefig(ARTIFACT_DIR / "benchmark_leaderboard_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved leaderboard plot to: {out_file1}")

    # 2. Moderate vs Severe Regimes
    fig, ax = plt.subplots(figsize=(10, 6))
    regime_grp = df_pp.groupby(["model", "regime"])["corner_err_px"].median().unstack()
    regime_grp = regime_grp.reindex(models)

    x = np.arange(len(models))
    ax.bar(x - width/2, regime_grp["moderate"], width, label="Moderate Regime", color="#3498db")
    ax.bar(x + width/2, regime_grp["severe"], width, label="Severe Regime (Tilt + Relighting)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Median Corner Transfer Error (px)")
    ax.set_title("Robustness Across Illumination & Perspective Regimes", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    out_file2 = BENCHMARK_DIR / "benchmark_regimes_comparison.png"
    plt.savefig(out_file2, dpi=200, bbox_inches="tight")
    if ARTIFACT_DIR.exists():
        plt.savefig(ARTIFACT_DIR / "benchmark_regimes_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved regime plot to: {out_file2}")

if __name__ == "__main__":
    main()
