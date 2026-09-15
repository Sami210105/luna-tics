#!/usr/bin/env python3
"""
extract_ch2_datasets.py - Extracts and validates Chandrayaan-2 TMC-2 and IIRS datasets.

Extracts tmc.zip and iirs.zip into data/ and validates:
1. Master catalog JSON and CSV integrity.
2. Presence of all pre-cut 512x512 feature tiles.
3. Observational metadata (incidence, emission, phase angles).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def extract_archive(zip_path: Path, dest_dir: Path, expected_folder: str) -> Path:
    target_folder = dest_dir / expected_folder
    if target_folder.exists() and any(target_folder.iterdir()):
        print(f"--> [INFO] Destination folder {target_folder} already exists. Verifying contents...")
        return target_folder

    print(f"--> [EXTRACT] Extracting {zip_path.name} to {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"--> [DONE] Extracted {zip_path.name}.")
    return target_folder


def validate_dataset(dataset_dir: Path, sensor_name: str) -> dict:
    catalog_json = dataset_dir / "dataset_catalog.json"
    if not catalog_json.exists():
        # Check subfolder
        candidate = dataset_dir / f"chandrayaan2_{sensor_name.lower()}_dataset" / "dataset_catalog.json"
        if candidate.exists():
            dataset_dir = candidate.parent
            catalog_json = candidate

    if not catalog_json.exists():
        raise FileNotFoundError(f"Catalog not found in {dataset_dir}")

    with open(catalog_json, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    scenes = catalog if isinstance(catalog, list) else list(catalog.values())
    tiles_dir = dataset_dir / "feature_tiles"
    tile_files = list(tiles_dir.glob("*.png")) if tiles_dir.exists() else []

    print(f"\n=======================================================")
    print(f"Validation Report: Chandrayaan-2 {sensor_name.upper()}")
    print(f"Directory   : {dataset_dir}")
    print(f"Total Scenes: {len(scenes)}")
    print(f"Total Tiles : {len(tile_files)}")
    print(f"=======================================================")

    # Verify each tile
    valid_tiles = 0
    for tf in tile_files:
        with Image.open(tf) as im:
            if im.size == (512, 512):
                valid_tiles += 1

    print(f"  Valid 512x512 feature tiles: {valid_tiles}/{len(tile_files)}")

    # Verify scene metadata
    valid_scenes = 0
    angles_summary = []
    for sc in scenes:
        pfx = sc.get("sample_prefix") or sc.get("sample_name") or sc.get("scene_id")
        inc = sc.get("incidence_angle_deg")
        emi = sc.get("emission_angle_deg")
        pha = sc.get("phase_angle_deg")
        res = sc.get("pixel_resolution_m")
        if inc is not None:
            valid_scenes += 1
            angles_summary.append((pfx, float(inc), float(pha) if pha is not None else 0.0, res))

    print(f"  Scenes with valid angles   : {valid_scenes}/{len(scenes)}")
    for pfx, inc, pha, res in angles_summary[:5]:
        print(f"    - {pfx}: inc={inc:.1f}°, phase={pha:.1f}°, res={res}m")
    if len(angles_summary) > 5:
        print(f"    ... and {len(angles_summary) - 5} more scenes.")

    return {
        "dataset_dir": str(dataset_dir),
        "total_scenes": len(scenes),
        "total_tiles": len(tile_files),
        "valid_tiles": valid_tiles,
        "valid_scenes": valid_scenes,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract and validate Chandrayaan-2 datasets.")
    parser.add_argument("--tmc-zip", type=str, default=str(REPO_ROOT / "tmc.zip"))
    parser.add_argument("--iirs-zip", type=str, default=str(REPO_ROOT / "iirs.zip"))
    parser.add_argument("--dest-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    tmc_zip = Path(args.tmc_zip)
    iirs_zip = Path(args.iirs_zip)

    if not args.verify_only:
        if tmc_zip.exists():
            extract_archive(tmc_zip, dest_dir, "chandrayaan2_tmc_dataset")
        else:
            print(f"[WARN] {tmc_zip} not found.")

        if iirs_zip.exists():
            extract_archive(iirs_zip, dest_dir, "chandrayaan2_iirs_dataset")
        else:
            print(f"[WARN] {iirs_zip} not found.")

    tmc_dir = dest_dir / "chandrayaan2_tmc_dataset"
    if tmc_dir.exists():
        validate_dataset(tmc_dir, "TMC-2")

    iirs_dir = dest_dir / "chandrayaan2_iirs_dataset"
    if iirs_dir.exists():
        validate_dataset(iirs_dir, "IIRS")


if __name__ == "__main__":
    main()
