#!/usr/bin/env bash
# run_pipeline.sh - Luna-tics production execution wrapper
# Configures memory allocator safeguards for constrained GPUs (e.g. RTX 3050 6GB)

set -euo pipefail

# Enable PyTorch CUDA expandable segments to avoid VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Execute the lunar registration pipeline with passed arguments
python -m lunar_registration.pipeline "$@"
