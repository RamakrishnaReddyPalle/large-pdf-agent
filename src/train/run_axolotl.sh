#!/usr/bin/env bash
set -euo pipefail

# Requires: conda/venv with axolotl installed, CUDA available, and HF_TOKEN set (gated model).
# If on Windows, run via WSL2.

CFG="configs/axolotl_lora.yaml"

echo "[Axolotl] Using config: $CFG"
accelerate launch -m axolotl.cli.train $CFG
