#!/usr/bin/env bash
set -euo pipefail

# Example Colab terminal usage after uploading/cloning this folder:
#   cd optimized_hri
#   bash scripts/colab_setup_and_run.sh

python -m pip install -r requirements.txt
bash scripts/run_hri_both.sh
