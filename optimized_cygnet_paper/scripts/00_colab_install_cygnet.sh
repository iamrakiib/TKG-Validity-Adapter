#!/usr/bin/env bash
set -euo pipefail
pip install -r requirements.txt
pip install -r requirements_hva_extra.txt
python -m hva.smoke_test_cygnet_hva
