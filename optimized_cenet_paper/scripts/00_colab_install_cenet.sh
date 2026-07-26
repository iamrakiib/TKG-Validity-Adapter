#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_hva_extra.txt
