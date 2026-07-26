#!/usr/bin/env bash
set -e
# TeMP is an older PyTorch-Lightning/DGL codebase. This installation command
# is a starting point for Colab/conda. Adjust CUDA/DGL versions as needed.
pip install -r requirements.txt || true
pip install numpy scipy scikit-learn tqdm
