# Verification note

This folder was checked for repository readiness.

Checked items:
- Python source files compile with `python -m py_compile src/*.py`.
- `src/run_hri.py` runs on ICEWS14 and ICEWS18 in debug mode with `--max-queries`.
- `src/tune_and_run_hri.py` runs in debug mode and writes validation grid, selected configuration, and test summary files.

The full non-debug experiment should still be rerun in the intended Colab/GPU/CPU environment before claiming exact final table reproduction.
