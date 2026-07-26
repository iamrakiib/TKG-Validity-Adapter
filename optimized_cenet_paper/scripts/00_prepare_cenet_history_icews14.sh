#!/usr/bin/env bash
set -e
cd data/ICEWS14
if [ ! -f valid.txt ]; then
  echo "ERROR: data/ICEWS14/valid.txt is missing in this upstream CENET copy."
  echo "For paper reproduction, place the official ICEWS14 valid.txt here before running CENET history preprocessing."
  exit 1
fi
python get_history_graph.py
