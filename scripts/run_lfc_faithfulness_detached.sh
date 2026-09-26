#!/usr/bin/env bash
# Lee-Fang-Chang reimplementation faithfulness check (50 calibration images).
#   setsid nohup bash scripts/run_lfc_faithfulness_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
OUT=analysis_outputs/lfc_faithfulness_20260926
mkdir -p "$OUT"
echo "started $(date -Is)" > "$OUT/STATUS"
OMP_NUM_THREADS=6 /home/sepi/jupyterenv/bin/python -u experiments/gate_trajectory_signatures/lfc_faithfulness.py \
  --images 50 --bern-scales 0.0,0.002,0.01 --output-dir "$OUT" > "$OUT/run.log" 2>&1
echo "exit=$? $(date -Is)" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
