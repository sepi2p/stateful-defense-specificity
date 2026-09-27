#!/usr/bin/env bash
# X13: sensitivity of the matched-objective result to the NES configuration (3 variants, 8 shards each).
#   setsid nohup bash scripts/run_nes_sensitivity_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
BASE=analysis_outputs/specificity_nes_sensitivity_20260927
SHARDS=8
mkdir -p "$BASE"
echo "started $(date -Is) pid $$" >> "$BASE/STATUS"
run_variant () {
  local name=$1 step=$2 accept=$3
  local out="$BASE/$name"
  mkdir -p "$out"
  cp -n analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv "$out/manifest.csv"
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u experiments/gate_trajectory_signatures/run_specificity_workloads.py \
      --stage main --output-dir "$out" --manifest "$out/manifest.csv" --optimizer nes \
      --objectives attack,restore,confidence_boost --splits fit,evaluation --lfc-seed 20260926 \
      --nes-step-255 "$step" --nes-accept "$accept" \
      --shard "$i" --num-shards "$SHARDS" >> "$out/shard$i.log" 2>&1 &
    pids+=($!)
    sleep 15
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  echo "$name $(date -Is): $(cat "$out"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$BASE/STATUS"
}
run_variant v1_step1 1.0 improve
run_variant v2_step2 2.0 improve
run_variant v3_step2_always 2.0 always
echo DONE >> "$BASE/STATUS"
