#!/usr/bin/env bash
# X11: acceptance-throttled SimBA attack (fit + evaluation images), 6 shards.
#   setsid nohup bash scripts/run_simba_throttled_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
OUT=analysis_outputs/specificity_simba_throttled_20260927
SHARDS=6
mkdir -p "$OUT"
cp -n analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv "$OUT/manifest.csv"
echo "started $(date -Is) pid $$" >> "$OUT/STATUS"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u experiments/gate_trajectory_signatures/run_specificity_workloads.py \
    --stage main --output-dir "$OUT" --manifest "$OUT/manifest.csv" --optimizer simba --objectives attack_throttled \
    --throttle-rates "$OUT/throttle_rates_restore_fit.json" --splits fit,evaluation --lfc-seed 20260926 \
    --shard "$i" --num-shards "$SHARDS" >> "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
  sleep 10
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "corpus $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
