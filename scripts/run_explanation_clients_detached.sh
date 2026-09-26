#!/usr/bin/env bash
# X2: explanation-traffic clients, 6 shards, then the P6 summary.
#   setsid nohup bash scripts/run_explanation_clients_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
OUT=analysis_outputs/explanation_clients_20260926
SHARDS=6
mkdir -p "$OUT"
echo "started $(date -Is) pid $$" > "$OUT/STATUS"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u experiments/gate_trajectory_signatures/run_explanation_clients.py \
    --output-dir "$OUT" --shard "$i" --num-shards "$SHARDS" >> "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "shards finished $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$OUT/STATUS"
"$PY" experiments/gate_trajectory_signatures/summarize_explanation_clients.py --root "$OUT" > "$OUT/summary.log" 2>&1
echo "summary exit=$? $(date -Is)" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
