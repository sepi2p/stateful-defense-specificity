#!/usr/bin/env bash
# X0 control corpus: 6 shards, then the P5 summary. Launch with:
#   setsid nohup bash scripts/run_specificity_controls_detached.sh > /dev/null 2>&1 &
# Resumable: rerunning skips sessions already in sessions_shard*.jsonl.
set -uo pipefail
ROOT=/home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
OUT=analysis_outputs/stateful_specificity_controls_20260925
SHARDS=6
cd "$ROOT"
mkdir -p "$OUT"
echo "started $(date -Is) pid $$" > "$OUT/STATUS"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u experiments/gate_trajectory_signatures/run_specificity_controls.py \
    --output-dir "$OUT" --shard "$i" --num-shards "$SHARDS" >> "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
n=$(cat "$OUT"/sessions_shard*.jsonl | wc -l)
echo "shards finished $(date -Is): $n sessions, $failed shard(s) nonzero exit" >> "$OUT/STATUS"
"$PY" experiments/gate_trajectory_signatures/summarize_specificity_controls.py --root "$OUT" > "$OUT/summary.log" 2>&1
echo "summary exit=$? $(date -Is)" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
