#!/usr/bin/env bash
# X14: resume the ImageNet part (two shards stopped without an error message at 253 of 300 sessions).
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
BASE=analysis_outputs/explanation_libraries_20260927
out="$BASE/imagenet"
echo "resume $(date -Is) pid $$" >> "$BASE/STATUS"
pids=()
for i in 0 1 2 3 4 5; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u experiments/gate_trajectory_signatures/run_library_explanations.py \
    --output-dir "$out" --dataset imagenet --manifest analysis_outputs/specificity_imagenet_20260926/manifest.csv --one-image-per-class \
    --shard "$i" --num-shards 6 >> "$out/shard$i.log" 2>&1 &
  pids+=($!)
  sleep 20
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "imagenet resumed $(date -Is): $(cat "$out"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$BASE/STATUS"
echo DONE >> "$BASE/STATUS"
