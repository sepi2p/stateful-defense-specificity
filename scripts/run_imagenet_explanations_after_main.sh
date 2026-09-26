#!/usr/bin/env bash
# X10: waits for the X9 ImageNet corpus to finish, smoke-tests one calibration image,
# then runs the ImageNet explanation tier on the evaluation split and summarizes it.
#   setsid nohup bash scripts/run_imagenet_explanations_after_main.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
MAIN=analysis_outputs/specificity_imagenet_20260926
OUT=analysis_outputs/explanation_clients_imagenet_20260926
SHARDS=6
mkdir -p "$OUT"
echo "waiting for X9 $(date -Is) pid $$" >> "$OUT/STATUS"
until grep -q DONE "$MAIN/STATUS" 2>/dev/null; do sleep 120; done
COMMON=(--manifest "$MAIN/manifest.csv" --dataset imagenet)
if ! "$PY" -u experiments/gate_trajectory_signatures/run_explanation_clients.py --output-dir "$OUT/smoke" "${COMMON[@]}" \
     --splits calibration --max-images 1 > "$OUT/smoke.log" 2>&1; then
  echo "SMOKE FAILED $(date -Is)" >> "$OUT/STATUS"; echo DONE >> "$OUT/STATUS"; exit 1
fi
echo "smoke ok $(date -Is)" >> "$OUT/STATUS"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u experiments/gate_trajectory_signatures/run_explanation_clients.py \
    --output-dir "$OUT" "${COMMON[@]}" --splits evaluation --shard "$i" --num-shards "$SHARDS" >> "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
  sleep 20
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "shards finished $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) sessions, $failed nonzero" >> "$OUT/STATUS"
"$PY" experiments/gate_trajectory_signatures/summarize_explanation_clients.py --root "$OUT" > "$OUT/summary.log" 2>&1
echo "summary exit=$? $(date -Is)" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
