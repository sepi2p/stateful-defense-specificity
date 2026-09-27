#!/usr/bin/env bash
# X14: explanation clients from libraries (lime, Captum). CIFAR-10 first, then ImageNet.
#   EXPLANATION_LIBS=<dir with lime and captum> setsid nohup bash scripts/run_library_explanations_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
BASE=analysis_outputs/explanation_libraries_20260927
mkdir -p "$BASE"
echo "started $(date -Is) pid $$ libs=${EXPLANATION_LIBS:-environment}" >> "$BASE/STATUS"
run_part () {
  local name=$1 shards=$2; shift 2
  local out="$BASE/$name"
  mkdir -p "$out"
  local pids=()
  for i in $(seq 0 $((shards - 1))); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u experiments/gate_trajectory_signatures/run_library_explanations.py \
      --output-dir "$out" "$@" --shard "$i" --num-shards "$shards" >> "$out/shard$i.log" 2>&1 &
    pids+=($!)
    sleep 20
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  echo "$name $(date -Is): $(cat "$out"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$BASE/STATUS"
}
run_part cifar10 8 --dataset cifar10
run_part imagenet 6 --dataset imagenet --manifest analysis_outputs/specificity_imagenet_20260926/manifest.csv --one-image-per-class
echo DONE >> "$BASE/STATUS"
