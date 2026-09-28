#!/usr/bin/env bash
# X15, part 1: library explanation clients under rejection (replay of the X14 sessions).
set -u
cd "$(dirname "$0")/.."
OUT=analysis_outputs/explanation_enforcement_20260927
PY=/home/sepi/jupyterenv/bin/python
export EXPLANATION_LIBS=${EXPLANATION_LIBS:-/tmp/claude-1000/-home-sepi-projects-MCG-Blackbox/19c044a2-8080-46c9-9cf3-cf349e8c7ea3/scratchpad/libs}
mkdir -p "$OUT/imagenet" "$OUT/cifar10"
echo "started $(date -Is)" > "$OUT/STATUS"
for s in 0 1 2; do
  $PY experiments/gate_trajectory_signatures/run_enforced_explanations.py --source-dir analysis_outputs/explanation_libraries_20260927/imagenet \
      --output-dir "$OUT/imagenet" --dataset imagenet --shard $s --num-shards 3 > "$OUT/imagenet/shard$s.log" 2>&1 &
done
$PY experiments/gate_trajectory_signatures/run_enforced_explanations.py --source-dir analysis_outputs/explanation_libraries_20260927/cifar10 \
    --output-dir "$OUT/cifar10" --dataset cifar10 > "$OUT/cifar10/shard0.log" 2>&1 &
wait
echo "finished $(date -Is)" >> "$OUT/STATUS"
