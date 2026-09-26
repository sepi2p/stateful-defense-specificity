#!/usr/bin/env bash
# X9: ImageNet corpus (224 px, ResNet-50), 10 shards, then the frozen analysis and P8-P11 scoring.
#   setsid nohup bash scripts/run_imagenet_corpus_detached.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
OUT=analysis_outputs/specificity_imagenet_20260926
SHARDS=10
mkdir -p "$OUT"
cp -n /tmp/claude-1000/-home-sepi-projects-MCG-Blackbox/19c044a2-8080-46c9-9cf3-cf349e8c7ea3/scratchpad/imagenet_manifest.csv "$OUT/manifest.csv"
echo "started $(date -Is) pid $$" >> "$OUT/STATUS"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u experiments/gate_trajectory_signatures/run_specificity_workloads.py \
    --stage main --output-dir "$OUT" --manifest "$OUT/manifest.csv" --dataset imagenet --model-name imagenet_resnet50 \
    --splits fit,evaluation --nes-tile 8 --nes-step-255 1 --lfc-seed 20260926 \
    --shard "$i" --num-shards "$SHARDS" >> "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
  sleep 20  # stagger CUDA/cuDNN initialization; 10 simultaneous starts failed with CUDNN_STATUS_INTERNAL_ERROR
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "corpus $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$OUT/STATUS"
"$PY" experiments/gate_trajectory_signatures/analyze_specificity_workloads.py --root "$OUT" > "$OUT/analysis.log" 2>&1
echo "analysis exit=$? $(date -Is)" >> "$OUT/STATUS"
"$PY" experiments/gate_trajectory_signatures/score_robustness_corpus.py "$OUT" > "$OUT/score.log" 2>&1
echo "score exit=$? $(date -Is)" >> "$OUT/STATUS"
echo DONE >> "$OUT/STATUS"
