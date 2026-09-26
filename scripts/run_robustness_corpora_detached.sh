#!/usr/bin/env bash
# X4-X6: robustness corpora, run back to back, each followed by the frozen AUROC analysis.
#   setsid nohup bash scripts/run_robustness_corpora_detached.sh > /dev/null 2>&1 < /dev/null &
# Resumable: finished sessions are skipped; a finished GTSRB model is reused.
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
LOG=analysis_outputs/robustness_corpora_20260926.STATUS
SHARDS=6
LFC=20260926
echo "started $(date -Is) pid $$" >> "$LOG"

corpus () {  # $1 = out dir; remaining args go to run_specificity_workloads.py
  local out=$1; shift
  mkdir -p "$out"
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u experiments/gate_trajectory_signatures/run_specificity_workloads.py \
      --stage main --output-dir "$out" --manifest "$out/manifest.csv" --lfc-seed "$LFC" \
      --shard "$i" --num-shards "$SHARDS" "$@" >> "$out/shard$i.log" 2>&1 &
    pids+=($!)
    # shard 0 builds the manifest if it does not exist yet; let it finish before the others start
    if [ "$i" -eq 0 ] && [ ! -f "$out/manifest.csv" ]; then
      until [ -f "$out/manifest.csv" ] || ! kill -0 "${pids[0]}" 2>/dev/null; do sleep 5; done
    fi
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  echo "$(basename "$out") corpus $(date -Is): $(cat "$out"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$LOG"
  "$PY" experiments/gate_trajectory_signatures/analyze_specificity_workloads.py --root "$out" > "$out/analysis.log" 2>&1
  echo "$(basename "$out") analysis exit=$? $(date -Is)" >> "$LOG"
}

# X5 prerequisite: GTSRB ResNet-18 at 32 px
G=analysis_outputs/eaai_gtsrb/checkpoints_32px
if [ ! -f "$G/resnet18/best.pt" ]; then
  "$PY" -u experiments/eaai_gtsrb/train_gtsrb_models.py --data-dir data/gtsrb --output-dir "$G" \
    --models resnet18 --image-size 32 --epochs 15 > analysis_outputs/gtsrb32_training.log 2>&1
  echo "gtsrb32 training exit=$? $(date -Is)" >> "$LOG"
fi

# X4: SimBA on the frozen CIFAR-10 manifest
mkdir -p analysis_outputs/specificity_simba_20260926
cp -n analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv analysis_outputs/specificity_simba_20260926/manifest.csv
corpus analysis_outputs/specificity_simba_20260926 --optimizer simba --objectives attack,restore,confidence_boost,boundary_probe,counterfactual

# X6: second models (own manifests)
corpus analysis_outputs/specificity_vgg19bn_20260926 --model-name bbb_vgg19_bn
corpus analysis_outputs/specificity_robust_engstrom_20260926 --model-name robustbench_Engstrom2019Robustness
corpus analysis_outputs/specificity_resnet18_seed1_20260926 --model-name resnet18_seed1 --checkpoint checkpoints/cifar10_resnet18_seed_study/resnet18_seed1.pt
corpus analysis_outputs/specificity_resnet18_seed2_20260926 --model-name resnet18_seed2 --checkpoint checkpoints/cifar10_resnet18_seed_study/resnet18_seed2.pt

# X5: GTSRB 32 px
corpus analysis_outputs/specificity_gtsrb32_20260926 --dataset gtsrb32 --model-name gtsrb32 --checkpoint "$G/resnet18/best.pt"

echo "ALL DONE $(date -Is)" >> "$LOG"
