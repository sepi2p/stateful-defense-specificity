#!/usr/bin/env bash
# X16: operating procedure on fresh ImageNet images, and the practical result on a second model.
#   setsid nohup bash scripts/run_x16_detached.sh > /dev/null 2>&1 < /dev/null &
# Waits until no process uses the graphics card, then runs the stages in order. Every runner is
# resumable: a stage that is started again skips the sessions it has already written.
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
G=experiments/gate_trajectory_signatures
OUT=analysis_outputs/operating_profile_imagenet_20260928
export EXPLANATION_LIBS=${EXPLANATION_LIBS:-/tmp/claude-1000/-home-sepi-projects-MCG-Blackbox/19c044a2-8080-46c9-9cf3-cf349e8c7ea3/scratchpad/libs}
mkdir -p "$OUT"
log() { echo "$1 $(date -Is)" >> "$OUT/STATUS"; }
disk_ok() {
  local free; free=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 3 ]; then log "ABORT: only ${free} GB free on disk before $1"; exit 1; fi
}
wait_gpu() {
  local idle=0
  while [ "$idle" -lt 2 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 60
  done
}
run_shards() {  # name, shards, command...
  local name=$1 shards=$2; shift 2
  local pids=() failed=0
  disk_ok "$name"
  for i in $(seq 0 $((shards - 1))); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$@" --shard "$i" --num-shards "$shards" >> "$OUT/$name.shard$i.log" 2>&1 &
    pids+=($!)
    sleep 20  # stagger CUDA initialization
  done
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  log "$name done, $failed shard(s) nonzero"
}

log "queued (pid $$); waiting for an idle graphics card"
wait_gpu
log "graphics card idle; starting"

[ -f "$OUT/manifest.csv" ] || "$PY" $G/extend_imagenet_manifest.py --output "$OUT/manifest.csv" > "$OUT/manifest.log" 2>&1
[ -f "$OUT/manifest.csv" ] || { log "ABORT: manifest extension failed"; exit 1; }
log "manifest: $(grep -c confirmation "$OUT/manifest.csv") confirmation images"

run_shards workloads 8 "$PY" -u $G/run_specificity_workloads.py --stage main --output-dir "$OUT/workloads" \
  --manifest "$OUT/manifest.csv" --dataset imagenet --model-name imagenet_resnet50 --splits calibration,confirmation \
  --objectives attack,confidence_boost --nes-tile 8 --nes-step-255 1 --lfc-seed 20260926
run_shards libraries_resnet50 3 "$PY" -u $G/run_library_explanations.py --output-dir "$OUT/libraries_resnet50" \
  --manifest "$OUT/manifest.csv" --dataset imagenet --splits calibration,confirmation
run_shards controls 3 "$PY" -u $G/run_specificity_controls.py --output-dir "$OUT/controls" --manifest "$OUT/manifest.csv" \
  --dataset imagenet --splits calibration,confirmation --controls shuffled --lfc-seed 20260926
run_shards libraries_convnext 2 "$PY" -u $G/run_library_explanations.py --output-dir "$OUT/libraries_convnext" \
  --manifest "$OUT/manifest.csv" --dataset imagenet --splits confirmation --imagenet-model convnext_tiny --no-detectors

for d in workloads libraries_resnet50 controls libraries_convnext; do
  log "sessions in $d: $(cat "$OUT/$d"/sessions_shard*.jsonl 2>/dev/null | wc -l)"
done

"$PY" $G/run_enforced_explanations.py --source-dir "$OUT/libraries_resnet50" --output-dir "$OUT/enforced_resnet50" \
  --dataset imagenet > "$OUT/enforced_resnet50.log" 2>&1; log "replay resnet50 exit=$?"
"$PY" $G/run_enforced_explanations.py --source-dir "$OUT/libraries_convnext" --mask-dir "$OUT/libraries_resnet50" \
  --output-dir "$OUT/enforced_convnext" --dataset imagenet > "$OUT/enforced_convnext.log" 2>&1; log "replay convnext exit=$?"

for m in resnet50 convnext; do
  model=$([ $m = convnext ] && echo convnext_tiny || echo resnet50)
  splits=$([ $m = convnext ] && echo confirmation || echo calibration,confirmation)
  "$PY" $G/analyze_explanation_validity.py --source-dir "$OUT/libraries_$m" --enforced-dir "$OUT/enforced_$m" \
    --output-dir "$OUT/validity_$m" --dataset imagenet --splits "$splits" --imagenet-model "$model" > "$OUT/validity_$m.log" 2>&1
  log "validity $m exit=$?"
  "$PY" $G/analyze_explanation_prior.py --source-dir "$OUT/libraries_$m" --enforced-dir "$OUT/enforced_$m" \
    --output-dir "$OUT/validity_$m" --dataset imagenet --splits "$splits" --imagenet-model "$model" > "$OUT/prior_$m.log" 2>&1
  log "prior $m exit=$?"
  "$PY" $G/analyze_centre_prior_r2.py --source-dir "$OUT/libraries_$m" --output-dir "$OUT/validity_$m" --dataset imagenet \
    --splits "$splits" --imagenet-model "$model" > "$OUT/centre_r2_$m.log" 2>&1
  log "centre prior R2 $m exit=$?"
done

"$PY" $G/analyze_operating_profile.py --setting imagenet --output-dir analysis_outputs/operating_profile_r1 > "$OUT/profile.log" 2>&1
log "profile exit=$?"
log FINISHED
