#!/usr/bin/env bash
# Completes X16 after the relaunch of libraries_resnet50 shard 0: waits for the launcher and the
# relaunched shard, then repeats the replays (resumable) and every analysis on the complete sessions.
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
G=experiments/gate_trajectory_signatures
OUT=analysis_outputs/operating_profile_imagenet_20260928
export EXPLANATION_LIBS=${EXPLANATION_LIBS:-/tmp/claude-1000/-home-sepi-projects-MCG-Blackbox/19c044a2-8080-46c9-9cf3-cf349e8c7ea3/scratchpad/libs}
log() { echo "$1 $(date -Is)" >> "$OUT/STATUS"; }
until grep -q "^FINISHED" "$OUT/STATUS" && ! pgrep -f "run_library_explanations.py --output-dir $OUT/libraries_resnet50" > /dev/null; do sleep 60; done
log "finish: sessions libraries_resnet50 $(cat "$OUT"/libraries_resnet50/sessions_shard*.jsonl | wc -l), convnext $(cat "$OUT"/libraries_convnext/sessions_shard*.jsonl | wc -l), controls $(cat "$OUT"/controls/sessions_shard*.jsonl | wc -l), workloads $(cat "$OUT"/workloads/sessions_shard*.jsonl | wc -l)"
"$PY" $G/run_enforced_explanations.py --source-dir "$OUT/libraries_resnet50" --output-dir "$OUT/enforced_resnet50" --dataset imagenet >> "$OUT/enforced_resnet50.log" 2>&1; log "finish: replay resnet50 exit=$?"
"$PY" $G/run_enforced_explanations.py --source-dir "$OUT/libraries_convnext" --mask-dir "$OUT/libraries_resnet50" --output-dir "$OUT/enforced_convnext" --dataset imagenet >> "$OUT/enforced_convnext.log" 2>&1; log "finish: replay convnext exit=$?"
for m in resnet50 convnext; do
  model=$([ $m = convnext ] && echo convnext_tiny || echo resnet50)
  splits=$([ $m = convnext ] && echo confirmation || echo calibration,confirmation)
  "$PY" $G/analyze_explanation_validity.py --source-dir "$OUT/libraries_$m" --enforced-dir "$OUT/enforced_$m" --output-dir "$OUT/validity_$m" --dataset imagenet --splits "$splits" --imagenet-model "$model" > "$OUT/validity_$m.log" 2>&1; log "finish: validity $m exit=$?"
  "$PY" $G/analyze_explanation_prior.py --source-dir "$OUT/libraries_$m" --enforced-dir "$OUT/enforced_$m" --output-dir "$OUT/validity_$m" --dataset imagenet --splits "$splits" --imagenet-model "$model" > "$OUT/prior_$m.log" 2>&1; log "finish: prior $m exit=$?"
  "$PY" $G/analyze_centre_prior_r2.py --source-dir "$OUT/libraries_$m" --output-dir "$OUT/validity_$m" --dataset imagenet --splits "$splits" --imagenet-model "$model" > "$OUT/centre_r2_$m.log" 2>&1; log "finish: centre prior R2 $m exit=$?"
done
"$PY" $G/analyze_operating_profile.py --setting imagenet --output-dir analysis_outputs/operating_profile_r1 > "$OUT/profile.log" 2>&1; log "finish: profile exit=$?"
log "COMPLETE"
