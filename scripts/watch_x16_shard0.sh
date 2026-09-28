#!/usr/bin/env bash
# Relaunches libraries_resnet50 shard 0 of X16 if it stops before its 201 sessions are written
# (it was killed once by the kernel for lack of memory). The runner is resumable.
cd /home/sepi/projects/MCG-Blackbox
OUT=analysis_outputs/operating_profile_imagenet_20260928
export EXPLANATION_LIBS=/tmp/claude-1000/-home-sepi-projects-MCG-Blackbox/19c044a2-8080-46c9-9cf3-cf349e8c7ea3/scratchpad/libs
for attempt in 1 2 3 4 5; do
  while pgrep -f "run_library_explanations.py --output-dir $OUT/libraries_resnet50 .*--shard 0" > /dev/null; do sleep 60; done
  n=$(grep -c . "$OUT/libraries_resnet50/sessions_shard0.jsonl")
  [ "$n" -ge 201 ] && { echo "watch: shard 0 complete ($n sessions) $(date -Is)" >> "$OUT/STATUS"; exit 0; }
  echo "watch: shard 0 stopped at $n sessions; relaunch $attempt $(date -Is)" >> "$OUT/STATUS"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 setsid nohup /home/sepi/jupyterenv/bin/python -u experiments/gate_trajectory_signatures/run_library_explanations.py \
    --output-dir "$OUT/libraries_resnet50" --manifest "$OUT/manifest.csv" --dataset imagenet --splits calibration,confirmation \
    --shard 0 --num-shards 3 >> "$OUT/libraries_resnet50.shard0.relaunch.log" 2>&1 < /dev/null &
  sleep 120
done
