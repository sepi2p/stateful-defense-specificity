#!/usr/bin/env bash
# Written by a Claude session on 2026-09-27. Shard 3 crashed at launch (CUDA OOM caused by that session's concurrent GPU experiment)
# and was relaunched outside the launcher (pid 302861). The launcher (pid 297200) will therefore report "1 nonzero" and summarize
# without shard 3 if it finishes first. This watcher waits for both, then reruns the summary so it covers all six shards.
cd /home/sepi/projects/MCG-Blackbox
OUT=analysis_outputs/explanation_clients_imagenet_20260926
while kill -0 297200 2>/dev/null || kill -0 302861 2>/dev/null; do sleep 60; done
/home/sepi/jupyterenv/bin/python experiments/gate_trajectory_signatures/summarize_explanation_clients.py --root "$OUT" > "$OUT/summary.log" 2>&1
echo "shard3 relaunch complete; summary rerun over all shards exit=$? $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) sessions" >> "$OUT/STATUS"
