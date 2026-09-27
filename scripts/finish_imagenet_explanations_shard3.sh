#!/usr/bin/env bash
# X10 follow-up. Shard 3 died with CUDA OOM at startup; a shard-3 worker was later started
# outside the launcher (PID 302861, 10:24:32). Wait until the launcher is done AND no
# shard-3 worker is running, then run shard 3 once more (resumable: only missing sessions
# are generated), check for duplicate session ids, and write the final summary.
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
OUT=analysis_outputs/explanation_clients_imagenet_20260926
shard3_running () {
  ps -eo args | grep "run_explanation_clients.py" | grep "explanation_clients_imagenet" | grep -q -- "--shard 3 "
}
until grep -q DONE "$OUT/STATUS" 2>/dev/null && ! shard3_running; do sleep 60; done
echo "shard 3 completion pass started $(date -Is)" >> "$OUT/STATUS.shard3"
"$PY" -u experiments/gate_trajectory_signatures/run_explanation_clients.py --output-dir "$OUT" \
  --manifest analysis_outputs/specificity_imagenet_20260926/manifest.csv --dataset imagenet \
  --splits evaluation --shard 3 --num-shards 6 >> "$OUT/shard3_rerun.log" 2>&1
echo "shard 3 exit=$? $(date -Is): $(cat "$OUT"/sessions_shard*.jsonl | wc -l) session lines, $(cat "$OUT"/sessions_shard*.jsonl | "$PY" -c 'import sys,json; ids=[json.loads(l)["session_id"] for l in sys.stdin]; print(len(ids)-len(set(ids)))') duplicates" >> "$OUT/STATUS.shard3"
"$PY" experiments/gate_trajectory_signatures/summarize_explanation_clients.py --root "$OUT" > "$OUT/summary.log" 2>&1
echo "summary exit=$? $(date -Is)" >> "$OUT/STATUS.shard3"
echo DONE >> "$OUT/STATUS.shard3"
