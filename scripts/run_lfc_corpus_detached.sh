#!/usr/bin/env bash
# X1: regenerate the NES workload corpus and the control corpus with the
# Lee-Fang-Chang Phase-1 observer online, then run Phase 2 and score P7.
#   setsid nohup bash scripts/run_lfc_corpus_detached.sh > /dev/null 2>&1 < /dev/null &
# Resumable: rerunning skips finished sessions.
set -uo pipefail
cd /home/sepi/projects/MCG-Blackbox
PY=/home/sepi/jupyterenv/bin/python
W=analysis_outputs/lfc_workloads_20260926
C=analysis_outputs/lfc_controls_20260926
SHARDS=6
SEED=20260926
mkdir -p "$W" "$C"
echo "started $(date -Is) pid $$" > "$W/STATUS"
run_shards () {  # $1 = script, $2 = out dir, $3.. = extra args
  local script=$1 out=$2; shift 2
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" -u "$script" --output-dir "$out" --lfc-seed "$SEED" \
      --shard "$i" --num-shards "$SHARDS" "$@" >> "$out/shard$i.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  echo "$(basename "$out") finished $(date -Is): $(cat "$out"/sessions_shard*.jsonl | wc -l) sessions, $failed shard(s) nonzero" >> "$W/STATUS"
}
run_shards experiments/gate_trajectory_signatures/run_specificity_workloads.py "$W" --stage main
run_shards experiments/gate_trajectory_signatures/run_specificity_controls.py "$C"
"$PY" experiments/gate_trajectory_signatures/analyze_lfc_corpus.py > "$W/analysis.log" 2>&1
echo "analysis exit=$? $(date -Is)" >> "$W/STATUS"
echo DONE >> "$W/STATUS"
