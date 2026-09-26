#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sepi/projects/MCG-Blackbox"
PYTHON="/home/sepi/jupyterenv/bin/python"
OUT="$ROOT/analysis_outputs/stateful_specificity_paper_gate_20260924"

mkdir -p "$OUT"
cd "$ROOT"
exec "$PYTHON" -u scripts/supervise_stateful_specificity_paper_gate.py
