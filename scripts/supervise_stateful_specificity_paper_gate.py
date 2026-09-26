#!/usr/bin/env python3
"""Run the frozen specificity paper gate sequentially with explicit stops."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/stateful_specificity_paper_gate_20260924"
STATUS = OUT / "status.json"
LOGICAL_STAGES = []


def write_status(state, stage, detail=""):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state, "stage": stage, "detail": detail,
        "pid": os.getpid(), "updated_unix": time.time(),
        "completed_stages": LOGICAL_STAGES,
        "output_root": str(OUT),
    }
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATUS)


def run(stage, command, allow_no_go=False):
    write_status("RUNNING", stage, " ".join(map(str, command)))
    print(f"[STAGE] {stage}", flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 0:
        LOGICAL_STAGES.append(stage)
        return True
    if allow_no_go and result.returncode == 20:
        write_status("NO_GO", stage, "Predeclared gate did not pass; downstream experiments were not launched.")
        return False
    write_status("FAILED", stage, f"command returned {result.returncode}")
    raise SystemExit(result.returncode)


def main():
    py = sys.executable
    OUT.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", "startup")
    if not run("official_gwad_parity", [
        py, "experiments/gate_trajectory_signatures/validate_official_gwad_adapter.py",
        "--output", str(OUT / "official_gwad_parity.json"),
    ]):
        return
    if not run("development_utility", [
        py, "experiments/gate_trajectory_signatures/run_stateful_specificity_sessions.py",
        "--stage", "development", "--output-dir", str(OUT / "development"),
    ]):
        return
    if not run("development_gate", [
        py, "experiments/gate_trajectory_signatures/analyze_stateful_specificity_gate.py",
        "--mode", "development", "--root", str(OUT / "development"), "--status", str(STATUS),
    ], allow_no_go=True):
        return
    gate = json.loads((OUT / "development/development_gate.json").read_text())
    denoise = str(gate["chosen_strengths"]["denoise"])
    deblur = str(gate["chosen_strengths"]["deblur"])
    common = ["--chosen-denoise-lambda", denoise, "--chosen-deblur-lambda", deblur]
    if not run("experiment1_nes", [
        py, "experiments/gate_trajectory_signatures/run_stateful_specificity_sessions.py",
        "--stage", "nes", "--output-dir", str(OUT / "nes"), *common,
    ]):
        return
    if not run("experiment1_gate", [
        py, "experiments/gate_trajectory_signatures/analyze_stateful_specificity_gate.py",
        "--mode", "experiment1", "--root", str(OUT / "nes"), "--status", str(STATUS),
    ], allow_no_go=True):
        return
    if not run("experiment2_repairs", [
        py, "experiments/gate_trajectory_signatures/repair_stateful_specificity.py",
        "--root", str(OUT / "nes"), "--status", str(STATUS),
    ], allow_no_go=True):
        return
    if not run("experiment3_simba", [
        py, "experiments/gate_trajectory_signatures/run_stateful_specificity_sessions.py",
        "--stage", "simba", "--output-dir", str(OUT / "simba"), *common,
    ]):
        return
    run("experiment3_analysis", [
        py, "experiments/gate_trajectory_signatures/analyze_stateful_specificity_simba.py",
        "--nes-root", str(OUT / "nes"), "--simba-root", str(OUT / "simba"), "--status", str(STATUS),
    ])
    write_status("BLOCKED", "independent_detector", "SimBA transfer completed; no faithful official implementation of the June 2026 detector was located, so no home-made substitute was used.")


if __name__ == "__main__":
    main()
