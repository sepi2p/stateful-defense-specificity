#!/usr/bin/env python3
"""Apply frozen NES repairs to SimBA and report calibration-only adaptation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.gate_trajectory_signatures.repair_stateful_specificity import load, rate, threshold, trace_features


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nes-root", type=Path, required=True)
    parser.add_argument("--simba-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.simba_root)
    reports = []
    for path in sorted((args.nes_root / "repair_models").glob("*.joblib")):
        bundle = joblib.load(path)
        model, method = bundle["model"], bundle["method"]
        scored = []
        for row in rows:
            if row["objective"] not in {"attack", "restore"}:
                continue
            x, q = trace_features(args.simba_root, row, method)
            values = model.predict_proba(x)[:, 1]
            scored.append(row | {"scores": values, "queries": q, "session_score": float(values.max())})
        for calibration_mode in ("frozen_nes", "simba_calibration_only"):
            cutoff = float(bundle["threshold"])
            if calibration_mode == "simba_calibration_only":
                controls = [r["session_score"] for r in scored if r["split"] == "calibration" and r["objective"] == "restore"]
                cutoff = threshold(controls)
            for row in scored:
                alarm_q = row["queries"][row["scores"] > cutoff]
                row["alarm"] = int(len(alarm_q) > 0)
                first = int(alarm_q.min()) if len(alarm_q) else -1
                row["early"] = int(row["objective"] == "attack" and row["first_success"] > 0 and 0 < first < row["first_success"])
            for workload in ("denoise", "deblur"):
                test = [r for r in scored if r["split"] == "evaluation" and r["workload"] == workload]
                benign = [r for r in test if r["objective"] == "restore"]
                attacks = [r for r in test if r["objective"] == "attack" and r["first_success"] > 0]
                reports.append({
                    "model": path.stem, "method": method, "calibration": calibration_mode,
                    "workload": workload, "threshold": cutoff,
                    "fpr": rate(benign, "alarm"),
                    "early_detection": rate(attacks, "early") if attacks else None,
                    "successful_sources": len({r["dataset_index"] for r in attacks}),
                })
    payload = {
        "stage": "experiment3_simba",
        "status": "BLOCKED_INDEPENDENT_DETECTOR",
        "simba_reports": reports,
        "independent_detector": {
            "paper": "Enhancing Stateful Detection of Adversarial Attacks with Soft-labels' Temporality and Robust Similarity Approximations (arXiv:2606.21592v1)",
            "status": "No official release located during the frozen setup audit; a home-made approximation is not substituted.",
        },
        "decision": "SimBA transfer is complete, but the predeclared strong-paper gate remains blocked pending a faithful independent-detector implementation.",
    }
    atomic_json(args.simba_root / "experiment3_report.json", payload)
    atomic_json(args.status, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
