#!/usr/bin/env python3
"""Apply the frozen development and published-detector investment gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_sessions(root: Path):
    with (root / "sessions.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frame = pd.json_normalize(rows, sep=".")
    return frame


def bootstrap_rate(frame, column, repetitions=5000, seed=20260924):
    grouped = frame.groupby("dataset_index")[column].mean()
    values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(1)
    return {
        "value": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "source_images": int(len(values)),
    }


def development(root: Path, status: Path):
    frame = load_sessions(root)
    expected = 20 * 2 * 2 * 3
    if len(frame) != expected:
        raise RuntimeError(f"development corpus incomplete: {len(frame)}/{expected}")
    choices = (
        frame.groupby(["workload", "lambda"], as_index=False).psnr_gain.mean()
        .sort_values(["workload", "psnr_gain", "lambda"], ascending=[True, False, True])
        .groupby("workload", as_index=False).first()
    )
    chosen = {row.workload: float(row["lambda"]) for _, row in choices.iterrows()}
    checks = []
    passed = True
    for workload, strength in chosen.items():
        selected = frame[(frame.workload == workload) & (frame["lambda"] == strength)]
        for optimizer, part in selected.groupby("optimizer"):
            row = {
                "workload": workload,
                "optimizer": optimizer,
                "strength": strength,
                "positive": int((part.psnr_gain > 0).sum()),
                "n": int(len(part)),
                "median_psnr_gain": float(part.psnr_gain.median()),
                "mean_psnr_gain": float(part.psnr_gain.mean()),
                "label_preservation": float(part.label_preserved.mean()),
            }
            row["pass"] = row["positive"] >= 14 and row["median_psnr_gain"] >= 0.02
            passed &= row["pass"]
            checks.append(row)
    payload = {
        "stage": "development",
        "status": "PASS" if passed else "NO_GO",
        "chosen_strengths": chosen,
        "checks": checks,
        "decision_rule": ">=14/20 PSNR gains and median gain >=0.02 dB in every workload/optimizer cell",
    }
    atomic_json(root / "development_gate.json", payload)
    atomic_json(status, payload)
    print(json.dumps(payload, indent=2))
    return passed


def conservative_threshold(scores):
    values = np.sort(np.asarray(scores, float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("inf")
    candidates = np.r_[values, np.inf]
    valid = [value for value in candidates if np.mean(values > value) <= 0.05]
    return float(min(valid))


def experiment1(root: Path, status: Path):
    frame = load_sessions(root)
    expected = 400 * 6
    if len(frame) != expected:
        raise RuntimeError(f"NES corpus incomplete: {len(frame)}/{expected}")
    results = []
    gates = []
    for workload in ("denoise", "deblur"):
        calibration = frame[(frame.split == "calibration") & (frame.workload == workload)]
        threshold = conservative_threshold(
            calibration[calibration.objective == "random_walk"]["detectors.gwad_plus.max_score"]
        )
        evaluation = frame[(frame.split == "evaluation") & (frame.workload == workload)].copy()
        evaluation["calibrated_alarm"] = evaluation["detectors.gwad_plus.max_score"].fillna(-np.inf) > threshold
        evaluation["native_alarm"] = evaluation["detectors.gwad_plus.first_alarm"] >= 0
        for objective, part in evaluation.groupby("objective"):
            for decision in ("calibrated_alarm", "native_alarm"):
                metric = bootstrap_rate(part, decision, seed=20260924 + len(results))
                results.append({
                    "workload": workload, "objective": objective,
                    "decision": decision, "threshold": threshold, **metric,
                })
        by = {(r["objective"], r["decision"]): r for r in results if r["workload"] == workload}
        restore = by[("restore", "calibrated_alarm")]
        random_walk = by[("random_walk", "calibrated_alarm")]
        attack = by[("attack", "calibrated_alarm")]
        gate = {
            "workload": workload,
            "restore_fpr": restore,
            "random_walk_fpr": random_walk,
            "attack_detection": attack,
            "pass": bool(
                restore["value"] >= 0.20 and restore["lower"] > 0.10
                and random_walk["value"] <= 0.10
                and attack["value"] >= 0.70
            ),
        }
        gates.append(gate)
    passed = all(item["pass"] for item in gates)
    payload = {
        "stage": "experiment1",
        "status": "PASS" if passed else "NO_GO",
        "gates": gates,
        "all_metrics": results,
        "decision_rule": "both workloads: restoration FPR>=20% (lower CI>10%), random-walk FPR<=10%, attack detection>=70%",
    }
    atomic_json(root / "experiment1_gate.json", payload)
    atomic_json(status, payload)
    pd.DataFrame(results).to_csv(root / "experiment1_metrics.csv", index=False)
    print(json.dumps(payload, indent=2))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["development", "experiment1"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    ok = development(args.root, args.status) if args.mode == "development" else experiment1(args.root, args.status)
    raise SystemExit(0 if ok else 20)


if __name__ == "__main__":
    main()
