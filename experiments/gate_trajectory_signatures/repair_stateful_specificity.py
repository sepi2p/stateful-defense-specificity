#!/usr/bin/env python3
"""Test fixed, ordinary repairs after the GWAD specificity gate passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load(root):
    rows = [json.loads(line) for line in (root / "sessions.jsonl").open() if line.strip()]
    return rows


def threshold(scores):
    values = np.sort(np.asarray(scores, float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("inf")
    valid = [v for v in np.r_[values, np.inf] if np.mean(values > v) <= 0.05]
    return float(min(valid))


def summarize_series(values, times):
    values = np.asarray(values, float)
    times = np.asarray(times, float)
    if len(values) == 0:
        return np.zeros(7, dtype=np.float32)
    slope = np.polyfit(times, values, 1)[0] if len(values) > 1 else 0.0
    return np.asarray([values[0], values[-1], values.min(), values.max(), values.mean(), values.std(), slope], np.float32)


def output_features(logits, label, prefixes=(256, 512, 768, 1024)):
    logits = np.asarray(logits, float)
    logits -= logits.max(1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(1)
    sorted_p = np.sort(probabilities, axis=1)
    gap = sorted_p[:, -1] - sorted_p[:, -2]
    source = probabilities[:, int(label)]
    rows = []
    query_indices = []
    for prefix in prefixes:
        stop = min(prefix, len(logits))
        t = np.arange(1, stop + 1)
        rows.append(np.concatenate([summarize_series(v[:stop], t) for v in (source, entropy, gap)]))
        query_indices.append(stop)
    return np.asarray(rows), np.asarray(query_indices)


def trace_features(root, row, method):
    with np.load(root / row["trace"]) as data:
        if method == "histogram":
            x = data["gwad_plus_histograms"]
            q = data["gwad_plus_query_indices"]
            if len(x) > 8:
                positions = np.linspace(0, len(x) - 1, 8).round().astype(int)
                x, q = x[positions], q[positions]
            return x.astype(np.float32), q.astype(int)
        return output_features(data["logits"], row["source_label"])


def fit_direction(rows, root, method, train_workloads):
    fit_rows = [r for r in rows if r["split"] == "fit" and r["workload"] in train_workloads and r["objective"] in {"attack", "restore"}]
    xs, ys, weights = [], [], []
    for row in fit_rows:
        x, _ = trace_features(root, row, method)
        xs.append(x)
        ys.extend([int(row["objective"] == "attack")] * len(x))
        weights.extend([1.0 / max(len(x), 1)] * len(x))
    x = np.concatenate(xs)
    y = np.asarray(ys)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=20260924))
    model.fit(x, y, logisticregression__sample_weight=np.asarray(weights))
    scored = []
    for row in rows:
        if row["objective"] not in {"attack", "restore"}:
            continue
        x, q = trace_features(root, row, method)
        scores = model.predict_proba(x)[:, 1]
        scored.append(row | {
            "scores": scores.tolist(), "score_queries": q.tolist(),
            "session_score": float(scores.max()),
        })
    calibration = [r for r in scored if r["split"] == "calibration" and r["objective"] == "restore" and r["workload"] in train_workloads]
    cutoff = threshold([r["session_score"] for r in calibration])
    for row in scored:
        alarm_queries = [q for s, q in zip(row["scores"], row["score_queries"]) if s > cutoff]
        row["alarm_query"] = min(alarm_queries, default=-1)
        row["alarm"] = int(row["alarm_query"] >= 0)
        row["early"] = int(row["objective"] == "attack" and row["first_success"] > 0 and 0 < row["alarm_query"] < row["first_success"])
    return scored, cutoff, model


def rate(rows, field):
    by_source = pd.DataFrame(rows).groupby("dataset_index")[field].mean().to_numpy(float)
    rng = np.random.default_rng(20260924 + len(rows) + len(field))
    draws = rng.choice(by_source, (5000, len(by_source)), replace=True).mean(1)
    return {"value": float(by_source.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)), "n": int(len(by_source))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.root)
    simple_thresholds = {}
    for workload in ("denoise", "deblur"):
        controls = [
            r["detectors"]["gwad_plus"]["max_score"] for r in rows
            if r["split"] == "calibration" and r["workload"] == workload
            and r["objective"] == "random_walk"
        ]
        simple_thresholds[workload] = threshold(controls)
    directions = [("denoise",), ("deblur",), ("denoise", "deblur")]
    results = []
    for method in ("histogram", "output"):
        for train_workloads in directions:
            scored, cutoff, model = fit_direction(rows, args.root, method, set(train_workloads))
            model_path = args.root / "repair_models" / f"{method}__{'+'.join(train_workloads)}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "threshold": cutoff, "method": method, "train_workloads": train_workloads}, model_path)
            for workload in ("denoise", "deblur"):
                test = [r for r in scored if r["split"] == "evaluation" and r["workload"] == workload]
                benign = [r for r in test if r["objective"] == "restore"]
                attacks = [r for r in test if r["objective"] == "attack"]
                successful = [r for r in attacks if r["first_success"] > 0]
                baseline_early = []
                for row in successful:
                    with np.load(args.root / row["trace"]) as data:
                        scores = data["gwad_plus_scores"]
                        queries = data["gwad_plus_query_indices"]
                    alarms = queries[scores > simple_thresholds[workload]]
                    first = int(alarms.min()) if len(alarms) else -1
                    baseline_early.append(row | {"baseline_early": int(0 < first < row["first_success"])})
                repaired_rate = rate(successful, "early") if successful else None
                baseline_rate = rate(baseline_early, "baseline_early") if baseline_early else None
                results.append({
                    "method": method, "train_workloads": "+".join(train_workloads),
                    "test_workload": workload, "threshold": cutoff,
                    "fpr": rate(benign, "alarm"),
                    "early_detection_conditional": repaired_rate,
                    "simple_control_early_detection": baseline_rate,
                    "early_detection_change": (
                        repaired_rate["value"] - baseline_rate["value"]
                        if repaired_rate and baseline_rate else None
                    ),
                    "successful_sources": len({r["dataset_index"] for r in successful}),
                })
    # A repair is considered ordinary and adequate only if one method trained on
    # each single workload transfers in both directions with <=10% FPR. The
    # early-detection comparison to the original threshold is retained for the
    # final paper decision after the SimBA transfer stage.
    transfer = [r for r in results if "+" not in r["train_workloads"] and r["train_workloads"] != r["test_workload"]]
    for row in transfer:
        row["adequate_repair"] = bool(
            row["fpr"]["value"] <= .10
            and row["early_detection_change"] is not None
            and row["early_detection_change"] >= -.10
        )
        row["persistent_problem"] = bool(
            (row["fpr"]["value"] >= .20 and row["fpr"]["lower"] > .10)
            or (row["early_detection_change"] is not None and row["early_detection_change"] <= -.20)
        )
    by_method = {}
    for method in ("histogram", "output"):
        cells = [r for r in transfer if r["method"] == method]
        by_method[method] = all(r["adequate_repair"] for r in cells)
    repair_succeeds = any(by_method.values())
    persistent = all(
        any(r["persistent_problem"] for r in transfer if r["method"] == method)
        for method in ("histogram", "output")
    )
    proceed = persistent and not repair_succeeds
    payload = {
        "stage": "experiment2", "status": "PASS" if proceed else "NO_GO",
        "proceed_to_simba": proceed, "ordinary_repair_succeeds": repair_succeeds,
        "adequate_repair_by_method": by_method,
        "persistent_heldout_failure": persistent, "results": results,
    }
    atomic_json(args.root / "experiment2_repair.json", payload)
    atomic_json(args.status, payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if proceed else 20)


if __name__ == "__main__":
    main()
