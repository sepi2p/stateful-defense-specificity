#!/usr/bin/env python3
"""Pilot: can query-geometry vs output-trajectory detectors separate attacks from
matched benign optimization? Uses saved traces only (no new target queries).

Detector-visible information only: query-derived GWAD/GWAD+ window scores and
the model's own logits. The true label is never used; the "reference label" is
the prediction on the first query of the session.
Orientation of single features and all model fitting use the `fit` split;
AUROCs are reported on the `evaluation` split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def slope(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, y, 1)[0]) if len(y) > 1 else 0.0


def session_features(trace: np.lib.npyio.NpzFile, prefix: int) -> dict[str, float]:
    logits = trace["logits"][:prefix].astype(np.float64)
    probs = softmax(logits)
    ref = int(logits[0].argmax())
    p_ref = probs[:, ref]
    others = np.delete(logits, ref, axis=1)
    ref_margin = logits[:, ref] - others.max(axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    out = {
        "out_pref_first": p_ref[0],
        "out_pref_last": p_ref[-1],
        "out_pref_min": p_ref.min(),
        "out_pref_slope": slope(p_ref),
        "out_margin_last_minus_first": ref_margin[-1] - ref_margin[0],
        "out_margin_min": ref_margin.min(),
        "out_margin_slope": slope(ref_margin),
        "out_gap_mean": float((top2[:, 1] - top2[:, 0]).mean()),
        "out_entropy_slope": slope(entropy),
        "out_label_changed": float((logits.argmax(1) != ref).any()),
    }
    for name in ("gwad", "gwad_plus"):
        idx = trace[f"{name}_query_indices"]
        keep = idx < prefix
        scores = trace[f"{name}_scores"][keep].astype(np.float64)
        # scores saturate near 1; use the complement on a log scale
        logc = -np.log10(np.clip(1.0 - scores, 1e-12, 1.0))
        out[f"{name}_max"] = float(logc.max()) if len(logc) else 0.0
        out[f"{name}_mean"] = float(logc.mean()) if len(logc) else 0.0
        hist = trace[f"{name}_histograms"][keep]
        out[f"{name}_hist_mean_bin"] = float((hist * np.arange(hist.shape[1])).sum(1).mean() / max(hist.sum(1).mean(), 1e-12)) if len(hist) else 0.0
    return out


FEATURE_GROUPS = {
    "query_geometry(GWAD/GWAD+)": lambda c: c.startswith("gwad"),
    "output_trajectory": lambda c: c.startswith("out_"),
    "both": lambda c: True,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="analysis_outputs/stateful_specificity_paper_gate_20260924/nes")
    parser.add_argument("--output-dir", default="analysis_outputs/stateful_specificity_paper_gate_20260924/separability_pilot")
    parser.add_argument("--prefixes", default="128,256,512,1024")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = [json.loads(line) for line in (root / "sessions.jsonl").read_text().splitlines()]
    prefixes = [int(p) for p in args.prefixes.split(",")]
    rows = []
    for s in sessions:
        trace = np.load(root / s["trace"])
        for prefix in prefixes:
            rows.append({
                "session_id": s["session_id"], "split": s["split"], "workload": s["workload"],
                "objective": s["objective"], "dataset_index": s["dataset_index"], "prefix": prefix,
                **session_features(trace, prefix),
            })
    feats = pd.DataFrame(rows)
    feats.to_csv(out_dir / "session_features.csv", index=False)
    feature_cols = [c for c in feats.columns if c.startswith(("out_", "gwad"))]

    results = []
    for prefix in prefixes:
        for workload in ("denoise", "deblur"):
            for negative in ("restore", "random_walk"):
                sub = feats[(feats.prefix == prefix) & (feats.workload == workload) & feats.objective.isin(["attack", negative])]
                fit = sub[sub.split == "fit"]
                ev = sub[sub.split == "evaluation"]
                y_fit = (fit.objective == "attack").to_numpy(int)
                y_ev = (ev.objective == "attack").to_numpy(int)
                for col in feature_cols:
                    sign = 1.0 if roc_auc_score(y_fit, fit[col]) >= 0.5 else -1.0
                    results.append({"prefix": prefix, "workload": workload, "negative": negative,
                                    "detector": f"single:{col}", "eval_auroc": roc_auc_score(y_ev, sign * ev[col])})
                for group, pick in FEATURE_GROUPS.items():
                    cols = [c for c in feature_cols if pick(c)]
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
                    clf.fit(fit[cols], y_fit)
                    score = clf.predict_proba(ev[cols])[:, 1]
                    results.append({"prefix": prefix, "workload": workload, "negative": negative,
                                    "detector": f"logreg:{group}", "eval_auroc": roc_auc_score(y_ev, score)})
    res = pd.DataFrame(results)
    res.to_csv(out_dir / "separability_auroc.csv", index=False)
    table = res[res.detector.str.startswith("logreg") | res.detector.isin(["single:gwad_plus_max", "single:out_pref_slope", "single:out_margin_min"])]
    print(table.pivot_table(index=["negative", "detector"], columns=["workload", "prefix"], values="eval_auroc").round(3).to_string())


if __name__ == "__main__":
    main()
