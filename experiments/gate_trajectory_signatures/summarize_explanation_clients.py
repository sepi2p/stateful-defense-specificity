#!/usr/bin/env python3
"""Summarize X2 explanation-client sessions against P6a-P6c (evaluation split).

Revision 2 applies the published Blacklight rule (blacklight_rule.py), reports the share of
queries that each detector flags, and writes explanation_*_r2 files; the files without the
suffix are the first analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402
from experiments.gate_trajectory_signatures.lfc_detector import phase2_alarm  # noqa: E402
from experiments.gate_trajectory_signatures.summarize_specificity_controls import wilson  # noqa: E402


def bootstrap_median_ci(diff: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(diff, len(diff))) for _ in range(reps)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("analysis_outputs/explanation_clients_20260926"))
    args = parser.parse_args()
    sessions = pd.DataFrame(blacklight_rule.load_sessions(args.root))
    rows = []
    for r in sessions.itertuples(index=False):
        trace = np.load(args.root / r.trace)
        d = r.detectors
        rows.append({
            "client": r.objective, "split": r.split, "queries": r.calls,
            "blacklight_flagged_fraction": d["blacklight"]["flagged_fraction"],
            "gwad_plus_windows": int(len(trace["gwad_plus_predictions"])),
            "gwad_plus_windows_flagged": int((trace["gwad_plus_predictions"] != 0).sum()),
            "blacklight": d["blacklight"]["first_alarm"], "gwad_plus": d["gwad_plus"]["first_alarm"],
            "gwad": d["gwad"]["first_alarm"],
            "lfc": phase2_alarm(trace["lfc_assignment"], trace["logits"], lags="min10", retest="every"),
            "lfc_largest_subsequence": int(np.bincount(trace["lfc_assignment"]).max()),
            "util_diff": r.deletion_auc_random - r.deletion_auc,
            "util_diff_pixel_blur": r.deletion_auc_random_pixel_blur - r.deletion_auc_blur,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.root / "explanation_sessions_r2.csv", index=False)
    ev = frame[frame.split == "evaluation"]
    out, checks = [], {}
    for client, g in ev.groupby("client"):
        entry = {"client": client, "n": len(g), "queries": int(g.queries.median()),
                 "lfc_median_largest_subsequence": float(g.lfc_largest_subsequence.median()),
                 "blacklight_flagged_queries": float((g.blacklight_flagged_fraction * g.queries).sum() / g.queries.sum()),
                 "gwad_plus_flagged_windows": (float(g.gwad_plus_windows_flagged.sum() / g.gwad_plus_windows.sum())
                                               if g.gwad_plus_windows.sum() else np.nan),
                 "blacklight_alarm_by_query_10": float(((g.blacklight > 0) & (g.blacklight <= 10)).mean())}
        for det in ("blacklight", "gwad_plus", "gwad", "lfc"):
            k = int((g[det] > 0).sum())
            lo, hi = wilson(k, len(g))
            entry[det] = f"{k / len(g):.3f} [{lo:.3f}, {hi:.3f}]"
            entry[f"{det}_median_first"] = float(g.loc[g[det] > 0, det].median()) if k else np.nan
        for col in ("util_diff", "util_diff_pixel_blur"):
            lo, hi = bootstrap_median_ci(g[col].to_numpy())
            entry[col] = f"{g[col].median():.3f} [{lo:.3f}, {hi:.3f}]"
        out.append(entry)
        bl_rate = float((g.blacklight > 0).mean())
        bl_first = float(g.loc[g.blacklight > 0, "blacklight"].median()) if (g.blacklight > 0).any() else np.inf
        if client in ("lime", "kernelshap", "occlusion"):
            checks[f"P6a_{client}_blacklight_ge_90pct_first_le_50"] = bool(bl_rate >= 0.90 and bl_first <= 50)
        if client == "rise":
            checks["P6b_rise_blacklight_le_10pct"] = bool(bl_rate <= 0.10)
        lo, _ = bootstrap_median_ci(g.util_diff.to_numpy())
        checks[f"P6c_{client}_utility"] = bool(g.util_diff.median() > 0 and lo > 0)
    table = pd.DataFrame(out)
    table.to_csv(args.root / "explanation_summary_r2.csv", index=False)
    (args.root / "explanation_predictions_r2.json").write_text(json.dumps(checks, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
