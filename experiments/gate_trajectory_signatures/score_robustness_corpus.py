#!/usr/bin/env python3
"""Score an X4-X6 robustness corpus against the frozen predictions P8-P11.

Reads the corpus's separability.csv (analyze_specificity_workloads.py) and
its traces (Lee-Fang-Chang Phase 2, native alarms). Evaluation split only.
Revision 2 reads and writes <root>/analysis_r2 (published Blacklight rule).
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

TIER_B = ("restore", "confidence_boost")
TIER_C = ("boundary_probe", "counterfactual")
QUERY_ONLY = ("logreg:gwad_plus", "logreg:blacklight", "logreg:query_only_all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--analysis-dir", default="analysis_r2")
    args = parser.parse_args()
    for root in args.roots:
        sep = pd.read_csv(root / args.analysis_dir / "separability.csv")
        cell = lambda neg, det, prefix: sep[(sep.negative == neg) & (sep.detector == det) & (sep.prefix == prefix)].auroc  # noqa: E731
        sessions = blacklight_rule.load_sessions(root)
        rows = []
        for s in sessions:
            if s["split"] != "evaluation":
                continue
            trace = np.load(root / s["trace"])
            rows.append({"workload": s["workload"], "objective": s["objective"],
                         "lfc": phase2_alarm(trace["lfc_assignment"], trace["logits"]) > 0,
                         "blacklight_native": s["detectors"]["blacklight"]["first_alarm"] > 0,
                         "gwad_plus_native": s["detectors"]["gwad_plus"]["first_alarm"] > 0,
                         "flipped": s["first_success"] > 0})
        ev = pd.DataFrame(rows)
        rates = ev.groupby(["workload", "objective"])[["lfc", "blacklight_native", "gwad_plus_native", "flipped"]].mean()
        query_b = {f"{neg}|{det}": float(cell(neg, det, "1024").max()) for neg in TIER_B for det in QUERY_ONLY}
        walk = {det: float(cell("random_walk", det, "1024").min()) for det in QUERY_ONLY} if "random_walk" in set(sep.negative) else {}
        p9 = {neg: float(cell(neg, "logreg:output_trajectory", "pre_either_flip").max()) for neg in TIER_C}
        p10 = {neg: float(cell(neg, "logreg:output_trajectory", "1024").min()) for neg in TIER_B}
        p11 = {neg: float(rates.xs(neg, level="objective").lfc.min()) for neg in TIER_B}
        checks = {
            "P8_query_only_vs_tierB_le_0.62": {k: (round(v, 3), v <= 0.62) for k, v in query_b.items()},
            "P8_query_only_vs_random_walk_ge_0.90": {k: (round(v, 3), v >= 0.90) for k, v in walk.items()},
            "P9_output_pre_flip_vs_tierC_le_0.70": {k: (round(v, 3), v <= 0.70) for k, v in p9.items()},
            "P10_output_vs_tierB_ge_0.90": {k: (round(v, 3), v >= 0.90) for k, v in p10.items()},
            "P11_lfc_tierB_ge_90pct": {k: (round(v, 3), v >= 0.90) for k, v in p11.items()},
        }
        (root / args.analysis_dir / "robustness_predictions.json").write_text(json.dumps(checks, indent=2, default=str))
        rates.to_csv(root / args.analysis_dir / "alarm_rates.csv")
        print(f"===== {root.name}")
        print(rates.round(3).to_string())
        for name, entries in checks.items():
            held = all(ok for _, ok in entries.values())
            print(f"{name}: {'HELD' if held else 'FAILED'}  " + ", ".join(f"{k}={v}" for k, (v, _) in entries.items()))


if __name__ == "__main__":
    main()
