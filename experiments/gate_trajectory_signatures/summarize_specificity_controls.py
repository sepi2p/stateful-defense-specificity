#!/usr/bin/env python3
"""Summarize X0 control streams against the frozen predictions P5a-P5d.

Revision 2 applies the published Blacklight rule (blacklight_rule.py) and writes
controls_summary_r2.csv and controls_predictions_r2.json; the files without the suffix are the
first analysis.
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
from experiments.gate_trajectory_signatures.pretest_detector_assumptions import ljung_box_p, p_first_class  # noqa: E402


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("analysis_outputs/stateful_specificity_controls_20260925"))
    args = parser.parse_args()
    sessions = pd.DataFrame(blacklight_rule.load_sessions(args.root))
    rows = []
    for r in sessions.itertuples(index=False):
        trace = np.load(args.root / r.trace)
        rows.append({
            "objective": r.objective, "split": r.split,
            "gwad_windows": r.detectors["gwad"]["eligible_windows"],
            "blacklight_alarm": r.detectors["blacklight"]["first_alarm"] > 0,
            "blacklight_first_alarm": r.detectors["blacklight"]["first_alarm"],
            "gwad_plus_alarm": r.detectors["gwad_plus"]["first_alarm"] > 0,
            "gwad_plus_windows": r.detectors["gwad_plus"]["eligible_windows"],
            "gwad_alarm": r.detectors["gwad"]["first_alarm"] > 0,
            "ljung_box_h20": ljung_box_p(p_first_class(trace["logits"].astype(np.float64)), 20) < 0.025,
            "seconds": r.elapsed_seconds,
        })
    frame = pd.DataFrame(rows)
    out = []
    for objective, g in frame.groupby("objective"):
        entry = {"objective": objective, "n": len(g), "median_seconds": float(g.seconds.median()),
                 "median_gwad_plus_windows": float(g.gwad_plus_windows.median()),
                 "median_gwad_windows": float(g.gwad_windows.median())}
        for col in ("blacklight_alarm", "gwad_plus_alarm", "gwad_alarm", "ljung_box_h20"):
            k = int(g[col].sum())
            lo, hi = wilson(k, len(g))
            entry[col] = f"{k}/{len(g)} = {k / len(g):.3f} [{lo:.3f}, {hi:.3f}]"
        out.append(entry)
    table = pd.DataFrame(out)
    table.to_csv(args.root / "controls_summary_r2.csv", index=False)
    rate = frame.groupby("objective")[["blacklight_alarm", "gwad_plus_alarm", "gwad_alarm", "ljung_box_h20"]].mean()
    checks = {
        "P5a_blacklight_shuffled_le_1pct": bool(rate.loc["shuffled", "blacklight_alarm"] <= 0.01),
        "P5a_gwad_family_shuffled_no_alarm": bool(rate.loc["shuffled", ["gwad_plus_alarm", "gwad_alarm"]].max() == 0),
        "P5b_gwad_plus_noise_ge_90pct": bool(rate.loc["noise", "gwad_plus_alarm"] >= 0.90),
        "P5c_blacklight_sweep_ge_90pct": bool(rate.loc["sweep", "blacklight_alarm"] >= 0.90),
        "P5d_ljung_box_shuffled_le_5pct": bool(rate.loc["shuffled", "ljung_box_h20"] <= 0.05),
        "P5d_ljung_box_noise_le_5pct": bool(rate.loc["noise", "ljung_box_h20"] <= 0.05),
    }
    (args.root / "controls_predictions_r2.json").write_text(json.dumps(checks, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
