#!/usr/bin/env python3
"""Decision-time specificity (exploratory; declared in the preregistration addendum of 2026-09-27).

Prefix sweep of the two-stage output diagnostic built on the exploratory trend statistic E1:
stage 1 = a similarity alarm by query P (Blacklight, GWAD+ released decision, or a Lee-Fang-Chang
Phase-1 subsequence of >= 15 queries); stage 2 = -Kendall tau between query index and the
reference-class margin within the largest Phase-1 subsequence, above a threshold set at 1% FPR on
the pooled benign traffic of the CALIBRATION split. For each prefix P the threshold is recalibrated
using only the first P queries of every calibration session. Reported on the evaluation split:
attack detection rate, per-workload false-positive rate, attack-equivalent rates, and the
distribution of attack first-success times.
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

from experiments.gate_trajectory_signatures.analyze_output_diagnostic import ATTACK_EQUIVALENT, BENIGN_POOL, load, trend  # noqa: E402

PREFIXES = (64, 128, 192, 256, 384, 512, 640, 768, 896, 1021)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["analysis_outputs/lfc_workloads_20260926",
                                                       "analysis_outputs/lfc_controls_20260926",
                                                       "analysis_outputs/explanation_clients_20260926"])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/decision_time_20260927"))
    parser.add_argument("--exclude-from-pool", default="", help="comma-separated benign workloads left out of the calibration pool (sensitivity)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    pool = set(BENIGN_POOL) - {x.strip() for x in args.exclude_from_pool.split(",") if x.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.concat([load(Path(r)) for r in args.roots], ignore_index=True)
    sessions = sessions[sessions.split.isin(["calibration", "evaluation"])].reset_index(drop=True)

    records = []
    for r in sessions.itertuples(index=False):
        trace = np.load(Path(r.root) / r.trace)
        logits = trace["logits"].astype(np.float64)
        assign = trace["lfc_assignment"]
        bl = r.detectors["blacklight"]["first_alarm"]
        gp = r.detectors["gwad_plus"]["first_alarm"]
        for prefix in PREFIXES:
            cut = min(prefix, len(logits))
            a = assign[:cut]
            counts = np.bincount(a)
            largest = int(counts.argmax())
            members = np.flatnonzero(a == largest)
            stage1 = (0 < bl <= cut) or (0 < gp <= cut) or counts.max() >= 15
            records.append({"objective": r.objective, "workload": r.workload, "split": r.split,
                            "session_id": r.session_id, "prefix": prefix, "stage1": bool(stage1),
                            "score": trend(logits[members]) if len(members) >= 3 else 0.0,
                            "first_success": int(getattr(r, "first_success", -1))})
    frame = pd.DataFrame(records)
    rows, thresholds = [], {}
    for prefix, g in frame.groupby("prefix"):
        cal = g[(g.split == "calibration") & g.objective.isin(pool)]
        scores = np.where(cal.stage1, cal.score, -np.inf)
        finite = np.sort(scores[np.isfinite(scores)])[::-1]
        allowed = int(np.floor(0.01 * len(scores)))
        tau = float(finite[allowed]) if allowed < len(finite) else -np.inf
        thresholds[int(prefix)] = tau
        ev = g[g.split == "evaluation"]
        for objective, e in ev.groupby("objective"):
            alarm = e.stage1 & (e.score > tau)
            tier = "attack" if objective == "attack" else ("attack-equivalent" if objective in ATTACK_EQUIVALENT else "benign")
            rows.append({"prefix": int(prefix), "objective": objective, "tier": tier, "n": len(e),
                         "alarm_rate": float(alarm.mean()), "threshold": tau})
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / f"decision_time_rates{args.tag}.csv", index=False)

    attack = frame[(frame.objective == "attack") & (frame.split == "evaluation")].copy()
    attack["alarm"] = [bool(s1 and sc > thresholds[int(p)]) for s1, sc, p in zip(attack.stage1, attack.score, attack.prefix)]
    detect = attack[attack.alarm].groupby("session_id").prefix.min().rename("detect_prefix")
    per = attack.drop_duplicates("session_id").set_index("session_id")[["first_success"]].join(detect)
    successful = per[per.first_success > 0]
    summary = {
        "attack_sessions": int(len(per)),
        "attacks_successful": int(len(successful)),
        "median_first_success": float(successful.first_success.median()),
        "detected_by_end": float(per.detect_prefix.notna().mean()),
        "successful_attacks_detected_before_success": float((successful.detect_prefix < successful.first_success).mean()),
        "successful_attacks_detected_at_all": float(successful.detect_prefix.notna().mean()),
        "success_cdf": {int(p): float(((per.first_success > 0) & (per.first_success <= p)).mean()) for p in PREFIXES},
        "thresholds": thresholds,
    }
    summary["calibration_pool"] = sorted(pool)
    (args.output_dir / f"decision_time_summary{args.tag}.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    print(table.pivot_table(index=["tier", "objective"], columns="prefix", values="alarm_rate").round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "thresholds"}, indent=1))


if __name__ == "__main__":
    main()
