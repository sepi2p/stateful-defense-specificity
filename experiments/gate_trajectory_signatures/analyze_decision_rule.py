#!/usr/bin/env python3
"""Decision time of a sequential rule built on the direction of the output drift (exploratory).

Replaces analyze_decision_time.py for the manuscript. That script set the threshold of each prefix
for a pooled false-positive rate of 1% AT THAT PREFIX, reported the benign rates per prefix, and
counted an attack as detected if the rule fired at ANY prefix. Applied as a monitor, which looks
repeatedly, such a rule has a pooled false-positive rate of 4 to 6%, so attack and benign sessions
were not counted alike. Here the rule is sequential for every session:

  the rule is evaluated after each of the prefixes below; a session is flagged at the first prefix at
  which (stage 1) a similarity alarm has been raised and (stage 2) the trend of the reference margin
  within the largest group of similar queries formed SO FAR exceeds the threshold of that prefix;

and it is calibrated as a whole: the thresholds of all prefixes are set at one common per-prefix
rate, the largest for which at most 1% of the pooled benign sessions of the CALIBRATION split are
flagged at any prefix. Everything reported is on the evaluation split and cumulative.

Stage 1: Blacklight (published rule), the released decision of GWAD+, or a group of at least 15
similar queries (Phase 1 of the reimplemented detector of Lee et al.).
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
from experiments.gate_trajectory_signatures.analyze_output_diagnostic import ATTACK_EQUIVALENT, BENIGN_POOL, trend  # noqa: E402

PREFIXES = (64, 128, 192, 256, 384, 512, 640, 768, 896, 1021)
TARGET = 0.01


def build_frame(roots: list[Path]) -> pd.DataFrame:
    records = []
    for root in roots:
        for s in blacklight_rule.load_sessions(root):
            if s["split"] not in ("calibration", "evaluation"):
                continue
            trace = np.load(root / s["trace"])
            logits = trace["logits"].astype(np.float64)
            assign = trace["lfc_assignment"]
            bl = s["detectors"]["blacklight"]["first_alarm"]
            gp = s["detectors"]["gwad_plus"]["first_alarm"]
            for prefix in PREFIXES:
                cut = min(prefix, len(logits))
                counts = np.bincount(assign[:cut])
                members = np.flatnonzero(assign[:cut] == int(counts.argmax()))
                records.append({"objective": s["objective"], "workload": s["workload"], "split": s["split"],
                                "session_id": s["session_id"], "prefix": prefix,
                                "stage1": bool((0 < bl <= cut) or (0 < gp <= cut) or counts.max() >= 15),
                                "score": trend(logits[members]) if len(members) >= 3 else 0.0,
                                "first_success": int(s.get("first_success", -1))})
    return pd.DataFrame(records)


def thresholds_at(cal: pd.DataFrame, level: float) -> dict[int, float]:
    out = {}
    for prefix, g in cal.groupby("prefix"):
        scores = np.where(g.stage1, g.score, -np.inf)
        finite = np.sort(scores[np.isfinite(scores)])[::-1]
        allowed = int(np.floor(level * len(scores)))
        out[int(prefix)] = float(finite[allowed]) if allowed < len(finite) else -np.inf
    return out


def first_flag(frame: pd.DataFrame, tau: dict[int, float]) -> pd.Series:
    fired = frame[frame.stage1 & (frame.score > frame.prefix.map(tau))]
    return fired.groupby("session_id").prefix.min()


def calibrate_sequential(cal: pd.DataFrame) -> tuple[float, dict[int, float], float]:
    n = cal.session_id.nunique()
    best = (0.0, thresholds_at(cal, 0.0), len(first_flag(cal, thresholds_at(cal, 0.0))) / n)
    for k in range(1, int(np.ceil(TARGET * n)) + 1):
        tau = thresholds_at(cal, k / n)
        rate = len(first_flag(cal, tau)) / n
        if rate <= TARGET:
            best = (k / n, tau, rate)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, default=[Path("analysis_outputs/lfc_workloads_20260926"),
                                                                   Path("analysis_outputs/lfc_controls_20260926"),
                                                                   Path("analysis_outputs/explanation_clients_20260926")])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/decision_time_r2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_frame(args.roots)
    frame.to_csv(args.output_dir / "session_prefix_scores.csv", index=False)
    pd.set_option("display.width", 250)
    for tag, excluded in (("", ()), ("_no_sweep_walk", ("sweep", "random_walk"))):
        pool = sorted(set(BENIGN_POOL) - set(excluded))
        cal = frame[(frame.split == "calibration") & frame.objective.isin(pool)]
        ev = frame[frame.split == "evaluation"]
        level, tau, cal_rate = calibrate_sequential(cal)
        per_prefix = thresholds_at(cal, TARGET)  # the rule of the first analysis, for comparison
        sessions = ev.drop_duplicates("session_id").set_index("session_id")[["objective", "first_success"]]
        summary = {"calibration_pool": pool, "calibration_sessions": int(cal.session_id.nunique()),
                   "success_cdf": {int(p): float(((sessions[sessions.objective == "attack"].first_success > 0)
                                                  & (sessions[sessions.objective == "attack"].first_success <= p)).mean()) for p in PREFIXES}}
        for name, thresholds in (("sequential", tau), ("per_prefix_1pct", per_prefix)):
            flagged = sessions.assign(first_flag=first_flag(ev, thresholds))
            rows = []
            for objective, g in flagged.groupby("objective"):
                tier = "attack" if objective == "attack" else ("attack-equivalent" if objective in ATTACK_EQUIVALENT else "benign")
                for p in PREFIXES:
                    rows.append({"rule": name, "objective": objective, "tier": tier, "prefix": p, "n": len(g),
                                 "flagged_by_prefix": float((g.first_flag <= p).mean())})
            table = pd.DataFrame(rows)
            table.to_csv(args.output_dir / f"{name}_rates{tag}.csv", index=False)
            attack = flagged[flagged.objective == "attack"]
            successful = attack[attack.first_success > 0]
            pooled = flagged[flagged.objective.isin(pool)]
            summary[name] = {
                "thresholds": thresholds,
                "per_prefix_level": level if name == "sequential" else TARGET,
                "pooled_rate_calibration": float(len(first_flag(cal, thresholds)) / cal.session_id.nunique()),
                "pooled_rate_evaluation": float(pooled.first_flag.notna().mean()),
                "attacks_flagged_by_end": float(attack.first_flag.notna().mean()),
                "successful_attacks": int(len(successful)),
                "successful_attacks_flagged_before_success": int((successful.first_flag < successful.first_success).sum()),
                "fraction_flagged_before_success": float((successful.first_flag < successful.first_success).mean()),
            }
            print(f"===== pool{tag or '_all'} rule {name}: per-prefix level {summary[name]['per_prefix_level']:.5f}, pooled rate "
                  f"calibration {summary[name]['pooled_rate_calibration']:.4f}, evaluation {summary[name]['pooled_rate_evaluation']:.4f}; "
                  f"flagged before success {summary[name]['successful_attacks_flagged_before_success']}/{len(successful)}")
            print((100 * table.pivot_table(index=["tier", "objective"], columns="prefix", values="flagged_by_prefix")).round(1).to_string())
        # snapshot: the rule evaluated once, after the stated number of queries (per-prefix 1% thresholds)
        snap = ev.assign(flag=ev.stage1 & (ev.score > ev.prefix.map(per_prefix)))
        snap.groupby(["objective", "prefix"]).flag.mean().rename("flagged_at_prefix").reset_index().to_csv(
            args.output_dir / f"snapshot_rates{tag}.csv", index=False)
        (args.output_dir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
