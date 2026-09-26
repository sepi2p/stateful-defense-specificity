#!/usr/bin/env python3
"""X1: full Lee-Fang-Chang detector (Phase 1 online, Phase 2 offline) on the specificity corpora.

Checks that the regenerated query streams are identical to the original corpora
(SHA-256 of the float32 query bytes), then reports alarm rates (Wilson 95% CI)
and first-alarm indices per objective on the evaluation split, for the frozen
primary configuration and the lag-20 sensitivity setting, and scores P7a-P7d.
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

from experiments.gate_trajectory_signatures.lfc_detector import phase2_alarm  # noqa: E402
from experiments.gate_trajectory_signatures.summarize_specificity_controls import wilson  # noqa: E402


def load(root: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["root"] = str(root)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=[
        "analysis_outputs/stateful_specificity_workloads_20260925:analysis_outputs/lfc_workloads_20260926",
        "analysis_outputs/stateful_specificity_controls_20260925:analysis_outputs/lfc_controls_20260926",
    ], help="original:regenerated corpus directory pairs")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/lfc_workloads_20260926/analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, sha_report = [], {}
    for pair in args.pairs:
        original, regenerated = (Path(p) for p in pair.split(":"))
        old, new = load(original), load(regenerated)
        merged = new.merge(old[["session_id", "query_sha256"]], on="session_id", how="left", suffixes=("", "_original"))
        sha_report[str(regenerated)] = {
            "sessions": int(len(new)),
            "matched_original": int(merged.query_sha256_original.notna().sum()),
            "sha_identical": int((merged.query_sha256 == merged.query_sha256_original).sum()),
        }
        frames.append(new)
    sessions = pd.concat(frames, ignore_index=True)
    rows = []
    for r in sessions.itertuples(index=False):
        trace = np.load(Path(r.root) / r.trace)
        a, logits = trace["lfc_assignment"], trace["logits"]
        sizes = np.bincount(a)
        rows.append({
            "session_id": r.session_id, "split": r.split, "workload": r.workload, "objective": r.objective,
            "largest_subsequence": int(sizes.max()), "n_subsequences": int(len(sizes)),
            "alarm_primary": phase2_alarm(a, logits, lags="min10", retest="every"),
            "alarm_lag20": phase2_alarm(a, logits, lags=20, retest="every"),
            "attack_first_success": int(r.first_success),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "lfc_sessions.csv", index=False)
    ev = frame[frame.split == "evaluation"]
    out = []
    for (workload, objective), g in ev.groupby(["workload", "objective"]):
        entry = {"workload": workload, "objective": objective, "n": len(g),
                 "median_largest_subsequence": float(g.largest_subsequence.median())}
        for setting in ("alarm_primary", "alarm_lag20"):
            k = int((g[setting] > 0).sum())
            lo, hi = wilson(k, len(g))
            entry[setting] = k / len(g)
            entry[f"{setting}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            entry[f"{setting}_median_first"] = float(g.loc[g[setting] > 0, setting].median()) if k else np.nan
        out.append(entry)
    table = pd.DataFrame(out)
    table.to_csv(args.output_dir / "lfc_alarm_rates.csv", index=False)
    rate = ev.assign(a=ev.alarm_primary > 0).groupby(["workload", "objective"]).a.mean()

    def cells(objective):
        return [rate[(w, objective)] for w in ("denoise", "deblur") if (w, objective) in rate]

    def check(values, op, bound):
        return None if not values else bool(op(values, bound))

    at_least = lambda v, b: min(v) >= b  # noqa: E731
    at_most = lambda v, b: max(v) <= b  # noqa: E731
    control = lambda o: [rate[("clean", o)]] if ("clean", o) in rate else []  # noqa: E731
    checks = {
        "P7a_restore_ge_90pct": check(cells("restore"), at_least, 0.90),
        "P7a_confidence_boost_ge_90pct": check(cells("confidence_boost"), at_least, 0.90),
        "P7b_attack_ge_95pct": check(cells("attack"), at_least, 0.95),
        "P7c_shuffled_le_2pct": check(control("shuffled"), at_most, 0.02),
        "P7c_noise_le_2pct": check(control("noise"), at_most, 0.02),
        "P7d_random_walk_ge_90pct": check(cells("random_walk"), at_least, 0.90),
    }
    (args.output_dir / "lfc_predictions.json").write_text(json.dumps({"sha": sha_report, "checks": checks}, indent=2))
    print(json.dumps(sha_report, indent=2))
    print(table.drop(columns=[c for c in table.columns if c.endswith("_ci")]).round(3).to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
