#!/usr/bin/env python3
"""X7: an interpretable output-footprint diagnostic and its limits.

Statistic: drawdown of the reference-class margin, max_t (max_{s<=t} m_s - m_t),
where m_t = logit_ref - max_other and the reference class is the prediction on
the first query of the group being scored. Larger drawdown = more attack-like;
the orientation is fixed a priori, nothing is fitted.

Two-stage design: stage 1 = any similarity alarm (Blacklight, GWAD+ native, or a
Lee-Fang-Chang Phase-1 subsequence of >= 15 queries); stage 2 = drawdown of the
largest Phase-1 subsequence above a threshold set at 1% FPR on the CALIBRATION
split of all benign traffic pooled (explanation clients, restore,
confidence_boost, shuffled/noise/sweep controls, random walk). Reported on the
evaluation split: attack TPR, per-workload FPR, and attack-equivalent clients
(boundary_probe, counterfactual) separately, never in the benign pool.

Reference poisoning (offline): an attacker's first query is a decoy image of
another class. Session-level scoring locks onto the decoy's class;
subsequence-level scoring does not if the decoy falls outside the attack's
Phase-1 subsequence (a dissimilar image cannot match its windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.summarize_specificity_controls import wilson  # noqa: E402

BENIGN_POOL = {"restore", "confidence_boost", "shuffled", "noise", "sweep", "random_walk",
               "lime", "kernelshap", "occlusion", "rise"}
ATTACK_EQUIVALENT = {"boundary_probe", "counterfactual"}


def drawdown(logits: np.ndarray, ref: int | None = None) -> float:
    if len(logits) < 2:
        return 0.0
    ref = int(logits[0].argmax()) if ref is None else ref
    m = logits[:, ref] - np.delete(logits, ref, axis=1).max(axis=1)
    return float((np.maximum.accumulate(m) - m).max())


def trend(logits: np.ndarray, ref: int | None = None) -> float:
    """E1: -Kendall tau between query index and reference margin (persistent decline -> large)."""
    if len(logits) < 3:
        return 0.0
    ref = int(logits[0].argmax()) if ref is None else ref
    m = logits[:, ref] - np.delete(logits, ref, axis=1).max(axis=1)
    tau = kendalltau(np.arange(len(m)), m).statistic
    return 0.0 if np.isnan(tau) else float(-tau)


def load(root: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["root"] = str(root)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["analysis_outputs/lfc_workloads_20260926",
                                                       "analysis_outputs/lfc_controls_20260926",
                                                       "analysis_outputs/explanation_clients_20260926"])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/output_diagnostic_20260926"))
    parser.add_argument("--prefix", type=int, default=256, help="early-decision prefix in queries (full session also reported)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.concat([load(Path(r)) for r in args.roots], ignore_index=True)
    sessions = sessions[sessions.split.isin(["calibration", "evaluation"])].reset_index(drop=True)

    decoys = sessions[sessions.objective == "shuffled"]
    rng = np.random.default_rng(0)
    rows = []
    for r in sessions.itertuples(index=False):
        trace = np.load(Path(r.root) / r.trace)
        logits = trace["logits"].astype(np.float64)
        assign = trace["lfc_assignment"]
        largest = int(np.bincount(assign).argmax())
        members = np.flatnonzero(assign == largest)
        d = r.detectors
        stage1 = (d["blacklight"]["first_alarm"] > 0) or (d["gwad_plus"]["first_alarm"] > 0) or len(members) >= 15
        row = {"objective": r.objective, "workload": r.workload, "split": r.split, "stage1": bool(stage1)}
        for tag, cut in (("early", args.prefix), ("full", len(logits))):
            sub = members[members < cut]
            row[f"dd_session_{tag}"] = drawdown(logits[:cut])
            row[f"dd_group_{tag}"] = drawdown(logits[sub]) if len(sub) >= 2 else 0.0
            row[f"tr_group_{tag}"] = trend(logits[sub]) if len(sub) >= 3 else 0.0
        if r.objective == "attack":
            # poisoning: a decoy query from an unrelated image precedes the attack
            decoy_row = decoys.iloc[int(rng.integers(len(decoys)))]
            decoy = np.load(Path(decoy_row.root) / decoy_row.trace)["logits"][:1].astype(np.float64)
            poisoned = np.vstack([decoy, logits])
            for tag, cut in (("early", args.prefix), ("full", len(logits))):
                row[f"dd_session_{tag}_poisoned"] = drawdown(poisoned[: cut + 1])
                # the decoy cannot match the attack's windows, so the group score is unchanged
                row[f"dd_group_{tag}_poisoned"] = row[f"dd_group_{tag}"]
                row[f"tr_group_{tag}_poisoned"] = row[f"tr_group_{tag}"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "diagnostic_sessions.csv", index=False)

    ev, cal = frame[frame.split == "evaluation"], frame[frame.split == "calibration"]
    out = {}
    # (a) AUROC, attack vs each other workload, by scoring unit and prefix
    auc_rows = []
    attack = ev[ev.objective == "attack"]
    for objective, g in ev[ev.objective != "attack"].groupby("objective"):
        for stat, unit in (("dd", "session"), ("dd", "group"), ("tr", "group")):
            for tag in ("early", "full"):
                col = f"{stat}_{unit}_{tag}"
                unit_label = f"{stat}_{unit}"
                y = np.r_[np.ones(len(attack)), np.zeros(len(g))]
                auc_rows.append({"benign": objective, "unit": unit_label, "prefix": tag,
                                 "auroc": roc_auc_score(y, np.r_[attack[col], g[col]]),
                                 "auroc_poisoned_attack": roc_auc_score(y, np.r_[attack[f"{col}_poisoned"], g[col]])})
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(args.output_dir / "diagnostic_auroc.csv", index=False)

    # (b) two-stage operating point, threshold at 1% FPR on pooled calibration benign traffic
    op_rows = []
    for stat, unit in (("dd", "session"), ("dd", "group"), ("tr", "group")):
        for tag in ("early", "full"):
            col = f"{stat}_{unit}_{tag}"
            pool = cal[cal.objective.isin(BENIGN_POOL)]
            scores = np.where(pool.stage1, pool[col], -np.inf)
            finite = np.sort(scores[np.isfinite(scores)])[::-1]
            allowed = int(np.floor(0.01 * len(scores)))
            tau = finite[allowed] if allowed < len(finite) else -np.inf
            for objective, g in ev.groupby("objective"):
                for poisoned in ((False, True) if objective == "attack" else (False,)):
                    s = g[f"{col}_poisoned"] if poisoned else g[col]
                    alarm = g.stage1 & (s > tau)
                    k = int(alarm.sum())
                    lo, hi = wilson(k, len(g))
                    op_rows.append({"unit": f"{stat}_{unit}", "prefix": tag, "threshold": float(tau),
                                    "objective": objective + (" (poisoned)" if poisoned else ""),
                                    "tier": "attack" if objective == "attack" else ("attack-equivalent" if objective in ATTACK_EQUIVALENT else "benign"),
                                    "n": len(g), "alarm_rate": k / len(g), "lo": lo, "hi": hi})
    ops = pd.DataFrame(op_rows)
    ops.to_csv(args.output_dir / "two_stage_operating_points.csv", index=False)
    print(auc.pivot_table(index="benign", columns=["unit", "prefix"], values="auroc").round(3).to_string())
    print()
    print(auc.pivot_table(index="benign", columns=["unit", "prefix"], values="auroc_poisoned_attack").round(3).to_string())
    print()
    print(ops.pivot_table(index=["tier", "objective"], columns=["unit", "prefix"], values="alarm_rate").round(3).to_string())


if __name__ == "__main__":
    main()
