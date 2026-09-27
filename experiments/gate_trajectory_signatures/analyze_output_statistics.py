#!/usr/bin/env python3
"""Two single statistics of the model's outputs, on every corpus (exploratory).

drawdown  largest drop of the reference margin below its running maximum (planned, X7)
trend     minus Kendall's tau between query index and reference margin (introduced after the drawdown result, E1)

Both are computed within the largest group of similar queries (Phase 1 of the reimplemented detector
of Lee et al.) for the class predicted for the group's first query. The group is determined FROM THE
PREFIX ONLY: the first analysis (analyze_output_diagnostic.py) chose the largest group of the whole
session and then cut it at the prefix, which uses information from later queries. Orientation is
fixed (larger = more attack-like); nothing is fitted. AUROC on the evaluation split, attack sessions
against the sessions of each other client, both starts pooled.

Non-optimizing clients (explanation methods, control streams) exist for the CIFAR-10 seed-0 model
and for the ImageNet model only; they are compared with the attack sessions of those corpora.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402
from experiments.gate_trajectory_signatures.analyze_output_diagnostic import drawdown, trend  # noqa: E402

A = ROOT / "analysis_outputs"
CORPORA = {
    "cifar_nes_seed0": ["lfc_workloads_20260926", "lfc_controls_20260926", "explanation_clients_20260926"],
    "cifar_nes_seed1": ["specificity_resnet18_seed1_20260926"],
    "cifar_nes_seed2": ["specificity_resnet18_seed2_20260926"],
    "cifar_nes_vgg19bn": ["specificity_vgg19bn_20260926"],
    "cifar_nes_robust": ["specificity_robust_engstrom_20260926"],
    "gtsrb_nes": ["specificity_gtsrb32_20260926"],
    "imagenet_nes": ["specificity_imagenet_20260926", "explanation_clients_imagenet_20260926"],
    "cifar_simba": ["specificity_simba_20260926"],
}
PREFIXES = {"256": 256, "full": 10**9}


def session_rows(corpus: str, directories: list[str]) -> list[dict]:
    rows = []
    for directory in directories:
        root = A / directory
        for s in blacklight_rule.load_sessions(root):
            if s["split"] != "evaluation":
                continue
            trace = np.load(root / s["trace"])
            logits = trace["logits"].astype(np.float64)
            assign = trace["lfc_assignment"]
            row = {"corpus": corpus, "objective": s["objective"], "start": s["workload"], "session_id": s["session_id"]}
            for name, cut in PREFIXES.items():
                a = assign[: min(cut, len(assign))]
                members = np.flatnonzero(a == int(np.bincount(a).argmax()))
                row[f"drawdown_{name}"] = drawdown(logits[members]) if len(members) >= 2 else 0.0
                row[f"trend_{name}"] = trend(logits[members]) if len(members) >= 3 else 0.0
                row[f"group_{name}"] = len(members)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=A / "output_statistics_r2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.DataFrame([r for corpus, dirs in CORPORA.items() for r in session_rows(corpus, dirs)])
    sessions.to_csv(args.output_dir / "sessions.csv", index=False)
    out = []
    for corpus, g in sessions.groupby("corpus"):
        attack = g[g.objective == "attack"]
        for objective, b in g[g.objective != "attack"].groupby("objective"):
            y = np.r_[np.ones(len(attack)), np.zeros(len(b))]
            for statistic in ("drawdown", "trend"):
                for name in PREFIXES:
                    col = f"{statistic}_{name}"
                    out.append({"corpus": corpus, "benign": objective, "statistic": statistic, "prefix": name,
                                "auroc": float(roc_auc_score(y, np.r_[attack[col], b[col]])),
                                "n_attack": len(attack), "n_benign": len(b)})
    table = pd.DataFrame(out)
    table.to_csv(args.output_dir / "auroc.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(table.pivot_table(index=["corpus", "benign"], columns=["statistic", "prefix"], values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
