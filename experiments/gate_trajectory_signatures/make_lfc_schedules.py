#!/usr/bin/env python3
"""Apply the two schedules of the reimplemented Lee-Fang-Chang detector to every logged session.

sequence-50: the paper's protocol. Phase 1 groups the first 50 queries; the Ljung-Box test is applied
             once to every group with at least 15 members (lfc_detector.phase2_batch).
online:      the test is repeated whenever a group grows (lfc_detector.phase2_alarm).

Evaluation split only. Writes, unrounded,
  paper/jisa_2026/numbers/lfc_schedules_sessions.csv   one row per session
  paper/jisa_2026/numbers/lfc_schedules.csv            per corpus and client, both starts pooled
  paper/jisa_2026/numbers/lfc_schedules_by_start.csv   per corpus, client and start
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.lfc_detector import phase2_alarm, phase2_batch  # noqa: E402

A = ROOT / "analysis_outputs"
OUT = ROOT / "paper/jisa_2026/numbers"
CORPORA = {
    "cifar_nes_seed0": "lfc_workloads_20260926",  # the main corpus, regenerated with the observer attached
    "cifar_nes_seed1": "specificity_resnet18_seed1_20260926",
    "cifar_nes_seed2": "specificity_resnet18_seed2_20260926",
    "cifar_nes_vgg19bn": "specificity_vgg19bn_20260926",
    "cifar_nes_robust": "specificity_robust_engstrom_20260926",
    "gtsrb_nes": "specificity_gtsrb32_20260926",
    "imagenet_nes": "specificity_imagenet_20260926",
    "cifar_simba": "specificity_simba_20260926",
    "cifar_controls": "lfc_controls_20260926",
    "cifar_explain": "explanation_clients_20260926",
    "imagenet_explain": "explanation_clients_imagenet_20260926",
}
MIN_LENGTH = 15


def sessions():
    for name, directory in CORPORA.items():
        root = A / directory
        for path in sorted(root.glob("sessions_shard*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                s = json.loads(line)
                if s["split"] != "evaluation":
                    continue
                trace = np.load(root / s["trace"])
                assignment, logits = trace["lfc_assignment"], trace["logits"]
                first = phase2_alarm(assignment, logits)
                grouped = int(np.bincount(assignment[:50]).max())
                yield {"corpus": name, "objective": s["objective"], "start": s["workload"], "session_id": s["session_id"],
                       "online": bool(first > 0), "online_first": int(first),
                       "seq50": bool(phase2_batch(assignment, logits, 50)),
                       "seq50_lag5": bool(phase2_batch(assignment, logits, 50, lags=5)),
                       "grouped50": grouped, "tested50": grouped >= MIN_LENGTH}


def summarize(frame, keys):
    g = frame.groupby(keys)
    out = g.agg(n=("online", "size"), online=("online", "mean"), seq50=("seq50", "mean"), seq50_lag5=("seq50_lag5", "mean"),
                grouped50_median=("grouped50", "median"), tested50=("tested50", "mean"))
    out["seq50_among_tested"] = frame[frame.tested50].groupby(keys).seq50.mean()
    fired = frame[frame.online]
    out["online_first_median"] = fired.groupby(keys).online_first.median()
    return out


def main():
    frame = pd.DataFrame(sessions())
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "lfc_schedules_sessions.csv", index=False)
    pooled = summarize(frame, ["corpus", "objective"])
    pooled.to_csv(OUT / "lfc_schedules.csv")
    summarize(frame, ["corpus", "objective", "start"]).to_csv(OUT / "lfc_schedules_by_start.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(pooled.round(4).to_string())


if __name__ == "__main__":
    main()
