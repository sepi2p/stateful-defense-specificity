#!/usr/bin/env python3
"""Apply the schedules of the reimplemented Lee-Fang-Chang detector to every logged session.

The paper sends a group of similar queries to the test whenever the group "is updated" and has at
least 15 members (Section 5.1, step 5), removes the oldest query of a group that exceeds a length
which it does not state, and evaluates attack sequences of 50 queries. Schedules reported:

update50   test on every update, first 50 queries of the session (the paper's design on the length of
           the sequences of its evaluation); a session counts if an alarm is raised by query 50
single50   the first 50 queries are grouped and every group of at least 15 is tested ONCE at the end
           (lfc_detector.phase2_batch); the most lenient reading, a lower bound on update50
online     test on every update over the whole session, groups of unbounded length
online_cap50  the same with groups capped at their 50 most recent members

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
    "cifar_libraries": "explanation_libraries_20260927/cifar10",
    "imagenet_libraries": "explanation_libraries_20260927/imagenet",
    "nes_v1_step1": "specificity_nes_sensitivity_20260927/v1_step1",
    "nes_v2_step2": "specificity_nes_sensitivity_20260927/v2_step2",
    "nes_v3_step2_always": "specificity_nes_sensitivity_20260927/v3_step2_always",
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
                capped = phase2_alarm(assignment, logits, cap=50)
                lag5 = phase2_alarm(assignment[:50], logits[:50], lags=5)
                grouped = int(np.bincount(assignment[:50]).max())
                yield {"corpus": name, "objective": s["objective"], "start": s["workload"], "session_id": s["session_id"],
                       "online": bool(first > 0), "online_first": int(first),
                       "online_cap50": bool(capped > 0), "online_cap50_first": int(capped),
                       "update50": bool(0 < first <= 50), "update50_lag5": bool(lag5 > 0),
                       "single50": bool(phase2_batch(assignment, logits, 50)),
                       "single50_lag5": bool(phase2_batch(assignment, logits, 50, lags=5)),
                       "grouped50": grouped, "tested50": grouped >= MIN_LENGTH}


def summarize(frame, keys):
    g = frame.groupby(keys)
    out = g.agg(n=("online", "size"), update50=("update50", "mean"), single50=("single50", "mean"), online=("online", "mean"),
                online_cap50=("online_cap50", "mean"), update50_lag5=("update50_lag5", "mean"),
                single50_lag5=("single50_lag5", "mean"), grouped50_median=("grouped50", "median"), tested50=("tested50", "mean"))
    out["update50_among_tested"] = frame[frame.tested50].groupby(keys).update50.mean()
    out["single50_among_tested"] = frame[frame.tested50].groupby(keys).single50.mean()
    out["online_first_median"] = frame[frame.online].groupby(keys).online_first.median()
    out["online_cap50_first_median"] = frame[frame.online_cap50].groupby(keys).online_cap50_first.median()
    out["update50_first_median"] = frame[frame.update50].groupby(keys).online_first.median()
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
