#!/usr/bin/env python3
"""Compute every number quoted in the manuscript text from the raw session logs and traces.

Writes paper/jisa_2026/numbers/*.csv and a JSON summary so that each statement in the paper can
be traced to a file. Nothing here fits a model; it only aggregates what the runs recorded.
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

from experiments.gate_trajectory_signatures.lfc_detector import phase2_alarm  # noqa: E402

A = ROOT / "analysis_outputs"
OUT = ROOT / "paper/jisa_2026/numbers"

CORPORA = {
    "cifar_nes_seed0": "lfc_workloads_20260926",          # SHA-identical replay of the main corpus, with LFC observer
    "cifar_nes_seed1": "specificity_resnet18_seed1_20260926",
    "cifar_nes_seed2": "specificity_resnet18_seed2_20260926",
    "cifar_nes_vgg19bn": "specificity_vgg19bn_20260926",
    "cifar_nes_robust": "specificity_robust_engstrom_20260926",
    "gtsrb_nes": "specificity_gtsrb32_20260926",
    "imagenet_nes": "specificity_imagenet_20260926",
    "cifar_simba": "specificity_simba_20260926",
}
CONTROLS = {"cifar_controls": "lfc_controls_20260926"}
EXPLAIN = {"cifar_explain": "explanation_clients_20260926", "imagenet_explain": "explanation_clients_imagenet_20260926"}


def load(directory: str) -> pd.DataFrame:
    root = A / directory
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["root"] = str(root)
    return frame


def acceptance(accepted, optimizer: str) -> float:
    a = np.asarray(accepted)
    if len(a) == 0:
        return np.nan
    return float((a >= 0).mean()) if optimizer == "simba" else float((a == 1).mean())


def q(series, p):
    return float(series.quantile(p)) if len(series) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    inventory, utility = [], []
    for name, directory in CORPORA.items():
        s = load(directory)
        manifest = A / directory / "manifest.csv"
        if not manifest.exists():
            manifest = A / "stateful_specificity_paper_gate_20260924/manifest.csv"
        m = pd.read_csv(manifest)
        inventory.append({
            "corpus": name, "directory": directory, "sessions": len(s),
            "source_images": int(s.dataset_index.nunique()),
            "eval_images": int(s[s.split == "evaluation"].dataset_index.nunique()),
            "fit_images": int(s[s.split == "fit"].dataset_index.nunique()),
            "calibration_images": int(s[s.split == "calibration"].dataset_index.nunique()),
            "manifest_images": len(m), "manifest_classes": int(m.source_label.nunique()),
            "manifest_splits": json.dumps(m.split.value_counts().to_dict()),
            "queries_per_session": json.dumps({k: int(v) for k, v in s.groupby("objective").calls.median().items()}),
        })
        ev = s[s.split == "evaluation"].copy()
        ev["accept"] = [acceptance(a, o) for a, o in zip(ev.accepted, ev.optimizer)]
        lfc = []
        for r in ev.itertuples(index=False):
            t = np.load(Path(r.root) / r.trace)
            lfc.append(phase2_alarm(t["lfc_assignment"], t["logits"]) if "lfc_assignment" in t.files else -1)
        ev["lfc_first"] = lfc
        ev["bl_first"] = [d["blacklight"]["first_alarm"] for d in ev.detectors]
        ev["gp_first"] = [d["gwad_plus"]["first_alarm"] for d in ev.detectors]
        ev["g_first"] = [d["gwad"]["first_alarm"] for d in ev.detectors]
        for (workload, objective), g in ev.groupby(["workload", "objective"]):
            succ = g.first_success[g.first_success > 0]
            utility.append({
                "corpus": name, "start": workload, "objective": objective, "n": len(g),
                "flipped_frac": float((g.first_success > 0).mean()),
                "median_first_flip": float(succ.median()) if len(succ) else np.nan,
                "label_kept_final": float(g.label_preserved.mean()),
                "p_label_start_median": float(g.start_p_label.median()), "p_label_final_median": float(g.final_p_label.median()),
                "margin_start_median": float(g.start_margin.median()), "margin_final_median": float(g.final_margin.median()),
                "psnr_gain_median": float(g.psnr_gain.median()),
                "accept_median": float(g.accept.median()), "accept_q25": q(g.accept, 0.25), "accept_q75": q(g.accept, 0.75),
                "blacklight_native": float((g.bl_first > 0).mean()), "gwad_plus_native": float((g.gp_first > 0).mean()),
                "gwad_native": float((g.g_first > 0).mean()), "lfc": float((g.lfc_first > 0).mean()),
                "blacklight_first_median": float(g.bl_first[g.bl_first > 0].median()) if (g.bl_first > 0).any() else np.nan,
                "gwad_plus_first_median": float(g.gp_first[g.gp_first > 0].median()) if (g.gp_first > 0).any() else np.nan,
                "lfc_first_median": float(g.lfc_first[g.lfc_first > 0].median()) if (g.lfc_first > 0).any() else np.nan,
            })
    pd.DataFrame(inventory).to_csv(OUT / "corpus_inventory.csv", index=False)
    util = pd.DataFrame(utility)
    util.to_csv(OUT / "workload_utility_all.csv", index=False)

    # Per-query Blacklight flag rates on the main corpus (evaluation split)
    s = load(CORPORA["cifar_nes_seed0"])
    ev = s[s.split == "evaluation"]
    rows = []
    for objective, g in ev.groupby("objective"):
        flagged = total = 0
        for r in g.itertuples(index=False):
            c = np.load(Path(r.root) / r.trace)["blacklight_counts"]
            flagged += int((c >= 25).sum())
            total += len(c)
        rows.append({"objective": objective, "queries": total, "flagged": flagged, "rate": flagged / total})
    pd.DataFrame(rows).to_csv(OUT / "blacklight_per_query_main.csv", index=False)
    summary["blacklight_per_query_main"] = {r["objective"]: round(r["rate"], 4) for r in rows}

    # Controls (evaluation + all splits), including per-query Blacklight rate on shuffled
    c = load(CONTROLS["cifar_controls"])
    rows = []
    for objective, g in c.groupby("objective"):
        flagged = total = 0
        lfc = []
        for r in g.itertuples(index=False):
            t = np.load(Path(r.root) / r.trace)
            flagged += int((t["blacklight_counts"] >= 25).sum())
            total += len(t["blacklight_counts"])
            lfc.append(phase2_alarm(t["lfc_assignment"], t["logits"]))
        rows.append({"control": objective, "sessions": len(g), "queries": total, "blacklight_flagged_queries": flagged,
                     "blacklight_session_alarm": float(np.mean([d["blacklight"]["first_alarm"] > 0 for d in g.detectors])),
                     "gwad_plus_session_alarm": float(np.mean([d["gwad_plus"]["first_alarm"] > 0 for d in g.detectors])),
                     "gwad_session_alarm": float(np.mean([d["gwad"]["first_alarm"] > 0 for d in g.detectors])),
                     "lfc_session_alarm": float(np.mean(np.asarray(lfc) > 0)),
                     "gwad_plus_median_windows": float(np.median([d["gwad_plus"]["eligible_windows"] for d in g.detectors]))})
    pd.DataFrame(rows).to_csv(OUT / "controls_all_splits.csv", index=False)

    # Explanation clients
    rows = []
    for name, directory in EXPLAIN.items():
        e = load(directory)
        e = e[e.split == "evaluation"]
        for client, g in e.groupby("objective"):
            rows.append({"corpus": name, "client": client, "n": len(g), "queries": int(g.calls.median()),
                         "del_auc_client_median": float(g.deletion_auc.median()), "del_auc_random_median": float(g.deletion_auc_random.median()),
                         "better_than_random_frac": float((g.deletion_auc < g.deletion_auc_random).mean()),
                         "del_auc_blur_median": float(g.deletion_auc_blur.median()),
                         "del_auc_random_pixel_blur_median": float(g.deletion_auc_random_pixel_blur.median())})
    pd.DataFrame(rows).to_csv(OUT / "explanation_utility.csv", index=False)

    # Rank equivalence (check_rank_equivalence.py; query numbers are 1-based)
    r = pd.read_csv(A / "stateful_specificity_workloads_20260925/rank_equivalence_simba_v2.csv")
    flipped = r[r.attack_first_flip >= 0]
    offsets = (flipped.first_divergence - flipped.attack_first_flip).value_counts().sort_index()
    summary["rank_equivalence"] = {
        "sessions": len(r), "attacks_flipped": len(flipped),
        "identical_through_flip": int(r.identical_through_flip.sum()),
        "divergence_minus_flip": {int(k): int(v) for k, v in offsets.items()},
        "median_flip_query": float(flipped.attack_first_flip.median()), "min_flip": int(flipped.attack_first_flip.min()),
        "max_flip": int(flipped.attack_first_flip.max()),
        "calls": int(r.calls.iloc[0]),
    }
    summary["total_distinct_sessions"] = int(sum(len(load(d)) for k, d in CORPORA.items() if k != "cifar_nes_seed0")
                                             + len(load("stateful_specificity_workloads_20260925"))
                                             + len(load("stateful_specificity_controls_20260925"))
                                             + sum(len(load(d)) for d in EXPLAIN.values()))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(inventory).drop(columns=["directory"]).to_string(index=False))
    print(json.dumps(summary, indent=1))
    cols = ["corpus", "start", "objective", "n", "flipped_frac", "median_first_flip", "p_label_final_median", "margin_final_median",
            "psnr_gain_median", "accept_median", "blacklight_native", "gwad_plus_native", "lfc"]
    print(util[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
