#!/usr/bin/env python3
"""Compute the numbers quoted in the manuscript from the raw session logs and traces.

Writes paper/jisa_2026/numbers/*.csv and a JSON summary so that each statement in the paper can be
traced to a file. Nothing here fits a model; it only aggregates what the runs recorded.

Revision 2 (2026-09-27): Blacklight's decisions follow the published rule (blacklight_rule.py). Added:
the share of queries that Blacklight flags and the share of decision windows that GWAD+ flags, for
every client; operating points on every corpus; attacks that succeed before the first alarm; alarms
raised before the first query that depends on the objective; the composition of the session count and
the query streams that occur in more than one corpus.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402

A = ROOT / "analysis_outputs"
OUT = ROOT / "paper/jisa_2026/numbers"

CORPORA = {
    "cifar_nes_seed0": "lfc_workloads_20260926",          # the main corpus, regenerated with the Lee et al. observer attached
    "cifar_nes_seed1": "specificity_resnet18_seed1_20260926",
    "cifar_nes_seed2": "specificity_resnet18_seed2_20260926",
    "cifar_nes_vgg19bn": "specificity_vgg19bn_20260926",
    "cifar_nes_robust": "specificity_robust_engstrom_20260926",
    "gtsrb_nes": "specificity_gtsrb32_20260926",
    "imagenet_nes": "specificity_imagenet_20260926",
    "cifar_simba": "specificity_simba_20260926",
}
ORIGINALS = {"cifar_nes_seed0": "stateful_specificity_workloads_20260925", "cifar_controls": "stateful_specificity_controls_20260925"}
CONTROLS = {"cifar_controls": "lfc_controls_20260926"}
EXPLAIN = {"cifar_explain": "explanation_clients_20260926", "imagenet_explain": "explanation_clients_imagenet_20260926"}
OPTIMIZING = ("attack", "restore", "confidence_boost", "boundary_probe", "counterfactual")
# queries that precede the first query which depends on the objective: the start and the first probes
OBJECTIVE_FREE_QUERIES = {"nes": 17, "simba": 3}


def load(directory: str) -> pd.DataFrame:
    root = A / directory
    frame = pd.DataFrame(blacklight_rule.load_sessions(root))
    frame["root"] = str(root)
    return frame


def acceptance(accepted, optimizer: str) -> float:
    a = np.asarray(accepted)
    if len(a) == 0:
        return np.nan
    return float((a >= 0).mean()) if optimizer == "simba" else float((a == 1).mean())


def quantile(series, p):
    return float(series.quantile(p)) if len(series) else np.nan


def detector_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bl_first"] = [d["blacklight"]["first_alarm"] for d in frame.detectors]
    frame["bl_flagged_fraction"] = [d["blacklight"]["flagged_fraction"] for d in frame.detectors]
    frame["gp_first"] = [d["gwad_plus"]["first_alarm"] for d in frame.detectors]
    frame["g_first"] = [d["gwad"]["first_alarm"] for d in frame.detectors]
    windows, flagged = [], []
    for r in frame.itertuples(index=False):
        p = np.load(Path(r.root) / r.trace)["gwad_plus_predictions"]
        windows.append(len(p))
        flagged.append(int((p != 0).sum()))
    frame["gp_windows"], frame["gp_windows_flagged"] = windows, flagged
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    inventory, utility, before = [], [], []
    digests = defaultdict(list)
    split_counts = Counter()
    for name, directory in CORPORA.items():
        s = load(directory)
        for r in s.itertuples(index=False):
            digests[r.query_sha256].append((name, r.objective))
        split_counts.update(s.split.tolist())
        manifest = A / directory / "manifest.csv"
        if not manifest.exists():
            manifest = A / "stateful_specificity_paper_gate_20260924/manifest.csv"
        m = pd.read_csv(manifest)
        m = m[m.split != "development"]
        inventory.append({
            "corpus": name, "directory": directory, "sessions": len(s),
            "source_images": int(s.dataset_index.nunique()),
            "eval_images": int(s[s.split == "evaluation"].dataset_index.nunique()),
            "fit_images": int(s[s.split == "fit"].dataset_index.nunique()),
            "calibration_images": int(s[s.split == "calibration"].dataset_index.nunique()),
            "classes_with_images": int(s.source_label.nunique()),
            "queries_per_session": json.dumps({k: int(v) for k, v in s.groupby("objective").calls.median().items()}),
        })
        ev = detector_columns(s[s.split == "evaluation"])
        ev["accept"] = [acceptance(a, o) for a, o in zip(ev.accepted, ev.optimizer)]
        free = OBJECTIVE_FREE_QUERIES["simba" if name == "cifar_simba" else "nes"]
        for (workload, objective), g in ev.groupby(["workload", "objective"]):
            succ = g.first_success[g.first_success > 0]
            bl = g.bl_first[g.bl_first > 0]
            utility.append({
                "corpus": name, "start": workload, "objective": objective, "n": len(g),
                "flipped_frac": float((g.first_success > 0).mean()),
                "median_first_flip": float(succ.median()) if len(succ) else np.nan,
                "label_kept_final": float(g.label_preserved.mean()),
                "p_label_start_median": float(g.start_p_label.median()), "p_label_final_median": float(g.final_p_label.median()),
                "margin_start_median": float(g.start_margin.median()), "margin_final_median": float(g.final_margin.median()),
                "psnr_gain_median": float(g.psnr_gain.median()),
                "accept_median": float(g.accept.median()), "accept_q25": quantile(g.accept, 0.25), "accept_q75": quantile(g.accept, 0.75),
                "sessions_without_accepted_step": float((g.accept == 0).mean()),
                "blacklight_alarm": float((g.bl_first > 0).mean()), "gwad_alarm": float((g.g_first > 0).mean()),
                "gwad_plus_alarm": float((g.gp_first > 0).mean()),
                "blacklight_first_median": float(bl.median()) if len(bl) else np.nan,
                "blacklight_first_q25": quantile(bl, 0.25), "blacklight_first_q75": quantile(bl, 0.75),
                "blacklight_first_max": float(bl.max()) if len(bl) else np.nan,
                "blacklight_alarm_before_objective": float(((g.bl_first > 0) & (g.bl_first <= free)).mean()),
                "gwad_plus_first_median": float(g.gp_first[g.gp_first > 0].median()) if (g.gp_first > 0).any() else np.nan,
                "gwad_first_median": float(g.g_first[g.g_first > 0].median()) if (g.g_first > 0).any() else np.nan,
                "blacklight_flagged_queries": float((g.bl_flagged_fraction * g.calls).sum() / g.calls.sum()),
                "gwad_plus_flagged_windows": float(g.gp_windows_flagged.sum() / g.gp_windows.sum()) if g.gp_windows.sum() else np.nan,
            })
        for start, g in [(w, ev[(ev.objective == "attack") & (ev.workload == w)]) for w in ("denoise", "deblur")] + \
                        [("pooled", ev[ev.objective == "attack"])]:
            succ = g[g.first_success > 0]
            row = {"corpus": name, "start": start, "attack_sessions": len(g), "successful": len(succ),
                   "success_rate": len(succ) / len(g), "median_first_success": float(succ.first_success.median()) if len(succ) else np.nan}
            for det, col in (("blacklight", "bl_first"), ("gwad", "g_first"), ("gwad_plus", "gp_first")):
                k = int(((succ[col] < 0) | (succ.first_success < succ[col])).sum())
                row[f"{det}_success_before_alarm"] = k
                row[f"{det}_success_before_alarm_frac"] = k / len(succ) if len(succ) else np.nan
            before.append(row)
        if name == "cifar_nes_seed0":
            first = {o: ev[ev.objective == o].bl_first for o in ("attack", "restore", "confidence_boost")}
            summary["blacklight_first_alarm_main"] = {
                o: {"median": float(v.median()), "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)), "max": int(v.max())}
                for o, v in first.items()}
            summary["blacklight_first_alarm_ks_vs_attack"] = {
                o: {"D": float(ks_2samp(first["attack"], first[o]).statistic), "p": float(ks_2samp(first["attack"], first[o]).pvalue)}
                for o in ("restore", "confidence_boost")}
            pairs = ev[ev.objective == "attack"].merge(ev[ev.objective.isin(["restore", "confidence_boost"])],
                                                       on=["dataset_index", "workload"], suffixes=("_a", "_b"))
            summary["blacklight_first_alarm_pairs_main"] = {
                "pairs": len(pairs), "benign_same_query": float((pairs.bl_first_a == pairs.bl_first_b).mean()),
                "benign_earlier": float((pairs.bl_first_b < pairs.bl_first_a).mean()),
                "benign_later": float((pairs.bl_first_b > pairs.bl_first_a).mean())}
    pd.DataFrame(inventory).to_csv(OUT / "corpus_inventory.csv", index=False)
    util = pd.DataFrame(utility)
    util.to_csv(OUT / "workload_utility_all.csv", index=False)
    pd.DataFrame(before).to_csv(OUT / "success_before_alarm.csv", index=False)

    # Session count and query streams that occur more than once
    controls = load(CONTROLS["cifar_controls"])
    explain = {k: load(d) for k, d in EXPLAIN.items()}
    split_counts.update(controls.split.tolist())
    for e in explain.values():
        split_counts.update(e.split.tolist())
    repeated = {k: v for k, v in digests.items() if len(v) > 1}
    summary["sessions"] = {
        "matched_corpora": int(sum(i["sessions"] for i in inventory)), "control_streams": len(controls),
        "explanation_32px": len(explain["cifar_explain"]), "explanation_224px": len(explain["imagenet_explain"]),
        "total": int(sum(i["sessions"] for i in inventory) + len(controls) + sum(len(e) for e in explain.values())),
        "by_split": dict(split_counts),
        "matched_corpora_distinct_query_streams": len(digests),
        "matched_corpora_sessions_repeating_a_stream": int(sum(len(v) for v in repeated.values()) - len(repeated)),
        "repeated_streams_by_client": dict(Counter(v[0][1] for v in repeated.values())),
        "repeated_streams_corpora": sorted({c for v in repeated.values() for c, _ in v}),
    }
    replay = {}
    for key, original in ORIGINALS.items():
        a = load(original).set_index("session_id").query_sha256
        b = (load(CORPORA[key]) if key in CORPORA else controls).set_index("session_id").query_sha256
        replay[key] = {"sessions": len(a), "identical_digests": int((a == b.reindex(a.index)).sum())}
    summary["replay_with_observer"] = replay

    # Control streams: all splits and evaluation split
    rows = []
    for scope, frame in (("all_splits", controls), ("evaluation", controls[controls.split == "evaluation"])):
        frame = detector_columns(frame)
        for objective, g in frame.groupby("objective"):
            rows.append({"scope": scope, "control": objective, "sessions": len(g), "queries": int(g.calls.sum()),
                         "blacklight_alarm": float((g.bl_first > 0).mean()), "blacklight_alarm_sessions": int((g.bl_first > 0).sum()),
                         "blacklight_alarm_recorded_rule_sessions": int(sum(d["blacklight"]["first_alarm_recorded"] > 0 for d in g.detectors)),
                         "blacklight_flagged_queries": float((g.bl_flagged_fraction * g.calls).sum() / g.calls.sum()),
                         "gwad_alarm": float((g.g_first > 0).mean()), "gwad_plus_alarm": float((g.gp_first > 0).mean()),
                         "gwad_median_windows": float(np.median([d["gwad"]["eligible_windows"] for d in g.detectors])),
                         "gwad_plus_median_windows": float(np.median([d["gwad_plus"]["eligible_windows"] for d in g.detectors])),
                         "gwad_plus_flagged_windows": float(g.gp_windows_flagged.sum() / g.gp_windows.sum()) if g.gp_windows.sum() else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "controls.csv", index=False)

    # Explanation clients
    rows = []
    for name, e in explain.items():
        e = detector_columns(e[e.split == "evaluation"])
        for client, g in e.groupby("objective"):
            bl = g.bl_first[g.bl_first > 0]
            rows.append({"corpus": name, "client": client, "n": len(g), "queries": int(g.calls.median()),
                         "blacklight_alarm": float((g.bl_first > 0).mean()), "blacklight_first_median": float(bl.median()) if len(bl) else np.nan,
                         "blacklight_first_q75": quantile(bl, 0.75), "blacklight_first_max": float(bl.max()) if len(bl) else np.nan,
                         "blacklight_alarm_by_query_10": float(((g.bl_first > 0) & (g.bl_first <= 10)).mean()),
                         "blacklight_flagged_queries": float((g.bl_flagged_fraction * g.calls).sum() / g.calls.sum()),
                         "gwad_alarm": float((g.g_first > 0).mean()), "gwad_plus_alarm": float((g.gp_first > 0).mean()),
                         "gwad_plus_first_median": float(g.gp_first[g.gp_first > 0].median()) if (g.gp_first > 0).any() else np.nan,
                         "gwad_plus_flagged_windows": float(g.gp_windows_flagged.sum() / g.gp_windows.sum()) if g.gp_windows.sum() else np.nan,
                         "del_auc_client_median": float(g.deletion_auc.median()), "del_auc_random_median": float(g.deletion_auc_random.median()),
                         "better_than_random_frac": float((g.deletion_auc < g.deletion_auc_random).mean()),
                         "del_auc_blur_median": float(g.deletion_auc_blur.median()),
                         "del_auc_random_pixel_blur_median": float(g.deletion_auc_random_pixel_blur.median()),
                         "better_than_random_pixel_blur_frac": float((g.deletion_auc_blur < g.deletion_auc_random_pixel_blur).mean())})
    pd.DataFrame(rows).to_csv(OUT / "explanation_clients.csv", index=False)

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

    # ImageNet restoration (declared invalid): how many sessions accept a step at all
    imagenet = load(CORPORA["imagenet_nes"])
    res = imagenet[imagenet.objective == "restore"]
    steps = np.array([int((np.asarray(a) == 1).sum()) for a in res.accepted])
    summary["imagenet_restoration"] = {"sessions": len(res), "sessions_without_accepted_step": int((steps == 0).sum()),
                                       "largest_number_of_accepted_steps": int(steps.max()), "accepted_steps_in_total": int(steps.sum()),
                                       "iterations_per_session": int(np.median([len(a) for a in res.accepted])),
                                       "largest_psnr_gain": float(res.psnr_gain.max())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(pd.DataFrame(inventory).drop(columns=["directory"]).to_string(index=False))
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(before).round(3).to_string(index=False))
    cols = ["corpus", "start", "objective", "n", "flipped_frac", "accept_median", "blacklight_alarm", "blacklight_first_median",
            "blacklight_alarm_before_objective", "blacklight_flagged_queries", "gwad_plus_alarm", "gwad_plus_flagged_windows"]
    print(util[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
