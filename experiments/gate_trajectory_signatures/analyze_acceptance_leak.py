#!/usr/bin/env python3
"""Acceptance-rate leak: post hoc analysis (declared in the preregistration addendum of 2026-09-27)
and the analysis of the preregistered throttled-attack experiment X11 (P14).

Part 1 (post hoc, all corpora): for every (corpus, start, benign objective) cell, the AUROC that the
session's acceptance rate ALONE achieves for attack vs benign, next to the GWAD+ and Blacklight AUROCs
from the main analysis. Orientation is fixed on the fit split; AUROC is on the evaluation split.

Part 2 (X11): the SimBA attack with acceptance throttled to the restore client's rate, scored with the
same logistic models, splits and bootstrap as the main analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.analyze_specificity_workloads import FAMILIES, bootstrap_auc, features  # noqa: E402

A = ROOT / "analysis_outputs"
CORPORA = {
    "CIFAR-10 NES (seed 0)": ("stateful_specificity_workloads_20260925", "nes"),
    "CIFAR-10 NES (seed 1)": ("specificity_resnet18_seed1_20260926", "nes"),
    "CIFAR-10 NES (seed 2)": ("specificity_resnet18_seed2_20260926", "nes"),
    "CIFAR-10 NES (VGG19-BN)": ("specificity_vgg19bn_20260926", "nes"),
    "CIFAR-10 NES (robust)": ("specificity_robust_engstrom_20260926", "nes"),
    "GTSRB NES": ("specificity_gtsrb32_20260926", "nes"),
    "ImageNet tiled NES": ("specificity_imagenet_20260926", "nes_tiled"),
    "CIFAR-10 SimBA": ("specificity_simba_20260926", "simba"),
}
BENIGN = ("restore", "confidence_boost", "boundary_probe", "counterfactual")
INVALID = {("ImageNet tiled NES", "restore")}


def load(directory: str) -> pd.DataFrame:
    root = A / directory
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["root"] = str(root)
    frame["accept"] = [float((np.asarray(a) >= 0).mean()) if o == "simba" else float((np.asarray(a) == 1).mean())
                       for a, o in zip(frame.accepted, frame.optimizer)]
    return frame


def oriented_auc(fit_y, fit_s, ev_y, ev_s):
    sign = 1.0 if roc_auc_score(fit_y, fit_s) >= 0.5 else -1.0
    return float(roc_auc_score(ev_y, sign * ev_s)), sign


def part1() -> pd.DataFrame:
    rows = []
    for label, (directory, optimizer) in CORPORA.items():
        s = load(directory)
        sep = pd.read_csv(A / directory / "analysis/separability.csv")
        for start in ("denoise", "deblur"):
            attack = s[(s.workload == start) & (s.objective == "attack")]
            for objective in BENIGN:
                if (label, objective) in INVALID:
                    continue
                benign = s[(s.workload == start) & (s.objective == objective)]
                if benign.empty:
                    continue
                both = pd.concat([attack.assign(y=1), benign.assign(y=0)])
                fit, ev = both[both.split == "fit"], both[both.split == "evaluation"]
                auc, sign = oriented_auc(fit.y, fit.accept, ev.y, ev.accept)
                cell = lambda det: float(sep[(sep.workload == start) & (sep.negative == objective) & (sep.detector == det)  # noqa: E731
                                             & (sep.prefix == "1024")].auroc.iloc[0])
                rows.append({
                    "corpus": label, "optimizer": optimizer, "start": start, "objective": objective,
                    "accept_attack_median": float(ev[ev.y == 1].accept.median()),
                    "accept_benign_median": float(ev[ev.y == 0].accept.median()),
                    "acceptance_only_auroc": auc, "attack_accepts_more": bool(sign > 0),
                    "gwad_plus_auroc": cell("logreg:gwad_plus"), "blacklight_auroc": cell("logreg:blacklight"),
                })
    return pd.DataFrame(rows)


def part2(bootstrap: int) -> pd.DataFrame:
    base = load("specificity_simba_20260926")
    thr = load("specificity_simba_throttled_20260927")
    rows = []
    feature_cache = {}

    def feats(r):
        key = (r.root, r.trace)
        if key not in feature_cache:
            feature_cache[key] = features(np.load(Path(r.root) / r.trace), 1024)
        return feature_cache[key]

    for start in ("denoise", "deblur"):
        positives = {"throttled attack": thr[(thr.workload == start)], "unthrottled attack": base[(base.workload == start) & (base.objective == "attack")]}
        for pos_name, pos in positives.items():
            for objective in ("restore", "confidence_boost"):
                neg = base[(base.workload == start) & (base.objective == objective)]
                paired = pos.merge(neg, on=["dataset_index", "split"], suffixes=("_a", "_b"))
                records = []
                for r in paired.itertuples(index=False):
                    for root, trace, accept, y in ((r.root_a, r.trace_a, r.accept_a, 1), (r.root_b, r.trace_b, r.accept_b, 0)):
                        f = feats(pd.Series({"root": root, "trace": trace}))
                        records.append({"dataset_index": r.dataset_index, "split": r.split, "y": y, "accept": accept, **f})
                frame = pd.DataFrame(records)
                fit, ev = frame[frame.split == "fit"], frame[frame.split == "evaluation"]
                rng = np.random.default_rng(0)
                out = {"start": start, "positive": pos_name, "negative": objective, "n_eval_pairs": int(ev.y.sum()),
                       "accept_pos_median": float(ev[ev.y == 1].accept.median()), "accept_neg_median": float(ev[ev.y == 0].accept.median())}
                acc_auc, _ = oriented_auc(fit.y, fit.accept, ev.y, ev.accept)
                out["acceptance_only_auroc"] = acc_auc
                cols_all = [c for c in frame.columns if c.startswith(("out_", "bl_", "gwad")) and not c.endswith("has_window")]
                for family, pick in FAMILIES.items():
                    use = [c for c in cols_all if pick(c)]
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
                    clf.fit(fit[use].fillna(0.0), fit.y)
                    score = clf.predict_proba(ev[use].fillna(0.0))[:, 1]
                    lo, hi = bootstrap_auc(ev.y.to_numpy(), score, ev.dataset_index.to_numpy(), bootstrap, rng)
                    out[f"{family}_auroc"] = float(roc_auc_score(ev.y, score))
                    out[f"{family}_lo"], out[f"{family}_hi"] = lo, hi
                rows.append(out)
        ev_thr = thr[(thr.workload == start) & (thr.split == "evaluation")]
        ev_att = base[(base.workload == start) & (base.objective == "attack") & (base.split == "evaluation")]
        for name, g in (("throttled attack", ev_thr), ("unthrottled attack", ev_att)):
            succ = g.first_success[g.first_success > 0]
            rows.append({"start": start, "positive": name, "negative": "(attack outcome)", "n_eval_pairs": len(g),
                         "success_rate": float((g.first_success > 0).mean()),
                         "median_first_success": float(succ.median()) if len(succ) else np.nan,
                         "blacklight_native": float(np.mean([d["blacklight"]["first_alarm"] > 0 for d in g.detectors])),
                         "gwad_plus_native": float(np.mean([d["gwad_plus"]["first_alarm"] > 0 for d in g.detectors]))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=A / "acceptance_leak_20260927")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-throttled", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    p1 = part1()
    p1.to_csv(args.output_dir / "acceptance_vs_detectors.csv", index=False)
    print(p1.round(3).to_string(index=False))
    if not args.skip_throttled:
        p2 = part2(args.bootstrap)
        p2.to_csv(args.output_dir / "throttled_attack.csv", index=False)
        cols = ["start", "positive", "negative", "n_eval_pairs", "accept_pos_median", "accept_neg_median", "acceptance_only_auroc",
                "gwad_plus_auroc", "gwad_plus_lo", "gwad_plus_hi", "blacklight_auroc", "output_trajectory_auroc",
                "success_rate", "median_first_success", "blacklight_native", "gwad_plus_native"]
        print(p2[[c for c in cols if c in p2.columns]].round(3).to_string(index=False))
        thr = p2[(p2.positive == "throttled attack") & (p2.negative == "restore")]
        outcome = p2[(p2.positive == "throttled attack") & (p2.negative == "(attack outcome)")]
        checks = {
            "P14a_gwad_plus_throttled_vs_restore_le_0.65": {r.start: (round(r.gwad_plus_auroc, 3), bool(r.gwad_plus_auroc <= 0.65)) for r in thr.itertuples()},
            "P14b_throttled_success_ge_0.50": {r.start: (round(r.success_rate, 3), bool(r.success_rate >= 0.50)) for r in outcome.itertuples()},
        }
        (args.output_dir / "throttled_predictions.json").write_text(json.dumps(checks, indent=2))
        print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
