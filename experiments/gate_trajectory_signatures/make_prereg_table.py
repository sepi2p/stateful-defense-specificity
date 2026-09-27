#!/usr/bin/env python3
"""Appendix A: every frozen prediction with its outcome, computed from the analysis outputs.

The prediction texts paraphrase docs/specificity_workloads_preregistration.md; the outcomes and
verdicts are evaluated here from the result files, not typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis_outputs"
N = ROOT / "paper/jisa_2026/numbers"
OUT = ROOT / "paper/jisa_2026/tables"

QUERY_ONLY = ["single:gwad_plus_max", "single:bl_max", "logreg:gwad_plus", "logreg:blacklight", "logreg:query_only_all"]
QUERY_ONLY_FAMILIES = ["logreg:gwad_plus", "logreg:blacklight", "logreg:query_only_all"]
MATCHED = ["restore", "confidence_boost", "boundary_probe", "counterfactual"]
LATER = [("SimBA", "specificity_simba_20260926"), ("VGG19-BN", "specificity_vgg19bn_20260926"),
         ("robust ResNet-50", "specificity_robust_engstrom_20260926"), ("seed 1", "specificity_resnet18_seed1_20260926"),
         ("seed 2", "specificity_resnet18_seed2_20260926"), ("GTSRB", "specificity_gtsrb32_20260926")]


def sep(directory):
    return pd.read_csv(A / directory / "analysis/separability.csv")


def cells(s, detectors, negatives, prefix):
    return s[s.detector.isin(detectors) & s.negative.isin(negatives) & (s.prefix == prefix)]


def r2(x):
    return f"{x:.2f}"


def main():
    rows = []

    def add(pid, corpus, prediction, outcome, held):
        rows.append((pid, corpus, prediction, outcome, "held" if held else "\\textbf{failed}"))

    m = sep("stateful_specificity_workloads_20260925")
    c = cells(m, QUERY_ONLY, MATCHED, "1024")
    over = c[c.auroc > 0.60]
    add("P1", "main", "Query-only detectors: AUROC $\\le0.60$ against every matched NES client, full session",
        f"{len(c) - len(over)} of {len(c)} cells; maximum {r2(c.auroc.max())}", over.empty)
    walk = cells(m, QUERY_ONLY, ["random_walk"], "1024").auroc.min()
    add("P2", "main", "Query-only detectors separate the random walk better than any matched client",
        f"random walk $\\ge{r2(walk)}$; matched clients $\\le{r2(c.auroc.max())}$", walk > c.auroc.max())
    o = m[(m.detector == "logreg:output_trajectory") & m.negative.isin(["restore", "confidence_boost"]) & m.prefix.isin(["128", "256", "512", "1024"])]
    add("P3", "main", "Model outputs: AUROC $\\ge0.90$ against restoration and confidence raising",
        f"{r2(o.auroc.min())}--{r2(o.auroc.max())} at 128 to 1{{,}}021 queries", o.auroc.min() >= 0.90)
    t = m[(m.detector == "logreg:output_trajectory") & m.negative.isin(["boundary_probe", "counterfactual"]) & (m.prefix == "attack_truncated")]
    add("P4", "main", "Model outputs: AUROC $\\le0.75$ against Tier~C before the attack's first label change",
        f"{r2(t.auroc.min())}--{r2(t.auroc.max())}", t.auroc.max() <= 0.75)

    ctl = pd.read_csv(A / "stateful_specificity_controls_20260925/controls_summary.csv").set_index("objective")
    rate = lambda obj, col: float(ctl.loc[obj, col].split("=")[1].split("[")[0])  # noqa: E731
    frac = lambda obj, col: ctl.loc[obj, col].split("=")[0].strip()  # noqa: E731
    k, n = (int(v) for v in frac("shuffled", "blacklight_alarm").split("/"))
    add("P5a", "controls", "Blacklight alarms on $\\le1\\%$ of shuffled sessions", f"{100 * k / n:.2f}\\% ({k}/{n})", k / n <= 0.01)
    add("P5a", "controls", "GWAD and GWAD+ never alarm on shuffled sessions",
        f"{frac('shuffled', 'gwad_alarm')} and {frac('shuffled', 'gwad_plus_alarm')}",
        rate("shuffled", "gwad_alarm") == 0 and rate("shuffled", "gwad_plus_alarm") == 0)
    add("P5b", "controls", "GWAD+ alarms on $\\ge90\\%$ of Gaussian-noise sessions",
        f"{100 * rate('noise', 'gwad_plus_alarm'):.0f}\\%", rate("noise", "gwad_plus_alarm") >= 0.90)
    add("P5c", "controls", "Blacklight alarms on $\\ge90\\%$ of sweep sessions",
        f"{100 * rate('sweep', 'blacklight_alarm'):.0f}\\%", rate("sweep", "blacklight_alarm") >= 0.90)
    lb = max(rate("shuffled", "ljung_box_h20"), rate("noise", "ljung_box_h20"))
    add("P5d", "controls", "Ljung--Box test alone (20 lags, whole session) flags $\\le5\\%$ of shuffled and noise sessions",
        f"{100 * rate('shuffled', 'ljung_box_h20'):.1f}\\% and {100 * rate('noise', 'ljung_box_h20'):.1f}\\%", lb <= 0.05)

    for pid, label, directory in (("P6", "explanation, 32\\,px", "explanation_clients_20260926"),
                                  ("P13", "explanation, 224\\,px", "explanation_clients_imagenet_20260926")):
        e = pd.read_csv(A / directory / "explanation_summary.csv").set_index("client")
        bl = {k: float(e.loc[k, "blacklight"].split(" ")[0]) for k in e.index}
        first = {k: e.loc[k, "blacklight_median_first"] for k in e.index}
        three = ["lime", "kernelshap", "occlusion"]
        if pid == "P6":
            ok = all(bl[k] >= 0.90 and first[k] <= 50 for k in three)
            add("P6a", label, "Blacklight blocks $\\ge90\\%$ of LIME, KernelSHAP and occlusion sessions, median first alarm $\\le50$",
                f"{100 * min(bl[k] for k in three):.0f}\\%; first alarm at {min(first[k] for k in three):.0f}--{max(first[k] for k in three):.0f}", ok)
            add("P6b", label, "Blacklight blocks $\\le10\\%$ of RISE sessions", f"{100 * bl['rise']:.0f}\\%", bl["rise"] <= 0.10)
        else:
            add("P13a", label, "Blacklight blocks $\\ge90\\%$ of LIME, KernelSHAP and occlusion sessions",
                f"{100 * min(bl[k] for k in three):.0f}\\%", all(bl[k] >= 0.90 for k in three))
        lows = {k: float(e.loc[k, "util_diff"].split("[")[1].split(",")[0]) for k in e.index}
        meds = {k: float(e.loc[k, "util_diff"].split(" ")[0]) for k in e.index}
        add("P6c" if pid == "P6" else "P13b", label, "Every client beats random orderings (median difference $>0$, interval excludes 0)",
            f"medians {min(meds.values()):.2f}--{max(meds.values()):.2f}; lower bounds $\\ge{min(lows.values()):.2f}$",
            all(v > 0 for v in lows.values()) and all(v > 0 for v in meds.values()))

    lfc = pd.read_csv(A / "lfc_workloads_20260926/analysis/lfc_alarm_rates.csv")
    g = lambda objs: lfc[lfc.objective.isin(objs)].alarm_primary  # noqa: E731
    add("P7a", "main", "Lee et al.\\ (online) flags $\\ge90\\%$ of restoration and confidence-raising sessions",
        f"{100 * g(['restore', 'confidence_boost']).min():.0f}--{100 * g(['restore', 'confidence_boost']).max():.0f}\\%",
        g(["restore", "confidence_boost"]).min() >= 0.90)
    add("P7b", "main", "Lee et al.\\ (online) flags $\\ge95\\%$ of attack sessions", f"{100 * g(['attack']).min():.0f}\\%", g(["attack"]).min() >= 0.95)
    add("P7c", "controls", "Lee et al.\\ (online) flags $\\le2\\%$ of shuffled and noise sessions",
        f"{100 * g(['shuffled', 'noise']).max():.0f}\\%", g(["shuffled", "noise"]).max() <= 0.02)
    add("P7d", "main", "Lee et al.\\ (online) flags $\\ge90\\%$ of random walks", f"{100 * g(['random_walk']).min():.0f}\\%", g(["random_walk"]).min() >= 0.90)

    for label, directory in LATER + [("ImageNet", "specificity_imagenet_20260926")]:
        s = sep(directory)
        tier_b = ["confidence_boost"] if label == "ImageNet" else ["restore", "confidence_boost"]
        pid = "P12" if label == "ImageNet" else ""
        q = cells(s, QUERY_ONLY_FAMILIES, tier_b, "1024")
        worst = q.loc[q.auroc.idxmax()]
        add(pid or "P8", label, "Query-only detectors: AUROC $\\le0.62$ against Tier~B",
            f"maximum {worst.auroc:.3f} ({worst.detector.split(':')[1].replace('_', ' ')})" if q.auroc.max() > 0.62 else f"maximum {r2(q.auroc.max())}",
            q.auroc.max() <= 0.62)
        w = cells(s, QUERY_ONLY_FAMILIES, ["random_walk"], "1024")
        if not w.empty:
            add(pid or "P8", label, "Query-only detectors: AUROC $\\ge0.90$ against the random walk", f"minimum {r2(w.auroc.min())}", w.auroc.min() >= 0.90)
        p = cells(s, ["logreg:output_trajectory"], ["boundary_probe", "counterfactual"], "pre_either_flip")
        add(pid or "P9", label, "Model outputs: AUROC $\\le0.70$ against Tier~C before either label change",
            f"maximum {p.auroc.max():.3f}" if p.auroc.max() > 0.69 else f"maximum {r2(p.auroc.max())}", p.auroc.max() <= 0.70)
        b = cells(s, ["logreg:output_trajectory"], tier_b, "1024")
        add(pid or "P10", label, "Model outputs: AUROC $\\ge0.90$ against Tier~B", f"minimum {r2(b.auroc.min())}", b.auroc.min() >= 0.90)
        rates = pd.read_csv(A / directory / "analysis/alarm_rates.csv")
        lf = rates[rates.objective.isin(tier_b)].lfc
        add(pid or "P11", label, "Lee et al.\\ (online) flags $\\ge90\\%$ of Tier-B sessions", f"minimum {100 * lf.min():.1f}\\%", lf.min() >= 0.90)

    thr = json.loads((A / "acceptance_leak_20260927/throttled_predictions.json").read_text())
    a = thr["P14a_gwad_plus_throttled_vs_restore_le_0.65"]
    add("P14a", "throttled SimBA", "GWAD+ AUROC $\\le0.65$, throttled attack against restoration",
        " and ".join(f"{v[0]:.2f}" for v in a.values()), all(v[1] for v in a.values()))
    b = thr["P14b_throttled_success_ge_0.50"]
    add("P14b", "throttled SimBA", "Throttled attack succeeds in $\\ge50\\%$ of the sessions",
        " and ".join(f"{100 * v[0]:.0f}\\%" for v in b.values()), all(v[1] for v in b.values()))

    sens = json.loads((A / "lfc_sensitivity_20260927/sensitivity_predictions.json").read_text())
    faithful = {k: v for k, v in sens.items() if v["reproduces_paper"]}
    add("P15", "sensitivity", "Every variant of the Lee et al.\\ detector that reproduces the paper flags $\\ge90\\%$ of Tier-B sessions (online)",
        f"{len(faithful)} of {len(sens)} variants reproduce the paper; minimum {100 * min(v['min_benign_optimizer_alarm'] for v in faithful.values()):.0f}\\%",
        all(v["P15_holds"] for v in faithful.values()))

    ops = pd.read_csv(A / "output_diagnostic_20260926/two_stage_operating_points.csv")
    e1 = ops[(ops.unit == "tr_group") & (ops.prefix == "full")]
    tpr = float(e1[e1.objective == "attack"].alarm_rate.iloc[0])
    worst = e1[e1.tier == "benign"].sort_values("alarm_rate").iloc[-1]
    add("E1", "main (exploratory)", "Trend statistic, two-stage, 1\\% pooled threshold: detects $\\ge50\\%$ of attacks with $\\le5\\%$ on every benign client",
        f"attacks {100 * tpr:.0f}\\%; worst benign client {100 * worst.alarm_rate:.1f}\\% ({worst.objective})",
        tpr >= 0.5 and worst.alarm_rate <= 0.05)

    frame = pd.DataFrame(rows, columns=["id", "corpus", "prediction", "outcome", "verdict"])
    frame.to_csv(N / "preregistration_outcomes.csv", index=False)
    body = "\n".join(f"{r.id} & {r.corpus} & {r.prediction} & {r.outcome} & {r.verdict} \\\\" for r in frame.itertuples())
    head = "ID & Corpus & Prediction & Outcome & Verdict \\\\\n\\midrule"
    text = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{L{0.9cm}L{1.9cm}L{5.0cm}L{3.4cm}L{1.1cm}}\n"
            "\\caption{Frozen predictions and their outcomes (evaluation split). ``Main'' is the CIFAR-10 ResNet-18 NES corpus. "
            "Predictions P8--P11 were frozen for each later corpus and P12 applies them to ImageNet, where restoration is excluded as "
            "an invalid workload.}\\label{tab:prereg}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n\\toprule\n" + head + "\n\\endhead\n"
            + body + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg.tex").write_text(text)
    failed = frame[frame.verdict != "held"]
    print(f"{len(frame)} checks, {len(failed)} failed")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(frame[["id", "corpus", "outcome", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
