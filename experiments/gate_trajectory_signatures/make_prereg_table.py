#!/usr/bin/env python3
"""Appendix A: every prediction that was written down in advance, with its outcome.

The prediction texts paraphrase docs/specificity_workloads_preregistration.md; outcomes and verdicts
are evaluated here from the result files, not typed. Every check is evaluated twice: on the first
analysis, which applied Blacklight's threshold as "25 or more matches", and on the corrected analysis
(published rule, "more than 25"; files with the suffix r2). The table reports the corrected analysis
and states the first outcome wherever outcome or verdict differ.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
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
SOURCES = {
    "first": {"analysis": "analysis", "suffix": "", "leak": "acceptance_leak_20260927"},
    "r2": {"analysis": "analysis_r2", "suffix": "_r2", "leak": "acceptance_leak_r2"},
}


def fmt(value, digits=0):
    """Format with rounding half up (Python's own formatting rounds binary floats half to even)."""
    return str(Decimal(f"{float(value):.9f}").quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def pc(value, digits=0):
    return fmt(100 * float(value), digits) + "\\%"


def cells(s, detectors, negatives, prefix):
    return s[s.detector.isin(detectors) & s.negative.isin(negatives) & (s.prefix == prefix)]


def evaluate(which: str) -> list[dict]:
    src = SOURCES[which]
    rows = []

    def add(pid, corpus, prediction, outcome, held):
        rows.append({"id": pid, "corpus": corpus, "prediction": prediction, "outcome": outcome, "held": bool(held)})

    def sep(directory):
        return pd.read_csv(A / directory / src["analysis"] / "separability.csv")

    m = sep("stateful_specificity_workloads_20260925")
    c = cells(m, QUERY_ONLY, MATCHED, "1024")
    over = c[c.auroc > 0.60]
    add("P1", "main", "Query-only detectors: AUROC $\\le0.60$ against every matched NES client, full session",
        f"{len(c) - len(over)} of {len(c)} cells; maximum {fmt(c.auroc.max(), 3)}", over.empty)
    walk = cells(m, QUERY_ONLY, ["random_walk"], "1024").auroc.min()
    add("P2", "main", "Query-only detectors separate the random walk better than any matched client",
        f"random walk $\\ge{fmt(walk, 2)}$ (single statistics; logistic models 1.00); matched clients $\\le{fmt(c.auroc.max(), 2)}$",
        walk > c.auroc.max())
    o = m[(m.detector == "logreg:output_trajectory") & m.negative.isin(["restore", "confidence_boost"]) & m.prefix.isin(["128", "256", "512", "1024"])]
    add("P3", "main", "Model outputs: AUROC $\\ge0.90$ against restoration and confidence raising",
        f"{fmt(o.auroc.min(), 2)}--{fmt(o.auroc.max(), 2)} at 128 to 1{{,}}021 queries", o.auroc.min() >= 0.90)
    t = m[(m.detector == "logreg:output_trajectory") & m.negative.isin(["boundary_probe", "counterfactual"]) & (m.prefix == "attack_truncated")]
    add("P4", "main", "Model outputs: AUROC $\\le0.75$ against Tier~C before the attack's first label change",
        f"{fmt(t.auroc.min(), 2)}--{fmt(t.auroc.max(), 2)}", t.auroc.max() <= 0.75)

    ctl = pd.read_csv(A / f"stateful_specificity_controls_20260925/controls_summary{src['suffix']}.csv").set_index("objective")
    rate = lambda obj, col: float(ctl.loc[obj, col].split("=")[1].split("[")[0])  # noqa: E731
    frac = lambda obj, col: ctl.loc[obj, col].split("=")[0].strip()  # noqa: E731
    k, n = (int(v) for v in frac("shuffled", "blacklight_alarm").split("/"))
    add("P5a", "controls", "Blacklight raises an alarm in $\\le1\\%$ of the shuffled sessions", f"{pc(k / n, 2)} ({k}/{n})", k / n <= 0.01)
    add("P5a", "controls", "GWAD and GWAD+ raise no alarm in shuffled sessions",
        f"{frac('shuffled', 'gwad_alarm')} and {frac('shuffled', 'gwad_plus_alarm')}",
        rate("shuffled", "gwad_alarm") == 0 and rate("shuffled", "gwad_plus_alarm") == 0)
    add("P5b", "controls", "GWAD+ raises an alarm in $\\ge90\\%$ of the Gaussian-noise sessions",
        f"{pc(rate('noise', 'gwad_plus_alarm'))} ({frac('noise', 'gwad_plus_alarm')})", rate("noise", "gwad_plus_alarm") >= 0.90)
    add("P5c", "controls", "Blacklight raises an alarm in $\\ge90\\%$ of the sweep sessions",
        f"{pc(rate('sweep', 'blacklight_alarm'), 1)} ({frac('sweep', 'blacklight_alarm')})", rate("sweep", "blacklight_alarm") >= 0.90)
    lb = max(rate("shuffled", "ljung_box_h20"), rate("noise", "ljung_box_h20"))
    add("P5d", "controls", "Ljung--Box test alone (20 lags, whole session) flags $\\le5\\%$ of shuffled and of noise sessions",
        f"{pc(rate('shuffled', 'ljung_box_h20'), 1)} and {pc(rate('noise', 'ljung_box_h20'), 1)}", lb <= 0.05)

    for pid, label, directory in (("P6", "explanation, 32\\,px", "explanation_clients_20260926"),
                                  ("P13", "explanation, 224\\,px", "explanation_clients_imagenet_20260926")):
        e = pd.read_csv(A / directory / f"explanation_summary{src['suffix']}.csv").set_index("client")
        bl = {k: float(e.loc[k, "blacklight"].split(" ")[0]) for k in e.index}
        first = {k: e.loc[k, "blacklight_median_first"] for k in e.index}
        three = ["lime", "kernelshap", "occlusion"]
        if pid == "P6":
            ok = all(bl[k] >= 0.90 and first[k] <= 50 for k in three)
            add("P6a", label, "Blacklight raises an alarm in $\\ge90\\%$ of the LIME, KernelSHAP and occlusion sessions, median first alarm $\\le50$",
                f"{pc(min(bl[k] for k in three))}; median first alarm {fmt(min(first[k] for k in three))}--{fmt(max(first[k] for k in three))}", ok)
            add("P6b", label, "Blacklight raises an alarm in $\\le10\\%$ of the RISE sessions", pc(bl["rise"], 1), bl["rise"] <= 0.10)
        else:
            add("P13a", label, "Blacklight raises an alarm in $\\ge90\\%$ of the LIME, KernelSHAP and occlusion sessions",
                pc(min(bl[k] for k in three)), all(bl[k] >= 0.90 for k in three))
        lows = {k: float(e.loc[k, "util_diff"].split("[")[1].split(",")[0]) for k in e.index}
        meds = {k: float(e.loc[k, "util_diff"].split(" ")[0]) for k in e.index}
        add("P6c" if pid == "P6" else "P13b", label,
            "Every client beats spatially smooth random orderings (amended rule: median difference $>0$, interval excludes 0)",
            f"medians {fmt(min(meds.values()), 3)}--{fmt(max(meds.values()), 3)}; smallest lower bound {fmt(min(lows.values()), 3)}",
            all(v > 0 for v in lows.values()) and all(v > 0 for v in meds.values()))

    lfc = pd.read_csv(A / "lfc_workloads_20260926/analysis/lfc_alarm_rates.csv")
    g = lambda objs: lfc[lfc.objective.isin(objs)].alarm_primary  # noqa: E731
    add("P7a", "main", "Lee et al.\\ (test on every update, whole session) flags $\\ge90\\%$ of restoration and confidence-raising sessions",
        f"{pc(g(['restore', 'confidence_boost']).min(), 1)}--{pc(g(['restore', 'confidence_boost']).max(), 1)}",
        g(["restore", "confidence_boost"]).min() >= 0.90)
    add("P7b", "main", "The same schedule flags $\\ge95\\%$ of attack sessions", pc(g(["attack"]).min(), 1), g(["attack"]).min() >= 0.95)
    add("P7c", "controls", "The same schedule flags $\\le2\\%$ of shuffled and noise sessions",
        pc(g(["shuffled", "noise"]).max(), 1), g(["shuffled", "noise"]).max() <= 0.02)
    add("P7d", "main", "The same schedule flags $\\ge90\\%$ of random walks", pc(g(["random_walk"]).min(), 1), g(["random_walk"]).min() >= 0.90)

    for label, directory in LATER + [("ImageNet", "specificity_imagenet_20260926")]:
        s = sep(directory)
        tier_b = ["confidence_boost"] if label == "ImageNet" else ["restore", "confidence_boost"]
        pid = "P12" if label == "ImageNet" else ""
        q = cells(s, QUERY_ONLY_FAMILIES, tier_b, "1024")
        worst = q.loc[q.auroc.idxmax()]
        by = {d.split(":")[1]: q[q.detector == d].auroc.max() for d in QUERY_ONLY_FAMILIES}
        add(pid or "P8", label, "Query-only detectors: AUROC $\\le0.62$ against Tier~B",
            f"maximum {fmt(worst.auroc, 3)} (GWAD+ {fmt(by['gwad_plus'], 2)}, Blacklight {fmt(by['blacklight'], 2)}, all {fmt(by['query_only_all'], 2)})",
            q.auroc.max() <= 0.62)
        w = cells(s, QUERY_ONLY_FAMILIES, ["random_walk"], "1024")
        if not w.empty:
            add(pid or "P8", label, "Query-only detectors: AUROC $\\ge0.90$ against the random walk", f"minimum {fmt(w.auroc.min(), 2)}", w.auroc.min() >= 0.90)
        p = cells(s, ["logreg:output_trajectory"], ["boundary_probe", "counterfactual"], "pre_either_flip")
        add(pid or "P9", label, "Model outputs: AUROC $\\le0.70$ against Tier~C before either label change",
            f"maximum {fmt(p.auroc.max(), 3)}", p.auroc.max() <= 0.70)
        b = cells(s, ["logreg:output_trajectory"], tier_b, "1024")
        add(pid or "P10", label, "Model outputs: AUROC $\\ge0.90$ against Tier~B", f"minimum {fmt(b.auroc.min(), 2)}", b.auroc.min() >= 0.90)
        rates = pd.read_csv(A / directory / src["analysis"] / "alarm_rates.csv")
        lf = rates[rates.objective.isin(tier_b)].lfc
        add(pid or "P11", label, "Lee et al.\\ (test on every update, whole session) flags $\\ge90\\%$ of Tier-B sessions, per start",
            f"minimum {pc(lf.min(), 1)}", lf.min() >= 0.90)

    thr = json.loads((A / src["leak"] / "throttled_predictions.json").read_text())
    a = thr["P14a_gwad_plus_throttled_vs_restore_le_0.65"]
    add("P14a", "throttled SimBA", "GWAD+ AUROC $\\le0.65$, throttled attack against restoration",
        " and ".join(fmt(v[0], 2) for v in a.values()), all(v[1] for v in a.values()))
    b = thr["P14b_throttled_success_ge_0.50"]
    add("P14b", "throttled SimBA", "The throttled attack succeeds in $\\ge50\\%$ of the sessions",
        " and ".join(pc(v[0], 1) for v in b.values()), all(v[1] for v in b.values()))

    sens = json.loads((A / "lfc_sensitivity_20260927/sensitivity_predictions.json").read_text())
    faithful = {k: v for k, v in sens.items() if v["reproduces_paper"]}
    add("P15", "sensitivity", "Every variant of the Lee et al.\\ detector that reproduces the paper's rates flags $\\ge90\\%$ of Tier-B sessions (whole session)",
        f"{len(faithful)} of {len(sens)} variants reproduce them; minimum {pc(min(v['min_benign_optimizer_alarm'] for v in faithful.values()))}",
        all(v["P15_holds"] for v in faithful.values()))
    return rows


def later_experiments() -> list[dict]:
    """P16 (X13) and P17 (X14): run after the correction, so there is one analysis only."""
    rows = []
    p = A / "specificity_nes_sensitivity_20260927/analysis/predictions.json"
    if p.exists():
        c = json.loads(p.read_text())
        a = c["P16a_alarm_ge_99pct"]
        rows.append({"id": "P16a", "corpus": "NES variants", "prediction": "Blacklight and GWAD+ raise an alarm in $\\ge99\\%$ of the sessions of every client",
                     "outcome": f"minimum {pc(min(a['minimum_blacklight'], a['minimum_gwad_plus']))}", "held": a["held"]})
        b = c["P16b_blacklight_le_0.62"]
        rows.append({"id": "P16b", "corpus": "NES variants", "prediction": "Blacklight: AUROC $\\le0.62$ against every valid Tier-B client",
                     "outcome": f"maximum {fmt(b['maximum'], 3)} ({b['cells']} cells)", "held": b["held"]})
        d = c["P16c_gwad_plus_minus_acceptance_le_0.05"]
        rows.append({"id": "P16c", "corpus": "NES variants", "prediction": "With an acceptance test: AUROC of GWAD+ exceeds that of the acceptance rate alone by $\\le0.05$",
                     "outcome": f"largest difference {fmt(d['maximum_difference'], 3).replace('-', '$-$')} ({d['cells']} cells)", "held": d["held"]})
        e = c["P16d_no_acceptance_test_gwad_plus_le_0.62"]
        rows.append({"id": "P16d", "corpus": "NES variants", "prediction": "Without an acceptance test: GWAD+ AUROC $\\le0.62$",
                     "outcome": f"maximum {fmt(e['maximum'], 3)} ({e['cells']} cells)", "held": e["held"]})
        f = c["P16e_gwad_plus_vs_restore_rises_by_0.10"]
        if f.get("held") is not None:
            gains = " and ".join("+" + fmt(v, 2) for v in f["gain_by_start"].values())
            where = "step 1/255 (restoration invalid at 2/255)" if f["variant"] == "v1_step1" else "step 2/255"
            rows.append({"id": "P16e", "corpus": "NES variants",
                         "prediction": "With a larger step the GWAD+ AUROC against restoration rises by $\\ge0.10$ on at least one start",
                         "outcome": f"{gains} at {where}", "held": f["held"]})
    for label, directory in (("libraries, 32\\,px", "cifar10"), ("libraries, 224\\,px", "imagenet")):
        p = A / "explanation_libraries_20260927" / directory / "explanation_summary_r2.csv"
        if not p.exists():
            continue
        e = pd.read_csv(p).set_index("client")
        if e.n.min() < (200 if directory == "cifar10" else 100):
            continue
        rate = {k: float(e.loc[k, "blacklight"].split(" ")[0]) for k in e.index}
        first = {k: float(e.loc[k, "blacklight_median_first"]) for k in e.index}
        rows.append({"id": "P17a", "corpus": label, "prediction": "Blacklight raises an alarm in $\\ge90\\%$ of the sessions of each library client, median first alarm $\\le50$",
                     "outcome": f"{pc(min(rate.values()))}; median first alarm {fmt(min(first.values()))}--{fmt(max(first.values()))}",
                     "held": min(rate.values()) >= 0.90 and max(first.values()) <= 50})
        flagged = e.blacklight_flagged_queries
        rows.append({"id": "P17b", "corpus": label, "prediction": "Blacklight flags $\\ge90\\%$ of the queries of each library client",
                     "outcome": f"{pc(flagged.min(), 1)}--{pc(flagged.max(), 1)}", "held": flagged.min() >= 0.90})
        if directory == "imagenet":
            gp = float(e.loc["captum_occlusion", "gwad_plus"].split(" ")[0])
            rows.append({"id": "P17c", "corpus": label, "prediction": "GWAD+ raises an alarm in $\\ge90\\%$ of the Captum occlusion sessions",
                         "outcome": pc(gp, 1), "held": gp >= 0.90})
        lows = {k: float(e.loc[k, "util_diff"].split("[")[1].split(",")[0]) for k in e.index}
        meds = {k: float(e.loc[k, "util_diff"].split(" ")[0]) for k in e.index}
        held = all(v > 0 for v in lows.values()) and all(v > 0 for v in meds.values())
        note = ""
        if directory == "cifar10":  # the package returns one superpixel for most images: its explanations are empty
            note = "; not evaluable for the \\texttt{lime} package, whose explanations are empty (Table~\\ref{tab:libraries})"
            held = None
        rows.append({"id": "P17d", "corpus": label, "prediction": "Every library client beats spatially smooth random orderings",
                     "outcome": f"medians {fmt(min(meds.values()), 3)}--{fmt(max(meds.values()), 3)}; smallest lower bound {fmt(min(lows.values()), 3)}" + note,
                     "held": held})
    names = {"lime_package": "LIME", "captum_kernelshap": "KernelSHAP", "captum_occlusion": "occlusion"}
    p = A / "explanation_validity_20260927/libraries_imagenet/summary.csv"
    if p.exists():
        s = pd.read_csv(p).set_index(["client", "variant"])
        rec = {c: s.loc[(c, "recorded")] for c in names}
        h1 = {c: s.loc[(c, "h1")] for c in names}
        rows.append({"id": "P18a", "corpus": "libraries, 224\\,px",
                     "prediction": "Without rejection $\\ge95\\%$ of the explanations of each library client have ten or more distinct values, and the client is informative by rule R2",
                     "outcome": f"{pc(min(r.distinct_ge_10 for r in rec.values()))}; medians of $d$ {fmt(min(r.d_median for r in rec.values()), 3)}--"
                                f"{fmt(max(r.d_median for r in rec.values()), 3)}, smallest lower bound {fmt(min(r.d_lo for r in rec.values()), 3)}",
                     "held": all(r.distinct_ge_10 >= 0.95 and bool(r.informative) for r in rec.values())})
        rows.append({"id": "P18b", "corpus": "libraries, 224\\,px",
                     "prediction": "With rejection (uniform substitution): median rank correlation with the explanation without rejection $<0.3$ for each client",
                     "outcome": ", ".join(f"{names[c]} {fmt(h1[c].spearman_median, 2)}" for c in names),
                     "held": all(h1[c].spearman_median < 0.3 for c in names)})
        rows.append({"id": "P18c", "corpus": "libraries, 224\\,px",
                     "prediction": "With rejection (uniform substitution): median of $d$ below one quarter of the median without rejection, for each client",
                     "outcome": "share retained: " + ", ".join(f"{names[c]} {fmt(h1[c].d_median / rec[c].d_median, 2)}" for c in names),
                     "held": all(h1[c].d_median < 0.25 * rec[c].d_median for c in names)})
    p = A / "explanation_validity_20260927/libraries_cifar10/summary.csv"
    if p.exists():
        r = pd.read_csv(p).set_index(["client", "variant"]).loc[("lime_package", "recorded")]
        rows.append({"id": "P18d", "corpus": "libraries, 32\\,px",
                     "prediction": "The \\texttt{lime} package is not informative by rule R2",
                     "outcome": f"median of $d$ {fmt(r.d_median, 3)}, interval {fmt(r.d_lo, 3)} to {fmt(r.d_hi, 3)}",
                     "held": not bool(r.informative)})
    return rows


def main():
    first = pd.DataFrame(evaluate("first"))
    final = pd.DataFrame(evaluate("r2"))
    assert list(first.id) == list(final.id) and list(first.corpus) == list(final.corpus)
    final["first_outcome"], final["first_held"] = first.outcome, first.held
    # the text in parentheses lists the detectors separately; the bracket is shown if the leading quantity or the verdict differs
    lead = lambda s: s.split(" (GWAD+")[0]  # noqa: E731
    final["changed"] = (final.outcome.map(lead) != final.first_outcome.map(lead)) | (final.held != final.first_held)
    later = pd.DataFrame(later_experiments())
    if len(later):
        later["first_outcome"], later["first_held"], later["changed"] = later.outcome, later.held, False
    frame = pd.concat([final, later], ignore_index=True)
    frame.to_csv(N / "preregistration_outcomes.csv", index=False)

    def verdict(held):
        if held is None or pd.isna(held):
            return "in part"
        return "held" if held else "\\textbf{failed}"

    lines = []
    for r in frame.itertuples():
        outcome = r.outcome
        if r.changed:
            outcome += f" [first analysis: {lead(r.first_outcome)}" + (f", {verdict(r.first_held)}" if r.first_held != r.held else "") + "]"
        lines.append(f"{r.id} & {r.corpus} & {r.prediction} & {outcome} & {verdict(r.held)} \\\\")
    head = "ID & Corpus & Prediction & Outcome & Verdict \\\\\n\\midrule"
    text = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{L{0.9cm}L{1.9cm}L{5.0cm}L{3.4cm}L{1.1cm}}\n"
            "\\caption{Predictions that were written down in advance, and their outcomes. Evaluation split, except P5 (all splits of the "
            "control streams). ``Main'' is the CIFAR-10 ResNet-18 NES corpus. P8--P11 were written for each later corpus; P12 applies "
            "them to ImageNet, where restoration is excluded as an invalid workload. Outcomes are those of the corrected analysis "
            "(Blacklight's published rule); where the first analysis gave a different outcome or verdict, it is given in brackets. "
            "P16, P17 and P18 were run after the correction. The rule of P6c, P13b and P17d does not establish that an explanation is informative; P18 uses the rule that replaced it (Section~\\ref{app:validity}).}\\label{tab:prereg}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n"
            + "\\multicolumn{5}{l}{\\emph{Table~\\thetable\\ (continued)}}\\\\\n\\toprule\n" + head + "\n\\endhead\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg.tex").write_text(text)
    decided = frame[frame.held.notna()]
    decided = decided.assign(held=decided.held.astype(bool))
    counts = {"checks": len(frame), "held": int(decided.held.sum()), "failed": int((~decided.held).sum()),
              "evaluable_in_part": int(frame.held.isna().sum()),
              "checks_P1_P15": len(final), "failed_P1_P15": int((~final.held).sum()),
              "failed_P1_P15_first_analysis": int((~first.held).sum()),
              "verdict_changed": final[final.held != final.first_held][["id", "corpus"]].values.tolist(),
              "failed_ids": decided[~decided.held][["id", "corpus"]].values.tolist()}
    (N / "preregistration_counts.json").write_text(json.dumps(counts, indent=2))
    (N / "prereg_counts.tex").write_text("% generated by make_prereg_table.py\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in (
            ("PredChecks", counts["checks"]), ("PredHeld", counts["held"]), ("PredFailed", counts["failed"]),
            ("PredPart", counts["evaluable_in_part"]))))
    print(json.dumps(counts, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(frame[["id", "corpus", "outcome", "held", "changed"]].to_string(index=False))


if __name__ == "__main__":
    main()
