#!/usr/bin/env python3
"""Generate the paper's results tables (LaTeX) directly from the analysis outputs.

No number in these tables is typed by hand: each cell is read from the CSV/JSON files the
analysis scripts wrote. Ranges are min-max over both corrupted starts (and over the listed
objectives where a column pools objectives). Output: paper/jisa_2026/tables/*.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis_outputs"
OUT = ROOT / "paper/jisa_2026/tables"

# (label, separability dir, alarm-rates source, objectives excluded as invalid workloads)
CORPORA = [
    ("CIFAR-10 ResNet-18 (seed 0)", "stateful_specificity_workloads_20260925", ("lfc", "lfc_workloads_20260926"), ()),
    ("CIFAR-10 ResNet-18 (seed 1)", "specificity_resnet18_seed1_20260926", ("score", None), ()),
    ("CIFAR-10 ResNet-18 (seed 2)", "specificity_resnet18_seed2_20260926", ("score", None), ()),
    ("CIFAR-10 VGG19-BN", "specificity_vgg19bn_20260926", ("score", None), ()),
    ("CIFAR-10 robust ResNet-50", "specificity_robust_engstrom_20260926", ("score", None), ()),
    ("GTSRB ResNet-18", "specificity_gtsrb32_20260926", ("score", None), ()),
    ("ImageNet ResNet-50", "specificity_imagenet_20260926", ("score", None), ("restore",)),
    ("CIFAR-10 ResNet-18, SimBA", "specificity_simba_20260926", ("score", None), ()),
]


def rng(values, digits=2):
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "--"
    lo, hi = min(values), max(values)
    return f"{lo:.{digits}f}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f}--{hi:.{digits}f}"


def auroc(sep, negatives, detector, prefix):
    cell = sep[sep.negative.isin(negatives) & (sep.detector == detector) & (sep.prefix == prefix)]
    return cell.auroc.tolist()


def lfc_rates(kind, source_dir, corpus_dir, objectives):
    if kind == "lfc":
        table = pd.read_csv(A / source_dir / "analysis/lfc_alarm_rates.csv")
        return table[table.objective.isin(objectives)].alarm_primary.tolist()
    table = pd.read_csv(A / corpus_dir / "analysis/alarm_rates.csv")
    return table[table.objective.isin(objectives)].lfc.tolist()


def matched_table():
    rows = []
    for label, directory, (kind, src), excluded in CORPORA:
        sep = pd.read_csv(A / directory / "analysis/separability.csv")
        tier_b = [o for o in ("restore", "confidence_boost") if o not in excluded]
        tier_c = ["boundary_probe", "counterfactual"]
        rows.append([
            label,
            rng(auroc(sep, tier_b, "logreg:gwad_plus", "1024")),
            rng(auroc(sep, tier_b, "logreg:blacklight", "1024")),
            rng(auroc(sep, ["random_walk"], "logreg:query_only_all", "1024")),
            rng(auroc(sep, tier_b, "logreg:output_trajectory", "1024")),
            rng(auroc(sep, tier_c, "logreg:output_trajectory", "pre_either_flip")),
            rng(lfc_rates(kind, src, directory, tier_b)),
        ])
    head = (r"& \multicolumn{3}{c}{Query-only detectors (AUROC)} & \multicolumn{2}{c}{Model outputs (AUROC)} & Lee et al. \\"
            "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
            r"Corpus & GWAD+ vs B & Blacklight vs B & vs drift & vs B & vs C, pre-flip & blocks B \\")
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        "\\begin{table*}[t]\n\\centering\\small\n"
        "\\caption{Matched-objective specificity across corpora (evaluation split, 1,024 queries unless noted). "
        "Columns 2--3: AUROC separating the attack from matched benign optimizers (Tier B: restoration and "
        "confidence raising; ImageNet: confidence raising only, restoration makes no progress at 224\\,px). "
        "``vs drift'': query-only AUROC against the objective-free random walk. Model outputs: logistic model on the "
        "classifier's own outputs, against Tier B and against attack-equivalent clients (Tier C) before either session changes the label. "
        "Last column: fraction of Tier-B sessions blocked by the reimplemented Lee et al.\\ detector. Ranges are over both corrupted starts and the pooled objectives.}\n"
        "\\label{tab:matched}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\n"
        + head + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def explanation_table():
    rows = []
    for res, directory in (("32 px", "explanation_clients_20260926"), ("224 px", "explanation_clients_imagenet_20260926")):
        summary = pd.read_csv(A / directory / "explanation_summary.csv")
        for r in summary.itertuples(index=False):
            def cell(value, first):
                rate = value.split(" ")[0]
                return f"{float(rate):.2f}" + (f" ({first:.0f})" if pd.notna(first) and float(rate) > 0 else "")
            rows.append([res, {"lime": "LIME", "kernelshap": "KernelSHAP", "occlusion": "Occlusion", "rise": "RISE"}[r.client],
                         str(r.queries), cell(r.blacklight, r.blacklight_median_first), cell(r.gwad_plus, r.gwad_plus_median_first),
                         cell(r.lfc, r.lfc_median_first), r.util_diff.replace("[", "[").split(" ")[0]])
    head = r"Resolution & Client & Queries & Blacklight & GWAD+ & Lee et al. & Utility \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption{Explanation clients: fraction of sessions blocked at released operating points "
        "(median first-alarm query in parentheses), 200 evaluation images per cell. Utility: median "
        "reduction in deletion AUC relative to a smooth random ordering (all bootstrap 95\\% CIs exclude 0).}\n"
        "\\label{tab:explanations}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrcccc}\n\\toprule\n" + head + "\n\\midrule\n" + body
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def controls_table():
    summary = pd.read_csv(A / "stateful_specificity_controls_20260925/controls_summary.csv")
    names = {"shuffled": "Shuffled test images", "noise": "i.i.d. $\\mathcal{N}(0,0.1^2)$ noise", "sweep": "JPEG/brightness sweep"}
    rows = []
    for r in summary.itertuples(index=False):
        def rate(text):
            return text.split("=")[1].split("[")[0].strip()
        rows.append([names[r.objective], str(r.n), rate(r.blacklight_alarm), rate(r.gwad_plus_alarm), rate(r.ljung_box_h20)])
    faith = pd.read_csv(A / "lfc_faithfulness_20260926/faithfulness_summary.csv")
    f = faith[(faith.bern_scale == 0.0) & (faith.lags == "min10") & (faith.retest == "every")].set_index("stream").alarm_rate
    head = r"Control stream & Sessions & Blacklight & GWAD+ & Lee et al.\ Phase 2 \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    note = (f"Faithfulness of the Lee et al.\\ reimplementation (50 images): SimBA {f['simba']:.2f}, Square {f['square']:.2f}, "
            f"shuffled {f['shuffled']:.2f}, noise {f['noise']:.2f} (paper: TPR 1.00, FPR 0.00).")
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption{Non-optimizing controls on CIFAR-10 (fraction of sessions alarmed at released operating points). "
        + note + "}\n\\label{tab:controls}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrccc}\n\\toprule\n" + head + "\n\\midrule\n" + body
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("matched", matched_table), ("explanations", explanation_table), ("controls", controls_table)):
        (OUT / f"{name}.tex").write_text(fn())
        print(f"wrote tables/{name}.tex")
    print((OUT / "matched.tex").read_text())


if __name__ == "__main__":
    main()
