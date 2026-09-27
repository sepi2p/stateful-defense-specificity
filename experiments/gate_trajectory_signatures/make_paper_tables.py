#!/usr/bin/env python3
"""Generate the manuscript's data tables (LaTeX) directly from the analysis outputs.

No number in these tables is typed by hand: every cell is read from the CSV/JSON files written by the
analysis scripts (run make_paper_numbers.py first). Ranges are min-max over both degraded starts and,
where a column pools objectives, over the pooled objectives. Output: paper/jisa_2026/tables/*.tex
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
N = ROOT / "paper/jisa_2026/numbers"
OUT = ROOT / "paper/jisa_2026/tables"

# key, label, separability directory, model label, accuracy key, optimizer label, invalid objectives
CORPORA = [
    ("cifar_nes_seed0", "CIFAR-10", "stateful_specificity_workloads_20260925", "ResNet-18 (seed 0)", "cifar10/resnet18_seed0", "NES", ()),
    ("cifar_nes_seed1", "CIFAR-10", "specificity_resnet18_seed1_20260926", "ResNet-18 (seed 1)", "cifar10/resnet18_seed1", "NES", ()),
    ("cifar_nes_seed2", "CIFAR-10", "specificity_resnet18_seed2_20260926", "ResNet-18 (seed 2)", "cifar10/resnet18_seed2", "NES", ()),
    ("cifar_nes_vgg19bn", "CIFAR-10", "specificity_vgg19bn_20260926", "VGG19-BN", "cifar10/bbb_vgg19_bn", "NES", ()),
    ("cifar_nes_robust", "CIFAR-10", "specificity_robust_engstrom_20260926", "robust ResNet-50", "cifar10/robustbench_Engstrom2019Robustness", "NES", ()),
    ("gtsrb_nes", "GTSRB", "specificity_gtsrb32_20260926", "ResNet-18", "gtsrb32/resnet18", "NES", ()),
    ("imagenet_nes", "ImageNet", "specificity_imagenet_20260926", "ResNet-50", "imagenet/resnet50_v1_full_val", "tiled NES", ("restore",)),
    ("cifar_simba", "CIFAR-10", "specificity_simba_20260926", "ResNet-18 (seed 0)", "cifar10/resnet18_seed0", "SimBA", ()),
]
OBJECTIVE_NAMES = {"attack": "Attack", "restore": "Restoration", "confidence_boost": "Confidence raising",
                   "boundary_probe": "Boundary probing", "counterfactual": "Counterfactual search", "random_walk": "Random walk",
                   "shuffled": "Shuffled test images", "noise": "Gaussian noise ($\\sigma=0.1$)", "sweep": "JPEG/brightness sweep",
                   "lime": "LIME", "kernelshap": "KernelSHAP", "occlusion": "Occlusion", "rise": "RISE"}


def corpus_label(c, short=False):
    _key, data, _d, model, _acc, opt, _inv = c
    return f"{data}, {model}, {opt}" if not short else f"{data} {model} ({opt})"


def rng(values, digits=2, scale=1.0):
    values = [v * scale for v in values if pd.notna(v)]
    if not values:
        return "--"
    lo, hi = min(values), max(values)
    a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    if float(a) == 0:
        a = a.lstrip("-")
    if float(b) == 0:
        b = b.lstrip("-")
    if a.startswith("-") or b.startswith("-"):  # negative values: math minus and a worded range
        m = lambda v: f"$-${v[1:]}" if v.startswith("-") else v  # noqa: E731
        return m(a) if a == b else f"{m(a)} to {m(b)}"
    return a if a == b else f"{a}--{b}"


def pct(values, digits=0):
    return rng(values if isinstance(values, (list, tuple, np.ndarray, pd.Series)) else [values], digits, 100.0)


def wrap(name, body, caption, label, spec, star=False, resize=True, size="\\small"):
    env = "table*" if star else "table"
    inner = f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    if resize:
        inner = f"\\resizebox{{\\textwidth}}{{!}}{{{inner}}}"
    text = f"\\begin{{{env}}}[!tbp]\n\\centering{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n{inner}\n\\end{{{env}}}\n"
    (OUT / f"{name}.tex").write_text(text)
    print(f"wrote tables/{name}.tex")


def rows_tex(rows):
    return "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)


def sep_cells(sep, negatives, detector, prefix):
    return sep[sep.negative.isin(negatives) & (sep.detector == detector) & (sep.prefix == prefix)].auroc.tolist()


# ------------------------------------------------------------------------------------------ main text

def table_corpora():
    inv = pd.read_csv(N / "corpus_inventory.csv").set_index("corpus")
    util = pd.read_csv(N / "workload_utility_all.csv")
    acc = json.loads((N / "model_accuracies.json").read_text())
    rows = []
    for c in CORPORA:
        key, data, _d, model, acc_key, opt, _inv = c
        i = inv.loc[key]
        att = util[(util.corpus == key) & (util.objective == "attack")]
        res = "224" if data == "ImageNet" else "32"
        rows.append([data, f"{model} ({100 * acc[acc_key][0]:.1f})", res, opt,
                     f"{i.fit_images}/{i.calibration_images}/{i.eval_images}", f"{i.sessions:,}".replace(",", "{,}"),
                     pct(att.flipped_frac.tolist()), rng(att.median_first_flip.tolist(), 0)])
    head = ("Dataset & Model (clean accuracy, \\%) & Pixels & Optimizer & Images fit/cal./eval. & Sessions & "
            "Attack success (\\%) & Median queries to success \\\\\n\\midrule")
    wrap("corpora", head + "\n" + rows_tex(rows),
         "Corpora. Each source image contributes two degraded starts and one session per client and start. "
         "Attack success is the fraction of evaluation sessions in which some query is classified differently from the "
         "source label within the budget; the last column is the median query index of the first such query among "
         "successful sessions. Ranges are over the two starts. The ImageNet corpus uses the fit and evaluation images only.",
         "tab:corpora", "llrlrrrr", star=True)


def table_matched():
    rows = []
    for c in CORPORA:
        _key, _data, directory, _model, _acc, _opt, invalid = c
        sep = pd.read_csv(A / directory / "analysis/separability.csv")
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        tier_c = ["boundary_probe", "counterfactual"]
        rows.append([corpus_label(c),
                     rng(sep_cells(sep, tier_b, "logreg:gwad_plus", "1024")),
                     rng(sep_cells(sep, tier_b, "logreg:blacklight", "1024")),
                     rng(sep_cells(sep, ["random_walk"], "logreg:gwad_plus", "1024")),
                     rng(sep_cells(sep, ["random_walk"], "logreg:blacklight", "1024")),
                     rng(sep_cells(sep, tier_b, "logreg:output_trajectory", "1024")),
                     rng(sep_cells(sep, tier_c, "logreg:output_trajectory", "pre_either_flip"))])
    head = ("& \\multicolumn{2}{c}{Attack vs Tier B} & \\multicolumn{2}{c}{Attack vs random walk} & "
            "\\multicolumn{2}{c}{Model outputs} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Corpus & GWAD+ & Blacklight & GWAD+ & Blacklight & vs Tier B & vs Tier C, pre-flip \\\\\n\\midrule")
    wrap("matched", head + "\n" + rows_tex(rows),
         "Matched-objective specificity (evaluation split; AUROC at the full session length unless noted). "
         "Tier B: restoration and confidence raising (ImageNet: confidence raising only, because restoration makes no "
         "progress at $224\\times224$ pixels, Section~\\ref{sec:threats}). The random walk was generated with the NES corpora only. "
         "Model outputs: logistic model on the classifier's own outputs, against Tier B and against attack-equivalent clients "
         "(Tier C) on the queries that precede the first label change of either session. Ranges are over both starts and the pooled objectives.",
         "tab:matched", "lcccccc", star=True)


def table_operating():
    t = pd.read_csv(A / "stateful_specificity_operating_points_20260926/operating_points.csv")
    names = {"blacklight": "Blacklight", "gwad": "GWAD", "gwad_plus": "GWAD+"}
    # Each detector is recalibrated on the control that is informative for it: shuffled streams never produce a
    # GWAD/GWAD+ decision window, and noise streams have Blacklight match counts below those of shuffled streams.
    informative = {"blacklight": "cal_shuffled", "gwad": "cal_noise", "gwad_plus": "cal_noise"}
    rows = []
    for det, det_name in names.items():
        ref = informative[det]
        ref_name = ref.removeprefix("cal_")
        settings = [("native", np.nan, "released"), (ref, 0.01, f"{ref_name}, 1\\%"), (ref, 0.001, f"{ref_name}, 0.1\\%")]
        for cal, target, label in settings:
            g = t[(t.detector == det) & (t.calibration == cal) & (t.target_fpr.isna() if pd.isna(target) else np.isclose(t.target_fpr, target))]
            cell = lambda obj: pct(g[g.objective == obj].blocked.tolist(), 1)  # noqa: E731
            med = lambda obj: rng(g[g.objective == obj].median_first_alarm.tolist(), 0)  # noqa: E731
            rows.append([det_name if cal == "native" else "", label, cell("attack"), cell("restore"), cell("confidence_boost"),
                         cell("shuffled"), cell("noise"), cell("sweep"), med("attack"), med("restore")])
        rows[-1][-1] += " \\\\[2pt]" if det != "gwad_plus" else ""
    body = "\n".join(" & ".join(r) + (r" \\" if not r[-1].endswith("[2pt]") else "") for r in rows)
    head = ("& & \\multicolumn{3}{c}{Optimizing clients blocked (\\%)} & \\multicolumn{3}{c}{Controls blocked (\\%)} & "
            "\\multicolumn{2}{c}{Median first alarm} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\n"
            "Detector & Threshold & Attack & Restoration & Conf.\\ raising & Shuffled & Noise & Sweep & Attack & Restoration \\\\\n\\midrule")
    wrap("operating", head + "\n" + body,
         "Operating points on the main corpus and its controls (evaluation split). ``Released'' is the detector's published "
         "decision rule; the other rows recalibrate the session-level threshold on the calibration split of a control so that "
         "at most the stated fraction of its sessions is blocked. Blacklight is recalibrated on shuffled test images. GWAD and "
         "GWAD+ never produce a decision window on shuffled images, so they are recalibrated on the Gaussian-noise control. "
         "Ranges are over the two starts.",
         "tab:operating", "llcccccccc", star=True)


def controls_frame():
    root = A / "lfc_controls_20260926"
    rows = []
    for p in sorted(root.glob("sessions_shard*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            t = np.load(root / s["trace"])
            rows.append({"control": s["objective"], "blacklight": s["detectors"]["blacklight"]["first_alarm"] > 0,
                         "gwad_plus": s["detectors"]["gwad_plus"]["first_alarm"] > 0, "gwad": s["detectors"]["gwad"]["first_alarm"] > 0,
                         "lfc_seq50": phase2_batch(t["lfc_assignment"], t["logits"], 50),
                         "lfc_online": phase2_alarm(t["lfc_assignment"], t["logits"]) > 0})
    return pd.DataFrame(rows)


def table_controls():
    f = controls_frame()
    f.groupby("control").mean().to_csv(N / "controls_table.csv")
    rows = []
    for control in ("shuffled", "noise", "sweep"):
        g = f[f.control == control]
        rows.append([OBJECTIVE_NAMES[control], f"{len(g):,}".replace(",", "{,}")] +
                    [pct(g[c].mean(), 1) for c in ("blacklight", "gwad", "gwad_plus", "lfc_seq50", "lfc_online")])
    head = ("& & & & & \\multicolumn{2}{c}{Lee et al.} \\\\\n\\cmidrule(lr){6-7}\n"
            "Control stream & Sessions & Blacklight & GWAD & GWAD+ & sequence-50 & online \\\\\n\\midrule")
    wrap("controls", head + "\n" + rows_tex(rows),
         "Non-optimizing control streams on CIFAR-10: sessions alarmed (\\%) at released operating points, all splits. "
         "Each stream has 1{,}024 queries.",
         "tab:controls", "lrccccc")


def table_explanations():
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for res, directory, key in (("32", "explanation_clients_20260926", "cifar_explain"), ("224", "explanation_clients_imagenet_20260926", "imagenet_explain")):
        summary = pd.read_csv(A / directory / "explanation_summary.csv").set_index("client")
        for client in ("lime", "kernelshap", "occlusion", "rise"):
            r = summary.loc[client]

            def cell(value, first):
                rate = float(value.split(" ")[0])
                return f"{100 * rate:.1f}".rstrip("0").rstrip(".") + (f" ({first:.0f})" if pd.notna(first) and rate > 0 else "")
            rows.append([res if client == "lime" else "", OBJECTIVE_NAMES[client], f"{int(r.queries):,}".replace(",", "{,}"),
                         cell(r.blacklight, r.blacklight_median_first), cell(r.gwad_plus, r.gwad_plus_median_first),
                         pct(sched.loc[(key, client), "seq50"], 1), cell(r.lfc, r.lfc_median_first),
                         r.util_diff.replace("-0.000", "0.000").replace("-", "$-$").replace("[", "{[}").replace("]", "{]}")])
        rows[-1][-1] += " \\\\[2pt]" if res == "32" else ""
    body = "\n".join(" & ".join(r) + (r" \\" if not r[-1].endswith("[2pt]") else "") for r in rows)
    head = ("& & & & & \\multicolumn{2}{c}{Lee et al.} & \\\\\n\\cmidrule(lr){6-7}\n"
            "Pixels & Client & Queries & Blacklight & GWAD+ & sequence-50 & online & Utility \\\\\n\\midrule")
    wrap("explanations", head + "\n" + body,
         "Explanation clients: sessions blocked (\\%) at released operating points, 200 evaluation images per row; the median "
         "query index of the first alarm is given in parentheses. Utility is the median reduction in deletion area relative to a "
         "spatially smooth random ordering, with its bootstrap 95\\% interval.",
         "tab:explanations", "rlrccccc", star=True)


def table_lfc():
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        r = [corpus_label(c)]
        for schedule in ("seq50", "online"):
            for obj in ("attack", "restore", "confidence_boost", "random_walk"):
                if obj in invalid or (key, obj) not in sched.index:
                    r.append("--")
                else:
                    r.append(pct(sched.loc[(key, obj), schedule], 1))
        rows.append(r)
    head = ("& \\multicolumn{4}{c}{Sequence-50 (the paper's protocol)} & \\multicolumn{4}{c}{Online re-testing} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Corpus & Attack & Restor. & Conf. & Walk & Attack & Restor. & Conf. & Walk \\\\\n\\midrule")
    wrap("lfc", head + "\n" + rows_tex(rows),
         "The reimplemented detector of Lee et al.\\ on matched clients: evaluation sessions flagged (\\%), both starts pooled. "
         "Sequence-50 applies the paper's evaluation protocol to the first 50 queries of a session; online re-testing repeats "
         "the test whenever a subsequence grows. Restor.: restoration; Conf.: confidence raising; Walk: objective-free random "
         "walk. Dashes: workload not valid or not generated for that corpus.",
         "tab:lfc", "lcccccccc", star=True)


def table_leak():
    cells = pd.read_csv(A / "acceptance_leak_20260927/acceptance_vs_detectors.csv")
    thr = pd.read_csv(A / "acceptance_leak_20260927/throttled_attack.csv")
    fam = {"nes": "NES (CIFAR-10, GTSRB; 6 corpora)", "nes_tiled": "Tiled NES (ImageNet)", "simba": "SimBA (CIFAR-10)"}
    tiers = {"Tier B": ["restore", "confidence_boost"], "Tier C": ["boundary_probe", "counterfactual"]}
    rows = []
    for key, label in fam.items():
        for tier, objs in tiers.items():
            g = cells[(cells.optimizer == key) & cells.objective.isin(objs)]
            rows.append([label if tier == "Tier B" else "", tier, rng(g.accept_attack_median.tolist()), rng(g.accept_benign_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()), rng(g.blacklight_auroc.tolist())])
    rows[-1][-1] += " \\\\[2pt]"
    for positive, label in (("unthrottled attack", "SimBA attack"), ("throttled attack", "SimBA attack, throttled")):
        for negative, neg_label in (("restore", "vs restoration"), ("confidence_boost", "vs conf.\\ raising")):
            g = thr[(thr.positive == positive) & (thr.negative == negative)]
            rows.append([label if negative == "restore" else "", neg_label, rng(g.accept_pos_median.tolist()), rng(g.accept_neg_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()), rng(g.blacklight_auroc.tolist())])
    body = "\n".join(" & ".join(r) + (r" \\" if not r[-1].endswith("[2pt]") else "") for r in rows)
    head = ("& & \\multicolumn{2}{c}{Median acceptance rate} & \\multicolumn{3}{c}{AUROC, attack vs benign client} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
            "Optimizer & Benign clients & Attack & Benign & Acceptance rate alone & GWAD+ & Blacklight \\\\\n\\midrule")
    wrap("leak", head + "\n" + body,
         "The acceptance-rate leak. Upper block (post hoc): per optimizer family, the fraction of iterations in which the "
         "client's current image moved, and the AUROC for separating attack from benign sessions using that fraction alone, "
         "GWAD+, or Blacklight. Lower block (preregistered experiment): the SimBA attack before and after its acceptance rate "
         "is throttled to that of the restoration client. Ranges are over starts and the pooled clients.",
         "tab:leak", "llccccc", star=True)


# ------------------------------------------------------------------------------------------ appendix

def table_effects():
    util = pd.read_csv(N / "workload_utility_all.csv")
    rows = []
    for c in CORPORA:
        key = c[0]
        start_p = rng(util[(util.corpus == key) & (util.objective == "attack")].p_label_start_median.tolist())
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{corpus_label(c)}}} ($p$ at the start: {start_p})}}"])
        for obj in ("attack", "restore", "confidence_boost", "boundary_probe", "counterfactual", "random_walk"):
            g = util[(util.corpus == key) & (util.objective == obj)]
            if g.empty:
                continue
            rows.append(["\\quad " + OBJECTIVE_NAMES[obj], pct(g.flipped_frac.tolist()),
                         rng(g.p_label_final_median.tolist()), rng(g.margin_final_median.tolist(), 1),
                         rng(g.psnr_gain_median.tolist()), rng(g.accept_median.tolist())])
    head = "Client & Flip (\\%) & $p$ end & Margin end & PSNR gain (dB) & Accepted \\\\\n\\midrule"
    text = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n"
            "\\begin{longtable}{@{\\extracolsep{\\fill}}lccccc@{}}\n"
            "\\caption{Effect of each optimizing client on the model and on the image (evaluation split; medians; ranges over the two "
            "starts). Flip: sessions in which some submitted query is classified differently from the source label. "
            "End: after the last query. "
            "$p$: probability of the source label (its median at the start image is given with the corpus). PSNR gain is measured "
            "against the clean source image. Accepted: fraction of "
            "iterations in which the current image moved.}\\label{tab:effects}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n"
            "\\toprule\n" + head + "\n\\endhead\n" + rows_tex(rows) + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    (OUT / "effects.tex").write_text(text)
    print("wrote tables/effects.tex")


def table_full():
    rows = []
    for c in CORPORA:
        _key, _data, directory, _model, _acc, _opt, invalid = c
        sep = pd.read_csv(A / directory / "analysis/separability.csv")
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{corpus_label(c)}}}}}"])
        for start in ("denoise", "deblur"):
            for obj in ("restore", "confidence_boost", "boundary_probe", "counterfactual", "random_walk"):
                g = sep[(sep.workload == start) & (sep.negative == obj) & (sep.prefix == "1024")]
                if g.empty:
                    continue

                def cell(det):
                    r = g[g.detector == det]
                    if r.empty or pd.isna(r.auroc.iloc[0]):
                        return "--"
                    r = r.iloc[0]
                    return f"{r.auroc:.2f} {{[}}{r.lo:.2f}, {r.hi:.2f}{{]}}"
                name = OBJECTIVE_NAMES[obj] + ("$^{\\ast}$" if obj in invalid else "")
                rows.append(["\\quad " + {"denoise": "noisy", "deblur": "blurred"}[start], name,
                             cell("logreg:gwad_plus"), cell("logreg:blacklight"), cell("logreg:query_only_all"), cell("logreg:output_trajectory")])
    head = "Start & Client & GWAD+ & Blacklight & All query-only & Model outputs \\\\\n\\midrule"
    text = ("\\begin{landscape}\n\\begin{footnotesize}\n\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n"
            "\\begin{longtable}{@{\\extracolsep{\\fill}}llcccc@{}}\n"
            "\\caption{AUROC for separating the attack from each client at the full session length, evaluation split, with bootstrap "
            "95\\% intervals. $^{\\ast}$Restoration makes no progress on ImageNet (no accepted step); its sessions are trivially "
            "separable and are excluded from all claims.}\\label{tab:full}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n"
            "\\toprule\n" + head + "\n\\endhead\n" + rows_tex(rows) + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n\\end{landscape}\n")
    (OUT / "full.tex").write_text(text)
    print("wrote tables/full.tex")


def table_explanation_utility():
    u = pd.read_csv(N / "explanation_utility.csv")
    rows = []
    for key, res, directory in (("cifar_explain", "32", "explanation_clients_20260926"), ("imagenet_explain", "224", "explanation_clients_imagenet_20260926")):
        summary = pd.read_csv(A / directory / "explanation_summary.csv").set_index("client")
        for client in ("lime", "kernelshap", "occlusion", "rise"):
            r = u[(u.corpus == key) & (u.client == client)].iloc[0]
            fix = lambda s: s.replace("-0.000", "0.000").replace("-", "$-$").replace("[", "{[}").replace("]", "{]}")  # noqa: E731
            rows.append([res if client == "lime" else "", OBJECTIVE_NAMES[client], f"{r.del_auc_client_median:.2f}", f"{r.del_auc_random_median:.2f}",
                         fix(summary.loc[client, "util_diff"]), f"{100 * r.better_than_random_frac:.0f}", fix(summary.loc[client, "util_diff_pixel_blur"])])
    head = ("& & \\multicolumn{4}{c}{Primary metric} & Sensitivity metric \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-7}\n"
            "Pixels & Client & Client & Random & Difference & Better (\\%) & Difference \\\\\n\\midrule")
    wrap("explanation_utility", head + "\n" + rows_tex(rows),
         "Utility of the explanation clients on the evaluation images. Primary metric: deletion area with per-channel mean fill; "
         "``Random'' is the mean over ten spatially smooth random orderings; ``Difference'' is the median paired difference "
         "(random minus client) with its bootstrap 95\\% interval; ``Better'' is the share of images on which the client's area is "
         "smaller. Sensitivity metric: blurred fill and random pixel orderings, the metric originally planned.",
         "tab:explutility", "rlccccc", star=True)


def table_sensitivity():
    path = A / "lfc_sensitivity_20260927/sensitivity_summary.csv"
    if not path.exists():
        print("sensitivity summary not available yet")
        return
    s = pd.read_csv(path).set_index(["stream", "variant"])
    variants = [("frozen", "Frozen configuration"), ("salt_unit", "Salt in $[0,1)$"), ("literal_fixed", "Literal rounding, fixed thresholds"),
                ("literal_fresh", "Literal rounding, redrawn per query")]
    streams = ["simba", "square", "shuffled", "noise", "noise_small", "sweep", "attack", "restore", "confidence_boost"]
    rows = []
    for schedule, name in (("seq50_flag", "Sequence-50"), ("alarm", "Online")):
        for v, label in variants:
            rows.append([name if v == "frozen" else "", label] + [pct(s.loc[(st, v), schedule], 1) if (st, v) in s.index else "--" for st in streams])
    head = ("& & \\multicolumn{2}{c}{Attacks of the paper} & \\multicolumn{4}{c}{Benign and control streams} & "
            "\\multicolumn{3}{c}{Matched NES sessions} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}\n"
            "Schedule & Phase-1 variant & SimBA & Square & Shuffled & Noise & Small noise & Sweep & Attack & Restor. & Conf. \\\\\n\\midrule")
    wrap("sensitivity", head + "\n" + rows_tex(rows),
         "Sensitivity of the reimplemented detector of Lee et al.\\ to choices the paper leaves open: sessions flagged (\\%). "
         "Streams of the first six columns: 50 calibration images; matched NES sessions: 50 evaluation images, both starts. "
         "Noise: $\\sigma=0.1$ on the $[0,1]$ scale; small noise: $\\sigma=0.1$ intensity levels on the 0--255 scale.",
         "tab:sensitivity", "llccccccccc", star=True)


def table_decision():
    rows = []
    for tag, label in (("", "All benign traffic"), ("_no_sweep_walk", "Without sweep and walk")):
        r = pd.read_csv(A / f"decision_time_20260927/decision_time_rates{tag}.csv")
        piv = r.pivot_table(index="objective", columns="prefix", values="alarm_rate")
        first = True
        for obj in ("attack", "counterfactual", "boundary_probe", "restore", "confidence_boost", "lime", "kernelshap", "occlusion", "rise",
                    "shuffled", "noise", "sweep", "random_walk"):
            rows.append([label if first else "", OBJECTIVE_NAMES[obj]] + [pct(piv.loc[obj, p], 1) for p in (128, 256, 512, 768, 1021)])
            first = False
        rows[-1][-1] += " \\\\[2pt]" if tag == "" else ""
    body = "\n".join(" & ".join(r) + (r" \\" if not r[-1].endswith("[2pt]") else "") for r in rows)
    head = "Calibration pool & Client & 128 & 256 & 512 & 768 & 1{,}021 \\\\\n\\midrule"
    wrap("decision", head + "\n" + body,
         "Exploratory two-stage output diagnostic: evaluation sessions flagged (\\%) after the stated number of queries, with the "
         "threshold set for a pooled false-positive rate of 1\\% on the calibration split of the stated pool.",
         "tab:decision", "llccccc")


def table_output_stats():
    d = pd.read_csv(A / "output_diagnostic_20260926/diagnostic_auroc.csv")
    piv = d.pivot_table(index="benign", columns=["unit", "prefix"], values="auroc")
    rows = []
    for obj in ("restore", "confidence_boost", "lime", "kernelshap", "occlusion", "rise", "shuffled", "noise", "sweep", "random_walk",
                "boundary_probe", "counterfactual"):
        rows.append([OBJECTIVE_NAMES[obj]] + [f"{piv.loc[obj, (u, p)]:.2f}" for u in ("dd_group", "tr_group") for p in ("early", "full")])
    head = ("& \\multicolumn{2}{c}{Margin drawdown} & \\multicolumn{2}{c}{Margin trend} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Attack vs & 256 queries & full & 256 queries & full \\\\\n\\midrule")
    wrap("outputstats", head + "\n" + rows_tex(rows),
         "AUROC of two single output statistics for separating attack sessions from each client (evaluation split; fixed "
         "orientation, larger means more attack-like; nothing is fitted). Both are computed within the largest group of similar "
         "queries. Drawdown was planned in advance; trend was introduced after the drawdown result and is exploratory.",
         "tab:outputstats", "lcccc", resize=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (table_corpora, table_matched, table_operating, table_controls, table_explanations, table_lfc, table_leak,
               table_effects, table_full, table_explanation_utility, table_sensitivity, table_decision, table_output_stats):
        fn()


if __name__ == "__main__":
    main()
