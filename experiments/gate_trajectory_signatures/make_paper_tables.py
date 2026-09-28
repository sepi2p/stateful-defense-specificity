#!/usr/bin/env python3
"""Generate the manuscript's data tables (LaTeX) from the analysis outputs.

No number in these tables is typed by hand: every cell is read from the files written by the analysis
scripts (run make_lfc_schedules.py and make_paper_numbers.py first). Revision 2: corrected Blacklight
rule (analysis_r2 and the other r2 outputs), rounding half up, schedules of the Lee et al. detector
as described in its paper, operating points on all corpora, X13 and X14.
Output: paper/jisa_2026/tables/*.tex
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A = ROOT / "analysis_outputs"
N = ROOT / "paper/jisa_2026/numbers"
OUT = ROOT / "paper/jisa_2026/tables"
R2 = "analysis_r2"

# key, dataset, directory, model, accuracy key, optimizer, invalid objectives, short label
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
                   "lime": "LIME", "kernelshap": "KernelSHAP", "occlusion": "Occlusion", "rise": "RISE",
                   "lime_package": "LIME (\\texttt{lime} package)", "captum_kernelshap": "KernelSHAP (Captum)",
                   "captum_occlusion": "Occlusion (Captum)"}


def fmt(value, digits=0):
    """Format with rounding half up. Python's own formatting rounds binary floats half to even, which prints
    78.5 as 78 and would make tables disagree with numbers quoted in the text."""
    text = str(Decimal(f"{float(value):.9f}").quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return text.lstrip("-") if float(text) == 0 else text


def corpus_label(c):
    _key, data, _d, model, _acc, opt, _inv = c
    return f"{data}, {model}, {opt}"


def rng(values, digits=2, scale=1.0):
    values = [v * scale for v in values if pd.notna(v)]
    if not values:
        return "--"
    a, b = fmt(min(values), digits), fmt(max(values), digits)
    if a.startswith("-") or b.startswith("-"):  # negative values: math minus and a worded range
        m = lambda v: f"$-${v[1:]}" if v.startswith("-") else v  # noqa: E731
        return m(a) if a == b else f"{m(a)} to {m(b)}"
    return a if a == b else f"{a}--{b}"


def med(values):
    """Range of medians: one decimal only where a median is not an integer."""
    return rng(values, 1).replace(".0", "")


def pct(values, digits=0):
    return rng(values if isinstance(values, (list, tuple, np.ndarray, pd.Series)) else [values], digits, 100.0)


def interval(text):
    return text.replace("-0.000", "0.000").replace("-", "$-$").replace("[", "{[}").replace("]", "{]}")


def wrap(name, body, caption, label, spec, size="\\footnotesize", colsep="4pt", sideways=False):
    inner = f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    env, place, width = ("sidewaystable", "[p]", "\\textheight") if sideways else ("table", "[!tbp]", "\\textwidth")
    text = (f"\\begin{{{env}}}{place}\n\\centering{size}\\setlength{{\\tabcolsep}}{{{colsep}}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\vspace{{3pt}}\n"
            f"\\begin{{adjustbox}}{{max width={width}}}\n{inner}\n\\end{{adjustbox}}\n\\end{{{env}}}\n")
    (OUT / f"{name}.tex").write_text(text)
    print(f"wrote tables/{name}.tex")


def rows_tex(rows):
    """One LaTeX row per list; a cell that already ends with a row break (e.g. with extra space) closes its row."""
    lines = []
    for r in rows:
        line = " & ".join(str(c) for c in r)
        lines.append(line if line.endswith("pt]") else line + r" \\")
    return "\n".join(lines)


def sep_cells(sep, negatives, detector, prefix):
    return sep[sep.negative.isin(negatives) & (sep.detector == detector) & (sep.prefix == prefix)].auroc.tolist()


def separability(directory):
    return pd.read_csv(A / directory / R2 / "separability.csv")


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
        rows.append([data, model, fmt(100 * acc[acc_key][0], 1), "224" if data == "ImageNet" else "32", opt,
                     f"{i.fit_images}/{i.calibration_images}/{i.eval_images}", f"{i.sessions:,}".replace(",", "{,}"),
                     pct(att.flipped_frac.tolist(), 1), med(att.median_first_flip.tolist())])
    head = ("Dataset & Model & Acc.\\ (\\%) & Pixels & Optimizer & Images & Sessions & Success (\\%) & Queries \\\\\n\\midrule")
    wrap("corpora", head + "\n" + rows_tex(rows),
         "Corpora. Each source image contributes two degraded starts and one session per client and start. Acc.: clean test "
         "accuracy (ImageNet: 73.5\\% on the 100 classes from which the source images are drawn). Images: source images in the "
         "fit, calibration and evaluation splits. Success: evaluation sessions of the attack in which some query is classified "
         "differently from the source label within the budget. Queries: median index of the first such query among successful "
         "sessions. Ranges are over the two starts.",
         "tab:corpora", "llrrllrrr")


def table_controls():
    c = pd.read_csv(N / "controls.csv")
    c = c[c.scope == "evaluation"].set_index("control")
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for control in ("shuffled", "noise", "sweep"):
        r = c.loc[control]
        s = sched.loc[("cifar_controls", control)]
        rows.append([OBJECTIVE_NAMES[control], f"{int(r.sessions):,}".replace(",", "{,}"), pct(r.blacklight_alarm, 1),
                     pct(r.blacklight_flagged_queries, 1), pct(r.gwad_alarm, 1), pct(r.gwad_plus_alarm, 1),
                     pct(s.update50, 1), pct(s.online, 1)])
    head = ("& & \\multicolumn{2}{c}{Blacklight} & & & \\multicolumn{2}{c}{Lee et al.} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){7-8}\n"
            "Control stream & Sessions & sessions & queries & GWAD & GWAD+ & first 50 & whole session \\\\\n\\midrule")
    wrap("controls", head + "\n" + rows_tex(rows),
         "Non-optimizing control streams on CIFAR-10 (evaluation split, 1{,}024 queries per stream): sessions in which the "
         "detector raises an alarm (\\%) at its released operating point and, for Blacklight, the share of queries that it flags. "
         "Lee et al.: the reimplemented detector with the test applied at every update of a group, within the first 50 queries "
         "and over the whole session.",
         "tab:controls", "lrcccccc")


def table_matched():
    rows = []
    for c in CORPORA:
        _key, _data, directory, _model, _acc, _opt, invalid = c
        sep = separability(directory)
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        tier_c = ["boundary_probe", "counterfactual"]
        rows.append([corpus_label(c),
                     rng(sep_cells(sep, tier_b, "logreg:gwad", "1024")),
                     rng(sep_cells(sep, tier_b, "logreg:gwad_plus", "1024")),
                     rng(sep_cells(sep, tier_b, "logreg:blacklight", "1024")),
                     rng(sep_cells(sep, ["random_walk"], "logreg:gwad_plus", "1024")),
                     rng(sep_cells(sep, ["random_walk"], "logreg:blacklight", "1024")),
                     rng(sep_cells(sep, tier_b, "logreg:output_trajectory", "1024")),
                     rng(sep_cells(sep, tier_c, "logreg:output_trajectory", "pre_either_flip"))])
    head = ("& \\multicolumn{3}{c}{Attack vs Tier B} & \\multicolumn{2}{c}{Attack vs random walk} & "
            "\\multicolumn{2}{c}{Model outputs} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Corpus & GWAD & GWAD+ & Blacklight & GWAD+ & Blacklight & vs Tier B & vs Tier C \\\\\n\\midrule")
    wrap("matched", head + "\n" + rows_tex(rows),
         "Matched-objective separability: AUROC of logistic models on each detector's session statistics (fitted on the fit "
         "split, evaluation split shown, full session). Tier B: restoration and confidence raising (ImageNet: confidence raising "
         "only, Section~\\ref{sec:threats}). The random walk was generated with the NES corpora only. Model outputs: logistic model "
         "on the classifier's own outputs; against Tier~C it uses only the queries that precede the first label change of either "
         "session. Ranges are over the two starts and the clients of the tier.",
         "tab:matched", "lccccccc")


def table_operating():
    t = pd.read_csv(A / "stateful_specificity_operating_points_r2/operating_points.csv")
    cal = pd.read_csv(A / "stateful_specificity_operating_points_r2/calibration.csv")
    settings = [("blacklight", "Blacklight", [("native", None, "released ($T=25$)"), ("cal_sweep", 0.01, "sweep, $\\le1$ of 100"),
                                              ("cal_sweep", 0.001, "sweep, none of 100")]),
                ("gwad", "GWAD", [("native", None, "released"), ("cal_shuffled", 0.01, "shuffled, $\\le1\\%$"),
                                  ("cal_noise", 0.01, "noise, $\\le1$ of 100"), ("cal_noise", 0.001, "noise, none of 100")]),
                ("gwad_plus", "GWAD+", [("native", None, "released"), ("cal_noise", 0.01, "noise, $\\le1$ of 100"),
                                        ("cal_noise", 0.001, "noise, none of 100")])]
    rows = []
    for det, det_name, options in settings:
        for k, (calibration, target, label) in enumerate(options):
            g = t[(t.detector == det) & (t.calibration == calibration) & (t.target_fpr.isna() if target is None else np.isclose(t.target_fpr, target))]
            cell = lambda obj: pct(g[g.objective == obj].blocked.tolist(), 1)  # noqa: E731
            med_ = lambda obj: med(g[g.objective == obj].median_first_alarm.tolist())  # noqa: E731
            if det == "blacklight" and calibration != "native":
                thr = cal[(cal.detector == det) & (cal.reference == "sweep") & np.isclose(cal.target_fpr, target)].threshold.iloc[0]
                label += f" ($T={int(thr)}$)"
            last = k == len(options) - 1 and det != "gwad_plus"
            rows.append([det_name if k == 0 else "", label, cell("attack"), cell("restore"), cell("confidence_boost"),
                         cell("shuffled"), cell("noise"), cell("sweep"), med_("attack"), med_("restore"),
                         med_("confidence_boost") + (" \\\\[2pt]" if last else "")])
    head = ("& & \\multicolumn{3}{c}{Optimizing clients (\\%)} & \\multicolumn{3}{c}{Controls (\\%)} & "
            "\\multicolumn{3}{c}{Median first alarm} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
            "Detector & Threshold & Attack & Restor. & Conf. & Shuffled & Noise & Sweep & Attack & Restor. & Conf. \\\\\n\\midrule")
    wrap("operating", head + "\n" + rows_tex(rows),
         "Operating points on the main corpus and its controls (evaluation split): sessions with an alarm (\\%) and median "
         "query index of the first alarm. ``Released'' is the published decision rule. The other rows set the session-level "
         "threshold on the calibration split of the named control so that at most the stated number of its sessions raise an "
         "alarm; only controls on which the released rule raises alarms give a threshold that is less sensitive than the "
         "released one. Calibrated on the sweep, GWAD and GWAD+ raise no alarm on any stream, because the released network "
         "gives the sweeps its largest score; these rows are omitted. Restor.: restoration; Conf.: confidence raising. "
         "Ranges are over the two starts.",
         "tab:operating", "llccccccccc", colsep="5pt", size="\\small", sideways=True)


def table_released():
    util = pd.read_csv(N / "workload_utility_all.csv")
    before = pd.read_csv(N / "success_before_alarm.csv")
    before = before[before.start == "pooled"].set_index("corpus")
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        a = util[(util.corpus == key) & (util.objective == "attack")]
        b = util[(util.corpus == key) & util.objective.isin(tier_b)]
        assert min(a.gwad_plus_alarm.min(), b.gwad_plus_alarm.min(), a.gwad_alarm.min(), b.gwad_alarm.min()) == 1.0  # stated in the caption
        assert set(a.gwad_plus_first_median) | set(b.gwad_plus_first_median) == {259.0}
        s = before.loc[key]
        rows.append([corpus_label(c),
                     pct(a.blacklight_alarm.tolist(), 1), pct(b.blacklight_alarm.tolist(), 1),
                     med(a.blacklight_first_median.tolist()), med(b.blacklight_first_median.tolist()),
                     pct(a.blacklight_flagged_queries.tolist(), 1), pct(b.blacklight_flagged_queries.tolist(), 1),
                     pct(s.blacklight_success_before_alarm_frac, 1), pct(s.gwad_plus_success_before_alarm_frac, 1)])
    head = ("& \\multicolumn{6}{c}{Blacklight} & \\multicolumn{2}{c}{Attack succeeds} \\\\\n"
            "& \\multicolumn{2}{c}{sessions (\\%)} & \\multicolumn{2}{c}{first alarm} & \\multicolumn{2}{c}{queries (\\%)} & "
            "\\multicolumn{2}{c}{before alarm (\\%)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Corpus & attack & Tier B & attack & Tier B & attack & Tier B & Blacklight & GWAD+ \\\\\n\\midrule")
    wrap("released", head + "\n" + rows_tex(rows),
         "Released operating points on every corpus (evaluation split). GWAD and GWAD+ raise an alarm in every session of the "
         "attack and of the Tier-B clients in every corpus, GWAD+ at query 259; the table therefore shows Blacklight. "
         "Sessions: sessions in which Blacklight raises an alarm. First alarm: median query index of its first alarm. "
         "Queries: share of queries that it flags, which it would reject. Attack succeeds before alarm: successful attack "
         "sessions whose first misclassified query precedes the first alarm of the detector, or that raise no alarm, as a share "
         "of the successful attack sessions. Tier B as in Table~\\ref{tab:main-matched}. Ranges are over the two starts and the "
         "clients of the tier.",
         "tab:released", "lcccccccc", colsep="4pt")


def explanation_rows(files, clients, sched_keys, numbers):
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for k, (res, summary_path, key) in enumerate(files):
        summary = pd.read_csv(summary_path).set_index("client")
        for j, client in enumerate(clients):
            r = summary.loc[client]
            n = numbers[(numbers.corpus == sched_keys[k]) & (numbers.client == client)].iloc[0] if numbers is not None else None

            def cell(value, first):
                rate = float(value.split(" ")[0])
                return fmt(100 * rate, 1).rstrip("0").rstrip(".") + (f" ({fmt(first)})" if pd.notna(first) and rate > 0 else "")
            s = sched.loc[(key, client)]
            last = j == len(clients) - 1 and k < len(files) - 1
            rows.append([res if j == 0 else "", OBJECTIVE_NAMES[client], f"{int(r.queries):,}".replace(",", "{,}"),
                         cell(r.blacklight, r.blacklight_median_first), pct(r.blacklight_flagged_queries, 1),
                         "n/a" if int(r.queries) < 259 else cell(r.gwad_plus, r.gwad_plus_median_first),
                         pct(s.update50, 1), pct(s.online, 1), interval(r.util_diff) + (" \\\\[2pt]" if last else "")])
    return rows


EXPLANATION_HEAD = ("& & & \\multicolumn{2}{c}{Blacklight} & & \\multicolumn{2}{c}{Lee et al.} & \\\\\n"
                    "\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}\n"
                    "Pixels & Client & Queries & sessions & queries & GWAD+ & first 50 & whole session & Deletion \\\\\n\\midrule")
EXPLANATION_NOTE = ("sessions with an alarm (\\%), with the median query index of the first alarm in parentheses, and the share of "
                    "queries that Blacklight flags (\\%). Lee et al.: test at every update of a group, within the first 50 queries "
                    "and over the whole session (one test after 50 queries: Tables~\\ref{tab:lfcother} and~\\ref{tab:lfcall}). "
                    "Deletion: median reduction of the deletion area relative to spatially smooth random orderings, with its bootstrap "
                    "95\\% interval under the first rule, which an empty explanation also meets (Section~\\ref{app:workloads}).")


def table_explanations():
    files = [("32", A / "explanation_clients_20260926/explanation_summary_r2.csv", "cifar_explain"),
             ("224", A / "explanation_clients_imagenet_20260926/explanation_summary_r2.csv", "imagenet_explain")]
    rows = explanation_rows(files, ("lime", "kernelshap", "occlusion", "rise"), None, None)
    wrap("explanations", EXPLANATION_HEAD + "\n" + rows_tex(rows),
         "Explanation clients, our implementations of the four query designs (200 evaluation images per row): " + EXPLANATION_NOTE,
         "tab:explanations", "rlrcccccc")


def table_libraries():
    files = [("32", A / "explanation_libraries_20260927/cifar10/explanation_summary_r2.csv", "cifar_libraries"),
             ("224", A / "explanation_libraries_20260927/imagenet/explanation_summary_r2.csv", "imagenet_libraries")]
    files = [f for f in files if f[1].exists()]
    rows = explanation_rows(files, ("lime_package", "captum_kernelshap", "captum_occlusion"), None, None)
    wrap("libraries", EXPLANATION_HEAD + "\n" + rows_tex(rows),
         "Explanation clients as implemented by libraries (200 evaluation images at $32\\times32$ pixels, 100 at "
         "$224\\times224$): " + EXPLANATION_NOTE + " n/a: the stream is shorter than the 259 queries that GWAD+ needs "
         "before its first decision. At $32\\times32$ pixels the default segmentation of the \\texttt{lime} package returns a "
         "median of one superpixel, so that its samples are copies of the image and its explanation carries no information; "
         "the row shows what the package sends.",
         "tab:libraries", "rlrcccccc")


def table_lfc():
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        r = [corpus_label(c)]
        for schedule in ("single50", "update50", "online"):
            for obj in ("attack", "restore", "confidence_boost"):
                r.append("--" if obj in invalid or (key, obj) not in sched.index else pct(sched.loc[(key, obj), schedule], 1))
        rows.append(r)
    rows[-1][-1] += " \\\\[2pt]"
    for key, label in (("nes_v1_step1", "NES variant, step 1/255"), ("nes_v2_step2", "NES variant, step 2/255"),
                       ("nes_v3_step2_always", "NES variant, step 2/255, always moves")):
        r = [label]
        for schedule in ("single50", "update50", "online"):
            for obj in ("attack", "restore", "confidence_boost"):
                star = "$^{\\ast}$" if (key, obj) == ("nes_v2_step2", "restore") else ""
                r.append(pct(sched.loc[(key, obj), schedule], 1) + star)
        rows.append(r)
    head = ("& \\multicolumn{3}{c}{First 50, one test} & \\multicolumn{3}{c}{First 50, every update} & "
            "\\multicolumn{3}{c}{Whole session} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Corpus & Attack & Restor. & Conf. & Attack & Restor. & Conf. & Attack & Restor. & Conf. \\\\\n\\midrule")
    wrap("lfc", head + "\n" + rows_tex(rows),
         "The reimplemented detector of Lee et al.\\ on matched clients: evaluation sessions flagged (\\%), both starts pooled. "
         "First 50, one test: the first 50 queries are grouped and every group of at least 15 queries is tested once. "
         "Every update: the test is applied whenever a group of at least 15 queries gains a member, and alarms up to "
         "query 50 are counted. Whole session: the same without a limit on the number of queries. "
         "Restor.: restoration; Conf.: confidence raising. Dashes: workload not valid. Last three rows: the variants of "
         "Section~\\ref{sec:res-accept} (CIFAR-10, ResNet-18); $^{\\ast}$invalid workload. The objective-free random walk "
         "is flagged in 98.8--100\\% of the sessions under every schedule.",
         "tab:lfc", "lccccccccc", colsep="3pt")


def table_leak():
    cells = pd.read_csv(A / "acceptance_leak_r2/acceptance_vs_detectors.csv")
    thr = pd.read_csv(A / "acceptance_leak_r2/throttled_attack.csv")
    fam = {"nes": "NES (CIFAR-10, GTSRB; 6 corpora)", "nes_tiled": "Tiled NES (ImageNet)", "simba": "SimBA (CIFAR-10)"}
    tiers = {"Tier B": ["restore", "confidence_boost"], "Tier C": ["boundary_probe", "counterfactual"]}
    rows = []
    for key, label in fam.items():
        for tier, objs in tiers.items():
            g = cells[(cells.optimizer == key) & cells.objective.isin(objs)]
            rows.append([label if tier == "Tier B" else "", tier, rng(g.accept_attack_median.tolist()), rng(g.accept_benign_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()), rng(g.blacklight_auroc.tolist())])
    rows[-1][-1] += " \\\\[2pt]"
    names = {"restore": "restoration", "confidence_boost": "conf.\\ raising", "boundary_probe": "boundary probing",
             "counterfactual": "counterfactual"}
    for positive, label in (("unthrottled attack", "SimBA attack"), ("throttled attack", "SimBA attack, throttled")):
        for k, negative in enumerate(names):
            g = thr[(thr.positive == positive) & (thr.negative == negative)]
            last = k == len(names) - 1 and positive == "unthrottled attack"
            rows.append([label if k == 0 else "", "vs " + names[negative], rng(g.accept_pos_median.tolist()), rng(g.accept_neg_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()),
                         rng(g.blacklight_auroc.tolist()) + (" \\\\[2pt]" if last else "")])
    head = ("& & \\multicolumn{2}{c}{Median acceptance rate} & \\multicolumn{3}{c}{AUROC, attack vs benign client} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
            "Optimizer & Benign clients & Attack & Benign & Acceptance alone & GWAD+ & Blacklight \\\\\n\\midrule")
    wrap("leak", head + "\n" + rows_tex(rows),
         "Acceptance rates. Upper block (not planned in advance): per optimizer, the fraction of iterations in which the "
         "client's current image moved, and the AUROC for separating attack sessions from those of the matched client by that fraction "
         "alone and by the logistic models on the session statistics of GWAD+ and of Blacklight. Lower blocks: the SimBA attack before and after its acceptance rate is throttled to that of "
         "the restoration client; the comparison with restoration was planned in advance (P14), the other three were not. "
         "Ranges are over the starts and, in the upper block, the clients of the tier.",
         "tab:leak", "llccccc")


def table_nes():
    c = pd.read_csv(A / "specificity_nes_sensitivity_20260927/analysis/cells.csv")
    a = pd.read_csv(A / "specificity_nes_sensitivity_20260927/analysis/alarms.csv")
    names = {"main": ("0.25/255", "if better"), "v1_step1": ("1/255", "if better"), "v2_step2": ("2/255", "if better"),
             "v3_step2_always": ("2/255", "always")}
    rows = []
    for k, (variant, (step, rule)) in enumerate(names.items()):
        att = a[(a.variant == variant) & (a.client == "attack")]
        for j, objective in enumerate(("restore", "confidence_boost")):
            g = c[(c.variant == variant) & (c.benign == objective)]
            valid = bool(g.valid.all())
            star = "" if valid else "$^{\\ast}$"
            last = j == 1 and k < len(names) - 1
            rows.append([step if j == 0 else "", rule if j == 0 else "", pct(att.success_rate.tolist()) if j == 0 else "",
                         OBJECTIVE_NAMES[objective] + star, rng(g.attack_acceptance_median.tolist()), rng(g.benign_acceptance_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()),
                         rng(g.blacklight_auroc.tolist()) + (" \\\\[2pt]" if last else "")])
    head = ("& & & & \\multicolumn{2}{c}{Median acceptance} & \\multicolumn{4}{c}{AUROC, attack vs benign client} \\\\\n"
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}\n"
            "Step & Moves & Success (\\%) & Benign client & attack & benign & Acceptance alone & GWAD & GWAD+ & Blacklight \\\\\n\\midrule")
    wrap("nes", head + "\n" + rows_tex(rows),
         "Sensitivity of the matched-objective result to the NES configuration (CIFAR-10, ResNet-18, 200 evaluation images, "
         "full session). Step: size of the sign step; the first row is the configuration of the main corpus. Moves: whether "
         "the client moves to the stepped point only if its objective improves (and, for restoration, the label is kept) or "
         "always, as in NES as published (restoration then also drops its label check). Success: attack success within the budget. "
         "AUROC: acceptance rate alone, and logistic models on the session statistics of each detector, fitted on the fit split. At the released operating points Blacklight "
         "and GWAD+ raise an alarm in every session of every row. $^{\\ast}$Invalid workload: the client accepts no step "
         "(median acceptance rate on the fit split below 0.10; 76 of its 400 evaluation sessions accept at least one step). "
         "Each variant has 1{,}800 sessions (100 fit and 200 evaluation images, two starts, three clients). Ranges are over "
         "the two starts.",
         "tab:nes", "llclcccccc", colsep="3pt")


# ------------------------------------------------------------------------------------------ appendix

def table_lfc_all():
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        r = [corpus_label(c), pct(sched.loc[(key, "attack"), "tested50"], 1)]
        for schedule in ("online_cap50", "update50_lag5"):
            for obj in ("attack", "restore", "confidence_boost"):
                r.append("--" if obj in invalid or (key, obj) not in sched.index else pct(sched.loc[(key, obj), schedule], 1))
        r.append("--" if (key, "random_walk") not in sched.index else rng([sched.loc[(key, "random_walk"), s] for s in ("single50", "update50", "online")], 1, 100.0))
        rows.append(r)
    head = ("& & \\multicolumn{3}{c}{Whole session, groups capped at 50} & \\multicolumn{3}{c}{First 50, every update, 5 lags} & \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Corpus & Tested & Attack & Restor. & Conf. & Attack & Restor. & Conf. & Walk \\\\\n\\midrule")
    wrap("lfc_all", head + "\n" + rows_tex(rows),
         "The reimplemented detector of Lee et al.\\ under further settings: evaluation sessions flagged (\\%), both starts "
         "pooled. Tested: attack sessions in which at least 15 of the first 50 queries fall into one group, so that a test is "
         "applied at all. Capped at 50: the test is applied at every update to the 50 most recent members of a group. "
         "5 lags: the test uses five lags instead of $\\min(10,\\lfloor n/5\\rfloor)$. Walk: the objective-free random walk "
         "under the three schedules of Table~\\ref{tab:lfc}.",
         "tab:lfcall", "lcccccccc", colsep="3pt")
    labels = {"cifar_controls": "Controls, 32 px", "cifar_explain": "Explanation, 32 px", "imagenet_explain": "Explanation, 224 px"}
    clients = {"cifar_controls": ("shuffled", "noise", "sweep"), "cifar_explain": ("lime", "kernelshap", "occlusion", "rise"),
               "imagenet_explain": ("lime", "kernelshap", "occlusion", "rise")}
    other = []
    for key, label in labels.items():
        for obj in clients[key]:
            s = sched.loc[(key, obj)]
            other.append([label + ": " + OBJECTIVE_NAMES[obj].split(" (")[0], pct(s.tested50, 1), pct(s.single50, 1), pct(s.update50, 1),
                          pct(s.online, 1), pct(s.online_cap50, 1)])
    head2 = "Stream & Tested & First 50, one test & First 50, every update & Whole session & Whole session, capped \\\\\n\\midrule"
    wrap("lfc_other", head2 + "\n" + rows_tex(other),
         "The reimplemented detector of Lee et al.\\ on control streams and on our explanation clients: evaluation sessions "
         "flagged (\\%) under four schedules. Tested: sessions in which at least 15 of the first 50 queries fall into one group.",
         "tab:lfcother", "lccccc")


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
            rows.append(["\\quad " + OBJECTIVE_NAMES[obj], pct(g.flipped_frac.tolist(), 1),
                         rng(g.p_label_final_median.tolist()), rng(g.margin_final_median.tolist(), 1),
                         rng(g.psnr_gain_median.tolist()), rng(g.accept_median.tolist())])
    head = "Client & Flip (\\%) & $p$ end & Margin end & PSNR gain (dB) & Accepted \\\\\n\\midrule"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n"
            "\\begin{longtable}{@{\\extracolsep{\\fill}}lccccc@{}}\n"
            "\\caption{Effect of each optimizing client on the model and on the image (evaluation split; medians; ranges over the two "
            "starts). Flip: sessions in which some submitted query is classified differently from the source label. "
            "End: after the last query. "
            "$p$: probability of the source label (its median at the start image is given with the corpus). PSNR gain is measured "
            "against the clean source image. Accepted: fraction of "
            "iterations in which the current image moved.}\\label{tab:effects}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n"
            "\\multicolumn{6}{l}{\\emph{Table~\\thetable\\ (continued)}}\\\\\n\\toprule\n" + head + "\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    (OUT / "effects.tex").write_text(text)
    print("wrote tables/effects.tex")


def table_full():
    rows = []
    for c in CORPORA:
        _key, _data, directory, _model, _acc, _opt, invalid = c
        sep = separability(directory)
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\emph{{{corpus_label(c)}}}}}"])
        for start in ("denoise", "deblur"):
            for obj in ("restore", "confidence_boost", "boundary_probe", "counterfactual", "random_walk"):
                g = sep[(sep.workload == start) & (sep.negative == obj) & (sep.prefix == "1024")]
                if g.empty:
                    continue

                def cell(det, ci=True):
                    r = g[g.detector == det]
                    if r.empty or pd.isna(r.auroc.iloc[0]):
                        return "--"
                    r = r.iloc[0]
                    return f"{fmt(r.auroc, 2)} {{[}}{fmt(r.lo, 2)}, {fmt(r.hi, 2)}{{]}}" if ci else fmt(r.auroc, 2)
                name = OBJECTIVE_NAMES[obj] + ("$^{\\ast}$" if obj in invalid else "")
                rows.append(["\\quad " + {"denoise": "noisy", "deblur": "blurred"}[start], name,
                             cell("logreg:gwad_plus"), cell("logreg:blacklight"), cell("logreg:output_trajectory"),
                             cell("single:gwad_plus_max", False), cell("single:bl_max", False)])
    head = ("& & \\multicolumn{3}{c}{Logistic models} & \\multicolumn{2}{c}{Single statistic} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
            "Start & Client & GWAD+ & Blacklight & Model outputs & GWAD+ & Blacklight \\\\\n\\midrule")
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = ("\\begin{landscape}\n\\begin{footnotesize}\n\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n"
            "\\begin{longtable}{@{\\extracolsep{\\fill}}llccccc@{}}\n"
            "\\caption{AUROC for separating the attack from each client at the full session length, evaluation split. Logistic "
            "models with bootstrap 95\\% intervals; the logistic model on the statistics of GWAD differs from that of GWAD+ by at "
            "most 0.01 in every cell and is omitted. Single statistic: the largest score of GWAD+ and the largest match count of "
            "Blacklight over the session, oriented on the fit split. $^{\\ast}$Restoration accepts almost no step on ImageNet; "
            "its sessions are trivially separable and are excluded from all claims.}\\label{tab:full}\\\\\n\\toprule\n"
            + head + "\n\\endfirsthead\n\\multicolumn{7}{l}{\\emph{Table~\\thetable\\ (continued)}}\\\\\n\\toprule\n" + head
            + "\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n\\end{landscape}\n")
    (OUT / "full.tex").write_text(text)
    print("wrote tables/full.tex")


def table_explanation_utility():
    u = pd.read_csv(N / "explanation_clients.csv")
    rows = []
    for key, res, directory in (("cifar_explain", "32", "explanation_clients_20260926"), ("imagenet_explain", "224", "explanation_clients_imagenet_20260926")):
        summary = pd.read_csv(A / directory / "explanation_summary_r2.csv").set_index("client")
        for client in ("lime", "kernelshap", "occlusion", "rise"):
            r = u[(u.corpus == key) & (u.client == client)].iloc[0]
            rows.append([res if client == "lime" else "", OBJECTIVE_NAMES[client], fmt(r.del_auc_client_median, 2), fmt(r.del_auc_random_median, 2),
                         interval(summary.loc[client, "util_diff"]), fmt(100 * r.better_than_random_frac, 1),
                         interval(summary.loc[client, "util_diff_pixel_blur"]), fmt(100 * r.better_than_random_pixel_blur_frac, 1)])
    head = ("& & \\multicolumn{4}{c}{Rule used} & \\multicolumn{2}{c}{Rule planned first} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
            "Pixels & Client & Client & Random & Difference & Better (\\%) & Difference & Better (\\%) \\\\\n\\midrule")
    wrap("explanation_utility", head + "\n" + rows_tex(rows),
         "Deletion metric of the explanation clients on the evaluation images. Rule used: deletion area with per-channel mean fill; "
         "``Random'' is the mean over ten spatially smooth random orderings; ``Difference'' is the median paired difference "
         "(random minus client) with its bootstrap 95\\% interval (2{,}000 resamples); ``Better'' is the share of images on which the "
         "client's area is smaller. Rule planned first: blurred fill and random orderings of single pixels, with the requirement "
         "that the client be better on 90\\% of the images, which no client meets.",
         "tab:explutility", "rlcccccc")


def table_sensitivity():
    # the repetition of X12 with the added variant; identical to the first run in all other cells
    s = pd.read_csv(A / "lfc_sensitivity_r2/sensitivity_summary.csv").set_index(["stream", "variant"])
    variants = [("frozen", "Configuration used"), ("strict_threshold", "Grouping above the threshold"), ("salt_unit", "Salt in $[0,1)$"),
                ("literal_fixed", "Literal, fixed rounding"), ("literal_fresh", "Literal, rounding redrawn")]
    streams = ["simba", "square", "shuffled", "noise", "noise_small", "sweep", "attack", "restore", "confidence_boost"]
    rows = []
    for schedule, name in (("seq50_flag", "First 50, one test"), ("alarm_within_50", "First 50, every update"), ("alarm", "Whole session")):
        for k, (v, label) in enumerate(variants):
            last = k == len(variants) - 1 and schedule != "alarm"
            cells = [pct(s.loc[(st, v), schedule], 1) if (st, v) in s.index else "--" for st in streams]
            cells[-1] += " \\\\[2pt]" if last else ""
            rows.append([name if k == 0 else "", label] + cells)
    head = ("& & \\multicolumn{2}{c}{Attacks of the paper} & \\multicolumn{4}{c}{Benign and control streams} & "
            "\\multicolumn{3}{c}{Matched NES sessions} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}\n"
            "Schedule & Phase-1 variant & SimBA & Square & Shuffled & Noise & Small noise & Sweep & Attack & Restor. & Conf. \\\\\n\\midrule")
    wrap("sensitivity", head + "\n" + rows_tex(rows),
         "Sensitivity of the reimplemented detector of Lee et al.\\ to choices that the paper leaves open: sessions flagged (\\%). "
         "Streams of the first six columns: 50 calibration images; matched NES sessions: 50 evaluation images, both starts. "
         "Noise: $\\sigma=0.1$ on the $[0,1]$ scale; small noise: $\\sigma=0.1$ intensity levels on the 0--255 scale. "
         "The two literal variants use the salt in $[0,1)$ and the rounding rule of the paper, with rounding thresholds drawn "
         "once per position or redrawn for every query.",
         "tab:sensitivity", "llccccccccc", colsep="5pt", size="\\small", sideways=True)


def table_decision():
    rows = []
    summaries = {}
    options = (("", "All benign traffic"), ("_no_sweep_walk", "Without sweep and walk"))
    for k, (tag, label) in enumerate(options):
        r = pd.read_csv(A / f"decision_time_r2/sequential_rates{tag}.csv")
        summaries[label] = json.loads((A / f"decision_time_r2/summary{tag}.json").read_text())
        piv = r.pivot_table(index="objective", columns="prefix", values="flagged_by_prefix")
        objectives = ("attack", "counterfactual", "boundary_probe", "restore", "confidence_boost", "lime", "kernelshap", "occlusion", "rise",
                      "shuffled", "noise", "sweep", "random_walk")
        for j, obj in enumerate(objectives):
            last = j == len(objectives) - 1 and k == 0
            cells = [pct(piv.loc[obj, p], 1) for p in (128, 256, 512, 768, 1021)]
            cells[-1] += " \\\\[2pt]" if last else ""
            rows.append([label if j == 0 else "", OBJECTIVE_NAMES[obj]] + cells)
        cdf = summaries[label]["success_cdf"]
        if k == 0:
            rows.insert(0, ["", "\\emph{Attack has succeeded}"] + [pct(cdf[str(p)], 1) for p in (128, 256, 512, 768, 1021)])
    head = "Calibration pool & Client & 128 & 256 & 512 & 768 & 1{,}021 \\\\\n\\midrule"
    wrap("decision", head + "\n" + rows_tex(rows),
         "Exploratory sequential rule on the direction of the output drift (main corpus, its controls and explanation clients at "
         "$32\\times32$ pixels): evaluation sessions flagged by the stated number of queries (\\%, cumulative). The rule is "
         "evaluated after 64, 128, 192, 256, 384, 512, 640, 768, 896 and 1{,}021 queries; its thresholds are set on the "
         "calibration split so that at most 1\\% of the pooled benign sessions of the stated pool are flagged at any of these "
         "points. First row: attack sessions in which a query has been misclassified by then.",
         "tab:decision", "llccccc")


def table_output_stats():
    d = pd.read_csv(A / "output_statistics_r2/auroc.csv")
    main = d[d.corpus == "cifar_nes_seed0"].pivot_table(index="benign", columns=["statistic", "prefix"], values="auroc")
    rows = []
    for obj in ("restore", "confidence_boost", "lime", "kernelshap", "occlusion", "rise", "shuffled", "noise", "sweep", "random_walk",
                "boundary_probe", "counterfactual"):
        rows.append([OBJECTIVE_NAMES[obj]] + [fmt(main.loc[obj, (u, p)], 2) for u in ("drawdown", "trend") for p in ("256", "full")])
    head = ("& \\multicolumn{2}{c}{Margin drawdown} & \\multicolumn{2}{c}{Margin trend} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Attack vs & 256 queries & full & 256 queries & full \\\\\n\\midrule")
    wrap("outputstats", head + "\n" + rows_tex(rows),
         "AUROC of two single statistics of the model's outputs for separating attack sessions from each client (main corpus, "
         "evaluation split; fixed orientation, larger means more attack-like; nothing is fitted). Both are computed within the "
         "largest group of similar queries among the queries seen so far. The drawdown was planned in advance; the trend was "
         "introduced after the drawdown result and is exploratory.",
         "tab:outputstats", "lcccc")
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        g = d[(d.corpus == key) & (d.statistic == "trend")]
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        cell = lambda objs, prefix: rng(g[g.benign.isin(objs) & (g.prefix == prefix)].auroc.tolist())  # noqa: E731
        expl = ("lime", "kernelshap", "occlusion", "rise")
        rows.append([corpus_label(c), cell(tier_b, "256"), cell(tier_b, "full"), cell(expl, "256"), cell(expl, "full"),
                     cell(["boundary_probe"], "256"), cell(["boundary_probe"], "full"),
                     cell(["counterfactual"], "256"), cell(["counterfactual"], "full")])
    head = ("& \\multicolumn{2}{c}{Tier B} & \\multicolumn{2}{c}{Explanation clients} & \\multicolumn{2}{c}{Boundary probing} & "
            "\\multicolumn{2}{c}{Counterfactual search} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Corpus & 256 & full & 256 & full & 256 & full & 256 & full \\\\\n\\midrule")
    wrap("outputstats_corpora", head + "\n" + rows_tex(rows),
         "AUROC of the margin trend for separating attack sessions from the sessions of other clients on every corpus, after 256 "
         "queries and over the full session (evaluation split, both starts pooled; exploratory). Explanation clients exist for "
         "two models only. Ranges are over the clients of the group.",
         "tab:outputstatsall", "lcccccccc")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (table_corpora, table_controls, table_matched, table_operating, table_released, table_explanations, table_libraries,
               table_lfc, table_leak, table_nes, table_lfc_all, table_effects, table_full, table_explanation_utility,
               table_sensitivity, table_decision, table_output_stats):
        try:
            fn()
        except (KeyError, FileNotFoundError) as error:  # an input of this table has not been produced yet
            print(f"SKIPPED {fn.__name__}: missing {error}")


if __name__ == "__main__":
    main()
