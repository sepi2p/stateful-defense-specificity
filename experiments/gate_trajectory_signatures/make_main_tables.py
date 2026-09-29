#!/usr/bin/env python3
"""Compact tables for the main text of the article (the complete tables go to the supplement).

Every cell is read from the files written by the analysis scripts, as in make_paper_tables.py, whose
helpers are used. Output: paper/jisa_2026/tables/main_*.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from experiments.gate_trajectory_signatures.make_paper_tables import (  # noqa: E402
    A, CORPORA, N, OBJECTIVE_NAMES, fmt, med, pct, rng, rows_tex, separability, wrap)

SHORT = {"cifar_nes_seed0": "CIFAR-10, ResNet-18, seed 0", "cifar_nes_seed1": "CIFAR-10, ResNet-18, seed 1",
         "cifar_nes_seed2": "CIFAR-10, ResNet-18, seed 2", "cifar_nes_vgg19bn": "CIFAR-10, VGG19-BN",
         "cifar_nes_robust": "CIFAR-10, robust ResNet-50", "gtsrb_nes": "GTSRB, ResNet-18",
         "imagenet_nes": "ImageNet, ResNet-50", "cifar_simba": "CIFAR-10, ResNet-18, SimBA"}
OPTIMIZER_NOTE = "Optimizer: NES; tiled NES on ImageNet; SimBA in the last row."
CLIENT = {"lime_package": ("LIME", "\\texttt{lime}"), "captum_kernelshap": ("KernelSHAP", "Captum"),
          "captum_occlusion": ("Occlusion", "Captum"), "lime": ("LIME", "ours"), "kernelshap": ("KernelSHAP", "ours"),
          "occlusion": ("Occlusion", "ours"), "rise": ("RISE", "ours")}


def corpus_label(c):
    return SHORT[c[0]]


def table_corpora():
    inv = pd.read_csv(N / "corpus_inventory.csv").set_index("corpus")
    util = pd.read_csv(N / "workload_utility_all.csv")
    acc = json.loads((N / "model_accuracies.json").read_text())
    rows = []
    for c in CORPORA:
        key, _data, _d, _model, acc_key, _opt, _inv = c
        i = inv.loc[key]
        att = util[(util.corpus == key) & (util.objective == "attack")]
        rows.append([corpus_label(c), fmt(100 * acc[acc_key][0], 1), f"{i.fit_images}/{i.calibration_images}/{i.eval_images}",
                     f"{i.sessions:,}".replace(",", "{,}"), pct(att.flipped_frac.tolist(), 1), med(att.median_first_flip.tolist())])
    head = "Corpus & Acc.\\ (\\%) & Images & Sessions & Success (\\%) & Queries \\\\\n\\midrule"
    wrap("main_corpora", head + "\n" + rows_tex(rows),
         "Corpora. " + OPTIMIZER_NOTE + " ImageNet at $224\\times224$ pixels, the others at $32\\times32$. Each source "
         "image contributes two starts and one session per client and start. Acc.: clean test accuracy. Images: source "
         "images in the fit, calibration and evaluation splits. Success: evaluation sessions of the attack in which a query "
         "is misclassified within the budget. Queries: median index of the first such query. Ranges are over the two starts.",
         "tab:corpora", "lrcrcc")


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
        s = before.loc[key]
        rows.append([corpus_label(c), pct(a.blacklight_alarm.tolist()), pct(b.blacklight_alarm.tolist()),
                     med(a.blacklight_first_median.tolist()), med(b.blacklight_first_median.tolist()),
                     pct(a.blacklight_flagged_queries.tolist(), 1), pct(b.blacklight_flagged_queries.tolist(), 1),
                     pct(s.blacklight_success_before_alarm_frac, 1), pct(s.gwad_plus_success_before_alarm_frac, 1)])
    head = ("& \\multicolumn{2}{c}{Sessions (\\%)} & \\multicolumn{2}{c}{First alarm} & \\multicolumn{2}{c}{Queries (\\%)} & "
            "\\multicolumn{2}{c}{Before alarm (\\%)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Corpus & attack & benign & attack & benign & attack & benign & Blackl. & GWAD+ \\\\\n\\midrule")
    wrap("main_released", head + "\n" + rows_tex(rows),
         "Released operating points (evaluation split). " + OPTIMIZER_NOTE + " Sessions, first alarm, queries: sessions in "
         "which Blacklight raises an alarm, median query index of its first alarm, and share of the queries that it flags, "
         "for the attack and for the Tier-B clients (restoration and confidence raising; on ImageNet confidence raising). "
         "GWAD and GWAD+ raise an alarm in every session of these clients in every corpus, GWAD+ at query 259. Before "
         "alarm: successful attacks whose first misclassified query precedes the first alarm of the detector, or that "
         "raise no alarm. Ranges are over the two starts and the clients.",
         "tab:released", "lcccccccc", colsep="3pt")



NES_32 = ["cifar_nes_seed0", "cifar_nes_seed1", "cifar_nes_seed2", "cifar_nes_vgg19bn", "cifar_nes_robust", "gtsrb_nes"]
VALIDITY = A / "explanation_validity_20260927"


EXPLANATION_BLOCKS = [
    ("224", A / "explanation_libraries_20260927/imagenet/explanation_summary_r2.csv", "imagenet_libraries",
     ("lime_package", "captum_kernelshap", "captum_occlusion")),
    ("224", A / "explanation_clients_imagenet_20260926/explanation_summary_r2.csv", "imagenet_explain",
     ("lime", "kernelshap", "occlusion", "rise")),
    ("32", A / "explanation_libraries_20260927/cifar10/explanation_summary_r2.csv", "cifar_libraries",
     ("lime_package", "captum_kernelshap", "captum_occlusion")),
    ("32", A / "explanation_clients_20260926/explanation_summary_r2.csv", "cifar_explain",
     ("lime", "kernelshap", "occlusion", "rise"))]


def _explanations(name, label, blocks, with_lee, caption, spec):
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    rows = []
    for k, (res, path, key, clients) in enumerate(blocks):
        summary = pd.read_csv(path).set_index("client")
        for j, client in enumerate(clients):
            r = summary.loc[client]

            def cell(value, first):
                rate = float(value.split(" ")[0])
                return fmt(100 * rate, 1).rstrip("0").rstrip(".") + (f" ({fmt(first)})" if pd.notna(first) and rate > 0 else "")
            name_, source = CLIENT[client]
            name_ += "$^{\\dagger}$" if (res, client) == ("32", "lime_package") else ""
            last = j == len(clients) - 1 and k < len(blocks) - 1
            row = [res if j == 0 else "", name_, source, f"{int(r.queries):,}".replace(",", "{,}"),
                   cell(r.blacklight, r.blacklight_median_first), pct(r.blacklight_flagged_queries, 1),
                   "n/a" if int(r.queries) < 259 else cell(r.gwad_plus, r.gwad_plus_median_first)]
            if with_lee:
                s = sched.loc[(key, client)]
                row += [pct(s.update50, 1), pct(s.online, 1)]
            row[-1] += " \\\\[2pt]" if last else ""
            rows.append(row)
    if with_lee:
        head = ("& & & & \\multicolumn{2}{c}{Blacklight} & & \\multicolumn{2}{c}{Lee et al.} \\\\\n"
                "\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}\n"
                "Pixels & Client & Code & Queries & sessions & queries & GWAD+ & first 50 & whole \\\\\n\\midrule")
    else:
        head = ("& & & & \\multicolumn{2}{c}{Blacklight} & \\\\\n\\cmidrule(lr){5-6}\n"
                "Pixels & Client & Code & Queries & sessions & queries flagged & GWAD+ sessions \\\\\n\\midrule")
    wrap(name, head + "\n" + rows_tex(rows), caption, label, spec)


def table_explanations():
    common = ("Code: the \\texttt{lime} package, Captum, or our implementation of the query design. Sessions: sessions "
              "with an alarm (\\%), with the median query index of the first alarm in parentheses. Queries: share of the "
              "queries that Blacklight flags (\\%). n/a: the stream is shorter than the 259 queries that GWAD+ needs.")
    _explanations("main_explanations", "tab:main-expl", EXPLANATION_BLOCKS[:2], False,
                   "Explanation clients at $224\\times224$ pixels (ResNet-50): one session per image and client, 100 "
                   "images for the library clients and 200 for ours. " + common.split(" n/a:")[0] + " Results at $32\\times32$ pixels: "
                   "Table~\\ref{tab:expl-all} of the supplementary material.", "rllrccc")
    _explanations("explanations_all", "tab:expl-all", EXPLANATION_BLOCKS, True,
                   "Explanation clients at both resolutions (100 images for the library clients at $224\\times224$ "
                   "pixels, 200 otherwise). " + common + " Lee et al.: our reconstruction, test at every update of a "
                   "group, within the first 50 queries and over the whole session. $^{\\dagger}$The default segmentation "
                   "of the package returns a median of one superpixel at this resolution, and the explanation is empty.",
                   "rllrccccc")


ENFORCEMENT_BLOCKS = [
    # label, directory with sessions.csv and prior_sessions.csv, session-id prefix of the images used
    ("Images of Section~\\ref{sec:res-expl}, ResNet-50", VALIDITY / "libraries_imagenet", "evaluation__"),
    ("Fresh images of Section~\\ref{sec:res-profile}, ResNet-50", A / "operating_profile_imagenet_20260928/validity_resnet50", "confirmation__"),
    ("Fresh images of Section~\\ref{sec:res-profile}, ConvNeXt-T", A / "operating_profile_imagenet_20260928/validity_convnext", "confirmation__"),
]


def _median_ci(values):
    from experiments.gate_trajectory_signatures.analyze_explanation_validity import bootstrap_median
    return bootstrap_median(np.asarray(values, dtype=float))


def enforcement_stats(directory, prefix):
    """Per (client, mode): median answered queries; d with and without rejection; the paired change of d per image;
    gains over the centre prior. Intervals: bootstrap over images (2,000 resamples, seed 0)."""
    vs = pd.read_csv(directory / "sessions.csv")
    vs = vs[vs.session_id.str.startswith(prefix)]
    ps = pd.read_csv(directory / "prior_sessions.csv")
    ps = ps[ps.session_id.str.startswith(prefix)]
    out = {}
    for client in ("lime_package", "captum_kernelshap", "captum_occlusion"):
        rec = vs[(vs.client == client) & (vs.variant == "recorded")].set_index("session_id")
        x = ps[ps.client == client]
        for mode in ("recorded", "h1", "h2"):
            v = vs[(vs.client == client) & (vs.variant == mode)].set_index("session_id")
            if len(v) == 0:
                continue
            both = rec.join(v, rsuffix="_m", how="inner")
            out[(client, mode)] = {"images": len(v), "d": _median_ci(v.d), "queries": float(v.queries.median()),
                                   "answered": float(v.answered.median()) if mode != "recorded" else float(v.queries.median()),
                                   "change": _median_ci((both.d_m - both.d).to_numpy()),
                                   "retained": float(v.d.median() / rec.d.median()) if rec.d.median() > 0 else float("nan"),
                                   "gain": _median_ci((x.area_prior_centre - x[f"area_{mode}"]).to_numpy()),
                                   "constant": float((v.distinct_values == 1).mean())}
    return out


def table_enforcement():
    entries = [("lime_package", "h1", "LIME, uniform"), ("lime_package", "h2", "LIME, dropped"),
               ("captum_kernelshap", "h1", "KernelSHAP, uniform"), ("captum_occlusion", "h1", "Occlusion, uniform")]

    def ci(v, digits=2):
        # estimate over its interval, so that the table keeps the body font at the text width
        s = f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{fmt(v[0], digits)}\\\\ {{\\scriptsize[{fmt(v[1], digits)}, {fmt(v[2], digits)}]}}\\end{{tabular}}"
        return s.replace("-", "$-$")
    rows = []
    blocks = [(label, d, pre) for label, d, pre in ENFORCEMENT_BLOCKS if (d / "sessions.csv").exists() and (d / "prior_sessions.csv").exists()]
    for k, (label, d, pre) in enumerate(blocks):
        s = enforcement_stats(d, pre)
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{label} ({s[('lime_package', 'recorded')]['images']} images)}}}}"])
        for n, (client, mode, name) in enumerate(entries):
            r, e = s[(client, "recorded")], s[(client, mode)]
            row = [f"\\quad {name}", f"{fmt(e['answered'])}", fmt(r["d"][0], 2), fmt(e["d"][0], 2), ci(e["change"]),
                   ci(e["gain"])]
            if n == len(entries) - 1 and k < len(blocks) - 1:
                row[-1] += " \\\\[4pt]"
            rows.append(row)
    head = ("& & \\multicolumn{3}{c}{$d$} & Gain over centre \\\\\n\\cmidrule(lr){3-5}\n"
            "Client, withheld & Answered & all & rejected & change & prior, rejected \\\\\n\\midrule")
    wrap("main_enforcement", head + "\n" + rows_tex(rows),
         "Library explanation clients at $224\\times224$ pixels when Blacklight's flagged queries are rejected in a replay "
         "(medians over images; under the paired change and the gain: bootstrap 95\\% interval over images). Fresh images: "
         "the 100 held-out images. LIME: the \\texttt{lime} package; KernelSHAP, occlusion: Captum. Answered: queries "
         "answered of 1{,}000 (occlusion: 785). Withheld answers: replaced by the uniform distribution, or dropped from "
         "the fit of the \\texttt{lime} package. $d$: gain in deletion area over randomizations of the explanation that "
         "keep its spatial structure (a constant explanation has $d=0$; a positive $d$ does not show that the information "
         "comes from the model), with every query answered, with the flagged queries rejected, and the paired change per "
         "image. Gain over centre prior: deletion area of a Gaussian around the image centre minus that of the rejected "
         "explanation (added after the first results were known); on ConvNeXt-T this prior is weaker than a constant "
         "explanation. Further measures: Tables~\\ref{tab:validity} and~\\ref{tab:priors} of the supplementary material.",
         "tab:main-enforce", "lccccc", size="\\footnotesize", colsep="4pt")


def table_matched():
    rows = []
    for c in CORPORA:
        _key, _data, directory, _model, _acc, _opt, invalid = c
        sep = separability(directory)
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        r = [corpus_label(c)]
        for det in ("logreg:gwad_plus", "logreg:blacklight"):
            g = sep[sep.negative.isin(tier_b) & (sep.detector == det) & (sep.prefix == "1024")]
            chance = int(((g.lo <= 0.5) & (g.hi >= 0.5)).sum())
            r += [rng(g.auroc.tolist()), f"{fmt(g.lo.min(), 2)}--{fmt(g.hi.max(), 2)}", f"{chance}/{len(g)}"]
        rows.append(r)
    head = ("& \\multicolumn{3}{c}{GWAD+} & \\multicolumn{3}{c}{Blacklight} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Corpus & AUROC & intervals & incl.\\ 0.5 & AUROC & intervals & incl.\\ 0.5 \\\\\n\\midrule")
    wrap("main_matched", head + "\n" + rows_tex(rows),
         "Matched AUROC, attack against Tier-B clients: logistic models on the session statistics of each detector "
         "(fitted on the fit split, evaluation split shown, whole session). " + OPTIMIZER_NOTE + " A cell is one benign "
         "client (restoration or confidence raising; on ImageNet confidence raising) on one start. AUROC: range over the "
         "cells. Intervals: smallest lower and largest upper limit of the bootstrap 95\\% intervals of the cells. "
         "Incl.\\ 0.5: cells whose interval contains 0.5. The values of GWAD agree with those of GWAD+ to within 0.01. "
         "Every cell with its interval: Table~\\ref{tab:full}.",
         "tab:main-matched", "lcccccc")


def table_acceptance():
    c = pd.read_csv(A / "specificity_nes_sensitivity_20260927/analysis/cells.csv")
    thr = pd.read_csv(A / "acceptance_leak_r2/throttled_attack.csv")
    short = {"restore": "Restoration", "confidence_boost": "Conf.\\ raising"}
    rows = []
    variants = {"main": "NES, 0.25/255, if better", "v1_step1": "NES, 1/255, if better",
                "v2_step2": "NES, 2/255, if better", "v3_step2_always": "NES, 2/255, always"}
    for variant, label in variants.items():
        for j, objective in enumerate(("restore", "confidence_boost")):
            g = c[(c.variant == variant) & (c.benign == objective)]
            star = "" if bool(g.valid.all()) else "$^{\\ast}$"
            if j == 0:
                rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}}"])
            rows.append([f"\\quad {short[objective]}{star}",
                         rng(g.attack_acceptance_median.tolist()), rng(g.benign_acceptance_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()),
                         rng(g.blacklight_auroc.tolist()) + (" \\\\[2pt]" if j == 1 else "")])
    for positive, label in (("unthrottled attack", "SimBA, 8/255, if better"), ("throttled attack", "SimBA, 8/255, throttled$^{\\ddagger}$")):
        for j, objective in enumerate(("restore", "confidence_boost")):
            g = thr[(thr.positive == positive) & (thr.negative == objective)]
            last = j == 1 and positive == "unthrottled attack"
            if j == 0:
                rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}}"])
            rows.append([f"\\quad {short[objective]}",
                         rng(g.accept_pos_median.tolist()), rng(g.accept_neg_median.tolist()),
                         rng(g.acceptance_only_auroc.tolist()), rng(g.gwad_plus_auroc.tolist()),
                         rng(g.blacklight_auroc.tolist()) + (" \\\\[2pt]" if last else "")])
    head = ("& \\multicolumn{2}{c}{Acceptance} & \\multicolumn{3}{c}{Matched AUROC} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
            "Benign client & attack & benign & acceptance & GWAD+ & Blacklight \\\\\n\\midrule")
    wrap("main_acceptance", head + "\n" + rows_tex(rows),
         "Acceptance behaviour and matched AUROC (CIFAR-10, ResNet-18, 200 evaluation images, whole session; ranges over "
         "the two starts). Panels: optimizer, step size and move rule of the attack. Move: the client moves to the evaluated point if its objective improves, or always, as NES "
         "does in its published form. Acceptance: median fraction of iterations in which the current image moves. "
         "Matched AUROC: of the acceptance rate alone, and of logistic models on the session statistics of GWAD+ and "
         "Blacklight. At the released operating points both detectors raise an alarm in every session of every row. "
         "$^{\\ast}$Invalid workload: the client accepts almost no step. $^{\\ddagger}$The attack accepts an improving "
         "step only while its running acceptance rate is at most a cap drawn from restoration sessions of the fit split; "
         "the comparison with confidence raising was not planned. Every cell with its interval: "
         "Table~\\ref{tab:accept-cells} of the supplementary material.",
         "tab:main-accept", "lccccc", colsep="4pt")


def table_lfc():
    sched = pd.read_csv(N / "lfc_schedules.csv").set_index(["corpus", "objective"])
    sens = pd.read_csv(A / "lfc_sensitivity_r2/sensitivity_summary.csv")
    sens = sens[(sens.part == "A") & (sens.variant == "frozen")].set_index("stream")
    groups = [("NES, step 0.25/255 (6 corpora)", NES_32, {}),
              ("NES, larger steps (3 variants)", ["nes_v1_step1", "nes_v2_step2", "nes_v3_step2_always"], {("nes_v2_step2", "restore")}),
              ("Tiled NES, ImageNet", ["imagenet_nes"], {("imagenet_nes", "restore")}),
              ("SimBA", ["cifar_simba"], {})]
    schedules = (("single50", "first 50, one test"), ("update50", "first 50, every update"), ("online", "whole session"))
    rows = []
    for label, keys, invalid in groups:
        for j, (schedule, name) in enumerate(schedules):
            r = [label if j == 0 else "", name]
            for obj in ("attack", "restore", "confidence_boost"):
                values = [sched.loc[(k, obj), schedule] for k in keys if (k, obj) not in invalid]
                r.append(pct(values, 1) if values else "--")
            if j == len(schedules) - 1:
                r[-1] += " \\\\[2pt]"
            rows.append(r)
    for stream, label in (("shuffled", "Unrelated test images"), ("noise_small", "Near-duplicate noise copies")):
        s = sens.loc[stream]
        for j, (column, name) in enumerate((("seq50_flag", "first 50, one test"), ("alarm_within_50", "first 50, every update"), ("alarm", "whole session"))):
            rows.append([label if j == 0 else "", name, f"\\multicolumn{{3}}{{c}}{{{pct(s[column], 1)}}}"
                         + (" \\\\[2pt]" if j == 2 and stream == "shuffled" else "")])
    head = "Traffic & Test applied & Attack & Restoration & Conf.\\ raising \\\\\n\\midrule"
    wrap("main_lfc", head + "\n" + rows_tex(rows),
         "Our reconstruction of the detector of Lee et al.: evaluation sessions flagged (\\%), both starts pooled; ranges "
         "over the corpora of a row; $32\\times32$ pixels except ImageNet. One test: the first 50 queries are grouped and "
         "every group of at least 15 queries is tested once. Every update: the test is applied whenever such a group "
         "gains a member. Invalid workloads are left out. Last two blocks: streams without an objective (50 sequences "
         "each, from calibration images); the soft labels of the noise copies are independent, so that their alarms are false alarms of the test. "
         "Every corpus: Table~\\ref{tab:lfc}.",
         "tab:main-lfc", "llccc", colsep="5pt")


def table_validity():
    """Supplement: rule R2 for every explanation client, without and with rejection."""
    blocks = [("224", "libraries_imagenet", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224*", "../operating_profile_imagenet_20260928/validity_resnet50", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224\\textsuperscript{c}", "../operating_profile_imagenet_20260928/validity_convnext", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224", "own_imagenet", ("lime", "kernelshap", "occlusion", "rise")),
              ("32", "libraries_cifar10", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("32", "own_cifar10", ("lime", "kernelshap", "occlusion", "rise"))]
    modes = {"recorded": "every query answered", "h1": "rejected, uniform", "h2": "rejected, dropped"}
    rows = []
    for res, directory, clients in blocks:
        path = VALIDITY / directory / "summary.csv"
        if not path.exists():
            print("skipped validity: missing", path)
            return
        s = pd.read_csv(path).set_index(["client", "variant"])
        for client in clients:
            for mode, label in modes.items():
                if (client, mode) not in s.index:
                    continue
                r = s.loc[(client, mode)]
                constant = r.distinct_median == 1

                def ci(name, digits=2):
                    if mode == "recorded" or constant:
                        return "--"
                    return f"{fmt(r[name + '_median'], digits)} {{[}}{fmt(r[name + '_lo'], digits)}, {fmt(r[name + '_hi'], digits)}{{]}}"
                rows.append([res if mode == "recorded" else "", OBJECTIVE_NAMES[client] if mode == "recorded" else "", label,
                             f"{int(r.images)}", f"{int(float(fmt(r.distinct_median))):,}".replace(",", "{,}"),
                             "--" if mode == "recorded" else fmt(r.answered_median),
                             f"{fmt(r.d_median, 3)} {{[}}{fmt(r.d_lo, 3)}, {fmt(r.d_hi, 3)}{{]}}".replace("-", "$-$"),
                             "--" if constant else pct(r.d_positive_share), ci("spearman"), ci("top_fifth")])
        rows[-1][-1] += " \\\\[2pt]"
    rows[-1][-1] = rows[-1][-1].replace(" \\\\[2pt]", "")
    head = ("Pixels & Client & Queries & Images & Regions & Answered & $d$ & $d>0$ (\\%) & Rank correlation & Overlap \\\\\n\\midrule")
    wrap("validity", head + "\n" + rows_tex(rows),
         "Rule R2 for every explanation client (evaluation images; 224*: the 200 fresh calibration and confirmation images "
         "of X16 with ResNet-50; 224\\textsuperscript{c}: the 100 fresh confirmation images with ConvNeXt-T). Clients with a "
         "source are library clients, the others our implementations. Queries: every query answered, or the queries flagged by Blacklight rejected, with the "
         "uniform distribution in the place of a rejected answer or, for the \\texttt{lime} package, with the rejected "
         "samples dropped from the fit. Regions: median number of distinct attribution values. Answered: median number of "
         "answered queries. $d$: median gain in deletion area over 20 randomizations of the explanation that keep its "
         "spatial structure, with its bootstrap 95\\% interval. $d>0$: share of the images. Rank correlation and overlap "
         "of the most important fifth (0.2 by chance) are relative to the explanation without rejection; dashes where an "
         "explanation is constant over the image.",
         "tab:validity", "rllrrrcccc", colsep="3pt", sideways=True)


PROFILE = A / "operating_profile_r1"


SHORTW = {"Occlusion (ours)": "occlusion", "Occlusion (Captum)": "occlusion", "KernelSHAP (Captum)": "KernelSHAP",
          "KernelSHAP (ours)": "KernelSHAP", "unrelated images": "unrelated", "LIME (lime)": "LIME", "LIME (ours)": "LIME"}


def _theta_text(det, theta):
    if det == "blacklight":
        return f"$T={int(float(theta))}$" + ("\\textsuperscript{n}" if int(float(theta)) == 50 else "")
    if str(theta) == "native":
        return "released"
    return f"$c={float(theta):g}$" + ("\\textsuperscript{n}" if float(theta) >= 12 else "")


def share(values):
    """Whole percentages, with one decimal where rounding would show 0 or 100 for a value that is not."""
    whole = [round(100 * v) for v in values]
    exact = all(v in (0.0, 1.0) for v in values)
    return pct(values, 0 if exact or not any(w in (0, 100) for w in whole) else 1)


def table_operating():
    """Main text: released operating points (Table 4) and selection under benign-alarm budgets (Table 5)."""
    util = pd.read_csv(N / "workload_utility_all.csv")
    before = pd.read_csv(N / "success_before_alarm.csv")
    before = before[before.start == "pooled"].set_index("corpus")
    a_rows, panel = [], None
    for c in CORPORA:
        key, _data, _d, _model, _acc, _opt, invalid = c
        tier_b = [o for o in ("restore", "confidence_boost") if o not in invalid]
        a = util[(util.corpus == key) & (util.objective == "attack")]
        b = util[(util.corpus == key) & util.objective.isin(tier_b)]
        s = before.loc[key]
        # rows grouped by dataset, so that the first column carries only the model
        dataset, model = corpus_label(c).split(", ", 1)
        if model.endswith(", SimBA"):
            dataset, model = f"{dataset}, SimBA", model[:-len(", SimBA")]
        if dataset != panel:
            a_rows.append([f"\\multicolumn{{10}}{{l}}{{\\emph{{{dataset}}}}}"])
            panel = dataset
        a_rows.append([model, pct(a.flipped_frac.tolist()), pct(a.blacklight_alarm.tolist()), pct(b.blacklight_alarm.tolist()),
                       med(a.blacklight_first_median.tolist()), med(b.blacklight_first_median.tolist()),
                       share(a.blacklight_flagged_queries.tolist()), share(b.blacklight_flagged_queries.tolist()),
                       pct(s.blacklight_success_before_alarm_frac, 1), pct(s.gwad_plus_success_before_alarm_frac, 1)])
    head_a = ("& & \\multicolumn{2}{c}{Alarms (\\%)} & \\multicolumn{2}{c}{First alarm} & \\multicolumn{2}{c}{Flagged (\\%)} & "
              "\\multicolumn{2}{c}{Before alarm} \\\\\n"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
              "Model & Success & attack & benign & attack & benign & attack & benign & Blackl. & GWAD+ \\\\\n\\midrule")
    wrap("main_operating", head_a + "\n" + rows_tex(a_rows),
         "Released operating points (evaluation split). " + OPTIMIZER_NOTE + " Success: attack sessions with a misclassified "
         "query within the budget (\\%). Alarms, first alarm, flagged: Blacklight's sessions with an alarm, median index of "
         "its first alarm and share of flagged queries, for the attack and the Tier-B clients (on ImageNet confidence "
         "raising only); GWAD and GWAD+ raise an alarm in every such session, GWAD+ at query 259. Before alarm: share (\\%) of the successful "
         "attacks whose first misclassified query precedes the detector's first alarm, or that raise none (the complement "
         "of the timely rate of Table~\\ref{tab:selection}, except for an alarm on the misclassified query itself). Ranges: over the "
         "two starts and, in the benign columns, the clients.",
         "tab:operating-main", "lccccccccc", colsep="2.5pt")
    names = {"blacklight": "Blacklight", "gwad_plus": "GWAD+", "gwad": "GWAD"}
    rows = []
    for setting, label in (("cifar10", "CIFAR-10, retrospective"), ("imagenet", "ImageNet, confirmatory")):
        path = PROFILE / setting / "profile.csv"
        if not path.exists():
            continue
        prof = pd.read_csv(path)
        rel = pd.read_csv(PROFILE / setting / "released.csv")
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\emph{{{label}}}}}"])
        for det in ("blacklight", "gwad_plus"):
            rr = rel[(rel.detector == det) & (rel.workload == "attack (timely)")].iloc[0]
            ben = rel[(rel.detector == det) & (rel.workload != "attack (timely)") & (rel.sessions > 0)]
            prof_workloads = ([g for g in ben.workload if g in ("unrelated images", "LIME (ours)", "KernelSHAP (ours)", "Occlusion (ours)", "RISE (ours)")]
                              if setting == "cifar10" else [g for g in ben.workload if g != "confidence raising"])
            worst = ben[ben.workload.isin(prof_workloads)].sort_values("rate").iloc[-1]
            tb = [g for g in ("confidence raising", "restoration") if g in set(ben.workload)]
            rows.append([names[det], "released", "--", "--", f"{pct(worst.rate)} ({SHORTW.get(worst.workload, worst.workload)})",
                         pct(ben[ben.workload.isin(tb)].rate.max()), f"{int(rr.alarms)}/{int(rr.sessions)}"])
            for profile, alpha in (("explanation", 0.01), ("explanation", 0.05), ("diagnostic", 0.01)):
                r = prof[(prof.detector == det) & (prof.profile == profile) & np.isclose(prof.alpha, alpha)].iloc[0]
                fcols = {c[2:-1]: r[c] for c in prof.columns if c.startswith("F[") and isinstance(r[c], str) and not r[c].endswith("/0")}
                g_in = [g for g in fcols if g in set(prof_workloads)]
                worst_g = max(g_in, key=lambda g: int(fcols[g].split("/")[0]) / int(fcols[g].split("/")[1]))
                k, n = map(int, fcols[worst_g].split("/"))
                tb_rates = [int(fcols[g].split("/")[0]) / int(fcols[g].split("/")[1]) for g in tb if g in fcols]
                R = f"{int(r.timely)}/{int(r.successful)}"
                if 0 < r.timely < r.successful:
                    R += f" [{pct(r.R_lo)}, {pct(r.R_hi)}]"
                rows.append(["", f"{profile[0].upper()}, {int(alpha * 100)}\\%", _theta_text(det, r.theta), str(int(r.feasible_candidates)),
                             f"{pct(k / n)} ({SHORTW.get(worst_g, worst_g)})", pct(max(tb_rates)) if tb_rates else "--", R])
    head = ("& & & & \\multicolumn{2}{c}{Held-out alarms (\\%)} & Timely / \\\\\n\\cmidrule(lr){5-6}\n"
            "Detector & Budget & Selected & Feasible & worst in $G$ & Tier B & successful \\\\\n\\midrule")
    wrap("main_selection", head + "\n" + rows_tex(rows),
         "Operation under benign-alarm budgets (Section~\\ref{sec:procedure}). Released: the published decision rule. The "
         "other rows select a configuration on the calibration split under the budget $\\alpha$ for every workload of the "
         "profile, E (explanation: unrelated images and the explanation clients of the setting) or D (diagnostic: these and "
         "the Tier-B clients), and evaluate it once on the held-out split. Selected: Blacklight flags a query with more "
         "than $T$ of 50 matching hashes; GWAD+ alarms when its score exceeds $c$; \\textsuperscript{n}never alarms. "
         "Feasible: number of feasible candidates. Held-out alarms: the largest rate among the workloads of the explanation "
         "profile, and that of the Tier-B clients, which the explanation profile does not constrain. Timely: first alarm "
         "before the first misclassified query; brackets: 95\\% interval from resampling source images. Every workload "
         "with its interval: Table~\\ref{tab:profile-full} of the supplementary material.",
         "tab:selection", "llllccc", colsep="4pt")


def table_profile_full():
    """Supplement: every selection with its per-workload held-out rates and intervals."""
    names = {"blacklight": "Blacklight", "gwad_plus": "GWAD+", "gwad": "GWAD"}
    lines = []
    for setting, label in (("cifar10", "CIFAR-10, retrospective (calibration / evaluation split)"),
                           ("imagenet", "ImageNet, confirmatory (calibration / confirmation split)")):
        path = PROFILE / setting / "profile.csv"
        if not path.exists():
            continue
        prof = pd.read_csv(path)
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for _, r in prof.iterrows():
            cells = []
            for col in prof.columns:
                if not col.startswith("F[") or not isinstance(r[col], str) or r[col].endswith("/0"):
                    continue
                g = col[2:-1]
                k, n = map(int, r[col].split("/"))
                lo, hi = (float(v) for v in r[f"Fci[{g}]"].split("-"))
                cells.append(f"{g} {k}/{n} [{pct(lo, 1)}, {pct(hi, 1)}]")
            R = f"{int(r.timely)}/{int(r.successful)} [{pct(r.R_lo, 1)}, {pct(r.R_hi, 1)}]"
            lines.append(f"{names[r.detector]} & {r.profile}, {int(r.alpha * 100)}\\% & {_theta_text(r.detector, r.theta)} & {R} \\\\")
            lines.append(f"\\multicolumn{{4}}{{p{{12.8cm}}}}{{\\raggedright\\quad {'; '.join(cells)}}} \\\\[4pt]")
    head = ("Detector & Profile, $\\alpha$ & Selected & Timely / successful (\\%) \\\\\n"
            "\\multicolumn{4}{p{12.8cm}}{\\quad Held-out alarms per workload: alarms/sessions [95\\% interval, \\%]} \\\\\n\\midrule")
    body = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{p{1.6cm}p{2.6cm}p{2.8cm}p{4.6cm}}\n"
            "\\caption{Selection under benign-alarm budgets, every detector, profile and budget: selected configuration, "
            "timely alarms among successful attacks with a 95\\% interval from resampling source images (exact Clopper--Pearson "
            "over source images when no or every successful attack is timely), and, on the held-out split, the alarms of every "
            "workload, constrained or not, with exact Clopper--Pearson 95\\% intervals over sessions (one session per image "
            "for the explanation clients and controls, except the CIFAR-10 streams of unrelated images, ten per image and "
            "treated as independent; the two starts of the optimizing clients are not independent).}\\label{tab:profile-full}\\\\\n"
            "\\toprule\n" + head + "\n\\endfirsthead\n\\toprule\n" + head + "\n\\endhead\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    (ROOT / "paper/jisa_2026/tables/profile_full.tex").write_text(body)
    print("wrote tables/profile_full.tex")


START = {"denoise": "noisy", "deblur": "blurred"}


def with_interval(value, lo, hi, digits=2):
    return f"{fmt(value, digits)} {{[}}{fmt(lo, digits)}, {fmt(hi, digits)}{{]}}"


def table_prefix():
    """Supplement: the model's outputs against Tier-C clients before a label change, every cell."""
    lines = []
    for c in CORPORA:
        sep = separability(c[2])
        g = sep[(sep.detector == "logreg:output_trajectory") & sep.negative.isin(["boundary_probe", "counterfactual"])]
        first = True
        for start in ("denoise", "deblur"):
            for objective in ("boundary_probe", "counterfactual"):
                cell = {}
                for prefix in ("attack_truncated", "pre_either_flip"):
                    r = g[(g.workload == start) & (g.negative == objective) & (g.prefix == prefix)]
                    cell[prefix] = r.iloc[0] if len(r) else None
                sym, tru = cell["pre_either_flip"], cell["attack_truncated"]
                lines.append([corpus_label(c) if first else "", START[start], OBJECTIVE_NAMES[objective],
                              "--" if tru is None else with_interval(tru.auroc, tru.lo, tru.hi),
                              "--" if sym is None else f"{int(sym.n_eval_pairs)}",
                              "--" if sym is None else fmt(sym.median_prefix, 1).replace(".0", ""),
                              "--" if sym is None else with_interval(sym.auroc, sym.lo, sym.hi)])
                first = False
        lines[-1][-1] += " \\\\[2pt]"
    lines[-1][-1] = lines[-1][-1].replace(" \\\\[2pt]", "")
    head = ("& & & Truncated at the attack's & \\multicolumn{3}{c}{Before the first label change of either session} \\\\\n"
            "\\cmidrule(lr){5-7}\n"
            "Corpus & Start & Benign client & first label change & pairs & median length & AUROC \\\\\n\\midrule")
    wrap("prefix", head + "\n" + rows_tex(lines),
         "The classifier's outputs before a label change: AUROC of the logistic model on the statistics of the model's "
         "outputs for separating the attack from the Tier-C clients, with bootstrap 95\\% intervals (evaluation split). "
         + OPTIMIZER_NOTE + " Truncated: both sessions of a pair end before the attack's first label change (the definition "
         "written down first). The symmetric prefix ends before the first label change of either session and omits the "
         "indicator of a label change; pairs with fewer than 17 queries are left out.",
         "tab:prefix", "lllcrrc", colsep="4pt", sideways=True)


def table_acceptance_cells():
    """Supplement: every cell of the acceptance experiments with its interval."""
    c = pd.read_csv(A / "specificity_nes_sensitivity_20260927/analysis/cells.csv")
    thr = pd.read_csv(A / "acceptance_leak_r2/throttled_attack.csv")
    thr = thr[thr.negative != "(attack outcome)"]
    lines = []

    def group(label):
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{label}}}}} \\\\")

    def row(cells):
        lines.append(" & ".join(cells) + " \\\\")
    variants = {"main": "NES, step 0.25/255, moves if better", "v1_step1": "NES, step 1/255, moves if better",
                "v2_step2": "NES, step 2/255, moves if better", "v3_step2_always": "NES, step 2/255, always moves"}
    for variant, label in variants.items():
        g = c[c.variant == variant]
        group(label)
        for objective in ("restore", "confidence_boost"):
            for start in ("denoise", "deblur"):
                r = g[(g.benign == objective) & (g.start == start)].iloc[0]
                star = "" if bool(r.valid) else "$^{\\ast}$"
                row([OBJECTIVE_NAMES[objective] + star, START[start], fmt(r.acceptance_only_auroc, 2),
                     with_interval(r.gwad_plus_auroc, r.gwad_plus_lo, r.gwad_plus_hi),
                     with_interval(r.blacklight_auroc, r.blacklight_lo, r.blacklight_hi)])
    for positive, label in (("unthrottled attack", "SimBA, step 8/255, moves if better"),
                            ("throttled attack", "SimBA, step 8/255, acceptance rate throttled")):
        group(label)
        for objective in ("restore", "confidence_boost", "boundary_probe", "counterfactual"):
            for start in ("denoise", "deblur"):
                r = thr[(thr.positive == positive) & (thr.negative == objective) & (thr.start == start)].iloc[0]
                row([OBJECTIVE_NAMES[objective], START[start], fmt(r.acceptance_only_auroc, 2),
                     with_interval(r.gwad_plus_auroc, r.gwad_plus_lo, r.gwad_plus_hi),
                     with_interval(r.blacklight_auroc, r.blacklight_lo, r.blacklight_hi)])
    head = ("& & \\multicolumn{3}{c}{Matched AUROC} \\\\\n\\cmidrule(lr){3-5}\n"
            "Benign client & Start & acceptance & GWAD+ & Blacklight \\\\\n\\midrule")
    body = ("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{5pt}\n\\begin{longtable}{llccc}\n"
            "\\caption{The acceptance experiments in every cell (CIFAR-10, ResNet-18, 200 evaluation images, whole session): "
            "matched AUROC of the acceptance rate alone and of the logistic models on the session statistics of GWAD+ and "
            "Blacklight, with bootstrap 95\\% intervals. $^{\\ast}$Invalid workload. The comparisons of the throttled attack "
            "with clients other than restoration were not planned.}\\label{tab:accept-cells}\\\\\n\\toprule\n" + head + "\n\\endfirsthead\n"
            "\\multicolumn{5}{l}{\\emph{Table~\\thetable\\ (continued)}}\\\\\n\\toprule\n" + head + "\n\\endhead\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")
    (ROOT / "paper/jisa_2026/tables/acceptance_cells.tex").write_text(body)
    print("wrote tables/acceptance_cells.tex")


def table_priors():
    """Supplement: explanations against orderings that use no answer of the model."""
    blocks = [("224", "libraries_imagenet", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224*", "../operating_profile_imagenet_20260928/validity_resnet50", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224\\textsuperscript{c}", "../operating_profile_imagenet_20260928/validity_convnext", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("224", "own_imagenet", ("lime", "kernelshap", "occlusion", "rise")),
              ("32", "libraries_cifar10", ("lime_package", "captum_kernelshap", "captum_occlusion")),
              ("32", "own_cifar10", ("lime", "kernelshap", "occlusion", "rise"))]
    modes = {"recorded": "all answered", "h1": "rejected, uniform", "h2": "rejected, dropped"}
    rows = []
    for res, directory, clients in blocks:
        s = pd.read_csv(VALIDITY / directory / "prior_summary.csv").set_index(["client", "variant"])
        for client in clients:
            for mode, label in modes.items():
                if (client, mode) not in s.index:
                    continue
                r = s.loc[(client, mode)]

                def cell(name):
                    if f"gain_over_{name}" not in r or pd.isna(r[f"gain_over_{name}"]):
                        return ["--", "--"]
                    text = with_interval(r[f"gain_over_{name}"], r[f"gain_over_{name}_lo"], r[f"gain_over_{name}_hi"]).replace("-", "$-$")
                    return [text, pct(r[f"better_than_{name}"])]
                name, source = CLIENT[client]
                rows.append([res if mode == "recorded" else "", f"{name} ({source})" if mode == "recorded" else "", label,
                             fmt(r.area_median, 2)] + cell("prior_centre") + cell("prior_regions"))
        rows[-1][-1] += " \\\\[2pt]"
    rows[-1][-1] = rows[-1][-1].replace(" \\\\[2pt]", "")
    head = ("& & & & \\multicolumn{2}{c}{Over the centre prior} & \\multicolumn{2}{c}{Over ranked regions} \\\\\n"
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Pixels & Client & Queries & Area & gain & better (\\%) & gain & better (\\%) \\\\\n\\midrule")
    wrap("priors", head + "\n" + rows_tex(rows),
         "Explanations against orderings that use no answer of the model (evaluation images; 224* and "
         "224\\textsuperscript{c} as in Table~\\ref{tab:validity}; not planned in advance). On ConvNeXt-T the centre prior is "
         "weaker than a smooth random ordering. "
         "Area: median deletion area of the explanation. Centre prior: a Gaussian around the centre of the image with a "
         "standard deviation of a quarter of its side (median area 0.23 at $224\\times224$ and 0.31 at $32\\times32$ "
         "pixels). Ranked regions: the regions of the explanation computed from all answers, ranked by the distance of "
         "their centroid from the centre; for maps with at most 256 distinct values and at least two. Gain: median of "
         "the area of the prior minus the area of the explanation, with its bootstrap 95\\% interval. Better: share of "
         "the images on which the explanation has the smaller area.",
         "tab:priors", "rllccccc", colsep="3pt", sideways=True)


if __name__ == "__main__":
    for table in (table_corpora, table_released, table_explanations, table_enforcement, table_matched, table_acceptance,
                  table_lfc, table_validity, table_prefix, table_acceptance_cells, table_priors, table_operating,
                  table_profile_full):
        table()
