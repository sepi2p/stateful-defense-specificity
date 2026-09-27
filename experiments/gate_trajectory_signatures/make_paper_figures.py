#!/usr/bin/env python3
"""Figures 2-4 of the manuscript (Figure 1 is produced by make_first_alarm_figure.py).

fig 2  protocol schematic (drawn, no data)
fig 3  acceptance-rate leak: GWAD+ AUROC against the AUROC of the acceptance rate alone (post hoc),
       with the preregistered throttled-attack experiment X11
fig 4  decision time of the exploratory output diagnostic against attack success (exploratory)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis_outputs"
OUT = ROOT / "paper/jisa_2026/figures"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"


# The manuscript's text width is 390 pt (5.4 in). Figures are drawn at the width at which they are
# printed, so that the type appears at its nominal size.
WIDTH = 5.4


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def box(ax, xy, w, h, title, lines, face, edge):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=1.2, facecolor=face, edgecolor=edge))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.16, title, ha="center", va="top", fontsize=8.2, color=INK, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(xy[0] + w / 2, xy[1] + h - 0.56 - 0.34 * i, line, ha="center", va="top", fontsize=7.2, color=INK2)


def arrow(ax, a, b, color=INK2, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, linewidth=1.1, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_protocol():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.75)
    ax.axis("off")
    box(ax, (0.1, 3.05), 3.5, 1.5, "Adversarial client", ["objective: make the model", "misclassify the image"], "#fdebe3", ORANGE)
    box(ax, (0.1, 0.2), 3.5, 1.5, "Benign client", ["objective: e.g. restore the", "image and keep its label"], "#e3f5ee", AQUA)
    ax.text(1.85, 2.38, "same optimizer, start image,\nperturbation ball, query budget", ha="center", va="center", fontsize=7.2,
            color=INK, style="italic")
    box(ax, (4.75, 1.55), 2.1, 1.65, "Classifier", ["model $f$ behind", "a prediction API"], "#eef1f6", INK2)
    box(ax, (8.0, 1.55), 2.5, 1.65, "Stateful detector", ["history of queries;", "raises alarms"], "#e6effb", BLUE)
    arrow(ax, (3.6, 3.6), (4.75, 2.8), ORANGE)
    arrow(ax, (3.6, 1.15), (4.75, 1.95), AQUA)
    ax.text(4.32, 3.42, "queries", ha="center", fontsize=7, color=INK2)
    ax.text(4.32, 1.12, "queries", ha="center", fontsize=7, color=INK2)
    arrow(ax, (6.85, 2.75), (8.0, 2.75), INK2)
    ax.text(7.42, 2.88, "queries", ha="center", fontsize=7, color=INK2)
    arrow(ax, (6.85, 1.95), (8.0, 1.95), INK2, ls=(0, (4, 2)))
    ax.text(7.42, 1.58, "outputs", ha="center", fontsize=7, color=INK2)
    ax.text(8.5, 1.3, "query-only detectors use the queries;\noutput-aware detectors use both", ha="center", va="top", fontsize=6.8,
            color=INK2)
    save(fig, "protocol_schematic")


def fig_leak():
    cells = pd.read_csv(A / "acceptance_leak_r2/acceptance_vs_detectors.csv")
    thr = pd.read_csv(A / "acceptance_leak_r2/throttled_attack.csv")
    nes = pd.read_csv(A / "specificity_nes_sensitivity_20260927/analysis/cells.csv")
    nes = nes[(nes.variant.isin(["v1_step1", "v2_step2"])) & nes.valid]
    groups = [("nes", "NES, CIFAR-10 and GTSRB (6 corpora)", BLUE, "o"), ("nes_tiled", "Tiled NES, ImageNet", ORANGE, "s"),
              ("simba", "SimBA, CIFAR-10", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(4.3, 3.7), constrained_layout=True)
    ax.plot([0.4, 1.0], [0.4, 1.0], color=INK2, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.axhline(0.5, color=GRID, linewidth=0.8)
    for key, label, color, marker in groups:
        g = cells[cells.optimizer == key]
        ax.scatter(g.acceptance_only_auroc, g.gwad_plus_auroc, s=26, marker=marker, facecolor=color, edgecolor=SURFACE, linewidth=0.7,
                   label=label, zorder=3)
    ax.scatter(nes.acceptance_only_auroc, nes.gwad_plus_auroc, s=30, marker="D", facecolor=YELLOW, edgecolor=SURFACE, linewidth=0.7,
               label="NES with steps of 1/255 and 2/255, CIFAR-10", zorder=3, clip_on=False)
    t = thr[(thr.positive == "throttled attack") & (thr.negative == "restore")]
    u = thr[(thr.positive == "unthrottled attack") & (thr.negative == "restore")]
    ax.scatter(t.acceptance_only_auroc, t.gwad_plus_auroc, s=46, marker="^", facecolor="none", edgecolor=INK, linewidth=1.2,
               label="SimBA, attack throttled to the restoration\nclient's acceptance rate (vs restoration)", zorder=4)
    ax.set_xlim(0.4, 1.03)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("AUROC of the acceptance rate alone")
    ax.set_ylabel("AUROC of GWAD+")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=6.6, handletextpad=0.4)
    save(fig, "acceptance_leak")


def fig_decision_time():
    panels = [("", "(a) calibrated on all benign traffic"), ("_no_sweep_walk", "(b) calibrated without sweep and walk")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True, constrained_layout=True)
    for ax, (tag, title) in zip(axes, panels):
        rates = pd.read_csv(A / f"decision_time_r2/sequential_rates{tag}.csv")
        summary = json.loads((A / f"decision_time_r2/summary{tag}.json").read_text())
        piv = rates.pivot_table(index="prefix", columns="objective", values="flagged_by_prefix")
        pool = [o for o in summary["calibration_pool"] if o in piv.columns]
        worst = piv[pool].max(axis=1)
        cdf = pd.Series({int(k): v for k, v in summary["success_cdf"].items()}).sort_index()
        ax.plot(piv.index, piv["attack"], color=BLUE, linewidth=2.0, marker="o", markersize=3, label="Attack flagged")
        ax.plot(cdf.index, cdf.values, color=ORANGE, linewidth=1.6, linestyle="--", marker="s", markersize=2.6, label="Attack has succeeded")
        ax.plot(piv.index, piv["counterfactual"], color=AQUA, linewidth=1.5, linestyle="-.", label="Counterfactual search flagged (Tier C)")
        ax.plot(piv.index, piv["boundary_probe"], color=YELLOW, linewidth=1.5, linestyle=(0, (5, 1.5, 1, 1.5)), label="Boundary probing flagged (Tier C)")
        ax.plot(worst.index, worst.values, color=GREEN, linewidth=1.5, linestyle=":", label="Most-flagged client of the calibration pool")
        if tag:
            ax.plot(piv.index, piv["sweep"], color=MAGENTA, linewidth=1.3, linestyle=(0, (1, 1.2)), marker="x", markersize=3,
                    label="Degradation sweep flagged (outside the pool)")
        ax.set_title(title, color=INK, fontsize=7.8, loc="left")
        ax.set_xlim(0, 1040)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Queries observed")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Fraction of sessions (cumulative)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=6.8, handlelength=3.2,
               columnspacing=1.2)
    save(fig, "decision_time")


def main():
    style()
    fig_protocol()
    fig_leak()
    fig_decision_time()
    src = A / "figures_20260926/first_alarm_curves.pdf"
    (OUT / "first_alarm_curves.pdf").write_bytes(src.read_bytes())
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
