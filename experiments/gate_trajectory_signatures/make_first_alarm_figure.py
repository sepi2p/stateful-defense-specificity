#!/usr/bin/env python3
"""Figure: when does each stateful detector fire, for attacks vs matched benign vs ordinary traffic?

Cumulative fraction of evaluation sessions alarmed by query q, one panel per detector at its
released/frozen operating point (Blacklight native, GWAD+ native, Lee-Fang-Chang reimplementation).
Also writes a table of median / IQR first-alarm query, alarm rate, and a two-sample KS test of
attack vs each matched benign client (restricted to sessions that alarmed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.lfc_detector import phase2_alarm  # noqa: E402

OUT = ROOT / "analysis_outputs/figures_20260926"
SOURCES = {
    "lfc_workloads_20260926": ("attack", "restore", "confidence_boost"),
    "explanation_clients_20260926": ("lime", "occlusion"),
    "lfc_controls_20260926": ("shuffled",),
}
# Fixed categorical order (validated palette, slots 1-6) + line style as secondary encoding.
SERIES = [
    ("attack", "Attack (NES)", "#2a78d6", "-", 2.6),
    ("restore", "Benign restoration (matched)", "#eb6834", "--", 2.0),
    ("confidence_boost", "Benign confidence raising (matched)", "#1baf7a", (0, (6, 2, 1, 2)), 2.0),
    ("lime", "LIME explanation", "#eda100", "-.", 2.0),
    ("occlusion", "Occlusion explanation", "#e87ba4", (0, (1, 1.5)), 2.2),
    ("shuffled", "Ordinary traffic (shuffled test images)", "#008300", ":", 2.0),
]
DETECTORS = [("blacklight", "Blacklight (released threshold)"),
             ("gwad_plus", "GWAD+ (released decision)"),
             ("lfc", "Lee-Fang-Chang 2026 (reimplemented)")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load():
    rows = []
    for folder, objectives in SOURCES.items():
        root = ROOT / "analysis_outputs" / folder
        for path in sorted(root.glob("sessions_shard*.jsonl")):
            for line in path.read_text().splitlines():
                s = json.loads(line)
                if s["split"] != "evaluation" or s["objective"] not in objectives:
                    continue
                trace = np.load(root / s["trace"])
                rows.append({
                    "objective": s["objective"], "calls": int(s["calls"]),
                    "first_success": int(s.get("first_success", -1)),
                    "blacklight": s["detectors"]["blacklight"]["first_alarm"],
                    "gwad_plus": s["detectors"]["gwad_plus"]["first_alarm"],
                    "lfc": phase2_alarm(trace["lfc_assignment"], trace["logits"]),
                })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    grid = np.unique(np.round(np.logspace(0, np.log10(1024), 300)).astype(int))
    success = frame.loc[(frame.objective == "attack") & (frame.first_success > 0), "first_success"]
    median_success = float(success.median())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True, constrained_layout=True)
    for ax, (det, title) in zip(axes, DETECTORS):
        for key, label, color, style, width in SERIES:
            first = frame.loc[frame.objective == key, det].to_numpy()
            curve = [(np.where(first > 0, first, np.inf) <= q).mean() for q in grid]
            ax.step(grid, curve, where="post", color=color, linestyle=style, linewidth=width, label=label)
        ax.axvline(median_success, color=INK2, linewidth=1, linestyle=(0, (2, 2)))
        ax.text(median_success * 1.06, 0.03, f"median attack\nsuccess (q={median_success:.0f})", color=INK2, fontsize=7.5, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1, 1024)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Query index q (log scale)")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Sessions blocked by query q")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8, handlelength=3.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"first_alarm_curves.{ext}", dpi=200)

    table = []
    for det, _ in DETECTORS:
        attack = frame.loc[(frame.objective == "attack") & (frame[det] > 0), det]
        for key, label, *_ in SERIES:
            first = frame.loc[frame.objective == key, det]
            fired = first[first > 0]
            row = {"detector": det, "client": key, "n": len(first), "blocked": round(float((first > 0).mean()), 3),
                   "median_first_alarm": float(fired.median()) if len(fired) else np.nan,
                   "iqr_low": float(fired.quantile(0.25)) if len(fired) else np.nan,
                   "iqr_high": float(fired.quantile(0.75)) if len(fired) else np.nan}
            if key in ("restore", "confidence_boost") and len(fired) and len(attack):
                ks = ks_2samp(attack, fired)
                row |= {"ks_vs_attack": round(float(ks.statistic), 3), "ks_p": float(ks.pvalue)}
            table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(OUT / "first_alarm_table.csv", index=False)
    print(f"median attack success query: {median_success:.0f} (n={len(success)})")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
