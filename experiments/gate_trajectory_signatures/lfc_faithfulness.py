#!/usr/bin/env python3
"""Faithfulness check for the Lee-Fang-Chang reimplementation (lfc_detector.py).

Before the reimplementation is applied to the specificity corpus it must
reproduce the paper's own CIFAR-10 behaviour: TPR near 1.00 on SimBA and
Square streams and FPR near 0 on the paper's benign types (random dataset
images, i.i.d. N(0, 0.1^2) noise around one image) plus a structured sweep.
Calibration-split images only; the evaluation split is not touched.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.lfc_detector import LFCPhase1, phase2_alarm  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_controls import build_stream  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_workloads import margin  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import load_cifar_model, project  # noqa: E402

EPS = 8 / 255


@torch.no_grad()
def simba_stream(model, x, label, budget, gen):
    order = torch.randperm(x[0].numel(), generator=gen, device=x.device)
    queries, cur = [x.clone()], x.clone()
    value = margin(model(cur), label)[0]
    it = 0
    while len(queries) + 2 <= budget:
        delta = torch.zeros_like(cur).flatten()
        delta[int(order[it % len(order)])] = EPS
        delta = delta.view_as(cur)
        cands = torch.cat([project(cur - delta, x, EPS), project(cur + delta, x, EPS)])
        vals = margin(model(cands), label)
        queries += [cands[:1], cands[1:]]
        i = int(vals.argmin())
        if vals[i] < value:
            cur, value = cands[i : i + 1], vals[i]
        it += 1
    return torch.cat(queries)


@torch.no_grad()
def square_stream(model, x, label, budget, gen, p_init=0.05):
    """Square Attack (L-inf, margin loss), Andriushchenko et al. 2020, single image."""
    c, h, w = x.shape[1:]
    init = torch.where(torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5, -EPS, EPS)
    cur = project(x + init, x, EPS)
    queries = [cur.clone()]
    best = margin(model(cur), label)[0]
    for it in range(budget - 1):
        frac = p_init * (0.5 ** sum(it >= t for t in (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000)))
        s = max(int(round((frac * h * w) ** 0.5)), 1)
        r0 = int(torch.randint(0, h - s + 1, (1,), generator=gen, device=x.device))
        c0 = int(torch.randint(0, w - s + 1, (1,), generator=gen, device=x.device))
        cand = cur.clone()
        signs = torch.where(torch.rand((1, c, 1, 1), generator=gen, device=x.device) < 0.5, -EPS, EPS)
        cand[:, :, r0 : r0 + s, c0 : c0 + s] = (x[:, :, r0 : r0 + s, c0 : c0 + s] + signs)
        cand = project(cand, x, EPS)
        queries.append(cand.clone())
        val = margin(model(cand), label)[0]
        if val < best:
            cur, best = cand, val
    return torch.cat(queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/lfc_faithfulness_20260926"))
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--bern-scales", default="0.05,0.0,1.0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", ROOT / "checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt", device).eval()
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    pool = np.array(sorted(set(range(len(dataset))) - set(manifest.dataset_index.astype(int))))
    rows_img = manifest[manifest.split == "calibration"].groupby("source_label").head(max(1, args.images // 10)).head(args.images)
    bern_scales = [float(b) for b in args.bern_scales.split(",")]
    records = []
    for n_img, row in enumerate(rows_img.itertuples(index=False)):
        clean = dataset[int(row.dataset_index)][0]
        x = clean[None].to(device)
        label = int(row.source_label)
        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.dataset_index))
        streams = {
            "simba": simba_stream(model, x, label, args.budget, gen).cpu(),
            "square": square_stream(model, x, label, args.budget, gen).cpu(),
        }
        for control in ("shuffled", "noise", "sweep"):
            streams[control] = build_stream(control, dataset, clean, pool, args.seed + int(row.dataset_index))[0][: args.budget]
        for name, stream in streams.items():
            with torch.no_grad():
                logits = torch.cat([model(stream[i : i + 256].to(device)) for i in range(0, len(stream), 256)]).cpu().numpy()
            for bern in bern_scales:
                det = LFCPhase1(d=stream[0].numel(), seed=args.seed, bern_scale=bern)
                for q in stream.numpy():
                    det.add(q)
                a = np.asarray(det.assignment)
                for lags, retest in itertools.product(("min10", 20), ("every", "once")):
                    records.append({
                        "image": int(row.dataset_index), "stream": name, "bern_scale": bern, "lags": str(lags),
                        "retest": retest, "largest_subsequence": int(np.bincount(a).max()),
                        "n_subsequences": int(a.max() + 1),
                        "phase2_first_alarm": phase2_alarm(a, logits, lags=lags, retest=retest),
                    })
        print(f"[{n_img + 1}/{len(rows_img)}] image {row.dataset_index} done", flush=True)
        pd.DataFrame(records).to_csv(args.output_dir / "faithfulness_rows.csv", index=False)
    frame = pd.DataFrame(records)
    frame["alarm"] = frame.phase2_first_alarm > 0
    table = frame.groupby(["bern_scale", "lags", "retest", "stream"]).agg(
        alarm_rate=("alarm", "mean"), median_first_alarm=("phase2_first_alarm", lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
        median_largest_subseq=("largest_subsequence", "median")).reset_index()
    table.to_csv(args.output_dir / "faithfulness_summary.csv", index=False)
    print(table.pivot_table(index=["bern_scale", "lags", "retest"], columns="stream", values="alarm_rate").round(2).to_string())
    print(frame.groupby(["bern_scale", "stream"]).largest_subsequence.median().unstack().to_string())


if __name__ == "__main__":
    main()
