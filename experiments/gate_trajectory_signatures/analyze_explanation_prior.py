#!/usr/bin/env python3
"""X15, exploratory addition: explanations against orderings that do not use the model.

Rule R2 (analyze_explanation_validity.py) compares an explanation with randomizations of itself.
A map that only prefers the centre of the image also meets that rule, because the objects of the
datasets are centred. This script compares the deletion area of each explanation, without and
with rejection of the flagged queries, with two orderings that use no answer of the model:

  centre      a Gaussian around the centre of the image (standard deviation: a quarter of its side)
  regions     the explanation's own regions (level sets of the recorded map, if it has at most 256
              of them), ranked by the distance of their centroid from the centre of the image

The gain of an explanation over a prior is area(prior) - area(explanation). This analysis was made
after the results of X15 were known; no prediction is attached to it.
"""

from __future__ import annotations

import argparse
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

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402
from experiments.gate_trajectory_signatures.analyze_explanation_validity import LEVELS, areas, bootstrap_median, ordering  # noqa: E402
from experiments.gate_trajectory_signatures.run_explanation_clients import session_seed, smooth_random_saliency  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import load_cifar_model  # noqa: E402


def centre_gaussian(h: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:h].astype(np.float64)
    c = (h - 1) / 2.0
    return np.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * (h / 4.0) ** 2))


def region_centrality(sal: np.ndarray) -> np.ndarray | None:
    values, inverse = np.unique(sal, return_inverse=True)
    if not 2 <= len(values) <= LEVELS:
        return None
    inverse = inverse.reshape(sal.shape)
    h = sal.shape[0]
    y, x = np.mgrid[0:h, 0:h].astype(np.float64)
    c = (h - 1) / 2.0
    distance = np.sqrt((y - c) ** 2 + (x - c) ** 2)
    out = np.zeros(sal.shape, np.float64)
    for k in range(len(values)):
        region = inverse == k
        out[region] = -distance[region].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--enforced-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--splits", default="evaluation")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if args.dataset == "imagenet":
        from torchvision.models import ResNet50_Weights, resnet50

        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), net).to(device).eval()
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
        batch = 102  # a multiple of 17, so that every ordering is evaluated in batches of the same composition
    else:
        model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
        dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
        batch = 1020
    splits = {v.strip() for v in args.splits.split(",")}
    rows = sorted((r for r in blacklight_rule.load_sessions(args.source_dir) if r["split"] in splits), key=lambda r: r["session_id"])
    records = []
    for count, row in enumerate(rows):
        sid, index, label, client = row["session_id"], int(row["dataset_index"]), int(row["source_label"]), row["objective"]
        x = dataset[index][0][None]
        h = x.shape[-1]
        field = smooth_random_saliency(np.random.default_rng(session_seed(args.seed, index, "field")), h=h)
        recorded = np.load(args.source_dir / row["trace"])["saliency"].astype(np.float32)
        maps = {"recorded": recorded, "prior_centre": centre_gaussian(h)}
        regions = region_centrality(recorded)
        if regions is not None:
            maps["prior_regions"] = regions
        if args.enforced_dir is not None and (args.enforced_dir / "traces" / f"{sid}.npz").exists():
            trace = np.load(args.enforced_dir / "traces" / f"{sid}.npz")
            for mode in ("h1", "h2"):
                if f"saliency_{mode}" in trace.files:
                    maps[mode] = trace[f"saliency_{mode}"].astype(np.float32)
        names = list(maps)
        a = areas(model, device, x, [ordering(maps[n], field) for n in names], label, batch)
        record = {"session_id": sid, "dataset_index": index, "client": client}
        record |= {f"area_{n}": float(v) for n, v in zip(names, a)}
        record |= {f"distinct_{n}": int(len(np.unique(maps[n]))) for n in names if not n.startswith("prior")}
        records.append(record)
        if (count + 1) % 50 == 0:
            print(f"{count + 1}/{len(rows)}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "prior_sessions.csv", index=False)
    summary = []
    for client, g in frame.groupby("client"):
        for variant in ("recorded", "h1", "h2"):
            if f"area_{variant}" not in g or g[f"area_{variant}"].isna().all():
                continue
            entry = {"client": client, "variant": variant, "images": int(g[f"area_{variant}"].notna().sum()),
                     "area_median": float(g[f"area_{variant}"].median())}
            for prior in ("prior_centre", "prior_regions"):
                if f"area_{prior}" not in g:
                    continue
                pair = g[[f"area_{prior}", f"area_{variant}"]].dropna()
                if len(pair) < 10:
                    continue
                gain = (pair[f"area_{prior}"] - pair[f"area_{variant}"]).to_numpy()
                med, lo, hi = bootstrap_median(gain)
                entry |= {f"{prior}_images": len(pair), f"{prior}_area_median": float(pair[f"area_{prior}"].median()),
                          f"gain_over_{prior}": med, f"gain_over_{prior}_lo": lo, f"gain_over_{prior}_hi": hi,
                          f"better_than_{prior}": float((gain > 0).mean())}
            summary.append(entry)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "prior_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
