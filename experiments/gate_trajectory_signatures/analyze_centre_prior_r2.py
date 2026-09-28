#!/usr/bin/env python3
"""X15, exploratory addition: rule R2 applied to a map that uses no answer of the model.

A Gaussian around the centre of the image is evaluated by rule R2 exactly as an explanation is
(analyze_explanation_validity.py), on the evaluation images of the library clients. If it is
"informative" by R2, the rule does not show that the information of an explanation comes from the
model. Output: <output-dir>/centre_prior_r2.csv and centre_prior_r2_summary.json
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
from experiments.gate_trajectory_signatures.analyze_explanation_prior import centre_gaussian  # noqa: E402
from experiments.gate_trajectory_signatures.analyze_explanation_validity import DRAWS, areas, bootstrap_median, ordering, randomize  # noqa: E402
from experiments.gate_trajectory_signatures.run_explanation_clients import session_seed, smooth_random_saliency  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import load_cifar_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--splits", default="evaluation")
    parser.add_argument("--imagenet-model", choices=["resnet50", "convnext_tiny"], default="resnet50")
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.dataset == "imagenet":
        from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50

        net = (convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if args.imagenet_model == "convnext_tiny"
               else resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
        model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), net).to(device).eval()
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
        batch = 119
    else:
        model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
        dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
        batch = 357
    images = sorted({(int(r["dataset_index"]), int(r["source_label"])) for r in blacklight_rule.load_sessions(args.source_dir)
                     if r["split"] in {s.strip() for s in args.splits.split(",")}})
    records = []
    for index, label in images:
        x = dataset[index][0][None]
        h = x.shape[-1]
        field = smooth_random_saliency(np.random.default_rng(session_seed(args.seed, index, "field")), h=h)
        sal = centre_gaussian(h).astype(np.float32)
        rng = np.random.default_rng(session_seed(args.seed, index, "centre|recorded"))
        orders = [ordering(sal, field)] + [ordering(randomize(sal, rng), field) for _ in range(DRAWS)]
        a = areas(model, device, x, orders, label, batch)
        records.append({"dataset_index": index, "distinct_values": int(len(np.unique(sal))), "area": float(a[0]),
                        "reference_area": float(a[1:].mean()), "d": float(a[1:].mean() - a[0])})
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "centre_prior_r2.csv", index=False)
    med, lo, hi = bootstrap_median(frame.d.to_numpy())
    summary = {"images": len(frame), "distinct_values_median": float(frame.distinct_values.median()), "d_median": med, "d_lo": lo,
               "d_hi": hi, "informative": bool(med > 0 and lo > 0), "d_positive_share": float((frame.d > 0).mean())}
    (args.output_dir / "centre_prior_r2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
