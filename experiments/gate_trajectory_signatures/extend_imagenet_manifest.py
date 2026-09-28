#!/usr/bin/env python3
"""X16: add a 'confirmation' split of fresh images to the ImageNet manifest of the study.

The manifest of X9 was drawn by prepare_manifest_generic (run_specificity_workloads.py): for each of the
100 classes, the validation images of the class in a seeded random order, of which the first five that
the model classifies correctly in their clean form and from both starts became development, fit,
calibration and two evaluation images. This script walks the same orders and takes, for each class, the
first image after the last of the five manifest images that the model classifies correctly in the same
way. Earlier images of the order are not judged again: whether an image near the decision boundary counts
as correctly classified can depend on the arithmetic of the device (on the CPU one manifest image of the
first classes does not), and the rule must not depend on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.run_specificity_workloads import corruption  # noqa: E402




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("analysis_outputs/specificity_imagenet_20260926/manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=0, help="testing only")
    args = parser.parse_args()
    from torchvision.models import ResNet50_Weights, resnet50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), net).to(device).eval()
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    labels = np.asarray(dataset.targets)
    classes = np.sort(np.random.default_rng(args.seed).choice(1000, args.classes, replace=False))
    old = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)
    rows = []
    for n, label in enumerate(classes):
        if args.max_classes and n >= args.max_classes:
            break
        candidates = list(rng.permutation(np.flatnonzero(labels == label)))
        mine = old[old.source_label == label].dataset_index.astype(int).tolist()
        if len(mine) != 5 or not set(mine) <= set(candidates):
            raise SystemExit(f"class {label}: manifest images not found in the seeded order")
        start = max(candidates.index(i) for i in mine) + 1
        chosen = None
        for index in candidates[start:]:
            clean = dataset[int(index)][0].unsqueeze(0).to(device)
            batch = torch.cat([clean, corruption(clean, "denoise", int(index), args.seed), corruption(clean, "deblur", int(index), args.seed)])
            with torch.no_grad():
                if bool(model(batch).argmax(1).eq(int(label)).all()):
                    chosen = int(index)
                    break
        if chosen is None:
            raise SystemExit(f"class {label}: no further qualifying image")
        rows.append({"dataset_index": chosen, "source_label": int(label), "split": "confirmation"})
        print(label, sorted(mine), chosen, flush=True)
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values(["split", "source_label", "dataset_index"])
    assert not new.dataset_index.duplicated().any()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(rows)} confirmation images")


if __name__ == "__main__":
    main()
