#!/usr/bin/env python3
"""Clean test accuracy of every classifier used in the paper, with the paper's preprocessing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.common import load_cifar_model  # noqa: E402


@torch.no_grad()
def accuracy(model, loader, device):
    correct = total = 0
    for x, y in loader:
        correct += int((model(x.to(device)).argmax(1).cpu() == y).sum())
        total += len(y)
    return correct / total, total


def main():
    device = torch.device("cuda")
    out = {}
    cifar = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    loader = DataLoader(cifar, batch_size=250, num_workers=4)
    for name, ckpt in [("resnet18_seed0", "checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"),
                       ("resnet18_seed1", "checkpoints/cifar10_resnet18_seed_study/resnet18_seed1.pt"),
                       ("resnet18_seed2", "checkpoints/cifar10_resnet18_seed_study/resnet18_seed2.pt"),
                       ("bbb_vgg19_bn", None), ("robustbench_Engstrom2019Robustness", None)]:
        model = load_cifar_model(name, Path(ckpt) if ckpt else None, device).eval()
        out[f"cifar10/{name}"] = accuracy(model, loader, device)
        print(name, out[f"cifar10/{name}"], flush=True)
        del model
        torch.cuda.empty_cache()
    from experiments.eaai_gtsrb.gtsrb_common import gtsrb_dataset, load_checkpoint
    model, _ = load_checkpoint("analysis_outputs/eaai_gtsrb/checkpoints_32px/resnet18/best.pt", device)
    out["gtsrb32/resnet18"] = accuracy(model, DataLoader(gtsrb_dataset("data/gtsrb", "test", 32), batch_size=250, num_workers=4), device)
    print("gtsrb", out["gtsrb32/resnet18"], flush=True)
    del model
    from torchvision.models import ResNet50_Weights, resnet50
    model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)).to(device).eval()
    imagenet = datasets.ImageFolder("/home/sepi/Study/coding/data/imagenet/val", transform=transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    out["imagenet/resnet50_v1_full_val"] = accuracy(model, DataLoader(imagenet, batch_size=100, num_workers=8), device)
    classes = np.sort(np.random.default_rng(20260924).choice(1000, 100, replace=False))
    idx = [i for i, t in enumerate(imagenet.targets) if t in set(classes.tolist())]
    out["imagenet/resnet50_v1_100_classes"] = accuracy(model, DataLoader(torch.utils.data.Subset(imagenet, idx), batch_size=100, num_workers=8), device)
    print("imagenet", out["imagenet/resnet50_v1_full_val"], out["imagenet/resnet50_v1_100_classes"], flush=True)
    Path("paper/jisa_2026/numbers").mkdir(parents=True, exist_ok=True)
    Path("paper/jisa_2026/numbers/model_accuracies.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
