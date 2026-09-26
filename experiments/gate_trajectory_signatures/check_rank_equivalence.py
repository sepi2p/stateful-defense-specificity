#!/usr/bin/env python3
"""Empirical check of Proposition 1 (docs/specificity_theory_notes.md).

SimBA is comparison-based. With the margin objective, the attack (accept any
margin decrease) and the benign boundary probe (accept a margin decrease only if
the label is kept) must issue identical queries and receive identical outputs
until the attack's first label change. NES, which uses objective values, is run
as a contrast between the attack and the boundary probe as defined in the corpus.
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

from experiments.gate_trajectory_signatures.run_specificity_workloads import margin  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    corruption,
    load_cifar_model,
    project,
)


@torch.no_grad()
def simba_margin(model, start, label, keep_label, seed, budget, eps=8 / 255, step=1 / 255):
    generator = torch.Generator(device=start.device).manual_seed(seed)
    order = torch.randperm(start[0].numel(), generator=generator, device=start.device)
    queries, outputs = [start.clone()], [model(start)]
    current, value = start.clone(), margin(outputs[0], label)[0]
    first_flip = -1
    for it in range(budget // 2):
        delta = torch.zeros_like(current).flatten()
        delta[int(order[it % len(order)])] = step
        delta = delta.view_as(current)
        candidates = torch.cat([project(current - delta, start, eps), project(current + delta, start, eps)])
        logits = model(candidates)
        queries += [candidates[:1], candidates[1:]]
        outputs += [logits[:1], logits[1:]]
        if first_flip < 0 and bool((logits.argmax(1) != label).any()):
            first_flip = len(queries) - 1 - int(logits[1].argmax() != label and logits[0].argmax() == label)
        values = margin(logits, label)
        best = None
        for i in range(2):
            if keep_label and int(logits[i].argmax()) != label:
                continue
            if values[i] < value and (best is None or values[i] < values[best]):
                best = i
        if best is not None:
            current, value = candidates[best : best + 1], values[best]
    return torch.cat(queries), torch.cat(outputs), first_flip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv")
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--step-255", type=float, default=8.0)
    parser.add_argument("--output", default="analysis_outputs/stateful_specificity_workloads_20260925/rank_equivalence_simba.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"), device).eval()
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    rows = []
    for row in manifest[manifest.split == "evaluation"].head(args.images).itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0].unsqueeze(0).to(device)
        for workload in ("denoise", "deblur"):
            start = corruption(clean, workload, int(row.dataset_index), 20260924)
            seed = int(row.dataset_index) * 7919 + (workload == "deblur")
            qa, oa, flip = simba_margin(model, start, int(row.source_label), False, seed, args.budget, step=args.step_255 / 255)
            qb, ob, _ = simba_margin(model, start, int(row.source_label), True, seed, args.budget, step=args.step_255 / 255)
            n = min(len(qa), len(qb))
            same = [(torch.equal(qa[i], qb[i]) and torch.equal(oa[i], ob[i])) for i in range(n)]
            first_diff = next((i for i, s in enumerate(same) if not s), -1)
            rows.append({"dataset_index": int(row.dataset_index), "workload": workload, "attack_first_flip": flip,
                         "first_divergence": first_diff, "calls": n,
                         "identical_before_flip": bool(all(same[: flip if flip >= 0 else n]))})
    frame = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    flipped = frame[frame.attack_first_flip >= 0]
    print(f"sessions={len(frame)} attack_flipped={len(flipped)} "
          f"identical_before_flip={frame.identical_before_flip.mean():.3f} "
          f"divergence_after_or_at_flip={(flipped.first_divergence >= flipped.attack_first_flip).mean():.3f}")
    print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
