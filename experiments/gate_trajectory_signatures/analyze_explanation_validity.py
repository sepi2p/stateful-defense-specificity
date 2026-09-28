#!/usr/bin/env python3
"""X15, part 2: are the explanations informative (rule R2), without and with enforcement?

Rule R2 (docs/specificity_workloads_preregistration.md, X15): deletion area with per-channel mean fill
in sixteen steps; ties in the attribution are broken by a spatially smooth random field, the same
for an explanation and its references; the reference is a randomization of the explanation itself
that keeps its spatial structure (permutation of the values among the level sets of the map if it
has at most 256 distinct values, random flips and a circular shift otherwise), 20 draws.
d = mean reference area - area of the explanation. A constant map has d = 0 exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402
from experiments.gate_trajectory_signatures.run_explanation_clients import session_seed, smooth_random_saliency  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import load_cifar_model  # noqa: E402

STEPS, DRAWS, LEVELS = 16, 20, 256


def ordering(sal: np.ndarray, field: np.ndarray) -> np.ndarray:
    return np.lexsort((field.reshape(-1), -sal.reshape(-1).astype(np.float64)))


def randomize(sal: np.ndarray, rng) -> np.ndarray:
    values, inverse = np.unique(sal, return_inverse=True)
    if len(values) <= LEVELS:
        return rng.permutation(values)[inverse].reshape(sal.shape)
    out = sal
    if rng.random() < 0.5:
        out = out[::-1]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.roll(out, (int(rng.integers(sal.shape[0])), int(rng.integers(sal.shape[1]))), axis=(0, 1)).copy()


@torch.no_grad()
def areas(model, device, x: torch.Tensor, orders: list[np.ndarray], label: int, batch: int) -> np.ndarray:
    """Deletion area for each ordering."""
    h, w = x.shape[-2:]
    base = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    masks = np.ones((len(orders), STEPS + 1, h * w), np.float32)
    for k, order in enumerate(orders):
        for i in range(1, STEPS + 1):
            removed = order[int(round((i - 1) * h * w / STEPS)) : int(round(i * h * w / STEPS))]
            masks[k, i:, removed] = 0.0
    keep = torch.from_numpy(masks).view(-1, 1, h, w)
    out = []
    for i in range(0, len(keep), batch):
        k = keep[i : i + batch].to(device)
        images = k * x.to(device) + (1 - k) * base.to(device)
        out.append(F.softmax(model(images), dim=1)[:, label].float().cpu())
    p = torch.cat(out).view(len(orders), STEPS + 1).numpy().astype(np.float64)
    return np.trapezoid(p, dx=1.0 / STEPS, axis=1)


def top_share(order_a: np.ndarray, order_b: np.ndarray, share: float = 0.2) -> float:
    n = int(round(share * len(order_a)))
    return len(np.intersect1d(order_a[:n], order_b[:n])) / n


def bootstrap_median(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(values, len(values))) for _ in range(reps)]
    return float(np.median(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="sessions with recorded explanations")
    parser.add_argument("--enforced-dir", type=Path, default=None, help="output of run_enforced_explanations.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--splits", default="evaluation")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--imagenet-model", choices=["resnet50", "convnext_tiny"], default="resnet50")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--no-constant", action="store_true", help="skip the constant map (its d is 0 by construction)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if args.dataset == "imagenet":
        from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50

        net = (convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if args.imagenet_model == "convnext_tiny"
               else resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
        model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), net).to(device).eval()
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
        batch = 96
    else:
        model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
        dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
        batch = 1024
    splits = {v.strip() for v in args.splits.split(",")}
    rows = sorted((r for r in blacklight_rule.load_sessions(args.source_dir) if r["split"] in splits), key=lambda r: r["session_id"])
    if args.max_sessions > 0:
        rows = rows[: args.max_sessions]
    # sessions already analysed are read from sessions.csv; the summary is always written again
    enforced = {}
    if args.enforced_dir is not None:
        for path in sorted(args.enforced_dir.glob("sessions_shard*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    enforced[record["session_id"]] = record
    out_path = args.output_dir / "sessions.csv"
    records = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {r["session_id"] for r in records}
    for count, row in enumerate(rows):
        sid = row["session_id"]
        if sid in done:
            continue
        index, label, client = int(row["dataset_index"]), int(row["source_label"]), row["objective"]
        x = dataset[index][0][None]
        h = x.shape[-1]
        field = smooth_random_saliency(np.random.default_rng(session_seed(args.seed, index, "field")), h=h)
        maps = {"recorded": np.load(args.source_dir / row["trace"])["saliency"].astype(np.float32)}
        if not args.no_constant:
            maps["constant"] = np.zeros_like(maps["recorded"])
        info = enforced.get(sid)
        if info is not None:
            trace = np.load(args.enforced_dir / "traces" / f"{sid}.npz")
            for mode in info["modes"]:
                if mode != "replay":
                    maps[mode] = trace[f"saliency_{mode}"].astype(np.float32)
        reference_order = ordering(maps["recorded"], field)
        for variant, sal in maps.items():
            rng = np.random.default_rng(session_seed(args.seed, index, f"{client}|{variant}"))
            orders = [ordering(sal, field)] + [ordering(randomize(sal, rng), field) for _ in range(DRAWS)]
            a = areas(model, device, x, orders, label, batch)
            record = {"session_id": sid, "dataset_index": index, "client": client, "variant": variant,
                      "distinct_values": int(len(np.unique(sal))), "area": float(a[0]), "reference_area": float(a[1:].mean()),
                      "d": float(a[1:].mean() - a[0]), "queries": int(row["calls"])}
            if variant not in ("recorded", "constant"):
                constant = len(np.unique(sal)) == 1
                record["spearman"] = 0.0 if constant else float(spearmanr(sal.reshape(-1), maps["recorded"].reshape(-1)).statistic)
                record["top_fifth_overlap"] = top_share(orders[0], reference_order)
                record["answered"] = info["answered"]
                record["first_query_answered"] = info["first_query_answered"]
                record["stream_identical"] = info["modes"][variant]["stream_identical"]
            records.append(record)
        if (count + 1) % 10 == 0 or count + 1 == len(rows):
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"{count + 1}/{len(rows)} sessions", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out_path, index=False)
    summary = []
    for (client, variant), g in frame.groupby(["client", "variant"]):
        med, lo, hi = bootstrap_median(g.d.to_numpy())
        entry = {"client": client, "variant": variant, "images": len(g), "distinct_ge_10": float((g.distinct_values >= 10).mean()),
                 "distinct_median": float(g.distinct_values.median()), "queries_median": float(g.queries.median()),
                 "area_median": float(g.area.median()),
                 "reference_area_median": float(g.reference_area.median()), "d_median": med, "d_lo": lo, "d_hi": hi,
                 "informative": bool(med > 0 and lo > 0), "d_positive_share": float((g.d > 0).mean())}
        if variant not in ("recorded", "constant"):
            s = bootstrap_median(g.spearman.to_numpy())
            t = bootstrap_median(g.top_fifth_overlap.to_numpy())
            entry |= {"spearman_median": s[0], "spearman_lo": s[1], "spearman_hi": s[2], "top_fifth_median": t[0],
                      "top_fifth_lo": t[1], "top_fifth_hi": t[2], "answered_share": float((g.answered / g.queries).mean()),
                      "answered_median": float(g.answered.median()), "first_query_answered": float(g.first_query_answered.mean()),
                      "stream_identical": float(g.stream_identical.mean())}
        summary.append(entry)
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string())


if __name__ == "__main__":
    main()
