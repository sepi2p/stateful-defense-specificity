#!/usr/bin/env python3
"""X2: explanation-traffic clients against stateful detectors.

Each session is the query stream a black-box explanation client sends to explain
the model's prediction on one clean manifest image. GWAD, GWAD+, Blacklight and
the Lee-Fang-Chang Phase-1 observer watch the stream online (same trace format
as the other corpora). Explanation utility is measured afterwards with model
calls that are NOT part of the detector stream: a deletion curve of p(label)
in saliency order vs random pixel orders. Predictions: docs/specificity_workloads_preregistration.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.pretest_detector_assumptions import lime_session, lime_weights  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_controls import run_stream  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_workloads import blacklight_salt  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    atomic_json,
    load_cifar_model,
    load_official_components,
)

CLIENTS = ("lime", "kernelshap", "occlusion", "rise")


def blur(x: torch.Tensor) -> torch.Tensor:
    """Strong blur used as the 'removed' fill (two 5x5 box passes, reflect padding)."""
    for _ in range(2):
        x = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    return x


@torch.no_grad()
def probs(model, device, images: torch.Tensor, label: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), 256):
        out.append(F.softmax(model(images[i : i + 256].to(device)), 1)[:, label].cpu())
    return torch.cat(out).numpy().astype(np.float64)


# ---------------- clients: each returns (query stream, function(p_label) -> pixel saliency) -------------

def client_lime(x, rng):
    queries, masks, segments, _means = lime_session(x, 40, 1000, rng)
    return queries, lambda p: lime_weights(masks, p)[segments]


def client_kernelshap(x, rng, grid=4, samples=500):
    m = grid * grid
    cell = x.shape[-1] // grid
    baseline = blur(x)
    sizes = np.arange(1, m)
    kernel = (m - 1) / (sizes * (m - sizes))  # Shapley kernel mass per coalition size (up to binomial factor)
    kernel = kernel / kernel.sum()
    coalitions = [np.ones(m, bool), np.zeros(m, bool)]
    while len(coalitions) < samples:
        k = int(rng.choice(sizes, p=kernel))
        z = np.zeros(m, bool)
        z[rng.choice(m, k, replace=False)] = True
        coalitions.append(z)
    z = np.stack(coalitions)
    keep = torch.from_numpy(z.reshape(-1, 1, grid, grid).astype(np.float32))
    keep = F.interpolate(keep, scale_factor=cell, mode="nearest")
    queries = keep * x + (1 - keep) * baseline

    def saliency(p):
        # KernelSHAP weights: huge weight on the full/empty coalitions enforces efficiency
        s = z.sum(1)
        w = np.where((s == 0) | (s == m), 1e6, (m - 1) / (np.array([math.comb(m, int(k)) for k in s]) * s * (m - s) + 1e-12))
        a = np.hstack([z.astype(np.float64), np.ones((len(z), 1))])
        coef = np.linalg.lstsq(a * np.sqrt(w)[:, None], p * np.sqrt(w), rcond=None)[0][:-1]
        return np.kron(coef.reshape(grid, grid), np.ones((cell, cell)))

    return queries, saliency


def client_occlusion(x, rng, patch=4):
    h, w = x.shape[-2:]
    fill = x.mean(dim=(2, 3), keepdim=True)
    positions = [(r, c) for r in range(h - patch + 1) for c in range(w - patch + 1)]
    queries = [x.clone()]
    for r, c in positions:
        q = x.clone()
        q[:, :, r : r + patch, c : c + patch] = fill
        queries.append(q)
    queries = torch.cat(queries)

    def saliency(p):
        total, count = np.zeros((h, w)), np.zeros((h, w))
        for (r, c), pv in zip(positions, p[1:]):
            total[r : r + patch, c : c + patch] += p[0] - pv
            count[r : r + patch, c : c + patch] += 1
        return total / np.maximum(count, 1)

    return queries, saliency


def client_rise(x, rng, n=1000, s=7, p1=0.5):
    h = x.shape[-1]
    cell = math.ceil(h / s)
    up = (s + 1) * cell
    grid = torch.from_numpy((rng.random((n, 1, s, s)) < p1).astype(np.float32))
    big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
    shifts = rng.integers(0, cell, size=(n, 2))
    masks = torch.stack([big[i, :, dx : dx + h, dy : dy + h] for i, (dx, dy) in enumerate(shifts)])
    queries = masks * x
    m = masks[:, 0].numpy().astype(np.float64)
    return queries, lambda p: np.tensordot(p, m, axes=1) / (n * p1)


BUILDERS = {"lime": client_lime, "kernelshap": client_kernelshap, "occlusion": client_occlusion, "rise": client_rise}


def smooth_random_saliency(rng, h: int = 32, grid: int = 4) -> np.ndarray:
    """Random saliency that removes coherent blobs, like real explanations (primary utility baseline)."""
    g = torch.from_numpy(rng.random((1, 1, grid, grid)).astype(np.float32))
    return F.interpolate(g, size=(h, h), mode="bilinear", align_corners=False)[0, 0].numpy() + 1e-3 * rng.random((h, h))


def deletion_auc(model, device, x, sal, label, rng, steps=16, fill="mean") -> float:
    h, w = sal.shape
    order = np.lexsort((rng.random(h * w), -sal.reshape(-1)))  # descending saliency, random tie-break
    base = blur(x) if fill == "blur" else x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    frames = []
    keep = np.ones(h * w, np.float32)
    frames.append(keep.copy())
    for i in range(1, steps + 1):
        keep[order[: int(round(i * h * w / steps))]] = 0.0
        frames.append(keep.copy())
    k = torch.from_numpy(np.stack(frames)).view(-1, 1, h, w)
    p = probs(model, device, k * x + (1 - k) * base, label)
    return float(np.trapezoid(p, dx=1.0 / steps))


def session_seed(base: int, dataset_index: int, client: str) -> int:
    text = f"{base}|{dataset_index}|{client}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--clients", default=",".join(CLIENTS))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--lfc-seed", type=int, default=20260926)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    subset = manifest[manifest.split != "development"]
    if args.max_images > 0:
        subset = subset.head(args.max_images)
    subset = subset.iloc[args.shard :: args.num_shards]
    _, _, delta_net = load_official_components(torch.device("cpu"))
    salt = blacklight_salt()
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    atomic_json(args.output_dir / f"metadata_shard{args.shard}.json", {"args": {k: str(v) for k, v in vars(args).items()}})
    summary_path = args.output_dir / f"sessions_shard{args.shard}.jsonl"
    completed = set()
    if summary_path.exists():
        completed = {json.loads(line)["session_id"] for line in summary_path.read_text().splitlines() if line.strip()}
    for row in subset.itertuples(index=False):
        x = dataset[int(row.dataset_index)][0][None]
        label = int(row.source_label)
        for client in clients:
            sid = f"{row.split}__{row.dataset_index}__clean__{client}__r0"
            if sid in completed:
                continue
            seed = session_seed(args.seed, int(row.dataset_index), client)
            rng = np.random.default_rng(seed)
            started = time.time()
            queries, saliency_fn = BUILDERS[client](x, rng)
            result, arrays = run_stream(model, device, delta_net, salt, queries, label, lfc_seed=args.lfc_seed)
            p = probs(model, device, queries, label)
            sal = saliency_fn(p)
            auc = deletion_auc(model, device, x, sal, label, rng, fill="mean")
            auc_random = float(np.mean([deletion_auc(model, device, x, smooth_random_saliency(rng), label, rng, fill="mean") for _ in range(10)]))
            auc_blur = deletion_auc(model, device, x, sal, label, rng, fill="blur")
            auc_random_pixel_blur = float(np.mean([deletion_auc(model, device, x, rng.random(sal.shape), label, rng, fill="blur") for _ in range(10)]))
            trace = args.output_dir / "traces" / f"{sid}.npz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace, **arrays, saliency=sal.astype(np.float32))
            record = {"session_id": sid, "split": row.split, "dataset_index": int(row.dataset_index),
                      "source_label": label, "workload": "clean", "objective": client, "optimizer": "none",
                      "session_seed": seed, "elapsed_seconds": time.time() - started,
                      "deletion_auc": auc, "deletion_auc_random": auc_random,
                      "deletion_auc_blur": auc_blur, "deletion_auc_random_pixel_blur": auc_random_pixel_blur,
                      "trace": str(trace.relative_to(args.output_dir))} | result
            with summary_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            d = result["detectors"]
            print(f"[DONE] {sid} q={result['calls']} bl={d['blacklight']['first_alarm']} gwad+={d['gwad_plus']['first_alarm']} "
                  f"del {auc:.3f}/{auc_random:.3f} t={record['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
