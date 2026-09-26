#!/usr/bin/env python3
"""Verify that the local observer preserves released GWAD decisions."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
GWAD_ROOT = ROOT / "third_party/GWAD_official"
for path in (ROOT, GWAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    MU, STD, NullStats, TrackingGWAD, load_official_components,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    args = parser.parse_args()
    device = torch.device("cpu")
    GWAD, Query, delta_net = load_official_components(device)
    cfg = {
        "screen": {"on": True, "n": 50, "thold": 0.4},
        "model": {"data_mu": [0.4914, 0.4822, 0.4465], "data_std": [0.247, 0.243, 0.261], "data_format": "tensor"},
    }
    original = GWAD(device, cfg, NullStats(), mode="simulate", model=None, delta_net=delta_net)
    observer = TrackingGWAD(True, device, delta_net)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    base = dataset[0][0].unsqueeze(0)
    generator = torch.Generator().manual_seed(20260924)
    stream = []
    for index in range(600):
        noise = torch.randn(base.shape, generator=generator) * min(index, 32) / (32 * 255.0)
        stream.append((base + noise).clamp(0, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        for query_index, raw in enumerate(stream, 1):
            normalized = (raw - MU) / STD
            original.query_cnt = query_index
            original.run(Query(t="attack", x=normalized))
            observer.submit(raw)
    original_counts = original.hx.numpy().astype(int).tolist()
    observed_counts = observer.detector.hx.numpy().astype(int).tolist()
    parity = original_counts == observed_counts

    # Two release-format smoke streams: correlated attack-shaped queries and
    # independent benign images. These verify warm-up and screening behavior;
    # they are not claimed as a reproduction of a published performance table.
    correlated = observer.summary()
    benign = TrackingGWAD(True, device, delta_net)
    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(600):
            benign.submit(dataset[(index * 17) % len(dataset)][0].unsqueeze(0))
    payload = {
        "status": "PASS" if parity and correlated["eligible_windows"] > 0 else "BLOCKED",
        "decision_parity": parity,
        "original_class_counts": original_counts,
        "observer_class_counts": observed_counts,
        "correlated_stream": correlated,
        "independent_benign_stream": benign.summary(),
        "queries_per_stream": 600,
        "scope": "adapter parity and release-format smoke, not full published-table reproduction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 21)


if __name__ == "__main__":
    main()
