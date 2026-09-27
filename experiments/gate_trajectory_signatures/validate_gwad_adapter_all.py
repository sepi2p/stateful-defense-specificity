#!/usr/bin/env python3
"""Decision parity of the observer wrapper with the released GWAD class, with and without the
screening stage, on a stream of near-duplicates and on a stream of unrelated images (600 queries each).

Extends validate_official_gwad_adapter.py, which covers GWAD+ on the near-duplicate stream.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

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
    device = torch.device("cpu")
    GWAD, Query, delta_net = load_official_components(device)
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    base = dataset[0][0].unsqueeze(0)
    generator = torch.Generator().manual_seed(20260924)
    near = [(base + torch.randn(base.shape, generator=generator) * min(i, 32) / (32 * 255.0)).clamp(0, 1) for i in range(600)]
    unrelated = [dataset[(i * 17) % len(dataset)][0].unsqueeze(0) for i in range(600)]
    out = {}
    for screen in (True, False):
        for name, stream in (("near_duplicates", near), ("unrelated", unrelated)):
            cfg = {"screen": {"on": screen, "n": 50, "thold": 0.4},
                   "model": {"data_mu": [0.4914, 0.4822, 0.4465], "data_std": [0.247, 0.243, 0.261], "data_format": "tensor"}}
            original = GWAD(device, cfg, NullStats(), mode="simulate", model=None, delta_net=delta_net)
            observer = TrackingGWAD(screen, device, delta_net)
            with contextlib.redirect_stdout(io.StringIO()):
                for index, raw in enumerate(stream, 1):
                    original.query_cnt = index
                    original.run(Query(t="attack", x=(raw - MU) / STD))
                    observer.submit(raw)
            a = original.hx.numpy().astype(int).tolist()
            b = observer.detector.hx.numpy().astype(int).tolist()
            out[f"{'gwad_plus' if screen else 'gwad'}|{name}"] = {"released_class_counts": a, "wrapper_class_counts": b, "identical": a == b,
                                                                  "wrapper_summary": observer.summary()}
    out["all_identical"] = all(v["identical"] for v in out.values())
    path = ROOT / "analysis_outputs/stateful_specificity_pretests_20260925/gwad_adapter_parity_all.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
