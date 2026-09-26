#!/usr/bin/env python3
"""X0: non-optimizing control streams for the stateful-specificity study.

Same session/trace format as run_specificity_workloads.py, so the analysis code
reads both. Controls (predictions frozen in docs/specificity_workloads_preregistration.md):
  shuffled  1,024 distinct CIFAR-10 test images outside the manifest (10 sessions per manifest image)
  noise     i.i.d. N(0, 0.1^2) around the clean manifest image (Lee-Fang-Chang's benign construction)
  sweep     JPEG quality 95->20 x brightness -0.2..+0.2 grid of the clean manifest image (32 x 32 = 1,024)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.io import decode_jpeg, encode_jpeg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.run_specificity_workloads import (  # noqa: E402
    BLACKLIGHT,
    BlacklightTracker,
    blacklight_salt,
)
from experiments.gate_trajectory_signatures.lfc_detector import LFCPhase1  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    GWAD_ROOT,
    Recorder,
    atomic_json,
    load_cifar_model,
    load_official_components,
    sha256,
)

CONTROLS = ("shuffled", "noise", "sweep")
SHUFFLED_PER_IMAGE = 10
QUERIES = 1024


def control_seed(base: int, dataset_index: int, control: str, replicate: int) -> int:
    text = f"{base}|{dataset_index}|{control}|{replicate}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def jpeg_brightness(image: torch.Tensor, quality: int, brightness: float) -> torch.Tensor:
    u8 = (image.clamp(0, 1) * 255).round().to(torch.uint8)
    decoded = decode_jpeg(encode_jpeg(u8, quality=int(quality))).float() / 255.0
    return (decoded + brightness).clamp(0, 1)


def build_stream(control: str, dataset, clean: torch.Tensor, pool: np.ndarray, seed: int) -> tuple[torch.Tensor, int | None]:
    """Returns the query stream (N x C x H x W, CPU) and, for shuffled streams, the first image's label."""
    if control == "shuffled":
        idx = np.random.default_rng(seed).choice(pool, QUERIES, replace=False)
        return torch.stack([dataset[int(i)][0] for i in idx]), int(dataset[int(idx[0])][1])
    if control == "noise":
        g = torch.Generator().manual_seed(seed)
        return (clean[None] + 0.1 * torch.randn((QUERIES, *clean.shape), generator=g)).clamp(0, 1), None
    if control == "sweep":
        qualities = np.linspace(95, 20, 32).round().astype(int)
        shifts = np.linspace(-0.2, 0.2, 32)
        return torch.stack([jpeg_brightness(clean, q, b) for q in qualities for b in shifts]), None
    raise ValueError(control)


def run_stream(model, device, delta_net, salt, stream: torch.Tensor, label: int, batch: int = 64, lfc_seed: int = -1,
               blacklight_params=None, lfc_params=None):
    recorder = Recorder.create(model, label, device, delta_net)
    blacklight = BlacklightTracker(salt, blacklight_params)
    recorder.detectors["blacklight"] = blacklight
    if lfc_seed >= 0:
        recorder.detectors["lfc_phase1"] = LFCPhase1(d=stream[0].numel(), seed=lfc_seed, params=lfc_params, bern_scale=0.0)
    for i in range(0, len(stream), batch):
        recorder.submit_batch(stream[i : i + batch].to(device))
    arrays = {
        "logits": np.asarray(recorder.logits, dtype=np.float32),
        "predictions": np.asarray(recorder.predictions, dtype=np.int16),
        "blacklight_counts": np.asarray(blacklight.counts, dtype=np.int16),
    }
    if lfc_seed >= 0:
        arrays["lfc_assignment"] = np.asarray(recorder.detectors["lfc_phase1"].assignment, dtype=np.int32)
        arrays["lfc_best_match"] = np.asarray(recorder.detectors["lfc_phase1"].best_match, dtype=np.int16)
    for name in ("gwad_plus", "gwad"):
        observations = recorder.detectors[name].detector.observations
        arrays[f"{name}_query_indices"] = np.asarray([r["query_index"] for r in observations], dtype=np.int16)
        arrays[f"{name}_scores"] = np.asarray([r["score"] for r in observations], dtype=np.float32)
        arrays[f"{name}_predictions"] = np.asarray([r["prediction"] for r in observations], dtype=np.int8)
        arrays[f"{name}_histograms"] = np.asarray([r["histogram"] for r in observations], dtype=np.float32).reshape((-1, 201))
    result = {
        "calls": recorder.calls,
        "first_success": recorder.first_success,
        "query_sha256": recorder.digest.hexdigest(),
        "detectors": {name: det.summary() for name, det in recorder.detectors.items()},
    }
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv"))
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--controls", default=",".join(CONTROLS))
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--lfc-seed", type=int, default=-1, help="add the Lee-Fang-Chang Phase-1 observer with this detector seed")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    pool = np.array(sorted(set(range(len(dataset))) - set(manifest.dataset_index.astype(int))))
    subset = manifest[manifest.split != "development"]
    if args.max_images > 0:
        subset = subset.head(args.max_images)
    subset = subset.iloc[args.shard :: args.num_shards]
    _, _, delta_net = load_official_components(torch.device("cpu"))
    salt = blacklight_salt()
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    atomic_json(args.output_dir / f"metadata_shard{args.shard}.json", {
        "args": {k: str(v) for k, v in vars(args).items()},
        "gwad_commit": __import__("subprocess").check_output(["git", "-C", str(GWAD_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "delta_net_sha256": sha256(GWAD_ROOT / "model/delta/delta_ann.pt"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "blacklight": BLACKLIGHT,
        "shuffled_per_image": SHUFFLED_PER_IMAGE,
    })
    summary_path = args.output_dir / f"sessions_shard{args.shard}.jsonl"
    completed = set()
    if summary_path.exists():
        completed = {json.loads(line)["session_id"] for line in summary_path.read_text().splitlines() if line.strip()}
    for row in subset.itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0]
        for control in controls:
            for replicate in range(SHUFFLED_PER_IMAGE if control == "shuffled" else 1):
                sid = f"{row.split}__{row.dataset_index}__clean__{control}__r{replicate}"
                if sid in completed:
                    continue
                seed = control_seed(args.seed, int(row.dataset_index), control, replicate)
                started = time.time()
                stream, first_label = build_stream(control, dataset, clean, pool, seed)
                label = first_label if first_label is not None else int(row.source_label)
                result, arrays = run_stream(model, device, delta_net, salt, stream, label, lfc_seed=args.lfc_seed)
                trace = args.output_dir / "traces" / f"{sid}.npz"
                trace.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trace, **arrays)
                record = {"session_id": sid, "split": row.split, "dataset_index": int(row.dataset_index),
                          "source_label": int(row.source_label), "workload": "clean", "objective": control,
                          "optimizer": "none", "replicate": replicate, "session_seed": seed,
                          "elapsed_seconds": time.time() - started,
                          "trace": str(trace.relative_to(args.output_dir))} | result
                with summary_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                d = result["detectors"]
                print(f"[DONE] {sid} bl={d['blacklight']['first_alarm']} gwad+={d['gwad_plus']['first_alarm']} "
                      f"t={record['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
