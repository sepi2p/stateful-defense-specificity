#!/usr/bin/env python3
"""Sensitivity of the Lee-Fang-Chang reimplementation to choices the paper leaves open (P15).

Variants of Phase 1 (Phase 2 is the frozen configuration throughout):
  frozen         deterministic quantization, salt in [0, N)  (configuration used for every corpus)
  salt_unit      deterministic quantization, salt in [0, 1) as the paper states
  literal_fixed  randomized rounding at the paper's rate (1/w)(x mod q)/q with one rounding threshold per
                 position drawn when the detector is created, salt in [0, 1)
  literal_fresh  the same rate with rounding redrawn for every query, salt in [0, 1)

Part A: the paper's own traffic types (SimBA and Square attacks; shuffled images; i.i.d. noise) on
calibration images. Part B: matched NES sessions (attack, restore, confidence_boost; both starts) on
evaluation images. All variants observe the same query streams.
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

from experiments.gate_trajectory_signatures.lfc_detector import LFCPhase1, phase2_alarm, phase2_batch  # noqa: E402
from experiments.gate_trajectory_signatures.lfc_faithfulness import simba_stream, square_stream  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_controls import build_stream  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_workloads import blacklight_salt, run_session, session_seed  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import corruption, load_cifar_model  # noqa: E402
from experiments.gate_trajectory_signatures.summarize_specificity_controls import wilson  # noqa: E402

VARIANTS = {
    "frozen": dict(bern_scale=0.0),
    "salt_unit": dict(bern_scale=0.0, salt_unit=True),
    "literal_fixed": dict(bern_scale=None, salt_unit=True, rounding="fixed", rate_unsalted=True),
    "literal_fresh": dict(bern_scale=None, salt_unit=True, rounding="fresh", rate_unsalted=True),
}
DETECTOR_SEED = 20260926


def observers(d: int) -> dict:
    return {f"lfcvar_{name}": LFCPhase1(d=d, seed=DETECTOR_SEED, **kw) for name, kw in VARIANTS.items()}


def score(obs: dict, logits: np.ndarray) -> dict:
    out = {}
    for name, det in obs.items():
        a = np.asarray(det.assignment)
        first = phase2_alarm(a, logits, lags="min10", retest="every")
        out[name.removeprefix("lfcvar_")] = {"first_alarm": first, "largest": int(np.bincount(a).max()),
                                             "largest_in_first_50": int(np.bincount(a[:50]).max()),
                                             "seq50_flag": bool(phase2_batch(a, logits, length=50)),
                                             "seq50_flag_lag5": bool(phase2_batch(a, logits, length=50, lags=5))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/lfc_sensitivity_20260927"))
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", ROOT / "checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt", device).eval()
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    pool = np.array(sorted(set(range(len(dataset))) - set(manifest.dataset_index.astype(int))))
    per_class = max(1, args.images // 10)
    rows = []

    # Part A: the paper's own traffic types, calibration images
    for row in manifest[manifest.split == "calibration"].groupby("source_label").head(per_class).itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0]
        x, label = clean[None].to(device), int(row.source_label)
        gen = torch.Generator(device=device).manual_seed(20260926 + int(row.dataset_index))
        streams = {"simba": simba_stream(model, x, label, 1024, gen).cpu(), "square": square_stream(model, x, label, 1024, gen).cpu()}
        for control in ("shuffled", "noise", "sweep"):
            streams[control] = build_stream(control, dataset, clean, pool, 20260926 + int(row.dataset_index))[0][:1024]
        # The paper's noise scale is ambiguous (its inputs live in [0, N]); sigma = 0.1 intensity levels keeps the
        # queries near-duplicates, which is what its Blacklight baseline (97/100 flagged) implies.
        g = torch.Generator().manual_seed(20260927 + int(row.dataset_index))
        streams["noise_small"] = (clean[None] + (0.1 / 255.0) * torch.randn((1024, *clean.shape), generator=g)).clamp(0, 1)
        for name, stream in streams.items():
            with torch.no_grad():
                logits = torch.cat([model(stream[i : i + 256].to(device)) for i in range(0, len(stream), 256)]).cpu().numpy()
            obs = observers(stream[0].numel())
            for q in stream.numpy():
                for det in obs.values():
                    det.add(q)
            for variant, r in score(obs, logits).items():
                rows.append({"part": "A", "image": int(row.dataset_index), "stream": name, "variant": variant, **r})
        print(f"[A] image {row.dataset_index}", flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "sensitivity_rows.csv", index=False)

    # Part B: matched NES sessions, evaluation images
    salt = blacklight_salt()
    for row in manifest[manifest.split == "evaluation"].groupby("source_label").head(per_class).itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0].unsqueeze(0).to(device)
        label = int(row.source_label)
        for workload, lam in (("denoise", 0.5), ("deblur", 1.0)):
            start = corruption(clean, workload, int(row.dataset_index), args.seed)
            for objective in ("attack", "restore", "confidence_boost"):
                seed = session_seed(args.seed, row.dataset_index, workload, objective, "nes", lam)
                obs = observers(start[0].numel())
                result, arrays = run_session(model, clean, start, label, workload, objective, lam, seed, 1024, device, None, salt,
                                             observe_detectors=False, extra_observers=obs)
                for variant, r in score(obs, arrays["logits"]).items():
                    rows.append({"part": "B", "image": int(row.dataset_index), "stream": f"{objective}", "start": workload,
                                 "variant": variant, "query_sha256": result["query_sha256"], **r})
        print(f"[B] image {row.dataset_index}", flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "sensitivity_rows.csv", index=False)

    frame = pd.DataFrame(rows)
    out = []
    for (part, stream, variant), g in frame.groupby(["part", "stream", "variant"]):
        alarm, within50 = g.first_alarm > 0, (g.first_alarm > 0) & (g.first_alarm <= 50)
        lo, hi = wilson(int(alarm.sum()), len(g))
        out.append({"part": part, "stream": stream, "variant": variant, "n": len(g), "alarm": float(alarm.mean()), "lo": lo, "hi": hi,
                    "alarm_within_50": float(within50.mean()), "seq50_flag": float(g.seq50_flag.mean()),
                    "seq50_flag_lag5": float(g.seq50_flag_lag5.mean()), "median_largest_in_first_50": float(g.largest_in_first_50.median()),
                    "median_first": float(g.first_alarm[alarm].median()) if alarm.any() else np.nan,
                    "median_largest_subsequence": float(g.largest.median())})
    table = pd.DataFrame(out)
    table.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    rate = table.set_index(["stream", "variant"]).alarm
    checks = {}
    for variant in VARIANTS:
        seq = table.set_index(["stream", "variant"]).seq50_flag
        faithful = bool(rate[("simba", variant)] >= 0.95 and rate[("square", variant)] >= 0.95
                        and rate[("shuffled", variant)] <= 0.05 and rate[("noise", variant)] <= 0.05)
        faithful_seq50 = bool(seq[("simba", variant)] >= 0.95 and seq[("square", variant)] >= 0.95
                              and max(seq[(c, variant)] for c in ("shuffled", "noise", "noise_small")) <= 0.05)
        benign = min(rate[("restore", variant)], rate[("confidence_boost", variant)])
        checks[variant] = {"reproduces_paper": faithful, "min_benign_optimizer_alarm": round(float(benign), 3),
                           "P15_holds": (bool(benign >= 0.90) if faithful else None),
                           "reproduces_paper_seq50_protocol": faithful_seq50,
                           "seq50_attack": round(float(seq[("attack", variant)]), 3),
                           "seq50_restore": round(float(seq[("restore", variant)]), 3),
                           "seq50_confidence_boost": round(float(seq[("confidence_boost", variant)]), 3)}
    (args.output_dir / "sensitivity_predictions.json").write_text(json.dumps(checks, indent=2))
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print(json.dumps(checks, indent=1))
    # replay check: part-B query streams must equal the corpus streams
    corpus = {json.loads(l)["session_id"]: json.loads(l)["query_sha256"]
              for p in (ROOT / "analysis_outputs/stateful_specificity_workloads_20260925").glob("sessions_shard*.jsonl") for l in p.read_text().splitlines() if l.strip()}
    b = frame[frame.part == "B"].drop_duplicates(["image", "stream", "start"])
    print("part-B sessions:", len(b), "| query streams found in the main corpus:", int(b.query_sha256.isin(set(corpus.values())).sum()))


if __name__ == "__main__":
    main()
