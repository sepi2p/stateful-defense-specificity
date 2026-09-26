#!/usr/bin/env python3
"""Generate the frozen stateful-specificity paper-gate sessions.

This runner is deliberately separate from the historical factorial generator.
It accounts for every submitted query, streams each query through the released
GWAD implementation, and records deterministic replay metadata plus query
digests instead of rounding queries or retaining a multi-gigabyte tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
GWAD_ROOT = ROOT / "third_party/GWAD_official"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GWAD_ROOT) not in sys.path:
    sys.path.insert(0, str(GWAD_ROOT))

from experiments.gate_trajectory_signatures.common import load_cifar_model  # noqa: E402

MU = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
STRENGTHS = {"denoise": (0.3, 0.5, 0.7), "deblur": (0.3, 0.5, 1.0)}
SPLIT_COUNTS = {"development": 2, "fit": 10, "calibration": 10, "evaluation": 20}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corruption(clean: torch.Tensor, kind: str, dataset_index: int, seed: int) -> torch.Tensor:
    if kind == "denoise":
        generator = torch.Generator(device=clean.device)
        generator.manual_seed(seed + 104729 * int(dataset_index))
        noise = torch.randn(clean.shape, generator=generator, device=clean.device)
        return (clean + (16.0 / 255.0) * noise).clamp(0, 1)
    if kind == "deblur":
        mean = F.avg_pool2d(F.pad(clean, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        return 0.5 * clean + 0.5 * mean
    raise ValueError(kind)


def restoration_target(observed: torch.Tensor, workload: str, strength: float) -> torch.Tensor:
    mean = F.avg_pool2d(F.pad(observed, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    if workload == "denoise":
        return ((1.0 - strength) * observed + strength * mean).clamp(0, 1)
    return (observed + strength * (observed - mean)).clamp(0, 1)


def restoration_loss(image: torch.Tensor, observed: torch.Tensor, workload: str, strength: float) -> torch.Tensor:
    target = restoration_target(observed, workload, strength)
    return (image - target).square().mean((1, 2, 3))


def psnr(image: torch.Tensor, clean: torch.Tensor) -> float:
    mse = float((image - clean).square().mean().item())
    return float(10.0 * np.log10(1.0 / max(mse, 1e-15)))


def prepare_manifest(path: Path, dataset, model, device, seed: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    eligible = {label: [] for label in range(10)}
    for start in range(0, len(dataset), 128):
        stop = min(len(dataset), start + 128)
        clean = torch.stack([dataset[i][0] for i in range(start, stop)]).to(device)
        labels = torch.tensor([dataset[i][1] for i in range(start, stop)], device=device)
        with torch.no_grad():
            clean_ok = model(clean).argmax(1).eq(labels)
        for offset in range(stop - start):
            if not clean_ok[offset]:
                continue
            index = start + offset
            one = clean[offset : offset + 1]
            starts = torch.cat([
                corruption(one, "denoise", index, seed),
                corruption(one, "deblur", index, seed),
            ])
            with torch.no_grad():
                ok = model(starts).argmax(1).eq(labels[offset]).all().item()
            if ok:
                eligible[int(labels[offset])].append(index)
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(10):
        need = sum(SPLIT_COUNTS.values())
        values = np.asarray(eligible[label], dtype=int)
        if len(values) < need:
            raise RuntimeError(f"class {label}: only {len(values)} eligible, need {need}")
        values = rng.permutation(values)[:need]
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            for index in values[cursor : cursor + count]:
                rows.append({"dataset_index": int(index), "source_label": label, "split": split})
            cursor += count
    frame = pd.DataFrame(rows).sort_values(["split", "source_label", "dataset_index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


class NullStats:
    def __init__(self):
        self.classes = []
        self.predictions = []

    def reset(self):
        return None

    def update(self, *_args):
        return None


def load_official_components(device: torch.device):
    from net.delta.ann_hist.ann_hist import ANN_HIST
    from gwad import GWAD
    from algorithm.attack.base import Query

    model = ANN_HIST(len_op=7)
    state = torch.load(GWAD_ROOT / "model/delta/delta_ann.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.classes = ["benign", "hsja", "nes", "simba", "sign-opt", "sign-flip", "ba"]
    model.eval().to(device)
    return GWAD, Query, model


class TrackingGWAD:
    """Thin observer around the unmodified released GWAD state machine."""

    def __init__(self, screen: bool, device: torch.device, delta_net):
        GWAD, Query, _ = load_official_components(device)
        self.Query = Query
        cfg = {
            "screen": {"on": screen, "n": 50, "thold": 0.4},
            "model": {
                "data_mu": [0.4914, 0.4822, 0.4465],
                "data_std": [0.247, 0.243, 0.261],
                "data_format": "tensor",
            },
        }

        class Observer(GWAD):
            def attack_detect(inner):
                if not inner.dss_ready():
                    return False
                m0, m1 = torch.min(inner.hods_vector), torch.max(inner.hods_vector)
                if float(m1 - m0) == 0.0:
                    hods = torch.zeros_like(inner.hods_vector)
                else:
                    hods = (inner.hods_vector - m0) / (m1 - m0)
                with torch.no_grad():
                    logp = inner.delta_net(hods)
                probability = logp.exp()[0]
                prediction = int(logp.argmax(1).item())
                score = float(1.0 - probability[0].item())
                alarm = prediction != 0
                inner.observations.append({
                    "query_index": int(inner.query_cnt),
                    "score": score,
                    "prediction": prediction,
                    "alarm": int(alarm),
                    "histogram": hods.detach().cpu().numpy()[0].astype(np.float32),
                })
                inner.hx[prediction] += 1
                return alarm

        self.detector = Observer(device, cfg, NullStats(), mode="simulate", model=None, delta_net=delta_net)
        self.detector.observations = []
        self.query_count = 0

    def submit(self, raw: torch.Tensor) -> None:
        self.query_count += 1
        normalized = ((raw.detach().cpu() - MU) / STD).float()
        self.detector.query_cnt = self.query_count
        self.detector.run(self.Query(t="attack", x=normalized))

    def summary(self):
        obs = self.detector.observations
        return {
            "eligible_windows": len(obs),
            "max_score": max((row["score"] for row in obs), default=np.nan),
            "first_alarm": next((row["query_index"] for row in obs if row["alarm"]), -1),
        }


@dataclass
class Recorder:
    model: torch.nn.Module
    label: int
    detectors: dict
    digest: object
    logits: list
    predictions: list
    features: list
    feature_module: torch.nn.Module | None
    first_success: int
    calls: int

    @classmethod
    def create(cls, model, label, device, delta_net, observe_detectors=True, capture_features=False):
        return cls(
            model=model,
            label=label,
            detectors={
                "gwad_plus": TrackingGWAD(True, torch.device("cpu"), delta_net),
                "gwad": TrackingGWAD(False, torch.device("cpu"), delta_net),
            } if observe_detectors else {},
            digest=hashlib.sha256(),
            logits=[],
            predictions=[],
            features=[],
            feature_module=model.layer4 if capture_features else None,
            first_success=-1,
            calls=0,
        )

    @torch.no_grad()
    def submit_batch(self, images: torch.Tensor) -> torch.Tensor:
        captured = []
        handle = None
        if self.feature_module is not None:
            handle = self.feature_module.register_forward_hook(
                lambda _module, _inputs, value: captured.append(
                    F.adaptive_avg_pool2d(value.detach(), 1).flatten(1).cpu()
                )
            )
        try:
            output = self.model(images)
        finally:
            if handle is not None:
                handle.remove()
        feature_batch = captured[0] if captured else [None] * len(images)
        for image, logits, feature in zip(images, output, feature_batch):
            self.calls += 1
            cpu_image = image.detach().contiguous().float().cpu()
            self.digest.update(cpu_image.numpy().tobytes())
            for detector in self.detectors.values():
                detector.submit(cpu_image.unsqueeze(0))
            values = logits.detach().float().cpu().numpy()
            self.logits.append(values)
            if feature is not None:
                self.features.append(feature.numpy().astype(np.float32, copy=False))
            prediction = int(np.argmax(values))
            self.predictions.append(prediction)
            if prediction != self.label and self.first_success < 0:
                self.first_success = self.calls
        return output


def project(image: torch.Tensor, start: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.maximum(torch.minimum(image, start + eps), start - eps).clamp(0, 1)


def values_for(images, logits, observed, workload, objective, lam, label):
    if objective == "attack":
        labels = torch.full((len(images),), label, device=images.device, dtype=torch.long)
        return F.cross_entropy(logits, labels, reduction="none"), "max"
    return restoration_loss(images, observed.expand_as(images), workload, lam), "min"


def better(candidate, current, direction):
    return candidate > current if direction == "max" else candidate < current


def run_session(model, clean, observed, label, workload, objective, optimizer, lam, seed, budget, device, delta_net, observe_detectors=True, capture_features=False):
    recorder = Recorder.create(model, label, device, delta_net, observe_detectors, capture_features)
    current = observed.clone()
    current_logits = recorder.submit_batch(current)
    current_value, direction = values_for(current, current_logits, observed, workload, objective, lam, label)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    eps = 8.0 / 255.0
    step = (1.0 if optimizer == "simba" else 0.25) / 255.0
    accepted = []
    iteration = 0
    dimension = current[0].numel()
    simba_dimension = current[0].numel() if workload == "denoise" else current.shape[1] * 8 * 8
    order = torch.randperm(simba_dimension, generator=generator, device=device)
    while recorder.calls < budget:
        if optimizer == "random_walk":
            delta = torch.where(
                torch.rand(current.shape, generator=generator, device=device) < 0.5,
                -torch.ones_like(current), torch.ones_like(current),
            )
            candidate = project(current + step * delta, observed, eps)
            recorder.submit_batch(candidate)
            current = candidate
            accepted.append(1)
            iteration += 1
            continue
        if optimizer == "simba":
            if recorder.calls + 2 > budget:
                break
            coordinate = int(order[iteration % simba_dimension].item())
            if workload == "denoise":
                delta = torch.zeros_like(current).flatten()
                delta[coordinate] = step
                delta = delta.view_as(current)
            else:
                low = torch.zeros((1, current.shape[1], 8, 8), device=device)
                low.flatten()[coordinate] = 1.0
                delta = step * F.interpolate(
                    low, size=current.shape[-2:], mode="bilinear", align_corners=False
                )
            candidates = torch.cat([project(current - delta, observed, eps), project(current + delta, observed, eps)])
            logits = recorder.submit_batch(candidates)
            values, direction = values_for(candidates, logits, observed, workload, objective, lam, label)
            choices = [i for i in range(2) if objective == "attack" or int(logits[i].argmax()) == label]
            best = None
            for i in choices:
                if better(values[i], current_value[0], direction) and (best is None or better(values[i], values[best], direction)):
                    best = i
            if best is not None:
                current = candidates[best : best + 1]
                current_logits = logits[best : best + 1]
                current_value = values[best : best + 1]
            accepted.append(-1 if best is None else int(best))
            iteration += 1
            continue
        if optimizer != "nes":
            raise ValueError(optimizer)
        pairs = 8
        needed = 2 * pairs + 1
        if recorder.calls + needed > budget:
            break
        directions = torch.randn((pairs, *current.shape[1:]), generator=generator, device=device)
        sigma = 2.0 / 255.0
        negative = project(current - sigma * directions, observed, eps)
        positive = project(current + sigma * directions, observed, eps)
        proposals = torch.cat([negative, positive])
        proposal_logits = recorder.submit_batch(proposals)
        proposal_values, direction = values_for(proposals, proposal_logits, observed, workload, objective, lam, label)
        coefficients = proposal_values[pairs:] - proposal_values[:pairs]
        if direction == "min":
            coefficients = -coefficients
        estimate = (coefficients.view(pairs, 1, 1, 1) * directions).mean(0, keepdim=True)
        candidate = project(current + step * estimate.sign(), observed, eps)
        candidate_logits = recorder.submit_batch(candidate)
        candidate_value, direction = values_for(candidate, candidate_logits, observed, workload, objective, lam, label)
        allowed = objective == "attack" or int(candidate_logits.argmax()) == label
        take = bool(allowed and better(candidate_value[0], current_value[0], direction))
        if take:
            current, current_logits, current_value = candidate, candidate_logits, candidate_value
        accepted.append(int(take))
        iteration += 1
    result = {
        "calls": recorder.calls,
        "iterations": iteration,
        "first_success": recorder.first_success,
        "final_prediction": int(current_logits.argmax().item()),
        "label_preserved": int(current_logits.argmax().item() == label),
        "start_psnr": psnr(observed, clean),
        "final_psnr": psnr(current, clean),
        "psnr_gain": psnr(current, clean) - psnr(observed, clean),
        "query_sha256": recorder.digest.hexdigest(),
        "accepted": accepted,
        "detectors": {name: detector.summary() for name, detector in recorder.detectors.items()},
    }
    arrays = {
        "logits": np.asarray(recorder.logits, dtype=np.float32),
        "predictions": np.asarray(recorder.predictions, dtype=np.int16),
    }
    if recorder.features:
        arrays["features"] = np.asarray(recorder.features, dtype=np.float32)
    for name, detector in recorder.detectors.items():
        observations = detector.detector.observations
        arrays[f"{name}_query_indices"] = np.asarray(
            [row["query_index"] for row in observations], dtype=np.int16
        )
        arrays[f"{name}_scores"] = np.asarray(
            [row["score"] for row in observations], dtype=np.float32
        )
        arrays[f"{name}_predictions"] = np.asarray(
            [row["prediction"] for row in observations], dtype=np.int8
        )
        arrays[f"{name}_histograms"] = np.asarray(
            [row["histogram"] for row in observations], dtype=np.float32
        ).reshape((-1, 201))
    return result, arrays


def session_seed(base, dataset_index, workload, objective, optimizer, lam):
    text = f"{base}|{dataset_index}|{workload}|{objective}|{optimizer}|{lam:.7g}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "nes", "simba"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--chosen-denoise-lambda", type=float, default=-1)
    parser.add_argument("--chosen-deblur-lambda", type=float, default=-1)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    manifest = prepare_manifest(args.output_dir.parent / "manifest.csv", dataset, model, device, args.seed)
    _, _, delta_net = load_official_components(torch.device("cpu"))
    if args.stage == "development":
        subset = manifest[manifest.split == "development"]
        optimizers, objectives, lambdas = ("nes", "simba"), ("restore",), None
    else:
        subset = manifest[manifest.split != "development"]
        optimizers = (args.stage,)
        objectives = ("attack", "restore", "random_walk") if args.stage == "nes" else ("attack", "restore")
        lambdas = (0.0,)
        if args.chosen_denoise_lambda < 0 or args.chosen_deblur_lambda < 0:
            raise ValueError("main stages require frozen chosen lambdas")
    if args.max_images > 0:
        subset = subset.head(args.max_images)
    metadata = {
        "stage": args.stage,
        "args": vars(args) | {"output_dir": str(args.output_dir), "checkpoint": str(args.checkpoint)},
        "gwad_commit": subprocess.check_output(["git", "-C", str(GWAD_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "delta_net_sha256": sha256(GWAD_ROOT / "model/delta/delta_ann.pt"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "query_accounting": "all submitted proposals and candidates; no hidden confirmation calls",
        "replay": "dataset index + frozen config + session seed + acceptance decisions; SHA256 validates exact float32 query byte stream",
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    summary_path = args.output_dir / "sessions.jsonl"
    completed = set()
    if summary_path.exists():
        with summary_path.open() as handle:
            for line in handle:
                completed.add(json.loads(line)["session_id"])
    for row in subset.itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0].unsqueeze(0).to(device)
        label = int(row.source_label)
        for workload in ("denoise", "deblur"):
            observed = corruption(clean, workload, int(row.dataset_index), args.seed)
            for optimizer in optimizers:
                for objective in objectives:
                    if objective == "random_walk" and optimizer != "nes":
                        continue
                    effective_optimizer = "random_walk" if objective == "random_walk" else optimizer
                    candidate_lambdas = STRENGTHS[workload] if args.stage == "development" else (
                        args.chosen_denoise_lambda if workload == "denoise" else args.chosen_deblur_lambda,
                    )
                    for lam in candidate_lambdas:
                        sid = f"{row.split}__{row.dataset_index}__{workload}__{objective}__{effective_optimizer}__l{lam:.7g}"
                        if sid in completed:
                            continue
                        seed = session_seed(args.seed, row.dataset_index, workload, objective, effective_optimizer, lam)
                        started = time.time()
                        result, arrays = run_session(
                            model, clean, observed, label, workload, objective, effective_optimizer,
                            float(lam), seed, args.budget, device, delta_net,
                            observe_detectors=args.stage != "development",
                        )
                        trace = args.output_dir / "traces" / f"{sid}.npz"
                        trace.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(trace, **arrays)
                        record = {
                            "session_id": sid, "split": row.split,
                            "dataset_index": int(row.dataset_index), "source_label": label,
                            "workload": workload, "objective": objective,
                            "optimizer": effective_optimizer, "lambda": float(lam),
                            "session_seed": seed, "elapsed_seconds": time.time() - started,
                            "trace": str(trace.relative_to(args.output_dir)),
                        } | result
                        with summary_path.open("a") as handle:
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        print(f"[DONE] {sid} calls={result['calls']} gain={result['psnr_gain']:.3f}", flush=True)


if __name__ == "__main__":
    main()
