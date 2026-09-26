#!/usr/bin/env python3
"""Matched benign-optimization workloads for the stateful-specificity study.

Extends the frozen paper-gate runner (same manifest, corrupted starts, NES
settings, 8/255 ball, 1,024-call budget, and session seeding) with benign
objectives whose effect on the classifier output ranges from label-preserving
to label-changing, and adds the official Blacklight fingerprint as an online
detector next to released GWAD / GWAD+.

Objectives (all optimized by the same NES routine unless noted):
  attack            maximize CE of the source label (untargeted evasion)
  restore           minimize restoration loss; reject label-changing updates
  random_walk       objective-free projected sign walk (non-optimizing control)
  confidence_boost  maximize log p(source label) ("unadversarial" optimization)
  boundary_probe    minimize the source-label margin; reject label-changing
                    updates (local robustness self-assessment of one's own image)
  counterfactual    minimize CE toward the start's runner-up class plus an L2
                    proximity penalty (black-box counterfactual explanation)
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.lfc_detector import LFCPhase1  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    GWAD_ROOT,
    Recorder,
    atomic_json,
    better,
    corruption,
    load_cifar_model,
    load_official_components,
    prepare_manifest,
    project,
    psnr,
    restoration_loss,
    session_seed,
    set_seed,
    sha256,
)

OBJECTIVES = ("attack", "restore", "random_walk", "confidence_boost", "boundary_probe", "counterfactual")
LABEL_PRESERVING = {"restore", "boundary_probe"}
COUNTERFACTUAL_L2 = 100.0
# Published CIFAR-10 Blacklight parameters (third_party/stateful_monitoring_baselines/blacklight).
BLACKLIGHT = {"window_size": 20, "num_hashes_keep": 50, "round": 50, "step_size": 1, "threshold": 25}


def blacklight_salt(shape=(32, 32, 3)) -> np.ndarray:
    """Official salt: np.random.seed(666) at import, then rand(*query.shape) * 255."""
    return np.random.RandomState(666).rand(*shape) * 255.0


class BlacklightTracker:
    """In-process port of the official InputTracker (same preprocessing, hashing, matching).

    The official class spawns a multiprocessing pool and uses the removed `imp`
    module; `test_blacklight_port.py` checks this port against it query by query.
    """

    def __init__(self, salt: np.ndarray):
        self.salt = salt
        self.hash_dict: dict[str, list[int]] = {}
        self.input_idx = 0
        self.counts: list[int] = []
        self.digest_cache: dict[bytes, str] = {}

    def hashes(self, hwc01: np.ndarray) -> list[str]:
        array = (np.array(hwc01) * 255.0 + self.salt) % 255.0
        array = array.reshape(-1)
        array = (np.around(array / BLACKLIGHT["round"], decimals=0) * BLACKLIGHT["round"]).astype(np.int16)
        width = BLACKLIGHT["window_size"] * array.itemsize
        stride = BLACKLIGHT["step_size"] * array.itemsize
        total = int((len(array) - BLACKLIGHT["window_size"] + 1) / BLACKLIGHT["step_size"])
        raw = array.tobytes()
        # Consecutive queries share almost all quantized windows, so memoize window -> reversed digest.
        cache = self.digest_cache
        found = set()
        for window in {raw[i * stride : i * stride + width] for i in range(total)}:
            digest = cache.get(window)
            if digest is None:
                digest = cache[window] = hashlib.sha256(window).hexdigest()[::-1]
            found.add(digest)
        # Same as the release: reverse-sorted digests, first num_hashes_keep.
        return heapq.nlargest(BLACKLIGHT["num_hashes_keep"], found)

    def add(self, hwc01: np.ndarray) -> int:
        self.input_idx += 1
        hashes = self.hashes(hwc01)
        matches = [idx for h in hashes for idx in self.hash_dict.get(h, ())]
        count = Counter(matches).most_common(1)[0][1] if matches else 0
        for h in hashes:
            self.hash_dict.setdefault(h, []).append(self.input_idx)
        return count

    def submit(self, raw: torch.Tensor) -> None:
        self.counts.append(self.add(raw[0].permute(1, 2, 0).numpy().astype(np.float64)))

    def summary(self):
        counts = np.asarray(self.counts)
        hit = np.flatnonzero(counts >= BLACKLIGHT["threshold"])
        return {"max_match": int(counts.max()) if len(counts) else 0,
                "first_alarm": int(hit[0] + 1) if len(hit) else -1}


def margin(logits: torch.Tensor, label: int) -> torch.Tensor:
    others = logits.clone()
    others[:, label] = -torch.inf
    return logits[:, label] - others.max(1).values


def values_for(images, logits, start, workload, objective, lam, label, target):
    """Objective value per candidate and whether it is maximized or minimized."""
    labels = torch.full((len(images),), label, device=images.device, dtype=torch.long)
    if objective == "attack":
        return F.cross_entropy(logits, labels, reduction="none"), "max"
    if objective == "restore":
        return restoration_loss(images, start.expand_as(images), workload, lam), "min"
    if objective == "confidence_boost":
        return -F.cross_entropy(logits, labels, reduction="none"), "max"
    if objective == "boundary_probe":
        return margin(logits, label), "min"
    if objective == "counterfactual":
        targets = torch.full_like(labels, target)
        proximity = (images - start).square().mean((1, 2, 3))
        return F.cross_entropy(logits, targets, reduction="none") + COUNTERFACTUAL_L2 * proximity, "min"
    raise ValueError(objective)


def prepare_manifest_generic(path: Path, dataset, labels: np.ndarray, model, device, seed: int, split_counts: dict) -> "pd.DataFrame":
    """Class-balanced manifest for any labelled dataset: clean and both corrupted starts correctly classified."""
    import pandas as pd

    if path.exists():
        return pd.read_csv(path)
    need = sum(split_counts.values())
    rng = np.random.default_rng(seed)
    rows = []
    for label in np.unique(labels):
        candidates = rng.permutation(np.flatnonzero(labels == label))
        chosen = []
        for index in candidates:
            clean = dataset[int(index)][0].unsqueeze(0).to(device)
            batch = torch.cat([clean, corruption(clean, "denoise", int(index), seed), corruption(clean, "deblur", int(index), seed)])
            with torch.no_grad():
                if bool(model(batch).argmax(1).eq(int(label)).all()):
                    chosen.append(int(index))
            if len(chosen) == need:
                break
        cursor = 0
        for split, count in split_counts.items():
            for index in chosen[cursor : cursor + count]:
                rows.append({"dataset_index": index, "source_label": int(label), "split": split})
            cursor += count
    frame = pd.DataFrame(rows).sort_values(["split", "source_label", "dataset_index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def run_session(model, clean, start, label, workload, objective, lam, seed, budget, device, delta_net, salt, lfc_seed=-1,
                optimizer="nes", simba_step=8.0 / 255.0):
    recorder = Recorder.create(model, label, device, delta_net)
    blacklight = BlacklightTracker(salt)
    recorder.detectors["blacklight"] = blacklight
    if lfc_seed >= 0:
        recorder.detectors["lfc_phase1"] = LFCPhase1(d=start[0].numel(), seed=lfc_seed, bern_scale=0.0)
    current = start.clone()
    current_logits = recorder.submit_batch(current)
    target = int(current_logits[0].clone().index_fill_(0, torch.tensor([label], device=device), -torch.inf).argmax())
    if objective != "random_walk":
        current_value, direction = values_for(current, current_logits, start, workload, objective, lam, label, target)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if optimizer == "simba":
        # Pixel-basis SimBA (the 8/255 setting of check_rank_equivalence.py) for every objective.
        order = torch.randperm(current[0].numel(), generator=generator, device=device)
    else:
        # The frozen runner draws a SimBA ordering before every session; keep the RNG stream identical.
        torch.randperm(current[0].numel() if workload == "denoise" else current.shape[1] * 8 * 8, generator=generator, device=device)
    eps, step, sigma, pairs = 8.0 / 255.0, 0.25 / 255.0, 2.0 / 255.0, 8
    accepted = []
    while recorder.calls < budget:
        if objective == "random_walk":
            delta = torch.where(torch.rand(current.shape, generator=generator, device=device) < 0.5,
                                -torch.ones_like(current), torch.ones_like(current))
            current = project(current + step * delta, start, eps)
            current_logits = recorder.submit_batch(current)
            accepted.append(1)
            continue
        if optimizer == "simba":
            if recorder.calls + 2 > budget:
                break
            delta = torch.zeros_like(current).flatten()
            delta[int(order[len(accepted) % len(order)])] = simba_step
            delta = delta.view_as(current)
            candidates = torch.cat([project(current - delta, start, eps), project(current + delta, start, eps)])
            cand_logits = recorder.submit_batch(candidates)
            cand_values, direction = values_for(candidates, cand_logits, start, workload, objective, lam, label, target)
            best = None
            for i in range(2):
                if objective in LABEL_PRESERVING and int(cand_logits[i].argmax()) != label:
                    continue
                if better(cand_values[i], current_value[0], direction) and (best is None or better(cand_values[i], cand_values[best], direction)):
                    best = i
            if best is not None:
                current, current_logits, current_value = candidates[best : best + 1], cand_logits[best : best + 1], cand_values[best : best + 1]
            accepted.append(-1 if best is None else best)
            continue
        if recorder.calls + 2 * pairs + 1 > budget:
            break
        directions = torch.randn((pairs, *current.shape[1:]), generator=generator, device=device)
        proposals = torch.cat([project(current - sigma * directions, start, eps),
                               project(current + sigma * directions, start, eps)])
        proposal_logits = recorder.submit_batch(proposals)
        proposal_values, direction = values_for(proposals, proposal_logits, start, workload, objective, lam, label, target)
        coefficients = proposal_values[pairs:] - proposal_values[:pairs]
        if direction == "min":
            coefficients = -coefficients
        estimate = (coefficients.view(pairs, 1, 1, 1) * directions).mean(0, keepdim=True)
        candidate = project(current + step * estimate.sign(), start, eps)
        candidate_logits = recorder.submit_batch(candidate)
        candidate_value, direction = values_for(candidate, candidate_logits, start, workload, objective, lam, label, target)
        allowed = objective not in LABEL_PRESERVING or int(candidate_logits.argmax()) == label
        take = bool(allowed and better(candidate_value[0], current_value[0], direction))
        if take:
            current, current_logits, current_value = candidate, candidate_logits, candidate_value
        accepted.append(int(take))

    probs0 = F.softmax(recorder_logits_first(recorder), dim=0)
    probs1 = F.softmax(current_logits[0].float().cpu(), dim=0)
    result = {
        "calls": recorder.calls,
        "first_success": recorder.first_success,
        "final_prediction": int(current_logits.argmax().item()),
        "label_preserved": int(current_logits.argmax().item() == label),
        "counterfactual_target": target,
        "reached_target": int(current_logits.argmax().item() == target),
        "start_margin": float(margin(torch.from_numpy(recorder.logits[0][None]), label)[0]),
        "final_margin": float(margin(current_logits.float().cpu(), label)[0]),
        "start_p_label": float(probs0[label]),
        "final_p_label": float(probs1[label]),
        "final_linf_255": float((current - start).abs().max().item() * 255.0),
        "final_l2": float((current - start).flatten().norm().item()),
        "start_psnr": psnr(start, clean),
        "final_psnr": psnr(current, clean),
        "psnr_gain": psnr(current, clean) - psnr(start, clean),
        "query_sha256": recorder.digest.hexdigest(),
        "accepted": accepted,
        "detectors": {name: detector.summary() for name, detector in recorder.detectors.items()},
    }
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
    return result, arrays


def recorder_logits_first(recorder) -> torch.Tensor:
    return torch.from_numpy(recorder.logits[0]).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "main"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv"))
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--denoise-lambda", type=float, default=0.5)
    parser.add_argument("--deblur-lambda", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--model-name", default="resnet18_seed0", help="resnet18_seedN, bbb_*, robustbench_<name>, or gtsrb32")
    parser.add_argument("--optimizer", choices=["nes", "simba"], default="nes")
    parser.add_argument("--dataset", choices=["cifar10", "gtsrb32"], default="cifar10")
    parser.add_argument("--gtsrb-root", default="data/gtsrb")
    parser.add_argument("--lfc-seed", type=int, default=-1, help="add the Lee-Fang-Chang Phase-1 observer with this detector seed")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "gtsrb32":
        from experiments.eaai_gtsrb.gtsrb_common import gtsrb_dataset, load_checkpoint
        model, _payload = load_checkpoint(args.checkpoint, device)
        dataset = gtsrb_dataset(args.gtsrb_root, "test", 32)
        labels = np.asarray([int(label) for _path, label in dataset._samples])
        manifest = prepare_manifest_generic(args.manifest, dataset, labels, model, device, args.seed,
                                            split_counts={"development": 1, "fit": 2, "calibration": 2, "evaluation": 5})
    else:
        checkpoint = args.checkpoint if args.model_name.startswith("resnet18_seed") else None
        model = load_cifar_model(args.model_name, checkpoint, device).eval()
        dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
        manifest = prepare_manifest(args.manifest, dataset, model, device, args.seed)
    _, _, delta_net = load_official_components(torch.device("cpu"))
    salt = blacklight_salt()
    subset = manifest[manifest.split == "development"] if args.stage == "development" else manifest[manifest.split != "development"]
    if args.max_images > 0:
        subset = subset.head(args.max_images)
    subset = subset.iloc[args.shard :: args.num_shards]
    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    atomic_json(args.output_dir / f"metadata_shard{args.shard}.json", {
        "stage": args.stage,
        "args": {k: str(v) for k, v in vars(args).items()},
        "gwad_commit": subprocess.check_output(["git", "-C", str(GWAD_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "delta_net_sha256": sha256(GWAD_ROOT / "model/delta/delta_ann.pt"),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint.exists() else None,
        "model_name": args.model_name, "optimizer": args.optimizer, "dataset": args.dataset,
        "blacklight": BLACKLIGHT,
        "counterfactual_l2": COUNTERFACTUAL_L2,
    })
    summary_path = args.output_dir / f"sessions_shard{args.shard}.jsonl"
    completed = set()
    if summary_path.exists():
        completed = {json.loads(line)["session_id"] for line in summary_path.read_text().splitlines() if line.strip()}
    for row in subset.itertuples(index=False):
        clean = dataset[int(row.dataset_index)][0].unsqueeze(0).to(device)
        label = int(row.source_label)
        for workload in ("denoise", "deblur"):
            start = corruption(clean, workload, int(row.dataset_index), args.seed)
            lam = args.denoise_lambda if workload == "denoise" else args.deblur_lambda
            for objective in objectives:
                if objective == "random_walk" and args.optimizer != "nes":
                    continue  # the objective-free walk is optimizer-independent; run once with the NES corpus
                optimizer = "random_walk" if objective == "random_walk" else args.optimizer
                sid = f"{row.split}__{row.dataset_index}__{workload}__{objective}__{optimizer}__l{lam:.7g}"
                if sid in completed:
                    continue
                # Same seed derivation as the frozen gate, so attack/restore/random_walk replay exactly.
                seed = session_seed(args.seed, row.dataset_index, workload, objective, optimizer, lam)
                started = time.time()
                result, arrays = run_session(model, clean, start, label, workload, objective, lam, seed,
                                             args.budget, device, delta_net, salt, args.lfc_seed, optimizer=args.optimizer)
                trace = args.output_dir / "traces" / f"{sid}.npz"
                trace.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trace, **arrays)
                record = {"session_id": sid, "split": row.split, "dataset_index": int(row.dataset_index),
                          "source_label": label, "workload": workload, "objective": objective,
                          "optimizer": optimizer, "lambda": float(lam), "session_seed": seed,
                          "model_name": args.model_name, "dataset": args.dataset,
                          "elapsed_seconds": time.time() - started,
                          "trace": str(trace.relative_to(args.output_dir))} | result
                with summary_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(f"[DONE] {sid} calls={result['calls']} margin {result['start_margin']:.2f}->{result['final_margin']:.2f} "
                      f"t={record['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
