#!/usr/bin/env python3
"""Shared data, model, feature, and artifact utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from experiments.activation_atlas.activation_recorder import (
    ActivationRecorder,
    activation_state,
    channel_activity,
)
from experiments.activation_atlas.run_cifar_gate_exit_pilot import (
    CLASS_NAMES,
    load_model as load_resnet18,
)


SPLIT_COUNTS = {
    "tune": 20,
    "train": 40,
    "validation": 20,
    "test": 40,
}


def class_balanced_subset(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select a deterministic near-equal number of rows from every source class."""

    if count <= 0 or count >= len(frame):
        return frame.reset_index(drop=True)
    classes = sorted(frame.source_label.unique())
    base, remainder = divmod(count, len(classes))
    parts = []
    for position, class_id in enumerate(classes):
        take = base + int(position < remainder)
        parts.append(frame[frame.source_label == class_id].head(take))
    selected = pd.concat(parts, ignore_index=True)
    if len(selected) != count:
        raise RuntimeError(
            f"Requested {count} class-balanced rows but selected {len(selected)}"
        )
    return selected.sort_values(
        ["source_label", "dataset_index"]
    ).reset_index(drop=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_cifar_model(
    model_name: str,
    checkpoint: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    if model_name.startswith("resnet18_seed"):
        checkpoint = resolve_checkpoint(model_name, checkpoint)
        return load_resnet18(checkpoint, device)

    aliases = {
        "bbb_resnet50",
        "bbb_vgg19_bn",
        "bbb_densenet",
        "bbb_inception_v3",
    }
    if model_name in aliases:
        from utils.load_models import load_blackboxbench_cifar_model

        return load_blackboxbench_cifar_model(model_name).to(device).eval()
    if model_name.startswith("robustbench_"):
        from robustbench.utils import load_model as load_robustbench_model

        robustbench_name = model_name.removeprefix("robustbench_")
        return load_robustbench_model(
            robustbench_name,
            model_dir="checkpoints/robustbench_cifar10",
            dataset="cifar10",
            threat_model="Linf",
        ).to(device).eval()
    raise ValueError(f"Unsupported CIFAR model: {model_name}")


def resolve_checkpoint(model_name: str, checkpoint: Path | None) -> Path | None:
    if checkpoint is not None:
        return checkpoint.resolve()
    if model_name.startswith("resnet18_seed"):
        seed = int(model_name.removeprefix("resnet18_seed"))
        return Path(
            "checkpoints/cifar10_resnet18_training_dynamics_v1"
            f"/seed{seed}/resnet18_seed{seed}_epoch080.pt"
        ).resolve()
    if model_name.startswith("robustbench_"):
        robustbench_name = model_name.removeprefix("robustbench_")
        candidate = Path(
            "checkpoints/robustbench_cifar10/cifar10/Linf"
        ) / f"{robustbench_name}.pt"
        return candidate.resolve() if candidate.exists() else None
    return None


def cifar_test_set(dataset_root: str):
    return datasets.CIFAR10(
        dataset_root,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )


@torch.no_grad()
def clean_correct_indices(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
) -> dict[int, list[int]]:
    by_class = {class_id: [] for class_id in range(len(CLASS_NAMES))}
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        images = torch.stack([dataset[index][0] for index in range(start, stop)]).to(device)
        labels = torch.as_tensor(
            [dataset[index][1] for index in range(start, stop)],
            device=device,
        )
        predictions = model(images).argmax(1)
        for offset in (predictions == labels).nonzero(as_tuple=False).flatten().tolist():
            index = start + offset
            by_class[int(labels[offset].item())].append(index)
    return by_class


def deterministic_target(dataset_index: int, source_label: int, seed: int) -> int:
    value = (dataset_index * 0x9E3779B185EBCA87 + seed) & ((1 << 64) - 1)
    offset = value % 9 + 1
    return int((source_label + offset) % len(CLASS_NAMES))


def prepare_split_manifest(
    output: Path,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    path = output / "split_manifest.csv"
    summary_path = output / "clean_model_summary.json"
    if path.exists() and summary_path.exists():
        return pd.read_csv(path)

    correct = clean_correct_indices(model, dataset, device, batch_size)
    atomic_json(
        summary_path,
        {
            "dataset_n": len(dataset),
            "clean_correct_n": int(sum(len(values) for values in correct.values())),
            "clean_accuracy": float(
                sum(len(values) for values in correct.values()) / len(dataset)
            ),
            "clean_correct_by_class": {
                str(class_id): len(values) for class_id, values in correct.items()
            },
            "evaluation_mode": True,
        },
    )
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(seed)
    rows = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.asarray(correct[class_id], dtype=int)
        required = sum(SPLIT_COUNTS.values())
        if len(candidates) < required:
            raise RuntimeError(
                f"Class {class_id} has {len(candidates)} clean-correct images; "
                f"{required} are required"
            )
        candidates = rng.permutation(candidates)[:required]
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            for dataset_index in candidates[cursor : cursor + count]:
                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "source_label": class_id,
                        "target_label": deterministic_target(
                            int(dataset_index), class_id, seed
                        ),
                        "split": split,
                    }
                )
            cursor += count
    manifest = pd.DataFrame(rows).sort_values(
        ["split", "source_label", "dataset_index"]
    )
    atomic_csv(path, manifest)
    return manifest


def tensor_batch(dataset, rows: pd.DataFrame, device: torch.device):
    indices = rows.dataset_index.astype(int).tolist()
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    source = torch.as_tensor(rows.source_label.to_numpy(), device=device, dtype=torch.long)
    target = torch.as_tensor(rows.target_label.to_numpy(), device=device, dtype=torch.long)
    return images, source, target, np.asarray(indices, dtype=np.int64)


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target[:, None], -torch.inf)
    return target_logits - masked.max(1).values


def perturbation_metrics(states: torch.Tensor, clean: torch.Tensor) -> dict[str, np.ndarray]:
    delta = states - clean[:, None]
    flat = delta.flatten(2)
    clipped = (
        (states <= torch.finfo(states.dtype).eps)
        | (states >= 1.0 - torch.finfo(states.dtype).eps)
    ).flatten(2)
    return {
        "linf": flat.abs().amax(2).cpu().numpy().astype(np.float32),
        "l2": flat.square().sum(2).sqrt().cpu().numpy().astype(np.float32),
        "l1": flat.abs().sum(2).cpu().numpy().astype(np.float32),
        "clipped_fraction": clipped.float().mean(2).cpu().numpy().astype(np.float32),
    }


def _channel_statistics(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if values.ndim <= 2:
        values = values.unsqueeze(-1)
    flat = values.flatten(2)
    mean = flat.mean(2)
    std = flat.std(2, unbiased=False)
    norm = flat.square().mean(2).sqrt()
    return mean, std, norm


@torch.no_grad()
def extract_trajectory_features(
    recorder: ActivationRecorder,
    states: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    clean: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract state and transition features from [N, S, C, H, W] states."""

    n_images, n_steps = states.shape[:2]
    gate_state_steps: list[np.ndarray] = []
    gate_on_steps: list[np.ndarray] = []
    gate_off_steps: list[np.ndarray] = []
    gate_cumulative_steps: list[np.ndarray] = []
    activation_mean_steps: list[np.ndarray] = []
    activation_std_steps: list[np.ndarray] = []
    activation_norm_steps: list[np.ndarray] = []
    logits_steps: list[np.ndarray] = []
    feature_names: list[str] | None = None
    site_slices: dict[str, list[int]] = {}

    previous_states: dict[str, torch.Tensor] | None = None
    initial_states: dict[str, torch.Tensor] | None = None
    for step in range(n_steps):
        per_step_gate = []
        per_step_on = []
        per_step_off = []
        per_step_cumulative = []
        per_step_mean = []
        per_step_std = []
        per_step_norm = []
        per_step_logits = []
        current_states: dict[str, list[torch.Tensor]] = {
            name: [] for name in recorder.sites
        }
        current_continuous: dict[str, list[torch.Tensor]] = {
            name: [] for name in recorder.sites
        }
        for start in range(0, n_images, batch_size):
            batch = states[start : start + batch_size, step]
            logits, activations = recorder.record(batch)
            per_step_logits.append(logits.detach().cpu())
            for name, values in activations.items():
                current_states[name].append(
                    activation_state(values, recorder.sites[name].family).detach().cpu()
                )
                current_continuous[name].append(values.detach().cpu())
        current = {name: torch.cat(parts) for name, parts in current_states.items()}
        continuous_by_site = {
            name: torch.cat(parts) for name, parts in current_continuous.items()
        }
        if previous_states is None:
            previous_states = {name: values.clone() for name, values in current.items()}
            initial_states = {name: values.clone() for name, values in current.items()}

        cursor = 0
        names = []
        for name, values in current.items():
            active = channel_activity(values)
            on = channel_activity(values & ~previous_states[name])
            off = channel_activity(~values & previous_states[name])
            cumulative = channel_activity(values ^ initial_states[name])
            per_step_gate.append(active.numpy())
            per_step_on.append(on.numpy())
            per_step_off.append(off.numpy())
            per_step_cumulative.append(cumulative.numpy())

            continuous = continuous_by_site[name]
            mean, std, norm = _channel_statistics(continuous)
            per_step_mean.append(mean.numpy())
            per_step_std.append(std.numpy())
            per_step_norm.append(norm.numpy())

            width = active.shape[1]
            site_slices[name] = [cursor, cursor + width]
            names.extend(f"{name}:channel_{index}" for index in range(width))
            cursor += width

        if feature_names is None:
            feature_names = names
        gate_state_steps.append(np.concatenate(per_step_gate, axis=1))
        gate_on_steps.append(np.concatenate(per_step_on, axis=1))
        gate_off_steps.append(np.concatenate(per_step_off, axis=1))
        gate_cumulative_steps.append(np.concatenate(per_step_cumulative, axis=1))
        activation_mean_steps.append(np.concatenate(per_step_mean, axis=1))
        activation_std_steps.append(np.concatenate(per_step_std, axis=1))
        activation_norm_steps.append(np.concatenate(per_step_norm, axis=1))
        logits_steps.append(torch.cat(per_step_logits).numpy())
        previous_states = current

    def stack_steps(values, dtype=np.float16):
        return np.stack(values, axis=1).astype(dtype)

    logits = stack_steps(logits_steps, np.float32)
    probability = torch.softmax(torch.from_numpy(logits), dim=2).numpy()
    predictions = logits.argmax(2)
    source_np = source.cpu().numpy()
    target_np = target.cpu().numpy()
    source_indices = np.repeat(source_np[:, None, None], n_steps, axis=1)
    target_indices = np.repeat(target_np[:, None, None], n_steps, axis=1)
    source_confidence = np.take_along_axis(
        probability, source_indices, axis=2
    ).squeeze(2)
    target_confidence = np.take_along_axis(
        probability, target_indices, axis=2
    ).squeeze(2)
    margin = target_margin(
        torch.from_numpy(logits.reshape(-1, logits.shape[-1])),
        target.repeat_interleave(n_steps).cpu(),
    ).reshape(n_images, n_steps).numpy()
    success = predictions == target_np[:, None]
    first_success = np.full(n_images, -1, dtype=np.int16)
    for index in range(n_images):
        positions = np.flatnonzero(success[index])
        if len(positions):
            first_success[index] = int(positions[0])

    arrays = {
        "gate_state": stack_steps(gate_state_steps),
        "gate_on": stack_steps(gate_on_steps),
        "gate_off": stack_steps(gate_off_steps),
        "gate_cumulative_flip": stack_steps(gate_cumulative_steps),
        "activation_mean": stack_steps(activation_mean_steps),
        "activation_std": stack_steps(activation_std_steps),
        "activation_norm": stack_steps(activation_norm_steps),
        "logits": logits,
        "prediction": predictions.astype(np.int16),
        "source_confidence": source_confidence.astype(np.float32),
        "target_confidence": target_confidence.astype(np.float32),
        "target_margin": margin.astype(np.float32),
        "success": success.astype(np.uint8),
        "first_success": first_success,
    }
    arrays.update(perturbation_metrics(states, clean))
    schema = {
        "feature_names": feature_names,
        "site_slices": site_slices,
        "activation_sites": [asdict(site) for site in recorder.sites.values()],
        "gate_definition": "exact post-ReLU output > 0; stored as per-channel spatial active fraction",
        "transition_definition": "per-channel fraction of exact spatial gates changing state",
    }
    return arrays, schema


def normalized_milestones(total: int, fractions: Iterable[float]) -> list[int]:
    values = {0, total}
    values.update(int(round(total * fraction)) for fraction in fractions)
    return sorted(max(0, min(total, value)) for value in values)
