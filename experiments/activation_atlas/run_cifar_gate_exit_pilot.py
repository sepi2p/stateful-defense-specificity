#!/usr/bin/env python3
"""Build a ReLU activation atlas and test class-prototype early exits on CIFAR-10."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.activation_atlas.activation_recorder import (  # noqa: E402
    ActivationRecorder,
    activation_state,
    channel_activity,
)
from surro_models.cifar10_models.resnet import ResNet18  # noqa: E402


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/cifar10_resnet18_training_dynamics_v1/seed0/resnet18_seed0_epoch080.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/activation_atlas/cifar10_resnet18_gate_exit_pilot",
    )
    parser.add_argument("--fit-per-class", type=int, default=500)
    parser.add_argument("--calibration-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47011)
    parser.add_argument("--prototype-smoothing", type=float, default=1.0)
    parser.add_argument("--heatmap-units", type=int, default=320)
    parser.add_argument("--risk-targets", default="0.90,0.95,0.98")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(path: Path, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet18()
    model.load_state_dict(state["net"] if isinstance(state, dict) and "net" in state else state)
    return model.to(device).eval()


def class_balanced_indices(
    targets: list[int] | np.ndarray,
    per_class: int,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> list[int]:
    excluded = excluded or set()
    labels = np.asarray(targets)
    chosen: list[int] = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.where(labels == class_id)[0]
        candidates = np.asarray([value for value in candidates if int(value) not in excluded])
        if len(candidates) < per_class:
            raise RuntimeError(f"Class {class_id} has {len(candidates)} candidates, need {per_class}")
        chosen.extend(rng.choice(candidates, per_class, replace=False).astype(int).tolist())
    return sorted(chosen)


def build_splits(args, train_set, test_set) -> dict[str, list[int]]:
    rng = np.random.default_rng(args.seed)
    fit = class_balanced_indices(train_set.targets, args.fit_per_class, rng)
    calibration = class_balanced_indices(
        train_set.targets,
        args.calibration_per_class,
        rng,
        excluded=set(fit),
    )
    test = class_balanced_indices(test_set.targets, args.test_per_class, rng)
    return {"fit": fit, "calibration": calibration, "test": test}


def loader(dataset, indices: list[int], args, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def estimate_compute_fractions(recorder: ActivationRecorder) -> dict[str, float]:
    graph = copy.deepcopy(recorder.graph_module).cpu().eval()
    ShapeProp(graph).propagate(torch.zeros(1, 3, 32, 32))
    cumulative = 0.0
    activation_cost: dict[str, float] = {}
    for node in graph.graph.nodes:
        if node.op == "call_module":
            module = graph.get_submodule(str(node.target))
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is None:
                continue
            shape = tuple(tensor_meta.shape)
            if isinstance(module, nn.Conv2d):
                _, out_channels, height, width = shape
                kernel_height, kernel_width = module.kernel_size
                cumulative += (
                    out_channels
                    * height
                    * width
                    * (module.in_channels / module.groups)
                    * kernel_height
                    * kernel_width
                )
            elif isinstance(module, nn.Linear):
                cumulative += module.in_features * module.out_features
        if node.name in recorder.sites:
            activation_cost[node.name] = cumulative
    total = max(cumulative, 1.0)
    return {name: float(value / total) for name, value in activation_cost.items()}


@torch.no_grad()
def fit_prototypes(recorder, data_loader, device, class_count, smoothing):
    counts: OrderedDict[str, torch.Tensor] = OrderedDict()
    channel_sums: OrderedDict[str, torch.Tensor] = OrderedDict()
    class_counts = torch.zeros(class_count, dtype=torch.float64, device=device)
    shapes: dict[str, tuple[int, ...]] = {}
    feature_batches = []
    label_batches = []
    correct = total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits, activations = recorder.record(images)
        correct += int((logits.argmax(1) == labels).sum())
        total += int(labels.numel())
        class_counts += torch.bincount(labels, minlength=class_count).double()
        batch_features = []
        for node_name, values in activations.items():
            site = recorder.sites[node_name]
            states = activation_state(values, site.family)
            if node_name not in counts:
                counts[node_name] = torch.zeros(
                    (class_count, *states.shape[1:]), dtype=torch.float64, device=device
                )
                channel_sums[node_name] = torch.zeros(
                    (class_count, states.shape[1]), dtype=torch.float64, device=device
                )
                shapes[node_name] = tuple(int(value) for value in states.shape[1:])
            rates = channel_activity(states).double()
            batch_features.append(rates.float().cpu().numpy())
            for class_id in labels.unique().tolist():
                mask = labels == class_id
                counts[node_name][class_id] += states[mask].double().sum(0)
                channel_sums[node_name][class_id] += rates[mask].sum(0)
        feature_batches.append(np.concatenate(batch_features, axis=1).astype(np.float32))
        label_batches.append(labels.cpu().numpy().astype(np.int64))
    probabilities: OrderedDict[str, np.ndarray] = OrderedDict()
    channel_profiles: OrderedDict[str, np.ndarray] = OrderedDict()
    for node_name in counts:
        denominator = class_counts.view(class_count, *([1] * (counts[node_name].ndim - 1)))
        probabilities[node_name] = (
            (counts[node_name] + smoothing) / (denominator + 2 * smoothing)
        ).float().cpu().numpy()
        channel_profiles[node_name] = (
            channel_sums[node_name] / class_counts[:, None]
        ).float().cpu().numpy()
    return (
        probabilities,
        channel_profiles,
        shapes,
        correct / max(total, 1),
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
    )


def prototype_tensors(probabilities, device):
    values = {}
    for node_name, probability in probabilities.items():
        p = torch.from_numpy(probability).to(device).float().flatten(1).clamp(1e-5, 1 - 1e-5)
        values[node_name] = {
            "logit": torch.log(p) - torch.log1p(-p),
            "base": torch.log1p(-p).sum(1),
            "units": p.shape[1],
        }
    return values


@torch.no_grad()
def evaluate_split(
    recorder,
    data_loader,
    device,
    prototypes,
    exit_nodes,
    selected_units,
    collect_class_zero,
):
    rows = []
    sample_channels: dict[int, dict[str, list[float]]] = {}
    class_zero_heatmap = []
    class_zero_indices = []
    feature_batches = []
    label_batches = []
    running_index = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits, activations = recorder.record(images)
        batch_size = labels.numel()
        cumulative = torch.zeros(batch_size, len(CLASS_NAMES), device=device)
        cumulative_units = 0
        layer_results = {}
        selected_parts = []
        batch_features = []
        for node_name, values in activations.items():
            site = recorder.sites[node_name]
            states = activation_state(values, site.family)
            flat = states.float().flatten(1)
            model = prototypes[node_name]
            cumulative += flat @ model["logit"].T + model["base"][None, :]
            cumulative_units += model["units"]
            if node_name in selected_units:
                selected_parts.append(
                    states.flatten(1)[:, selected_units[node_name]].detach().cpu().numpy().astype(np.uint8)
                )
            if node_name in exit_nodes:
                normalized = cumulative / cumulative_units
                top = torch.topk(normalized, 2, dim=1)
                layer_results[node_name] = (
                    top.indices[:, 0].cpu().numpy(),
                    (top.values[:, 0] - top.values[:, 1]).cpu().numpy(),
                )
            rates = channel_activity(states).cpu().numpy()
            batch_features.append(rates.astype(np.float32))
            for class_id in range(len(CLASS_NAMES)):
                matches = np.where(labels.cpu().numpy() == class_id)[0]
                if len(matches) and class_id not in sample_channels:
                    position = int(matches[0])
                    sample_channels[class_id] = {}
                if class_id in sample_channels and node_name not in sample_channels[class_id]:
                    matches = np.where(labels.cpu().numpy() == class_id)[0]
                    if len(matches):
                        sample_channels[class_id][node_name] = rates[int(matches[0])].tolist()
        labels_np = labels.cpu().numpy()
        feature_batches.append(np.concatenate(batch_features, axis=1))
        label_batches.append(labels_np.astype(np.int64))
        full_predictions = logits.argmax(1).cpu().numpy()
        for position in range(batch_size):
            row = {
                "sample_order": running_index + position,
                "label": int(labels_np[position]),
                "full_prediction": int(full_predictions[position]),
                "full_correct": int(full_predictions[position] == labels_np[position]),
            }
            for node_name, (prediction, gap) in layer_results.items():
                row[f"{node_name}_prediction"] = int(prediction[position])
                row[f"{node_name}_gap"] = float(gap[position])
                row[f"{node_name}_correct"] = int(prediction[position] == labels_np[position])
            rows.append(row)
        if collect_class_zero and selected_parts:
            selected_batch = np.concatenate(selected_parts, axis=1)
            mask = labels_np == 0
            class_zero_heatmap.append(selected_batch[mask])
            class_zero_indices.extend((running_index + np.where(mask)[0]).astype(int).tolist())
        running_index += batch_size
    heatmap = (
        np.concatenate(class_zero_heatmap, axis=0)
        if class_zero_heatmap
        else np.empty((0, sum(len(value) for value in selected_units.values())), dtype=np.uint8)
    )
    return (
        pd.DataFrame(rows),
        sample_channels,
        heatmap,
        class_zero_indices,
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
    )


def channel_feature_ends(recorder, shapes):
    ends = {}
    cumulative = 0
    for node_name, site in recorder.sites.items():
        cumulative += int(shapes[node_name][0])
        if site.is_exit_candidate:
            ends[node_name] = cumulative
    return ends


def fit_gate_linear_probes(
    fit_features,
    fit_labels,
    calibration_features,
    calibration_labels,
    test_features,
    test_labels,
    feature_ends,
    compute_fractions,
    full_predictions,
    seed,
    targets,
):
    layer_rows = []
    calibration = pd.DataFrame({"label": calibration_labels})
    test = pd.DataFrame({"label": test_labels, "full_prediction": full_predictions})
    for layer_index, (node_name, end) in enumerate(feature_ends.items()):
        estimator = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=150,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=seed + layer_index,
                average=True,
            ),
        )
        estimator.fit(fit_features[:, :end], fit_labels)
        calibration_scores = estimator.decision_function(calibration_features[:, :end])
        test_scores = estimator.decision_function(test_features[:, :end])
        calibration_prediction = calibration_scores.argmax(1)
        test_prediction = test_scores.argmax(1)
        calibration[f"{node_name}_prediction"] = calibration_prediction
        calibration[f"{node_name}_correct"] = calibration_prediction == calibration_labels
        calibration_sorted = np.sort(calibration_scores, axis=1)
        calibration[f"{node_name}_gap"] = calibration_sorted[:, -1] - calibration_sorted[:, -2]
        test[f"{node_name}_prediction"] = test_prediction
        test[f"{node_name}_correct"] = test_prediction == test_labels
        test_sorted = np.sort(test_scores, axis=1)
        test[f"{node_name}_gap"] = test_sorted[:, -1] - test_sorted[:, -2]
        layer_rows.append(
            {
                "node_name": node_name,
                "feature_count": end,
                "compute_fraction": compute_fractions[node_name],
                "calibration_accuracy": accuracy_score(calibration_labels, calibration_prediction),
                "test_accuracy": accuracy_score(test_labels, test_prediction),
            }
        )
    policies = calibrate_policies(
        calibration,
        list(feature_ends),
        compute_fractions,
        targets,
    )
    policy_rows = []
    for target in targets:
        prediction, exit_layer, compute = apply_policy(
            test,
            list(feature_ends),
            policies[str(target)],
            compute_fractions,
        )
        policy_rows.append(
            {
                "target_accuracy": target,
                "test_accuracy": accuracy_score(test_labels, prediction),
                "early_exit_fraction": float(np.mean(exit_layer != "full_model")),
                "mean_compute_fraction": float(compute.mean()),
            }
        )
    return (
        pd.DataFrame(layer_rows),
        pd.DataFrame(policy_rows),
        calibration,
        test,
        policies,
    )


def choose_top_units(probabilities, sites, limit):
    candidates = []
    for node_name, probability in probabilities.items():
        flat = probability.reshape(len(CLASS_NAMES), -1)
        specificity = flat[0] - flat[1:].mean(0)
        take = min(limit, len(specificity))
        local = np.argpartition(specificity, -take)[-take:]
        shape = probability.shape[1:]
        for index in local:
            coordinates = np.unravel_index(int(index), shape)
            candidates.append(
                {
                    "node_name": node_name,
                    "flat_index": int(index),
                    "specificity": float(specificity[index]),
                    "class0_probability": float(flat[0, index]),
                    "channel": int(coordinates[0]),
                    "row": int(coordinates[1]) if len(coordinates) > 1 else 0,
                    "column": int(coordinates[2]) if len(coordinates) > 2 else 0,
                    "display_name": sites[node_name].display_name,
                }
            )
    candidates = sorted(candidates, key=lambda row: row["specificity"], reverse=True)[:limit]
    by_node: OrderedDict[str, list[int]] = OrderedDict()
    for row in candidates:
        by_node.setdefault(row["node_name"], []).append(row["flat_index"])
    order = {(node, index): rank for rank, row in enumerate(candidates) for node, index in [(row["node_name"], row["flat_index"])]}
    for node_name in by_node:
        by_node[node_name] = sorted(by_node[node_name], key=lambda index: order[(node_name, index)])
    return candidates, by_node


def calibrate_policies(calibration, exit_nodes, compute_fractions, targets):
    policies = {}
    for target in targets:
        remaining = np.ones(len(calibration), dtype=bool)
        thresholds = {}
        calibration_exits = np.full(len(calibration), "", dtype=object)
        for node_name in exit_nodes:
            gaps = calibration[f"{node_name}_gap"].to_numpy()
            correct = calibration[f"{node_name}_correct"].to_numpy().astype(bool)
            eligible = np.where(remaining)[0]
            if len(eligible) < 10:
                thresholds[node_name] = float("inf")
                continue
            values = np.unique(np.quantile(gaps[eligible], np.linspace(0, 1, 201)))
            best_threshold = float("inf")
            best_count = 0
            for threshold in values:
                chosen = eligible[gaps[eligible] >= threshold]
                if len(chosen) < 10:
                    continue
                if correct[chosen].mean() >= target and len(chosen) > best_count:
                    best_count = len(chosen)
                    best_threshold = float(threshold)
            thresholds[node_name] = best_threshold
            if np.isfinite(best_threshold):
                chosen = eligible[gaps[eligible] >= best_threshold]
                remaining[chosen] = False
                calibration_exits[chosen] = node_name
        policies[str(target)] = {
            "thresholds": thresholds,
            "calibration_exit_fraction": float((~remaining).mean()),
            "calibration_exits": calibration_exits.tolist(),
            "compute_fractions": compute_fractions,
        }
    return policies


def apply_policy(frame, exit_nodes, policy, compute_fractions):
    remaining = np.ones(len(frame), dtype=bool)
    predictions = frame.full_prediction.to_numpy().copy()
    exit_layer = np.full(len(frame), "full_model", dtype=object)
    compute = np.ones(len(frame), dtype=float)
    for node_name in exit_nodes:
        threshold = policy["thresholds"][node_name]
        if not np.isfinite(threshold):
            continue
        gaps = frame[f"{node_name}_gap"].to_numpy()
        chosen = np.where(remaining & (gaps >= threshold))[0]
        predictions[chosen] = frame[f"{node_name}_prediction"].to_numpy()[chosen]
        exit_layer[chosen] = node_name
        compute[chosen] = compute_fractions[node_name]
        remaining[chosen] = False
    return predictions, exit_layer, compute


def make_heatmap(
    probabilities,
    selected_rows,
    class_zero_heatmap,
    output,
):
    prototype_matrix = np.zeros((len(CLASS_NAMES), len(selected_rows)), dtype=float)
    for column, row in enumerate(selected_rows):
        probability = probabilities[row["node_name"]].reshape(len(CLASS_NAMES), -1)
        prototype_matrix[:, column] = probability[:, row["flat_index"]]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        gridspec_kw={"height_ratios": [1, 3]},
        constrained_layout=True,
    )
    image = axes[0].imshow(prototype_matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[0].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axes[0].set_title("Class-conditional activation probability for class-0-selective spatial neurons")
    axes[0].set_ylabel("Class")
    axes[1].imshow(class_zero_heatmap.T, aspect="auto", vmin=0, vmax=1, cmap="Greys_r")
    axes[1].set_title("Exact gate states across held-out class-0 images")
    axes[1].set_xlabel("Held-out class-0 image")
    axes[1].set_ylabel("Persistent spatial neuron")
    figure.colorbar(image, ax=axes[0], location="right", shrink=0.8, label="Activation probability")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_summary_figure(exit_summary, policy_summary, linear_layers, linear_policies, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(
        exit_summary.compute_fraction,
        exit_summary.prototype_accuracy,
        marker="o",
        color="#2A6F97",
        label="Bernoulli prototype",
    )
    axes[0].plot(
        linear_layers.compute_fraction,
        linear_layers.test_accuracy,
        marker="s",
        color="#2F855A",
        label="Linear gate readout",
    )
    axes[0].axhline(
        exit_summary.full_model_accuracy.iloc[0],
        color="#C44536",
        linestyle="--",
        label="Full model",
    )
    axes[0].set_xlabel("Cumulative model compute fraction")
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_title("Gate-prototype accuracy by depth")
    axes[0].legend(frameon=False)
    positions = np.arange(len(linear_policies))
    axes[1].bar(
        positions - 0.18,
        linear_policies.test_accuracy,
        width=0.36,
        color="#2A6F97",
        label="Accuracy",
    )
    axes[1].bar(
        positions + 0.18,
        1 - linear_policies.mean_compute_fraction,
        width=0.36,
        color="#F2C14E",
        label="Compute saved",
    )
    axes[1].set_xticks(positions, [f"{value:.0%}" for value in linear_policies.target_accuracy])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Calibration accuracy target")
    axes[1].set_title("Sequential early-exit policies")
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def per_class_gate_accuracy(linear_test, linear_layers):
    rows = []
    for layer in linear_layers.itertuples(index=False):
        for class_id, class_name in enumerate(CLASS_NAMES):
            group = linear_test[linear_test.label == class_id]
            rows.append(
                {
                    "node_name": layer.node_name,
                    "compute_fraction": layer.compute_fraction,
                    "class_id": class_id,
                    "class_name": class_name,
                    "accuracy": float(group[f"{layer.node_name}_correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_per_class_figure(per_class, linear_layers, output):
    matrix = (
        per_class.pivot(index="class_name", columns="node_name", values="accuracy")
        .reindex(index=CLASS_NAMES, columns=linear_layers.node_name)
        .to_numpy()
    )
    figure, axis = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", vmin=0.1, vmax=1, cmap="viridis")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xticks(
        range(len(linear_layers)),
        [f"{value:.0%}" for value in linear_layers.compute_fraction],
    )
    axis.set_xlabel("Cumulative model compute")
    axis.set_ylabel("Class")
    axis.set_title("Held-out accuracy from cumulative ReLU gate states")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.72 else "black",
            )
    figure.colorbar(image, ax=axis, label="Accuracy")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_set = datasets.CIFAR10(
        args.dataset_root, train=True, download=False, transform=transforms.ToTensor()
    )
    test_set = datasets.CIFAR10(
        args.dataset_root, train=False, download=False, transform=transforms.ToTensor()
    )
    splits = build_splits(args, train_set, test_set)
    model = load_model(Path(args.checkpoint), device)
    recorder = ActivationRecorder(model).to(device).eval()
    compute_fractions = estimate_compute_fractions(recorder)
    fit_loader = loader(train_set, splits["fit"], args)
    (
        probabilities,
        channel_profiles,
        shapes,
        fit_accuracy,
        fit_channel_features,
        fit_labels,
    ) = fit_prototypes(
        recorder,
        fit_loader,
        device,
        len(CLASS_NAMES),
        args.prototype_smoothing,
    )
    prototype_models = prototype_tensors(probabilities, device)
    exit_nodes = [
        name for name, site in recorder.sites.items() if site.is_exit_candidate
    ]
    selected_rows, selected_units = choose_top_units(
        probabilities, recorder.sites, args.heatmap_units
    )
    (
        calibration,
        _,
        _,
        _,
        calibration_channel_features,
        calibration_labels,
    ) = evaluate_split(
        recorder,
        loader(train_set, splits["calibration"], args),
        device,
        prototype_models,
        exit_nodes,
        {},
        False,
    )
    (
        test,
        _sample_channels,
        class_zero_heatmap,
        class_zero_indices,
        test_channel_features,
        test_labels,
    ) = evaluate_split(
        recorder,
        loader(test_set, splits["test"], args),
        device,
        prototype_models,
        exit_nodes,
        selected_units,
        True,
    )
    risk_targets = [float(value) for value in args.risk_targets.split(",") if value.strip()]
    policies = calibrate_policies(
        calibration, exit_nodes, compute_fractions, risk_targets
    )
    policy_rows = []
    for target in risk_targets:
        prediction, exit_layer, compute = apply_policy(
            test, exit_nodes, policies[str(target)], compute_fractions
        )
        policy_rows.append(
            {
                "target_accuracy": target,
                "test_accuracy": accuracy_score(test.label, prediction),
                "early_exit_fraction": float(np.mean(exit_layer != "full_model")),
                "mean_compute_fraction": float(compute.mean()),
                "full_model_accuracy": float(test.full_correct.mean()),
            }
        )
        test[f"exit_{target}_prediction"] = prediction
        test[f"exit_{target}_layer"] = exit_layer
        test[f"exit_{target}_compute"] = compute
    policy_summary = pd.DataFrame(policy_rows)
    exit_rows = []
    for node_name in exit_nodes:
        exit_rows.append(
            {
                "node_name": node_name,
                "display_name": recorder.sites[node_name].display_name,
                "compute_fraction": compute_fractions[node_name],
                "calibration_accuracy": float(calibration[f"{node_name}_correct"].mean()),
                "prototype_accuracy": float(test[f"{node_name}_correct"].mean()),
                "full_model_accuracy": float(test.full_correct.mean()),
            }
        )
    exit_summary = pd.DataFrame(exit_rows)
    (
        linear_layers,
        linear_policies,
        linear_calibration,
        linear_test,
        linear_policy_definitions,
    ) = fit_gate_linear_probes(
        fit_channel_features,
        fit_labels,
        calibration_channel_features,
        calibration_labels,
        test_channel_features,
        test_labels,
        channel_feature_ends(recorder, shapes),
        compute_fractions,
        test.full_prediction.to_numpy(),
        args.seed,
        risk_targets,
    )
    registry_rows = []
    for order, (node_name, site) in enumerate(recorder.sites.items()):
        registry_rows.append(
            {
                "order": order,
                "node_name": node_name,
                "display_name": site.display_name,
                "family": site.family,
                "module_path": site.module_path,
                "stage": site.stage,
                "shape": "x".join(str(value) for value in shapes[node_name]),
                "channels": shapes[node_name][0],
                "spatial_units": int(np.prod(shapes[node_name][1:])),
                "total_units": int(np.prod(shapes[node_name])),
                "exit_candidate": int(site.is_exit_candidate),
                "compute_fraction": compute_fractions[node_name],
            }
        )
    registry = pd.DataFrame(registry_rows)
    prototype_archive = {
        f"{node_name}__spatial": probability.astype(np.float16)
        for node_name, probability in probabilities.items()
    }
    prototype_archive.update(
        {
            f"{node_name}__channel": profile.astype(np.float16)
            for node_name, profile in channel_profiles.items()
        }
    )
    np.savez_compressed(output / "class_activation_prototypes.npz", **prototype_archive)
    np.savez_compressed(
        output / "class0_selected_gate_heatmap.npz",
        gates=class_zero_heatmap,
        sample_order=np.asarray(class_zero_indices),
    )
    registry.to_csv(output / "activation_site_registry.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "class0_selective_neurons.csv", index=False)
    calibration.to_csv(output / "calibration_predictions.csv", index=False)
    test.to_csv(output / "heldout_predictions.csv", index=False)
    exit_summary.to_csv(output / "layerwise_gate_prototype_accuracy.csv", index=False)
    policy_summary.to_csv(output / "early_exit_policy_summary.csv", index=False)
    linear_layers.to_csv(output / "linear_gate_probe_accuracy.csv", index=False)
    linear_policies.to_csv(output / "linear_gate_early_exit_summary.csv", index=False)
    linear_calibration.to_csv(output / "linear_gate_calibration_predictions.csv", index=False)
    linear_test.to_csv(output / "linear_gate_heldout_predictions.csv", index=False)
    (output / "linear_gate_early_exit_policies.json").write_text(
        json.dumps(linear_policy_definitions, indent=2)
    )
    per_class = per_class_gate_accuracy(linear_test, linear_layers)
    per_class.to_csv(output / "linear_gate_per_class_accuracy.csv", index=False)
    (output / "early_exit_policies.json").write_text(json.dumps(policies, indent=2))
    make_heatmap(
        probabilities,
        selected_rows,
        class_zero_heatmap,
        output / "class0_activation_fingerprint.png",
    )
    make_summary_figure(
        exit_summary,
        policy_summary,
        linear_layers,
        linear_policies,
        output / "gate_exit_summary.png",
    )
    make_per_class_figure(
        per_class,
        linear_layers,
        output / "gate_exit_per_class.png",
    )
    split_frame = pd.concat(
        [
            pd.DataFrame({"split": split_name, "dataset_index": indices})
            for split_name, indices in splits.items()
        ],
        ignore_index=True,
    )
    split_frame.to_csv(output / "split_indices.csv", index=False)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": {
            key: value
            for key, value in torch.load(
                args.checkpoint, map_location="cpu", weights_only=False
            ).items()
            if key not in {"net", "optimizer", "scheduler"}
        },
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "fit_per_class": args.fit_per_class,
        "calibration_per_class": args.calibration_per_class,
        "test_per_class": args.test_per_class,
        "fit_accuracy": fit_accuracy,
        "heldout_full_model_accuracy": float(test.full_correct.mean()),
        "activation_state": "exact post-ReLU output > 0",
        "prototype": "per-spatial-neuron class-conditional Bernoulli probability",
        "early_exit_score": "cumulative mean Bernoulli log likelihood",
        "linear_gate_readout": "averaged logistic SGD classifier on cumulative per-channel active fractions",
        "seed": args.seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nLayerwise gate-prototype accuracy")
    print(exit_summary.to_string(index=False))
    print("\nEarly-exit policies")
    print(policy_summary.to_string(index=False))
    print("\nLinear gate readouts")
    print(linear_layers.to_string(index=False))
    print("\nLinear gate early-exit policies")
    print(linear_policies.to_string(index=False))


if __name__ == "__main__":
    main()
