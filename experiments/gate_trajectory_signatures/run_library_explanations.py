#!/usr/bin/env python3
"""X14: explanation clients as implemented by widely used libraries, against the stateful detectors.

The explanation clients of X2 and X10 are our own implementations of the four query designs. Here the
query streams come from library code that we do not control:

  lime_package       lime 0.2.0.1 (the implementation of the method's authors), lime_image with its
                     default settings: quickshift segmentation, 1,000 samples, batches of 10, hidden
                     superpixels replaced by their mean colour, explanations for the top 5 labels
  captum_kernelshap  Captum 0.9.0, KernelShap on SLIC superpixels (50 segments, compactness 30, sigma 3,
                     the settings of the image example in the SHAP documentation), 1,000 samples,
                     hidden superpixels replaced by the mean colour of the image
  captum_occlusion   Captum 0.9.0, Occlusion with the settings of the Captum tutorial for ImageNet
                     (window 15 x 15, stride 8, baseline 0); at 32 x 32 pixels window 4 x 4, stride 2

Every tensor that the library passes to the model is recorded, in the order in which it is passed, and
the recorded stream is shown to the detectors exactly as the streams of the other corpora are
(run_specificity_controls.run_stream). Trace and log format as in run_explanation_clients.py.

The libraries are imported from the environment, or from the directory named by EXPLANATION_LIBS.
"""

from __future__ import annotations

import argparse
import json
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
if os.environ.get("EXPLANATION_LIBS"):
    sys.path.insert(0, os.environ["EXPLANATION_LIBS"])

from experiments.gate_trajectory_signatures.run_explanation_clients import (  # noqa: E402
    deletion_auc,
    session_seed,
    smooth_random_saliency,
)
from experiments.gate_trajectory_signatures.run_specificity_controls import run_stream  # noqa: E402
from experiments.gate_trajectory_signatures.run_specificity_workloads import blacklight_salt  # noqa: E402
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    atomic_json,
    load_cifar_model,
    load_official_components,
)

CLIENTS = ("lime_package", "captum_kernelshap", "captum_occlusion")


class Capture(torch.nn.Module):
    """Passes inputs to the model and keeps a copy of every input, in order."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.queries: list[torch.Tensor] = []

    @torch.no_grad()
    def forward(self, x):
        self.queries.append(x.detach().float().cpu().clone())
        return F.softmax(self.model(x.float()), dim=1)

    def stream(self) -> torch.Tensor:
        return torch.cat(self.queries)


def client_lime_package(capture, x, label, seed, device):
    from lime import lime_image

    def classifier_fn(images):
        batch = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float().to(device)
        return capture(batch).cpu().numpy()

    image = x[0].permute(1, 2, 0).numpy().astype(np.float64)
    explainer = lime_image.LimeImageExplainer(random_state=seed % (2**31))
    explanation = explainer.explain_instance(image, classifier_fn, random_seed=seed % (2**31))  # every other setting: default
    weights = dict(explanation.local_exp[label]) if label in explanation.local_exp else {}
    saliency = np.zeros(explanation.segments.shape, dtype=np.float64)
    for segment, weight in weights.items():
        saliency[explanation.segments == segment] = weight
    return saliency, {"segments": int(len(np.unique(explanation.segments))), "label_explained": bool(label in explanation.local_exp)}


def client_captum_kernelshap(capture, x, label, seed, device):
    from captum.attr import KernelShap
    from skimage.segmentation import slic

    image = x[0].permute(1, 2, 0).numpy().astype(np.float64)
    segments = slic(image, n_segments=50, compactness=30, sigma=3, start_label=0)
    mask = torch.from_numpy(segments).long()[None, None].expand(1, x.shape[1], -1, -1).to(device)
    baseline = x.mean(dim=(2, 3), keepdim=True).expand_as(x).to(device)
    torch.manual_seed(seed % (2**31))
    np.random.seed(seed % (2**31))
    attribution = KernelShap(capture).attribute(x.to(device), baselines=baseline, target=label, feature_mask=mask,
                                                n_samples=1000, perturbations_per_eval=16)
    return attribution[0].sum(0).cpu().numpy().astype(np.float64), {"segments": int(len(np.unique(segments)))}


def client_captum_occlusion(capture, x, label, seed, device):
    from captum.attr import Occlusion

    window, stride = (15, 8) if x.shape[-1] > 64 else (4, 2)
    attribution = Occlusion(capture).attribute(x.to(device), sliding_window_shapes=(3, window, window),
                                               strides=(3, stride, stride), baselines=0, target=label, perturbations_per_eval=16)
    return attribution[0].sum(0).cpu().numpy().astype(np.float64), {"window": window, "stride": stride}


BUILDERS = {"lime_package": client_lime_package, "captum_kernelshap": client_captum_kernelshap,
            "captum_occlusion": client_captum_occlusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt"))
    parser.add_argument("--clients", default=",".join(CLIENTS))
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--lfc-seed", type=int, default=20260926)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--one-image-per-class", action="store_true")
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], default="cifar10")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--splits", default="evaluation")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--imagenet-model", choices=["resnet50", "convnext_tiny"], default="resnet50")
    parser.add_argument("--no-detectors", action="store_true",
                        help="record the model's outputs only; the queries do not depend on the answers, so the detector "
                             "results are those of the run with the same images and seed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blacklight_params, lfc_params, salt_shape = None, None, (32, 32, 3)
    if args.dataset == "imagenet":
        from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50

        from experiments.gate_trajectory_signatures.lfc_detector import LFC_IMAGENET
        from experiments.gate_trajectory_signatures.run_specificity_workloads import BLACKLIGHT_IMAGENET
        net = (convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if args.imagenet_model == "convnext_tiny"
               else resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
        model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), net).to(device).eval()
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
        blacklight_params, lfc_params, salt_shape = BLACKLIGHT_IMAGENET, LFC_IMAGENET, (224, 224, 3)
    else:
        model = load_cifar_model("resnet18_seed0", args.checkpoint, device).eval()
        dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    subset = manifest[manifest.split.isin([v.strip() for v in args.splits.split(",")])]
    if args.one_image_per_class:
        subset = subset.groupby("source_label").head(1)
    if args.max_images > 0:
        subset = subset.head(args.max_images)
    subset = subset.iloc[args.shard :: args.num_shards]
    _, _, delta_net = load_official_components(torch.device("cpu"))
    salt = blacklight_salt(salt_shape)
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    import captum
    import lime  # noqa: F401
    from importlib.metadata import version
    atomic_json(args.output_dir / f"metadata_shard{args.shard}.json", {
        "args": {k: str(v) for k, v in vars(args).items()}, "captum": captum.__version__, "lime": version("lime"),
        "torch": torch.__version__})
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
            capture = Capture(model)
            sal, info = BUILDERS[client](capture, x, label, seed, device)
            queries = capture.stream()
            result, arrays = run_stream(model, device, delta_net, salt, queries, label, lfc_seed=args.lfc_seed,
                                        batch=16 if args.dataset == "imagenet" else 64,
                                        blacklight_params=blacklight_params, lfc_params=lfc_params,
                                        observe_detectors=not args.no_detectors)
            auc = deletion_auc(model, device, x, sal, label, rng, fill="mean")
            auc_random = float(np.mean([deletion_auc(model, device, x, smooth_random_saliency(rng, h=sal.shape[0]), label, rng, fill="mean")
                                        for _ in range(10)]))
            auc_blur = deletion_auc(model, device, x, sal, label, rng, fill="blur")
            auc_random_pixel_blur = float(np.mean([deletion_auc(model, device, x, rng.random(sal.shape), label, rng, fill="blur")
                                                   for _ in range(10)]))
            trace = args.output_dir / "traces" / f"{sid}.npz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace, **arrays, saliency=sal.astype(np.float32))
            record = {"session_id": sid, "split": row.split, "dataset_index": int(row.dataset_index),
                      "source_label": label, "workload": "clean", "objective": client, "optimizer": "none",
                      "session_seed": seed, "elapsed_seconds": time.time() - started, "client_info": info,
                      "deletion_auc": auc, "deletion_auc_random": auc_random,
                      "deletion_auc_blur": auc_blur, "deletion_auc_random_pixel_blur": auc_random_pixel_blur,
                      "trace": str(trace.relative_to(args.output_dir))} | result
            with summary_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            d = result["detectors"]
            alarms = (f"bl={d['blacklight']['first_alarm_published_rule']} gwad+={d['gwad_plus']['first_alarm']} "
                      if d else "detectors off ")
            print(f"[DONE] {sid} q={result['calls']} {alarms}del {auc:.3f}/{auc_random:.3f} t={record['elapsed_seconds']:.1f}s {info}",
                  flush=True)


if __name__ == "__main__":
    main()
