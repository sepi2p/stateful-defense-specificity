#!/usr/bin/env python3
"""Short pre-tests before the long specificity runs (development data only).

T1/T2/T7: shuffled distinct-image streams through Blacklight and GWAD/GWAD+
          (false alarms on benign non-duplicate traffic; Blacklight cost).
T3:       Ljung-Box (Lee-Fang-Chang Phase 2) on their own benign constructions
          (i.i.d. N(0, 0.1^2) noise around one image; random images) vs attack
          sessions from the existing corpus.
T4/T5:    LIME-style explanation traffic (SLIC 40 segments, 1,000 Bernoulli(0.5)
          masks, segment-mean fill) on the 20 development images: detector
          alarms, and deletion-AUC utility vs random saliency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from skimage.segmentation import slic
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.lfc_detector import LB_ALPHA, LB_MIN_LEN, ljung_box_p  # noqa: E402,F401
from experiments.gate_trajectory_signatures.run_specificity_workloads import (  # noqa: E402
    BLACKLIGHT,
    BlacklightTracker,
    blacklight_salt,
)
from experiments.gate_trajectory_signatures.run_stateful_specificity_sessions import (  # noqa: E402
    Recorder,
    load_cifar_model,
    load_official_components,
)

def observe(model, device, delta_net, salt, images: torch.Tensor, label: int, batch: int = 64):
    """Stream images through the model with GWAD, GWAD+ and Blacklight observing."""
    recorder = Recorder.create(model, label, device, delta_net)
    blacklight = BlacklightTracker(salt)
    recorder.detectors["blacklight"] = blacklight
    started = time.time()
    for i in range(0, len(images), batch):
        recorder.submit_batch(images[i : i + batch].to(device))
    elapsed = time.time() - started
    logits = np.asarray(recorder.logits, dtype=np.float64)
    summary = {name: det.summary() for name, det in recorder.detectors.items()}
    counts = np.asarray(blacklight.counts)
    return {
        "seconds": elapsed,
        "logits": logits,
        "blacklight_first_alarm": summary["blacklight"]["first_alarm"],
        "blacklight_max_match": summary["blacklight"]["max_match"],
        "blacklight_query_flag_frac": float((counts >= BLACKLIGHT["threshold"]).mean()),
        "gwad_plus_windows": summary["gwad_plus"]["eligible_windows"],
        "gwad_plus_first_alarm": summary["gwad_plus"]["first_alarm"],
        "gwad_windows": summary["gwad"]["eligible_windows"],
        "gwad_first_alarm": summary["gwad"]["first_alarm"],
    }


def p_first_class(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p[:, int(logits[0].argmax())]


@torch.no_grad()
def lime_session(image: torch.Tensor, n_segments: int, samples: int, rng: np.random.Generator):
    hwc = image[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    segments = slic(hwc, n_segments=n_segments, compactness=10, start_label=0, channel_axis=-1)
    n_seg = int(segments.max()) + 1
    means = np.zeros_like(hwc)
    for s in range(n_seg):
        means[segments == s] = hwc[segments == s].mean(0)
    masks = rng.random((samples, n_seg)) < 0.5
    masks[0] = True  # LIME's first sample is the unperturbed image
    keep = masks[:, segments]  # samples x H x W
    queries = np.where(keep[..., None], hwc[None], means[None])
    return torch.from_numpy(queries).permute(0, 3, 1, 2).float(), masks, segments, means


def lime_weights(masks: np.ndarray, p: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
    distance = 1.0 - masks.mean(1)  # cosine-like distance to the all-on vector, as in LIME's default
    weights = np.sqrt(np.exp(-(distance**2) / kernel_width**2))
    x = np.hstack([masks.astype(np.float64), np.ones((len(masks), 1))])
    w = weights[:, None]
    ridge = np.eye(x.shape[1]) * 1.0
    ridge[-1, -1] = 0.0
    coef = np.linalg.solve(x.T @ (w * x) + ridge, x.T @ (weights * p))
    return coef[:-1]


@torch.no_grad()
def deletion_auc(model, device, image, segments, means, order, label) -> float:
    hwc = image[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    current = hwc.copy()
    frames = [current.copy()]
    for s in order:
        current[segments == s] = means[segments == s]
        frames.append(current.copy())
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device)
    p = F.softmax(model(batch), 1)[:, label].cpu().numpy()
    return float(np.trapz(p, dx=1.0 / (len(p) - 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis_outputs/stateful_specificity_paper_gate_20260924/manifest.csv")
    parser.add_argument("--corpus", default="analysis_outputs/stateful_specificity_workloads_20260925")
    parser.add_argument("--output-dir", default="analysis_outputs/stateful_specificity_pretests_20260925")
    parser.add_argument("--shuffled-sessions", type=int, default=20)
    parser.add_argument("--noise-sessions", type=int, default=20)
    parser.add_argument("--lime-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("resnet18_seed0", ROOT / "checkpoints/cifar10_resnet18_seed_study/resnet18_seed0.pt", device).eval()
    dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    manifest = pd.read_csv(args.manifest)
    used = set(manifest.dataset_index.astype(int))
    pool = np.array([i for i in range(len(dataset)) if i not in used])
    _, _, delta_net = load_official_components(torch.device("cpu"))
    salt = blacklight_salt()
    rng = np.random.default_rng(args.seed)
    results = {}

    # T1/T2/T7: shuffled distinct-image streams.
    rows = []
    for s in range(args.shuffled_sessions):
        idx = rng.choice(pool, 1024, replace=False)
        images = torch.stack([dataset[int(i)][0] for i in idx])
        r = observe(model, device, delta_net, salt, images, int(dataset[int(idx[0])][1]))
        r["lb_p_h20"] = ljung_box_p(p_first_class(r["logits"]), 20)
        r.pop("logits")
        rows.append({"session": s, **r})
        print(f"[shuffled {s}] bl_first={r['blacklight_first_alarm']} bl_max={r['blacklight_max_match']} "
              f"gwad+ windows={r['gwad_plus_windows']} first={r['gwad_plus_first_alarm']} t={r['seconds']:.1f}s", flush=True)
    shuffled = pd.DataFrame(rows)
    shuffled.to_csv(out / "t1_shuffled_streams.csv", index=False)

    # T3: Lee-Fang-Chang benign construction (i.i.d. Gaussian noise around one image).
    rows = []
    dev = manifest[manifest.split == "development"]
    for s, row in enumerate(dev.head(args.noise_sessions).itertuples(index=False)):
        base = dataset[int(row.dataset_index)][0]
        g = torch.Generator().manual_seed(args.seed + s)
        images = (base[None] + 0.1 * torch.randn((1024, *base.shape), generator=g)).clamp(0, 1)
        r = observe(model, device, delta_net, salt, images, int(row.source_label))
        p = p_first_class(r.pop("logits"))
        rows.append({"session": s, **r, **{f"lb_p_h{h}": ljung_box_p(p, h) for h in (5, 10, 20, 40)}})
        print(f"[noise {s}] bl_first={r['blacklight_first_alarm']} gwad+ first={r['gwad_plus_first_alarm']} "
              f"lb_p(h20)={rows[-1]['lb_p_h20']:.3f}", flush=True)
    noise = pd.DataFrame(rows)
    noise.to_csv(out / "t3_noise_streams.csv", index=False)

    corpus = Path(args.corpus)
    sessions = pd.DataFrame([json.loads(line) for p in sorted(corpus.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()])
    lb_rows = []
    for objective in ("attack", "restore", "random_walk"):
        sub = sessions[(sessions.objective == objective) & (sessions.split == "calibration")].head(100)
        for r in sub.itertuples(index=False):
            p = p_first_class(np.load(corpus / r.trace)["logits"].astype(np.float64))
            lb_rows.append({"objective": objective, **{f"lb_p_h{h}": ljung_box_p(p, h) for h in (5, 10, 20, 40)}})
    lb_corpus = pd.DataFrame(lb_rows)
    lb_corpus.to_csv(out / "t3_corpus_ljung_box.csv", index=False)

    # T4/T5: LIME-style explanation traffic on development images.
    rows = []
    for s, row in enumerate(dev.itertuples(index=False)):
        image = dataset[int(row.dataset_index)][0][None]
        label = int(row.source_label)
        queries, masks, segments, means = lime_session(image, 40, args.lime_samples, rng)
        r = observe(model, device, delta_net, salt, queries, label)
        logits = r.pop("logits")
        z = logits - logits.max(1, keepdims=True)
        p = (np.exp(z) / np.exp(z).sum(1, keepdims=True))[:, label]
        coef = lime_weights(masks, p)
        lime_order = np.argsort(-coef)
        auc_lime = deletion_auc(model, device, image, segments, means, lime_order, label)
        auc_rand = float(np.mean([deletion_auc(model, device, image, segments, means, rng.permutation(len(coef)), label) for _ in range(10)]))
        rows.append({"session": s, "n_segments": len(coef), **r, "deletion_auc_lime": auc_lime,
                     "deletion_auc_random": auc_rand, "lb_p_h20": ljung_box_p(p_first_class(logits), 20)})
        print(f"[lime {s}] segs={len(coef)} bl_first={r['blacklight_first_alarm']} bl_flag={r['blacklight_query_flag_frac']:.2f} "
              f"gwad+ windows={r['gwad_plus_windows']} first={r['gwad_plus_first_alarm']} del_auc {auc_lime:.3f} vs rand {auc_rand:.3f} "
              f"t={r['seconds']:.1f}s", flush=True)
    lime = pd.DataFrame(rows)
    lime.to_csv(out / "t4_lime_streams.csv", index=False)

    def alarm_rate(frame, col):
        return float((frame[col] > 0).mean())

    results = {
        "T1_blacklight_shuffled_alarm_rate": alarm_rate(shuffled, "blacklight_first_alarm"),
        "T1_blacklight_shuffled_max_match_median": float(shuffled.blacklight_max_match.median()),
        "T2_gwad_plus_shuffled_alarm_rate": alarm_rate(shuffled, "gwad_plus_first_alarm"),
        "T2_gwad_plus_shuffled_median_windows": float(shuffled.gwad_plus_windows.median()),
        "T2_gwad_shuffled_alarm_rate": alarm_rate(shuffled, "gwad_first_alarm"),
        "T7_seconds_per_shuffled_session": float(shuffled.seconds.median()),
        "T3_noise_blacklight_alarm_rate": alarm_rate(noise, "blacklight_first_alarm"),
        "T3_noise_gwad_plus_alarm_rate": alarm_rate(noise, "gwad_plus_first_alarm"),
        "T3_noise_lb_flag_rate": {h: float((noise[f"lb_p_h{h}"] < LB_ALPHA).mean()) for h in (5, 10, 20, 40)},
        "T3_shuffled_lb_flag_rate_h20": float((shuffled.lb_p_h20 < LB_ALPHA).mean()),
        "T3_corpus_lb_flag_rate": {
            obj: {h: float((g[f"lb_p_h{h}"] < LB_ALPHA).mean()) for h in (5, 10, 20, 40)}
            for obj, g in lb_corpus.groupby("objective")
        },
        "T4_lime_blacklight_alarm_rate": alarm_rate(lime, "blacklight_first_alarm"),
        "T4_lime_blacklight_median_first_alarm": float(lime.blacklight_first_alarm.median()),
        "T4_lime_gwad_plus_alarm_rate": alarm_rate(lime, "gwad_plus_first_alarm"),
        "T4_lime_gwad_plus_median_windows": float(lime.gwad_plus_windows.median()),
        "T4_lime_lb_flag_rate_h20": float((lime.lb_p_h20 < LB_ALPHA).mean()),
        "T5_deletion_auc_lime_median": float(lime.deletion_auc_lime.median()),
        "T5_deletion_auc_random_median": float(lime.deletion_auc_random.median()),
        "T5_lime_better_than_random_frac": float((lime.deletion_auc_lime < lime.deletion_auc_random).mean()),
        "T7_seconds_per_lime_session": float(lime.seconds.median()),
    }
    (out / "pretest_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
