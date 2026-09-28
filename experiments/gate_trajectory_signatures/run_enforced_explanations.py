#!/usr/bin/env python3
"""X15, part 1: the library explanation clients of X14 when Blacklight rejects the flagged queries.

The three clients choose their queries without regard to the answers, and every query enters the
history of Blacklight whether it is flagged or not (add_img of the release). The set of flagged
queries under enforcement is therefore the recorded one. The library is run again with the seed of
the session; the stream is compared with the recorded stream (SHA-256 of all queries); answers are
served from the recorded outputs.

Modes
  replay  every query is answered (check: the recorded explanation is reproduced)
  h1      a rejected answer is replaced by the uniform distribution; library code unchanged
  h2      lime package only: the surrogate model is fitted on the answered samples alone

Output: one trace per session with the attribution maps of the modes, and sessions.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if os.environ.get("EXPLANATION_LIBS"):
    sys.path.insert(0, os.environ["EXPLANATION_LIBS"])

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402
from experiments.gate_trajectory_signatures.run_library_explanations import BUILDERS  # noqa: E402


class Replay(torch.nn.Module):
    """Serves recorded answers in the order of the queries; rejected answers are replaced if asked."""

    def __init__(self, logits: np.ndarray, rejected: np.ndarray, substitute: bool):
        super().__init__()
        self.probs = torch.softmax(torch.from_numpy(logits).float(), dim=1)
        self.rejected = torch.from_numpy(rejected.astype(bool))
        self.substitute = substitute
        self.pointer = 0
        self.digest = hashlib.sha256()
        self.overrun = False

    @torch.no_grad()
    def forward(self, x):
        n = x.shape[0]
        for image in x:
            self.digest.update(image.detach().contiguous().float().cpu().numpy().tobytes())
        lo, hi = self.pointer, self.pointer + n
        self.pointer = hi
        if hi > len(self.probs):
            self.overrun = True
            return torch.full((n, self.probs.shape[1]), 1.0 / self.probs.shape[1], device=x.device)
        out = self.probs[lo:hi].clone()
        if self.substitute:
            out[self.rejected[lo:hi]] = 1.0 / self.probs.shape[1]
        return out.to(x.device)


def run_mode(client, mode, x, label, seed, device, logits, rejected):
    replay = Replay(logits, rejected, substitute=(mode == "h1"))
    if mode == "h2":
        from lime import lime_base

        original = lime_base.LimeBase.explain_instance_with_data

        def answered_only(self, data, labels, distances, *args, **kwargs):
            keep = ~rejected[: len(data)]
            return original(self, data[keep], labels[keep], distances[keep], *args, **kwargs)

        lime_base.LimeBase.explain_instance_with_data = answered_only
        try:
            sal, info = BUILDERS[client](replay, x, label, seed, device)
        finally:
            lime_base.LimeBase.explain_instance_with_data = original
    else:
        sal, info = BUILDERS[client](replay, x, label, seed, device)
    complete = (replay.pointer == len(logits)) and not replay.overrun
    return sal.astype(np.float32), info, replay.digest.hexdigest(), complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="directory of the X14 sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="take the flagged queries from the run in this directory (same session ids, identical query "
                             "streams); needed for runs recorded without detectors")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "traces").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "imagenet":
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    else:
        dataset = datasets.CIFAR10("/home/sepi/data/cifar10", train=False, download=False, transform=transforms.ToTensor())
    rows = sorted(blacklight_rule.load_sessions(args.source_dir), key=lambda r: r["session_id"])
    masks = {r["session_id"]: r for r in blacklight_rule.load_sessions(args.mask_dir)} if args.mask_dir else {}
    rows = rows[args.shard :: args.num_shards]
    if args.max_sessions > 0:
        rows = rows[: args.max_sessions]
    out_path = args.output_dir / f"sessions_shard{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["session_id"] for line in out_path.read_text().splitlines() if line.strip()}
    for row in rows:
        if row["session_id"] in done:
            continue
        client = row["objective"]
        trace = np.load(args.source_dir / row["trace"])
        logits = trace["logits"]
        if args.mask_dir is not None:
            other = masks[row["session_id"]]
            if other["query_sha256"] != row["query_sha256"]:
                raise RuntimeError(f"{row['session_id']}: the query streams of the two runs differ")
            counts = np.load(args.mask_dir / other["trace"])["blacklight_counts"]
        else:
            counts = trace["blacklight_counts"]
        rejected = blacklight_rule.flagged(counts)
        x = dataset[int(row["dataset_index"])][0][None]
        label, seed = int(row["source_label"]), int(row["session_seed"])
        record = {"session_id": row["session_id"], "dataset_index": int(row["dataset_index"]), "objective": client,
                  "queries": int(len(counts)), "answered": int((~rejected).sum()), "first_query_answered": bool(not rejected[0]),
                  "first_rejected": int(np.flatnonzero(rejected)[0]) + 1 if rejected.any() else -1, "modes": {}}
        arrays = {"saliency_recorded": trace["saliency"], "rejected": rejected}
        for mode in ("replay", "h1") + (("h2",) if client == "lime_package" else ()):
            sal, info, digest, complete = run_mode(client, mode, x, label, seed, device, logits, rejected)
            arrays[f"saliency_{mode}"] = sal
            record["modes"][mode] = {"stream_identical": bool(complete and digest == row["query_sha256"]), "info": info}
            if mode == "replay":
                record["modes"][mode]["max_abs_difference_to_recorded"] = float(np.abs(sal - trace["saliency"]).max())
                record["modes"][mode]["scale_of_recorded"] = float(np.abs(trace["saliency"]).max())
        np.savez_compressed(args.output_dir / "traces" / f"{row['session_id']}.npz", **arrays)
        with out_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print("[DONE]", row["session_id"], "answered", record["answered"], "of", record["queries"],
              {m: v["stream_identical"] for m, v in record["modes"].items()}, flush=True)


if __name__ == "__main__":
    main()
