#!/usr/bin/env python3
"""Attack-vs-benign separability for matched optimization workloads.

Implements docs/specificity_workloads_preregistration.md: detectors are fitted
(orientation or logistic model) on the fit split and scored on the evaluation
split, with source-image bootstrap intervals. Only detector-visible signals are
used: the query stream (GWAD, GWAD+, Blacklight) and the classifier's own logits
with the first-query prediction as reference label.

Revision 2 (2026-09-27): Blacklight's decisions follow the published rule (a query is
flagged if it shares MORE THAN 25 hashes with an earlier one, see blacklight_rule.py); the
first analysis counted 25 or more. A logistic model on the GWAD statistics alone was added.
Results are written to <root>/analysis_r2; <root>/analysis holds the first analysis.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402

BLACKLIGHT_THRESHOLD = blacklight_rule.THRESHOLD
FIXED_PREFIXES = (128, 256, 512, 1024)
FAMILIES = {
    "gwad_plus": lambda c: c.startswith("gwad_plus_"),
    "blacklight": lambda c: c.startswith("bl_"),
    "query_only_all": lambda c: c.startswith(("gwad", "bl_")),
    "output_trajectory": lambda c: c.startswith("out_"),
    # added in revision 2; placed last so that the bootstrap draws of the families above are unchanged
    "gwad": lambda c: c.startswith("gwad_") and not c.startswith("gwad_plus_"),
}
CANONICAL = ("gwad_plus_max", "bl_max", "out_pref_slope")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def slope(y: np.ndarray) -> float:
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]) if len(y) > 1 else 0.0


def features(trace, prefix: int) -> dict[str, float]:
    logits = trace["logits"][:prefix].astype(np.float64)
    probs = softmax(logits)
    ref = int(logits[0].argmax())
    p_ref = probs[:, ref]
    ref_margin = logits[:, ref] - np.delete(logits, ref, axis=1).max(axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    runner_up = int(np.delete(np.arange(logits.shape[1]), ref)[np.delete(logits[0], ref).argmax()])
    out = {
        "out_pref_last": p_ref[-1], "out_pref_min": p_ref.min(), "out_pref_slope": slope(p_ref),
        "out_margin_last_minus_first": ref_margin[-1] - ref_margin[0], "out_margin_min": ref_margin.min(),
        "out_margin_slope": slope(ref_margin), "out_gap_mean": float((top2[:, 1] - top2[:, 0]).mean()),
        "out_entropy_slope": slope(entropy), "out_label_changed": float((logits.argmax(1) != ref).any()),
        # Class dynamics beyond the reference label: which competitor is gaining.
        "out_runnerup_slope": slope(probs[:, runner_up]),
        "out_top_competitor_changes": float(np.count_nonzero(np.diff(np.argsort(-probs, axis=1)[:, 1]))),
    }
    counts = trace["blacklight_counts"][:prefix].astype(np.float64)
    out |= {"bl_max": counts.max(), "bl_mean": counts.mean(), "bl_frac_alarm": float(blacklight_rule.flagged(counts).mean())}
    for name in ("gwad", "gwad_plus"):
        keep = trace[f"{name}_query_indices"] < prefix
        scores = trace[f"{name}_scores"][keep].astype(np.float64)
        logc = -np.log10(np.clip(1.0 - scores, 1e-12, 1.0))
        hist = trace[f"{name}_histograms"][keep]
        out[f"{name}_has_window"] = float(len(scores) > 0)
        out[f"{name}_max"] = float(logc.max()) if len(logc) else np.nan
        out[f"{name}_mean"] = float(logc.mean()) if len(logc) else np.nan
        out[f"{name}_hist_mean_bin"] = (
            float((hist * np.arange(hist.shape[1])).sum(1).mean() / max(hist.sum(1).mean(), 1e-12)) if len(hist) else np.nan
        )
    return out


def bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, reps: int, rng) -> tuple[float, float]:
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(reps):
        take = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(y[take])) == 2:
            values.append(roc_auc_score(y[take], s[take]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("analysis_outputs/stateful_specificity_workloads_20260925"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--analysis-dir", default="analysis_r2")
    args = parser.parse_args()
    out_dir = args.root / args.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.DataFrame(blacklight_rule.load_sessions(args.root))
    sessions = sessions.set_index("session_id")

    @lru_cache(maxsize=None)
    def feats(session_id: str, prefix: int) -> dict:
        return features(np.load(args.root / sessions.loc[session_id, "trace"]), prefix)

    @lru_cache(maxsize=None)
    def first_label_change(session_id: str) -> int:
        """Number of calls strictly before the first output whose argmax differs from the first query's."""
        logits = np.load(args.root / sessions.loc[session_id, "trace"])["logits"]
        changed = np.flatnonzero(logits.argmax(1) != logits[0].argmax())
        return int(changed[0]) if len(changed) else len(logits)

    benign = sorted(set(sessions.objective) - {"attack"})
    rows = []
    for workload in ("denoise", "deblur"):
        for negative in benign:
            pos = sessions[(sessions.workload == workload) & (sessions.objective == "attack")]
            neg = sessions[(sessions.workload == workload) & (sessions.objective == negative)]
            paired = pos.reset_index().merge(neg.reset_index(), on=["dataset_index", "split"], suffixes=("_a", "_b"))
            for prefix_name in [*map(str, FIXED_PREFIXES), "attack_truncated", "pre_either_flip"]:
                records = []
                for r in paired.itertuples(index=False):
                    if prefix_name == "attack_truncated":
                        # Calls strictly before the attack's first label change (full session if it never flips).
                        prefix = int(r.first_success_a) - 1 if r.first_success_a > 0 else int(r.calls_a)
                        if prefix < 17:
                            continue
                    elif prefix_name == "pre_either_flip":
                        # Post hoc (NES CIFAR-10): calls before EITHER session's first label change, so neither
                        # prefix contains a flip and the label-change indicator carries no information.
                        prefix = min(first_label_change(r.session_id_a), first_label_change(r.session_id_b))
                        if prefix < 17:
                            continue
                    else:
                        prefix = int(prefix_name)
                    for sid, label in ((r.session_id_a, 1), (r.session_id_b, 0)):
                        records.append({"session_id": sid, "dataset_index": r.dataset_index, "split": r.split,
                                        "y": label, "prefix": prefix, **feats(sid, prefix)})
                frame = pd.DataFrame(records)
                fit, ev = frame[frame.split == "fit"], frame[frame.split == "evaluation"]
                cols = [c for c in frame.columns if c.startswith(("out_", "bl_", "gwad")) and not c.endswith("has_window")]
                if prefix_name == "pre_either_flip":
                    cols = [c for c in cols if c != "out_label_changed"]
                gwad_ready = bool(frame.gwad_plus_has_window.all())
                base = {"workload": workload, "negative": negative, "prefix": prefix_name,
                        "n_eval_pairs": int(ev.y.sum()), "median_prefix": float(ev.prefix.median())}
                rng = np.random.default_rng(0)
                for col in CANONICAL:
                    if col.startswith("gwad") and not gwad_ready:
                        rows.append(base | {"detector": f"single:{col}", "auroc": np.nan, "note": "no GWAD window yet"})
                        continue
                    sign = 1.0 if roc_auc_score(fit.y, fit[col]) >= 0.5 else -1.0
                    score = sign * ev[col].to_numpy()
                    lo, hi = bootstrap_auc(ev.y.to_numpy(), score, ev.dataset_index.to_numpy(), args.bootstrap, rng)
                    rows.append(base | {"detector": f"single:{col}", "auroc": roc_auc_score(ev.y, score), "lo": lo, "hi": hi})
                for family, pick in FAMILIES.items():
                    use = [c for c in cols if pick(c) and (gwad_ready or not c.startswith("gwad"))]
                    if not use:
                        rows.append(base | {"detector": f"logreg:{family}", "auroc": np.nan, "note": "no GWAD window yet"})
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
                    clf.fit(fit[use].fillna(0.0), fit.y)
                    score = clf.predict_proba(ev[use].fillna(0.0))[:, 1]
                    lo, hi = bootstrap_auc(ev.y.to_numpy(), score, ev.dataset_index.to_numpy(), args.bootstrap, rng)
                    rows.append(base | {"detector": f"logreg:{family}", "auroc": roc_auc_score(ev.y, score), "lo": lo, "hi": hi})
                print(f"[{workload} {negative} {prefix_name}] done", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "separability.csv", index=False)
    utility = sessions.groupby(["workload", "objective"]).agg(
        n=("calls", "size"), label_preserved=("label_preserved", "mean"), reached_target=("reached_target", "mean"),
        start_p=("start_p_label", "median"), final_p=("final_p_label", "median"),
        final_margin=("final_margin", "median"), psnr_gain=("psnr_gain", "median"),
        blacklight_alarm=("detectors", lambda d: np.mean([x["blacklight"]["first_alarm"] > 0 for x in d])),
        gwad_plus_native_alarm=("detectors", lambda d: np.mean([x["gwad_plus"]["first_alarm"] > 0 for x in d])),
    )
    utility.to_csv(out_dir / "workload_utility.csv")
    print(utility.round(3).to_string())
    table = result[result.detector.isin([f"logreg:{f}" for f in FAMILIES])]
    print(table.pivot_table(index=["negative", "detector"], columns=["workload", "prefix"], values="auroc").round(2).to_string())


if __name__ == "__main__":
    main()
