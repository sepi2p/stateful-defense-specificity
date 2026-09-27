#!/usr/bin/env python3
"""X3: operating-point table for stateful detectors on the specificity corpora.

For each detector, a session-level threshold is calibrated on the calibration
split of a benign reference control (shuffled test images; i.i.d. noise as the
alternative reference) at a target session FPR (1%, 0.1%), plus the detector's
native threshold. The table reports, on the evaluation split of every workload
and control, the blocked fraction (Wilson 95% CI) and the median first-alarm
query index among blocked sessions.

Revision 2 (2026-09-27): Blacklight follows its published rule, count > T with T = 25 (the first
analysis used count >= 25, so its "released" row was one match too sensitive and its rows calibrated
on shuffled images, threshold 26 under ">=", were the published rule). Every detector is calibrated on
each of the three controls; the calibration sizes and the rates achieved on the calibration split are
written next to the thresholds. Blacklight's session alarm is also reported for a sweep of T.
Output: analysis_outputs/stateful_specificity_operating_points_r2.

Session scores and first-alarm rules:
  blacklight   max match count; alarm at the first query whose count > threshold (released: 25)
  gwad_plus    max over windows of -log10(1 - p_attack); first window above threshold
               (native: the release's argmax decision); no window => never alarms
  gwad         same as gwad_plus without the screener
  ljung_box    Ljung-Box (h=20) on p_{c1} evaluated at 128/256/512/full calls; score = -log10(min p);
               alarm at the first checkpoint below threshold (native p < 0.025)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.pretest_detector_assumptions import ljung_box_p, p_first_class  # noqa: E402
from experiments.gate_trajectory_signatures.summarize_specificity_controls import wilson  # noqa: E402

CHECKPOINTS = (128, 256, 512, 100000)
NATIVE_BLACKLIGHT = 25
NATIVE_LB = -np.log10(0.025)


def load_sessions(root: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["root"] = str(root)
    return frame


def per_session_signals(row) -> dict:
    trace = np.load(Path(row.root) / row.trace)
    out = {"blacklight_counts": trace["blacklight_counts"].astype(np.int32)}
    for name in ("gwad_plus", "gwad"):
        scores = trace[f"{name}_scores"].astype(np.float64)
        out[f"{name}_idx"] = trace[f"{name}_query_indices"].astype(np.int32)
        out[f"{name}_score"] = -np.log10(np.clip(1.0 - scores, 1e-12, 1.0))
        out[f"{name}_native"] = trace[f"{name}_predictions"] != 0
    p = p_first_class(trace["logits"].astype(np.float64))
    lb = []
    for c in CHECKPOINTS:
        n = min(c, len(p))
        lb.append((n, -np.log10(max(ljung_box_p(p[:n], 20), 1e-300))))
        if n == len(p):
            break
    out["lb"] = lb
    return out


def first_alarm(sig: dict, detector: str, threshold) -> int:
    """1-based query index of the first alarm, or -1."""
    if detector == "blacklight":
        hit = np.flatnonzero(sig["blacklight_counts"] > threshold)
        return int(hit[0] + 1) if len(hit) else -1
    if detector in ("gwad_plus", "gwad"):
        flags = sig[f"{detector}_native"] if threshold == "native" else sig[f"{detector}_score"] > threshold
        hit = np.flatnonzero(flags)
        return int(sig[f"{detector}_idx"][hit[0]]) if len(hit) else -1
    if detector == "ljung_box":
        for n, score in sig["lb"]:
            if score > threshold:
                return int(n)
        return -1
    raise ValueError(detector)


def session_score(sig: dict, detector: str) -> float:
    if detector == "blacklight":
        return float(sig["blacklight_counts"].max())
    if detector in ("gwad_plus", "gwad"):
        s = sig[f"{detector}_score"]
        return float(s.max()) if len(s) else -np.inf
    return float(max(score for _, score in sig["lb"]))


def calibrate(scores: np.ndarray, target_fpr: float, detector: str):
    """Smallest threshold with empirical FPR <= target under the strict '>' rule that every detector uses."""
    finite = np.sort(scores[np.isfinite(scores)])
    if len(finite) == 0:
        return -np.inf  # reference never produces a score: any score alarms
    allowed = int(np.floor(target_fpr * len(scores)))
    # threshold at the (allowed+1)-th largest score so at most `allowed` sessions exceed it
    return finite[::-1][min(allowed, len(finite) - 1)] if allowed < len(finite) else -np.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, default=Path("analysis_outputs/stateful_specificity_workloads_20260925"))
    parser.add_argument("--controls", type=Path, default=Path("analysis_outputs/stateful_specificity_controls_20260925"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/stateful_specificity_operating_points_r2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.concat([load_sessions(args.workloads), load_sessions(args.controls)], ignore_index=True)
    sessions = sessions[sessions.split.isin(["calibration", "evaluation"])].reset_index(drop=True)
    signals = [per_session_signals(r) for r in sessions.itertuples(index=False)]
    detectors = ("blacklight", "gwad_plus", "gwad", "ljung_box")
    for d in detectors:
        sessions[f"score_{d}"] = [session_score(s, d) for s in signals]

    thresholds, calibration = {}, []
    for reference in ("shuffled", "noise", "sweep"):
        ref = sessions[(sessions.objective == reference) & (sessions.split == "calibration")]
        for target in (0.01, 0.001):
            for d in detectors:
                scores = ref[f"score_{d}"].to_numpy()
                thr = calibrate(scores, target, d)
                thresholds[(reference, target, d)] = thr
                calibration.append({"reference": reference, "target_fpr": target, "detector": d, "threshold": float(thr),
                                    "calibration_sessions": len(scores), "sessions_with_a_score": int(np.isfinite(scores).sum()),
                                    "allowed_sessions": int(np.floor(target * len(scores))),
                                    "achieved_on_calibration": float((scores > thr).mean())})
    pd.DataFrame(calibration).to_csv(args.output_dir / "calibration.csv", index=False)
    native = {"blacklight": NATIVE_BLACKLIGHT, "gwad_plus": "native", "gwad": "native", "ljung_box": NATIVE_LB}

    evaluation = sessions[sessions.split == "evaluation"]
    rows = []
    settings = [("native", None, d, native[d]) for d in detectors]
    settings += [(f"cal_{ref}", tgt, d, thr) for (ref, tgt, d), thr in thresholds.items()]
    for label, target, d, thr in settings:
        for (workload, objective), g in evaluation.groupby(["workload", "objective"]):
            alarms = np.array([first_alarm(signals[i], d, thr) for i in g.index])
            blocked = alarms > 0
            k, n = int(blocked.sum()), len(blocked)
            lo, hi = wilson(k, n)
            rows.append({
                "calibration": label, "target_fpr": target, "detector": d,
                "threshold": thr if isinstance(thr, str) else float(thr),
                "workload": workload, "objective": objective, "n": n,
                "blocked": k / n, "blocked_lo": lo, "blocked_hi": hi,
                "median_first_alarm": float(np.median(alarms[blocked])) if k else np.nan,
            })
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "operating_points.csv", index=False)
    sweep = []
    for threshold in (25, 30, 35, 40, 45, 48, 49):
        for objective, g in evaluation.groupby("objective"):
            alarms = np.array([first_alarm(signals[i], "blacklight", threshold) for i in g.index])
            sweep.append({"threshold": threshold, "objective": objective, "n": len(g), "blocked": float((alarms > 0).mean()),
                          "median_first_alarm": float(np.median(alarms[alarms > 0])) if (alarms > 0).any() else np.nan})
    pd.DataFrame(sweep).to_csv(args.output_dir / "blacklight_threshold_sweep.csv", index=False)
    (args.output_dir / "thresholds.json").write_text(json.dumps(
        {f"{r}|{t}|{d}": (v if isinstance(v, str) else float(v)) for (r, t, d), v in thresholds.items()}, indent=2))
    view = table[(table.calibration.isin(["native", "cal_shuffled"])) & (table.target_fpr.isin([None, 0.01]) | table.target_fpr.isna())]
    pivot = view.pivot_table(index=["workload", "objective"], columns=["detector", "calibration"], values="blocked").round(2)
    print(pivot.to_string())
    print(json.dumps({f"{r}|{t}|{d}": (v if isinstance(v, str) else round(float(v), 3)) for (r, t, d), v in thresholds.items()}, indent=1))


if __name__ == "__main__":
    main()
