#!/usr/bin/env python3
"""Evaluation of a stateful detector under benign-alarm constraints (reporting and configuration selection).

A setting fixes the classifier, the attack and its budget, the detector's state scope (reset per session
here), the permitted benign workloads G and the alarm budget alpha. For a configuration theta:

  F_g(theta)  share of the sessions of benign workload g in which the detector raises any alarm
  R(theta)    among attack sessions that succeed within the budget, the share whose first alarm comes
              strictly before the first misclassified query (timely alarm); undefined without successes

Procedure (docs/specificity_workloads_preregistration.md, X16):
  1. candidates Theta and G are declared before any calibration result is seen;
  2. feasible = {theta : F_g(theta) <= alpha for every g in G} on the calibration split;
  3. theta* maximizes R on the calibration split among feasible candidates; ties go to the smaller
     largest F_g, then to the less sensitive candidate (later in the declared order);
     without calibration attack sessions, theta* is the most sensitive feasible candidate;
  4. theta* is frozen and evaluated once on the held-out split.
Outcomes are reported with their status: "no feasible candidate in the declared set", "feasible",
"held-out constraint violated", or "not estimable". A feasible candidate with R = 0 is not the same as
an empty feasible set.

The first alarm of a session is replayed from its trace, so any candidate can be evaluated without
running the detector again. Indices are 1-based query indices; no alarm is +inf.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

INF = math.inf
BLACKLIGHT_GRID = tuple(range(25, 51))                                  # alarm if count > T; released T = 25; T = 50 never alarms
GWAD_GRID = ("native",) + tuple(round(0.25 * k, 2) for k in range(0, 49))  # score > c, c = 0 .. 12; released: argmax


def candidates(detector: str) -> tuple:
    """Declared candidate configurations, in order of decreasing sensitivity (released first)."""
    if detector == "blacklight":
        return BLACKLIGHT_GRID
    if detector in ("gwad", "gwad_plus"):
        return GWAD_GRID
    raise ValueError(detector)


SIGNAL_KEYS = ("blacklight_counts",) + tuple(f"{n}_{k}" for n in ("gwad_plus", "gwad")
                                             for k in ("scores", "query_indices", "predictions"))
SIGNALS_FILE = "detector_signals.npz"   # the arrays above for every session, written by export_detector_signals.py
_signal_files: dict = {}


def load_signals(root: Path, trace: str) -> dict:
    path = Path(root) / trace
    if path.exists():
        t = np.load(path)
    else:
        # without the per-query traces: the extract of the detector quantities released with the session logs
        f = Path(root) / SIGNALS_FILE
        if f not in _signal_files:
            _signal_files[f] = np.load(f)
        t = {k: _signal_files[f][f"{Path(trace).name}/{k}"] for k in SIGNAL_KEYS}
    out = {"blacklight_counts": t["blacklight_counts"].astype(np.int32)}
    for name in ("gwad_plus", "gwad"):
        p = t[f"{name}_scores"].astype(np.float64)
        out[f"{name}_idx"] = t[f"{name}_query_indices"].astype(np.int64)
        out[f"{name}_score"] = -np.log10(np.clip(1.0 - p, 1e-12, 1.0))
        out[f"{name}_native"] = t[f"{name}_predictions"] != 0
    return out


def first_alarm(sig: dict, detector: str, theta) -> float:
    if detector == "blacklight":
        hit = np.flatnonzero(sig["blacklight_counts"] > theta)
        return float(hit[0] + 1) if len(hit) else INF
    flags = sig[f"{detector}_native"] if theta == "native" else sig[f"{detector}_score"] > theta
    hit = np.flatnonzero(flags)
    return float(sig[f"{detector}_idx"][hit[0]]) if len(hit) else INF


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


@dataclass
class Session:
    workload: str          # benign workload name, or "attack"
    split: str
    image: int             # source image (independent unit)
    success: float         # 1-based first misclassified query of an attack session, +inf if none
    signals: dict = field(repr=False)


def rates(sessions: list[Session], detector: str, theta, workloads) -> dict:
    """Per-workload alarm counts and the timely-alarm count on successful attacks."""
    out = {"benign": {}, "attack": {}}
    for g in workloads:
        s = [x for x in sessions if x.workload == g]
        k = sum(first_alarm(x.signals, detector, theta) < INF for x in s)
        out["benign"][g] = (k, len(s))
    attacks = [x for x in sessions if x.workload == "attack"]
    succ = [x for x in attacks if x.success < INF]
    timely = sum(first_alarm(x.signals, detector, theta) < x.success for x in succ)
    out["attack"] = {"sessions": len(attacks), "successful": len(succ), "timely": timely,
                     "alarm_any": sum(first_alarm(x.signals, detector, theta) < INF for x in attacks)}
    return out


def feasible(r: dict, alpha: float) -> bool:
    return all(n > 0 and k <= math.floor(alpha * n + 1e-9) for k, n in r["benign"].values())


def timely_rate(r: dict) -> float:
    a = r["attack"]
    return a["timely"] / a["successful"] if a["successful"] else math.nan


def select(calibration: list[Session], detector: str, workloads, alpha: float, grid=None) -> dict:
    """Steps 2-3 of the procedure; uses the calibration sessions only."""
    grid = candidates(detector) if grid is None else grid
    table = []
    for order, theta in enumerate(grid):
        r = rates(calibration, detector, theta, workloads)
        max_f = max(k / n for k, n in r["benign"].values()) if r["benign"] else math.nan
        table.append({"theta": theta, "order": order, "feasible": feasible(r, alpha), "R": timely_rate(r),
                      "max_F": max_f, "rates": r})
    ok = [t for t in table if t["feasible"]]
    if not ok:
        return {"status": "no feasible candidate in the declared set", "theta": None, "table": table}
    has_attacks = any(t["rates"]["attack"]["successful"] > 0 for t in ok)
    if has_attacks:
        best = max(ok, key=lambda t: (t["R"], -t["max_F"], t["order"]))
    else:
        best = min(ok, key=lambda t: t["order"])
    return {"status": "feasible", "theta": best["theta"], "table": table, "selected_on_attacks": has_attacks}


def timely_cluster_ci(sessions: list[Session], detector: str, theta, reps: int = 2000, seed: int = 0) -> tuple:
    """Percentile interval of R from resampling source images (both starts of an image stay together).

    When every or no successful attack is timely the bootstrap interval has zero width; the exact
    Clopper-Pearson interval over the source images with a successful attack is returned instead."""
    succ = [x for x in sessions if x.workload == "attack" and x.success < INF]
    if not succ:
        return (math.nan, math.nan), "undefined"
    by_image = {}
    for x in succ:
        by_image.setdefault(x.image, []).append(first_alarm(x.signals, detector, theta) < x.success)
    images = list(by_image)
    k = sum(sum(v) for v in by_image.values())
    if k in (0, len(succ)):
        return clopper_pearson(k if k == 0 else len(images), len(images)), "exact over source images"
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(images), len(images))
        hits = [by_image[images[i]] for i in pick]
        n = sum(len(h) for h in hits)
        draws.append(sum(sum(h) for h in hits) / n)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))), "bootstrap over source images"


def evaluate(held_out: list[Session], detector: str, theta, workloads, alpha: float, extra_workloads=()) -> dict:
    """Step 4: the frozen configuration on the held-out split. Benign rates (one session per image and workload)
    get Clopper-Pearson intervals; R gets an interval from resampling source images."""
    r = rates(held_out, detector, theta, tuple(workloads) + tuple(extra_workloads))
    r_ci, r_method = timely_cluster_ci(held_out, detector, theta)
    in_profile = {g: r["benign"][g] for g in workloads}
    violated = [g for g, (k, n) in in_profile.items() if n == 0 or k > math.floor(alpha * n + 1e-9)]
    a = r["attack"]
    return {
        "theta": theta,
        "status": "held-out constraint violated" if violated else "within budget on held-out split",
        "violated": violated,
        "benign": {g: {"alarms": k, "sessions": n, "rate": k / n if n else math.nan, "ci": clopper_pearson(k, n),
                       "in_profile": g in workloads} for g, (k, n) in r["benign"].items()},
        "attack": {**a, "success_rate": a["successful"] / a["sessions"] if a["sessions"] else math.nan,
                   "R": timely_rate(r), "R_ci": r_ci, "R_ci_method": r_method,
                   "R_ci_sessions_exact": clopper_pearson(a["timely"], a["successful"]),
                   "timely_of_all": a["timely"] / a["sessions"] if a["sessions"] else math.nan},
    }


def sessions_from_logs(roots_and_names: list[tuple[Path, dict]], splits) -> list[Session]:
    """roots_and_names: (directory with sessions_shard*.jsonl, {objective in log: workload name or None})."""
    out = []
    for root, names in roots_and_names:
        for p in sorted(Path(root).glob("sessions_shard*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                name = names.get(r["objective"])
                if name is None or r["split"] not in splits:
                    continue
                success = float(r["first_success"]) if name == "attack" and r.get("first_success", -1) > 0 else INF
                out.append(Session(name, r["split"], int(r["dataset_index"]), success, load_signals(root, r["trace"])))
    return out


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
