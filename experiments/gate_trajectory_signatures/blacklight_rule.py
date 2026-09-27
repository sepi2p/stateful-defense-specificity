#!/usr/bin/env python3
"""The published decision rule of Blacklight, applied to the stored per-query match counts.

Blacklight flags a query if its fingerprint shares MORE THAN T = 25 hashes with the fingerprint of an
earlier query (Li et al., USENIX Security 2022, Section 6.3: "shares more than T hash entries"; release,
example.ipynb: `match_num > threshold`, threshold = 25).

The session generators of this project recorded, as `detectors.blacklight.first_alarm`, the first query
with a count of AT LEAST T. That is one match more sensitive than the published rule. The match counts of
every query are stored in the traces (`blacklight_counts`), so the published rule can be applied afterwards.
Every analysis obtains Blacklight's decisions through this module; the session logs are left as recorded,
and the recorded value is kept as `first_alarm_recorded`.

Run as a script to write the sidecar file `blacklight_published_rule.csv` of a corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 25  # T of the paper, for every dataset
SIDECAR = "blacklight_published_rule.csv"


def flagged(counts: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Per-query decisions under the published rule."""
    return np.asarray(counts) > threshold


def first_alarm(counts: np.ndarray, threshold: float = THRESHOLD) -> int:
    """1-based index of the first flagged query, -1 if there is none."""
    hit = np.flatnonzero(flagged(counts, threshold))
    return int(hit[0]) + 1 if len(hit) else -1


def sidecar(root: Path, rows: list[dict] | None = None, refresh: bool = False) -> pd.DataFrame:
    """One row per session: decisions under the published rule and under the recorded one."""
    root = Path(root)
    path = root / SIDECAR
    if rows is None:
        rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        if set(frame.session_id) == {r["session_id"] for r in rows}:
            return frame
    out = []
    for r in rows:
        counts = np.load(root / r["trace"])["blacklight_counts"]
        out.append({"session_id": r["session_id"], "queries": len(counts), "max_match": int(counts.max()) if len(counts) else 0,
                    "first_alarm": first_alarm(counts), "flagged_queries": int(flagged(counts).sum()),
                    "first_alarm_recorded": int(r["detectors"]["blacklight"]["first_alarm"]),
                    "flagged_queries_recorded_rule": int((counts >= THRESHOLD).sum())})
    frame = pd.DataFrame(out)
    frame.to_csv(path, index=False)
    return frame


def apply(rows: list[dict], root: Path) -> list[dict]:
    """Replace Blacklight's first alarm in loaded session records by the published rule (in memory only)."""
    table = sidecar(root, rows).set_index("session_id")
    for r in rows:
        rec = table.loc[r["session_id"]]
        d = r["detectors"]["blacklight"]
        if "first_alarm_recorded" not in d:
            d["first_alarm_recorded"] = d["first_alarm"]
        d["first_alarm"] = int(rec.first_alarm)
        d["flagged_fraction"] = float(rec.flagged_queries) / max(int(rec.queries), 1)
    return rows


def load_sessions(root: Path) -> list[dict]:
    """Session records of a corpus with Blacklight's decisions under the published rule."""
    root = Path(root)
    rows = [json.loads(line) for p in sorted(root.glob("sessions_shard*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    return apply(rows, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for root in args.roots:
        frame = sidecar(root, refresh=args.refresh)
        changed = int((frame.first_alarm != frame.first_alarm_recorded).sum())
        lost = int(((frame.first_alarm < 0) & (frame.first_alarm_recorded > 0)).sum())
        print(f"{root.name}: {len(frame)} sessions; first alarm differs in {changed}; alarm only under the recorded rule in {lost}; "
              f"flagged queries {int(frame.flagged_queries.sum())} (recorded rule {int(frame.flagged_queries_recorded_rule.sum())}) "
              f"of {int(frame.queries.sum())}")


if __name__ == "__main__":
    main()
