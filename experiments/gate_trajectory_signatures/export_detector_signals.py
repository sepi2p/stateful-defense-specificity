#!/usr/bin/env python3
"""Extract the per-query detector quantities that the operating procedure reads (operating_profile.load_signals)
from the traces into one compressed file per source directory, <source>/detector_signals.npz, with the key
'<trace file name>/<array>' and the dtypes of the traces. The released repository carries these files
instead of the traces, so that analyze_operating_profile.py runs without them and reproduces its outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.analyze_operating_profile import SETTINGS  # noqa: E402
from experiments.gate_trajectory_signatures.operating_profile import SIGNAL_KEYS, SIGNALS_FILE  # noqa: E402


def main():
    for setting in SETTINGS.values():
        splits = {setting["calibration"], setting["held_out"]}
        for root, names in setting["sources"]:
            arrays = {}
            for p in sorted(Path(root).glob("sessions_shard*.jsonl")):
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if names.get(r["objective"]) is None or r["split"] not in splits:
                        continue
                    t = np.load(Path(root) / r["trace"])
                    for k in SIGNAL_KEYS:
                        arrays[f"{Path(r['trace']).name}/{k}"] = t[k]
            out = Path(root) / SIGNALS_FILE
            np.savez_compressed(out, **arrays)
            print(f"{out}: {len(arrays) // len(SIGNAL_KEYS)} sessions, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
