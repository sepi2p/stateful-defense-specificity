#!/usr/bin/env python3
"""Checks of the procedure in operating_profile.py on synthetic sessions (no data needed).

Run: python experiments/gate_trajectory_signatures/test_operating_profile.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.operating_profile import (  # noqa: E402
    INF, Session, clopper_pearson, evaluate, first_alarm, rates, select)


def bl(counts, workload, split="calibration", image=0, success=INF):
    return Session(workload, split, image, success, {"blacklight_counts": np.asarray(counts, dtype=np.int32)})


def gw(idx, native, score, workload="attack", success=INF):
    sig = {"gwad_plus_idx": np.asarray(idx), "gwad_plus_native": np.asarray(native, bool),
           "gwad_plus_score": np.asarray(score, float)}
    return Session(workload, "calibration", 0, success, sig)


def test_boundary_cases():
    s = bl([0, 30, 50], "attack", success=2)
    assert first_alarm(s.signals, "blacklight", 25) == 2.0          # alarm and success on the same query
    r = rates([s], "blacklight", 25, ())
    assert r["attack"]["timely"] == 0                                # equal index is not timely
    assert first_alarm(bl([0, 10], "g").signals, "blacklight", 25) == INF  # no alarm
    early = bl([30, 0, 0], "attack", success=3)
    assert rates([early], "blacklight", 25, ())["attack"]["timely"] == 1   # alarm before success
    late = bl([0, 0, 0, 30], "attack", success=2)
    assert rates([late], "blacklight", 25, ())["attack"]["timely"] == 0    # success before alarm
    none = bl([30], "attack")                                        # attack never succeeds
    r = rates([none], "blacklight", 25, ())
    assert r["attack"]["successful"] == 0 and math.isnan(select([none, bl([0], "g")], "blacklight", ("g",), 0.01,
                                                                    grid=(25,))["table"][0]["R"])
    short = gw([], [], [], success=100)                              # session ends before the warm-up
    assert first_alarm(short.signals, "gwad_plus", "native") == INF
    assert rates([short], "gwad_plus", "native", ())["attack"] == {"sessions": 1, "successful": 1, "timely": 0,
                                                                    "alarm_any": 0}


def test_constraint_is_per_workload():
    # 99 quiet sessions of an easy workload cannot hide one hard workload whose every session alarms
    cal = [bl([0], "easy", image=i) for i in range(99)] + [bl([50], "hard", image=100)]
    out = select(cal, "blacklight", ("easy", "hard"), 0.05, grid=(25, 45))
    assert [t["feasible"] for t in out["table"]] == [False, False]
    assert out["status"] == "no feasible candidate in the declared set" and out["theta"] is None


def test_feasible_zero_detection_differs_from_infeasible():
    cal = [bl([30], "g"), bl([30], "attack", success=5)]
    out = select(cal, "blacklight", ("g",), 0.0, grid=(25, 40))
    assert out["status"] == "feasible" and out["theta"] == 40
    assert out["table"][1]["R"] == 0.0


def test_selection_ignores_held_out_outcomes():
    rng = np.random.default_rng(0)
    cal = [bl([int(c)], "g", image=i) for i, c in enumerate(rng.integers(0, 40, 100))]
    cal += [bl([int(c), 50], "attack", image=200 + i, success=2) for i, c in enumerate(rng.integers(20, 50, 50))]
    chosen = select(cal, "blacklight", ("g",), 0.05)["theta"]
    for flip in (0, 50):
        held = [bl([flip], "g", "evaluation", i) for i in range(100)]
        assert select(cal, "blacklight", ("g",), 0.05)["theta"] == chosen
        ev = evaluate(held, "blacklight", chosen, ("g",), 0.05)
        assert (ev["status"] == "held-out constraint violated") == (flip > chosen)


def test_tie_break_is_deterministic():
    cal = [bl([0], "g"), bl([30], "attack", success=5)]
    out = select(cal, "blacklight", ("g",), 0.0, grid=(25, 26, 27))
    assert out["theta"] == 27   # equal R and max F: the least sensitive candidate


def test_interval():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and abs(hi - 0.0362) < 1e-3
    lo, hi = clopper_pearson(100, 100)
    assert hi == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
