#!/usr/bin/env python3
"""Check the in-process Blacklight port against the official InputTracker, query by query."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFICIAL = ROOT / "third_party/stateful_monitoring_baselines/blacklight"

from experiments.gate_trajectory_signatures.run_specificity_workloads import (  # noqa: E402
    BLACKLIGHT,
    BlacklightTracker,
    blacklight_salt,
)


def load_official():
    # The release uses the removed `imp` module; provide the one call it needs.
    shim = types.ModuleType("imp")

    def load_source(name, path):
        spec = importlib.util.spec_from_file_location(name, OFFICIAL / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    shim.load_source = load_source
    sys.modules["imp"] = shim
    sys.modules.setdefault("config", types.ModuleType("config"))  # imported but unused by utils.py
    spec = importlib.util.spec_from_file_location("probabilistic_fingerprint", OFFICIAL / "probabilistic_fingerprint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["probabilistic_fingerprint"] = module  # the worker pool pickles hash_helper by module name
    cwd = os.getcwd()
    os.chdir(OFFICIAL)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


def main():
    official = load_official()
    rng = np.random.default_rng(0)
    base = rng.random((32, 32, 3))
    stream = [np.clip(base + rng.normal(0, s, base.shape), 0, 1) for s in [0.0, 0.002, 0.004, 0.01, 0.03] * 6]
    stream += [rng.random((32, 32, 3)) for _ in range(5)]
    salt = blacklight_salt()
    reference = official.InputTracker(stream[0], BLACKLIGHT["window_size"], BLACKLIGHT["num_hashes_keep"],
                                      round=BLACKLIGHT["round"], step_size=BLACKLIGHT["step_size"], workers=2)
    # Default salt path: the release seeds 666 at import (an explicit array salt hits `salt != None`).
    assert np.array_equal(reference.salt, salt), "port salt differs from the official default"
    port = BlacklightTracker(salt)
    try:
        expected = [reference.add_img(x) for x in stream]
    finally:
        reference.delete()
    got = [port.add(x) for x in stream]
    print("official:", expected)
    print("port:    ", got)
    assert expected == got, "Blacklight port disagrees with the official tracker"
    print("PASS: identical match counts on", len(stream), "queries; salt matches the official default")


if __name__ == "__main__":
    main()
