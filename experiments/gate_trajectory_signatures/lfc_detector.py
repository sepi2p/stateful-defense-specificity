#!/usr/bin/env python3
"""Reimplementation of the two-phase stateful detector of Lee, Fang & Chang (arXiv 2606.21592).

No official code exists. Implemented from the paper; every underspecified step is
a named parameter here and listed in docs/lfc_reimplementation_notes.md.

Phase 1 (online, query geometry). Each query x in [0,1]^d is mapped to [0, N)
with N = 256, salted with s ~ U[0,1) * N, and quantized per entry as
floor(((x + s) mod N) / q) + B, B ~ Bernoulli(bern_scale * (((x + s) mod N) mod q) / q)
(paper: q = 80 for CIFAR-10; the paper's Bernoulli rate includes a 1/w factor,
so bern_scale = 1/w by default). r = 45 windows of w = 20 entries of the
flattened query are hashed; window start positions are drawn ONCE per detector
instance with pairwise overlap <= k = 15 (a per-query redraw makes the paper's
reported CIFAR-10 TPR unreachable, since two queries would share almost no
windows). Candidate subsequences each keep a set of (position, content) window
hashes (an exact set is an ideal Bloom filter). A query joins the subsequence
with the most matching windows if that count >= threshold (33); otherwise it
starts a new subsequence. Its windows are added to the chosen subsequence.

Phase 2 (offline from logged outputs). For each subsequence with >= min_len (15)
members, Ljung-Box on p_{c1} of its members in arrival order, where c1 is the
predicted class of the subsequence's first query; alarm if p < 0.025. The paper
does not state the lag count or whether the test is re-run on every update;
both are parameters of phase2_alarm().
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy.stats import chi2

LB_ALPHA = 0.025
LB_MIN_LEN = 15

def ljung_box_p(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < max(LB_MIN_LEN, lags + 2):
        return 1.0
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0.0:
        return 1.0
    r = np.array([(x[:-k] * x[k:]).sum() / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    return float(chi2.sf(q, lags))


LFC_CIFAR = {"N": 256, "q": 80, "r": 45, "w": 20, "k": 15, "threshold": 33, "min_len": 15, "alpha": 0.025}
LFC_IMAGENET = LFC_CIFAR | {"r": 75, "w": 50, "k": 20, "threshold": 7}  # paper's ImageNet setting


def draw_window_starts(d: int, r: int, w: int, k: int, rng: np.random.Generator, max_tries: int = 100000) -> np.ndarray:
    """r window start positions in [0, d - w] with pairwise overlap <= k."""
    starts: list[int] = []
    tries = 0
    while len(starts) < r:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place windows under the overlap constraint")
        s = int(rng.integers(0, d - w + 1))
        if all(max(0, w - abs(s - t)) <= k for t in starts):
            starts.append(s)
    return np.sort(np.asarray(starts))


class LFCPhase1:
    def __init__(self, d: int = 3072, seed: int = 0, params: dict | None = None, bern_scale: float | None = None,
                 salt_unit: bool = False, rounding: str = "fresh", rate_unsalted: bool = False):
        """Defaults reproduce the configuration frozen for the corpus runs when bern_scale = 0.

        Sensitivity options (Appendix B): salt_unit draws the salt from [0, 1) as the paper states
        (default: [0, N)); rounding = "fixed" draws one rounding threshold per position when the
        detector is created, so that identical inputs quantize identically ("fresh" redraws them for
        every query); rate_unsalted computes the rounding rate from x mod q as written in the paper
        (default: from the salted value).
        """
        self.p = dict(LFC_CIFAR if params is None else params)
        rng = np.random.default_rng(seed)
        self.salt = float(rng.random()) * (1.0 if salt_unit else self.p["N"])
        self.starts = draw_window_starts(d, self.p["r"], self.p["w"], self.p["k"], rng)
        self.bern_scale = 1.0 / self.p["w"] if bern_scale is None else bern_scale
        self.noise = np.random.default_rng(seed + 1)  # per-query Bernoulli rounding
        self.rounding, self.rate_unsalted = rounding, rate_unsalted
        self.fixed_u = np.random.default_rng(seed + 2).random(d)
        self.subsequences: list[set[bytes]] = []
        self.index: dict[bytes, set[int]] = {}  # window hash -> subsequences containing it
        self.assignment: list[int] = []  # subsequence id per query
        self.best_match: list[int] = []

    def window_hashes(self, x01: np.ndarray) -> list[bytes]:
        N, q, w = self.p["N"], self.p["q"], self.p["w"]
        x = np.asarray(x01, dtype=np.float64).reshape(-1) * (N - 1)
        v = np.mod(x + self.salt, N)
        rate = self.bern_scale * np.mod(x if self.rate_unsalted else v, q) / q
        u = self.fixed_u if self.rounding == "fixed" else self.noise.random(v.shape)
        level = np.floor(v / q) + (u < rate)
        level = level.astype(np.int16)
        return [hashlib.blake2b(np.int32(s).tobytes() + level[s : s + w].tobytes(), digest_size=16).digest() for s in self.starts]

    def add(self, x01: np.ndarray) -> int:
        hashes = self.window_hashes(x01)
        counts = Counter(i for h in set(hashes) for i in self.index.get(h, ()))
        # most matching windows; ties go to the earliest subsequence
        best, best_count = min(counts.items(), key=lambda kv: (-kv[1], kv[0])) if counts else (-1, 0)
        if best_count >= self.p["threshold"]:
            chosen = best
        else:
            self.subsequences.append(set())
            chosen = len(self.subsequences) - 1
        for h in hashes:
            if h not in self.subsequences[chosen]:
                self.subsequences[chosen].add(h)
                self.index.setdefault(h, set()).add(chosen)
        self.assignment.append(chosen)
        self.best_match.append(max(best_count, 0))
        return chosen

    def submit(self, raw) -> None:  # raw: torch tensor 1 x C x H x W in [0,1]
        self.add(raw[0].numpy())

    def summary(self):
        a = np.asarray(self.assignment)
        sizes = np.bincount(a) if len(a) else np.zeros(0, int)
        return {"n_subsequences": int(len(sizes)), "largest_subsequence": int(sizes.max()) if len(sizes) else 0}


def phase2_alarm(assignment: np.ndarray, p_matrix_logits: np.ndarray, min_len: int = 15, alpha: float = 0.025,
                 lags: str | int = "min10", retest: str = "every") -> int:
    """1-based query index of the first Phase-2 alarm, or -1.

    lags: an int, or "min10" = min(10, n // 5) (a common software default).
    retest: "every" = re-run the test whenever a subsequence of length >= min_len gains a member;
            "once" = test each subsequence once, when it first reaches min_len.
    """
    logits = np.asarray(p_matrix_logits, dtype=np.float64)
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    members: dict[int, list[int]] = {}
    for t, sid in enumerate(np.asarray(assignment)):
        members.setdefault(int(sid), []).append(t)
        idx = members[int(sid)]
        n = len(idx)
        if n < min_len or (retest == "once" and n != min_len):
            continue
        c1 = int(logits[idx[0]].argmax())
        h = min(10, n // 5) if lags == "min10" else int(lags)
        if h < 1:
            continue
        if ljung_box_p(probs[idx, c1], h) < alpha:
            return t + 1
    return -1


def phase2_batch(assignment: np.ndarray, logits: np.ndarray, length: int = 50, min_len: int = 15, alpha: float = 0.025,
                 lags: str | int = "min10") -> bool:
    """The paper's evaluation protocol: a sequence of `length` queries is given, Phase 1 groups it, and the
    Ljung-Box test is applied ONCE to every subsequence with at least min_len members."""
    assignment = np.asarray(assignment)[:length]
    logits = np.asarray(logits, dtype=np.float64)[:length]
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    for sid in np.unique(assignment):
        idx = np.flatnonzero(assignment == sid)
        n = len(idx)
        if n < min_len:
            continue
        h = min(10, n // 5) if lags == "min10" else int(lags)
        if h >= 1 and ljung_box_p(probs[idx, int(logits[idx[0]].argmax())], h) < alpha:
            return True
    return False

