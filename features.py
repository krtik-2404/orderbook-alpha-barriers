#!/usr/bin/env python3
"""Engineered microstructure features - the honest baseline's input.

    python3 features.py                       # ./dataset -> dataset/features.npy

Every feature is CAUSAL: computed from frames at or before t, never after.
Rolling windows are cut at trusted-interval boundaries, so no feature is ever
computed across a gap.

Why this exists: the reference model must be a real one. Logistic regression on
raw book levels is a straw man, and beating a straw man says nothing. If these
fourteen features match DeepLOB, the finding is that the predictive content is
in known microstructure quantities and the deep model adds nothing - which is
worth reporting, and is a result a lot of papers quietly avoid producing.

What is NOT here: trade-flow features. Signed volume, trade imbalance and trade
count all need the aggTrade stream, which the dataset builder does not yet
join to the book. Named rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAMES = [
    "ofi1_1s", "ofi1_5s", "ofi5_1s",
    "imb1", "imb5", "imb10",
    "spread_bp", "micro_bp",
    "ret_1s_bp", "ret_5s_bp", "ret_30s_bp",
    "rv_30s_bp", "slope_bid", "slope_ask",
]


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling sum: out[i] = sum(x[i-w+1 .. i]), zero-padded at the
    start of the array (which is a trusted-interval start, never a splice)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = np.zeros(len(x))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w + 1)
    out = c[idx + 1] - c[lo]
    return out


def ofi(ask_p, ask_v, bid_p, bid_v) -> np.ndarray:
    """Order flow imbalance, Cont-Kukanov-Stoikov.

    Counts net pressure from CHANGES at the touch, not from levels. A quote
    that improves adds its full size; a quote that retreats removes the size
    that was there. This is why OFI predicts better than volume imbalance:
    volume imbalance describes the book, OFI describes what just happened.
    """
    e = np.zeros(len(bid_p))
    bp, bpp = bid_p[1:], bid_p[:-1]
    bv, bvp = bid_v[1:], bid_v[:-1]
    ap, app = ask_p[1:], ask_p[:-1]
    av, avp = ask_v[1:], ask_v[:-1]
    e[1:] = (np.where(bp >= bpp, bv, 0.0) - np.where(bp <= bpp, bvp, 0.0)
             - np.where(ap <= app, av, 0.0) + np.where(ap >= app, avp, 0.0))
    return e


def compute_interval(F: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Features for one contiguous trusted run."""
    n = len(F)
    ap = F[:, 0::4]          # (n, 10) ask prices
    av = F[:, 1::4]
    bp = F[:, 2::4]
    bv = F[:, 3::4]

    a1, b1 = ap[:, 0], bp[:, 0]
    av1, bv1 = av[:, 0], bv[:, 0]
    spread = a1 - b1
    m = mid

    e1 = ofi(a1, av1, b1, bv1)
    e5 = sum(ofi(ap[:, i], av[:, i], bp[:, i], bv[:, i]) for i in range(5))

    depth_b1, depth_a1 = bv1, av1
    depth_b5, depth_a5 = bv[:, :5].sum(1), av[:, :5].sum(1)
    depth_b10, depth_a10 = bv.sum(1), av.sum(1)

    def imb(b, a):
        d = b + a
        return np.where(d > 0, (b - a) / np.maximum(d, 1e-9), 0.0)

    micro = (a1 * bv1 + b1 * av1) / np.maximum(av1 + bv1, 1e-9)

    logm = np.log(np.maximum(m, 1e-9))
    def ret(w):
        out = np.zeros(n)
        if n > w:
            out[w:] = (logm[w:] - logm[:-w]) * 10_000
        return out

    r1 = np.zeros(n)
    r1[1:] = (logm[1:] - logm[:-1]) * 10_000
    rv = np.sqrt(np.maximum(_roll_sum(r1 ** 2, 300), 0.0))

    # Depth-weighted slope: how fast size accumulates away from the touch.
    lv = np.arange(1, 11)
    slope_b = (bv * lv).sum(1) / np.maximum(bv.sum(1), 1e-9)
    slope_a = (av * lv).sum(1) / np.maximum(av.sum(1), 1e-9)

    cols = [
        _roll_sum(e1, 10), _roll_sum(e1, 50), _roll_sum(e5, 10),
        imb(depth_b1, depth_a1), imb(depth_b5, depth_a5),
        imb(depth_b10, depth_a10),
        spread / m * 10_000,
        (micro - m) / m * 10_000,
        ret(10), ret(50), ret(300),
        rv, slope_b, slope_a,
    ]
    return np.column_stack(cols).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    a = p.parse_args()
    ds = Path(a.dataset)

    frames = np.load(ds / "frames.npy", mmap_mode="r")
    mid = np.load(ds / "mid.npy")
    intervals = json.loads((ds / "intervals.json").read_text())

    out = np.zeros((frames.shape[0], len(NAMES)), dtype=np.float32)
    for i, (lo, hi) in enumerate(intervals, 1):
        # Per interval, never across one: a rolling sum spanning a gap mixes
        # book states from either side of a discontinuity.
        out[lo:hi] = compute_interval(np.asarray(frames[lo:hi]), mid[lo:hi])
        print(f"\r  interval {i}/{len(intervals)}  {hi - lo:,} frames",
              end="", flush=True)
    print()

    np.save(ds / "features.npy", out)
    (ds / "feature_names.json").write_text(json.dumps(NAMES))

    finite = np.isfinite(out).all(axis=1)
    print(f"\n  features   {out.shape[0]:,} x {out.shape[1]}")
    print(f"  non-finite {(~finite).sum():,} rows")
    print(f"\n  {'name':<12}{'p05':>12}{'p50':>12}{'p95':>12}")
    for j, nm in enumerate(NAMES):
        c = out[finite, j]
        print(f"  {nm:<12}{np.percentile(c, 5):>12.4f}"
              f"{np.percentile(c, 50):>12.4f}{np.percentile(c, 95):>12.4f}")
    print(f"\n  -> {ds}/features.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())