#!/usr/bin/env python3
"""Label distribution across horizons and thresholds.

    python3 label_report.py

Run this BEFORE training. Choosing k and alpha by looking at what they do to
the class balance costs seconds; discovering a degenerate label set after an
hour of training costs an evening.

Two failure modes to watch for:

  overwhelmingly flat   the model learns to predict flat and scores well.
                        Accuracy becomes meaningless and macro F1 collapses.

  almost no flat        alpha is below the noise floor of the mid, so the
                        label is mostly recording tick jitter. The model can
                        appear to succeed while predicting something untradeable.

BTCUSDT sits at a one-tick spread, so the mid moves in steps of half a tick and
is unchanged in ~98.6% of consecutive frames. alpha = one median half-spread is
therefore exactly one mid tick - a much finer threshold than it looks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def valid_ends(intervals, window, k):
    out = []
    for a, b in intervals:
        lo, hi = a + window - 1, b - k - 1
        if hi >= lo:
            out.append(np.arange(lo, hi + 1, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--window", type=int, default=100)
    a = p.parse_args()
    ds = Path(a.dataset)

    mid = np.load(ds / "mid.npy")
    spread = np.load(ds / "spread.npy")
    intervals = json.loads((ds / "intervals.json").read_text())
    csum = np.concatenate([[0.0], np.cumsum(mid, dtype=np.float64)])

    tick_bp = 0.05 / np.median(mid) * 10_000
    half_bp = float(np.median((spread / 2) / mid * 10_000))
    print(f"frames {len(mid):,}   trusted runs {len(intervals)}")
    print(f"median half-spread {half_bp:.5f} bp"
          f"   one mid tick {tick_bp:.5f} bp")
    unchanged = (np.diff(mid) == 0).mean()
    print(f"mid unchanged in {unchanged:.1%} of consecutive frames\n")

    print(f"  {'k':>5}{'horizon':>10}{'windows':>12}"
          f"{'alpha_bp':>11}{'down':>9}{'flat':>9}{'up':>9}")
    for k in (25, 50, 100, 200, 400):
        ends = valid_ends(intervals, a.window, k)
        if len(ends) == 0:
            continue
        m_minus = (csum[ends + 1] - csum[ends + 1 - k]) / k
        m_plus = (csum[ends + 1 + k] - csum[ends + 1]) / k
        ell = (m_plus - m_minus) / m_minus * 10_000
        for mult in (0.5, 1.0, 2.0, 4.0):
            alpha = half_bp * mult
            up = float((ell > alpha).mean())
            dn = float((ell < -alpha).mean())
            fl = 1.0 - up - dn
            flag = ""
            if fl > 0.85:
                flag = "  <- almost all flat"
            elif fl < 0.10:
                flag = "  <- almost no flat"
            print(f"  {k:>5}{k / 10:>8.0f}s{len(ends):>12,}"
                  f"{alpha:>11.5f}{dn:>9.1%}{fl:>9.1%}{up:>9.1%}{flag}")
        print()

    print("  A balanced-ish split - flat somewhere around 20-50% - keeps all")
    print("  three classes learnable. The mult that gets you there is the one")
    print("  to pass as --alpha-mult, and it must be reported: it is a choice,")
    print("  and it belongs in the trial ledger like any other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())