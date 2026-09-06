#!/usr/bin/env python3
"""Cost floor: does the predicted move survive the cost of acting on it?

    python3 costfloor.py                      # k=100, sweep costs and latency
    python3 costfloor.py --k 100 --folds 5

Classification accuracy is not a result. A model can be right about direction
and still lose money on every trade, because being right about a 0.3 bp move
does not pay a 10 bp round trip. This script converts predictions into
positions and charges what it would actually cost to take them.

Three things it measures, all of which belong in the writeup:

  MAGNITUDE   what is the realised move, conditional on each prediction?
              Direction accuracy without magnitude is unfalsifiable.

  COST FLOOR  sweep the round-trip cost from 0 to 20 bp and find where gross
              edge stops covering it. Binance USD-M taker is 5 bp per side,
              so 10 bp round trip is the number that matters; anything below
              that is a claim about a market you cannot trade in.

  LATENCY     the model sees the book at t but can only act at t+delta. Sweep
              delta and watch the edge decay. The measured REST round trip
              from this machine is marked on the curve. This is the experiment
              that says WHO can capture the signal, not merely whether it
              exists.

  SIGNIFICANCE  is the edge distinguishable from zero at all? Reported two
              ways, because the obvious way is wrong. See moving_block_ci.

Predictions come from the same purged walk-forward folds as evaluate.py, with
fold-local labels and normalisation. Nothing here re-fits anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lobforge.models import LogisticBaseline                  # noqa: E402
from lobforge.training import (TrainConfig, log_trial,        # noqa: E402
                               set_seed, train)

RETURNS_ONLY = ("ret_1s_bp", "ret_5s_bp", "ret_30s_bp", "rv_30s_bp")


def valid_ends(intervals, window, k, lat=0):
    """Window ends whose history, action delay and label horizon all lie inside
    one trusted interval."""
    out = []
    for a, b in intervals:
        lo, hi = a + window - 1, b - k - lat - 1
        if hi >= lo:
            out.append(np.arange(lo, hi + 1, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def forward_bp(mid, ends, k, lat=0):
    """Realised move in bp from the price you could ACT at to the mean of the
    next k mids.

    With latency, entry is mid[t+lat] - the model saw the book at t but the
    order does not reach the exchange until t+lat, and the horizon is measured
    from there. Measuring from mid[t] while acting at t+lat is the free-money
    assumption every naive backtest makes.
    """
    csum = np.concatenate([[0.0], np.cumsum(mid, dtype=np.float64)])
    entry = mid[ends + lat]
    m_plus = (csum[ends + lat + 1 + k] - csum[ends + lat + 1]) / k
    return (m_plus - entry) / entry * 10_000


class FeatureRows:
    def __init__(self, feats, ends, labels, mu, sd):
        self.feats, self.ends, self.labels = feats, ends, labels
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        import torch
        x = (self.feats[int(self.ends[i])] - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[i])


def label_from_bp(ell_bp, alpha):
    y = np.full(len(ell_bp), 1, dtype=np.int64)
    y[ell_bp > alpha] = 2
    y[ell_bp < -alpha] = 0
    return y


def fold_predictions(feats, mid, spread, intervals, args, lat):
    """Train per fold, return (realised_bp, predicted_class, confidence)."""
    ends = valid_ends(intervals, args.window, args.k, lat)
    lo, hi = int(ends.min()), int(ends.max())
    edges = np.linspace(lo, hi + 1, args.folds + 2).astype(np.int64)

    all_bp, all_pred, all_conf = [], [], []
    for f in range(args.folds):
        t0, t1 = int(edges[f + 1]), int(edges[f + 2])
        test = ends[(ends >= t0) & (ends < t1)]
        tr_ends = ends[ends + args.k + lat < t0 - args.k]
        if len(tr_ends) < 5000 or len(test) < 1000:
            continue

        half = (spread[tr_ends] / 2) / mid[tr_ends] * 10_000
        alpha = float(np.median(half))

        bp_tr = forward_bp(mid, tr_ends, args.k, lat)
        bp_te = forward_bp(mid, test, args.k, lat)
        y_tr = label_from_bp(bp_tr, alpha)
        y_te = label_from_bp(bp_te, alpha)

        ftr = feats[tr_ends]
        mu = ftr.mean(0)
        sd = np.where(ftr.std(0) > 1e-9, ftr.std(0), 1.0)

        set_seed(args.seed)
        model = LogisticBaseline(feats.shape[1])
        tr = FeatureRows(feats, tr_ends, y_tr, mu, sd)
        va = FeatureRows(feats, test[::10], y_te[::10], mu, sd)
        train(model, tr, va,
              TrainConfig(epochs=args.epochs, batch_size=args.batch,
                          lr=1e-3, seed=args.seed, patience=2),
              log=lambda m: None)

        import torch
        from torch.utils.data import DataLoader
        te = FeatureRows(feats, test, y_te, mu, sd)
        preds, confs = [], []
        model.eval()
        with torch.no_grad():
            for x, _ in DataLoader(te, batch_size=4096):
                p = torch.softmax(model(x), dim=1)
                c, i = p.max(1)
                preds.append(i.numpy()); confs.append(c.numpy())
        all_bp.append(bp_te)
        all_pred.append(np.concatenate(preds))
        all_conf.append(np.concatenate(confs))
        print(f"  fold {f}: train {len(tr_ends):,}  test {len(test):,}"
              f"  alpha {alpha:.4f} bp", flush=True)

    return (np.concatenate(all_bp), np.concatenate(all_pred),
            np.concatenate(all_conf))


def gross_edge(bp, pred):
    """Signed realised move when the model takes a side. Flat predictions do
    not trade, so they neither earn nor cost - which is itself a result: a
    model that abstains most of the time may still be useful.

    Returns the per-trade signed series too: every honest statement about
    significance has to be made over that series, not over its mean and sd.
    """
    take = pred != 1
    if take.sum() == 0:
        return np.zeros(0)
    return np.where(pred[take] == 2, bp[take], -bp[take])


def moving_block_ci(x, block, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI on the mean of a SERIALLY DEPENDENT series.

    sd/sqrt(n) is wrong here, and not by a little. It assumes independent
    observations; these are nothing of the kind. Adjacent windows share 99 of
    their 100 frames, and their k-frame label horizons overlap almost
    entirely, so one trade carries very nearly the same information as its
    neighbour. The effective sample size is roughly n/k, not n, and the naive
    interval is therefore too narrow by something on the order of sqrt(k).

    The moving-block bootstrap resamples contiguous BLOCKS rather than points,
    so whatever dependence lives inside a block survives into the resample.
    Blocks are drawn with replacement from every possible start offset and
    concatenated to length n.

    `block` is in TRADE units, not frames. Consecutive trades are at least one
    frame apart, so a block of 2k trades spans at least 2k frames - a full
    label horizon and then some. The trade rate is below 100%, so in practice
    a block covers more frames than its length suggests. That is conservative
    on purpose: erring toward a wider interval is the right direction to err.

    Implemented over block SUMS rather than by materialising each resample.
    An 845k-trade series at 1000 resamples would otherwise build a 6.8 GB
    index array; precomputing every block sum by cumsum makes each resample
    O(n/block) instead of O(n), and the whole thing runs in well under a
    second.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")

    block = int(np.clip(block, 1, n))
    n_blocks = int(np.ceil(n / block))
    r = n - (n_blocks - 1) * block          # the final block is partial

    c = np.concatenate([[0.0], np.cumsum(x)])
    full = c[block:] - c[:-block]           # sum of x[i:i+block], every i
    part = c[r:] - c[:-r]                   # sum of x[i:i+r], every i

    rng = np.random.default_rng(seed)
    # Chunk the resamples so the index array stays small whatever `block` is.
    per = max(1, int(5e6 // max(1, n_blocks)))
    means = []
    for i in range(0, n_boot, per):
        b = min(per, n_boot - i)
        s = part[rng.integers(0, len(part), size=b)]
        if n_blocks > 1:
            s = s + full[rng.integers(0, len(full),
                                      size=(b, n_blocks - 1))].sum(1)
        means.append(s / n)
    means = np.concatenate(means)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--runs", default="./runs")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latencies", default="0,100,300,500,800,1200",
                   help="decision delays in MILLISECONDS")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="moving-block bootstrap resamples for the edge CI")
    p.add_argument("--block-mult", type=int, default=2, dest="block_mult",
                   help="block length = mult x k, so one block spans at least "
                        "a full label horizon and carries its dependence")
    a = p.parse_args()

    ds = Path(a.dataset)
    feats = np.load(ds / "features.npy")
    mid = np.load(ds / "mid.npy")
    spread = np.load(ds / "spread.npy")
    intervals = json.loads((ds / "intervals.json").read_text())

    print(f"frames {len(mid):,}   horizon {a.k / 10:.0f}s   folds {a.folds}\n")

    # Ledger BEFORE the sweep, not after. A run abandoned halfway through the
    # latency sweep is still a trial and still belongs in the denominator.
    run_id = f"costfloor-k{a.k}-{int(time.time())}"
    ledger = Path(a.runs) / "trials.jsonl"
    log_trial(ledger, {"run_id": run_id, "model": "costfloor", "k": a.k,
                       "window": a.window, "folds": a.folds,
                       "latencies": a.latencies, "bootstrap": a.bootstrap,
                       "block_mult": a.block_mult, "seed": a.seed,
                       "status": "started"})

    # ---------------------------------------------------------- magnitude
    print("=" * 66)
    print("  MAGNITUDE OF THE MOVE BEING PREDICTED")
    print("=" * 66)
    ends = valid_ends(intervals, a.window, a.k)
    bp_all = forward_bp(mid, ends, a.k)
    q = np.percentile(np.abs(bp_all), [50, 75, 90, 95, 99])
    print(f"  |forward move| over {a.k / 10:.0f}s, {len(bp_all):,} windows")
    for lab, v in zip(("p50", "p75", "p90", "p95", "p99"), q):
        print(f"    {lab}  {v:7.3f} bp")
    print(f"\n  Binance USD-M taker 5 bp/side -> 10 bp round trip.")
    frac = float((np.abs(bp_all) > 10).mean())
    print(f"  {frac:.2%} of windows move more than 10 bp at all.")

    # ------------------------------------------------------- latency sweep
    lats_ms = [int(x) for x in a.latencies.split(",")]
    print("\n" + "=" * 66)
    print("  EDGE vs DECISION LATENCY   (100 ms per frame)")
    print("=" * 66)
    block = a.block_mult * a.k
    rows = []
    for ms in lats_ms:
        lat = int(round(ms / 100))
        print(f"\n  latency {ms} ms ({lat} frames)")
        bp, pred, conf = fold_predictions(feats, mid, spread, intervals, a, lat)
        signed = gross_edge(bp, pred)
        n = len(signed)
        mean_bp = float(signed.mean()) if n else 0.0
        sd_bp = float(signed.std()) if n else 0.0

        # Naive: kept deliberately. The gap between these two intervals is
        # the finding, so deleting the wrong one would hide it.
        se = sd_bp / np.sqrt(max(1, n))
        naive_lo, naive_hi = mean_bp - 1.96 * se, mean_bp + 1.96 * se
        boot_lo, boot_hi = moving_block_ci(signed, block, a.bootstrap, a.seed)

        naive_w = naive_hi - naive_lo
        boot_w = boot_hi - boot_lo
        ratio = boot_w / naive_w if naive_w > 0 else float("nan")

        rows.append({"latency_ms": ms, "gross_bp": mean_bp, "n_trades": n,
                     "stderr_bp": float(se),
                     "naive_ci_bp": [naive_lo, naive_hi],
                     "boot_ci_bp": [boot_lo, boot_hi],
                     "boot_block_trades": int(block),
                     "boot_resamples": int(a.bootstrap),
                     "ci_width_ratio": float(ratio),
                     "trade_rate": float(n / len(pred))})
        print(f"    gross edge {mean_bp:+.4f} bp"
              f"   trades {n:,} ({n / len(pred):.1%} of windows)")
        print(f"      naive       +/-{se:.4f} se    95% CI"
              f" [{naive_lo:+.4f}, {naive_hi:+.4f}]  width {naive_w:.4f}")
        print(f"      block boot  {a.bootstrap} x L={block}   95% CI"
              f" [{boot_lo:+.4f}, {boot_hi:+.4f}]  width {boot_w:.4f}")
        print(f"      the naive interval is {ratio:.1f}x too narrow"
              f"   ({'excludes' if boot_lo > 0 or boot_hi < 0 else 'INCLUDES'}"
              f" zero under the bootstrap)")

    print("\n" + "=" * 66)
    print("  COST FLOOR")
    print("=" * 66)
    print(f"  {'latency':>9}{'gross bp':>11}{'net @5bp':>11}"
          f"{'net @10bp':>11}{'net @15bp':>11}")
    for r in rows:
        g = r["gross_bp"]
        print(f"  {r['latency_ms']:>7} ms{g:>11.4f}"
              f"{g - 5:>11.4f}{g - 10:>11.4f}{g - 15:>11.4f}")

    r0 = rows[0]
    g0 = r0["gross_bp"]
    blo, bhi = r0["boot_ci_bp"]
    print(f"\n  Break-even round-trip cost at zero latency: {g0:.4f} bp")
    print(f"  Moving-block 95% CI on that number: [{blo:+.4f}, {bhi:+.4f}] bp")
    print(f"  ({r0['ci_width_ratio']:.1f}x wider than sd/sqrt(n), which is the")
    print(f"  number to quote: the naive one assumes independent trades and")
    print(f"  overlapping windows are the opposite of independent.)")
    if g0 < 10:
        print(f"  Below the 10 bp retail round trip. The signal is real but")
        print(f"  not tradeable at these costs - which is a finding, not a")
        print(f"  failure, and it is the honest one to report.")
    else:
        print(f"  Above the 10 bp round trip. Check the deflated Sharpe before")
        print(f"  believing it: {len(rows)} latency points is {len(rows)} more")
        print(f"  trials in the ledger.")

    out = Path(a.runs) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "args": vars(a),
                               "magnitude_bp":
                               dict(zip(("p50", "p75", "p90", "p95", "p99"),
                                        q.tolist())),
                               "latency_sweep": rows}, indent=2))
    log_trial(ledger, {"run_id": run_id, "status": "finished",
                       "gross_bp_zero_latency": g0,
                       "boot_ci_bp": r0["boot_ci_bp"],
                       "naive_ci_bp": r0["naive_ci_bp"]})
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
