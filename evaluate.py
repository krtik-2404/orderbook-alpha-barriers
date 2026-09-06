#!/usr/bin/env python3
"""Purged walk-forward evaluation.

    python3 evaluate.py --model baseline --k 100
    python3 evaluate.py --model deeplob --k 100 --folds 5 --epochs 15

This tier owns the three things that decide whether a result means anything,
and it owns them because each is a leak that throws no error:

**Labels are built per fold.** The threshold alpha is the median half-spread of
that fold's TRAINING portion. Computing it over the whole dataset uses future
information to label the past.

**Normalization is fitted per fold.** Volume statistics come from training
frames only. Global statistics are the most common silent leak in LOB papers
and they inflate results without ever failing.

**Splits are chronological and purged.** Adjacent windows share 99 of their 100
frames, so a random split puts near-duplicates on both sides of the boundary
and "accuracy" becomes memorisation. Labels look k frames forward, so the k
frames before each test block are dropped from training - otherwise they are
labelled using data that lives in the test set.

Every run appends to a trial ledger BEFORE it starts. Abandoned runs must be in
there too, or the deflated-Sharpe denominator is understated and the headline
number overstates itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lobforge.models import DeepLOB, LogisticBaseline          # noqa: E402
from lobforge.training import (CLASSES, TrainConfig, evaluate,  # noqa: E402
                               log_trial, predict, set_seed, train)

CLASS_DOWN, CLASS_FLAT, CLASS_UP = 0, 1, 2


# ------------------------------------------------------------------ dataset

class Fold:
    def __init__(self, index, train_ends, test_ends, alpha_bp, vol_mu, vol_sd,
                 test_range):
        self.index = index
        self.train_ends = train_ends
        self.test_ends = test_ends
        self.alpha_bp = alpha_bp
        self.vol_mu = vol_mu
        self.vol_sd = vol_sd
        self.test_range = test_range


def valid_ends(intervals, window: int, k: int) -> np.ndarray:
    """Window end indices whose history AND label horizon lie inside one
    trusted interval. A window spanning a gap contains a book state that never
    existed; a label reaching across one is computed from prices that were
    never observed."""
    out = []
    for a, b in intervals:
        lo, hi = a + window - 1, b - k - 1
        if hi >= lo:
            out.append(np.arange(lo, hi + 1, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def make_labels(mid: np.ndarray, ends: np.ndarray, k: int, alpha_bp: float,
                mode: str = "forward") -> np.ndarray:
    """Direction of the smoothed future mid.

    Both variants average the next k mids, because a single tick is mostly
    bid-ask bounce and a model trained on that learns the bounce. They differ
    in what they compare it AGAINST, and the difference is not cosmetic:

      smoothed  l = (m+ - m-) / m-      the published DeepLOB/Ntakaris label
      forward   l = (m+ - mid_t) / mid_t

    m- is the mean of the PAST k mids, which is known at prediction time. For a
    random walk E[m+] ~ mid_t, so the smoothed label reduces in expectation to
    (mid_t - m-)/m- - the recent return. It therefore encodes past momentum by
    construction, and a model that sees only returns recovers much of it with
    no forecasting skill at all. Measured on this archive: a price-history-only
    null model scores macro F1 0.61 under `smoothed`.

    `forward` compares the future against the price you could actually trade at
    now, which is the quantity a cost analysis can speak to. It is the default
    for that reason; `smoothed` is kept for comparison with the literature.
    """
    csum = np.concatenate([[0.0], np.cumsum(mid, dtype=np.float64)])
    m_plus = (csum[ends + 1 + k] - csum[ends + 1]) / k
    if mode == "smoothed":
        ref = (csum[ends + 1] - csum[ends + 1 - k]) / k
    elif mode == "forward":
        ref = mid[ends]
    else:
        raise ValueError(f"unknown label mode: {mode}")
    ell_bp = (m_plus - ref) / ref * 10_000
    y = np.full(len(ends), CLASS_FLAT, dtype=np.int64)
    y[ell_bp > alpha_bp] = CLASS_UP
    y[ell_bp < -alpha_bp] = CLASS_DOWN
    return y


def make_folds(ends, mid, spread, n_folds, k, purge, embargo, alpha_mult):
    """Expanding-window walk-forward over the frame timeline."""
    lo, hi = int(ends.min()), int(ends.max())
    edges = np.linspace(lo, hi + 1, n_folds + 2).astype(np.int64)
    folds = []
    for f in range(n_folds):
        t0, t1 = int(edges[f + 1]), int(edges[f + 2])
        test = ends[(ends >= t0) & (ends < t1)]
        # Purge: a training window whose label horizon reaches into the test
        # block is labelled with test data. Drop it.
        train = ends[ends + k < t0 - purge]
        # Embargo guards the mirror case - training data drawn from AFTER a
        # test block. An expanding forward walk never produces any, so this is
        # a tripwire for a future change to the split scheme, not a no-op that
        # got left in.
        train = train[~((train > t1) & (train <= t1 + embargo))]
        if len(train) < 5000 or len(test) < 1000:
            continue

        # alpha anchored to the cost floor of the TRAINING fold: below the
        # half-spread you are predicting moves you could not have traded.
        half_bp = (spread[train] / 2) / mid[train] * 10_000
        alpha = float(np.median(half_bp)) * alpha_mult

        folds.append(Fold(len(folds), train, test, alpha, 0.0, 1.0, (t0, t1)))
    return folds


class FeatureRows:
    """Engineered features at the window end, z-scored with TRAINING stats.

    The baseline sees one vector, not a window: these features already
    aggregate history (rolling OFI, returns, realised volatility), so handing
    it a sequence would just duplicate that. Whether a deep model beats this
    is the actual question the project asks.
    """

    def __init__(self, feats, ends, labels, mu, sd):
        self.feats, self.ends, self.labels = feats, ends, labels
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        import torch
        x = (self.feats[int(self.ends[i])] - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[i])


class Windows:
    """Slices windows from the flat (T, 40) array and normalises on the fly.

    Materialising windows is not an option: 13M frames x 100 x 40 x 4 bytes is
    ~208 TB against ~2 GB for the flat array.

    Prices are expressed in basis points relative to the window's LAST mid, so
    the representation is translation-invariant - otherwise the network learns
    the price level, and BTC moved 1,200 dollars across this archive.
    """

    def __init__(self, frames, mid, ends, labels, window, vol_mu, vol_sd):
        self.frames, self.mid = frames, mid
        self.ends, self.labels = ends, labels
        self.window = window
        self.vol_mu, self.vol_sd = vol_mu, vol_sd
        self.price_cols = np.arange(0, frames.shape[1], 2)
        self.vol_cols = np.arange(1, frames.shape[1], 2)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        import torch
        e = int(self.ends[i])
        x = self.frames[e - self.window + 1: e + 1].astype(np.float32).copy()
        ref = self.mid[e]
        x[:, self.price_cols] = (x[:, self.price_cols] - ref) / ref * 10_000
        v = np.log1p(x[:, self.vol_cols])
        x[:, self.vol_cols] = (v - self.vol_mu) / self.vol_sd
        return torch.from_numpy(x), int(self.labels[i])


def volume_stats(frames, ends, window, sample=20000):
    """Fitted on training ends only. Sampled because the statistic is stable
    long before the data runs out."""
    idx = ends if len(ends) <= sample else np.random.default_rng(0).choice(
        ends, sample, replace=False)
    cols = np.arange(1, frames.shape[1], 2)
    vals = np.concatenate([np.log1p(frames[e - window + 1: e + 1][:, cols]).ravel()
                           for e in idx[:2000]])
    return float(vals.mean()), float(vals.std() or 1.0)


# --------------------------------------------------------------------- main

def run(args) -> int:
    ds = Path(args.dataset)
    frames = np.load(ds / "frames.npy", mmap_mode="r")
    mid = np.load(ds / "mid.npy")
    spread = np.load(ds / "spread.npy")
    intervals = json.loads((ds / "intervals.json").read_text())

    # The label is (m+ - m-)/m-, and m- is the mean of the PAST k mids - known
    # at prediction time. Past returns therefore predict the label partially
    # through mean reversion, with no forecasting whatever. On pure noise this
    # scores macro F1 ~0.46 against ~0.10 for a book-only model. So the
    # majority class is NOT the reference: this returns-only null model is.
    # The question the project actually asks is whether book state beats it.
    RETURNS_ONLY = ("ret_1s_bp", "ret_5s_bp", "ret_30s_bp", "rv_30s_bp")

    feats = None
    if args.model in ("baseline", "returns"):
        fp = ds / "features.npy"
        if not fp.exists():
            print("no features.npy - run:  python3 features.py")
            return 1
        feats = np.load(fp)
        names = json.loads((ds / "feature_names.json").read_text())
        if args.model == "returns":
            keep = [i for i, n in enumerate(names) if n in RETURNS_ONLY]
            feats = feats[:, keep]
            names = [names[i] for i in keep]
            print(f"NULL MODEL - price history only, no book information: "
                  + ", ".join(names))
        else:
            print(f"baseline on {len(names)} engineered features: "
                  + ", ".join(names))

    ends = valid_ends(intervals, args.window, args.k)
    print(f"frames {frames.shape[0]:,} x {frames.shape[1]}   "
          f"trusted runs {len(intervals)}   valid windows {len(ends):,}")

    folds = make_folds(ends, mid, spread, args.folds, args.k,
                       args.purge or args.k, args.embargo or args.k,
                       args.alpha_mult)
    if not folds:
        print("no usable folds - try fewer --folds or a smaller --k")
        return 1

    run_id = f"{args.model}-{args.label}-k{args.k}-{int(time.time())}"
    ledger = Path(args.runs) / "trials.jsonl"
    log_trial(ledger, {"run_id": run_id, "model": args.model, "k": args.k,
                       "window": args.window, "folds": len(folds),
                       "alpha_mult": args.alpha_mult, "label": args.label,
                       "epochs": args.epochs,
                       "stride": args.stride, "val_stride": args.val_stride,
                       "seed": args.seed, "status": "started"})

    results = []
    for fd in folds:
        if args.stride > 1:
            fd.train_ends = fd.train_ends[::args.stride]
        if args.model == "deeplob" and len(fd.train_ends) < 50_000:
            print(f"\n  WARNING fold {fd.index}: {len(fd.train_ends):,} training "
                  f"windows for a 143k-parameter network. Any comparison "
                  f"against the baseline, which trains on the full set, is "
                  f"not like-for-like. Lower --stride.")
        # Cheap slice for per-epoch early stopping; every reported metric
        # below is computed on the FULL test set regardless.
        val_ends = fd.test_ends[::max(1, args.val_stride)]
        y_val = make_labels(mid, val_ends, args.k, fd.alpha_bp, args.label)
        y_tr = make_labels(mid, fd.train_ends, args.k, fd.alpha_bp, args.label)
        y_te = make_labels(mid, fd.test_ends, args.k, fd.alpha_bp, args.label)
        mu, sd = volume_stats(frames, fd.train_ends, args.window)

        dist = np.bincount(y_tr, minlength=3) / len(y_tr)
        print(f"\nfold {fd.index}  test frames {fd.test_range[0]:,}"
              f"..{fd.test_range[1]:,}")
        print(f"  train {len(fd.train_ends):,}  val {len(val_ends):,}"
              f"  test {len(fd.test_ends):,}  alpha {fd.alpha_bp:.4f} bp")
        print("  train label mix  " + "  ".join(
            f"{c} {p:.1%}" for c, p in zip(CLASSES, dist)))

        if args.model in ("baseline", "returns"):
            # Fold-local standardisation: statistics from TRAINING rows only.
            ftr = feats[fd.train_ends]
            fmu = ftr.mean(0)
            fsd = np.where(ftr.std(0) > 1e-9, ftr.std(0), 1.0)
            tr = FeatureRows(feats, fd.train_ends, y_tr, fmu, fsd)
            va = FeatureRows(feats, val_ends, y_val, fmu, fsd)
            te = FeatureRows(feats, fd.test_ends, y_te, fmu, fsd)
            model = LogisticBaseline(feats.shape[1])
        else:
            tr = Windows(frames, mid, fd.train_ends, y_tr, args.window, mu, sd)
            va = Windows(frames, mid, val_ends, y_val, args.window, mu, sd)
            te = Windows(frames, mid, fd.test_ends, y_te, args.window, mu, sd)
            model = DeepLOB(levels=frames.shape[1] // 4)

        set_seed(args.seed)
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch,
                          lr=args.lr, seed=args.seed, patience=args.patience)
        train(model, tr, va, cfg, log=lambda m: print("   ", m))

        from torch.utils.data import DataLoader
        yt, yp = predict(model, DataLoader(te, batch_size=args.batch),
                         cfg.resolved_device())
        m = evaluate(yt, yp)
        print(m.summary())
        results.append({"fold": fd.index, "alpha_bp": fd.alpha_bp,
                        "n_train": len(fd.train_ends),
                        **m.as_dict()})

    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    base = [r["majority_accuracy"] for r in results]
    print("\n" + "=" * 62)
    print(f"  {args.model}  label={args.label}  k={args.k}"
          f" ({args.k * 100} ms horizon)"
          f"  {len(results)} folds")
    print(f"  accuracy   {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  majority   {np.mean(base):.4f}   "
          f"lift {np.mean(accs) - np.mean(base):+.4f}")
    print(f"  macro F1   {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("=" * 62)
    print("  Accuracy alone is not a result on a flat-dominated label set.")
    print("  Macro F1 and the lift over the majority baseline are.")

    out = Path(args.runs) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "args": vars(args),
                               "folds": results,
                               "mean_accuracy": float(np.mean(accs)),
                               "mean_macro_f1": float(np.mean(f1s))}, indent=2))
    log_trial(ledger, {"run_id": run_id, "status": "finished",
                       "mean_macro_f1": float(np.mean(f1s)),
                       "mean_accuracy": float(np.mean(accs))})
    print(f"\n  -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--runs", default="./runs")
    p.add_argument("--model", choices=("returns", "baseline", "deeplob"),
                   default="baseline",
                   help="returns = price history only (the null model); "
                        "baseline = engineered book features; "
                        "deeplob = raw book windows")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--k", type=int, default=100, help="label horizon in frames (x100ms)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--purge", type=int, default=0, help="default: k")
    p.add_argument("--embargo", type=int, default=0, help="default: k")
    p.add_argument("--label", choices=("forward", "smoothed"), default="forward",
                   help="forward: future mean vs the CURRENT mid (default). "
                        "smoothed: the published DeepLOB label, which encodes "
                        "past momentum and inflates any model seeing returns.")
    p.add_argument("--alpha-mult", type=float, default=1.0,
                   dest="alpha_mult", help="alpha = mult x median half-spread")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-stride", type=int, default=10, dest="val_stride",
                   help="stride the validation set used DURING training only. "
                        "Per-epoch validation over the full test set dominates "
                        "runtime - 164k val windows is 20x the work of 8k "
                        "train windows - and buys nothing, since it is only "
                        "used for early stopping. The FINAL reported metrics "
                        "always use the complete test set.")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth TRAINING window. Adjacent windows "
                        "share 99 of 100 frames, so stride 10 costs almost no "
                        "information and cuts training time tenfold. Test "
                        "windows are never strided - the evaluation stays full.")
    p.add_argument("--seed", type=int, default=0)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())