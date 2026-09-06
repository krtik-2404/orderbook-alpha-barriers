#!/usr/bin/env python3
"""What does a price-history-only null model score on FI-2010?

    python3 fi2010.py --download            # fetch, then run
    python3 fi2010.py --data ./fi2010-data  # use an existing extraction

FI-2010 is the public benchmark DeepLOB was published on. Its labels are built
by the smoothed scheme

    l = (m+ - m-) / m-

where m+ is the mean of the NEXT k mid-prices and m- is the mean of the PAST k.
m- is known at prediction time. For a random walk E[m+] ~ mid_t, so in
expectation the label reduces to (mid_t - m-)/m- - a past return. The label
therefore encodes recent momentum by construction, and a model that sees only
past returns can score on it without forecasting anything.

Measured on this project's own Binance archive, a returns-only model scores
macro F1 0.61 under that label and 0.39 under a forward label. On synthetic
pure noise it scores 0.46 under the smoothed label and 0.26 under a forward
one. Both say the same thing: the label leaks.

What nobody appears to have published is the null model's score ON THE
BENCHMARK ITSELF. That is what this script measures. The features are built
from level-1 prices only - mid, returns at several lags, realised volatility.
No depth, no imbalance, no book shape. If that scores well, the benchmark's
headline numbers are measuring less forecasting skill than they appear to.

--------------------------------------------------------------------- caveats

**The brief asked for the published numbers "from the paper's abstract". The
abstract contains no numerical results at all** - it is qualitative throughout
("outperforms all existing state-of-the-art algorithms"). The numbers below are
therefore cited from the results TABLES of the arXiv source, which is a
stronger citation anyway. See PUBLISHED.

**The paper's F1 is weighted, not macro.** In every DeepLOB row of those tables
Recall equals Accuracy exactly (84.47/84.47, 78.91/78.91), which is the
signature of weighted averaging - and Precision differs from Accuracy, which
rules out micro. A macro F1 set beside a weighted F1 is not like-for-like, so
this script reports BOTH for the null model and puts the weighted one in the
comparison column.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lobforge.models import LogisticBaseline                     # noqa: E402
from lobforge.training import (CLASSES, TrainConfig, evaluate,   # noqa: E402
                               log_trial, predict, set_seed, train)

# The DeepLOB repo ships the standard NoAuction_DecPre extraction of FI-2010.
# Verified reachable on 2026-09-05: HTTP 200, 56,278,154 bytes, application/zip,
# containing exactly the four Setup-2 files. Probed at runtime regardless -
# a URL that worked once is not a URL that works now.
DATA_URL = ("https://raw.githubusercontent.com/zcakhaa/"
            "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
            "master/data/data.zip")

TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = ("Test_Dst_NoAuction_DecPre_CF_7.txt",
              "Test_Dst_NoAuction_DecPre_CF_8.txt",
              "Test_Dst_NoAuction_DecPre_CF_9.txt")

HORIZONS = (10, 20, 30, 50, 100)      # the 5 label rows, in file order
N_FEATURES = 144                      # rows 0..143; rows 144..148 are labels

# Zhang, Zohren & Roberts, "DeepLOB: Deep Convolutional Neural Networks for
# Limit Order Books", arXiv:1808.03668. F1 % as printed in the results tables
# of the arXiv LaTeX source. WEIGHTED F1, not macro - see the module docstring.
#
#   setup2: "Setup 2: Experiment Results for the FI-2010 Dataset"
#           train = first 7 days, test = last 3 days. This is the split the
#           downloaded files implement, so it is the comparable one.
#   setup1: "Setup 1: Experiment Results for the FI-2010 Dataset"
#           9-fold anchored forward CV. Different protocol; shown only because
#           it is the sole published source for k=100.
#
# Absent horizons are absent from the paper. They are left blank rather than
# interpolated or borrowed from another table.
PUBLISHED = {
    "setup2": {10: 83.40, 20: 72.82, 50: 80.35},
    "setup1": {10: 77.66, 50: 74.96, 100: 76.58},
}

# FI-2010 codes direction as 1=up, 2=stationary, 3=down. The project orders
# classes (down, flat, up) = (0, 1, 2). Macro F1 is invariant to a relabelling,
# so this matters for reading the confusion matrix, not for the headline - but
# a confusion matrix with the axes silently swapped is worse than none.
FI_TO_PROJECT = {1: 2, 2: 1, 3: 0}

RETURN_LAGS = (1, 5, 10, 20, 50)
RV_WINDOWS = (10, 50)
FEATURE_NAMES = ([f"ret_{l}" for l in RETURN_LAGS]
                 + [f"rv_{w}" for w in RV_WINDOWS])
WARMUP = max(max(RETURN_LAGS), max(RV_WINDOWS))


# --------------------------------------------------------------- acquisition

def instructions(data: Path) -> None:
    print(f"""
  No FI-2010 files found in {data}/

  Expected (the standard NoAuction_DecPre extraction):
      {TRAIN_FILE}
""" + "".join(f"      {f}\n" for f in TEST_FILES) + f"""
  Either:
    1. python3 fi2010.py --download --data {data}
       (tries {DATA_URL[:60]}...)

    2. Download FI-2010 yourself, extract the four files above into
       {data}/, and re-run with --data {data}

  The dataset's canonical home is the Etsin/Fairdata repository:
      https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  Cite Ntakaris et al., "Benchmark dataset for mid-price forecasting of limit
  order book data with machine learning methods" (arXiv:1705.03233).
""")


def download(data: Path) -> bool:
    """Fetch and extract. Returns False - loudly - if the URL is not reachable.

    Nothing here falls back to a second mirror or retries in a loop. If the one
    verified URL is gone, the honest move is to say so and let the operator
    supply the files.
    """
    data.mkdir(parents=True, exist_ok=True)
    zpath = data / "fi2010.zip"
    print(f"  GET {DATA_URL}")
    try:
        req = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "lobforge-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            size = int(resp.headers.get("Content-Length") or 0)
            ctype = resp.headers.get("Content-Type", "?")
            print(f"  {resp.status}  {size:,} bytes  {ctype}")
            with zpath.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"\n  DOWNLOAD FAILED: {e}")
        print("  The URL above did not resolve. Not substituting another one.")
        return False

    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            print(f"  archive contains {len(names)} files")
            z.extractall(data)
    except zipfile.BadZipFile as e:
        print(f"\n  NOT A ZIP: {e} - the URL returned something else.")
        return False
    zpath.unlink(missing_ok=True)

    missing = [f for f in (TRAIN_FILE, *TEST_FILES) if not (data / f).exists()]
    if missing:
        print(f"\n  archive did not contain: {missing}")
        return False
    print(f"  -> {data}/")
    return True


# ------------------------------------------------------------------- loading

def load_matrix(path: Path) -> np.ndarray:
    """FI-2010 stores FEATURES AS ROWS: (149, n_events), whitespace separated.

    149 lines, so parsing line by line is both the simplest and the fastest
    thing available - np.loadtxt over the 607 MB training file is minutes of
    work for the same answer, and pandas is not a dependency of this project.

    Streamed rather than read whole: the training file is 607 MB of text and
    holding the string and its split copy at once costs over a gigabyte for no
    reason. One line at a time is ~4 MB.
    """
    t0 = time.time()
    with path.open() as fh:
        rows = [np.array(ln.split(), dtype=np.float64) for ln in fh if ln.strip()]
    a = np.vstack(rows)
    if a.shape[0] != N_FEATURES + len(HORIZONS):
        raise ValueError(f"{path.name}: expected "
                         f"{N_FEATURES + len(HORIZONS)} rows, got {a.shape[0]}")
    print(f"    {path.name:<42}{a.shape[1]:>9,} events  "
          f"{time.time() - t0:.1f}s")
    return a


def load_split(data: Path):
    """Setup 2: train on the first 7 days, test on the last 3.

    Returns (train_matrix, test_matrix, test_file_edges). The edges mark where
    one test FILE ends and the next begins; those joins are not continuous
    price series and must not be treated as such.
    """
    print("  train:")
    tr = load_matrix(data / TRAIN_FILE)
    print("  test:")
    mats = [load_matrix(data / f) for f in TEST_FILES]
    edges, n = [], 0
    for m in mats[:-1]:
        n += m.shape[1]
        edges.append(n)
    return tr, np.hstack(mats), edges


# ------------------------------------------------------------------ features

def segment_bounds(mid: np.ndarray, known_edges, jump_bp: float) -> list:
    """Contiguous runs of one instrument.

    FI-2010 concatenates five stocks (and ten days) into one column axis with
    no delimiter. DecPre normalisation scales each instrument by its own power
    of ten, so a stock boundary appears as an enormous jump in the mid - and a
    return computed across one is fiction.

    This is the same discipline as the trusted-interval index in
    build_dataset.py: a rolling statistic never spans a discontinuity. The
    separation is not marginal. On Test_CF_9 the largest genuine one-step move
    is 14.9 bp (p99.9) and the smallest boundary jump is 2,271 bp, so any
    threshold in between finds exactly the four boundaries that five stocks
    imply.
    """
    r = np.abs(np.diff(np.log(mid))) * 10_000
    cuts = sorted(set([0, len(mid)]) | set(known_edges)
                  | set((np.where(r > jump_bp)[0] + 1).tolist()))
    return [[a, b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def build_features(mid: np.ndarray, segments) -> np.ndarray:
    """Returns at several lags plus realised volatility. Nothing else.

    Deliberately blind to the book: no depth, no imbalance, no spread, no
    level-2..10 anything. Only the level-1 mid and its own history. That is
    what makes a high score damning rather than interesting - there is no
    microstructure information in here to explain it.
    """
    n = len(mid)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    logm = np.log(np.maximum(mid, 1e-12))

    for lo, hi in segments:
        lm = logm[lo:hi]
        m = hi - lo
        cols = []
        for lag in RETURN_LAGS:
            c = np.zeros(m)
            if m > lag:
                c[lag:] = (lm[lag:] - lm[:-lag]) * 10_000
            cols.append(c)
        r1 = np.zeros(m)
        r1[1:] = (lm[1:] - lm[:-1]) * 10_000
        csq = np.concatenate([[0.0], np.cumsum(r1 ** 2)])
        idx = np.arange(m)
        for w in RV_WINDOWS:
            lo_i = np.maximum(0, idx - w + 1)
            cols.append(np.sqrt(np.maximum(csq[idx + 1] - csq[lo_i], 0.0)))
        out[lo:hi] = np.column_stack(cols).astype(np.float32)
    return out


def valid_rows(segments, n: int, max_k: int) -> np.ndarray:
    """Rows with a full feature warm-up AND a label that does not look past the
    end of its own instrument.

    The second half of that matters and is not something the benchmark itself
    does: FI-2010's labels are computed over the concatenated column axis, so
    the last k rows of every stock are labelled using the FIRST rows of the
    next stock. Those labels are meaningless. Dropping them costs under 2% of
    rows and removes contamination rather than adding any.
    """
    out = []
    for a, b in segments:
        lo, hi = a + WARMUP, b - max_k
        if hi > lo:
            out.append(np.arange(lo, hi, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def split_fit_val(segments, max_k: int, val_frac: float):
    """Hold out the tail of EVERY instrument, not the tail of the file.

    The obvious split - last 10% of the training rows - lands entirely inside
    the last stock, because the file is instrument-major. Early stopping then
    selects on one instrument and fires at epoch 1, which understates the
    model. Taking a tail from each segment keeps validation representative
    while still being strictly later-in-time within each instrument.

    Purged by max_k at each internal boundary so no fitting row carries a label
    computed from validation rows.
    """
    fit, val = [], []
    for a, b in segments:
        lo, hi = a + WARMUP, b - max_k
        if hi <= lo:
            continue
        idx = np.arange(lo, hi, dtype=np.int64)
        cut = int(len(idx) * (1 - val_frac))
        fit.append(idx[: max(0, cut - max_k)])
        val.append(idx[cut:])
    return np.concatenate(fit), np.concatenate(val)


class Rows:
    """Feature vectors z-scored with TRAINING statistics."""

    def __init__(self, feats, idx, labels, mu, sd):
        self.feats, self.idx, self.labels = feats, idx, labels
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        x = (self.feats[int(self.idx[i])] - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[i])


def weighted_f1(m) -> float:
    """Support-weighted F1 - what the DeepLOB tables report."""
    tot = sum(p["support"] for p in m.per_class.values())
    if not tot:
        return 0.0
    return sum(p["f1"] * p["support"] for p in m.per_class.values()) / tot


# ---------------------------------------------------------------------- main

def run(a) -> int:
    data = Path(a.data)

    if a.download and not download(data):
        instructions(data)
        return 1
    if not (data / TRAIN_FILE).exists():
        instructions(data)
        return 1

    print("\n" + "=" * 72)
    print("  FI-2010, Setup 2 (train = first 7 days, test = last 3)")
    print("=" * 72)
    tr_m, te_m, te_edges = load_split(data)

    tr_mid = (tr_m[0] + tr_m[2]) / 2          # level-1 ask / bid
    te_mid = (te_m[0] + te_m[2]) / 2
    if (tr_m[0] <= tr_m[2]).any() or (te_m[0] <= te_m[2]).any():
        print("  WARNING: crossed level-1 quotes present")

    tr_seg = segment_bounds(tr_mid, [], a.jump_bp)
    te_seg = segment_bounds(te_mid, te_edges, a.jump_bp)
    print(f"\n  instruments/segments   train {len(tr_seg)}   test {len(te_seg)}")
    print(f"  features               {', '.join(FEATURE_NAMES)}")
    print(f"                         level-1 mid only - no depth, no imbalance")

    tr_f = build_features(tr_mid, tr_seg)
    te_f = build_features(te_mid, te_seg)

    max_k = max(HORIZONS)
    tr_idx = valid_rows(tr_seg, tr_m.shape[1], max_k)
    te_idx = valid_rows(te_seg, te_m.shape[1], max_k)
    print(f"  usable rows            train {len(tr_idx):,}"
          f" of {tr_m.shape[1]:,}   test {len(te_idx):,} of {te_m.shape[1]:,}")

    # Validation is a per-instrument tail of TRAIN. Using the test set for
    # early stopping - even strided - would select the model on the data it is
    # scored on.
    fit_idx, val_idx = split_fit_val(tr_seg, max_k, a.val_frac)
    print(f"  fit / val / test       {len(fit_idx):,} / {len(val_idx):,}"
          f" / {len(te_idx):,}")

    run_id = f"fi2010-null-{int(time.time())}"
    ledger = Path(a.runs) / "trials.jsonl"
    log_trial(ledger, {"run_id": run_id, "model": "returns-null",
                       "dataset": "FI-2010 NoAuction_DecPre Setup2",
                       "horizons": list(HORIZONS), "features": FEATURE_NAMES,
                       "epochs": a.epochs, "seed": a.seed, "status": "started"})

    # Normalisation fitted on the FIT rows only - not train+val, not the whole
    # file. sd guarded so a constant column cannot divide by zero.
    ffit = tr_f[fit_idx]
    mu = ffit.mean(0)
    sd = np.where(ffit.std(0) > 1e-9, ffit.std(0), 1.0)

    results = []
    for h_i, k in enumerate(HORIZONS):
        row = N_FEATURES + h_i
        y_tr_all = np.array([FI_TO_PROJECT[int(v)] for v in tr_m[row]])
        y_te_all = np.array([FI_TO_PROJECT[int(v)] for v in te_m[row]])

        y_fit, y_val = y_tr_all[fit_idx], y_tr_all[val_idx]
        y_te = y_te_all[te_idx]
        mix = np.bincount(y_fit, minlength=3) / len(y_fit)

        print(f"\n  {'-' * 68}")
        print(f"  horizon k = {k}")
        print("    train label mix  " + "  ".join(
            f"{c} {p:.1%}" for c, p in zip(CLASSES, mix)))

        set_seed(a.seed)
        model = LogisticBaseline(len(FEATURE_NAMES))
        train(model,
              Rows(tr_f, fit_idx, y_fit, mu, sd),
              Rows(tr_f, val_idx, y_val, mu, sd),
              TrainConfig(epochs=a.epochs, batch_size=a.batch, lr=1e-3,
                          seed=a.seed, patience=a.patience),
              log=lambda m: print("     ", m))

        from torch.utils.data import DataLoader
        te_ds = Rows(te_f, te_idx, y_te, mu, sd)
        yt, yp = predict(model, DataLoader(te_ds, batch_size=4096), "cpu")
        m = evaluate(yt, yp)
        wf1 = weighted_f1(m)
        print(f"    macro F1 {m.macro_f1:.4f}   weighted F1 {wf1:.4f}   "
              f"accuracy {m.accuracy:.4f}   (majority {m.majority_accuracy:.4f})")
        results.append({"k": k, "macro_f1": m.macro_f1, "weighted_f1": wf1,
                        "accuracy": m.accuracy,
                        "majority_accuracy": m.majority_accuracy,
                        "n_test": m.n, **m.as_dict()})

    # -------------------------------------------------------------- the table
    print("\n" + "=" * 72)
    print("  NULL MODEL (price history only) vs PUBLISHED DeepLOB")
    print("=" * 72)
    print(f"  {'horizon':>8}{'null macro':>12}{'null wtd':>10}"
          f"{'DeepLOB S2':>12}{'gap':>9}{'DeepLOB S1':>12}{'gap':>9}")
    print(f"  {'k':>8}{'F1':>12}{'F1':>10}{'wtd F1':>12}{'(wtd)':>9}"
          f"{'wtd F1':>12}{'(wtd)':>9}")
    print("  " + "-" * 70)
    for r in results:
        k = r["k"]
        line = f"  {k:>8}{r['macro_f1']:>12.4f}{r['weighted_f1']:>10.4f}"
        for setup in ("setup2", "setup1"):
            pub = PUBLISHED[setup].get(k)
            if pub is None:
                line += f"{'-':>12}{'-':>9}"
            else:
                line += f"{pub / 100:>12.4f}{pub / 100 - r['weighted_f1']:>+9.4f}"
        print(line)
    print("  " + "-" * 70)
    print("  S2 = paper's Setup 2 (same split as these files) - the comparable")
    print("       column. S1 = Setup 1, 9-fold anchored CV, shown only because")
    print("       it is the sole published source for k=100.")
    print("  '-' = not reported by the paper for that horizon. Not estimated.")
    print("  Published F1 is WEIGHTED, not macro (Recall == Accuracy in every")
    print("  DeepLOB row). The null's weighted column is the like-for-like one.")

    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\n  Null model peaks at macro F1 {best['macro_f1']:.4f} "
          f"(k={best['k']}), using no book information whatever.")

    out = Path(a.runs) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "args": vars(a),
                               "features": FEATURE_NAMES,
                               "published": PUBLISHED,
                               "horizons": results}, indent=2))
    log_trial(ledger, {"run_id": run_id, "status": "finished",
                       "macro_f1": {r["k"]: r["macro_f1"] for r in results},
                       "weighted_f1": {r["k"]: r["weighted_f1"]
                                       for r in results}})
    print(f"\n  -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./fi2010-data",
                   help="directory holding the extracted FI-2010 .txt files")
    p.add_argument("--download", action="store_true",
                   help="fetch the dataset first; fails loudly if unreachable")
    p.add_argument("--runs", default="./runs")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac",
                   help="tail fraction of EACH instrument held out for early "
                        "stopping; never the test set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jump-bp", type=float, default=100.0, dest="jump_bp",
                   help="one-step |log return| above which a column join is "
                        "treated as an instrument boundary rather than a "
                        "price move (real p99.9 is ~15 bp, boundaries ~2000+)")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
