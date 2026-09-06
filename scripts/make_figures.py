#!/usr/bin/env python3
"""Every figure in the README, from the run artefacts. Nothing is retrained.

    python3 scripts/make_figures.py                  # all six
    python3 scripts/make_figures.py latency ablation # just those

Reads runs/*.json (measurements), dataset/ (the arrays those measurements were
computed on) and data/ (the raw archive, for coverage and the REST round trip).
Writes docs/figures/*.png plus figure-data.json, which records every number
that appears on a figure and the file it came from - the figures are committed
but runs/ is not, so that sidecar is what makes the README checkable.
"""

from __future__ import annotations

import json
import gzip
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATASET = ROOT / "dataset"
DATA = ROOT / "data"
OUT = ROOT / "docs" / "figures"

# One place for the look, so six figures cannot drift apart.
INK, GRID = "#1b1b1b", "#d8d8d8"
BLUE, RED, GREY, GREEN = "#1f6fb2", "#c0392b", "#8c8c8c", "#3f7d4e"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

FACTS: dict = {}          # what each figure asserts, and where it came from


def newest(pattern: str) -> Path:
    hits = list(RUNS.glob(pattern))
    if not hits:
        raise SystemExit("no run matching runs/" + pattern + " - run the "
                         "experiment that produces it first (see the README)")
    return max(hits, key=lambda p: p.stat().st_mtime)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print("  " + str(path.relative_to(ROOT)))


def rest_rtt_ms():
    """REST round trips the collector actually measured, from the archive.

    Every snapshot record carries the monotonic clock at send and at receive,
    so this is the machine's real distance from the exchange rather than a
    ping estimate taken at a different time under different load.
    """
    rtts = []
    for f in sorted(DATA.glob("snapshots/date=*/hour=*/*.jsonl.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("sent"):
                        rtts.append((r["t"] - r["sent"]) / 1e6)
        except (OSError, EOFError, json.JSONDecodeError):
            continue          # a live .part or a torn tail; the rest still counts
    rtts.sort()
    return rtts, len(rtts)


# --------------------------------------------------------------- figure 1
def fig_latency() -> None:
    src = newest("costfloor-k*.json")
    d = json.loads(src.read_text())
    sweep = [r for r in d["latency_sweep"] if "boot_ci_bp" in r]
    if not sweep:
        raise SystemExit(src.name + " carries no bootstrap CI; rerun "
                         "costfloor.py with --bootstrap above zero")
    x = np.array([r["latency_ms"] for r in sweep], float)
    y = np.array([r["gross_bp"] for r in sweep], float)
    lo = np.array([r["boot_ci_bp"][0] for r in sweep], float)
    hi = np.array([r["boot_ci_bp"][1] for r in sweep], float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.18, linewidth=0,
                    label="95% moving-block bootstrap CI")
    ax.plot(x, y, color=BLUE, marker="o", markersize=4.5, linewidth=1.8,
            label="gross edge per trade")
    ax.axhline(10.0, color=RED, linestyle="--", linewidth=1.6,
               label="round-trip cost, 10 bp (Binance USD-M taker, 5 bp a side)")

    rtts, n_rtt = rest_rtt_ms()
    p50 = float(np.median(rtts)) if rtts else None
    if p50 is not None:
        ax.axvline(p50, color=GREY, linestyle=":", linewidth=1.6)
        ax.annotate("measured REST round trip\np50 %.0f ms  (n=%d)"
                    % (p50, n_rtt), xy=(p50, y.min()),
                    xytext=(p50 + 40, y.min() * 0.78),
                    color=GREY, fontsize=9, va="top")

    ax.set_yscale("log")
    ax.set_xlabel("decision latency (milliseconds)")
    ax.set_ylabel("gross edge per trade (basis points, log scale)")
    ax.set_title("The edge decays with latency, and never approaches the cost line")
    ax.set_ylim(y.min() * 0.5, 22)
    ax.legend(loc="upper left", fontsize=9)
    ax.annotate("the gap is the finding: %.0fx short of the cost, before "
                "latency" % (10.0 / y[0]),
                xy=(0.5, 0.42), xycoords="axes fraction", ha="center",
                color=RED, fontsize=9.5)
    save(fig, "fig1-latency-decay.png")

    at_p50 = float(np.interp(p50, x, y)) if p50 is not None else None
    FACTS["latency_decay"] = {
        "source": src.name,
        "gross_bp_by_latency_ms": dict((int(r["latency_ms"]), r["gross_bp"])
                                       for r in sweep),
        "boot_ci_bp_at_0ms": sweep[0]["boot_ci_bp"],
        "naive_ci_bp_at_0ms": sweep[0].get("naive_ci_bp"),
        "ci_width_ratio_at_0ms": sweep[0].get("ci_width_ratio"),
        "n_trades_at_0ms": sweep[0]["n_trades"],
        "trade_rate_at_0ms": sweep[0].get("trade_rate"),
        "rest_rtt_p50_ms": p50, "rest_rtt_samples": n_rtt,
        "rest_rtt_source": "data/snapshots/**/*.jsonl.gz  (t - sent)",
        "gross_bp_at_rest_p50": at_p50,
        "edge_retained_at_rest_p50": (at_p50 / sweep[0]["gross_bp"]
                                      if at_p50 else None),
        "round_trip_cost_bp": 10.0,
    }


# --------------------------------------------------------------- figure 2
def fig_costfloor() -> None:
    src = newest("costfloor-k*.json")
    d = json.loads(src.read_text())
    g0 = d["latency_sweep"][0]["gross_bp"]
    costs = [0, 5, 10, 15]
    net = [g0 - c for c in costs]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar([str(c) for c in costs], net,
                  color=[GREEN if v > 0 else RED for v in net], width=0.6)
    ax.axhline(0, color=INK, linewidth=1.0)
    for b, v in zip(bars, net):
        ax.annotate("%+.3f" % v, (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v > 0 else "top", fontsize=9.5,
                    xytext=(0, 3 if v > 0 else -3), textcoords="offset points")
    ax.set_xlabel("round-trip cost charged (basis points)")
    ax.set_ylabel("net edge per trade (basis points)")
    ax.set_title("Net edge at zero latency, the most generous case there is")
    ax.annotate("gross edge %.4f bp, so the floor is crossed between\n"
                "0 and 5 bp. Retail taker pays 10." % g0,
                xy=(0.03, 0.30), xycoords="axes fraction", fontsize=9,
                color=GREY)
    save(fig, "fig2-cost-floor.png")
    FACTS["cost_floor"] = {"source": src.name, "gross_bp_zero_latency": g0,
                           "net_bp_by_round_trip_cost_bp": dict(zip(costs, net))}


# --------------------------------------------------------------- figure 3
def dataset_windows_k(d: dict) -> int:
    """Thousands of windows in the dataset build a run was computed on.

    Walk-forward with f folds tests on f/(f+1) of the data, so the fold count
    has to be divided back out before two runs with different fold counts can
    be recognised as coming from the same dataset. Rounded to 1k: far below the
    gap between successive dataset builds, far above the split remainder.
    """
    folds = len(d["folds"])
    return round(sum(f["n"] for f in d["folds"]) * (folds + 1) / folds / 1000)


def fig_ablation() -> None:
    by_key: dict = {}
    for p in RUNS.glob("*-forward-k*.json"):
        d = json.loads(p.read_text())
        model = d["args"]["model"]
        by_key.setdefault(dataset_windows_k(d), {}) \
              .setdefault(model, []).append((p.stat().st_mtime, p, d))

    want = set(["returns", "baseline", "deeplob"])
    shared = [k for k, v in by_key.items() if want <= set(v)]
    if not shared:
        raise SystemExit(
            "no dataset build has runs for all of returns/baseline/deeplob, so "
            "the three-model comparison would be across different data. Have: "
            + json.dumps(dict((k, sorted(v)) for k, v in by_key.items())))
    key = max(shared)                    # the largest build that has all three
    picked = dict((m, max(v)[1:]) for m, v in by_key[key].items() if m in want)

    order = ["returns", "baseline", "deeplob"]
    labels = {"returns": "returns only\n(4 price features)",
              "baseline": "logistic baseline\n(14 book features)",
              "deeplob": "DeepLOB\n(raw 40-wide book)"}

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for i, m in enumerate(order):
        d = picked[m][1]
        f1 = [f["macro_f1"] for f in d["folds"]]
        ax.scatter(i + np.linspace(-.08, .08, len(f1)), f1, s=34, color=BLUE,
                   zorder=3, alpha=.85, label="per fold" if i == 0 else None)
        ax.hlines(d["mean_macro_f1"], i - .22, i + .22, color=RED,
                  linewidth=2.2, zorder=4, label="mean" if i == 0 else None)
        ax.annotate("%.3f" % d["mean_macro_f1"], (i + .26, d["mean_macro_f1"]),
                    color=RED, fontsize=9.5, va="center")
        ax.annotate("%d folds" % len(f1), (i, 0.262), ha="center",
                    fontsize=8.5, color=GREY)
    ax.set_xticks(range(3))
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(0.25, 0.68)
    ax.set_ylabel("macro $F_1$ over 3 classes (down / flat / up)")
    ax.set_title("Architecture buys nothing the book features do not already give"
                 "\npurged walk-forward, k=100 frames (10 s), ~%dk windows" % key)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig3-ablation.png")

    FACTS["ablation"] = {
        "dataset_windows_thousands": key,
        "models": dict(
            (m, {"source": p.name, "folds": len(d["folds"]),
                 "stride": d["args"].get("stride"),
                 "epochs": d["args"].get("epochs"),
                 "mean_macro_f1": d["mean_macro_f1"],
                 "mean_accuracy": d["mean_accuracy"],
                 "per_fold_macro_f1": [f["macro_f1"] for f in d["folds"]],
                 "per_fold_accuracy": [f["accuracy"] for f in d["folds"]],
                 "per_fold_majority_accuracy":
                     [f["majority_accuracy"] for f in d["folds"]],
                 "n_train_per_fold": [f["n_train"] for f in d["folds"]],
                 "n_test_per_fold": [f["n"] for f in d["folds"]]})
            for m, (p, d) in picked.items()),
    }


# --------------------------------------------------------------- figure 4
def fig_fi2010() -> None:
    src = newest("fi2010-null-*.json")
    d = json.loads(src.read_text())
    hs = [h["k"] for h in d["horizons"]]
    null = [h["weighted_f1"] * 100 for h in d["horizons"]]
    pub = d["published"]
    s1 = [pub["setup1"].get(str(k)) for k in hs]
    s2 = [pub["setup2"].get(str(k)) for k in hs]

    x = np.arange(len(hs), dtype=float)
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - w, null, w, color=BLUE, label="null model, price history only")
    for series, off, col, lab in ((s1, 0.0, GREY, "DeepLOB published, setup 1"),
                                  (s2, w, GREEN, "DeepLOB published, setup 2")):
        xs = [x[i] + off for i, v in enumerate(series) if v is not None]
        ys = [v for v in series if v is not None]
        ax.bar(xs, ys, w, color=col, label=lab)
    for i, v in enumerate(null):
        ax.annotate("%.1f" % v, (x[i] - w, v), ha="center", va="bottom",
                    fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    missing = [str(k) for k, a, b in zip(hs, s1, s2) if a is None and b is None]
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in hs])
    ax.set_xlabel("prediction horizon k (events ahead)" + (
        "\nno published bar at k=%s: the paper does not report that horizon"
        % ", ".join(missing) if missing else ""))
    ax.set_ylabel("weighted $F_1$ (%)")
    ax.set_ylim(0, 100)
    ax.set_title("FI-2010: a model that never sees the order book,\n"
                 "against the published order-book architecture")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig4-fi2010-null.png")

    FACTS["fi2010"] = {
        "source": src.name, "features": d["features"],
        "n_test": d["horizons"][0]["n_test"],
        "null_weighted_f1_pct": dict((h["k"], h["weighted_f1"] * 100)
                                     for h in d["horizons"]),
        "null_macro_f1": dict((h["k"], h["macro_f1"]) for h in d["horizons"]),
        "majority_accuracy": dict((h["k"], h["majority_accuracy"])
                                  for h in d["horizons"]),
        "published_deeplob_weighted_f1_pct": pub,
    }


# --------------------------------------------------------------- figure 5
def fig_coverage() -> None:
    sizes: dict = {}
    for p in DATA.glob("depth/date=*/hour=*/*"):
        date = p.parent.parent.name.split("=")[1]
        hour = int(p.parent.name.split("=")[1])
        t = datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]), hour,
                     tzinfo=timezone.utc)
        sizes[t] = sizes.get(t, 0) + p.stat().st_size
    if not sizes:
        raise SystemExit("no depth partitions under data/ - nothing to plot")

    lo, hi = min(sizes), max(sizes)
    span = int((hi - lo).total_seconds() // 3600) + 1
    hours = [lo + timedelta(hours=i) for i in range(span)]
    mb = [sizes.get(h, 0) / 1e6 for h in hours]
    covered = sum(1 for v in mb if v > 0)

    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.bar(hours, mb, width=1 / 24.0 * 0.9, color=BLUE, linewidth=0)
    ax.set_xlabel("UTC hour partition")
    ax.set_ylabel("archived depth events (MB per hour)")
    ax.set_title("Collection coverage: %d of the %d hours from %s to %s "
                 "carry data" % (covered, span, lo.strftime("%b %d"),
                                 hi.strftime("%b %d")))
    ax.annotate("Gaps are the collector down, not the market quiet:\n"
                "this ran on a laptop that sleeps. Every gap\n"
                "ends a trusted run.",
                xy=(0.615, 0.97), xycoords="axes fraction", fontsize=9,
                color=GREY, va="top")
    fig.autofmt_xdate()
    save(fig, "fig5-coverage.png")

    FACTS["coverage"] = {
        "source": "data/depth/date=*/hour=*",
        "first_hour_utc": lo.isoformat(), "last_hour_utc": hi.isoformat(),
        "hours_spanned": span, "hours_with_data": covered,
        "coverage_fraction": covered / float(span),
        "total_depth_bytes": int(sum(sizes.values())),
        "median_mb_per_covered_hour": float(np.median([v for v in mb if v > 0])),
    }


# --------------------------------------------------------------- figure 6
def fig_heatmap(minutes: int = 60) -> None:
    frames = np.load(DATASET / "frames.npy", mmap_mode="r")
    mid = np.load(DATASET / "mid.npy")
    event_ms = np.load(DATASET / "event_ms.npy")
    intervals = json.loads((DATASET / "intervals.json").read_text())
    tick = json.loads((DATASET / "manifest.json").read_text())["tick"]

    a, b = max(intervals, key=lambda ab: ab[1] - ab[0])
    want = minutes * 60 * 10                       # frames, at 100 ms each
    if (b - a) > want:                             # take it from the middle
        a = a + ((b - a) - want) // 2
    b = min(b, a + want)
    step = max(1, (b - a) // 900)
    idx = np.arange(a, b, step)

    row = np.asarray(frames[idx], dtype=np.float64)
    m = mid[idx]
    ap, aq = row[:, 0::4], row[:, 1::4]
    bp, bq = row[:, 2::4], row[:, 3::4]

    off = np.concatenate([(ap - m[:, None]) / tick,
                          (bp - m[:, None]) / tick], axis=1)
    qty = np.concatenate([aq, bq], axis=1)
    span = int(np.ceil(np.percentile(np.abs(off), 99)))
    grid = np.zeros((2 * span + 1, len(idx)))
    r = np.rint(off).astype(int) + span
    c = np.broadcast_to(np.arange(len(idx))[:, None], off.shape)
    keep = (r >= 0) & (r < grid.shape[0])
    np.add.at(grid, (r[keep], c[keep]), qty[keep])

    t0 = datetime.fromtimestamp(event_ms[idx[0]] / 1000.0, timezone.utc)
    mins = (event_ms[idx] - event_ms[idx[0]]) / 60000.0

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    im = ax.imshow(np.log1p(grid), aspect="auto", origin="lower", cmap="magma",
                   extent=[0, float(mins[-1]), -span * tick, span * tick])
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.55)
    ax.grid(False)
    ax.set_xlabel("minutes from " + t0.strftime("%Y-%m-%d %H:%M") + " UTC")
    ax.set_ylabel("price offset from mid (USDT, %g tick)" % tick)
    ax.set_title("Resting depth around the mid: one unbroken trusted run, "
                 "%s frames\nsampled every %.1f s into %d columns"
                 % (format(b - a, ","), step / 10.0, len(idx)))
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("log(1 + resting size) (BTC)")
    cb.outline.set_visible(False)
    save(fig, "fig6-book-heatmap.png")

    FACTS["book_heatmap"] = {
        "source": "dataset/frames.npy", "tick": tick,
        "frame_range": [int(a), int(b)], "frames_plotted": int(len(idx)),
        "start_utc": t0.isoformat(), "minutes": float(mins[-1]),
        "levels_per_side": int(ap.shape[1]),
        "price_span_usdt": float(span * tick),
    }


FIGURES = {"latency": fig_latency, "costfloor": fig_costfloor,
           "ablation": fig_ablation, "fi2010": fig_fi2010,
           "coverage": fig_coverage, "heatmap": fig_heatmap}


def main() -> int:
    want = sys.argv[1:] or list(FIGURES)
    bad = [w for w in want if w not in FIGURES]
    if bad:
        raise SystemExit("unknown figure(s) %s; have %s" % (bad, list(FIGURES)))
    for name in want:
        FIGURES[name]()
    if set(want) == set(FIGURES):
        path = OUT / "figure-data.json"
        path.write_text(json.dumps(FACTS, indent=2, default=float))
        print("  " + str(path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
