#!/usr/bin/env python3
"""Does undetected desync manufacture predictability?

    python3 gapstudy.py                       # 24 hour-partitions, default rates
    python3 gapstudy.py --hours 0             # the whole archive (slow)
    python3 gapstudy.py --rates 0,0.01,0.05

-------------------------------------------------------------- the hypothesis

A book reconstructed across a sequence gap is not a book. The events that went
missing were the ones that moved or removed levels, so a pipeline that splices
straight across the gap carries stale levels forward and then jumps when some
later event finally touches them. That jump is a PHANTOM: it never happened in
the market, it happened in the reconstruction.

Phantom jumps should be trivially predictable. The stale book carries a visible
signature - a level sitting at a price the market has already left, an
imbalance that cannot persist - and the jump that follows is mechanically
determined by it. A model that sees the book immediately before the splice can
learn that signature, and it will be rewarded for it.

So the prediction under test is:

    ignoring the trust index INFLATES measured predictability,
    and the inflation GROWS with the gap rate p.

If that holds, then every LOB result computed on a pipeline without an explicit
trust index is contaminated by an unknown amount, and the amount depends on
data quality nobody reports.

------------------------------------------------------------- why this is here

This is the study that needs raw archived events with their sequence numbers
intact. Vendor data arrives pre-cleaned: the gaps have already been repaired,
or silently interpolated, and the sequence fields are gone. You cannot ask this
question of a dataset that has already answered it for you.

The gap injection reuses the sandbox's FaultInjector, which is the same
protocol-correct dropper the collector is soak-tested against - applied offline
to archived events rather than to a live socket.

---------------------------------------------------------------- what it does

For each rate p, drop that fraction of depth events, then reconstruct twice:

    trusted   sequence discontinuities desync the engine and end the trusted
              run. Windows never span a gap. This is what build_dataset.py does.
    naive     the pu check is disabled, so the engine splices across the gap
              and emits one continuous run. This is what a pipeline that does
              not track sequence numbers produces.

Both then get the same features, the same purged walk-forward, the same model.
The only difference is whether the trust index was respected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Reused rather than reimplemented: this script's whole point is to run the
# EXISTING measurement over perturbed inputs. Redefining the window/label/
# normalisation logic here would let it drift from what costfloor.py actually
# does, and then the comparison would measure the drift instead of the gaps.
from build_dataset import partitions, read_jsonl                 # noqa: E402
from costfloor import (FeatureRows, forward_bp, gross_edge,      # noqa: E402
                       label_from_bp, moving_block_ci, valid_ends)
from features import compute_interval                            # noqa: E402
from lobforge.book import BookEngine, BookState, Desync                     # noqa: E402
from lobforge.models import LogisticBaseline                     # noqa: E402
from lobforge.sandbox.simulator import FaultConfig, FaultInjector  # noqa: E402
from lobforge.training import (TrainConfig, evaluate, log_trial,  # noqa: E402
                               predict, set_seed, train)

DEFAULT_RATES = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)


# --------------------------------------------------------------- reconstruct

def stream_events(data: Path, hours: int):
    """Archived depth events with their snapshots, partition by partition.

    Materialised per partition rather than all at once: the full archive is
    834 MB compressed and holding every event as a dict would not fit
    comfortably, but one hour at a time is nothing.
    """
    depth = partitions(data, "depth")
    snaps = partitions(data, "snapshots")
    keys = sorted(depth)
    if hours:
        keys = keys[:hours]
    for h in keys:
        s = [r["m"] for f in snaps.get(h, []) for r in read_jsonl(f)
             if "m" in r and "lastUpdateId" in r.get("m", {})]
        e = [r["m"]["data"] for f in depth[h] for r in read_jsonl(f)
             if "m" in r and "data" in r["m"]]
        yield h, s, e


def reconstruct(data: Path, hours: int, p: float, trust: bool, seed: int,
                tick: float, levels: int):
    """Replay the archive with `p` of depth events dropped.

    Two arms, one variable. Both see the same surviving events; they differ
    only in what they do when the sequence breaks.

      trusted  desync, record a run boundary, then re-anchor on a fresh book -
               the offline stand-in for the collector's gap refetch. Windows
               never span a gap, and the book after one is correct.
      naive    the pu check is off, so the break is invisible: the stale
               levels the missing events would have moved stay in the book and
               the run runs on.

    Without the re-anchor the trusted arm would be a strawman: the archive
    snapshots every 30 minutes, so a single dropped packet would blind it for
    half an hour and the comparison would measure that blindness instead of
    the splicing. See BookEngine.resync_over.

    strict=False for BOTH arms. The naive book WILL cross - that is the
    phantom showing up as an invariant violation - and raising on it would
    just prevent the measurement. Crossings are counted and reported instead,
    because the count is itself part of the answer.
    """
    engine = BookEngine(tick=tick, levels=levels, strict=False,
                        trust_sequence=trust)
    # One injector across the whole replay, so the dropped events are the same
    # positions in the stream regardless of how it is partitioned - and the
    # same positions in both arms, which is what makes them comparable.
    inj = FaultInjector(FaultConfig(drop_rate=p), seed=seed) if p else None

    rows, mids, spreads, intervals, uids = [], [], [], [], []
    run_start = 0
    missed, resyncs = [], 0
    for _, snaps, events in stream_events(data, hours):
        engine.add_snapshots(snaps)
        for ev in events:
            if inj is not None and not inj.process(ev):
                missed.append(ev)
                continue
            item = next(engine.feed([ev]), None)
            if isinstance(item, Desync):
                if len(rows) > run_start:
                    intervals.append([run_start, len(rows)])
                run_start = len(rows)
                # Repair ONLY a hole this script punched. The archive has
                # real gaps of its own, and pu must chain exactly onto the
                # last event we withheld for the missing events to be the
                # whole story; if it does not, the data is genuinely absent
                # and the engine waits for a snapshot, exactly as it would
                # live. Repairing anyway splices across a real hole and
                # leaves a crossed book for tens of thousands of frames -
                # which is the very contamination this arm is the control
                # for.
                if (item.reason == "sequence_break" and missed
                        and int(ev["pu"]) == int(missed[-1]["u"])):
                    engine.resync_over(missed)
                    resyncs += 1
                    item = next(engine.feed([ev]), None)
            missed.clear()
            if not isinstance(item, BookState):
                continue
            rows.append(item.as_row(levels))
            mids.append(item.mid)
            spreads.append(item.spread)
            uids.append(item.update_id)
    if len(rows) > run_start:
        intervals.append([run_start, len(rows)])

    frames = np.asarray(rows, dtype=np.float32)
    mid = np.asarray(mids, dtype=np.float64)
    spread = np.asarray(spreads, dtype=np.float64)
    return SimpleNamespace(
        frames=frames, mid=mid, spread=spread, intervals=intervals,
        uid=np.asarray(uids, dtype=np.int64),
        desyncs=engine.desyncs, unanchored=engine.events_unanchored,
        dropped=(inj.dropped if inj else 0), resyncs=resyncs,
        crossed=int((spread <= 0).sum()) if len(spread) else 0)


def build_features(ds) -> np.ndarray:
    """Per trusted interval, never across one - same rule as features.py."""
    out = np.zeros((len(ds.mid), 14), dtype=np.float32)
    for lo, hi in ds.intervals:
        if hi - lo > 1:
            out[lo:hi] = compute_interval(ds.frames[lo:hi], ds.mid[lo:hi])
    return out


# ------------------------------------------------------------------ evaluate

def measure(ds, feats, a, ends=None):
    """Purged walk-forward, fold-local alpha and normalisation.

    Deliberately the same shape as costfloor.fold_predictions, with the labels
    kept so macro F1 can be reported beside the edge.

    `ends` overrides the window set. Passing the trusted arm's windows while
    handing over the naive arm's book is what makes the third row possible:
    same windows, same folds, same count - only the book differs.
    """
    if ends is None:
        ends = valid_ends(ds.intervals, a.window, a.k)
    if len(ends) < a.min_ends:
        return None
    lo, hi = int(ends.min()), int(ends.max())
    edges = np.linspace(lo, hi + 1, a.folds + 2).astype(np.int64)

    y_all, p_all, bp_all = [], [], []
    for f in range(a.folds):
        t0, t1 = int(edges[f + 1]), int(edges[f + 2])
        test = ends[(ends >= t0) & (ends < t1)]
        tr = ends[ends + a.k < t0 - a.k]
        if len(tr) < 5000 or len(test) < 1000:
            continue

        # alpha from the TRAINING fold only. A crossed book gives a negative
        # spread, so clip at zero rather than letting it drag alpha down.
        half = np.maximum((ds.spread[tr] / 2) / ds.mid[tr] * 10_000, 0.0)
        alpha = float(np.median(half))

        bp_tr = forward_bp(ds.mid, tr, a.k)
        bp_te = forward_bp(ds.mid, test, a.k)
        y_tr, y_te = label_from_bp(bp_tr, alpha), label_from_bp(bp_te, alpha)

        ftr = feats[tr]
        mu = ftr.mean(0)
        sd = np.where(ftr.std(0) > 1e-9, ftr.std(0), 1.0)

        set_seed(a.seed)
        model = LogisticBaseline(feats.shape[1])
        train(model, FeatureRows(feats, tr, y_tr, mu, sd),
              FeatureRows(feats, test[::10], y_te[::10], mu, sd),
              TrainConfig(epochs=a.epochs, batch_size=a.batch, lr=1e-3,
                          seed=a.seed, patience=2), log=lambda m: None)

        from torch.utils.data import DataLoader
        te = FeatureRows(feats, test, y_te, mu, sd)
        yt, yp = predict(model, DataLoader(te, batch_size=4096), "cpu")
        y_all.append(yt); p_all.append(yp); bp_all.append(bp_te)

    if not y_all:
        return None
    y = np.concatenate(y_all)
    pr = np.concatenate(p_all)
    bp = np.concatenate(bp_all)
    m = evaluate(y, pr)
    signed = gross_edge(bp, pr)
    edge = float(signed.mean()) if len(signed) else 0.0
    ci = moving_block_ci(signed, 2 * a.k, 400, a.seed) if len(signed) else (0, 0)
    return {"macro_f1": m.macro_f1, "accuracy": m.accuracy,
            "gross_bp": edge, "boot_ci_bp": list(ci),
            "n_trades": int(len(signed)), "n_windows": int(len(y))}


def matched_ends(t_ds, n_ds, t_ends, k):
    """The trusted arm's windows, expressed as positions in the naive arm.

    The two arms do not line up by index. The archive has real gaps of its own,
    and after each one the trusted arm goes quiet until the next 30-minute
    snapshot while the naive arm splices straight on, so the naive arm is the
    longer array. Update id is the clock they share: both arms applied the same
    surviving events, so every trusted state has a naive twin with the same u.

    Windows that fall off the end of the naive arm are dropped; there are none
    in practice, because the naive arm is never shorter.
    """
    want = t_ds.uid[t_ends]
    pos = np.searchsorted(n_ds.uid, want)
    keep = pos < len(n_ds.uid)
    pos, want = pos[keep], want[keep]
    pos = pos[n_ds.uid[pos] == want]
    return pos[pos + k + 1 < len(n_ds.uid)]


# ---------------------------------------------------------------------- main

def run(a) -> int:
    data = Path(a.data)
    rates = [float(x) for x in a.rates.split(",")]

    run_id = f"gapstudy-{int(time.time())}"
    ledger = Path(a.runs) / "trials.jsonl"
    log_trial(ledger, {"run_id": run_id, "model": "gapstudy", "k": a.k,
                       "window": a.window, "folds": a.folds, "rates": rates,
                       "hours": a.hours, "epochs": a.epochs, "seed": a.seed,
                       "status": "started"})

    print(f"  archive {data}   partitions "
          f"{a.hours or 'all'}   rates {rates}\n")

    results = []
    for p in rates:
        arms = {}
        for trust in (True, False):
            mode = "trusted" if trust else "naive"
            t0 = time.time()
            ds = reconstruct(data, a.hours, p, trust, a.seed, a.tick, a.levels)
            if not len(ds.mid):
                print(f"  p={p:<7} {mode:<8} no states reconstructed")
                continue
            arms[mode] = (ds, build_features(ds), valid_ends(
                ds.intervals, a.window, a.k), time.time() - t0)
        if len(arms) != 2:
            continue

        (t_ds, t_feats, t_ends, t_secs) = arms["trusted"]
        (n_ds, n_feats, n_ends, n_secs) = arms["naive"]

        todo = [("trusted", t_ds, t_feats, t_ends, t_secs),
                ("naive", n_ds, n_feats, n_ends, n_secs),
                # The control. Without it the trusted/naive gap is confounded
                # with sample size: the trusted arm trains on far fewer windows
                # at every p > 0, and less training data alone moves macro F1.
                ("naive@trusted", n_ds, n_feats,
                 matched_ends(t_ds, n_ds, t_ends, a.k), 0.0)]

        for mode, ds, feats, ends, secs in todo:
            m = measure(ds, feats, a, ends)
            # Usable windows is a headline number, not diagnostics: it is what
            # respecting the trust index COSTS, and it is why people skip it.
            rec = {"p": p, "mode": mode, "frames": int(len(ds.mid)),
                   "runs": len(ds.intervals), "desyncs": ds.desyncs,
                   "resyncs": ds.resyncs, "dropped": ds.dropped,
                   "crossed": ds.crossed, "usable_ends": int(len(ends)),
                   "secs": round(secs, 1), **(m or {})}
            results.append(rec)
            got = "macro_f1" in rec
            print(f"  p={p:<7} {mode:<14} frames {rec['frames']:>9,}"
                  f"  runs {rec['runs']:>7,}  windows {rec['usable_ends']:>9,}"
                  f"  crossed {rec['crossed']:>8,}"
                  + (f"  macroF1 {rec['macro_f1']:.4f}"
                     f"  edge {rec['gross_bp']:+.4f} bp"
                     if got else f"  {'too few windows':>28}"), flush=True)

    # ------------------------------------------------------------- the table
    W = 100
    print("\n" + "=" * W)
    print("  RECONSTRUCTION QUALITY AS A CONFOUND")
    print("=" * W)
    print(f"  {'drop p':>8}{'frames':>11}{'runs':>9}{'windows':>10}"
          f"{'crossed':>10}{'macro F1':>11}{'gross bp':>11}{'trades':>10}")
    for mode in ("trusted", "naive", "naive@trusted"):
        print(f"  -- {mode} " + "-" * (W - 8 - len(mode)))
        for r in results:
            if r["mode"] != mode:
                continue
            got = "macro_f1" in r
            print(f"  {r['p']:>8.3f}{r['frames']:>11,}{r['runs']:>9,}"
                  f"{r['usable_ends']:>10,}{r['crossed']:>10,}"
                  + (f"{r['macro_f1']:>11.4f}{r['gross_bp']:>11.4f}"
                     f"{r['n_trades']:>10,}" if got else
                     f"{'-':>11}{'-':>11}{'-':>10}"))

    print("\n" + "=" * W)
    print("  INFLATION FROM IGNORING THE TRUST INDEX")
    print("=" * W)
    print("  matched = the naive book scored on the TRUSTED arm's windows, so")
    print("  the only difference left is the book itself. full = what a naive")
    print("  pipeline would actually report, sample-size advantage included.")
    print(f"\n  {'drop p':>8}{'trusted F1':>13}{'matched F1':>13}"
          f"{'d matched':>12}{'full F1':>11}{'d full':>10}"
          f"{'windows kept':>15}")
    by = {(r["p"], r["mode"]): r for r in results}
    deltas = []
    for p in rates:
        t = by.get((p, "trusted"))
        n = by.get((p, "naive"))
        c = by.get((p, "naive@trusted"))
        if not (t and n):
            continue
        kept = t["usable_ends"] / max(1, n["usable_ends"])
        if "macro_f1" not in t:
            # The honest pipeline has nothing to say at this rate, while the
            # naive one still reports a number to four decimal places.
            full = f"{n['macro_f1']:>11.4f}" if "macro_f1" in n else f"{'-':>11}"
            print(f"  {p:>8.3f}{'-':>13}{'-':>13}{'-':>12}" + full +
                  f"{'-':>10}{kept:>14.1%}"
                  "  <- trusted arm cannot answer")
            continue
        d_c = c["macro_f1"] - t["macro_f1"] if c and "macro_f1" in c else None
        d_n = n["macro_f1"] - t["macro_f1"] if "macro_f1" in n else None
        if d_c is not None:
            deltas.append((p, d_c))
        print(f"  {p:>8.3f}{t['macro_f1']:>13.4f}"
              + (f"{c['macro_f1']:>13.4f}{d_c:>+12.4f}" if d_c is not None
                 else f"{'-':>13}{'-':>12}")
              + (f"{n['macro_f1']:>11.4f}{d_n:>+10.4f}" if d_n is not None
                 else f"{'-':>11}{'-':>10}")
              + f"{kept:>14.1%}")

    if len(deltas) > 1:
        ps = np.array([d[0] for d in deltas])
        f1s = np.array([d[1] for d in deltas])
        # Spearman via ranks - the hypothesis is monotone growth, not linearity.
        rho = float(np.corrcoef(ps.argsort().argsort(),
                                f1s.argsort().argsort())[0, 1])
        print(f"\n  rank correlation of (drop rate, matched F1 inflation): "
              f"{rho:+.3f}   over {len(deltas)} rates")
        print("  The hypothesis predicts a strongly positive value: splicing")
        print("  across more gaps should manufacture more predictability.")
        if rho <= 0.5:
            print("  IT DOES NOT. The hypothesis as stated is not supported by")
            print("  this archive - see the changelog for what happens instead.")
    elif deltas:
        print(f"\n  Only one rate produced a matched pair, so the trend the")
        print("  hypothesis is about cannot be tested. Raise --hours.")

    out = Path(a.runs) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "args": vars(a),
                               "results": results}, indent=2))
    log_trial(ledger, {"run_id": run_id, "status": "finished",
                       "n_configs": len(results)})
    print(f"\n  -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--runs", default="./runs")
    p.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    p.add_argument("--hours", type=int, default=24,
                   help="hour partitions to replay; 0 = all. Each rate is "
                        "reconstructed twice, so the full archive is 12 "
                        "replays and takes over an hour.")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--tick", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-ends", type=int, default=20_000,
                   help="below this many usable windows the arm reports "
                        "nothing rather than a number built on noise")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
