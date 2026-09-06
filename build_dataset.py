#!/usr/bin/env python3
"""Reconstruct the raw archive into model-ready arrays.

    python3 build_dataset.py                     # ./data -> ./dataset
    python3 build_dataset.py --data ./data --out ./dataset --levels 10

Output:

    dataset/frames.npy      (T, 40) float32   raw book rows, unnormalized
    dataset/mid.npy         (T,)    float64   mid price per frame
    dataset/spread.npy      (T,)    float64   spread per frame
    dataset/event_ms.npy    (T,)    int64     exchange event time
    dataset/intervals.json            contiguous trusted runs [start, end)
    dataset/manifest.json             inputs, counts, code version

Two things this does NOT do, on purpose.

**No labels.** The label threshold alpha is anchored to the median half-spread
of the TRAINING fold, and computing it over the whole dataset would use future
information to label the past. Labels are built per fold by the evaluation
tier. Mid and spread are emitted so it can.

**No normalization.** Same reason: mean and standard deviation must be fitted
on each fold's training portion only. Normalizing here would bake a leak into
the artifact and it would never throw an error.

What it does own is TRUST. Every desync ends the current interval, and only
windows lying entirely inside one interval are ever valid. A window spanning a
gap contains a book state that never existed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from lobforge.book import BookEngine, BookState, Desync  # noqa: E402


def read_jsonl(path: str):
    """Tolerates a live .part with no gzip trailer."""
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (OSError, EOFError, zlib.error):
        return


def partitions(root: Path, stream: str) -> dict:
    out: dict = {}
    base = root / stream
    if not base.exists():
        return out
    for f in sorted(base.rglob("*")):
        # .part files too: a session that ended without sealing still holds
        # real data, and read_jsonl tolerates the missing gzip trailer. Hours
        # 17-18 on 2026-08-13 have their only snapshot in an unsealed file.
        if f.is_file() and ".jsonl.gz" in f.name:
            key = "/".join(f.parts[-3:-1])
            out.setdefault(key, []).append(str(f))
    return out


def code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def build(data: Path, out: Path, levels: int, tick: float, strict: bool,
          keep: int = 500):
    depth_parts = partitions(data, "depth")
    snap_parts = partitions(data, "snapshots")
    hours = sorted(depth_parts)
    if not hours:
        print(f"no depth partitions under {data}")
        return 1

    rows: list = []
    mids: list = []
    spreads: list = []
    times: list = []
    intervals: list = []          # [start, end) into the row arrays
    per_hour: list = []

    run_start = 0
    total_events = total_desync = total_skipped = total_unanchored = 0
    t0 = time.time()

    print(f"{len(hours)} hour partitions\n", flush=True)
    print(f"  {'partition':<26}{'events':>10}{'states':>10}"
          f"{'desync':>8}{'unanch':>9}{'runs':>6}")

    # ONE engine for the whole archive, fed partition by partition in order.
    # Partitions are a file-rotation artifact; the stream runs straight through
    # an hour boundary and the engine should too. A real session break desyncs
    # it on its own and the next snapshot resyncs it.
    engine = BookEngine(tick=tick, levels=levels, strict=strict, keep=keep)
    prev_applied = prev_desync = prev_unanchored = prev_skipped = 0

    for h in hours:
        snaps = [r["m"] for f in snap_parts.get(h, []) for r in read_jsonl(f)
                 if "m" in r and "lastUpdateId" in r.get("m", {})]
        engine.add_snapshots(snaps)
        events = (r["m"]["data"] for f in depth_parts[h] for r in read_jsonl(f)
                  if "m" in r and "data" in r["m"])

        n_before = len(rows)
        runs_here = 0
        n_ev = 0
        # Progress inside a partition: 35k events take ~7s and silence for
        # that long is indistinguishable from a hang.
        print(f"  {h:<26}{'':>10}", end="", flush=True)
        for item in engine.feed(events):
            n_ev += 1
            if n_ev % 5000 == 0:
                print(".", end="", flush=True)
            if isinstance(item, Desync):
                if len(rows) > run_start:
                    intervals.append([run_start, len(rows)])
                    runs_here += 1
                run_start = len(rows)
                continue
            st: BookState = item
            rows.append(st.as_row(levels))
            mids.append(st.mid)
            spreads.append(st.spread)
            times.append(st.event_ms)

        produced = len(rows) - n_before
        # Counters are cumulative on the shared engine; report the delta.
        d_applied = engine.events_applied - prev_applied
        d_desync = engine.desyncs - prev_desync
        d_unanch = engine.events_unanchored - prev_unanchored
        d_skip = engine.events_skipped - prev_skipped
        prev_applied, prev_desync = engine.events_applied, engine.desyncs
        prev_unanchored, prev_skipped = (engine.events_unanchored,
                                         engine.events_skipped)
        per_hour.append({"partition": h, "states": produced,
                         "desyncs": d_desync, "skipped": d_skip,
                         "unanchored": d_unanch})
        print(f"\r  {h:<26}{d_applied:>10,}{produced:>10,}"
              f"{d_desync:>8}{d_unanch:>9,}{runs_here:>6}{'':>12}", flush=True)

    # Close the final run only after every partition: a trusted interval may
    # legitimately span hour boundaries and must not be cut at one.
    if len(rows) > run_start:
        intervals.append([run_start, len(rows)])

    total_events = engine.events_applied
    total_desync = engine.desyncs
    total_skipped = engine.events_skipped
    total_unanchored = engine.events_unanchored

    if not rows:
        print("\nno states reconstructed")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(rows, dtype=np.float32)
    np.save(out / "frames.npy", frames)
    np.save(out / "mid.npy", np.asarray(mids, dtype=np.float64))
    np.save(out / "spread.npy", np.asarray(spreads, dtype=np.float64))
    np.save(out / "event_ms.npy", np.asarray(times, dtype=np.int64))
    (out / "intervals.json").write_text(json.dumps(intervals))

    inputs = {f: os.path.getsize(f)
              for files in depth_parts.values() for f in files}
    manifest = {
        "built_ms": int(time.time() * 1000),
        "code_version": code_version(),
        "schema": "dataset/v1",
        "levels": levels,
        "tick": tick,
        "frames": int(frames.shape[0]),
        "features": int(frames.shape[1]),
        "intervals": len(intervals),
        "longest_run": max(b - a for a, b in intervals),
        "events_applied": total_events,
        "desyncs": total_desync,
        "events_skipped": total_skipped,
        "events_unanchored": total_unanchored,
        "partitions": per_hour,
        "input_bytes": sum(inputs.values()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    spread_arr = np.asarray(spreads)
    mid_arr = np.asarray(mids)
    half_bp = (spread_arr / 2) / mid_arr * 10_000

    print(f"\n{'-' * 62}")
    print(f"  frames         {frames.shape[0]:,} x {frames.shape[1]}"
          f"   ({frames.nbytes / 1e6:.0f} MB float32)")
    print(f"  trusted runs   {len(intervals)}"
          f"   longest {manifest['longest_run']:,} frames")
    print(f"  desyncs        {total_desync}")
    if total_unanchored:
        pct = total_unanchored / (total_events + total_unanchored) * 100
        print(f"  UNANCHORED     {total_unanchored:,} events dropped "
              f"({pct:.1f}%) - no snapshot could place them on a trusted book")
    print(f"  mid range      {mid_arr.min():,.2f} .. {mid_arr.max():,.2f}")
    print(f"  half-spread    p50 {np.median(half_bp):.3f} bp"
          f"   p90 {np.percentile(half_bp, 90):.3f} bp")
    print(f"  elapsed        {time.time() - t0:.1f}s")

    for w, k in ((100, 50), (100, 100)):
        n = sum(max(0, (b - a) - w - k + 1) for a, b in intervals)
        print(f"  windows w={w} k={k:<4} {n:,}")
    print(f"\n  -> {out}/  (frames, mid, spread, event_ms, intervals, manifest)")
    print("  labels and normalization are per-fold and belong to the")
    print("  evaluation tier - deliberately not baked in here.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./dataset")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--tick", type=float, default=0.10)
    p.add_argument("--keep", type=int, default=500,
                   help="book levels retained per side; only --levels are "
                        "emitted, the rest is margin (bounds the per-event sort)")
    p.add_argument("--lenient", action="store_true",
                   help="do not raise on invariant violations, just count them")
    a = p.parse_args()
    return build(Path(a.data), Path(a.out), a.levels, a.tick, not a.lenient, a.keep)


if __name__ == "__main__":
    sys.exit(main())