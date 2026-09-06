#!/usr/bin/env python3
"""Inspect the raw LOBForge archive.

    python3 explore.py            # run everything
    python3 explore.py seq        # just one section

Sections: event, seq, removals, snapshot, latency, book
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

ROOT = Path("./data")


def files(stream: str) -> list[str]:
    pat = str(ROOT / stream / "date=*" / "hour=*" / "*.jsonl.gz")
    found = sorted(glob.glob(pat))
    if not found:
        found = sorted(glob.glob(pat + ".part"))
    return found


def records(stream: str, limit: int):
    n = 0
    for path in files(stream):
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n += 1
                    if n >= limit:
                        return
        except (OSError, EOFError, Exception):
            continue


def head(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


# ------------------------------------------------------------------ sections


def s_event() -> None:
    head("1. one depth event, in full")
    for rec in records("depth", 1):
        d = rec["m"]["data"]
        print(f"  e   {d['e']}          event type")
        print(f"  E   {d['E']}   exchange event time (ms)")
        print(f"  T   {d['T']}   transaction time (ms)")
        print(f"  U   {d['U']}      first update id in this event")
        print(f"  u   {d['u']}      final update id in this event")
        print(f"  pu  {d['pu']}      final update id of the PREVIOUS event")
        print(f"\n  b   {len(d['b'])} bid level updates, first 3: {d['b'][:3]}")
        print(f"  a   {len(d['a'])} ask level updates, first 3: {d['a'][:3]}")
        print("\n  Each [price, qty] REPLACES that level. qty 0 means remove it.")


def s_seq() -> None:
    head("2. sequence continuity — pu must equal the previous u")
    prev = None
    for i, rec in enumerate(records("depth", 8)):
        d = rec["m"]["data"]
        ok = "" if prev is None else ("  <- continuous" if d["pu"] == prev else "  <- GAP")
        print(f"  U {d['U']}  u {d['u']}  pu {d['pu']}{ok}")
        prev = d["u"]
    print("\n  This one relationship is what SequenceChecker enforces.")
    print("  Break it and every book state after the break is fiction.")


def s_removals() -> None:
    head("3. level removals — qty 0 means DELETE, not 'set to zero'")
    zero = total = 0
    example = None
    for rec in records("depth", 3000):
        d = rec["m"]["data"]
        for side in ("b", "a"):
            for price, qty in d[side]:
                total += 1
                if float(qty) == 0:
                    zero += 1
                    if example is None:
                        example = (side, price, qty)
    pct = (zero / total * 100) if total else 0
    print(f"  {zero} removals out of {total} level updates  ({pct:.1f}%)")
    if example:
        side = "bid" if example[0] == "b" else "ask"
        print(f"  example: {side} {example[1]} qty {example[2]}  ->  drop this price")
    print("\n  Treat qty 0 as a value and your book fills with dead levels.")
    print("  This is the classic reconstruction bug.")


def s_snapshot() -> None:
    head("4. the snapshot — the anchor reconstruction starts from")
    for rec in records("snapshots", 1):
        m = rec["m"]
        print(f"  lastUpdateId  {m['lastUpdateId']}")
        print(f"  bid levels    {len(m['bids'])}")
        print(f"  ask levels    {len(m['asks'])}")
        print(f"\n  top 3 bids  {m['bids'][:3]}")
        print(f"  top 3 asks  {m['asks'][:3]}")
        bb, ba = float(m["bids"][0][0]), float(m["asks"][0][0])
        mid = (bb + ba) / 2
        print(f"\n  best bid {bb}   best ask {ba}")
        print(f"  mid {mid:.2f}   spread {ba - bb:.2f} "
              f"({(ba - bb) / mid * 10000:.2f} bp)")
        print("\n  Replay starts here, then applies every event with u > lastUpdateId.")


def s_latency() -> None:
    head("5. your transport latency — both clocks are in every record")
    deltas = []
    for rec in records("depth", 2000):
        deltas.append(rec["t"] / 1e6 - rec["m"]["data"]["E"])
    if not deltas:
        print("  no records")
        return
    deltas.sort()
    n = len(deltas)
    print(f"  samples  {n}")
    print(f"  p50      {deltas[n // 2]:.0f} ms")
    print(f"  p90      {deltas[int(n * 0.9)]:.0f} ms")
    print(f"  p99      {deltas[int(n * 0.99)]:.0f} ms")
    print("\n  local receive time minus exchange event time.")
    print("  Includes clock skew, so treat it as an upper bound.")
    print("  This is the input to the latency-decay experiment.")


def s_book() -> None:
    head("6. build a book by hand — snapshot plus 3 events")
    snap = next(records("snapshots", 1), None)
    if snap is None:
        print("  no snapshot found")
        return
    m = snap["m"]
    last = m["lastUpdateId"]
    bids = {float(p): float(q) for p, q in m["bids"]}
    asks = {float(p): float(q) for p, q in m["asks"]}

    def top(d: dict, rev: bool, n: int = 3):
        return [(p, d[p]) for p in sorted(d, reverse=rev)[:n]]

    print(f"  start: lastUpdateId {last}")
    print(f"    bids {top(bids, True)}")
    print(f"    asks {top(asks, False)}")

    applied = 0
    for rec in records("depth", 400):
        d = rec["m"]["data"]
        if d["u"] <= last:
            continue  # already reflected in the snapshot
        for price, qty in d["b"]:
            p, q = float(price), float(qty)
            bids.pop(p, None) if q == 0 else bids.update({p: q})
        for price, qty in d["a"]:
            p, q = float(price), float(qty)
            asks.pop(p, None) if q == 0 else asks.update({p: q})
        last = d["u"]
        applied += 1
        print(f"\n  after event u={d['u']} "
              f"({len(d['b'])} bid, {len(d['a'])} ask updates)")
        print(f"    bids {top(bids, True)}")
        print(f"    asks {top(asks, False)}")
        bb, ba = max(bids), min(asks)
        flag = "" if bb < ba else "   <- CROSSED BOOK, invariant violated"
        print(f"    best bid {bb}  best ask {ba}  spread {ba - bb:.2f}{flag}")
        if applied >= 3:
            break

    print("\n  That is the whole reconstruction algorithm. BookEngine is this,")
    print("  plus sequence checking and invariant assertions.")


SECTIONS = {
    "event": s_event, "seq": s_seq, "removals": s_removals,
    "snapshot": s_snapshot, "latency": s_latency, "book": s_book,
}


def main() -> int:
    if not ROOT.exists():
        print("no ./data directory - run this from ~/projects/lobforge")
        return 1
    if not files("depth"):
        print("no depth files found yet - is the collector running?")
        return 1

    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        fn = SECTIONS.get(name)
        if fn is None:
            print(f"unknown section: {name}\navailable: {', '.join(SECTIONS)}")
            return 1
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  section failed: {type(exc).__name__}: {exc}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
