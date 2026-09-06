#!/usr/bin/env python3
"""Analyse a LOBForge archive after a soak run and print a verdict.

    python soak-report.py ./data

Answers the questions a 24-hour run exists to answer, and flags the ones that
only show up over long periods: rotation, the UTC date boundary, memory and
queue pressure, gap clustering by session, and disk growth.
"""

from __future__ import annotations

import gzip
import json
import zlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

OK, WARN, BAD = "  ok  ", " WARN ", " FAIL "


UNREADABLE: set = set()


def read_jsonl(paths):
    """Yield records, tolerating damaged files.

    A file killed mid-write and then appended to raises zlib.error partway
    through. Reporting must degrade to a warning, never crash - the whole point
    of the report is to tell you something is wrong.
    """
    for p in paths:
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (OSError, EOFError, zlib.error):
            # A .part still being appended to has no trailer yet. That is the
            # writer working, not damage - do not cry wolf about it.
            try:
                fresh = (time.time() - Path(p).stat().st_mtime) < 300
            except OSError:
                fresh = False
            if not (str(p).endswith(".part") and fresh):
                UNREADABLE.add(str(p))
            continue


def hour_of(path: Path) -> str:
    parts = {k: v for k, v in (s.split("=", 1) for s in path.parts if "=" in s)}
    return f"{parts.get('date', '?')} {parts.get('hour', '??')}h"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./data")
    if not root.exists():
        print(f"no archive at {root}")
        return 1

    print(f"\nLOBForge soak report - {root}\n" + "=" * 62)
    checks: list[tuple[str, str]] = []

    # ---------------------------------------------------------- file integrity
    sealed = sorted(root.rglob("*.jsonl.gz"))
    partial = sorted(root.rglob("*.part"))
    # Files seal on the hourly boundary, so a sub-hour run has data only in
    # .part files. Count both, or every short run looks like total data loss.
    all_files = sealed + partial
    hours = sorted({hour_of(p) for p in sealed + partial})

    print(f"\nsealed files      : {len(sealed)}")
    print(f"unsealed (.part)  : {len(partial)}")
    print(f"hour partitions   : {len(hours)}")
    if hours:
        print(f"first / last      : {hours[0]}  ->  {hours[-1]}")

    # Rotation is per-file, not per-hour-directory: a run can rotate several
    # times inside one hour partition (and does, under a short rotate interval).
    depth_files = [p for p in sealed if p.relative_to(root).parts[0] == "depth"]
    short_run = len(sealed) == 0 and len(partial) > 0
    if short_run:
        print("\n  (run shorter than one hour: nothing has sealed yet, "
              "so rotation and the date boundary cannot be assessed)")
    checks.append((OK if len(depth_files) > 1 else WARN,
                   f"rotation exercised ({len(depth_files)} sealed depth files)"))
    dates = {h.split()[0] for h in hours}
    checks.append((OK if len(dates) > 1 else WARN,
                   f"UTC date boundary crossed ({len(dates)} date(s))"))
    stale = [p for p in partial
             if (time.time() - p.stat().st_mtime) > 3600]
    checks.append((OK if not stale else WARN,
                   f"{len(stale)} stale .part file(s) from ended sessions"
                   + (f" - oldest {stale[0].name}" if stale else "")))

    # ---------------------------------------------------------- volume by hour
    per_hour: dict[str, Counter] = defaultdict(Counter)
    bytes_by_stream: Counter = Counter()
    for p in all_files:
        stream = p.relative_to(root).parts[0]
        per_hour[hour_of(p)][stream] += sum(1 for _ in read_jsonl([p]))
        bytes_by_stream[stream] += p.stat().st_size

    if per_hour:
        print("\nper-hour volume")
        print(f"  {'hour':<18}{'depth':>9}{'trades':>9}{'gaps':>7}{'snaps':>7}")
        for h in sorted(per_hour):
            c = per_hour[h]
            print(f"  {h:<18}{c['depth']:>9}{c['trades']:>9}"
                  f"{c['gaps']:>7}{c['snapshots']:>7}")

    total_depth = sum(c["depth"] for c in per_hour.values())
    total_trades = sum(c["trades"] for c in per_hour.values())
    total_gaps = sum(c["gaps"] for c in per_hour.values())

    checks.append((OK if total_trades > 0 else BAD,
                   f"trades present ({total_trades}) - a zero here is the routing bug"))
    checks.append((OK if total_depth > 0 else BAD, f"depth present ({total_depth})"))

    # depth should be ~36000/hour at 100ms; flag hours far below
    # ~36000 depth events/hour at 100ms; only meaningful for full hours
    full_hours = sorted(per_hour)[1:-1] if len(per_hour) > 2 else []
    thin = [h for h in full_hours if 0 < per_hour[h]["depth"] < 20_000]
    checks.append((OK if not thin else WARN,
                   f"{len(thin)} hour(s) with unexpectedly thin depth"
                   + (f" ({', '.join(sorted(thin)[:3])})" if thin else "")))

    # ------------------------------------------------------------------- gaps
    gap_files = [p for p in all_files if p.relative_to(root).parts[0] == "gaps"]
    deltas, kinds = [], Counter()
    for rec in read_jsonl(gap_files):
        deltas.append(abs(int(rec.get("delta", 0))))
        kinds[rec.get("kind", "?")] += 1

    print(f"\ngaps              : {total_gaps}")
    if deltas:
        deltas.sort()
        lost = sum(d for d in deltas if d > 0)
        print(f"  kinds           : {dict(kinds)}")
        print(f"  events lost     : ~{lost}")
        print(f"  median / max    : {deltas[len(deltas)//2]} / {deltas[-1]}")
        if total_depth:
            print(f"  loss rate       : {lost / (total_depth + lost) * 100:.4f}%")
        worst = max(per_hour.items(), key=lambda kv: kv[1]["gaps"])
        print(f"  worst hour      : {worst[0]} ({worst[1]['gaps']} gaps)")

    rate = (total_gaps / len(hours)) if hours else 0
    checks.append((OK if rate < 5 else (WARN if rate < 40 else BAD),
                   f"gap rate {rate:.1f}/hour"))

    # -------------------------------------------------------------- heartbeat
    hb_path = root / "heartbeat"
    if hb_path.exists():
        hb = json.loads(hb_path.read_text())
        print("\nfinal heartbeat")
        for k in ("healthy", "reconnects", "snapshots", "snapshot_failures",
                  "snapshots_throttled", "queue", "missing_streams"):
            if k in hb:
                print(f"  {k:<20}{hb[k]}")
        checks.append((OK if hb.get("healthy", True) else BAD, "collector self-reported healthy"))
        checks.append((OK if not hb.get("missing_streams") else BAD, "no missing streams"))
        rc = hb.get("reconnects", 0)
        checks.append((OK if rc < 20 else WARN, f"{rc} reconnects over the run"))
        checks.append((OK if hb.get("snapshot_failures", 0) == 0 else WARN,
                       f"{hb.get('snapshot_failures', 0)} snapshot failures"))

    # ------------------------------------------------------------------- disk
    total_mb = sum(bytes_by_stream.values()) / 1e6
    print(f"\ndisk              : {total_mb:.1f} MB total")
    for s, b in bytes_by_stream.most_common():
        print(f"  {s:<18}{b/1e6:>8.1f} MB")
    if hours:
        per_day = total_mb / len(hours) * 24
        print(f"  projected        : {per_day:.0f} MB/day, {per_day*21/1000:.1f} GB for 3 weeks")

    if UNREADABLE:
        print(f"\nDAMAGED FILES    : {len(UNREADABLE)}")
        for u in sorted(UNREADABLE)[:5]:
            print(f"  {Path(u).name}")
        print("  recover with:  python3 salvage.py ./data --repair")
    checks.append((OK if not UNREADABLE else WARN,
                   f"{len(UNREADABLE)} unreadable file(s)"))

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 62)
    for status, text in checks:
        print(f"[{status}] {text}")
    bad = sum(1 for s, _ in checks if s == BAD)
    warn = sum(1 for s, _ in checks if s == WARN)
    print("=" * 62)
    print(f"\n{len(checks)-bad-warn} passed, {warn} warnings, {bad} failures")
    print("ready for the three-week run\n" if bad == 0
          else "resolve the failures before committing three weeks\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
