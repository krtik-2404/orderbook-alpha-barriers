#!/usr/bin/env python3
"""Recover records from LOBForge archive files that gzip refuses to read.

    python3 salvage.py ./data              # report only, changes nothing
    python3 salvage.py ./data --repair     # rewrite damaged files cleanly

Why files break: a process killed without close() leaves a gzip member with no
trailer. If a later run appends a new member to that same file, the decompressor
reaches the end of the broken member, reads the next member's header as deflate
blocks, and fails with "invalid block type" - losing the entire file to the
standard reader even though almost all of the data is intact.

This walks each gzip member independently, keeps whatever decompresses, then
scans forward for the next member header. Loss is bounded by the flush interval
at each break, not the whole file.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
import sys
import zlib
from pathlib import Path

MAGIC = b"\x1f\x8b"
CHUNK = 1 << 12      # coarse pass
FINE = 64            # retry granularity for a damaged member


def salvage_bytes(raw: bytes) -> tuple[bytes, int, int]:
    """Return (decompressed, members_read, members_damaged)."""
    out = bytearray()
    pos, members, damaged = 0, 0, 0

    while pos < len(raw):
        start = raw.find(MAGIC, pos)
        if start < 0:
            break
        members += 1
        consumed = start
        broke = False

        def decode(step: int) -> tuple[bytes, bool, int]:
            """Decode one member from `start` in `step`-sized bites.

            Returns (data, failed, bytes_consumed). Feeding in small bites
            matters: decompress() raises without returning partial output, so
            anything in the same call as the fault is lost. A truncated member
            followed by another member faults at the boundary, and a coarse
            step throws away the whole member with it.
            """
            dd = zlib.decompressobj(31)
            buf = bytearray()
            j = start
            while j < len(raw):
                try:
                    buf += dd.decompress(raw[j:j + step])
                except zlib.error:
                    return bytes(buf), True, j
                j += step
                if dd.eof:
                    break
            if not dd.eof:
                return bytes(buf), True, j
            try:
                buf += dd.flush()
            except zlib.error:
                pass
            end = (len(raw) - len(dd.unused_data)) if dd.unused_data else len(raw)
            return bytes(buf), False, end

        data, failed, endpos = decode(CHUNK)
        if failed:
            # Retry finely to salvage the records lost inside the failing bite.
            fine_data, _, _ = decode(FINE)
            if len(fine_data) > len(data):
                data = fine_data
            damaged += 1
            broke = True
        out += data
        i = endpos
        if False:
            pass
        consumed = (start + 2) if broke else endpos
        pos = max(consumed, start + 2)

    return bytes(out), members, damaged


def check(path: Path) -> tuple[bool, int]:
    """(readable_by_standard_gzip, line_count)."""
    try:
        with gzip.open(path, "rb") as fh:
            return True, sum(1 for _ in fh)
    except Exception:
        return False, 0


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./data")
    repair = "--repair" in sys.argv
    # A .part touched recently is almost certainly OPEN in a running collector.
    # Rewriting it replaces the path while the writer still holds a handle to the
    # old inode: the collector then appends to a deleted file and everything
    # until the next rotation is lost. Never repair a live file.
    LIVE_WINDOW_S = 300
    _now = time.time()
    files, live = [], []
    for _f in sorted(root.rglob("*.jsonl.gz")) + sorted(root.rglob("*.part*")):
        if ".part" in _f.name and (_now - _f.stat().st_mtime) < LIVE_WINDOW_S:
            live.append(_f)
        else:
            files.append(_f)
    if live:
        print(f"\nSKIPPING {len(live)} file(s) currently being written:")
        for _f in live:
            print(f"  {_f.name}")
        print("  An unsealed live file has no gzip trailer yet - that is the")
        print("  writer working, not damage. Repairing it would destroy data.")
    if not files:
        print(f"no archive files under {root}")
        return 1

    bad, recovered_lines, repaired = [], 0, 0
    print(f"\nscanning {len(files)} file(s) under {root}\n" + "=" * 62)

    for p in files:
        ok, n = check(p)
        if ok:
            continue
        raw = p.read_bytes()
        data, members, damaged = salvage_bytes(raw)
        # Records straddling a corruption boundary are truncated or spliced,
        # and they are NOT only at the end of the file - one sits at every
        # member break. Validate each line rather than trusting position.
        lines, dropped = [], 0
        for ln in data.split(b"\n"):
            if not ln.strip():
                continue
            try:
                json.loads(ln)
            except (json.JSONDecodeError, UnicodeDecodeError):
                dropped += 1
                continue
            lines.append(ln)
        bad.append(p)
        recovered_lines += len(lines)
        print(f"\n  DAMAGED {p.relative_to(root)}")
        print(f"    {len(raw)/1024:.0f} KB, {members} gzip member(s), "
              f"{damaged} damaged")
        print(f"    recovered {len(lines)} records"
              + (f", dropped {dropped} unparseable" if dropped else ""))

        if repair and lines:
            backup = p.with_suffix(p.suffix + ".broken")
            shutil.copy2(p, backup)
            tmp = p.with_suffix(p.suffix + ".rebuilt")
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                fh.write(b"\n".join(lines) + b"\n")
            ok2, n2 = check(tmp)
            if ok2:
                tmp.replace(p)
                repaired += 1
                print(f"    repaired -> {n2} records; original kept as "
                      f"{backup.name}")
            else:
                tmp.unlink(missing_ok=True)
                print("    repair FAILED, original left untouched")

    print("\n" + "=" * 62)
    if not bad:
        print("all files read cleanly - nothing to salvage\n")
        return 0
    print(f"{len(bad)} damaged file(s), {recovered_lines} records recoverable")
    if repair:
        print(f"{repaired} file(s) rewritten; .broken backups kept alongside")
    else:
        print("run again with --repair to rewrite them cleanly")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
