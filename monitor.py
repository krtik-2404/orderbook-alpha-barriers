#!/usr/bin/env python3
"""LOBForge terminal monitor.

    python3 monitor.py              # watch ./data
    python3 monitor.py ./data 1.0   # custom root, refresh interval

Runs beside the collector in a second terminal. Strictly READ-ONLY: opens no
socket, holds no lock, writes nothing. Kill it any time; the collector does
not notice.

Everything comes from files the collector already writes:

  heartbeat        counters, health flags, missing streams, outages
  .part size       sampled each tick -> live throughput
  trades stream    execution price -> the price chart
  hour partitions  file sizes -> the collection history
  gaps / outages   -> the integrity score
  sealed files     incremental gzip verification -> the corruption index

Throughput is measured from FILE GROWTH, never from the counters. The counters
kept climbing while writes went into a deleted inode after a bad repair; a
monitor trusting them would have shown a healthy screen throughout.

Charts are drawn with box-drawing glyphs (U+2500 range), which render in
practically every console font. Partial-block and braille glyphs, which the
higher-resolution terminal plotting libraries rely on, do not.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import sys
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ESC = "\033["
DIM, BOLD, OFF = f"{ESC}2m", f"{ESC}1m", f"{ESC}0m"
GREY, GREEN, YELLOW, RED, CYAN = (f"{ESC}90m", f"{ESC}32m",
                                  f"{ESC}33m", f"{ESC}31m", f"{ESC}36m")
GUT = 9


# ------------------------------------------------------------------ format

def yfmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def hhmm(ns: float) -> str:
    try:
        return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%H:%M")
    except Exception:
        return "--:--"


def vlen(s: str) -> int:
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] not in "mK":
                i += 1
        else:
            n += 1
        i += 1
    return n


def fit(line: str, cols: int) -> str:
    if vlen(line) <= cols:
        return line
    out, n, i = [], 0, 0
    while i < len(line) and n < cols:
        if line[i] == "\033":
            j = i
            while j < len(line) and line[j] not in "mK":
                j += 1
            out.append(line[i:j + 1]); i = j + 1
        else:
            out.append(line[i]); n += 1; i += 1
    return "".join(out) + OFF


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - vlen(s))


# ------------------------------------------------------------------- chart

def line_chart(vals, width: int, height: int):
    """Connected line chart in box-drawing glyphs.

    Each transition between adjacent columns is rendered as a corner pair with
    a vertical run between, so the series reads as one continuous stroke:

        going up    ..╭──      going down   ──╮
                    ╯                        ╰──
    """
    plot_w = max(12, width - GUT - 2)
    if len(vals) < 2:
        return [" " * GUT + "│" for _ in range(height)], 0.0, 0.0

    step = len(vals) / plot_w
    cols = [vals[min(len(vals) - 1, int(i * step))] for i in range(plot_w)]
    lo, hi = min(cols), max(cols)
    span = (hi - lo) or 1.0
    row = lambda v: max(0, min(height - 1,
                               height - 1 - int((v - lo) / span * (height - 1))))

    grid = [[" "] * plot_w for _ in range(height)]
    r0 = row(cols[0])
    grid[r0][0] = "┼"
    for x in range(1, plot_w):
        rp, rc = row(cols[x - 1]), row(cols[x])
        if rp == rc:
            grid[rc][x] = "─"
        elif rc < rp:                       # rising
            grid[rp][x] = "╯"
            grid[rc][x] = "╭"
            for y in range(rc + 1, rp):
                grid[y][x] = "│"
        else:                               # falling
            grid[rp][x] = "╮"
            grid[rc][x] = "╰"
            for y in range(rp + 1, rc):
                grid[y][x] = "│"

    out = []
    for i, r in enumerate(grid):
        if i % 2 == 0 or i == height - 1:
            v = hi - (hi - lo) * i / max(1, height - 1)
            out.append(yfmt(v).rjust(GUT - 1) + " ┤" + "".join(r))
        else:
            out.append(" " * (GUT - 1) + " │" + "".join(r))
    return out, lo, hi


def xaxis(left: str, mid: str, right: str, width: int):
    plot_w = max(12, width - GUT - 2)
    rule = " " * (GUT - 1) + " └" + "─" * plot_w
    s = left + " " * max(1, plot_w // 2 - len(left) - len(mid) // 2) + mid
    s += " " * max(1, plot_w - len(s) - len(right)) + right
    return rule, " " * (GUT + 1) + s[:plot_w]


def hbar(value: float, peak: float, width: int) -> str:
    if peak <= 0:
        return " " * width
    n = int(round(value / peak * width))
    return "█" * n + DIM + "·" * (width - n) + OFF


def strip(values, width: int) -> str:
    """One-row coverage strip: filled where collecting, dim where idle."""
    vals = list(values)[-width:]
    if not vals:
        return DIM + "·" * width + OFF
    hi = max(vals) or 1
    return "".join("█" if v > hi * .5 else "▒" if v else DIM + "·" + OFF
                   for v in vals)


def tail_lines(path: str, keep: int):
    """Last `keep` lines, tolerating a live .part with no gzip trailer. Raw
    lines are buffered and only survivors parsed - JSON-decoding every line of
    a multi-MB file every few seconds is not free."""
    buf: deque = deque(maxlen=keep)
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                buf.append(line)
    except (OSError, EOFError, zlib.error):
        pass
    return list(buf)


# ----------------------------------------------------------------- monitor

class Monitor:
    def __init__(self, root: Path, interval: float = 1.0) -> None:
        self.root = root
        self.interval = interval
        self.started = time.time()
        self.hist = {"depth": deque(maxlen=40), "trades": deque(maxlen=40)}
        self.last_size: dict = {}
        self.prices: list = []
        self.ptimes: list = []
        self.price_at = 0.0
        self.hours: list = []
        self.volume: list = []
        self.loss: list = []
        self.hours_at = 0.0
        self.verified: dict = {}
        self.scan_at = 0.0
        self.lat = None
        self.lat_at = 0.0
        self.skewed = False
        self.stale_parts = 0

    def files(self, stream):
        return sorted(glob.glob(
            str(self.root / stream / "date=*" / "hour=*" / "*")))

    def open_part(self, stream):
        f = [p for p in self.files(stream) if p.endswith(".part")]
        return Path(max(f, key=os.path.getmtime)) if f else None

    def heartbeat(self):
        try:
            hb = json.loads((self.root / "heartbeat").read_text())
            return hb, time.time() - hb.get("ts", 0)
        except Exception:
            return {}, -1.0

    def sample(self, stream):
        p = self.open_part(stream)
        if p is None:
            self.hist[stream].append(0)
            return 0.0, 0
        try:
            size = p.stat().st_size
        except OSError:
            return 0.0, 0
        prev = self.last_size.get(stream)
        self.last_size[stream] = size
        rate = 0.0 if prev is None else max(0, size - prev) / self.interval
        self.hist[stream].append(rate)
        return rate, size

    def refresh_prices(self):
        """Execution price from aggTrade - no reconstruction needed, every
        trade record carries its price. This is last-trade; true mid, spread
        and the band around a prediction arrive with BookEngine."""
        if time.time() - self.price_at < 5:
            return
        self.price_at = time.time()
        px, ts = [], []
        for f in self.files("trades")[-2:]:
            for line in tail_lines(f, 5000):
                try:
                    r = json.loads(line)
                    px.append(float(r["m"]["data"]["p"]))
                    ts.append(r["t"])
                except Exception:
                    continue
        if px:
            self.prices, self.ptimes = px[-5000:], ts[-5000:]

    def refresh_hours(self):
        """Per-hour volume from FILE SIZES - no decompression, so scanning the
        whole archive costs a handful of stat() calls."""
        if time.time() - self.hours_at < 60:
            return
        self.hours_at = time.time()
        vol: dict = {}
        for f in self.files("depth"):
            h = "/".join(Path(f).parts[-3:-1])
            try:
                vol[h] = vol.get(h, 0) + os.path.getsize(f)
            except OSError:
                pass
        hours = sorted(vol)[-96:]
        counts = dict.fromkeys(hours, 0)
        for s in ("gaps", "outages"):
            for f in self.files(s):
                h = "/".join(Path(f).parts[-3:-1])
                if h in counts:
                    counts[h] += len(tail_lines(f, 5000))
        self.hours, self.volume = hours, [vol[h] for h in hours]
        self.loss = [counts[h] for h in hours]
        now = time.time()
        self.stale_parts = sum(
            1 for s in ("depth", "trades", "snapshots") for f in self.files(s)
            if f.endswith(".part") and now - os.path.getmtime(f) > 3600)

    def scan_corruption(self):
        """Verify sealed files incrementally, a few per cycle, cached by size.
        Sealed files are immutable so one check each suffices; re-reading the
        archive every refresh would not scale past a day of collection."""
        if time.time() - self.scan_at < 20:
            return
        self.scan_at = time.time()
        todo = [f for s in ("depth", "trades", "snapshots") for f in self.files(s)
                if f.endswith(".jsonl.gz")
                and self.verified.get(f, (None,))[0] != os.path.getsize(f)]
        for f in todo[:4]:
            ok = True
            try:
                with gzip.open(f, "rb") as fh:
                    while fh.read(1 << 20):
                        pass
            except Exception:
                ok = False
            try:
                self.verified[f] = (os.path.getsize(f), ok)
            except OSError:
                pass

    def refresh_latency(self):
        if time.time() - self.lat_at < 30:
            return
        self.lat_at = time.time()
        rtts, skew = [], []
        for f in self.files("snapshots")[-3:]:
            for line in tail_lines(f, 400):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "sent" in r:
                    rtts.append((r["t"] - r["sent"]) / 1e6)
                if "E" in r.get("m", {}):
                    skew.append(r["t"] / 1e6 - r["m"]["E"])
        if rtts:
            rtts.sort(); n = len(rtts)
            self.lat = (rtts[n // 2], rtts[min(n - 1, int(n * .9))], rtts[-1])
        if skew:
            skew.sort()
            self.skewed = skew[len(skew) // 2] < 0

    def integrity(self, hb):
        """One 0-100 index for glancing, itemised deductions for acting."""
        bad = sum(1 for _, ok in self.verified.values() if not ok)
        losses = sum(self.loss)
        score = 100
        score -= min(30, losses * 2)
        score -= min(30, bad * 10)
        score -= min(15, self.stale_parts * 5)
        score -= min(15, hb.get("write_failures", 0))
        score -= 20 if hb.get("missing_streams") else 0
        score -= min(10, hb.get("snapshot_failures", 0) * 2)
        return max(0, score), bad, len(self.verified), losses

    # -------------------------------------------------------------- render

    def render(self, cols, rows):
        hb, age = self.heartbeat()
        d_rate, d_size = self.sample("depth")
        t_rate, t_size = self.sample("trades")
        self.refresh_prices(); self.refresh_hours()
        self.refresh_latency(); self.scan_corruption()

        missing = hb.get("missing_streams", [])
        stalled = len(self.hist["depth"]) > 5 and sum(self.hist["depth"]) == 0
        if age < 0 or age > 180:
            status, tone = "NO HEARTBEAT", RED
        elif missing or hb.get("healthy") is False:
            status, tone = "DEGRADED", RED
        elif stalled:
            status, tone = "NOT WRITING", RED
        else:
            status, tone = "LIVE", GREEN

        w = max(56, min(cols - 1, 104))
        L: list = []; A = L.append

        head = f" LOBFORGE MONITOR {DIM}·{OFF} BTCUSDT "
        tail = f" {tone}● {status}{OFF} {DIM}{hms(time.time() - self.started)}{OFF} "
        A(f"{DIM}╭{OFF}{BOLD}{head}{OFF}"
          f"{DIM}{'─' * max(1, w - vlen(head) - vlen(tail) - 2)}{OFF}"
          f"{tail}{DIM}╮{OFF}")

        # ---- price
        ph = max(5, min(11, rows - 26))
        if len(self.prices) > 2:
            last, first = self.prices[-1], self.prices[0]
            pct = (last - first) / first * 100
            ct = GREEN if pct >= 0 else RED
            A(f" {DIM}PRICE{OFF}  {BOLD}{last:,.1f}{OFF} {ct}{pct:+.2f}%{OFF}"
              f"   {DIM}last trade · {len(self.prices):,} ticks{OFF}")
            body, lo, hi = line_chart(self.prices, w, ph)
            for r in body:
                A(" " + ct + r + OFF)
            rule, labels = xaxis(hhmm(self.ptimes[0]),
                                 hhmm(self.ptimes[len(self.ptimes) // 2]),
                                 hhmm(self.ptimes[-1]), w)
            A(" " + DIM + rule + OFF)
            A(" " + DIM + labels + OFF)
        else:
            A(f" {DIM}PRICE   sampling trades...{OFF}")
        A("")

        # ---- collection
        show = min(6, max(3, rows - 30))
        peak = max(self.volume) if self.volume else 0
        A(f" {DIM}COLLECTION{OFF}  {DIM}per hour ·"
          f" {human(sum(self.volume))} over {len(self.hours)}h{OFF}")
        for h, v, ls in list(zip(self.hours, self.volume, self.loss))[-show:]:
            d, hh = h.split("/")
            flag = f"  {RED}{ls} loss{OFF}" if ls else ""
            A(f"  {DIM}{d[-5:]} {hh.split('=')[1]}h{OFF} "
              f"{CYAN}{hbar(v, peak, w - 34)}{OFF} {human(v):>9}{flag}")
        if self.hours:
            A(f"  {DIM}history{OFF}    {CYAN}{strip(self.volume, w - 34)}{OFF}"
              f" {DIM}{len(self.hours)}h span{OFF}")
        A("")

        # ---- three metric columns
        score, bad, total, losses = self.integrity(hb)
        sc = GREEN if score >= 95 else YELLOW if score >= 80 else RED

        def ok(v, good=0):
            return (GREEN if v == good else RED) + str(v) + OFF

        cw = (w - 4) // 3
        rtt = self.lat or (0, 0, 0)
        left = [
            f"{DIM}INTEGRITY{OFF}  {sc}{score}/100{OFF}",
            f"gaps        {ok(hb.get('gaps', 0))}",
            f"outages     {ok(hb.get('outages', 0))}",
            f"reconnects  {ok(hb.get('reconnects', 0))}",
            f"corrupt     {ok(bad)}{DIM}/{total}{OFF}",
            f"stale parts {ok(self.stale_parts)}",
        ]
        mid = [
            f"{DIM}THROUGHPUT{OFF}",
            f"depth   {human(d_rate)}/s",
            f"trades  {human(t_rate)}/s",
            f"frames  {hb.get('depth', 0):,}",
            f"queue   {hb.get('queue', 0)}",
            f"writes  {ok(hb.get('write_failures', 0))} failed",
        ]
        right = [
            f"{DIM}LATENCY{OFF}",
            f"p50   {rtt[0]:.0f} ms",
            f"p90   {rtt[1]:.0f} ms",
            f"max   {rtt[2]:.0f} ms",
            (YELLOW + "clock skewed" + OFF) if self.skewed else "clock ok",
            (RED + "MISSING STREAM" + OFF) if missing else "subs verified",
        ]
        for a, b, c in zip(left, mid, right):
            A(" " + pad(a, cw) + pad(b, cw) + c)

        A("")
        A(f" {DIM}BOOK{OFF}  {GREY}mid · spread · trust % · crossed-book rate"
          f" · book-vs-REST error {DIM}—{OFF} require BookEngine{OFF}")
        A(f"{DIM}╰{'─' * (w - 2)}╯{OFF}")
        A(f" {DIM}read-only · refresh {self.interval}s · counters"
          f" {int(max(0, age))}s old · Ctrl+C quits{OFF}")
        return L

    def run(self):
        # Alternate screen buffer: a surface outside scrollback, so a frame
        # taller than the window overwrites in place instead of scrolling.
        sys.stdout.write(f"{ESC}?1049h{ESC}?25l")
        try:
            while True:
                cols, rows = shutil.get_terminal_size((100, 40))
                lines = self.render(cols, rows)[: rows - 1]
                out = [f"{ESC}H"]
                for line in lines:
                    out.append(fit(line, cols) + f"{ESC}K\n")
                out.append(f"{ESC}J")
                sys.stdout.write("".join(out)); sys.stdout.flush()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write(f"{ESC}?25h{ESC}?1049l"); sys.stdout.flush()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else "./data")
    if not root.exists():
        print(f"no archive at {root} - run from ~/projects/lobforge")
        return 1
    Monitor(root, float(args[1]) if len(args) > 1 else 1.0).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
