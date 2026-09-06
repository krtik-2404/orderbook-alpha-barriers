"""LOBForge raw capture daemon.

Captures the Binance USD-M futures diff-depth and aggTrade streams verbatim to
gzipped JSONL, together with periodic REST order book snapshots.

It deliberately does NOT reconstruct the order book. Book reconstruction happens
offline, replaying this archive. Rationale: a reconstruction bug discovered in
week five destroys five weeks of work if reconstruction happened inline, and
destroys nothing if it happens on replay.

Every record carries a local receive timestamp in nanoseconds (``t``) alongside
the exchange's own timestamps, so clock skew and transport latency stay
measurable after the fact.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import signal
from collections import Counter
import time
from pathlib import Path

import httpx
import websockets

from .config import Config
from .seqcheck import SequenceChecker
from .writer import RotatingJsonlWriter

log = logging.getLogger("lobforge.capture")

STOP = object()  # queue sentinel


class Metrics:
    def __init__(self) -> None:
        self.depth_msgs = 0
        self.trade_msgs = 0
        self.snapshots = 0
        self.snapshot_failures = 0
        self.snapshot_fail_streak = 0
        self.snapshots_throttled = 0
        self.write_failures = 0
        self.write_fail_streak = 0
        self.gaps = 0
        self.reconnects = 0
        self.queue_high_water = 0
        # Per-endpoint liveness: /public and /market are separate connections and
        # either can stall independently while the other looks healthy.
        self.last_msg: dict[str, float] = {}
        # Frames seen per stream NAME (not per endpoint): the unit that can go
        # silently missing when an exchange re-routes a channel.
        self.streams_seen: Counter = Counter()
        self.missing_streams: set[str] = set()
        # Endpoints whose websocket handshake has succeeded at least once.
        # Distinguishes "never reached the server" from "server sent nothing".
        self.endpoints_connected: set[str] = set()
        # Endpoints with a LIVE socket right now. The watchdog exists to catch a
        # connected-but-silent socket; applying it while a handshake is still in
        # flight cancels the very attempt that would restore service.
        self.live: set[str] = set()
        self.outages = 0
        self.outage_seconds = 0.0
        self.subscription_verified = False


class Capture:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_maxsize)
        self.metrics = Metrics()
        self.seq = SequenceChecker()
        self._shutdown = asyncio.Event()
        # asyncio holds only a WEAK reference to tasks. A fire-and-forget task
        # whose reference we drop can be garbage-collected mid-execution - which
        # aborts an in-flight HTTP request and looks like a server-side error.
        self._bg: set[asyncio.Task] = set()
        self._http: httpx.AsyncClient | None = None
        self._last_gap_snapshot = 0.0
        self._outage_start: dict[str, int] = {}

    # ------------------------------------------------------------------ enqueue

    async def _put(self, name: str, payload: bytes) -> None:
        """Enqueue for the writer. Blocks rather than drops: silent data loss is
        worse than backpressure, and backpressure is visible in the metrics."""
        qsize = self.queue.qsize()
        if qsize > self.metrics.queue_high_water:
            self.metrics.queue_high_water = qsize
        if qsize > self.cfg.queue_maxsize * 0.5:
            log.warning("writer queue at %d/%d - disk may be the bottleneck",
                        qsize, self.cfg.queue_maxsize)
        await self.queue.put((name, payload))

    # ------------------------------------------------------------------ streams

    async def ws_loop(self, endpoint: str, url: str) -> None:
        backoff = 1.0
        self.metrics.last_msg[endpoint] = time.monotonic()
        while not self._shutdown.is_set():
            try:
                log.info("[%s] connecting %s", endpoint, url)
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=None,
                ) as ws:
                    backoff = 1.0
                    self.metrics.endpoints_connected.add(endpoint)
                    self.metrics.live.add(endpoint)
                    self.metrics.last_msg[endpoint] = time.monotonic()
                    started = self._outage_start.pop(endpoint, None)
                    if started is not None:
                        now_ns = time.time_ns()
                        secs = (now_ns - started) / 1e9
                        self.metrics.outages += 1
                        self.metrics.outage_seconds += secs
                        log.warning("[%s] outage ended after %.1fs", endpoint, secs)
                        # An outage leaves NO gap record: nothing was lost
                        # mid-stream, the socket was simply absent. Without an
                        # explicit record the archive cannot tell the replay
                        # tier that this window is discontinuous.
                        await self._put("outages", json.dumps({
                            "t": now_ns, "endpoint": endpoint,
                            "start_ns": started, "end_ns": now_ns,
                            "duration_s": round(secs, 3),
                        }).encode())
                    if endpoint == "public":
                        self.seq.reset()
                        await self.fetch_snapshot(reason="connect")
                    connected_at = time.monotonic()

                    async for raw in ws:
                        recv_ns = time.time_ns()
                        self.metrics.last_msg[endpoint] = time.monotonic()
                        await self._handle(raw, recv_ns)

                        # Recycle before Binance's 24h forced close, on our terms.
                        if time.monotonic() - connected_at > self.cfg.connection_max_age_s:
                            log.info("[%s] recycling connection at max age", endpoint)
                            break

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must never die
                log.warning("[%s] websocket error: %s: %s", endpoint, type(exc).__name__, exc)

            self.metrics.live.discard(endpoint)
            self._outage_start.setdefault(endpoint, time.time_ns())
            if self._shutdown.is_set():
                break
            self.metrics.reconnects += 1
            sleep = min(backoff, self.cfg.reconnect_max_backoff_s)
            sleep *= 0.5 + random.random()  # jitter: avoid lockstep reconnects
            log.info("[%s] reconnecting in %.1fs", endpoint, sleep)
            await asyncio.sleep(sleep)
            backoff = min(backoff * 2, self.cfg.reconnect_max_backoff_s)

    async def _handle(self, raw: str | bytes, recv_ns: int) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        try:
            env = json.loads(raw)
        except json.JSONDecodeError:
            log.error("undecodable frame, archiving to malformed")
            await self._put("malformed", json.dumps({"t": recv_ns, "raw": raw}).encode())
            return

        stream = env.get("stream", "")
        data = env.get("data", {})
        if stream:
            self.metrics.streams_seen[stream] += 1

        # Store the envelope verbatim - no re-serialisation, no field loss.
        line = b'{"t":%d,"m":%s}' % (recv_ns, raw.encode("utf-8"))

        if stream.endswith("aggTrade"):
            self.metrics.trade_msgs += 1
            await self._put("trades", line)
            return

        # diff-depth
        self.metrics.depth_msgs += 1
        await self._put("depth", line)

        gap = self.seq.check(int(data["U"]), int(data["u"]), int(data["pu"]))
        if gap is not None:
            self.metrics.gaps += 1
            log.warning("sequence gap: %s", gap.as_dict())
            rec = {"t": recv_ns, "E": data.get("E"), **gap.as_dict()}
            await self._put("gaps", json.dumps(rec).encode())
            # A gap invalidates the local book. Resnapshot so replay can resync.
            now_mono = time.monotonic()
            if now_mono - self._last_gap_snapshot >= self.cfg.gap_snapshot_cooldown_s:
                self._last_gap_snapshot = now_mono
                self._spawn(self.fetch_snapshot(reason="gap"))
            else:
                # Skipping is safe: the gap is already in the ledger, and the next
                # snapshot re-anchors the whole burst. Losing the archive to an IP
                # ban would not be safe.
                self.metrics.snapshots_throttled += 1

    def _spawn(self, coro) -> None:
        """Fire-and-forget, but keep a reference until the task completes."""
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ---------------------------------------------------------------- snapshots

    async def fetch_snapshot(self, reason: str) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)
        try:
            sent_ns = time.time_ns()
            resp = await self._http.get(self.cfg.snapshot_url)
            recv_ns = time.time_ns()
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self.metrics.snapshot_failures += 1
            self.metrics.snapshot_fail_streak += 1
            streak = self.metrics.snapshot_fail_streak
            # Snapshots are the ONLY resync anchors. Without them the archive
            # becomes unreconstructable after the first gap - but depth frames
            # keep flowing, so every other health signal still looks green.
            # Escalate loudly rather than let a dead subsystem look healthy.
            if streak in (1, 3) or streak % 10 == 0:
                log.error(
                    "snapshot fetch failed (%s), %d consecutive: %s | "
                    "NO RESYNC ANCHORS ARE BEING WRITTEN - check LOBF_REST_BASE",
                    reason, streak, exc,
                )
            return

        self.metrics.snapshots += 1
        self.metrics.snapshot_fail_streak = 0
        line = b'{"t":%d,"sent":%d,"reason":"%s","m":%s}' % (
            recv_ns, sent_ns, reason.encode(), resp.content,
        )
        await self._put("snapshots", line)
        log.info("snapshot ok (%s) rtt=%.0fms", reason, (recv_ns - sent_ns) / 1e6)

    async def snapshot_loop(self) -> None:
        while not self._shutdown.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.cfg.snapshot_interval_s
                )
            if self._shutdown.is_set():
                return
            await self.fetch_snapshot(reason="periodic")

    # ------------------------------------------------------------------- writer

    async def writer_loop(self) -> None:
        writers: dict[str, RotatingJsonlWriter] = {}

        def writer_for(name: str) -> RotatingJsonlWriter:
            if name not in writers:
                # Low-rate streams flush on every record. The flush interval
                # is only consulted DURING a write, so a stream that writes
                # once every 30 minutes keeps its record in the gzip buffer
                # until the next one - and an unclean stop in between loses it,
                # leaving a file that exists and is empty. Snapshots are the
                # only anchors reconstruction has; losing one strands an entire
                # session. These streams are tiny, so flushing always is free.
                rare = name in ("snapshots", "gaps", "outages", "malformed")
                writers[name] = RotatingJsonlWriter(
                    root=self.cfg.data_root,
                    name=name,
                    rotate_seconds=self.cfg.rotate_seconds,
                    flush_seconds=0 if rare else self.cfg.flush_seconds,
                )
            return writers[name]

        try:
            while True:
                item = await self.queue.get()
                if item is STOP:
                    break
                name, payload = item
                try:
                    await asyncio.to_thread(writer_for(name).write, payload)
                    self.metrics.write_fail_streak = 0
                except Exception as exc:  # noqa: BLE001
                    self.metrics.write_failures += 1
                    streak = self.metrics.write_fail_streak = (
                        self.metrics.write_fail_streak + 1)
                    # Rate-limit: a broken mount fails on EVERY record, and
                    # thousands of identical lines bury the one that matters.
                    if streak <= 3 or streak % 250 == 0:
                        log.error("write failed for %s (%d consecutive): %s",
                                  name, streak, exc)
                    if streak >= self.cfg.max_write_failures:
                        log.critical(
                            "STORAGE UNWRITABLE - %d consecutive write failures: %s. "
                            "Nothing is being archived. Common cause: the bind-mounted "
                            "data directory is not writable by the container user "
                            "(uid 10001) - try: sudo chown -R 10001:10001 ./data",
                            streak, exc,
                        )
                        self._shutdown.set()
                        break
        finally:
            for w in writers.values():
                with contextlib.suppress(Exception):
                    w.close()
            log.info("writers closed")

    # ------------------------------------------------------- subscription check

    async def subscription_check_loop(self) -> None:
        """Verify every subscribed stream actually delivers.

        Binance moved aggTrade from the unrouted endpoint to /market in 2026-03.
        Unrouted connections kept receiving depth and silently received no
        trades - no error, no close, no gap. Depth-only archives looked perfectly
        healthy. This check is the only defence against the next such change.
        """
        deadline = self.cfg.subscription_timeout_s
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._shutdown.wait(), timeout=deadline)
        if self._shutdown.is_set():
            return

        expected = self.cfg.expected_streams
        seen = set(self.metrics.streams_seen)
        missing = expected - seen
        self.metrics.missing_streams = missing
        self.metrics.subscription_verified = not missing

        if not missing:
            log.info(
                "subscription verified: all %d streams delivering (%s)",
                len(expected),
                ", ".join(f"{s}={self.metrics.streams_seen[s]}" for s in sorted(seen)),
            )
            return

        # Same symptom, two very different causes. Saying "check the routing"
        # when the socket never opened sends you hunting in the wrong place.
        stream_endpoint = {
            self.cfg.depth_stream: "public",
            self.cfg.trade_stream: "market",
        }
        never_connected, connected_but_silent = [], []
        for stream in sorted(missing):
            ep = stream_endpoint.get(stream, "?")
            (connected_but_silent if ep in self.metrics.endpoints_connected
             else never_connected).append(f"{stream} [{ep}]")

        if never_connected:
            log.critical(
                "NO CONNECTION - %s never completed a websocket handshake after "
                "%ds (%d reconnect attempts). This is a transport problem, not a "
                "routing one: check LOBF_WS_BASE, DNS and network reachability.",
                ", ".join(never_connected), deadline, self.metrics.reconnects,
            )
        if connected_but_silent:
            log.critical(
                "SUBSCRIPTION INCOMPLETE - %s connected but delivered zero frames "
                "in %ds. The archive will be missing this data entirely and "
                "NOTHING else will flag it: no gap, no error, no stall - the data "
                "was never lost, it was never sent. Check endpoint routing: "
                "Binance serves different channels from /public vs /market.",
                ", ".join(connected_but_silent), deadline,
            )
        if self.cfg.fail_on_missing_stream:
            log.critical("LOBF_FAIL_ON_MISSING_STREAM set - shutting down")
            self._shutdown.set()

    # ---------------------------------------------------------------- watchdog

    async def watchdog_loop(self) -> None:
        """Binance can hold a socket open while sending nothing. A silent socket
        is indistinguishable from a healthy one at the TCP layer, so we time it."""
        # Poll well inside the timeout, or detection latency becomes
        # stale_timeout_s + poll_interval instead of stale_timeout_s.
        poll = max(0.25, self.cfg.stale_timeout_s / 4)
        while not self._shutdown.is_set():
            await asyncio.sleep(poll)
            now = time.monotonic()
            # Only endpoints with a live socket. A disconnected endpoint is
            # already being retried with backoff; cancelling it there just
            # aborts the in-flight handshake and prolongs the outage.
            for endpoint in list(self.metrics.live):
                last = self.metrics.last_msg.get(endpoint)
                if last is None:
                    continue
                idle = now - last
                if idle <= self.cfg.stale_timeout_s:
                    continue
                log.error("[%s] no messages for %.0fs - forcing reconnect",
                          endpoint, idle)
                self.metrics.last_msg[endpoint] = now
                for task in asyncio.all_tasks():
                    if task.get_name() == f"ws-{endpoint}":
                        task.cancel()

    async def metrics_loop(self) -> None:
        prev_depth = prev_trades = 0
        while not self._shutdown.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.cfg.metrics_interval_s
                )
            if self._shutdown.is_set():
                return
            m = self.metrics
            log.info(
                "depth=%d (+%d) trades=%d (+%d) snaps=%d/%dfail/%dthr gaps=%d "
                "reconnects=%d out=%d/%.0fs queue=%d hwm=%d%s",
                m.depth_msgs, m.depth_msgs - prev_depth,
                m.trade_msgs, m.trade_msgs - prev_trades,
                m.snapshots, m.snapshot_failures, m.snapshots_throttled,
                m.gaps, m.reconnects, m.outages, m.outage_seconds,
                self.queue.qsize(), m.queue_high_water,
                ("  <<< SNAPSHOTS FAILING" if m.snapshot_fail_streak >= 3 else "")
                + ("  <<< MISSING STREAMS: " + ",".join(sorted(m.missing_streams))
                   if m.missing_streams else ""),
            )
            prev_depth, prev_trades = m.depth_msgs, m.trade_msgs
            # External liveness signal. A supervisor checks this file's mtime;
            # an in-process check would report healthy from inside a wedged loop.
            try:
                hb = Path(self.cfg.data_root) / "heartbeat"
                hb.write_text(json.dumps({
                    "ts": time.time(),
                    "depth": m.depth_msgs, "trades": m.trade_msgs,
                    "gaps": m.gaps, "reconnects": m.reconnects,
                    "snapshots": m.snapshots,
                    "snapshot_failures": m.snapshot_failures,
                    "snapshot_fail_streak": m.snapshot_fail_streak,
                    "snapshots_throttled": m.snapshots_throttled,
                    "write_failures": m.write_failures,
                    "outages": m.outages,
                    "outage_seconds": round(m.outage_seconds, 1),
                    "live_endpoints": sorted(m.live),
                    "streams": dict(m.streams_seen),
                    "missing_streams": sorted(m.missing_streams),
                    # External supervisors gate on this single field.
                    "healthy": (not m.missing_streams
                                and m.snapshot_fail_streak < 3
                                and m.write_fail_streak == 0),
                    "queue": self.queue.qsize(),
                }))
            except OSError as exc:
                log.error("heartbeat write failed: %s", exc)

    # -------------------------------------------------------------------- main

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown.set)
            except NotImplementedError:
                # Windows proactor loop has no add_signal_handler. Fall back to
                # the sync handler; it is less precise but still reaches the
                # drain-and-seal path instead of killing the process outright.
                signal.signal(sig, lambda *_: self._shutdown.set())

        writer = asyncio.create_task(self.writer_loop(), name="writer")
        supervised = [
            asyncio.create_task(
                self._supervise(
                    lambda: self.ws_loop("public", self.cfg.public_ws_url), "ws-public"
                ),
                name="ws-public-sup",
            ),
            asyncio.create_task(
                self._supervise(
                    lambda: self.ws_loop("market", self.cfg.market_ws_url), "ws-market"
                ),
                name="ws-market-sup",
            ),
            asyncio.create_task(self.snapshot_loop(), name="snapshot"),
            asyncio.create_task(self.watchdog_loop(), name="watchdog"),
            asyncio.create_task(self.metrics_loop(), name="metrics"),
            asyncio.create_task(self.subscription_check_loop(), name="subcheck"),
        ]

        await self._shutdown.wait()
        log.info("shutdown requested, draining")

        for t in supervised:
            t.cancel()
        await asyncio.gather(*supervised, return_exceptions=True)

        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()

        await self.queue.put(STOP)
        await writer  # drains remaining records and seals files
        log.info("clean shutdown: depth=%d trades=%d gaps=%d",
                 self.metrics.depth_msgs, self.metrics.trade_msgs, self.metrics.gaps)

    async def _supervise(self, coro_fn, name: str) -> None:
        """Restart a task if it dies for any reason short of shutdown."""
        while not self._shutdown.is_set():
            task = asyncio.create_task(coro_fn(), name=name)
            try:
                await task
                return
            except asyncio.CancelledError:
                if self._shutdown.is_set():
                    raise
                log.warning("%s cancelled by watchdog, restarting", name)
                self.metrics.reconnects += 1
            except Exception as exc:  # noqa: BLE001
                log.exception("%s died: %s - restarting in 5s", name, exc)
                await asyncio.sleep(5)


def main() -> None:
    cfg = Config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )
    logging.Formatter.converter = time.gmtime
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    log.info("LOBForge capture starting")
    log.info("  public (depth)  : %s", cfg.public_ws_url)
    log.info("  market (trades) : %s", cfg.market_ws_url)
    asyncio.run(Capture(cfg).run())


if __name__ == "__main__":
    main()