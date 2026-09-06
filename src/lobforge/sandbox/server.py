"""A local server that impersonates Binance USD-M futures market data.

Serves both endpoints the collector uses:

    GET  /fapi/v1/depth?symbol=...&limit=...   REST order book snapshot
    WS   /stream?streams=a/b                   combined diff-depth + aggTrade

The collector needs **no code changes** to talk to it - point ``LOBF_WS_BASE``
and ``LOBF_REST_BASE`` at this server and the same binary that will later talk
to Binance talks to the sandbox instead. That is the whole point: we test the
artifact we ship, not a parallel one.

Two frame sources:
  * synthetic  - generated on the fly (default; no data needed)
  * replay     - real captured frames from a LOBForge archive (--replay PATH)
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import json
import logging
import time
from pathlib import Path

from aiohttp import web, WSMsgType

from .simulator import BookSimulator, FaultConfig, FaultInjector, envelope

log = logging.getLogger("lobforge.sandbox")


class SandboxServer:
    def __init__(
        self,
        symbol: str = "btcusdt",
        rate_hz: float = 10.0,
        speed: float = 1.0,
        faults: FaultConfig | None = None,
        replay_path: str | None = None,
        silent_endpoints: set[str] | None = None,
    ) -> None:
        self.symbol = symbol.lower()
        self.rate_hz = rate_hz
        self.speed = speed
        self.faults = faults or FaultConfig()
        self.replay_path = replay_path
        # Endpoints that accept the connection and then send nothing - exactly
        # how Binance behaves for a channel served from a different routed path.
        self.silent_endpoints = silent_endpoints or set()
        self.sim = BookSimulator(symbol=symbol.upper())
        self.injector = FaultInjector(self.faults)
        self.snapshot_requests = 0

    # ------------------------------------------------------------------ sources

    def _synthetic_frames(self, want: str):
        """Protocol-correct frames for one routed endpoint.

        Binance serves diff-depth on /public and aggTrade on /market, so a
        connection only ever carries one of the two.
        """
        depth_stream = f"{self.symbol}@depth@100ms"
        trade_stream = f"{self.symbol}@aggTrade"
        while True:
            if want == "public":
                yield envelope(depth_stream, self.sim.depth_event())
            else:
                # roughly 3 trades per depth event, matching live proportions
                for _ in range(3):
                    yield envelope(trade_stream, self.sim.agg_trade())

    def _replay_frames(self):
        """Replay real captured frames from a LOBForge archive directory."""
        pattern = str(Path(self.replay_path) / "**" / "*.jsonl.gz")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise SystemExit(f"no *.jsonl.gz found under {self.replay_path}")
        log.info("replaying %d archive files", len(files))
        for path in files:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    m = rec.get("m")
                    if m is not None:
                        yield json.dumps(m, separators=(",", ":"))

    def frames(self, want: str):
        if self.replay_path:
            return self._filtered_replay(want)
        return self._synthetic_frames(want)

    def _filtered_replay(self, want: str):
        for frame in self._replay_frames():
            is_trade = '"aggTrade"' in frame
            if (want == "market") == is_trade:
                yield frame

    # ----------------------------------------------------------------- handlers

    async def depth_snapshot(self, request: web.Request) -> web.Response:
        self.snapshot_requests += 1
        log.info("REST snapshot #%d", self.snapshot_requests)
        return web.json_response(self.sim.snapshot())

    async def stream(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)

        # Which routed endpoint was this? /public carries depth, /market carries
        # aggTrade. An unrouted path gets /public only - exactly as Binance now
        # behaves, so the collector's own migration bug would reproduce here.
        want = "market" if request.path.startswith("/market") else "public"
        log.info("[%s] client connected: streams=%s", want, request.query.get("streams"))

        if want in self.silent_endpoints:
            log.warning("[%s] endpoint is SILENT - connection open, no frames", want)
            async for _ in ws:
                pass
            return ws

        interval = 1.0 / (self.rate_hz * self.speed)
        source = self.frames(want)
        injector = FaultInjector(self.faults)

        async def drain_client() -> None:
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break

        reader = asyncio.create_task(drain_client())
        try:
            for frame in source:
                out = injector.process(frame)
                if out is None:
                    log.warning("[%s] injecting abrupt disconnect", want)
                    await ws.close(code=1006)
                    break
                for f in out:
                    if ws.closed:
                        break
                    await ws.send_str(f)
                if ws.closed:
                    break
                await asyncio.sleep(interval)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            reader.cancel()
            log.info("[%s] client gone: %s", want, injector.report())
        return ws

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/fapi/v1/depth", self.depth_snapshot)
        # Routed endpoints, matching Binance's post-2026-04 topology
        app.router.add_get("/public/stream", self.stream)
        app.router.add_get("/market/stream", self.stream)
        app.router.add_get("/public/ws/{stream:.*}", self.stream)
        app.router.add_get("/market/ws/{stream:.*}", self.stream)
        # Unrouted legacy paths: serve /public only, as Binance does
        app.router.add_get("/stream", self.stream)
        app.router.add_get("/ws/{stream:.*}", self.stream)
        return app


def main() -> None:
    p = argparse.ArgumentParser(description="Binance-shaped market data sandbox")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--rate", type=float, default=10.0, help="depth events/sec at 1x")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    p.add_argument("--replay", help="path to a LOBForge archive to replay instead of synthesising")
    p.add_argument("--drop", type=float, default=0.0, help="fraction of events dropped")
    p.add_argument("--duplicate", type=float, default=0.0)
    p.add_argument("--reorder", type=float, default=0.0)
    p.add_argument("--malformed", type=float, default=0.0)
    p.add_argument("--stall-after", type=int, help="stop sending after N frames, keep socket open")
    p.add_argument("--disconnect-after", type=int, help="close socket after N frames")
    p.add_argument("--silent-market", action="store_true",
                   help="accept /market connections but send nothing (the 2026 Binance routing trap)")
    p.add_argument("--silent-public", action="store_true",
                   help="accept /public connections but send nothing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(name)s %(message)s")
    # The per-request access log drowns out the lines that matter here.
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    logging.Formatter.converter = time.gmtime

    faults = FaultConfig(
        drop_rate=args.drop,
        duplicate_rate=args.duplicate,
        reorder_rate=args.reorder,
        malformed_rate=args.malformed,
        stall_after_n=args.stall_after,
        disconnect_after_n=args.disconnect_after,
    )
    silent = set()
    if args.silent_market:
        silent.add("market")
    if args.silent_public:
        silent.add("public")
    srv = SandboxServer(args.symbol, args.rate, args.speed, faults, args.replay, silent)

    print(f"\n  sandbox listening on http://127.0.0.1:{args.port}")
    print("  point the collector at it with:\n")
    print(f"    export LOBF_WS_BASE=ws://127.0.0.1:{args.port}")
    print(f"    export LOBF_REST_BASE=http://127.0.0.1:{args.port}")
    print("    lobforge-capture\n")
    if faults.any_enabled():
        print(f"  faults active: {faults}\n")

    web.run_app(srv.app(), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
