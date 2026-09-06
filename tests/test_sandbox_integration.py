"""End-to-end sandbox tests.

These run the REAL collector (``lobforge.capture.Capture``) against the sandbox
server over a real websocket, with faults injected, and assert on what actually
landed on disk. Nothing is mocked: the code path under test is the code path
that will later talk to Binance.

This is the "prove it before you trust it" layer.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

import pytest
from aiohttp import web

from lobforge.capture import Capture
from lobforge.config import Config
from lobforge.sandbox.simulator import FaultConfig
from lobforge.sandbox.server import SandboxServer


async def _serve(srv: SandboxServer) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(srv.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def _cfg(tmp_path: Path, port: int, **over) -> Config:
    base = dict(
        symbol="btcusdt",
        ws_base=f"ws://127.0.0.1:{port}",
        rest_base=f"http://127.0.0.1:{port}",
        data_root=tmp_path,
        rotate_seconds=3600,
        flush_seconds=0,
        stale_timeout_s=2,
        snapshot_interval_s=3600,
        metrics_interval_s=3600,
        reconnect_max_backoff_s=1,
    )
    base.update(over)
    return Config(**base)


def _read(root: Path, name: str) -> list[dict]:
    out = []
    for p in sorted((root / name).rglob("*")):
        if p.is_file():
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except (json.JSONDecodeError, EOFError):
                        break
    return out


async def _run_collector(cfg: Config, seconds: float) -> Capture:
    cap = Capture(cfg)
    task = asyncio.create_task(cap.run())
    await asyncio.sleep(seconds)
    cap._shutdown.set()
    await asyncio.wait_for(task, timeout=10)
    return cap


# --------------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_clean_stream_produces_no_gaps(tmp_path):
    srv = SandboxServer(rate_hz=50)
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port), 2.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.depth_msgs > 20, "collector received depth events"
    assert cap.metrics.trade_msgs > 20, "collector received trades"
    assert cap.metrics.gaps == 0, "a clean stream must produce zero gaps"
    assert cap.metrics.snapshots >= 1, "snapshot fetched on connect"
    assert len(_read(tmp_path, "depth")) > 20
    assert _read(tmp_path, "gaps") == []


@pytest.mark.asyncio
async def test_dropped_events_are_detected(tmp_path):
    srv = SandboxServer(rate_hz=50, faults=FaultConfig(drop_rate=0.10))
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port), 2.5)
    finally:
        await runner.cleanup()

    gaps = _read(tmp_path, "gaps")
    assert cap.metrics.gaps > 0, "dropped events must surface as gaps"
    assert len(gaps) == cap.metrics.gaps, "every gap is persisted, not just counted"
    assert all(g["kind"] == "missing" for g in gaps)
    assert all(g["delta"] > 0 for g in gaps), "missing events mean pu ran ahead"
    # the ledger must identify WHICH ids were lost, not merely that loss occurred
    assert all("expected_pu" in g and "got_pu" in g for g in gaps)


@pytest.mark.asyncio
async def test_reordered_events_are_detected_and_distinguished(tmp_path):
    srv = SandboxServer(rate_hz=50, faults=FaultConfig(reorder_rate=0.15))
    runner, port = await _serve(srv)
    try:
        await _run_collector(_cfg(tmp_path, port), 2.5)
    finally:
        await runner.cleanup()

    gaps = _read(tmp_path, "gaps")
    assert gaps, "reordering must be detected"
    kinds = {g["kind"] for g in gaps}
    assert "duplicate_or_reorder" in kinds, "late arrival is not the same fault as loss"


@pytest.mark.asyncio
async def test_malformed_frames_are_quarantined_not_dropped(tmp_path):
    srv = SandboxServer(rate_hz=50, faults=FaultConfig(malformed_rate=0.15))
    runner, port = await _serve(srv)
    try:
        await _run_collector(_cfg(tmp_path, port), 2.0)
    finally:
        await runner.cleanup()

    bad = _read(tmp_path, "malformed")
    assert bad, "undecodable frames must be preserved, never silently discarded"
    assert all("raw" in r for r in bad), "the original bytes are kept for inspection"
    assert _read(tmp_path, "depth"), "good frames still flow alongside bad ones"


@pytest.mark.asyncio
async def test_abrupt_disconnect_triggers_reconnect_and_resnapshot(tmp_path):
    srv = SandboxServer(rate_hz=80, faults=FaultConfig(disconnect_after_n=40))
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port), 4.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.reconnects >= 1, "collector must reconnect after a drop"
    assert cap.metrics.snapshots >= 2, "each reconnect plants a fresh resync anchor"


@pytest.mark.asyncio
async def test_silent_socket_is_detected_by_watchdog(tmp_path):
    """The failure that kills naive collectors: socket open, nothing flowing."""
    srv = SandboxServer(rate_hz=80, faults=FaultConfig(stall_after_n=30))
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port, stale_timeout_s=1), 5.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.reconnects >= 1, (
        "a stalled-but-open socket must be caught by the idle timer, "
        "since TCP reports it as perfectly healthy"
    )


@pytest.mark.asyncio
async def test_shutdown_seals_every_file(tmp_path):
    srv = SandboxServer(rate_hz=50)
    runner, port = await _serve(srv)
    try:
        await _run_collector(_cfg(tmp_path, port), 1.5)
    finally:
        await runner.cleanup()

    assert list(tmp_path.rglob("*.jsonl.gz")), "sealed files exist"
    assert not list(tmp_path.rglob("*.part")), "clean shutdown leaves no .part files"


@pytest.mark.asyncio
async def test_dead_snapshot_endpoint_is_surfaced_not_hidden(tmp_path):
    """Regression: snapshots can fail 100% while every other signal looks green.

    Depth frames keep flowing, the queue stays empty, the heartbeat stays fresh.
    Without an explicit failure counter the collector reports itself healthy
    while writing an archive that can never be reconstructed past the first gap.
    """
    srv = SandboxServer(rate_hz=50, faults=FaultConfig(drop_rate=0.10))
    runner, port = await _serve(srv)
    try:
        # point REST at a closed port - the websocket still works fine
        # cooldown off: this test is about failure *reporting*, not trigger rate
        cfg = _cfg(tmp_path, port, rest_base="http://127.0.0.1:1",
                   gap_snapshot_cooldown_s=0)
        cap = await _run_collector(cfg, 2.5)
    finally:
        await runner.cleanup()

    assert cap.metrics.depth_msgs > 20, "data still flows - that is the trap"
    assert cap.metrics.snapshots == 0, "no snapshot could succeed"
    assert cap.metrics.snapshot_failures > 0, "failures must be counted"
    assert cap.metrics.snapshot_fail_streak >= 3, "streak must be tracked"

    hb = json.loads((tmp_path / "heartbeat").read_text()) if (tmp_path / "heartbeat").exists() else None
    if hb is not None:
        assert hb["snapshot_failures"] > 0, "heartbeat must expose the failure"


@pytest.mark.asyncio
async def test_depth_and_trades_arrive_on_separate_routed_endpoints(tmp_path):
    """Regression: Binance split futures streams across routed base URLs.

    Diff depth is a /public stream; aggTrade is a /market stream. A connection
    to an unrouted path receives /public data ONLY - silently, with no error and
    no rejection. The old single-combined-stream collector therefore recorded
    depth forever while trades stayed at exactly zero, and nothing in the gap
    ledger could catch it, because nothing was lost - it was never subscribed.
    """
    srv = SandboxServer(rate_hz=60)
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port), 2.5)
    finally:
        await runner.cleanup()

    assert cap.metrics.depth_msgs > 10, "depth must arrive on /public"
    assert cap.metrics.trade_msgs > 10, "trades must arrive on /market"
    assert _read(tmp_path, "depth"), "depth persisted"
    assert _read(tmp_path, "trades"), "trades persisted"

    # both endpoints must be independently liveness-tracked
    assert set(cap.metrics.last_msg) == {"public", "market"}


@pytest.mark.asyncio
async def test_silent_stream_is_caught_by_subscription_check(tmp_path):
    """The failure that has no other signature.

    /market accepts the connection and sends nothing. Depth flows normally, so:
    no gap (nothing was lost), no error (nothing failed), no stall on the public
    endpoint, snapshots fine, heartbeat fresh, queue empty. Every existing signal
    reads green while a third of the dataset is absent. Only an explicit
    'did each subscribed stream actually deliver?' deadline catches it.
    """
    srv = SandboxServer(rate_hz=60, silent_endpoints={"market"})
    runner, port = await _serve(srv)
    try:
        cfg = _cfg(tmp_path, port, subscription_timeout_s=2, stale_timeout_s=60)
        cap = await _run_collector(cfg, 4.0)
    finally:
        await runner.cleanup()

    # the trap: everything else looks fine
    assert cap.metrics.depth_msgs > 10, "depth flows normally"
    assert cap.metrics.trade_msgs == 0, "trades never arrive"
    assert cap.metrics.gaps == 0, "no gap - nothing was lost, it was never sent"
    assert cap.metrics.snapshot_failures == 0, "snapshots are healthy"

    # the check that actually catches it
    assert cap.metrics.missing_streams == {"btcusdt@aggTrade"}
    assert cap.metrics.subscription_verified is False

    hb_path = tmp_path / "heartbeat"
    if hb_path.exists():
        hb = json.loads(hb_path.read_text())
        assert hb["healthy"] is False
        assert "btcusdt@aggTrade" in hb["missing_streams"]


@pytest.mark.asyncio
async def test_healthy_subscription_verifies_cleanly(tmp_path):
    srv = SandboxServer(rate_hz=60)
    runner, port = await _serve(srv)
    try:
        cfg = _cfg(tmp_path, port, subscription_timeout_s=2)
        cap = await _run_collector(cfg, 3.5)
    finally:
        await runner.cleanup()

    assert cap.metrics.subscription_verified is True
    assert cap.metrics.missing_streams == set()
    assert set(cap.metrics.streams_seen) == {"btcusdt@depth@100ms", "btcusdt@aggTrade"}


@pytest.mark.asyncio
async def test_unreachable_endpoint_is_diagnosed_as_transport_not_routing(tmp_path):
    """Same symptom, different cause: zero frames because nothing ever connected.

    A missing-stream alarm that says 'check the routing' when the socket never
    opened sends you hunting in the wrong place. The check must distinguish
    endpoints that handshook successfully from ones that never did.
    """
    cfg = _cfg(tmp_path, 1, subscription_timeout_s=2, stale_timeout_s=60)
    cap = await _run_collector(cfg, 3.5)

    assert cap.metrics.depth_msgs == 0
    assert cap.metrics.endpoints_connected == set(), "nothing ever connected"
    assert cap.metrics.missing_streams == {"btcusdt@depth@100ms", "btcusdt@aggTrade"}
    assert cap.metrics.reconnects > 0, "it was retrying, not idle"


@pytest.mark.asyncio
async def test_gap_snapshot_cooldown_caps_rest_call_rate(tmp_path):
    """Gaps cluster during volatility; unthrottled they burst REST calls into
    Binance's weight limit, and sustained 429s escalate to an IP ban. Skipping a
    resync anchor is recoverable; losing exchange access is not."""
    srv = SandboxServer(rate_hz=80, faults=FaultConfig(drop_rate=0.30))
    runner, port = await _serve(srv)
    try:
        cfg = _cfg(tmp_path, port, gap_snapshot_cooldown_s=1.0)
        cap = await _run_collector(cfg, 3.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.gaps > 5, "plenty of gaps to trigger on"
    assert cap.metrics.snapshots_throttled > 0, "bursts must be throttled"
    # ~3s at 1 per second, plus the connect snapshot
    assert cap.metrics.snapshots <= 6, f"too many REST calls: {cap.metrics.snapshots}"
    assert _read(tmp_path, "gaps"), "every gap is still recorded in the ledger"


@pytest.mark.asyncio
async def test_unwritable_storage_fails_fast_instead_of_spinning(tmp_path):
    """A collector that cannot write is not collecting.

    Without a cap, a bad bind-mount produces thousands of identical error lines
    per minute, an empty archive, and a process that still looks alive to
    anything watching the pid. Fail loudly and stop.
    """
    # A regular file where a directory must go: mkdir fails with ENOTDIR for
    # every user, root included, so this reproduces the broken-mount case
    # regardless of who runs the tests.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x")
    srv = SandboxServer(rate_hz=80)
    runner, port = await _serve(srv)
    try:
        cfg = _cfg(tmp_path, port, data_root=blocked, max_write_failures=10,
                   subscription_timeout_s=60)
        cap = await _run_collector(cfg, 6.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.write_failures >= 10, "failures counted"
    assert cap._shutdown.is_set(), "must stop, not spin forever"


@pytest.mark.asyncio
async def test_watchdog_does_not_cancel_inflight_reconnects(tmp_path):
    """Regression: during a real outage the watchdog fought its own recovery.

    Observed in production: the socket dropped, each reconnect took ~10s to fail
    its handshake, and the watchdog cancelled the in-flight attempt at 20s of
    idleness - repeatedly. 118 reconnects and six minutes of frozen counters.
    The watchdog must only police endpoints that HAVE a live socket.
    """
    cfg = _cfg(tmp_path, 1, stale_timeout_s=1, subscription_timeout_s=60)
    cap = await _run_collector(cfg, 4.0)

    assert cap.metrics.live == set(), "nothing ever connected"
    # backoff-driven retries only; no watchdog cancellations piled on top
    assert cap.metrics.reconnects < 12, (
        f"{cap.metrics.reconnects} reconnects in 4s - the watchdog is "
        "cancelling handshakes that are still in flight"
    )


@pytest.mark.asyncio
async def test_outage_is_recorded_as_data_not_just_a_log_line(tmp_path):
    """An outage produces NO gap record: nothing was lost mid-stream, the
    socket was simply absent. Without an explicit outage record the archive
    cannot tell the replay tier that the window is discontinuous."""
    srv = SandboxServer(rate_hz=60, faults=FaultConfig(disconnect_after_n=30))
    runner, port = await _serve(srv)
    try:
        cap = await _run_collector(_cfg(tmp_path, port), 5.0)
    finally:
        await runner.cleanup()

    assert cap.metrics.outages > 0, "outage counted"
    recs = _read(tmp_path, "outages")
    assert recs, "outage persisted to the archive, not only to the log"
    r = recs[0]
    assert r["endpoint"] in ("public", "market")
    assert r["duration_s"] >= 0 and r["end_ns"] > r["start_ns"]
