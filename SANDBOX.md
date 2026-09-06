# Sandbox — prove the collector before trusting a live feed

The sandbox impersonates Binance USD-M futures market data locally, so the
collector's failure and recovery behaviour can be tested deterministically,
thousands of times, without waiting for a real network fault.

**The collector needs no code changes.** Point `LOBF_WS_BASE` and
`LOBF_REST_BASE` at the sandbox and the same binary that will later talk to
Binance talks to localhost instead. We test the artifact we ship, not a parallel
one built for testing.

## Run it

    pip install -e ".[dev]"

    # terminal 1 — sandbox
    lobforge-sandbox --rate 20

    # terminal 2 — the real collector, pointed at it
    export LOBF_WS_BASE=ws://127.0.0.1:8765
    export LOBF_REST_BASE=http://127.0.0.1:8765
    export LOBF_DATA_ROOT=./sandbox-data
    lobforge-capture

## Break it on purpose

    lobforge-sandbox --drop 0.10          # 10% of events vanish -> gaps
    lobforge-sandbox --reorder 0.15       # events arrive late
    lobforge-sandbox --duplicate 0.10     # events arrive twice
    lobforge-sandbox --malformed 0.05     # non-JSON garbage
    lobforge-sandbox --disconnect-after 200   # abrupt socket close
    lobforge-sandbox --stall-after 100    # socket stays open, nothing flows

Then check what the collector noticed:

    zcat sandbox-data/gaps/date=*/hour=*/*.gz | head
    cat sandbox-data/heartbeat

## Two frame sources

**Synthetic** (default) — generated on the fly with correct `U`/`u`/`pu`
semantics. Needs no data. Prices are not economically realistic, and that is
fine: the collector never inspects prices, only framing and sequence.

**Replay** — real frames from a previous capture:

    lobforge-sandbox --replay ./data --speed 10

Ten minutes of real capture makes a far better fixture than any simulator,
because it contains edge cases you would not have thought to fake. Speed
multipliers let you compress hours of stream into minutes.

## Automated fault suite

    pytest tests/test_sandbox_integration.py -v

Seven end-to-end tests run the real collector against the sandbox over a real
websocket and assert on what landed on disk:

| Test | Fault | Asserted behaviour |
|---|---|---|
| clean stream | none | zero gaps, snapshot on connect, data persisted |
| dropped events | 10% loss | every gap detected AND persisted with the lost id range |
| reordered events | 15% late | detected, and classified apart from loss |
| malformed frames | 5% garbage | quarantined, never silently dropped; good frames still flow |
| abrupt disconnect | socket close | reconnect + fresh resync snapshot |
| **silent socket** | open, no data | watchdog fires — TCP reports this as healthy |
| clean shutdown | SIGTERM | every file sealed, no `.part` left behind |

## Two bugs this suite already caught

Both would have been invisible against a live feed until they cost data.

1. **Watchdog poll interval was hardcoded to 5s**, independent of
   `stale_timeout_s`. Real detection latency was the timeout *plus* up to five
   seconds, and short timeouts were effectively ignored. Now polls at a quarter
   of the configured timeout.

2. **Watchdog-forced restarts never incremented `reconnects`.** The metric that
   exists to make silent-socket failures visible was blind to exactly that
   failure. A stall storm would have shown a flat counter while data went
   missing.

The second one is the argument for the sandbox in miniature: the code *looked*
correct, the watchdog *did* fire, and the observability that was supposed to
tell you about it reported nothing.
