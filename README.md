# LOBForge

Limit order book capture and microstructure research on Binance USD-M futures.

**This repo currently contains the capture tier only.** See
[ARCHITECTURE.md](ARCHITECTURE.md) for the design and its rationale.

## What it does

Archives the `@depth@100ms` diff stream and `@aggTrade` stream verbatim to
gzipped JSONL, with periodic REST order book snapshots and a first-class log of
every sequence discontinuity. It does **not** reconstruct the order book — that
happens offline, replaying this archive.

No API key, no account, no authentication: these are public market data streams.

## Run

    cp .env.example .env
    docker compose up -d
    docker compose logs -f

## Develop

    pip install -e ".[dev]"
    pytest -q
    ruff check .

## Layout

    src/lobforge/
      config.py     env-driven configuration
      capture.py    supervision tree: ws reader, snapshot poller, writer, watchdog
      writer.py     rotating gzip JSONL, atomic sealing, crash-recoverable flush
      seqcheck.py   diff-stream continuity checking
