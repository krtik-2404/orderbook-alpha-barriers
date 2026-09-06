# LOBForge Capture — Architecture

Status: v0.1. Scope is the **capture tier only**. Reconstruction, features, models
and evaluation are separate tiers with their own documents.

---

## 1. The governing decision

**Capture and reconstruction are separate processes, separated by an immutable
on-disk archive.**

Everything else in this document follows from that. The alternative design —
maintaining a live order book inside the collector and persisting reconstructed
snapshots — is the obvious one, and it is wrong here for three reasons:

1. **Bug blast radius.** A reconstruction defect found in week five destroys five
   weeks of data if reconstruction is inline. It destroys nothing if the archive
   is raw: fix the code, replay, done.
2. **Testability.** Raw archives are deterministic replay fixtures. Reconstruction
   becomes a pure function over a fixed input, so it can be unit-tested and
   regression-tested. Inline reconstruction can only be tested against live data,
   which is never the same twice.
3. **Latency budget.** The capture process must drain the socket faster than the
   exchange fills it. Book maintenance in the hot path adds per-message work
   proportional to update size, for no benefit that can't be had offline.

The cost is disk. That cost is roughly ₹300/month. It is the cheapest insurance
in the project.

**Corollary:** the capture process is forbidden from making interpretive
decisions. It never repairs a gap, never drops a "bad" message, never normalises
a field. Detect, record, move on. Judgement happens offline where it can be
reviewed and re-run.

---

## 2. Component topology

```
                         Binance USD-M Futures
              ┌──────────────────────┬──────────────────────┐
              │  REST /fapi/v1/depth │  wss://fstream       │
              └──────────┬───────────┴──────────┬───────────┘
                         │                      │
                 ┌───────▼───────┐      ┌───────▼────────┐
                 │ Snapshot Task │      │  WS Reader     │
                 │ • periodic    │      │  • decode      │
                 │ • on connect  │      │  • stamp t_ns  │
                 │ • on gap      │      │  • route       │
                 └───────┬───────┘      └───┬────────┬───┘
                         │                  │        │
                         │                  │   ┌────▼─────────┐
                         │                  │   │ SequenceCheck│
                         │                  │   │ pu == last_u │
                         │                  │   └────┬─────────┘
                         │                  │        │ Gap
                         └──────────┬───────┴────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  asyncio.Queue (bounded)│   ← backpressure boundary
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  Writer Task           │
                        │  • to_thread(write)    │   ← blocking I/O off the loop
                        │  • per-stream writers  │
                        └───────────┬────────────┘
                                    │
        ┌───────────┬───────────┬───┴───────┬───────────┬────────────┐
        ▼           ▼           ▼           ▼           ▼            ▼
     depth/      trades/    snapshots/    gaps/    malformed/   heartbeat

  Out of band:  Watchdog (idle timer)      Metrics (counters + heartbeat file)
```

**Why a queue between reader and writer.** The socket arrives on the event loop;
gzip compression and disk writes are blocking and bursty. Without the queue, a
slow fsync stalls the socket read, the kernel buffer fills, and Binance drops the
connection. The queue converts a latency spike into a memory spike, which is
survivable and observable.

**Why `asyncio.to_thread` for writes.** `gzip.write` releases the GIL during
compression. Running it on the loop thread would block every other task —
including the watchdog, which would then be unable to detect that anything is
wrong.

---

## 3. Process model

A supervision tree, not a flat set of tasks:

```
run()
├── writer_loop            (unsupervised — owns file handles, must outlive all)
├── _supervise(ws_loop)    (restarted on any exception or watchdog cancel)
├── snapshot_loop          (cancelled on shutdown)
├── watchdog_loop          (cancelled on shutdown)
└── metrics_loop           (cancelled on shutdown)
```

`writer_loop` is deliberately *outside* the supervised set and is the last thing
to stop. Shutdown ordering is:

1. SIGTERM/SIGINT sets the shutdown event.
2. Producer tasks are cancelled and awaited.
3. A `STOP` sentinel is pushed to the queue.
4. The writer drains everything already queued, then seals every open file with
   an atomic rename.

That ordering is what makes shutdown lossless. Reversing steps 2 and 3 would
discard whatever was in flight.

---

## 4. Failure modes

The table below is the actual specification. Each row is a thing that *will*
happen during a three-week run — not a hypothetical.

| Failure | Detection | Response | Data consequence |
|---|---|---|---|
| Transport drop / TLS reset | exception from `async for` | jittered exponential backoff to 60s; `seq.reset()`; snapshot on reconnect | gap logged, replay resyncs from snapshot |
| **Silent socket** (TCP alive, no frames) | watchdog: idle > 20s | cancel WS task; supervisor restarts | as above |
| Binance 24h forced close | connection age > 20h | proactive recycle on our schedule | none — reconnect is clean |
| Sequence gap (`pu != last_u`) | `SequenceChecker` | write to `gaps/`; trigger out-of-band snapshot | window marked untrusted **offline**, not here |
| Duplicate / reordered events | same, `delta < 0` | recorded with distinct `kind` | resolved at replay |
| Disk slower than stream | queue depth > 50% | warn; producer blocks (backpressure) | none, until disk fills |
| Disk full | write raises `OSError` | log ERROR, keep draining | records lost — needs alerting + retention policy |
| SIGKILL / OOM / power loss | none possible | — | ≤ `flush_seconds` of data; `.part` still decompresses to last `Z_SYNC_FLUSH` |
| SIGTERM (deploy, reboot) | signal handler | drain queue, seal files | none |
| REST snapshot failure | httpx exception | log, continue; next periodic attempt | fewer resync anchors |
| Malformed frame | `JSONDecodeError` | quarantine to `malformed/` | none — nothing is discarded |
| NTP step / clock skew | — | watchdog uses `monotonic`; records use `time_ns` | measurable post-hoc against exchange `E`/`T` |

Two entries deserve emphasis.

**Silent socket** is the failure that kills naive collectors. A stalled stream is
indistinguishable from a healthy idle one at the TCP layer; websocket pings can
continue to be answered by a server that has stopped publishing. Only a
data-plane idle timer catches it. Discovering this after the fact means finding
a twelve-hour hole in your dataset.

**Partial-file recovery** is why `GzipFile.flush()` is called on an interval.
Default gzip buffering means a SIGKILL leaves a file that decompresses to
nothing. `Z_SYNC_FLUSH` makes the stream readable up to the last flush, bounding
worst-case loss at `flush_seconds`.

---

## 5. Storage contract

Hive-style partitioning, so the research tier can point Polars/DuckDB at the root
and get partition pruning for free:

```
data/
  depth/      date=2026-08-04/hour=13/depth-20260804T130000Z.jsonl.gz
  trades/     date=2026-08-04/hour=13/trades-20260804T130000Z.jsonl.gz
  snapshots/  date=2026-08-04/hour=13/snapshots-20260804T130000Z.jsonl.gz
  gaps/       date=2026-08-04/hour=13/gaps-20260804T130000Z.jsonl.gz
  malformed/  ...
  heartbeat
```

**Sealing protocol.** Files are written as `*.part` and atomically `os.replace`d
on rotation or clean close. A consumer may treat any file *without* the `.part`
suffix as complete and immutable. This single rule makes the archive safe to read
while collection is still running — no lockfiles, no coordination.

**Record schemas.**

```jsonc
// depth/, trades/  — exchange payload stored verbatim, never re-serialised
{"t": 1754312345678901234, "m": {"stream":"btcusdt@depth@100ms","data":{...}}}

// snapshots/
{"t": <recv_ns>, "sent": <sent_ns>, "reason": "connect|periodic|gap", "m": {...}}

// gaps/
{"t": <recv_ns>, "E": <event_ms>, "kind": "missing", "expected_pu": …, "got_pu": …, "delta": 20}
```

`t` is the **local** receive time in nanoseconds. Keeping it beside the exchange's
own `E`/`T` fields is what lets you measure transport latency and clock drift
later. Collectors that store only exchange timestamps throw that away
permanently.

The `m` field holds the raw frame by string splicing, not `json.dumps(parsed)`.
Re-serialising would silently normalise number formatting and key order, and any
field added by Binance that your parser doesn't know about would be lost.

---

## 6. Capacity

BTCUSDT perpetual, depth@100ms plus aggTrade:

| | rate | raw | gzipped |
|---|---|---|---|
| depth | ~10 msg/s, 2–4 KB each | ~2.5 GB/day | ~400–600 MB/day |
| trades | ~20–60 msg/s, ~200 B | ~0.7 GB/day | ~100–200 MB/day |
| **total** | | **~3 GB/day** | **~0.5–0.9 GB/day** |

Three weeks lands at roughly **15–25 GB compressed**. A 40 GB volume is
comfortable; 80 GB removes the need to think about it at all.

*(Correction to an earlier estimate I gave you: I said 3–6 GB/day compressed.
That was too high by roughly 5×. The number above is the one to budget against.)*

Memory is bounded by the queue: 200k records × ~3 KB ≈ 600 MB worst case at full
backpressure. On a 2 GB VPS that is survivable but tight — drop
`LOBF_QUEUE_MAXSIZE` to 50k if the box is small.

---

## 7. Observability

Three layers, deliberately:

- **Counters** logged every 60s: message rates, gap count, reconnect count, queue
  depth, queue high-water mark. Rates matter more than totals — a depth rate that
  quietly halves means something is wrong upstream.
- **Heartbeat file** written alongside the counters. External supervisors check
  its mtime. An in-process health endpoint would happily report healthy from
  inside a wedged event loop; a file that stops being touched cannot lie.
- **`gaps/` as a first-class dataset.** Not a log — a queryable artifact. At
  writeup time this produces the line that does more for credibility than any
  model result: *"3.1% of the collection window is flagged untrusted and
  excluded, per the gap log."*

---

## 8. Deployment

```
Kali (dev)  ──git push──▶  GitHub  ──git pull──▶  Ubuntu 24.04 VPS
                                                   └─ docker compose up -d
                                                      restart: unless-stopped
                                                      volume: ./data
```

`restart: unless-stopped` plus the clean-shutdown path means a reboot costs at
most `flush_seconds`. Snapshot-on-connect means every restart plants a fresh
resync anchor, so restarts are cheap by construction rather than by luck.

---

## 9. Deliberately out of scope

Listed so the boundary is a decision rather than an omission:

- **Book reconstruction** — offline replay tier.
- **Multiple symbols** — one binary, one symbol, one config. Scale by running
  more containers, not by adding concurrency inside one.
- **Prometheus / Grafana** — the heartbeat file and log counters are sufficient
  for a single-symbol single-box deployment. Adding a metrics stack here is
  infrastructure cosplay.
- **Kafka or any broker** — a bounded in-process queue is the correct
  abstraction for one producer and one consumer on one host.
- **Live dashboards, alerting integrations, auto-scaling.**

Each of these is defensible in a larger system and indefensible here. Being able
to say *why* something was left out is worth more than having built it.
