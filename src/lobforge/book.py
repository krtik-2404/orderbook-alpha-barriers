"""Order book reconstruction from a REST snapshot plus diff-depth events.

A pure function over a fixed input: snapshot in, ordered events in, book states
out. No sockets, no files, no clock. That is what makes it deterministically
testable, and it is why reconstruction lives here rather than in the collector.

Two decisions that are easy to get wrong and expensive to get wrong late:

**Prices are integer ticks, never floats.** 63590.70 is not exactly
representable in binary. A level inserted at one float and deleted at another
that *should* compare equal leaks a phantom level that never goes away, and the
book slowly fills with dead prices. Keying on ``round(price / tick)`` makes
insert and delete agree exactly.

**Quantity zero DELETES a level.** It does not set it to zero. Around 8% of
level updates on BTCUSDT are removals; treating them as values is the classic
reconstruction bug and it produces a book that looks plausible and is wrong.

Trust is explicit. The engine is desynced until a snapshot arrives, and any
sequence discontinuity desyncs it again. A desynced engine emits nothing:
a book state that cannot be justified from the data is worse than no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

BTCUSDT_TICK = 0.10


class InvariantError(Exception):
    """The reconstructed book violated something that must always hold."""


@dataclass(frozen=True)
class BookState:
    """Top-N view of the book after one event. Prices are floats here because
    this is the boundary where the data leaves the engine; internally they are
    integer ticks."""

    event_ms: int          # exchange event time, field E
    update_id: int         # final update id of the applied event, field u
    bids: tuple            # ((price, qty), ...) descending
    asks: tuple            # ((price, qty), ...) ascending

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bp(self) -> float:
        return self.spread / self.mid * 10_000

    def as_row(self, levels: int = 10) -> list:
        """Flat [ask_p1, ask_v1, bid_p1, bid_v1, ...] - the DeepLOB layout.

        40 numbers for 10 levels. Short sides are padded with the last known
        price and zero quantity so every row has the same width; a ragged
        feature matrix is not usable and silently dropping the row would bias
        the sample toward liquid moments."""
        row = []
        for i in range(levels):
            ap, aq = self.asks[i] if i < len(self.asks) else (self.asks[-1][0], 0.0)
            bp, bq = self.bids[i] if i < len(self.bids) else (self.bids[-1][0], 0.0)
            row += [ap, aq, bp, bq]
        return row


@dataclass
class Desync:
    """Why the engine stopped trusting itself. Recorded, never repaired here."""

    reason: str
    expected: int | None = None
    got: int | None = None
    update_id: int | None = None

    def as_dict(self) -> dict:
        return {"reason": self.reason, "expected": self.expected,
                "got": self.got, "u": self.update_id}


@dataclass
class BookEngine:
    tick: float = BTCUSDT_TICK
    levels: int = 10
    strict: bool = True
    # Depth diffs carry far-out levels (BTCUSDT sends updates at 1000.00 while
    # trading near 63000), and nothing ever removes them, so an unpruned book
    # grows without bound and top() - which sorts it on EVERY event - turns the
    # replay quadratic. We only ever emit `levels` deep; keeping `keep` is a
    # large margin. This makes the output a top-K book by construction, which
    # is what the feature tier consumes anyway.
    keep: int = 500
    # Setting this False makes the engine behave like a pipeline that never
    # checks pu: it splices straight across a sequence discontinuity instead of
    # desyncing, keeping levels the missing events would have moved or removed.
    # That is WRONG, deliberately, and exists so gapstudy.py can measure what
    # the wrongness costs. Nothing in the real pipeline may set it.
    trust_sequence: bool = True

    _bids: dict = field(default_factory=dict, init=False)   # ticks -> qty
    _asks: dict = field(default_factory=dict, init=False)
    _last_id: int | None = field(default=None, init=False)
    _first_after_snapshot: bool = field(default=False, init=False)

    events_applied: int = field(default=0, init=False)
    events_skipped: int = field(default=0, init=False)
    desyncs: int = field(default=0, init=False)
    events_unanchored: int = field(default=0, init=False)
    _pending: list = field(default_factory=list, init=False)

    # ------------------------------------------------------------- helpers

    def _to_ticks(self, price) -> int:
        # round-half-up, not banker's rounding. round() sends .5 to the nearest
        # EVEN integer, so two adjacent half-tick prices can collapse onto the
        # same key and silently merge two levels into one. Real Binance prices
        # sit on the tick grid, but a latent tie-break bug in the one function
        # every level passes through is not worth carrying.
        return int(float(price) / self.tick + 0.5)

    def _to_price(self, ticks: int) -> float:
        return round(ticks * self.tick, 8)

    @property
    def synced(self) -> bool:
        return self._last_id is not None

    @property
    def last_update_id(self) -> int | None:
        return self._last_id

    def desync(self, reason: str, **kw) -> Desync:
        self._last_id = None
        self._first_after_snapshot = False
        self.desyncs += 1
        return Desync(reason, **kw)

    # ------------------------------------------------------------ snapshot

    def apply_snapshot(self, snap: dict) -> None:
        """Reset to the exchange's own view. This is the only way to become
        synced, and the only way to recover from a discontinuity."""
        self._bids = {self._to_ticks(p): float(q)
                      for p, q in snap["bids"] if float(q) > 0}
        self._asks = {self._to_ticks(p): float(q)
                      for p, q in snap["asks"] if float(q) > 0}
        self._last_id = int(snap["lastUpdateId"])
        self._first_after_snapshot = True

    # --------------------------------------------------------------- event

    def apply_event(self, ev: dict) -> BookState | Desync | None:
        """Apply one depthUpdate.

        Returns a BookState when the book is trustworthy, a Desync when the
        stream broke, or None when the event is stale or the engine is not
        synced. Never guesses.
        """
        if not self.synced:
            return None

        U, u, pu = int(ev["U"]), int(ev["u"]), int(ev["pu"])

        # Everything at or before the snapshot is already reflected in it.
        if u <= self._last_id:
            self.events_skipped += 1
            return None

        if self._first_after_snapshot:
            # Binance: the first processed event must straddle lastUpdateId.
            # If it starts after it, the events in between were never seen and
            # the snapshot cannot anchor this stream.
            if self.trust_sequence and not (U <= self._last_id + 1):
                return self.desync("snapshot_stale", expected=self._last_id,
                                   got=U, update_id=u)
            self._first_after_snapshot = False
        elif self.trust_sequence and pu != self._last_id:
            # The one relationship that proves continuity. Anything else means
            # events were lost, duplicated or reordered.
            return self.desync("sequence_break", expected=self._last_id,
                               got=pu, update_id=u)

        self._apply_levels(ev)

        self._last_id = u
        self.events_applied += 1
        if len(self._bids) > 2 * self.keep:
            self._bids = dict(sorted(self._bids.items(),
                                     reverse=True)[: self.keep])
        if len(self._asks) > 2 * self.keep:
            self._asks = dict(sorted(self._asks.items())[: self.keep])
        return self.state(int(ev.get("E", 0)), u)

    def _apply_levels(self, ev: dict) -> None:
        """Fold one event's level deltas into the book. No sequence checking,
        no state emitted - that is apply_event's job."""
        for price, qty in ev.get("b", ()):
            t, q = self._to_ticks(price), float(qty)
            if q == 0:
                self._bids.pop(t, None)      # removal, not a zero value
            else:
                self._bids[t] = q
        for price, qty in ev.get("a", ()):
            t, q = self._to_ticks(price), float(qty)
            if q == 0:
                self._asks.pop(t, None)
            else:
                self._asks[t] = q

    def resync_over(self, missed) -> None:
        """Re-anchor after a gap by folding in the events the gap swallowed.

        The live collector answers a gap by refetching a REST snapshot
        (capture.py, reason="gap"), so a real pipeline is back on a correct
        book within a second; it does not sit blind until the next periodic
        snapshot half an hour later. An archive has no snapshot at that
        instant, but it does still contain the events that went missing, and
        replaying them reproduces exactly the book that refetch would have
        returned.

        This is the offline stand-in for that refetch and exists for
        gapstudy.py, which has to model the honest pipeline's real behaviour
        rather than a strawman that goes blind on the first dropped packet.
        Nothing in the live path calls it - live, the events are gone.
        """
        if not missed:
            return
        for ev in missed:
            self._apply_levels(ev)
        self._last_id = int(missed[-1]["u"])
        self._first_after_snapshot = False

    # --------------------------------------------------------------- state

    def top(self):
        bids = sorted(self._bids.items(), reverse=True)[: self.levels]
        asks = sorted(self._asks.items())[: self.levels]
        return ([(self._to_price(t), q) for t, q in bids],
                [(self._to_price(t), q) for t, q in asks])

    def state(self, event_ms: int, update_id: int) -> BookState:
        bids, asks = self.top()
        problems = self.check(bids, asks)
        if problems and self.strict:
            raise InvariantError(f"u={update_id}: " + "; ".join(problems))
        return BookState(event_ms, update_id, tuple(bids), tuple(asks))

    def check(self, bids, asks) -> list:
        """Things that must always hold. A book that violates one of these is
        not a book, and emitting it poisons every downstream tier."""
        bad = []
        if not bids or not asks:
            bad.append("empty side")
            return bad
        if bids[0][0] >= asks[0][0]:
            bad.append(f"crossed: bid {bids[0][0]} >= ask {asks[0][0]}")
        if any(bids[i][0] <= bids[i + 1][0] for i in range(len(bids) - 1)):
            bad.append("bids not strictly descending")
        if any(asks[i][0] >= asks[i + 1][0] for i in range(len(asks) - 1)):
            bad.append("asks not strictly ascending")
        if any(q <= 0 for _, q in bids + asks):
            bad.append("non-positive quantity")
        return bad

    # -------------------------------------------------------------- replay

    def add_snapshots(self, snapshots: list) -> None:
        """Add anchors to the pool, newest last. Safe to call repeatedly."""
        for sn in snapshots:
            self._pending.append(sn)
        self._pending.sort(key=lambda s: int(s["lastUpdateId"]))

    def feed(self, events) -> Iterator:
        """Drive the engine over an ordered event stream.

        Stateful across calls on purpose. Hour partitions are an artifact of
        file rotation, not of the stream: the book at the end of one hour IS
        the book at the start of the next, and pu-chaining proves it. Resetting
        per partition would strand every hour that happens to contain no
        snapshot - and snapshots arrive every 30 minutes, so half of them do
        not. A genuine break between sessions desyncs the engine anyway.
        """
        for ev in events:
            if not self.synced:
                U, u = int(ev["U"]), int(ev["u"])
                # An anchor is usable only if it straddles this event:
                # U <= lastUpdateId + 1 <= u. Older anchors have been overtaken;
                # newer ones belong to events still ahead.
                cand = None
                for j, sn in enumerate(self._pending):
                    last = int(sn["lastUpdateId"])
                    if U <= last + 1 <= u:
                        cand = j
                        break
                    if last >= u:
                        break
                if cand is None:
                    self.events_unanchored += 1
                    continue
                self.apply_snapshot(self._pending[cand])
                del self._pending[: cand + 1]
            out = self.apply_event(ev)
            if out is not None:
                yield out

    def replay(self, snapshots: list, events) -> Iterator:
        """One-shot convenience wrapper around add_snapshots + feed."""
        self.add_snapshots(snapshots)
        yield from self.feed(events)