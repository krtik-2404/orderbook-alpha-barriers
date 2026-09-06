"""Synthetic Binance USD-M futures market data.

Generates frames that are *protocol-correct* rather than economically realistic.
The collector under test never inspects prices — it inspects framing and the
U/u/pu sequence relationship — so protocol fidelity is what matters here.

Sequence semantics being reproduced (USD-M futures diff depth):
    U  = first update id in this event
    u  = final update id in this event
    pu = final update id of the PREVIOUS event on this stream

A stream is continuous iff ``pu == u_of_previous_event``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field


@dataclass
class BookSimulator:
    """Maintains a plausible book and emits diff events with valid sequence ids."""

    symbol: str = "BTCUSDT"
    mid: float = 118_500.0
    tick: float = 0.10
    levels: int = 20
    seed: int | None = 42

    _update_id: int = field(default=1_000_000, init=False)
    _last_u: int = field(default=0, init=False)
    _agg_id: int = field(default=500_000, init=False)
    _bids: dict[float, float] = field(default_factory=dict, init=False)
    _asks: dict[float, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        for i in range(self.levels):
            self._bids[round(self.mid - (i + 1) * self.tick, 2)] = round(self._rng.uniform(0.1, 8), 3)
            self._asks[round(self.mid + (i + 1) * self.tick, 2)] = round(self._rng.uniform(0.1, 8), 3)
        self._last_u = self._update_id

    # ------------------------------------------------------------------ helpers

    def _drift(self) -> None:
        """Random-walk the mid so the book isn't static."""
        if self._rng.random() < 0.15:
            self.mid = round(self.mid + self._rng.choice([-1, 1]) * self.tick, 2)

    def _mutate(self, side: dict[float, float], sign: int) -> list[list[str]]:
        changes: list[list[str]] = []
        for _ in range(self._rng.randint(1, 6)):
            depth = self._rng.randint(1, self.levels)
            price = round(self.mid + sign * depth * self.tick, 2)
            # qty 0 means "level removed" - the collector must pass this through
            qty = 0.0 if self._rng.random() < 0.12 else round(self._rng.uniform(0.05, 10), 3)
            if qty == 0.0:
                side.pop(price, None)
            else:
                side[price] = qty
            changes.append([f"{price:.2f}", f"{qty:.3f}"])
        return changes

    # ------------------------------------------------------------------- events

    def depth_event(self) -> dict:
        self._drift()
        pu = self._last_u
        first = self._last_u + 1
        last = first + self._rng.randint(0, 9)
        self._last_u = last
        now_ms = int(time.time() * 1000)
        return {
            "e": "depthUpdate",
            "E": now_ms,
            "T": now_ms - 1,
            "s": self.symbol,
            "U": first,
            "u": last,
            "pu": pu,
            "b": self._mutate(self._bids, -1),
            "a": self._mutate(self._asks, +1),
        }

    def agg_trade(self) -> dict:
        self._agg_id += 1
        now_ms = int(time.time() * 1000)
        return {
            "e": "aggTrade",
            "E": now_ms,
            "s": self.symbol,
            "a": self._agg_id,
            "p": f"{self.mid:.2f}",
            "q": f"{self._rng.uniform(0.001, 2):.3f}",
            "f": self._agg_id * 3,
            "l": self._agg_id * 3 + 2,
            "T": now_ms - 2,
            "m": self._rng.random() < 0.5,
        }

    def snapshot(self) -> dict:
        now_ms = int(time.time() * 1000)
        top = lambda d, rev: [  # noqa: E731
            [f"{p:.2f}", f"{q:.3f}"]
            for p, q in sorted(d.items(), key=lambda kv: kv[0], reverse=rev)[: self.levels]
        ]
        return {
            "lastUpdateId": self._last_u,
            "E": now_ms,
            "T": now_ms - 1,
            "bids": top(self._bids, True),
            "asks": top(self._asks, False),
        }


# ---------------------------------------------------------------------- faults


@dataclass
class FaultConfig:
    """Every field is the probability (or trigger) for one injected fault."""

    drop_rate: float = 0.0          # silently discard an event -> sequence gap
    duplicate_rate: float = 0.0     # send the same event twice
    reorder_rate: float = 0.0       # hold an event back and send it late
    malformed_rate: float = 0.0     # emit non-JSON garbage
    stall_after_n: int | None = None      # stop sending, keep socket open
    disconnect_after_n: int | None = None  # close the socket abruptly

    def any_enabled(self) -> bool:
        return any(
            [self.drop_rate, self.duplicate_rate, self.reorder_rate,
             self.malformed_rate, self.stall_after_n, self.disconnect_after_n]
        )


class FaultInjector:
    """Wraps a frame stream and corrupts it in protocol-realistic ways."""

    def __init__(self, cfg: FaultConfig, seed: int | None = 7) -> None:
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._held: str | None = None
        self.sent = 0
        self.dropped = 0
        self.duplicated = 0
        self.reordered = 0
        self.malformed = 0

    def process(self, frame: str) -> list[str] | None:
        """Returns frames to send. ``None`` means: close the connection now."""
        self.sent += 1
        c = self.cfg

        if c.disconnect_after_n and self.sent >= c.disconnect_after_n:
            return None
        if c.stall_after_n and self.sent >= c.stall_after_n:
            return []  # socket stays open, nothing flows - the silent-socket case

        out: list[str] = []

        # A held-back frame arrives late, after its successor.
        if self._held is not None and self._rng.random() < 0.5:
            out.append(self._held)
            self._held = None

        if c.drop_rate and self._rng.random() < c.drop_rate:
            self.dropped += 1
            return out

        if c.reorder_rate and self._held is None and self._rng.random() < c.reorder_rate:
            self._held = frame
            self.reordered += 1
            return out

        out.append(frame)

        if c.duplicate_rate and self._rng.random() < c.duplicate_rate:
            out.append(frame)
            self.duplicated += 1

        if c.malformed_rate and self._rng.random() < c.malformed_rate:
            out.append('{"e":"depthUpdate","U":  <<CORRUPT')
            self.malformed += 1

        return out

    def report(self) -> dict[str, int]:
        return {
            "frames_sent": self.sent,
            "dropped": self.dropped,
            "duplicated": self.duplicated,
            "reordered": self.reordered,
            "malformed": self.malformed,
        }


def envelope(stream: str, data: dict) -> str:
    """Binance combined-stream envelope: {"stream": ..., "data": ...}."""
    return json.dumps({"stream": stream, "data": data}, separators=(",", ":"))
