"""Sequence continuity checking for the USD-M futures diff-depth stream.

Each ``depthUpdate`` event carries:
    U  -- first update id in this event
    u  -- final update id in this event
    pu -- final update id of the *previous* event on this stream

The stream is continuous iff ``pu == u_of_previous_event``. Anything else means
we lost, duplicated, or reordered events, and the affected window cannot be used
to reconstruct a trustworthy book.

We do not attempt to repair gaps here. We record them, and the offline replay
stage resyncs from the next REST snapshot and marks the window untrusted. That
separation is deliberate: the capture process should never make a judgement call
that could silently corrupt the archive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    kind: str  # "missing" | "duplicate_or_reorder"
    expected_pu: int  # the u we last saw
    got_pu: int
    got_U: int
    got_u: int
    delta: int  # got_pu - expected_pu; >0 events lost, <0 replayed/reordered

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "expected_pu": self.expected_pu,
            "got_pu": self.got_pu,
            "U": self.got_U,
            "u": self.got_u,
            "delta": self.delta,
        }


class SequenceChecker:
    """Stateful continuity check. Not thread-safe; use one per stream."""

    def __init__(self) -> None:
        self.last_u: int | None = None
        self.events_seen = 0
        self.gaps_seen = 0

    def reset(self) -> None:
        """Call on reconnect: the first event of a new connection has no predecessor."""
        self.last_u = None

    def check(self, U: int, u: int, pu: int) -> Gap | None:
        self.events_seen += 1

        if self.last_u is None:
            self.last_u = u
            return None

        if pu == self.last_u:
            self.last_u = u
            return None

        delta = pu - self.last_u
        gap = Gap(
            kind="missing" if delta > 0 else "duplicate_or_reorder",
            expected_pu=self.last_u,
            got_pu=pu,
            got_U=U,
            got_u=u,
            delta=delta,
        )
        self.gaps_seen += 1
        # Advance regardless: we resync forward rather than stalling on the gap.
        self.last_u = u
        return gap
