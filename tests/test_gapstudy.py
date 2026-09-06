"""The two arms gapstudy.py compares must actually differ in the way it claims.

Everything downstream of reconstruction - the features, the folds, the model -
is shared code already covered elsewhere. What is new here is the reconstruction
itself, so that is what these test: after a gap the trusted arm is CORRECT and
segmented, the naive arm is CONTINUOUS and wrong.
"""

from lobforge.book import BookEngine, BookState, Desync
from lobforge.sandbox.simulator import BookSimulator

DROP_EVERY = 37


def stream(n=600, seed=3):
    sim = BookSimulator(seed=seed, levels=20)
    snap = sim.snapshot()
    return snap, [sim.depth_event() for _ in range(n)]


def replay(snap, events, trust, drops=()):
    """gapstudy.reconstruct in miniature: drop events, and on a sequence break
    either re-anchor over what was dropped (trusted) or never notice (naive)."""
    e = BookEngine(strict=False, trust_sequence=trust)
    e.add_snapshots([snap])
    states, runs, missed, resyncs = [], [], [], 0
    start = 0
    for i, ev in enumerate(events):
        if i in drops:
            missed.append(ev)
            continue
        item = next(e.feed([ev]), None)
        if isinstance(item, Desync):
            if len(states) > start:
                runs.append((start, len(states)))
            start = len(states)
            if item.reason == "sequence_break" and missed:
                e.resync_over(missed)
                resyncs += 1
                item = next(e.feed([ev]), None)
        missed.clear()
        if isinstance(item, BookState):
            states.append(item)
    if len(states) > start:
        runs.append((start, len(states)))
    return e, states, runs, resyncs


def test_resync_over_reproduces_the_true_book():
    """The whole design rests on this: the events a gap swallowed are still in
    the archive, so replaying them gives exactly the book a REST refetch would
    have returned. If that is not exact, the trusted arm is not a control."""
    snap, evs = stream()
    drops = set(range(DROP_EVERY, len(evs), DROP_EVERY))

    truth, _, _, _ = replay(snap, evs, trust=True)
    trusted, states, runs, resyncs = replay(snap, evs, trust=True, drops=drops)

    assert resyncs == len(drops), "every gap should re-anchor, not just the first"
    assert trusted.top() == truth.top()
    assert trusted.last_update_id == truth.last_update_id
    # One run per gap, and no more data lost than the dropped events themselves.
    assert len(runs) == len(drops) + 1
    assert len(states) == len(evs) - len(drops)


def test_the_naive_arm_splices_and_carries_a_wrong_book():
    """No desync, one unbroken run, and levels the missing events would have
    moved are still sitting there. That staleness is what the study measures."""
    snap, evs = stream()
    drops = set(range(DROP_EVERY, len(evs), DROP_EVERY))

    _, good, _, _ = replay(snap, evs, trust=True, drops=drops)
    naive, states, runs, resyncs = replay(snap, evs, trust=False, drops=drops)

    assert naive.desyncs == 0 and resyncs == 0
    assert len(runs) == 1, "a naive pipeline sees one continuous stream"
    assert len(states) == len(good)
    # Same surviving events, same count of states - and different books, which
    # is the entire premise. (Compared state by state, not just at the end:
    # a stale level is eventually overwritten, so the damage is transient and
    # a final-book check would miss it.)
    wrong = sum(a.bids != b.bids or a.asks != b.asks
                for a, b in zip(states, good))
    assert wrong > len(drops), "splicing must actually corrupt the book"


def test_trust_sequence_is_on_by_default():
    """Nothing in the real pipeline may accidentally get the naive engine."""
    assert BookEngine().trust_sequence is True
