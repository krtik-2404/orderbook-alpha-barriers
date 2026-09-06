import pytest

from lobforge.book import BookEngine, BookState, Desync, InvariantError


def snap(last_id=1000, mid=63590.0, n=12):
    """On-tick prices, as Binance actually sends them: multiples of 0.10."""
    return {
        "lastUpdateId": last_id,
        "bids": [[f"{mid - (i + 1) * 0.1:.2f}", f"{1.0 + i:.3f}"] for i in range(n)],
        "asks": [[f"{mid + (i + 1) * 0.1:.2f}", f"{1.0 + i:.3f}"] for i in range(n)],
    }


def ev(U, u, pu, b=(), a=(), E=1_700_000_000_000):
    return {"e": "depthUpdate", "E": E, "U": U, "u": u, "pu": pu,
            "b": [list(x) for x in b], "a": [list(x) for x in a]}


# ------------------------------------------------------------------ basics

def test_snapshot_gives_a_valid_book():
    e = BookEngine()
    e.apply_snapshot(snap())
    bids, asks = e.top()
    assert len(bids) == 10 and len(asks) == 10
    assert bids[0][0] == 63589.9
    assert asks[0][0] == 63590.1
    assert e.check(bids, asks) == []


def test_engine_emits_nothing_before_a_snapshot():
    """A book that cannot be justified from the data is worse than no book."""
    e = BookEngine()
    assert e.synced is False
    assert e.apply_event(ev(1001, 1005, 1000)) is None


def test_applies_an_event_and_advances():
    e = BookEngine()
    e.apply_snapshot(snap())
    st = e.apply_event(ev(1001, 1005, 1000, b=[("63589.90", "9.000")]))
    assert isinstance(st, BookState)
    assert st.update_id == 1005
    assert st.bids[0] == (63589.9, 9.0)
    assert e.last_update_id == 1005


# ------------------------------------------------------------- the classics

def test_zero_quantity_deletes_the_level():
    """~8% of BTCUSDT level updates are removals. Treating qty 0 as a value
    fills the book with dead prices that never clear."""
    e = BookEngine()
    e.apply_snapshot(snap())
    before = e.top()[0][0][0]
    e.apply_event(ev(1001, 1005, 1000, b=[(f"{before:.2f}", "0")]))
    bids, _ = e.top()
    assert all(p != before for p, _ in bids), "level should be gone entirely"
    assert all(q > 0 for _, q in bids), "no zero-quantity levels survive"


def test_integer_ticks_make_insert_and_delete_agree():
    """63590.70 is not exactly representable. Keying on floats leaks a phantom
    level when the same nominal price is written one way and removed another."""
    e = BookEngine()
    e.apply_snapshot(snap())
    e.apply_event(ev(1001, 1002, 1000, b=[("63589.70", "5.0")]))
    n_after_insert = len(e._bids)
    # same price, reached by a different float path: 6358.97 * 10 is not
    # bit-identical to float("63589.70")
    e.apply_event(ev(1003, 1004, 1002, b=[(str(6358.97 * 10), "0")]))
    assert len(e._bids) == n_after_insert - 1, "removal must find the level"


def test_stale_events_before_the_snapshot_are_skipped():
    e = BookEngine()
    e.apply_snapshot(snap(last_id=1000))
    assert e.apply_event(ev(900, 950, 899)) is None
    assert e.events_skipped == 1
    assert e.last_update_id == 1000, "stale event must not move the cursor"


# ------------------------------------------------------------------- trust

def test_sequence_break_desyncs_and_stops_emitting():
    e = BookEngine()
    e.apply_snapshot(snap())
    e.apply_event(ev(1001, 1005, 1000))
    out = e.apply_event(ev(1020, 1030, 1019))       # pu should be 1005
    assert isinstance(out, Desync)
    assert out.reason == "sequence_break"
    assert (out.expected, out.got) == (1005, 1019)
    assert e.synced is False
    assert e.apply_event(ev(1031, 1040, 1030)) is None, "stays dark until resync"


def test_resync_from_a_later_snapshot():
    e = BookEngine()
    e.apply_snapshot(snap())
    e.apply_event(ev(1001, 1005, 1000))
    e.apply_event(ev(1020, 1030, 1019))             # break
    e.apply_snapshot(snap(last_id=2000))
    st = e.apply_event(ev(2001, 2005, 2000))
    assert isinstance(st, BookState) and st.update_id == 2005


def test_snapshot_older_than_the_stream_is_rejected():
    """If the first event starts after lastUpdateId+1, the events in between
    were never seen and this snapshot cannot anchor the stream."""
    e = BookEngine()
    e.apply_snapshot(snap(last_id=1000))
    out = e.apply_event(ev(5000, 5010, 4999))
    assert isinstance(out, Desync) and out.reason == "snapshot_stale"


# -------------------------------------------------------------- invariants

def test_crossed_book_raises_rather_than_emitting():
    e = BookEngine()
    e.apply_snapshot(snap())
    with pytest.raises(InvariantError, match="crossed"):
        e.apply_event(ev(1001, 1005, 1000, b=[("63590.50", "1.0")]))


def test_non_strict_mode_still_reports_the_problem():
    e = BookEngine(strict=False)
    e.apply_snapshot(snap())
    st = e.apply_event(ev(1001, 1005, 1000, b=[("63590.50", "1.0")]))
    assert isinstance(st, BookState)
    assert e.check(list(st.bids), list(st.asks)), "problem is still detectable"


def test_ordering_holds_after_many_updates():
    e = BookEngine()
    e.apply_snapshot(snap())
    last = 1000
    for i in range(200):
        U, u = last + 1, last + 3
        e.apply_event(ev(U, u, last,
                         b=[(f"{63589.9 - (i % 7) * 0.1:.2f}", f"{(i % 5) + 1}.0")],
                         a=[(f"{63590.1 + (i % 7) * 0.1:.2f}", f"{(i % 4) + 1}.0")]))
        last = u
    bids, asks = e.top()
    assert e.check(bids, asks) == []
    assert bids == sorted(bids, key=lambda x: -x[0])
    assert asks == sorted(asks, key=lambda x: x[0])


# ------------------------------------------------------------------ output

def test_as_row_is_the_deeplob_layout():
    e = BookEngine()
    e.apply_snapshot(snap())
    st = e.apply_event(ev(1001, 1005, 1000))
    row = st.as_row(10)
    assert len(row) == 40, "10 levels x 2 sides x (price, qty)"
    assert row[0] == st.asks[0][0] and row[2] == st.bids[0][0]


def test_as_row_pads_a_thin_side_rather_than_dropping_the_row():
    e = BookEngine()
    s = snap(n=3)
    e.apply_snapshot(s)
    st = e.state(0, 1000)
    row = st.as_row(10)
    assert len(row) == 40
    assert row[-3] == 0.0 or row[-1] == 0.0, "padding uses zero quantity"


def test_mid_and_spread():
    e = BookEngine()
    e.apply_snapshot(snap(mid=63590.0))
    st = e.state(0, 1000)
    assert st.best_bid == 63589.9 and st.best_ask == 63590.1
    assert abs(st.mid - 63590.0) < 1e-9
    assert abs(st.spread - 0.20) < 1e-9
    assert 0 < st.spread_bp < 1


# ------------------------------------------------------------------ replay

def test_replay_resyncs_across_a_break():
    e = BookEngine()
    snaps = [snap(last_id=1000), snap(last_id=2000)]
    events = [ev(1001, 1005, 1000),
              ev(1006, 1010, 1005),
              ev(1900, 1950, 1899),      # break -> desync
              ev(2001, 2005, 2000),      # resync from the second snapshot
              ev(2006, 2010, 2005)]
    out = list(e.replay(snaps, events))
    states = [o for o in out if isinstance(o, BookState)]
    desyncs = [o for o in out if isinstance(o, Desync)]
    assert len(desyncs) == 1
    assert [s.update_id for s in states] == [1005, 1010, 2005, 2010]


def test_replay_is_deterministic():
    """Same input, byte-identical output - or a code change cannot be told
    apart from a data change."""
    snaps, events = [snap()], [ev(1000 + i, 1002 + i, 999 + i) for i in range(1, 60, 2)]
    a = [x for x in BookEngine().replay(snaps, list(events))]
    b = [x for x in BookEngine().replay(snaps, list(events))]
    assert a == b
