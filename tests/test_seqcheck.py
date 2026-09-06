from lobforge.seqcheck import SequenceChecker


def test_first_event_never_reports_gap():
    s = SequenceChecker()
    assert s.check(U=100, u=110, pu=99) is None
    assert s.last_u == 110


def test_continuous_stream_has_no_gaps():
    s = SequenceChecker()
    s.check(U=100, u=110, pu=99)
    assert s.check(U=111, u=120, pu=110) is None
    assert s.check(U=121, u=130, pu=120) is None
    assert s.gaps_seen == 0
    assert s.events_seen == 3


def test_missing_events_detected():
    s = SequenceChecker()
    s.check(U=100, u=110, pu=99)
    gap = s.check(U=131, u=140, pu=130)  # pu should have been 110
    assert gap is not None
    assert gap.kind == "missing"
    assert gap.expected_pu == 110
    assert gap.got_pu == 130
    assert gap.delta == 20


def test_reorder_or_duplicate_detected():
    s = SequenceChecker()
    s.check(U=100, u=200, pu=99)
    gap = s.check(U=111, u=120, pu=110)  # pu behind our last u
    assert gap is not None
    assert gap.kind == "duplicate_or_reorder"
    assert gap.delta == -90


def test_resyncs_forward_after_gap():
    """A gap must not wedge the checker into reporting on every later event."""
    s = SequenceChecker()
    s.check(U=100, u=110, pu=99)
    assert s.check(U=131, u=140, pu=130) is not None   # gap
    assert s.check(U=141, u=150, pu=140) is None       # back in sync
    assert s.gaps_seen == 1


def test_reset_on_reconnect():
    s = SequenceChecker()
    s.check(U=100, u=110, pu=99)
    s.reset()
    assert s.check(U=900, u=910, pu=899) is None, "new connection has no predecessor"
    assert s.gaps_seen == 0


def test_env_values_tolerate_inline_comments(monkeypatch):
    """Docker's --env-file keeps everything after '=', comments included.

    A shell-sourced .env and a Docker env-file disagree on this, so the same
    file that works locally crashes in a container. Tolerate both.
    """
    from lobforge.config import Config

    monkeypatch.setenv("LOBF_FLUSH_SECONDS", "5          # bounds worst-case loss")
    monkeypatch.setenv("LOBF_SYMBOL", "ethusdt   # not btc")
    monkeypatch.setenv("LOBF_GAP_SNAPSHOT_COOLDOWN_S", "2.5  # throttle")
    cfg = Config()
    assert cfg.flush_seconds == 5
    assert cfg.symbol == "ethusdt"
    assert cfg.gap_snapshot_cooldown_s == 2.5


def test_env_values_without_comments_still_parse(monkeypatch):
    from lobforge.config import Config

    monkeypatch.setenv("LOBF_FLUSH_SECONDS", "9")
    monkeypatch.setenv("LOBF_SYMBOL", "solusdt")
    cfg = Config()
    assert cfg.flush_seconds == 9
    assert cfg.symbol == "solusdt"
