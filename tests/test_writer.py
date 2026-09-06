import gzip
from pathlib import Path

from lobforge.writer import RotatingJsonlWriter


def _read_gz(p: Path) -> list[str]:
    with gzip.open(p, "rt") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def test_writes_and_seals_on_close(tmp_path):
    w = RotatingJsonlWriter(tmp_path, "depth", rotate_seconds=3600)
    w.write(b'{"a":1}', now=1_700_000_000.0)
    w.write(b'{"a":2}', now=1_700_000_001.0)
    w.close()

    finals = list(tmp_path.rglob("*.jsonl.gz"))
    parts = list(tmp_path.rglob("*.part"))
    assert len(finals) == 1, "exactly one sealed file"
    assert parts == [], "no .part left behind after clean close"
    assert _read_gz(finals[0]) == ['{"a":1}', '{"a":2}']


def test_rotates_on_period_boundary(tmp_path):
    w = RotatingJsonlWriter(tmp_path, "depth", rotate_seconds=3600)
    w.write(b'{"h":1}', now=1_700_000_000.0)      # inside hour N
    w.write(b'{"h":2}', now=1_700_000_000.0 + 3700)  # hour N+1
    w.close()

    finals = sorted(tmp_path.rglob("*.jsonl.gz"))
    assert len(finals) == 2
    assert _read_gz(finals[0]) == ['{"h":1}']
    assert _read_gz(finals[1]) == ['{"h":2}']


def test_hive_style_partition_layout(tmp_path):
    w = RotatingJsonlWriter(tmp_path, "trades", rotate_seconds=3600)
    w.write(b'{"x":1}', now=1_700_000_000.0)
    w.close()
    p = next(tmp_path.rglob("*.jsonl.gz"))
    rel = p.relative_to(tmp_path).parts
    assert rel[0] == "trades"
    assert rel[1].startswith("date=")
    assert rel[2].startswith("hour=")


def test_partial_file_is_readable_after_flush(tmp_path):
    """Simulates SIGKILL: the .part file must still decompress up to last flush."""
    w = RotatingJsonlWriter(tmp_path, "depth", rotate_seconds=3600, flush_seconds=0)
    w.write(b'{"a":1}', now=1_700_000_000.0)
    w.write(b'{"a":2}', now=1_700_000_002.0)
    # deliberately do NOT close - the gzip trailer is never written
    part = next(tmp_path.rglob("*.part"))

    with gzip.open(part, "rt") as f:
        lines = []
        try:
            for ln in f:
                lines.append(ln.rstrip("\n"))
        except EOFError:
            pass  # expected: truncated stream
    assert lines == ['{"a":1}', '{"a":2}']


def test_counters(tmp_path):
    w = RotatingJsonlWriter(tmp_path, "depth")
    for i in range(10):
        w.write(b'{"i":%d}' % i, now=1_700_000_000.0)
    w.close()
    assert w.records_written == 10
    assert w.bytes_written > 0


def test_never_appends_to_an_orphaned_part_file(tmp_path):
    """A .part left by a killed process has no gzip trailer.

    Appending a new member after it yields a stream that fails with
    "invalid block type" at the boundary and takes the WHOLE file with it -
    including all the data that was written before the crash. The orphan must
    be moved aside so the new file starts clean.
    """
    import gzip as _gzip

    w = RotatingJsonlWriter(tmp_path, "depth", rotate_seconds=3600, flush_seconds=0)
    w.write(b'{"a":1}', now=1_700_000_000.0)
    w.write(b'{"a":2}', now=1_700_000_000.0)
    part = next(tmp_path.rglob("*.part"))
    # simulate SIGKILL: flushed but never closed, so no trailer
    w._fh.flush()
    w._fh = None

    w2 = RotatingJsonlWriter(tmp_path, "depth", rotate_seconds=3600, flush_seconds=0)
    w2.write(b'{"b":1}', now=1_700_000_000.0)
    w2.close()

    orphans = list(tmp_path.rglob("*.orphan*"))
    assert orphans, "the headerless file must be preserved, not appended to"

    final = next(tmp_path.rglob("*.jsonl.gz"))
    with _gzip.open(final, "rt") as fh:
        assert [ln.strip() for ln in fh if ln.strip()] == ['{"b":1}']

    # and the orphan still yields its records up to the last flush
    # Accumulate line by line: a comprehension raises before it can assign,
    # discarding everything read so far - the same trap the salvage tool hits.
    got = []
    with _gzip.open(orphans[0], "rt") as fh:
        try:
            for ln in fh:
                if ln.strip():
                    got.append(ln.strip())
        except EOFError:
            pass  # expected: truncated stream, no trailer
    assert '{"a":1}' in got, "orphan stays readable up to its last flush"


def test_rare_stream_record_survives_an_unclean_stop(tmp_path):
    """A stream that writes once per half hour must not hold its record in the
    gzip buffer waiting for a next write that may never come.

    Snapshots are the only anchors reconstruction has. One lost snapshot
    strands a whole session's events with nothing to place them on - which is
    exactly what happened to 2026-08-13 hours 17 and 18.
    """
    import gzip as _gzip

    w = RotatingJsonlWriter(tmp_path, "snapshots", rotate_seconds=3600,
                            flush_seconds=0)
    w.write(b'{"lastUpdateId":123}', now=1_700_000_000.0)
    # simulate SIGKILL: no close(), so no gzip trailer
    part = next(tmp_path.rglob("*.part"))

    got = []
    with _gzip.open(part, "rt") as fh:
        try:
            for ln in fh:
                if ln.strip():
                    got.append(ln.strip())
        except EOFError:
            pass
    assert got == ['{"lastUpdateId":123}'], "the single record must be on disk"