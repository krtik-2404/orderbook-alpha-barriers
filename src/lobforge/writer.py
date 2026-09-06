"""Rotating, gzip-compressed JSONL writer.

Design notes (these are the bits that make it production-grade):

* Files are written as ``*.jsonl.gz.part`` and atomically ``os.rename``d to their
  final name on rotation/close. A consumer scanning the directory therefore never
  sees a half-written file under its final name.
* ``GzipFile.flush()`` issues a ``Z_SYNC_FLUSH``, which makes the compressed stream
  readable up to that point. If the process is SIGKILLed, the ``.part`` file is
  still decompressible up to the last flush instead of being a total loss.
* Rotation is on wall-clock boundaries, so hour buckets line up across restarts.
"""

from __future__ import annotations

import gzip
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


class RotatingJsonlWriter:
    def __init__(
        self,
        root: Path,
        name: str,
        rotate_seconds: int = 3600,
        flush_seconds: int = 5,
    ) -> None:
        self._root = Path(root)
        self._name = name
        self._rotate_seconds = rotate_seconds
        self._flush_seconds = flush_seconds

        self._fh: gzip.GzipFile | None = None
        self._part_path: Path | None = None
        self._final_path: Path | None = None
        self._period_start: int = 0
        self._last_flush: float = 0.0

        self.records_written = 0
        self.bytes_written = 0

    # ---------- lifecycle ----------

    def _period_for(self, ts: float) -> int:
        return int(ts) - (int(ts) % self._rotate_seconds)

    def _paths_for(self, period_start: int) -> tuple[Path, Path]:
        tm = time.gmtime(period_start)
        directory = (
            self._root
            / self._name
            / f"date={time.strftime('%Y-%m-%d', tm)}"
            / f"hour={time.strftime('%H', tm)}"
        )
        stem = f"{self._name}-{time.strftime('%Y%m%dT%H%M%SZ', tm)}.jsonl.gz"
        return directory / (stem + ".part"), directory / stem

    def _open(self, period_start: int, now: float) -> None:
        part, final = self._paths_for(period_start)
        part.parent.mkdir(parents=True, exist_ok=True)

        # NEVER append to a pre-existing .part. If a previous process died
        # without close(), that file ends mid-deflate with no gzip trailer;
        # appending a fresh member after it produces a stream that fails with
        # "invalid block type" at the boundary and takes the WHOLE file with it.
        # Move it aside instead - it stays readable up to its last flush.
        if part.exists() and part.stat().st_size > 0:
            n = 0
            while True:
                orphan = part.with_name(f"{part.name}.orphan{n or ''}")
                if not orphan.exists():
                    break
                n += 1
            part.rename(orphan)
            log.warning(
                "found orphaned %s from an unclean stop - preserved as %s "
                "and starting a fresh file (never append to a headerless member)",
                part.name, orphan.name,
            )
        # mtime=0 keeps the gzip header deterministic, which makes files hashable
        # for reproducibility checks.
        self._fh = gzip.GzipFile(filename=str(part), mode="ab", compresslevel=6, mtime=0)
        self._part_path, self._final_path = part, final
        self._period_start = period_start
        # Seed from the caller's clock, not wall time: they must be the same clock
        # or the flush interval silently never fires.
        self._last_flush = now
        log.info("opened %s", part)

    def _close(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        if self._part_path and self._final_path and self._part_path.exists():
            os.replace(self._part_path, self._final_path)
            log.info("sealed %s", self._final_path)
        self._part_path = self._final_path = None

    # ---------- writing ----------

    def write(self, line: bytes, now: float | None = None) -> None:
        """Append one already-serialised record. ``line`` must not contain a newline."""
        now = time.time() if now is None else now
        period = self._period_for(now)

        if self._fh is None:
            self._open(period, now)
        elif period != self._period_start:
            self._close()
            self._open(period, now)

        assert self._fh is not None
        self._fh.write(line)
        self._fh.write(b"\n")
        self.records_written += 1
        self.bytes_written += len(line) + 1

        if now - self._last_flush >= self._flush_seconds:
            self._fh.flush()
            self._last_flush = now

    def close(self) -> None:
        self._close()

    def __enter__(self) -> RotatingJsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
