"""Configuration, loaded from environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _clean(raw: str) -> str:
    """Strip a trailing inline comment and surrounding whitespace.

    Docker's --env-file parser does NOT strip comments the way a shell does:
    `KEY=5  # why` yields the literal string `5  # why`. Rather than rely on
    every env file being comment-free, tolerate it here.
    """
    return raw.split(" #", 1)[0].split("\t#", 1)[0].strip()


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    return _clean(raw) if raw is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    cleaned = _clean(raw)
    return int(cleaned) if cleaned else default


@dataclass(frozen=True)
class Config:
    # --- market ---
    symbol: str = field(default_factory=lambda: _env_str("LOBF_SYMBOL", "btcusdt").lower())
    depth_interval: str = field(default_factory=lambda: _env_str("LOBF_DEPTH_INTERVAL", "100ms"))

    # --- endpoints (USD-M futures) ---
    ws_base: str = field(default_factory=lambda: _env_str("LOBF_WS_BASE", "wss://fstream.binance.com"))
    rest_base: str = field(default_factory=lambda: _env_str("LOBF_REST_BASE", "https://fapi.binance.com"))
    snapshot_limit: int = field(default_factory=lambda: _env_int("LOBF_SNAPSHOT_LIMIT", 1000))

    # --- storage ---
    data_root: Path = field(default_factory=lambda: Path(_env_str("LOBF_DATA_ROOT", "./data")))
    rotate_seconds: int = field(default_factory=lambda: _env_int("LOBF_ROTATE_SECONDS", 3600))
    flush_seconds: int = field(default_factory=lambda: _env_int("LOBF_FLUSH_SECONDS", 5))

    # --- resilience ---
    queue_maxsize: int = field(default_factory=lambda: _env_int("LOBF_QUEUE_MAXSIZE", 200_000))
    stale_timeout_s: int = field(default_factory=lambda: _env_int("LOBF_STALE_TIMEOUT_S", 20))
    snapshot_interval_s: int = field(default_factory=lambda: _env_int("LOBF_SNAPSHOT_INTERVAL_S", 1800))
    # Every gap triggers a resync snapshot. Gaps cluster during volatility, so an
    # unthrottled trigger can burst REST calls straight into Binance's weight
    # limit (2400/min; depth?limit=1000 costs 20 => ~120 req/min). Sustained 429s
    # escalate to an IP ban, which costs far more than the missed anchors.
    gap_snapshot_cooldown_s: float = field(
        default_factory=lambda: float(_env_str("LOBF_GAP_SNAPSHOT_COOLDOWN_S", "5")))
    reconnect_max_backoff_s: int = field(default_factory=lambda: _env_int("LOBF_RECONNECT_MAX_BACKOFF_S", 60))
    # Binance force-closes a websocket after 24h. Recycle before that, on our terms.
    connection_max_age_s: int = field(default_factory=lambda: _env_int("LOBF_CONN_MAX_AGE_S", 20 * 3600))

    # --- observability ---
    metrics_interval_s: int = field(default_factory=lambda: _env_int("LOBF_METRICS_INTERVAL_S", 60))
    # A subscribed stream that delivers nothing is invisible to every other
    # health signal: no gap, no error, no stall - the data was never lost,
    # it was never sent. This deadline is the only thing that catches it.
    subscription_timeout_s: int = field(
        default_factory=lambda: _env_int("LOBF_SUBSCRIPTION_TIMEOUT_S", 30))
    fail_on_missing_stream: int = field(
        default_factory=lambda: _env_int("LOBF_FAIL_ON_MISSING_STREAM", 0))
    # A collector that cannot write is not collecting. Persisting through it
    # produces thousands of log lines, an empty archive, and a process that
    # still looks alive to anything watching the pid.
    max_write_failures: int = field(
        default_factory=lambda: _env_int("LOBF_MAX_WRITE_FAILURES", 50))
    log_level: str = field(default_factory=lambda: _env_str("LOBF_LOG_LEVEL", "INFO"))

    @property
    def depth_stream(self) -> str:
        return f"{self.symbol}@depth@{self.depth_interval}"

    @property
    def trade_stream(self) -> str:
        return f"{self.symbol}@aggTrade"

    # Binance split the futures websocket into routed endpoints (2026-03, legacy
    # URLs decommissioned 2026-04-23). Diff depth is a /public stream; aggTrade is
    # a /market stream. An unrouted connection silently receives ONLY /public -
    # no error, no rejection, just nothing. Hence two connections, not one.
    @property
    def public_ws_url(self) -> str:
        return f"{self.ws_base}/public/stream?streams={self.depth_stream}"

    @property
    def market_ws_url(self) -> str:
        return f"{self.ws_base}/market/stream?streams={self.trade_stream}"

    @property
    def ws_url(self) -> str:  # kept for logging/back-compat
        return self.public_ws_url

    @property
    def expected_streams(self) -> set[str]:
        """Stream names that MUST deliver frames for the archive to be complete."""
        return {self.depth_stream, self.trade_stream}

    @property
    def snapshot_url(self) -> str:
        return (
            f"{self.rest_base}/fapi/v1/depth"
            f"?symbol={self.symbol.upper()}&limit={self.snapshot_limit}"
        )
