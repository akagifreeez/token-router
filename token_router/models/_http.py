"""Shared resilience layer for the HTTP model backends.

This is a direct adaptation of the proven retry/backoff/rate-limit/cache layer
from ``hl-read`` (``hl_read/info.py``): the same ``_is_transient`` error
classifier, the same exponential-backoff-with-jitter ``_call`` loop, the same
thread-safe TTL ``_cached`` and per-minute ``_throttle``. Both ``FireworksModel``
and ``LocalModel`` are thin REST clients on top of this, so they get identical,
well-tested resilience for free.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .base import ModelError

# Exception class names that always mean "retry might help", independent of the
# concrete library (requests / urllib3). Mirrors hl-read's _TRANSIENT_NAMES.
_TRANSIENT_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ConnectionResetError",
        "ChunkedEncodingError",
        "ProtocolError",
        "ReadTimeout",
        "ReadTimeoutError",
        "RemoteDisconnected",
        "ServerError",
        "Timeout",
    }
)
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpError(ModelError):
    """An HTTP call returned a non-success status. Carries ``status_code`` so
    ``_is_transient`` can decide whether retrying is worthwhile."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_transient(e: Exception) -> bool:
    """True if ``e`` is a transient error worth retrying (vs a real 4xx/bug)."""
    if type(e).__name__ in _TRANSIENT_NAMES:
        return True
    code = getattr(e, "status_code", None)
    if code is None:
        code = getattr(e, "code", None)
    try:
        if int(code) in _TRANSIENT_STATUS:
            return True
    except (TypeError, ValueError):
        pass
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


class ResilientClient:
    """Mixin giving a REST client retry/backoff, rate-limiting and TTL caching.

    Resilience knobs (all keyword-only, sensible defaults):

    * ``max_retries`` - retry attempts for transient failures (default 4).
    * ``backoff_base`` / ``backoff_max`` - exponential backoff window, seconds.
    * ``rate_limit_per_min`` - if set, space calls to at most N per minute.
    * ``cache_ttl`` - seconds to cache identical calls (0 = always fresh).
    * ``http_timeout`` - per-request timeout so a stalled socket can't hang.
    """

    def __init__(
        self,
        *,
        max_retries: int = 4,
        backoff_base: float = 0.4,
        backoff_max: float = 8.0,
        rate_limit_per_min: Optional[float] = None,
        cache_ttl: float = 0.0,
        http_timeout: Optional[float] = 30.0,
    ) -> None:
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.cache_ttl = float(cache_ttl)
        self.http_timeout = http_timeout
        self._min_interval = (60.0 / rate_limit_per_min) if rate_limit_per_min else 0.0

        self._lock = threading.Lock()        # guards the cache
        self._rate_lock = threading.Lock()   # guards only the rate-limiter clock
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._last_call = 0.0

    def _throttle(self) -> None:
        # Uses its own lock, never the cache lock, so a throttle sleep can never
        # block an unrelated cache read.
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Invoke ``fn(*args)`` with rate limiting and retry/backoff.

        Transient failures (network errors, HTTP 429/5xx) are retried with
        exponential backoff plus jitter; a non-transient error (a real 4xx, a
        bug) is raised immediately. On exhaustion raises ``ModelError``.
        """
        last_err: Optional[Exception] = None
        attempt = 0
        while True:
            self._throttle()
            try:
                return fn(*args)
            except Exception as e:  # noqa: BLE001 - classified by _is_transient
                if not _is_transient(e):
                    raise
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    break
                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
                time.sleep(delay * (0.7 + 0.6 * random.random()))  # full-ish jitter
        raise ModelError(
            f"request failed after {self.max_retries} retries: "
            f"{type(last_err).__name__}: {last_err}"
        ) from last_err

    def _cached(self, key: str, ttl: float, producer: Callable[[], Any]) -> Any:
        if ttl <= 0:
            return producer()
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and (now - hit[0]) < ttl:
                return hit[1]
        value = producer()  # produced outside the lock (network call)
        with self._lock:
            self._cache[key] = (time.monotonic(), value)
        return value

    def clear_cache(self) -> None:
        """Drop all cached responses; the next call fetches fresh."""
        with self._lock:
            self._cache.clear()
