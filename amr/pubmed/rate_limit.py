"""Thread-safe request pacing for PubMed E-utilities."""

from __future__ import annotations

import threading
import time


class RequestRateLimiter:
    """Pace request starts across threads sharing one limiter."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        """Block until this caller may start its next request."""
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + self.interval_seconds
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def reset(self) -> None:
        """Reset pacing state, primarily useful for isolated tests."""
        with self._lock:
            self._next_allowed_at = 0.0

    def penalize(self, delay_seconds: float) -> None:
        """Delay all future callers after a server-side throttling response."""
        delay_seconds = max(0.0, float(delay_seconds))
        with self._lock:
            self._next_allowed_at = max(
                self._next_allowed_at,
                time.monotonic() + delay_seconds,
            )


__all__ = ["RequestRateLimiter"]
