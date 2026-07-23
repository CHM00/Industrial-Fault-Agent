"""Rate limiting and bounded concurrency for pilot deployments."""

from __future__ import annotations

import collections
import os
import threading
import time


class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int | None = None):
        self.limit = requests_per_minute or int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
        self._events: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        current = now if now is not None else time.monotonic()
        cutoff = current - 60
        with self._lock:
            events = self._events.setdefault(key, collections.deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(60 - (current - events[0])))
                return False, retry_after
            events.append(current)
            return True, 0


class DiagnosisConcurrency:
    def __init__(self, maximum: int | None = None):
        self.maximum = maximum or int(os.environ.get("MAX_CONCURRENT_DIAGNOSES", "2"))
        self._semaphore = threading.BoundedSemaphore(self.maximum)
        self._lock = threading.Lock()
        self._active = 0

    def acquire(self, timeout: float | None = None) -> bool:
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active += 1
        return acquired

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active
