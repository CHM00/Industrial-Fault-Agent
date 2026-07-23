"""Minimal dependency-free metrics for the pilot service."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)

    def inc(self, name: str, value: float = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    @contextmanager
    def timer(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.inc(f"{name}_seconds_total", elapsed)
            self.inc(f"{name}_count", 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {"counters": dict(self._counters), "gauges": dict(self._gauges)}

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = []
        for group in ("counters", "gauges"):
            for name, value in sorted(snapshot[group].items()):
                safe = "".join(ch if ch.isalnum() or ch in "_:" else "_" for ch in name)
                lines.append(f"langgraph_agent_{safe} {value:g}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
