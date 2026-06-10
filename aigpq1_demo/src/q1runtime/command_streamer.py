from __future__ import annotations

import threading
import time
from typing import Any, Callable


class CommandStreamer:
    """Runs a command callback at a fixed rate, independent of vision processing."""

    def __init__(
        self,
        tick_fn: Callable[[float], Any],
        *,
        hz: float = 250.0,
        on_result: Callable[[Any], None] | None = None,
    ):
        self.tick_fn = tick_fn
        self.hz = float(hz)
        self.interval_s = 1.0 / max(1e-6, self.hz)
        self.on_result = on_result
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_tick_s: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._next_tick_s = None
        self._thread = threading.Thread(target=self._run, name="q1runtime-command-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def step(self, now_s: float) -> Any | None:
        now = float(now_s)
        if self._next_tick_s is None:
            self._next_tick_s = now
        if now + 1e-9 < self._next_tick_s:
            return None
        result = self.tick_fn(now)
        if self.on_result is not None:
            self.on_result(result)
        self._next_tick_s = now + self.interval_s
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "hz": self.hz,
            "interval_s": self.interval_s,
            "running": self._running.is_set(),
            "next_tick_s": self._next_tick_s,
        }

    def _run(self) -> None:
        sleep_s = min(0.002, self.interval_s / 2.0)
        while self._running.is_set():
            self.step(time.monotonic())
            time.sleep(sleep_s)
