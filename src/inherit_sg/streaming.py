"""Bounded asynchronous producer/consumer with backpressure and error propagation."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Queue, Full, Empty
from threading import Thread, Event
from time import perf_counter


@dataclass
class QueueStats:
    capacity: int
    submitted: int = 0
    completed: int = 0
    high_watermark: int = 0
    full_observations: int = 0
    blocked_seconds: float = 0.0


def consume_async(items, consume, capacity=50):
    """items owns geometric tracking; consume owns semantic state.

    Items must contain immutable data or owned snapshots. FIFO order is retained;
    no frames are dropped. The producer propagates worker failures even while the
    queue is full. All accepted work completes before returning successfully.
    """
    if capacity < 1:
        raise ValueError("queue capacity must be positive")
    queue = Queue(maxsize=capacity)
    stop = Event()
    errors = []
    sentinel = object()
    stats = QueueStats(capacity)

    def worker():
        while not stop.is_set():
            try:
                item = queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                if item is sentinel:
                    return
                consume(item)
                stats.completed += 1
            except BaseException as error:
                errors.append(error)
                stop.set()
                return
            finally:
                queue.task_done()

    def put(item):
        observed_full = False
        started = perf_counter()
        while True:
            if errors:
                raise RuntimeError("semantic worker failed") from errors[0]
            try:
                queue.put(item, timeout=0.05)
                stats.high_watermark = max(stats.high_watermark, queue.qsize())
                if observed_full:
                    stats.blocked_seconds += perf_counter() - started
                return
            except Full:
                if not observed_full:
                    stats.full_observations += 1
                    observed_full = True

    thread = Thread(target=worker, name="inherit-semantic")
    thread.start()
    try:
        for item in items:
            put(item)
            stats.submitted += 1
        put(sentinel)
        thread.join()
        if errors:
            raise RuntimeError("semantic worker failed") from errors[0]
        return stats
    finally:
        stop.set()
        thread.join()
