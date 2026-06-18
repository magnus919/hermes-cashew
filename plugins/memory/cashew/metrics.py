"""Lightweight operational metrics for the cashew memory provider.

Tracks query latency, sync throughput, queue depth, and sleep cycle
duration using thread-safe counters and timers. Metrics are emitted as
structured log events at INFO level for downstream monitoring.

No external dependencies — stdlib only. Designed for Hermes Agent's
multi-threaded environment (sync worker, cron jobs, main agent loop).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginMetrics:
    """Thread-safe operational metrics for the cashew plugin.

    All counters and timestamps are protected by a lock so the sync
    worker, cron job, and main agent loop can safely read/write.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Query metrics
    query_count: int = 0
    query_cache_hits: int = 0
    query_total_ms: float = 0.0

    # Sync metrics
    sync_extracted: int = 0
    sync_failed: int = 0
    sync_dropped: int = 0
    sync_queue_depth: int = 0

    # Sleep cycle metrics
    sleep_cycle_count: int = 0
    sleep_cycle_total_ms: float = 0.0
    sleep_nodes_processed: int = 0

    # Startup
    startup_time: float = field(default_factory=time.time)

    def _snapshot(self) -> dict[str, Any]:
        """Return a dict copy of all counters under the lock."""
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.startup_time, 1),
                "query_count": self.query_count,
                "query_cache_hits": self.query_cache_hits,
                "query_cache_hit_pct": round(
                    self.query_cache_hits / max(self.query_count, 1) * 100, 1
                ),
                "query_avg_ms": round(
                    self.query_total_ms / max(self.query_count, 1), 1
                ),
                "sync_extracted": self.sync_extracted,
                "sync_failed": self.sync_failed,
                "sync_dropped": self.sync_dropped,
                "sync_queue_depth": self.sync_queue_depth,
                "sleep_cycle_count": self.sleep_cycle_count,
                "sleep_avg_ms": round(
                    self.sleep_cycle_total_ms / max(self.sleep_cycle_count, 1), 1
                ),
                "sleep_nodes_processed": self.sleep_nodes_processed,
            }

    def record_query(self, cache_hit: bool, elapsed_ms: float) -> None:
        """Record a completed cashew_query operation."""
        with self._lock:
            self.query_count += 1
            self.query_total_ms += elapsed_ms
            if cache_hit:
                self.query_cache_hits += 1

    def record_sync_success(self) -> None:
        """Record a sync turn that was successfully extracted."""
        with self._lock:
            self.sync_extracted += 1

    def record_sync_failure(self) -> None:
        """Record a sync turn that failed extraction."""
        with self._lock:
            self.sync_failed += 1

    def record_sync_dropped(self) -> None:
        """Record a sync turn dropped due to queue overflow or shutdown."""
        with self._lock:
            self.sync_dropped += 1

    def set_queue_depth(self, depth: int) -> None:
        """Update the current sync queue depth."""
        with self._lock:
            self.sync_queue_depth = depth

    def record_sleep_cycle(self, elapsed_ms: float, nodes: int) -> None:
        """Record a completed sleep cycle tick."""
        with self._lock:
            self.sleep_cycle_count += 1
            self.sleep_cycle_total_ms += elapsed_ms
            self.sleep_nodes_processed += nodes

    def emit(self) -> None:
        """Emit current metrics as a structured log event."""
        snap = self._snapshot()
        logger.info(
            "cashew_metrics "
            "uptime_s=%(uptime_s).1f "
            "queries=%(query_count)d "
            "query_avg_ms=%(query_avg_ms).1f "
            "cache_hit_pct=%(query_cache_hit_pct).1f "
            "sync_extracted=%(sync_extracted)d "
            "sync_failed=%(sync_failed)d "
            "sync_dropped=%(sync_dropped)d "
            "queue_depth=%(sync_queue_depth)d "
            "sleep_cycles=%(sleep_cycle_count)d "
            "sleep_avg_ms=%(sleep_avg_ms).1f "
            "sleep_nodes=%(sleep_nodes_processed)d",
            snap,
        )


# Module-level singleton — one metrics instance per Python process.
# The provider accesses this via `from .metrics import _METRICS`.
_METRICS = PluginMetrics()
