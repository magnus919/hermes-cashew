"""Provider-local health and work outcome bookkeeping.

The provider owns one ledger per lifecycle generation.  The module-level metrics
object remains aggregate telemetry; it is deliberately not used as an answer to
the provider's operational status question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SAFE_ERROR_CLASSES = frozenset(
    {
        "Exception",
        "ImportError",
        "KeyError",
        "OSError",
        "OperationalError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)


def safe_error_class(error: BaseException) -> str:
    """Return an allowlisted exception class for bounded diagnostics."""
    name = type(error).__name__
    return name if name in _SAFE_ERROR_CLASSES else "Exception"


@dataclass
class OutcomeLedger:
    """Account accepted sync work and synchronous tool outcomes.

    Callers hold the provider state lock while invoking these methods.  Pending
    represents queued work and in_flight represents work removed by the worker;
    together they are the non-terminal portion of accepted work.
    """

    accepted: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    rejected: int = 0
    pending: int = 0
    in_flight: int = 0
    query_completed: int = 0
    query_failed: int = 0
    query_empty: int = 0
    extract_completed: int = 0
    extract_failed: int = 0
    extract_empty: int = 0

    def admit(self) -> None:
        self.accepted += 1
        self.pending += 1

    def reject(self) -> None:
        self.rejected += 1

    def start(self) -> None:
        if self.pending:
            self.pending -= 1
        self.in_flight += 1

    def complete(self) -> None:
        if self.in_flight:
            self.in_flight -= 1
        self.completed += 1

    def fail(self) -> None:
        if self.in_flight:
            self.in_flight -= 1
        self.failed += 1

    def drop_pending(self) -> None:
        if self.pending:
            self.pending -= 1
        self.dropped += 1

    def drop_in_flight(self) -> None:
        if self.in_flight:
            self.in_flight -= 1
        self.dropped += 1

    def record_tool(self, tool: str, *, success: bool, empty: bool = False) -> None:
        if tool == "cashew_query":
            if success:
                self.query_completed += 1
                if empty:
                    self.query_empty += 1
            else:
                self.query_failed += 1
        elif tool == "cashew_extract":
            if success:
                self.extract_completed += 1
                if empty:
                    self.extract_empty += 1
            else:
                self.extract_failed += 1

    def work_snapshot(self) -> dict[str, int | bool]:
        reconciled = self.accepted == (
            self.completed
            + self.failed
            + self.dropped
            + self.pending
            + self.in_flight
        )
        return {
            "accepted": self.accepted,
            "completed": self.completed,
            "failed": self.failed,
            "dropped": self.dropped,
            "rejected": self.rejected,
            "pending": self.pending,
            "in_flight": self.in_flight,
            "reconciled": reconciled,
        }

    def tool_snapshot(self) -> dict[str, int]:
        return {
            "query_completed": self.query_completed,
            "query_failed": self.query_failed,
            "query_empty": self.query_empty,
            "extract_completed": self.extract_completed,
            "extract_failed": self.extract_failed,
            "extract_empty": self.extract_empty,
        }

    def snapshot(self) -> dict[str, Any]:
        return {"work": self.work_snapshot(), "tools": self.tool_snapshot()}
