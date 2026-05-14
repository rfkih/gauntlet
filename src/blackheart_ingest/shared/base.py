"""Source-module contract + standard result shape.

Every source under ``blackheart_ingest.sources`` implements
:class:`SourceModule`. The FastAPI dispatcher (``workers.server``) looks up
modules by name, calls ``fetch``, and forwards the result to the caller.

Design notes
------------
- Sources are *synchronous* — Postgres writes happen inside ``fetch`` via
  the shared ``db.write_raw_rows`` helper. The HTTP layer is async, but each
  source runs on a thread pool worker (FastAPI handles this transparently
  for sync ``def`` endpoints).

- Sources are *stateless* — every call must produce the same rows for the
  same ``(symbol, start, end, config)`` inputs. Idempotency lives at the DB
  layer via ``ON CONFLICT DO NOTHING`` on the natural-key unique indexes.

- Sources are *PIT-honest* — every row carries ``event_time`` (the
  publisher's timestamp) and ``ingestion_time`` (our server clock). Sources
  must reject any row where ``event_time > now + skew_tolerance``.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class IngestionRequest:
    """Parameters every source accepts. Sources interpret ``config``
    individually (FRED reads ``series_ids``, Binance reads ``feeds`` etc.).
    """

    start: datetime
    end: datetime
    symbol: str | None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result envelope returned to the Java handler via HTTP."""

    source: str
    symbol: str | None
    start: datetime
    end: datetime
    rows_fetched: int = 0
    rows_inserted: int = 0
    rows_rejected_pit: int = 0
    rows_skipped_duplicate: int = 0
    series_seen: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "rows_fetched": self.rows_fetched,
            "rows_inserted": self.rows_inserted,
            "rows_rejected_pit": self.rows_rejected_pit,
            "rows_skipped_duplicate": self.rows_skipped_duplicate,
            "series_seen": self.series_seen,
            "duration_seconds": round(self.duration_seconds, 3),
            "note": self.note,
            "stub": False,
        }


class SourceModule(Protocol):
    """Module-level interface every source under ``sources/`` exposes."""

    name: str
    """Canonical identifier matching ``ml_ingest_schedule.source``."""

    raw_table: str
    """Which ``*_raw`` table this source writes to (``macro_raw`` etc.)."""

    @abstractmethod
    def fetch(self, request: IngestionRequest) -> IngestionResult:
        """Pull data for the given window and write to the raw table."""
        ...
