"""Point-in-time integrity guards.

Every row a source produces passes through ``validate_row`` before insert.
Three rejection conditions enforced uniformly across all sources:

1. **Future event_time**: publisher's claimed event_time exceeds our server
   clock by more than ``CLOCK_SKEW_TOLERANCE_SECONDS``. Catches sources
   misreporting their timezone.

2. **Inverted publisher timestamp**: ``event_time > ingestion_time + skew``.
   Source backfilled an article and lied about when it was published.

3. **Out-of-window backfill**: a row's ``event_time`` lies more than
   ``max_backfill_lag_hours`` *before* the requested window
   (``request_start``). This catches sources that silently rescore historical
   data and ingest old event_times the operator did not ask for — the
   leakage vector the blueprint warns about — without dropping rows that
   the operator *did* explicitly request via an admin backfill.

   When ``request_start`` is not provided (legacy / live-tick callers that
   don't know their own window), we fall back to the older
   ``ingestion_time - event_time > max_backfill_lag_hours`` semantics, which
   is correct for live ingestion but throws away historical backfills.

Rejected rows are counted and returned to the caller as
``rows_rejected_pit`` so the operator can see suspicious sources in the
dashboard.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Sources can be a few seconds ahead of our clock without it meaning
# anything (NTP drift, network latency). Anything more is a real concern.
CLOCK_SKEW_TOLERANCE_SECONDS = 60


@dataclass
class PitConfig:
    """Per-source PIT thresholds. Sources pass their own instance."""

    max_backfill_lag_hours: int = 72
    clock_skew_tolerance_seconds: int = CLOCK_SKEW_TOLERANCE_SECONDS


def validate_row(
    row: dict[str, Any],
    *,
    config: PitConfig,
    now: datetime | None = None,
    request_start: datetime | None = None,
) -> tuple[bool, str | None]:
    """Apply PIT checks to a single row.

    Parameters
    ----------
    row
        The candidate row dict containing ``event_time`` and ``ingestion_time``.
    config
        Per-source thresholds.
    now
        Server clock at the start of the dispatch. Defaults to
        ``datetime.utcnow()`` when not supplied.
    request_start
        Lower bound of the operator-requested window. When provided, rule #3
        compares ``event_time`` against ``request_start`` (so deep-history
        backfills are allowed). When absent, the legacy ``now - event_time``
        lag check applies.

    Returns
    -------
    (ok, reason)
        ``ok=True`` → row should be inserted.
        ``ok=False`` → reject; ``reason`` is a short human-readable string
        suitable for logging at WARNING.
    """
    now = now or datetime.utcnow()
    event_time: datetime = row["event_time"]
    ingestion_time: datetime = row["ingestion_time"]

    # 1. Future event_time
    if event_time > now + timedelta(seconds=config.clock_skew_tolerance_seconds):
        return False, (
            f"event_time {event_time.isoformat()} > now {now.isoformat()} "
            f"+ {config.clock_skew_tolerance_seconds}s skew"
        )

    # 2. Inverted timestamps — publisher's claimed event_time is AFTER our
    # ingestion clock. Means the source backfilled an article and lied about
    # when it was published. Trust the smaller value going forward.
    if event_time > ingestion_time + timedelta(seconds=config.clock_skew_tolerance_seconds):
        return False, (
            f"event_time {event_time.isoformat()} > ingestion_time {ingestion_time.isoformat()} "
            f"+ skew (publisher backfill?)"
        )

    # 3. Out-of-window backfill — anti-PIT-poison guard.
    #
    # When request_start is known (admin backfill, scheduled tick with a
    # declared window): reject rows whose event_time predates the requested
    # window by more than max_backfill_lag_hours. A small slop window lets
    # sources legitimately publish data slightly earlier than the requested
    # start (e.g. UTC day boundaries) without rejection.
    #
    # When request_start is unknown (legacy callers): fall back to the older
    # "stale data" heuristic comparing event_time to the dispatch clock.
    slop = timedelta(hours=config.max_backfill_lag_hours)
    if request_start is not None:
        if event_time < request_start - slop:
            return False, (
                f"event_time {event_time.isoformat()} predates request_start "
                f"{request_start.isoformat()} by more than {config.max_backfill_lag_hours}h"
            )
    else:
        lag = ingestion_time - event_time
        if lag > slop:
            return False, (
                f"backfill lag {lag} exceeds {config.max_backfill_lag_hours}h threshold "
                f"(no request_start supplied)"
            )

    return True, None


def partition_by_pit(
    rows: list[dict[str, Any]],
    *,
    config: PitConfig,
    now: datetime | None = None,
    request_start: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Split a batch into accepted + rejected lists.

    Pass ``request_start`` for admin backfills / scheduled ticks with a known
    lower window bound — without it, rule #3 falls back to comparing
    ``event_time`` against ``now`` and silently rejects all historical
    backfill rows older than ``max_backfill_lag_hours``.

    Returns
    -------
    (accepted, rejected)
        ``rejected`` carries the reject reason alongside each row.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        ok, reason = validate_row(row, config=config, now=now, request_start=request_start)
        if ok:
            accepted.append(row)
        else:
            rejected.append((row, reason or "unknown"))
            logger.warning(
                "PIT reject | source=%s series=%s event_time=%s reason=%s",
                row.get("source"),
                row.get("series_id"),
                row.get("event_time"),
                reason,
            )
    return accepted, rejected
