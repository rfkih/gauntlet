"""alternative.me Fear & Greed Index → ``macro_raw``.

Public REST API: ``https://api.alternative.me/fng/?limit=N``
- No auth, no rate limit documented (be courteous: 1 req/min in live ingest).
- Returns last N days of the daily Crypto Fear & Greed Index.
- Each entry: ``value`` (0-100, integer), ``value_classification`` (string),
  ``timestamp`` (UNIX seconds, UTC midnight of the publication day).

We write one row per day:
    series_id      = 'fear_and_greed'
    symbol         = NULL (macro feature, not symbol-scoped)
    event_time     = UTC midnight of the day the score represents
    ingestion_time = our server clock at write
    value          = the numeric score (0-100)
    value_text     = the classification ('Extreme Fear' / 'Fear' / 'Neutral' / 'Greed' / 'Extreme Greed')
    source_uri     = 'alternative.me/fng/<YYYY-MM-DD>' for dedupe
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..shared.base import IngestionRequest, IngestionResult
from ..shared.db import content_hash, get_connection, update_source_health, write_macro_raw_rows
from ..shared.pit_guards import PitConfig, partition_by_pit

logger = logging.getLogger(__name__)

name = "alternative_me"
raw_table = "macro_raw"

_API_URL = "https://api.alternative.me/fng/"
_SERIES_ID = "fear_and_greed"

# alternative.me publishes once per day. A 48h max-backfill-lag matches what
# the V67 seed has for this source.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=48)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _fetch_api(limit: int) -> dict:
    """One HTTP GET with retries. ``limit=0`` would return all history."""
    response = httpx.get(
        _API_URL,
        params={"limit": str(limit), "format": "json"},
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()


def fetch(request: IngestionRequest) -> IngestionResult:
    """Pull Fear & Greed history between request.start and request.end.

    alternative.me only supports "last N days" — there's no date-range
    parameter. We over-fetch and filter client-side.
    """
    started = time.monotonic()
    now = datetime.utcnow()

    # Compute how many days of history we need (today minus request.start).
    # Cap at a defensive max to avoid pulling years of data on accident.
    days_back = max(1, (now.date() - request.start.date()).days + 2)
    days_back = min(days_back, 4000)  # alternative.me has ~3000 days of history

    logger.info(
        "alternative_me fetch starting | start=%s end=%s days_back=%d",
        request.start,
        request.end,
        days_back,
    )

    try:
        payload = _fetch_api(days_back)
    except Exception as e:  # noqa: BLE001 — we want any failure to update health
        logger.exception("alternative_me API call failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    raw_data = payload.get("data") or []
    if not isinstance(raw_data, list):
        msg = f"alternative.me returned unexpected payload shape: {type(raw_data)}"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    # Build candidate rows ----------------------------------------------------
    candidates: list[dict] = []
    for entry in raw_data:
        try:
            value_int = int(entry["value"])
            classification = str(entry.get("value_classification") or "")
            ts_unix = int(entry["timestamp"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("alternative_me skipping malformed entry %s: %s", entry, e)
            continue

        event_time = datetime.fromtimestamp(ts_unix, tz=timezone.utc).replace(tzinfo=None)

        # Filter to the requested window.
        if event_time < request.start or event_time > request.end:
            continue

        date_str = event_time.strftime("%Y-%m-%d")
        candidates.append(
            {
                "source": name,
                "source_uri": f"alternative.me/fng/{date_str}",
                "symbol": None,
                "series_id": _SERIES_ID,
                "event_time": event_time,
                "ingestion_time": now,
                "value": value_int,
                "value_text": classification,
                "content_hash": content_hash(_SERIES_ID, date_str, value_int),
                "schema_version": 1,
            }
        )

    # PIT filter --------------------------------------------------------------
    accepted, rejected = partition_by_pit(
        candidates, config=_PIT_CONFIG, now=now, request_start=request.start
    )

    # Write -------------------------------------------------------------------
    rows_inserted = 0
    rows_skipped_duplicate = 0
    try:
        with get_connection() as conn:
            rows_inserted, rows_skipped_duplicate = write_macro_raw_rows(accepted, conn=conn)
            update_source_health(
                name,
                success=True,
                rows_inserted=rows_inserted,
                rows_rejected_pit=len(rejected),
                conn=conn,
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("alternative_me DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    result = IngestionResult(
        source=name,
        symbol=None,
        start=request.start,
        end=request.end,
        rows_fetched=len(candidates),
        rows_inserted=rows_inserted,
        rows_rejected_pit=len(rejected),
        rows_skipped_duplicate=rows_skipped_duplicate,
        series_seen=[_SERIES_ID],
        duration_seconds=duration,
        note=None if rows_inserted else "All rows already present or filtered.",
    )
    logger.info(
        "alternative_me fetch complete | fetched=%d inserted=%d skipped=%d pit_reject=%d duration=%.2fs",
        result.rows_fetched,
        result.rows_inserted,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
