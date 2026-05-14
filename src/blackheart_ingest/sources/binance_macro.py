"""Binance public futures macro feeds → ``macro_raw``.

Four feeds, all unauthenticated, served from ``fapi.binance.com``:

1. **Funding rate** (``feeds=["funding_rate"]``):
   ``/fapi/v1/fundingRate?symbol=…`` — every-8h funding payments. No
   ``period`` parameter (native cadence).

   Series: ``binance_funding_rate_{symbol_lower}``

2. **Open interest** (``feeds=["open_interest"]``):
   ``/futures/data/openInterestHist?symbol=…&period=…`` — aggregate open
   interest. We store the USD-value (``sumOpenInterestValue``) for
   cross-coin comparability.

   Series: ``binance_open_interest_{symbol_lower}_{period}``

3. **Top trader long/short ratio** (``feeds=["top_long_short_ratio"]``):
   ``/futures/data/topLongShortAccountRatio?symbol=…&period=…`` — % of top
   accounts long vs short. Account-weighted (not position-weighted) since
   that's the convention most desks track.

   Series: ``binance_long_short_ratio_{symbol_lower}_{period}``

4. **Taker buy/sell volume ratio** (``feeds=["taker_buy_sell"]``):
   ``/futures/data/takerlongshortRatio?symbol=…&period=…`` — aggressor
   side imbalance. >1 = net taker-buy pressure.

   Series: ``binance_taker_buy_sell_ratio_{symbol_lower}_{period}``

Pagination: each feed is walked forward via ``startTime`` until the API
returns an empty page or the cursor advances past ``end_ms``. Defensive
cap of 50 pages per (feed, period) tuple guards against runaway loops.

PIT note: Binance reports each row with the exchange's UTC timestamp.
Cadence is fixed by the publisher — there's no historical revision —
so the 24h slop in ``_PIT_CONFIG`` exists only to tolerate operator
boundary mistakes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

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

name = "binance_macro"
raw_table = "macro_raw"

_FAPI_BASE = "https://fapi.binance.com"

# Per-endpoint pagination limits from Binance docs.
_FUNDING_RATE_LIMIT = 1000
_FUTURES_DATA_LIMIT = 500
_PAGINATION_CAP = 50  # 50 pages × max 500 rows = 25k rows per (feed, period)

_VALID_FEEDS: frozenset[str] = frozenset(
    {"funding_rate", "open_interest", "top_long_short_ratio", "taker_buy_sell"}
)
_PERIOD_FEEDS: frozenset[str] = frozenset(
    {"open_interest", "top_long_short_ratio", "taker_buy_sell"}
)
_VALID_PERIODS: frozenset[str] = frozenset(
    {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
)

# Binance only retains the last 30 days of /futures/data/* statistics
# (open interest, long/short ratio, taker buy-sell). Requests for older
# windows return HTTP 400. Funding rate (/fapi/v1/fundingRate) has full
# history per contract, so it's exempt from this clamp.
#
# Effective window is shaved by an hour (29d 23h) because Binance enforces
# the limit against its own server clock at request time, not against the
# endTime we send — so an exact 30d span computed from a slightly stale
# ``now`` rolls just over the boundary and gets rejected.
_FUTURES_DATA_RETENTION_DAYS = 30
_FUTURES_DATA_RETENTION_BUFFER_SECONDS = 3600

# Matches V67 seed: max_backfill_lag_hours=24 for binance_macro.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=24)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _http_get(client: httpx.Client, path: str, params: dict[str, Any]) -> Any:
    response = client.get(
        f"{_FAPI_BASE}{path}",
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_ms(dt: datetime) -> int:
    # Treat naive datetimes as UTC (project convention).
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _paginate(
    client: httpx.Client,
    path: str,
    base_params: dict[str, Any],
    *,
    start_ms: int,
    end_ms: int,
    limit: int,
    ts_key: str,
) -> Iterable[list[dict[str, Any]]]:
    """Walk forward through a Binance time-series endpoint.

    Stops when:
    - the API returns a non-list or empty page;
    - the last row's timestamp doesn't advance the cursor (guards against
      stuck pagination);
    - the page cap is reached.
    """
    cursor = start_ms
    pages = 0
    while cursor < end_ms and pages < _PAGINATION_CAP:
        params = {
            **base_params,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        page = _http_get(client, path, params)
        if not isinstance(page, list) or not page:
            return
        yield page
        last_ts = page[-1].get(ts_key)
        if not isinstance(last_ts, (int, float)) or int(last_ts) <= cursor:
            return
        cursor = int(last_ts) + 1
        pages += 1
    if pages >= _PAGINATION_CAP:
        logger.warning(
            "binance_macro pagination cap reached at page=%d path=%s — truncating",
            pages,
            path,
        )


def _fetch_funding_rate(
    client: httpx.Client,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    now: datetime,
) -> tuple[list[dict[str, Any]], str]:
    series_id = f"binance_funding_rate_{symbol.lower()}"
    rows: list[dict[str, Any]] = []
    for page in _paginate(
        client,
        "/fapi/v1/fundingRate",
        {"symbol": symbol},
        start_ms=start_ms,
        end_ms=end_ms,
        limit=_FUNDING_RATE_LIMIT,
        ts_key="fundingTime",
    ):
        for entry in page:
            if not isinstance(entry, dict):
                continue
            ts_ms = entry.get("fundingTime")
            rate = _to_float(entry.get("fundingRate"))
            if rate is None or not isinstance(ts_ms, (int, float)):
                continue
            event_time = _from_ms(int(ts_ms))
            iso_ts = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append(
                {
                    "source": name,
                    "source_uri": f"binance/fundingRate/{symbol}/{iso_ts}",
                    "symbol": symbol,
                    "series_id": series_id,
                    "event_time": event_time,
                    "ingestion_time": now,
                    "value": rate,
                    "value_text": None,
                    "content_hash": content_hash(series_id, iso_ts, rate),
                    "schema_version": 1,
                }
            )
    return rows, series_id


def _fetch_period_series(
    client: httpx.Client,
    *,
    path: str,
    symbol: str,
    period: str,
    series_id: str,
    value_key: str,
    source_tag: str,
    start_ms: int,
    end_ms: int,
    now: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in _paginate(
        client,
        path,
        {"symbol": symbol, "period": period},
        start_ms=start_ms,
        end_ms=end_ms,
        limit=_FUTURES_DATA_LIMIT,
        ts_key="timestamp",
    ):
        for entry in page:
            if not isinstance(entry, dict):
                continue
            ts_ms = entry.get("timestamp")
            value = _to_float(entry.get(value_key))
            if value is None or not isinstance(ts_ms, (int, float)):
                continue
            event_time = _from_ms(int(ts_ms))
            iso_ts = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append(
                {
                    "source": name,
                    "source_uri": f"binance/{source_tag}/{symbol}/{period}/{iso_ts}",
                    "symbol": symbol,
                    "series_id": series_id,
                    "event_time": event_time,
                    "ingestion_time": now,
                    "value": value,
                    "value_text": f"period={period}",
                    "content_hash": content_hash(series_id, iso_ts, value),
                    "schema_version": 1,
                }
            )
    return rows


def fetch(request: IngestionRequest) -> IngestionResult:
    config = request.config or {}
    feeds: list[str] = [
        str(f).strip().lower() for f in (config.get("feeds") or []) if str(f).strip()
    ]
    intervals: list[str] = [
        str(i).strip().lower() for i in (config.get("intervals") or []) if str(i).strip()
    ]

    if not feeds:
        msg = "binance_macro requires non-empty config.feeds"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    unknown_feeds = [f for f in feeds if f not in _VALID_FEEDS]
    if unknown_feeds:
        msg = (
            f"binance_macro unknown feeds: {unknown_feeds}. "
            f"Valid: {sorted(_VALID_FEEDS)}"
        )
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    invalid_periods = [p for p in intervals if p not in _VALID_PERIODS]
    if invalid_periods:
        msg = (
            f"binance_macro unknown intervals: {invalid_periods}. "
            f"Valid: {sorted(_VALID_PERIODS)}"
        )
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    needs_period = any(f in _PERIOD_FEEDS for f in feeds)
    if needs_period and not intervals:
        msg = (
            "binance_macro period feeds (open_interest, long_short_ratio, taker_buy_sell) "
            "require non-empty config.intervals (e.g. ['1h','4h'])"
        )
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    if not request.symbol:
        msg = "binance_macro requires request.symbol (e.g. BTCUSDT) — schedule must set it"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)
    symbol = request.symbol.upper()

    started = time.monotonic()
    now = datetime.utcnow()
    start_ms = _to_ms(request.start)
    end_ms = _to_ms(request.end)

    # /futures/data/* endpoints retain only ~30d of history. Anything older
    # returns HTTP 400. Funding rate keeps its full window.
    #
    # We also clamp endTime to ``now`` — Binance rejects future-anchored
    # windows because their server clock sees them as out-of-range. Combined
    # with the 1-hour buffer, this keeps us safely inside the retention
    # boundary regardless of the operator's chosen end_ms.
    now_ms = _to_ms(now)
    period_end_ms = min(end_ms, now_ms)
    retention_floor_ms = period_end_ms - (
        _FUTURES_DATA_RETENTION_DAYS * 86400 - _FUTURES_DATA_RETENTION_BUFFER_SECONDS
    ) * 1000
    period_start_ms = max(start_ms, retention_floor_ms)
    period_window_clamped = period_start_ms > start_ms or period_end_ms < end_ms

    if period_window_clamped:
        logger.info(
            "binance_macro period feeds clamped to last %dd "
            "(Binance /futures/data/* retention limit); funding rate keeps full window",
            _FUTURES_DATA_RETENTION_DAYS,
        )

    all_rows: list[dict[str, Any]] = []
    series_seen: list[str] = []

    def _record(rows: list[dict[str, Any]], series_id: str) -> None:
        all_rows.extend(rows)
        if series_id not in series_seen:
            series_seen.append(series_id)

    try:
        with httpx.Client(headers={"Accept": "application/json"}) as client:
            if "funding_rate" in feeds:
                logger.info(
                    "binance_macro fetching funding_rate | symbol=%s window=%s→%s",
                    symbol, request.start, request.end,
                )
                rows, sid = _fetch_funding_rate(
                    client, symbol, start_ms=start_ms, end_ms=end_ms, now=now
                )
                _record(rows, sid)

            for period in intervals:
                if "open_interest" in feeds:
                    logger.info(
                        "binance_macro fetching open_interest | symbol=%s period=%s",
                        symbol, period,
                    )
                    sid = f"binance_open_interest_{symbol.lower()}_{period}"
                    rows = _fetch_period_series(
                        client,
                        path="/futures/data/openInterestHist",
                        symbol=symbol,
                        period=period,
                        series_id=sid,
                        # USD-denominated open interest for cross-instrument compare.
                        value_key="sumOpenInterestValue",
                        source_tag="oiHist",
                        start_ms=period_start_ms,
                        end_ms=period_end_ms,
                        now=now,
                    )
                    _record(rows, sid)

                if "top_long_short_ratio" in feeds:
                    logger.info(
                        "binance_macro fetching top_long_short_ratio | symbol=%s period=%s",
                        symbol, period,
                    )
                    sid = f"binance_long_short_ratio_{symbol.lower()}_{period}"
                    rows = _fetch_period_series(
                        client,
                        path="/futures/data/topLongShortAccountRatio",
                        symbol=symbol,
                        period=period,
                        series_id=sid,
                        value_key="longShortRatio",
                        source_tag="lsAccountRatio",
                        start_ms=period_start_ms,
                        end_ms=period_end_ms,
                        now=now,
                    )
                    _record(rows, sid)

                if "taker_buy_sell" in feeds:
                    logger.info(
                        "binance_macro fetching taker_buy_sell | symbol=%s period=%s",
                        symbol, period,
                    )
                    sid = f"binance_taker_buy_sell_ratio_{symbol.lower()}_{period}"
                    rows = _fetch_period_series(
                        client,
                        path="/futures/data/takerlongshortRatio",
                        symbol=symbol,
                        period=period,
                        series_id=sid,
                        value_key="buySellRatio",
                        source_tag="takerRatio",
                        start_ms=period_start_ms,
                        end_ms=period_end_ms,
                        now=now,
                    )
                    _record(rows, sid)
    except Exception as e:  # noqa: BLE001
        logger.exception("binance_macro fetch failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    # PIT filter ────────────────────────────────────────────────────────────
    accepted, rejected = partition_by_pit(
        all_rows, config=_PIT_CONFIG, now=now, request_start=request.start
    )

    # Write ─────────────────────────────────────────────────────────────────
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
        logger.exception("binance_macro DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    note_parts: list[str] = [f"feeds={feeds} intervals={intervals}"]
    if period_window_clamped:
        note_parts.append(
            f"Period feeds (OI/LSR/Taker) clamped to last {_FUTURES_DATA_RETENTION_DAYS}d — "
            f"Binance retention limit. Funding rate window unchanged."
        )
    result = IngestionResult(
        source=name,
        symbol=symbol,
        start=request.start,
        end=request.end,
        rows_fetched=len(all_rows),
        rows_inserted=rows_inserted,
        rows_rejected_pit=len(rejected),
        rows_skipped_duplicate=rows_skipped_duplicate,
        series_seen=series_seen,
        duration_seconds=duration,
        note=" | ".join(note_parts),
    )
    logger.info(
        "binance_macro fetch complete | symbol=%s feeds=%s intervals=%s "
        "series=%d fetched=%d inserted=%d skipped=%d pit_reject=%d duration=%.2fs",
        symbol, feeds, intervals, len(series_seen),
        result.rows_fetched, result.rows_inserted,
        result.rows_skipped_duplicate, result.rows_rejected_pit, result.duration_seconds,
    )
    return result
