"""FRED + ALFRED → ``macro_raw``.

FRED is the St. Louis Fed's macro data API. ALFRED is its vintage archive —
returns each series ``as it was known on a given date``, including all
revisions. Critical for backtest honesty: CPI/M2/GDP get revised 1-3
times after first print, and using current-revised values in historical
training is a silent leakage source.

Series we pull (driven by ``config["series_ids"]``):
    DTWEXBGS  trade-weighted USD index (no revisions)
    DFII10    10-year TIPS yield, real (no revisions)
    DGS10     10-year Treasury constant maturity
    DGS2      2-year Treasury constant maturity
    T10Y2Y    2s10s spread
    VIXCLS    VIX close
    M2SL      M2 money supply (REVISED — use ALFRED)
    CPIAUCSL  CPI all urban (REVISED — use ALFRED)

When ``config["use_alfred_vintage"]`` is true we use the ALFRED vintage API
for revision-prone series and the regular FRED API for the rest. The
``content_hash`` for ALFRED rows includes the vintage date so a series can
hold multiple revisions side-by-side.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any

import pandas as pd
from fredapi import Fred
from tenacity import retry, stop_after_attempt, wait_exponential

from ..shared.base import IngestionRequest, IngestionResult
from ..shared.db import content_hash, get_connection, update_source_health, write_macro_raw_rows
from ..shared.pit_guards import PitConfig, partition_by_pit
from ..shared.settings import get_settings

logger = logging.getLogger(__name__)

name = "fred"
raw_table = "macro_raw"

# Series for which we ALWAYS use ALFRED vintage when use_alfred_vintage=true.
# Series not in this set use regular FRED (no revisions, vintage is identical
# to current).
_REVISED_SERIES: frozenset[str] = frozenset(
    {
        "M2SL",
        "M2",
        "CPIAUCSL",
        "GDPC1",
        "GDP",
        "UNRATE",
        "PAYEMS",  # nonfarm payrolls — heavily revised
        "PCE",
        "PCEC96",
    }
)

# PIT publication lag — days between a value's observation_date (the date
# the data describes) and the date it was first publicly released.
# fredapi.get_series_first_release returns values keyed by observation_date,
# so we shift event_time forward by this lag to model "when this value
# became publicly known". A backtest joining on event_time <= bar.start_time
# would otherwise leak future macro releases backward.
#
# Values are conservative (err high) — actual first-release timing varies
# month-to-month. For series not in this map, lag=0 means the value is
# considered public on its observation_date (true for DXY, VIX, daily yields).
#
# Sources for the lag estimates:
#   - CPIAUCSL  ~ 14d   BLS Consumer Price Index, mid-month release
#   - M2SL      ~ 28d   H.6 monthly money supply
#   - GDPC1/GDP ~ 28d   BEA advance estimate, end of next month
#   - UNRATE    ~ 7d    BLS Employment Situation, first Friday of next month
#   - PAYEMS    ~ 7d    same release as UNRATE
#   - PCE/PCEC96 ~ 28d  BEA Personal Income & Outlays, end of next month
_PUBLICATION_LAG_DAYS: dict[str, int] = {
    "CPIAUCSL": 14,
    "M2SL": 28,
    "M2": 28,
    "GDPC1": 28,
    "GDP": 28,
    "UNRATE": 7,
    "PAYEMS": 7,
    "PCE": 28,
    "PCEC96": 28,
}

# FRED publishes most series same-day after release. 72h max-backfill-lag is
# generous — anything older means we missed a tick. Matches V67 seed config.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=72)


def _publication_lag(series_id: str) -> int:
    return _PUBLICATION_LAG_DAYS.get(series_id.upper(), 0)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
)
def _fred_get_series(client: Fred, series_id: str, start: date, end: date) -> pd.Series:
    """Wrap fredapi to give us retries with backoff."""
    return client.get_series(series_id, observation_start=start, observation_end=end)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
)
def _alfred_get_series_first_release(
    client: Fred, series_id: str, start: date, end: date
) -> pd.Series:
    """Return the *first release* values from ALFRED. Effectively what
    market participants knew on the original publication date.

    Uses a boolean-mask filter rather than ``[start:end]`` slicing because
    pandas date-string slicing on a DatetimeIndex is fragile across
    versions (pandas 2.x deprecated some forms).
    """
    series = client.get_series_first_release(series_id)
    if series is None or series.empty:
        return series
    idx_dates = series.index.date
    mask = (idx_dates >= start) & (idx_dates <= end)
    return series[mask]


def _build_rows(
    series_id: str,
    series_data: pd.Series,
    *,
    use_vintage: bool,
    now: datetime,
) -> list[dict[str, Any]]:
    """Build raw rows for a series.

    For revised series, ``event_time`` is shifted forward by
    ``_publication_lag(series_id)`` days. The observation_date stays
    visible via ``value_text`` so the operator can trace the lag.
    """
    lag_days = _publication_lag(series_id)
    rows: list[dict[str, Any]] = []
    for ts, value in series_data.items():
        if pd.isna(value):
            continue
        # FRED returns pandas Timestamps. Convert to naive datetime UTC.
        if isinstance(ts, pd.Timestamp):
            observation_time = ts.to_pydatetime().replace(tzinfo=None)
        elif isinstance(ts, date):
            observation_time = datetime.combine(ts, dt_time(0, 0, 0))
        else:
            observation_time = datetime.fromisoformat(str(ts))

        # PIT shift: event_time = "when was this value publicly known".
        # For series with publication lag (CPI, M2, GDP, etc.) this is later
        # than the observation date. For real-time series (DXY, VIX) lag=0.
        event_time = observation_time + timedelta(days=lag_days)

        obs_date_str = observation_time.strftime("%Y-%m-%d")
        # ALFRED-vintage rows get a different source_uri so they can coexist
        # with regular FRED rows for the same date — useful when comparing
        # first-release vs current-revised in backtest research.
        suffix = "alfred" if use_vintage else "fred"
        rows.append(
            {
                "source": name,
                "source_uri": f"fred.stlouisfed.org/{series_id}/{obs_date_str}/{suffix}",
                "symbol": None,
                "series_id": series_id,
                "event_time": event_time,
                "ingestion_time": now,
                "value": float(value),
                # Persist observation_date in value_text so the lag is
                # auditable from the row alone. Format: "obs=YYYY-MM-DD lag=Nd"
                "value_text": f"obs={obs_date_str} lag={lag_days}d",
                "content_hash": content_hash(series_id, obs_date_str, suffix, float(value)),
                "schema_version": 1,
            }
        )
    return rows


def fetch(request: IngestionRequest) -> IngestionResult:
    settings = get_settings()
    if not settings.fred_api_key:
        msg = "INGEST_FRED_API_KEY not configured — refusing to call FRED unauthenticated."
        update_source_health(name, success=False, error_message=msg)
        raise RuntimeError(msg)

    config = request.config or {}
    series_ids: list[str] = list(config.get("series_ids") or [])
    if not series_ids:
        msg = "fred requires non-empty config.series_ids"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    use_vintage_global: bool = bool(config.get("use_alfred_vintage", False))

    started = time.monotonic()
    now = datetime.utcnow()

    client = Fred(api_key=settings.fred_api_key)

    all_rows: list[dict[str, Any]] = []
    series_seen: list[str] = []

    for series_id in series_ids:
        use_vintage = use_vintage_global and series_id.upper() in _REVISED_SERIES

        logger.info(
            "fred fetching series=%s vintage=%s start=%s end=%s",
            series_id,
            use_vintage,
            request.start.date(),
            request.end.date(),
        )

        try:
            if use_vintage:
                series_data = _alfred_get_series_first_release(
                    client, series_id, request.start.date(), request.end.date()
                )
            else:
                series_data = _fred_get_series(
                    client, series_id, request.start.date(), request.end.date()
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("fred series fetch failed | series=%s", series_id)
            update_source_health(
                name,
                success=False,
                error_message=f"series {series_id}: {str(e)[:400]}",
            )
            raise

        rows = _build_rows(series_id, series_data, use_vintage=use_vintage, now=now)
        all_rows.extend(rows)
        series_seen.append(series_id)

    # PIT filter --------------------------------------------------------------
    accepted, rejected = partition_by_pit(
        all_rows, config=_PIT_CONFIG, now=now, request_start=request.start
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
        logger.exception("fred DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    result = IngestionResult(
        source=name,
        symbol=None,
        start=request.start,
        end=request.end,
        rows_fetched=len(all_rows),
        rows_inserted=rows_inserted,
        rows_rejected_pit=len(rejected),
        rows_skipped_duplicate=rows_skipped_duplicate,
        series_seen=series_seen,
        duration_seconds=duration,
        note=(
            f"ALFRED vintage used for revision-prone series: "
            f"{sorted(s for s in series_seen if s.upper() in _REVISED_SERIES and use_vintage_global)}"
            if use_vintage_global
            else None
        ),
    )
    logger.info(
        "fred fetch complete | series=%d fetched=%d inserted=%d skipped=%d pit_reject=%d duration=%.2fs",
        len(series_seen),
        result.rows_fetched,
        result.rows_inserted,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
