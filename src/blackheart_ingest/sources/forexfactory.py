"""ForexFactory economic calendar → ``macro_raw``.

ForexFactory has no official public API. This module uses the
community-maintained faireconomy.media JSON mirror, which publishes the
current-week calendar as a small JSON array. The mirror is the most
reliable free source — direct HTML scraping of forexfactory.com runs into
CloudFlare challenges and breaks frequently.

Scope (MVP):
- Only the current-week feed is fetched. Historical backfills (request.end
  more than ~10 days in the past) return zero rows with an explanatory
  ``note``; the operator can wire a paid historical feed later without
  changing this contract.
- Filters by ``config.calendars`` (event-title keywords, OR-matched via
  the ``_TITLE_ALIASES`` map) and ``config.impact_filter`` (e.g.
  ``["High","Medium"]``).
- Emits three rows per matched event: one each for ``actual``,
  ``forecast``, ``previous``. ``value`` carries the parsed numeric (handles
  ``%``, ``K``/``M``/``B`` suffixes); ``value_text`` keeps the raw string
  so unit metadata isn't lost.

Series ids: ``forexfactory_{slug}_{metric_kind}`` — e.g.
``forexfactory_cpi_m_m_actual``, ``forexfactory_non_farm_employment_change_forecast``.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

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

name = "forexfactory"
raw_table = "macro_raw"

_THISWEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_USER_AGENT = "blackheart-ingest/0.1 (+contact via blackheart admin)"

# Matches V67 seed: max_backfill_lag_hours=168 (= one week).
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=168)

# The current-week mirror covers roughly Sun→Sat (depending on publisher's
# rollover). We allow a 10-day floor so requests landing on Mon morning still
# pick up the prior Friday's events.
_CURRENT_WEEK_FLOOR_DAYS = 10

# Operator-friendly tokens → substring matchers against event titles.
# Substring match is case-insensitive; OR semantics within a token group.
_TITLE_ALIASES: dict[str, list[str]] = {
    "FOMC": ["fomc"],
    "FED": ["fomc", "fed chair", "federal funds rate"],
    "CPI": ["cpi"],
    "PCE": ["pce"],
    "NFP": ["non-farm", "nonfarm", "nfp"],
    "UNEMPLOYMENT": ["unemployment"],
    "PMI": ["pmi"],
    "GDP": ["gdp"],
    "RETAIL": ["retail sales"],
    "ECB": ["ecb"],
    "BOJ": ["boj"],
    "PPI": ["ppi"],
}


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _http_get_json(url: str) -> Any:
    response = httpx.get(
        url,
        timeout=30.0,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def _parse_numeric(raw: Any) -> float | None:
    """Parse a metric value like ``'0.3%'``, ``'12.5K'``, ``'4.2M'``, ``'-'``
    into a float.

    Returns ``None`` for empty values, ``"-"``, ``"—"``, or unparseable
    strings. The raw string is preserved separately in ``value_text`` so
    unit suffixes aren't lost to the parser.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "—", "N/A"):
        return None
    s_clean = s.rstrip("%").replace(",", "").strip()
    multiplier = 1.0
    if s_clean.endswith(("K", "k")):
        multiplier = 1e3
        s_clean = s_clean[:-1]
    elif s_clean.endswith(("M", "m")):
        multiplier = 1e6
        s_clean = s_clean[:-1]
    elif s_clean.endswith(("B", "b")):
        multiplier = 1e9
        s_clean = s_clean[:-1]
    elif s_clean.endswith(("T", "t")):
        multiplier = 1e12
        s_clean = s_clean[:-1]
    try:
        return float(s_clean) * multiplier
    except (TypeError, ValueError):
        return None


def _matches_calendars(title: str, calendars: list[str]) -> bool:
    """Return True if any operator-requested calendar token matches.

    Empty ``calendars`` means "accept everything that passed the impact
    filter" — operator opted out of title-level filtering.
    """
    if not calendars:
        return True
    title_lower = title.lower()
    for cal in calendars:
        tokens = _TITLE_ALIASES.get(cal.upper(), [cal.lower()])
        for tok in tokens:
            if tok and tok in title_lower:
                return True
    return False


def _slugify_event(title: str) -> str:
    out: list[str] = []
    prev_under = False
    for ch in title.lower():
        if ch.isalnum():
            out.append(ch)
            prev_under = False
        elif not prev_under:
            out.append("_")
            prev_under = True
    return "".join(out).strip("_")[:80]


def _parse_event_time(raw: Any) -> datetime | None:
    """Parse ISO-8601 with offset (the mirror's format) → naive UTC."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def fetch(request: IngestionRequest) -> IngestionResult:
    config = request.config or {}
    calendars = [str(c).strip() for c in (config.get("calendars") or []) if str(c).strip()]
    impact_filter = [
        str(i).strip().lower() for i in (config.get("impact_filter") or []) if str(i).strip()
    ]

    started = time.monotonic()
    now = datetime.utcnow()

    # MVP boundary: the mirror only covers the current week. If the operator
    # asked for a window strictly before the current-week floor, return a
    # zero-row success with an explanatory note instead of failing — the
    # admin frontend should still show the dispatch as succeeded.
    week_floor = now - timedelta(days=_CURRENT_WEEK_FLOOR_DAYS)
    if request.end < week_floor:
        duration = time.monotonic() - started
        update_source_health(name, success=True, rows_inserted=0)
        return IngestionResult(
            source=name,
            symbol=None,
            start=request.start,
            end=request.end,
            rows_fetched=0,
            rows_inserted=0,
            rows_rejected_pit=0,
            rows_skipped_duplicate=0,
            series_seen=[],
            duration_seconds=duration,
            note=(
                "ForexFactory historical calendar not supported in MVP — only the "
                "faireconomy.media current-week mirror is fetched. "
                "Operator's request.end predates the current-week floor."
            ),
        )

    try:
        events = _http_get_json(_THISWEEK_URL)
    except Exception as e:  # noqa: BLE001
        logger.exception("forexfactory mirror fetch failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    if not isinstance(events, list):
        msg = f"forexfactory mirror returned unexpected shape: {type(events).__name__}"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    all_rows: list[dict[str, Any]] = []
    series_seen: list[str] = []
    filtered_count = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        title = str(ev.get("title") or "").strip()
        if not title:
            continue

        impact = str(ev.get("impact") or "").strip().lower()
        if impact_filter and impact not in impact_filter:
            filtered_count += 1
            continue
        if not _matches_calendars(title, calendars):
            filtered_count += 1
            continue

        event_time = _parse_event_time(ev.get("date"))
        if event_time is None:
            continue
        if event_time < request.start or event_time > request.end:
            continue

        slug = _slugify_event(title)
        if not slug:
            continue

        iso_ts = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        country = str(ev.get("country") or "").upper()

        for metric_kind, raw_val in (
            ("actual", ev.get("actual")),
            ("forecast", ev.get("forecast")),
            ("previous", ev.get("previous")),
        ):
            value = _parse_numeric(raw_val)
            value_text = (
                str(raw_val).strip()
                if raw_val not in (None, "") and str(raw_val).strip() not in ("-", "—")
                else None
            )
            if value is None and value_text is None:
                continue

            series_id = f"forexfactory_{slug}_{metric_kind}"
            all_rows.append(
                {
                    "source": name,
                    "source_uri": f"forexfactory/{country}/{slug}/{metric_kind}/{iso_ts}",
                    "symbol": None,
                    "series_id": series_id,
                    "event_time": event_time,
                    "ingestion_time": now,
                    "value": value,
                    "value_text": value_text,
                    "content_hash": content_hash(
                        series_id, iso_ts, value, value_text, impact
                    ),
                    "schema_version": 1,
                }
            )
            if series_id not in series_seen:
                series_seen.append(series_id)

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
        logger.exception("forexfactory DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    note = (
        f"Filtered {filtered_count} events by impact/calendar config. "
        f"MVP: current-week mirror only; historical lookups deferred."
    )
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
        note=note,
    )
    logger.info(
        "forexfactory fetch complete | series=%d fetched=%d inserted=%d filtered=%d "
        "skipped=%d pit_reject=%d duration=%.2fs",
        len(series_seen),
        result.rows_fetched,
        result.rows_inserted,
        filtered_count,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
