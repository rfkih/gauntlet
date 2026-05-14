"""CoinMetrics Community API (free tier) → ``macro_raw``.

CoinMetrics offers a free community-API tier with daily-cadence on-chain
metrics. We use it for the canonical on-chain "smart-money" features —
exchange flows, active addresses, transaction count, realized cap, market
cap — that the V67 seed config requests.

Endpoint:
    GET ``https://community-api.coinmetrics.io/v4/timeseries/asset-metrics``

Query parameters (all free for community tier):
    assets         comma-sep asset slugs (``btc``, ``eth``)
    metrics        comma-sep metric ids (``FlowOutNative``, ``AdrActCnt``, ...)
    frequency      ``1d`` (daily — only granularity free)
    start_time     ISO date or datetime, UTC
    end_time       ISO date or datetime, UTC
    page_size      max rows per page (10000 free)

Response shape::

    {"data": [{"asset": "btc", "time": "2024-01-01T00:00:00.000000000Z",
                "FlowOutNative": "12345.6", ...}, ...]}

We emit one row per (asset, metric, time):
    series_id = "coinmetrics_{asset}_{metric_lower}"
    e.g.       "coinmetrics_btc_flowoutnative"
               "coinmetrics_eth_adractcnt"

The symbol column maps to our trading symbol convention when the metric
references the asset we trade (``BTCUSDT`` for ``btc``); for assets we
don't trade (none today) we'd set NULL. This makes downstream feature
joins straightforward.

PIT note: CoinMetrics computes daily metrics at UTC end-of-day and
publishes within a few hours. The 72h max-backfill-lag from V67 leaves
generous slack for both publication delay and any operator-induced gaps.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
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

name = "coinmetrics"
raw_table = "macro_raw"

_BASE_URL = "https://community-api.coinmetrics.io/v4"

# Map our trading symbols to CoinMetrics asset slugs. Extend when new
# instruments are plumbed in.
_SYMBOL_TO_ASSET: dict[str, str] = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
}

# Best-effort allow-list of community-tier (free) metrics. CoinMetrics
# rejects the entire batch with 403 if any requested metric is paid-only —
# we surface a clearer error rather than passing the raw HTTPStatusError up
# the stack. Not authoritative (CoinMetrics revisits their free catalog
# periodically); kept here only to compose the diagnostic.
_KNOWN_PAID_METRICS: frozenset[str] = frozenset(
    {
        "FlowInNative",
        "FlowOutNative",
        "FlowInUSD",
        "FlowOutUSD",
        "CapRealUSD",
        "NVTAdj",
        "NVTAdj90",
        "AdrBalUSD1Cnt",
        "AdrBalUSD10Cnt",
        "AdrBalUSD100Cnt",
        "AdrBalUSD1KCnt",
        "AdrBalUSD10KCnt",
        "AdrBalUSD100KCnt",
        "AdrBalUSD1MCnt",
    }
)

# Matches V67 seed: max_backfill_lag_hours=72 for coinmetrics.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=72)


class CoinMetricsPaidTierError(RuntimeError):
    """Raised when CoinMetrics returns 403 — at least one requested metric
    is outside the free community-tier catalog. We surface this distinctly
    so the operator sees a config problem, not a generic server error.
    """


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _http_get(client: httpx.Client, path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.get(
            f"{_BASE_URL}{path}",
            params=params,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            # Don't retry tier-violation 403s — they will never succeed without
            # config changes. Re-raise as a domain-specific error so the
            # caller can format an actionable message.
            requested = params.get("metrics", "")
            suspect = [
                m.strip()
                for m in str(requested).split(",")
                if m.strip() in _KNOWN_PAID_METRICS
            ]
            raise CoinMetricsPaidTierError(
                "CoinMetrics returned 403. The community-tier free API does not "
                "expose all metrics. "
                + (
                    f"Suspected paid-only metric(s) in your request: {suspect}. "
                    if suspect
                    else "Could not pre-identify which metric is paid-only. "
                )
                + "Edit the coinmetrics schedule config to drop them. "
                "Known-free starter set: ['AdrActCnt','TxCnt','CapMrktCurUSD','BlkCnt','HashRate']."
            ) from e
        raise
    return response.json()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_cm_time(raw: str) -> datetime | None:
    """CoinMetrics returns nanosecond-precision UTC ISO timestamps like
    ``2024-01-01T00:00:00.000000000Z``. Python's fromisoformat handles
    microsecond precision, so we truncate ns → µs and swap Z → +00:00.
    """
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.replace("Z", "+00:00")
    # Truncate >6 fractional digits (Python max). Find the dot and slice.
    if "." in s:
        head, tail = s.split(".", 1)
        # tail looks like "000000000+00:00" — split off the tz portion.
        if "+" in tail:
            frac, tz = tail.split("+", 1)
            frac = frac[:6]
            s = f"{head}.{frac}+{tz}"
        elif "-" in tail:
            frac, tz = tail.rsplit("-", 1)
            frac = frac[:6]
            s = f"{head}.{frac}-{tz}"
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return None


def _fetch_metrics_page(
    client: httpx.Client,
    assets: list[str],
    metrics: list[str],
    *,
    start: datetime,
    end: datetime,
    next_page_token: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "assets": ",".join(assets),
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "page_size": "10000",
    }
    if next_page_token:
        params["next_page_token"] = next_page_token
    return _http_get(client, "/timeseries/asset-metrics", params)


def fetch(request: IngestionRequest) -> IngestionResult:
    config = request.config or {}
    metrics: list[str] = [
        str(m).strip() for m in (config.get("metrics") or []) if str(m).strip()
    ]
    if not metrics:
        msg = "coinmetrics requires non-empty config.metrics"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    # Resolve which assets to pull. Honor request.symbol if set; otherwise
    # default to all trading symbols we know how to map.
    if request.symbol:
        asset = _SYMBOL_TO_ASSET.get(request.symbol.upper())
        if asset is None:
            msg = (
                f"coinmetrics has no asset mapping for symbol={request.symbol}. "
                f"Known: {sorted(_SYMBOL_TO_ASSET)}"
            )
            update_source_health(name, success=False, error_message=msg)
            raise ValueError(msg)
        assets = [asset]
        symbol_for_rows: str | None = request.symbol.upper()
    else:
        assets = list(_SYMBOL_TO_ASSET.values())
        symbol_for_rows = None  # multi-asset run; per-row symbol resolved below

    # Asset slug → trading symbol (reverse map) so we can stamp symbol per row.
    asset_to_symbol = {v: k for k, v in _SYMBOL_TO_ASSET.items()}

    started = time.monotonic()
    now = datetime.utcnow()

    all_rows: list[dict[str, Any]] = []
    metrics_lower = [m.lower() for m in metrics]
    series_seen: list[str] = []

    next_page_token: str | None = None
    pages = 0
    try:
        with httpx.Client(headers={"Accept": "application/json"}) as client:
            while True:
                pages += 1
                payload = _fetch_metrics_page(
                    client,
                    assets,
                    metrics,
                    start=request.start,
                    end=request.end,
                    next_page_token=next_page_token,
                )
                data = payload.get("data") or []
                if not isinstance(data, list):
                    raise ValueError(
                        f"coinmetrics returned unexpected data shape: {type(data)}"
                    )

                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    asset_slug = str(entry.get("asset") or "").lower()
                    if not asset_slug:
                        continue
                    event_time = _parse_cm_time(entry.get("time"))
                    if event_time is None:
                        continue
                    if event_time < request.start or event_time > request.end:
                        continue

                    row_symbol = symbol_for_rows or asset_to_symbol.get(asset_slug)

                    for metric_id, metric_lc in zip(metrics, metrics_lower):
                        value = _to_float(entry.get(metric_id))
                        if value is None:
                            continue
                        series_id = f"coinmetrics_{asset_slug}_{metric_lc}"
                        iso_ts = event_time.strftime("%Y-%m-%d")
                        all_rows.append(
                            {
                                "source": name,
                                "source_uri": f"coinmetrics/{asset_slug}/{metric_id}/{iso_ts}",
                                "symbol": row_symbol,
                                "series_id": series_id,
                                "event_time": event_time,
                                "ingestion_time": now,
                                "value": value,
                                "value_text": metric_id,
                                "content_hash": content_hash(series_id, iso_ts, value),
                                "schema_version": 1,
                            }
                        )
                        if series_id not in series_seen:
                            series_seen.append(series_id)

                next_page_token = payload.get("next_page_token") or None
                if not next_page_token:
                    break
                # Defensive: cap pagination at 20 pages (200k rows) per dispatch.
                if pages >= 20:
                    logger.warning(
                        "coinmetrics pagination cap reached at page=%d — truncating",
                        pages,
                    )
                    break
    except Exception as e:  # noqa: BLE001
        logger.exception("coinmetrics fetch failed | pages_seen=%d", pages)
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
        logger.exception("coinmetrics DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    result = IngestionResult(
        source=name,
        symbol=symbol_for_rows,
        start=request.start,
        end=request.end,
        rows_fetched=len(all_rows),
        rows_inserted=rows_inserted,
        rows_rejected_pit=len(rejected),
        rows_skipped_duplicate=rows_skipped_duplicate,
        series_seen=series_seen,
        duration_seconds=duration,
        note=f"Fetched across {pages} page(s); assets={assets}",
    )
    logger.info(
        "coinmetrics fetch complete | assets=%s series=%d fetched=%d inserted=%d "
        "skipped=%d pit_reject=%d duration=%.2fs",
        assets,
        len(series_seen),
        result.rows_fetched,
        result.rows_inserted,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
