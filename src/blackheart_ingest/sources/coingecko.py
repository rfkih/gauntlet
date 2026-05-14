"""CoinGecko Public API → ``macro_raw``.

Two payloads ingested per dispatch:

1. **Global snapshot** (``config.global_metrics=true``, default ON):
   GET ``https://api.coingecko.com/api/v3/global`` → one row per series_id
   at the current ``updated_at`` timestamp.

   The snapshot endpoint has no historical view — it always returns the
   present moment. We therefore only emit it when the request window is
   "now-ish" (``request.end >= now - 1h``). Pure-historical backfills skip
   the snapshot entirely, since otherwise a backfill for 2024-01 would
   silently land a `now`-stamped row outside the requested window.

   Series:
     - ``total_market_cap_usd``  total crypto market cap in USD
     - ``total_volume_usd``      24h global trading volume in USD
     - ``btc_dominance_pct``     BTC share of total market cap (%)
     - ``eth_dominance_pct``     ETH share of total market cap (%)

2. **Per-coin history** (``config.per_coin=["bitcoin","ethereum"]``):
   GET ``/coins/{id}/market_chart?vs_currency=usd&days=N`` → time series of
   price / market_cap / volume.

   Series written per coin (e.g. for ``bitcoin``):
     - ``bitcoin_price_usd``
     - ``bitcoin_market_cap_usd``
     - ``bitcoin_volume_usd``

CoinGecko free tier rules (as of 2026):
- 5–30 req/min depending on load; no auth required.
- Granularity is auto-chosen by ``days``: ``days=1`` → 5-min, ``2–90`` →
  hourly, ``>=91`` → daily (UTC 00:00). The ``interval`` parameter is gated
  behind paid tiers — we accept whatever resolution comes back and let the
  feature_store roll up.
- ``days`` is capped at 365 on the public tier.

PIT note: CoinGecko publishes near-realtime. The 12h PIT slop from V67
allows for UTC boundary effects between the request window and the
publisher's actual sample times without rejection.
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

name = "coingecko"
raw_table = "macro_raw"

_BASE_URL = "https://api.coingecko.com/api/v3"
_FREE_TIER_DAYS_CAP = 365

# Matches V67 seed: max_backfill_lag_hours=12. Used as PIT slop tolerance
# against the operator's request.start (V69+ semantics) rather than a hard
# cutoff against ``now``.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=12)

# When request.end is older than this, treat the dispatch as a pure
# historical backfill — skip the global snapshot which only carries
# "now" data.
_SNAPSHOT_RECENCY_HOURS = 1


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _http_get(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(
        f"{_BASE_URL}{path}",
        params=params or {},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _fetch_global_snapshot(client: httpx.Client, now: datetime) -> list[dict[str, Any]]:
    """One snapshot row per series at the publisher's ``updated_at`` time."""
    payload = _http_get(client, "/global")
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError(f"coingecko /global returned unexpected shape: {type(data)}")

    # updated_at is UNIX seconds (UTC). Falls back to now if missing.
    ts_unix = data.get("updated_at")
    if isinstance(ts_unix, (int, float)) and ts_unix > 0:
        event_time = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).replace(tzinfo=None)
    else:
        event_time = now

    total_mc = (data.get("total_market_cap") or {}).get("usd")
    total_vol = (data.get("total_volume") or {}).get("usd")
    dominance = data.get("market_cap_percentage") or {}
    btc_dom = dominance.get("btc")
    eth_dom = dominance.get("eth")

    series_values: list[tuple[str, float | None]] = [
        ("total_market_cap_usd", _to_float(total_mc)),
        ("total_volume_usd", _to_float(total_vol)),
        ("btc_dominance_pct", _to_float(btc_dom)),
        ("eth_dominance_pct", _to_float(eth_dom)),
    ]

    rows: list[dict[str, Any]] = []
    iso_ts = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    for series_id, value in series_values:
        if value is None:
            logger.warning("coingecko /global missing value for %s — skipping row", series_id)
            continue
        rows.append(
            {
                "source": name,
                # source_uri MUST include series_id — the UNIQUE constraint
                # (source, source_uri, event_time) on macro_raw would
                # otherwise collapse all 4 snapshot rows to one (first one
                # wins, the rest hit ON CONFLICT DO NOTHING silently).
                "source_uri": f"coingecko/global/{series_id}/{iso_ts}",
                "symbol": None,
                "series_id": series_id,
                "event_time": event_time,
                "ingestion_time": now,
                "value": value,
                "value_text": "snapshot",
                "content_hash": content_hash(series_id, iso_ts, value),
                "schema_version": 1,
            }
        )
    return rows


def _fetch_coin_history(
    client: httpx.Client,
    coin_id: str,
    days_back: int,
    now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Pull ``/coins/{id}/market_chart`` and emit three series per timestamp."""
    payload = _http_get(
        client,
        f"/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": str(days_back)},
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"coingecko market_chart returned unexpected shape for {coin_id}: {type(payload)}"
        )

    prices = payload.get("prices") or []
    market_caps = payload.get("market_caps") or []
    volumes = payload.get("total_volumes") or []

    rows: list[dict[str, Any]] = []
    rows.extend(
        _emit_series(
            f"{coin_id}_price_usd",
            prices,
            now=now,
            window_start=window_start,
            window_end=window_end,
        )
    )
    rows.extend(
        _emit_series(
            f"{coin_id}_market_cap_usd",
            market_caps,
            now=now,
            window_start=window_start,
            window_end=window_end,
        )
    )
    rows.extend(
        _emit_series(
            f"{coin_id}_volume_usd",
            volumes,
            now=now,
            window_start=window_start,
            window_end=window_end,
        )
    )
    return rows


def _emit_series(
    series_id: str,
    raw_points: list[Any],
    *,
    now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in raw_points:
        # Each point is [unix_ms, value].
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        ts_ms = entry[0]
        value = _to_float(entry[1])
        if value is None or not isinstance(ts_ms, (int, float)):
            continue

        event_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        if event_time < window_start or event_time > window_end:
            continue

        iso_ts = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(
            {
                "source": name,
                "source_uri": f"coingecko/{series_id}/{iso_ts}",
                "symbol": None,
                "series_id": series_id,
                "event_time": event_time,
                "ingestion_time": now,
                "value": value,
                "value_text": None,
                "content_hash": content_hash(series_id, iso_ts, value),
                "schema_version": 1,
            }
        )
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch(request: IngestionRequest) -> IngestionResult:
    config = request.config or {}
    want_global: bool = bool(config.get("global_metrics", True))
    per_coin: list[str] = [str(c).strip() for c in (config.get("per_coin") or []) if str(c).strip()]

    if not want_global and not per_coin:
        msg = (
            "coingecko requires either config.global_metrics=true or a non-empty "
            "config.per_coin list"
        )
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    started = time.monotonic()
    now = datetime.utcnow()

    # CoinGecko free tier caps history at 365 days. We over-fetch by 2 days to
    # absorb tz boundary effects, then filter client-side.
    days_back = max(1, (now.date() - request.start.date()).days + 2)
    days_back = min(days_back, _FREE_TIER_DAYS_CAP)

    # Skip the global snapshot if the operator is asking for a window that
    # doesn't include "now" — the snapshot endpoint has no historical mode
    # and a now-stamped row outside the requested window would land in
    # a default partition with misleading metadata.
    snapshot_enabled = want_global and request.end >= now - timedelta(hours=_SNAPSHOT_RECENCY_HOURS)
    snapshot_skipped = want_global and not snapshot_enabled

    all_rows: list[dict[str, Any]] = []
    series_seen: list[str] = []

    with httpx.Client(headers={"Accept": "application/json"}) as client:
        # ── Global snapshot ────────────────────────────────────────────────
        if snapshot_enabled:
            logger.info("coingecko fetching global snapshot")
            try:
                rows = _fetch_global_snapshot(client, now)
            except Exception as e:  # noqa: BLE001
                logger.exception("coingecko /global fetch failed")
                update_source_health(name, success=False, error_message=str(e)[:500])
                raise
            all_rows.extend(rows)
            series_seen.extend(sorted({r["series_id"] for r in rows}))
        elif snapshot_skipped:
            logger.info(
                "coingecko global snapshot skipped — request.end=%s is older than now-%dh",
                request.end,
                _SNAPSHOT_RECENCY_HOURS,
            )

        # ── Per-coin history ───────────────────────────────────────────────
        for coin_id in per_coin:
            logger.info(
                "coingecko fetching coin=%s days_back=%d (window %s → %s)",
                coin_id,
                days_back,
                request.start,
                request.end,
            )
            try:
                rows = _fetch_coin_history(
                    client,
                    coin_id,
                    days_back,
                    now=now,
                    window_start=request.start,
                    window_end=request.end,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("coingecko market_chart fetch failed | coin=%s", coin_id)
                update_source_health(
                    name, success=False, error_message=f"{coin_id}: {str(e)[:400]}"
                )
                raise
            all_rows.extend(rows)
            if rows:
                for sid in sorted({r["series_id"] for r in rows}):
                    if sid not in series_seen:
                        series_seen.append(sid)

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
        logger.exception("coingecko DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    note_parts: list[str] = []
    if snapshot_skipped:
        note_parts.append(
            f"Global snapshot skipped (request.end older than now-{_SNAPSHOT_RECENCY_HOURS}h)."
        )
    if days_back == _FREE_TIER_DAYS_CAP and (now.date() - request.start.date()).days + 2 > _FREE_TIER_DAYS_CAP:
        note_parts.append(
            f"Per-coin history clamped to {_FREE_TIER_DAYS_CAP}-day public-tier limit. "
            f"Older data needs paid /market_chart/range."
        )
    note = " ".join(note_parts) if note_parts else None

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
        "coingecko fetch complete | series=%d fetched=%d inserted=%d skipped=%d "
        "pit_reject=%d duration=%.2fs",
        len(series_seen),
        result.rows_fetched,
        result.rows_inserted,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
