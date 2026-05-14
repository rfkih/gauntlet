"""DefiLlama Public API → ``macro_raw``.

DefiLlama publishes free, unauthenticated stablecoin and TVL data. We use
it to track stablecoin supply (a leading indicator for crypto inflows) and
chain TVL distribution.

Two payloads ingested per dispatch:

1. **Per-stablecoin charts** (``config.stablecoins=["USDT","USDC"]``):
   GET ``https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={id}``
   per requested symbol. Symbol → DefiLlama numeric id is resolved via the
   directory endpoint on first use and cached in-process. Emits one row
   per (asset, date) with the asset's circulating USD-pegged supply.

   Series written per asset (e.g. ``USDT``):
     - ``stablecoin_usdt_circulating_usd``
     - ``stablecoin_usdc_circulating_usd``

2. **Chain TVL** (``config.chains=["Ethereum","Tron","BSC"]``):
   GET ``https://api.llama.fi/v2/historicalChainTvl/{chain}`` (chain name
   URL-encoded) → time series of TVL per chain in USD. Emits one row per
   (chain, date).

   Series written per chain:
     - ``chain_tvl_ethereum_usd``
     - ``chain_tvl_tron_usd``
     - ``chain_tvl_bsc_usd``

PIT note: DefiLlama recomputes historical values when new protocols are
added to its index — that's a textbook PIT-poison vector. Rows outside the
operator-requested window are PIT-rejected; ``ingestion_time`` preserves
the value as-of-now in case we later want a per-vintage reconstruction.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

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

name = "defillama"
raw_table = "macro_raw"

_STABLECOINS_BASE = "https://stablecoins.llama.fi"
_TVL_BASE = "https://api.llama.fi"

# Matches V67 seed: max_backfill_lag_hours=48 for defillama. Now interpreted
# as the slop tolerance for event_time relative to request.start (anti-PIT
# poison) rather than a hard "stale data" cutoff.
_PIT_CONFIG = PitConfig(max_backfill_lag_hours=48)

# In-process cache: DefiLlama assigns each stablecoin a numeric id. The
# directory endpoint is small (~150 entries, ~50 KB), so we fetch once per
# process and reuse. Maps upper-case symbol → numeric id (as string).
_stablecoin_id_cache: dict[str, str] | None = None


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _http_get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(
        url,
        params=params or {},
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_stablecoin_directory(client: httpx.Client) -> dict[str, str]:
    """Resolve symbol → id from DefiLlama's stablecoins directory.

    Cached in-process so repeat fetch() calls only pay the directory hit
    once per server lifetime. The directory shape is::

        {"peggedAssets": [{"id": "1", "symbol": "USDT", ...}, ...]}
    """
    global _stablecoin_id_cache
    if _stablecoin_id_cache is not None:
        return _stablecoin_id_cache

    payload = _http_get(
        client,
        f"{_STABLECOINS_BASE}/stablecoins",
        params={"includePrices": "false"},
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"defillama /stablecoins returned unexpected shape: {type(payload)}"
        )
    pegged = payload.get("peggedAssets") or []
    if not isinstance(pegged, list):
        raise ValueError("defillama /stablecoins missing peggedAssets array")

    directory: dict[str, str] = {}
    for entry in pegged:
        if not isinstance(entry, dict):
            continue
        sym = entry.get("symbol")
        asset_id = entry.get("id")
        if isinstance(sym, str) and isinstance(asset_id, (str, int)):
            directory[sym.upper()] = str(asset_id)

    _stablecoin_id_cache = directory
    return directory


def _extract_stablecoin_value(entry: dict[str, Any]) -> float | None:
    """Pull a single peggedUSD value out of one bulk-chart entry.

    Defensively prefers ``totalCirculatingUSD.peggedUSD`` (USD-denominated)
    but falls back to ``totalCirculating.peggedUSD`` when the former is
    absent — both shapes occur depending on whether the chart endpoint is
    bulk or filtered.
    """
    for key in ("totalCirculatingUSD", "totalCirculating"):
        bucket = entry.get(key)
        if isinstance(bucket, dict):
            val = _to_float(bucket.get("peggedUSD"))
            if val is not None:
                return val
    return None


def _fetch_stablecoin_chart(
    client: httpx.Client,
    symbol: str,
    asset_id: str,
    *,
    now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Fetch one stablecoin's USD-circulating timeseries.

    Uses the bulk chart endpoint with a ``stablecoin`` filter parameter so
    we get back the same response shape (a list of date entries) but scoped
    to a single asset.
    """
    payload = _http_get(
        client,
        f"{_STABLECOINS_BASE}/stablecoincharts/all",
        params={"stablecoin": asset_id},
    )
    if not isinstance(payload, list):
        raise ValueError(
            f"defillama stablecoincharts (id={asset_id}) returned unexpected shape: "
            f"{type(payload)}"
        )

    series_id = f"stablecoin_{symbol.lower()}_circulating_usd"
    rows: list[dict[str, Any]] = []

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        ts_unix = entry.get("date")
        try:
            ts_unix_int = int(ts_unix) if ts_unix is not None else None
        except (TypeError, ValueError):
            ts_unix_int = None
        if ts_unix_int is None or ts_unix_int <= 0:
            continue

        event_time = datetime.fromtimestamp(ts_unix_int, tz=timezone.utc).replace(tzinfo=None)
        if event_time < window_start or event_time > window_end:
            continue

        value = _extract_stablecoin_value(entry)
        if value is None:
            continue

        iso_ts = event_time.strftime("%Y-%m-%d")
        rows.append(
            {
                "source": name,
                "source_uri": f"defillama/stablecoincharts/{asset_id}/{iso_ts}",
                "symbol": None,
                "series_id": series_id,
                "event_time": event_time,
                "ingestion_time": now,
                "value": value,
                "value_text": symbol.upper(),
                "content_hash": content_hash(series_id, iso_ts, value),
                "schema_version": 1,
            }
        )
    return rows


def _fetch_chain_tvl(
    client: httpx.Client,
    chain: str,
    *,
    now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    # URL-encode chain path segment: handles "Polygon zkEVM", "BNB Smart Chain",
    # accented characters, etc. without breaking the request URL.
    encoded_chain = quote(chain, safe="")
    payload = _http_get(client, f"{_TVL_BASE}/v2/historicalChainTvl/{encoded_chain}")
    if not isinstance(payload, list):
        raise ValueError(
            f"defillama historicalChainTvl/{chain} returned unexpected shape: {type(payload)}"
        )

    # Lower-case + underscore the chain name for the series_id while keeping
    # value_text as the human-friendly form.
    slug = chain.lower().replace(" ", "_").replace("/", "_")
    series_id = f"chain_tvl_{slug}_usd"

    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        ts_unix = entry.get("date")
        try:
            ts_unix_int = int(ts_unix) if ts_unix is not None else None
        except (TypeError, ValueError):
            ts_unix_int = None
        if ts_unix_int is None or ts_unix_int <= 0:
            continue

        event_time = datetime.fromtimestamp(ts_unix_int, tz=timezone.utc).replace(tzinfo=None)
        if event_time < window_start or event_time > window_end:
            continue

        tvl = _to_float(entry.get("tvl"))
        if tvl is None:
            continue

        iso_ts = event_time.strftime("%Y-%m-%d")
        rows.append(
            {
                "source": name,
                "source_uri": f"defillama/chain_tvl/{encoded_chain}/{iso_ts}",
                "symbol": None,
                "series_id": series_id,
                "event_time": event_time,
                "ingestion_time": now,
                "value": tvl,
                "value_text": chain,
                "content_hash": content_hash(series_id, iso_ts, tvl),
                "schema_version": 1,
            }
        )
    return rows


def fetch(request: IngestionRequest) -> IngestionResult:
    config = request.config or {}
    stablecoins: list[str] = [
        str(s).strip() for s in (config.get("stablecoins") or []) if str(s).strip()
    ]
    chains: list[str] = [
        str(c).strip() for c in (config.get("chains") or []) if str(c).strip()
    ]

    if not stablecoins and not chains:
        msg = "defillama requires config.stablecoins or config.chains to be non-empty"
        update_source_health(name, success=False, error_message=msg)
        raise ValueError(msg)

    started = time.monotonic()
    now = datetime.utcnow()

    all_rows: list[dict[str, Any]] = []
    series_seen: list[str] = []
    unresolved_symbols: list[str] = []

    with httpx.Client(headers={"Accept": "application/json"}) as client:
        # ── Per-stablecoin charts ──────────────────────────────────────────
        if stablecoins:
            try:
                directory = _load_stablecoin_directory(client)
            except Exception as e:  # noqa: BLE001
                logger.exception("defillama stablecoin directory fetch failed")
                update_source_health(name, success=False, error_message=str(e)[:500])
                raise

            for symbol in stablecoins:
                upper = symbol.upper()
                asset_id = directory.get(upper)
                if asset_id is None:
                    unresolved_symbols.append(upper)
                    logger.warning(
                        "defillama no asset id for stablecoin symbol=%s — skipping",
                        upper,
                    )
                    continue

                logger.info(
                    "defillama fetching stablecoin=%s id=%s window=%s→%s",
                    upper,
                    asset_id,
                    request.start,
                    request.end,
                )
                try:
                    rows = _fetch_stablecoin_chart(
                        client,
                        upper,
                        asset_id,
                        now=now,
                        window_start=request.start,
                        window_end=request.end,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "defillama stablecoin chart fetch failed | symbol=%s id=%s",
                        upper,
                        asset_id,
                    )
                    update_source_health(
                        name,
                        success=False,
                        error_message=f"{upper}: {str(e)[:400]}",
                    )
                    raise
                all_rows.extend(rows)
                if rows:
                    sid = rows[0]["series_id"]
                    if sid not in series_seen:
                        series_seen.append(sid)

        # ── Chain TVL ──────────────────────────────────────────────────────
        for chain in chains:
            logger.info(
                "defillama fetching chain TVL | chain=%s window=%s→%s",
                chain,
                request.start,
                request.end,
            )
            try:
                rows = _fetch_chain_tvl(
                    client,
                    chain,
                    now=now,
                    window_start=request.start,
                    window_end=request.end,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("defillama chain TVL fetch failed | chain=%s", chain)
                update_source_health(
                    name, success=False, error_message=f"{chain}: {str(e)[:400]}"
                )
                raise
            all_rows.extend(rows)
            if rows:
                sid = rows[0]["series_id"]
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
        logger.exception("defillama DB write failed")
        update_source_health(name, success=False, error_message=str(e)[:500])
        raise

    duration = time.monotonic() - started
    note = None
    if unresolved_symbols:
        note = (
            f"Could not resolve DefiLlama id for symbols: "
            f"{sorted(set(unresolved_symbols))} — skipped."
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
        "defillama fetch complete | series=%d fetched=%d inserted=%d skipped=%d "
        "pit_reject=%d duration=%.2fs",
        len(series_seen),
        result.rows_fetched,
        result.rows_inserted,
        result.rows_skipped_duplicate,
        result.rows_rejected_pit,
        result.duration_seconds,
    )
    return result
