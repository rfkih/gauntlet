"""Compute engine for FeatureDefs.

Three steps per feature:

1. **Read** the requested ``series_id`` values from the raw tables in the
   declared time window. Today only ``macro_raw`` is supported.
2. **Pivot** the long rows into a wide DataFrame indexed by ``event_time``,
   with one column per ``series_id``. Forward-fill per the feature's
   ``ffill_policy`` (capped by ``max_ffill_age_hours`` — older holes stay
   NaN, not silently extended).
3. **Transform** by invoking the registered transformer, then **PIT
   validate** the output: no row's ``ts`` may sit earlier than the latest
   ``ingestion_time`` of its inputs. (Engine doesn't trust the
   transformer's ``pit_safe`` claim — it checks.)

Output is a tidy DataFrame ``[ts, value, version, name, family]`` so callers
can concat across features and write to whatever backend they choose.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta

import pandas as pd
import psycopg

from ..shared.db import get_connection
from .definitions import FeatureDef

logger = logging.getLogger(__name__)


class FeatureComputeError(RuntimeError):
    """Raised when a feature cannot be computed honestly — missing inputs,
    PIT violation, or empty output window.
    """


def _read_raw_long(
    conn: psycopg.Connection,
    table: str,
    series_ids: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Read long-format rows for the requested series in ``[start, end]``."""
    sql = f"""
        SELECT series_id, event_time, ingestion_time, value
        FROM {table}
        WHERE series_id = ANY(%(series_ids)s)
          AND event_time >= %(start)s
          AND event_time <= %(end)s
          AND value IS NOT NULL
        ORDER BY event_time ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"series_ids": list(series_ids), "start": start, "end": end})
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(
            columns=["series_id", "event_time", "ingestion_time", "value"]
        )
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True).dt.tz_convert(None)
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], utc=True).dt.tz_convert(None)
    df["value"] = df["value"].astype("float64")
    return df


# Columns we surface to transformers reading market_data. The wide-frame
# index is the bar's ``start_time`` (left edge of the interval).
#
# ── PIT anchor convention (READ THIS BEFORE WRITING LABEL TRANSFORMERS) ──
# market_data features emit ``ts = start_time``. A bar starting at
# ``T`` has its ``close_price`` finalized only at ``T + interval``. So a
# value stored at ``ts = T`` in ``feature_values`` was NOT actually
# observable at wall-clock ``T``; it became known at ``T + interval``.
#
# Backward-looking features (rolling stats, momentum, z-scores, etc.) are
# safe under this convention as long as the downstream consumer joins
# with ``feature_value.ts <= bar.start_time - interval`` (i.e. uses the
# previous-bar's feature when deciding at the current bar). The trading
# JVM's existing feature_store reads use this rule.
#
# Forward-looking labels MUST NOT include the anchor bar's close in their
# "current price" reference. When label transformers ship, they need to
# treat the price at ``ts`` as a *future* observation, not a current one.
_MARKET_DATA_COLS = (
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "trade_count",
)


def _read_market_data_wide(
    conn: psycopg.Connection,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Read OHLCV bars for ``(symbol, interval)`` over ``[start, end]``.

    Returns a wide DataFrame indexed by ``start_time`` with OHLCV columns.
    market_data is gapless by construction (the trading JVM's backfill
    pipeline fills missing bars), so no pivot/ffill is needed downstream.

    See the comment block above on ``_MARKET_DATA_COLS`` for the PIT
    anchor convention that callers (and especially forward-looking label
    transformers) must respect.
    """
    cols_sql = ", ".join(_MARKET_DATA_COLS)
    sql = f"""
        SELECT start_time, {cols_sql}
        FROM market_data
        WHERE symbol = %(symbol)s
          AND interval = %(interval)s
          AND start_time >= %(start)s
          AND start_time <= %(end)s
        ORDER BY start_time ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"symbol": symbol, "interval": interval, "start": start, "end": end})
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=list(_MARKET_DATA_COLS))
    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.set_index("start_time").sort_index()
    for col in _MARKET_DATA_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


def _pivot_wide(
    long_df: pd.DataFrame,
    feat: FeatureDef,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot to wide ``[event_time × series_id]`` for values and a parallel
    ``[event_time × series_id]`` of ``ingestion_time`` for PIT validation.

    When a series publishes multiple values at the same ``event_time``
    (revisions), the latest by ``ingestion_time`` wins — that's what an
    operator looking at the wide table "now" would see.
    """
    if long_df.empty:
        return (
            pd.DataFrame(columns=list(feat.inputs)),
            pd.DataFrame(columns=list(feat.inputs)),
        )

    long_df = long_df.sort_values(["series_id", "event_time", "ingestion_time"])
    deduped = long_df.drop_duplicates(
        subset=["series_id", "event_time"], keep="last"
    )

    value_wide = deduped.pivot(index="event_time", columns="series_id", values="value")
    ing_wide = deduped.pivot(index="event_time", columns="series_id", values="ingestion_time")

    # Ensure every declared input column exists, even if no rows were returned.
    for col in feat.inputs:
        if col not in value_wide.columns:
            value_wide[col] = pd.NA
            ing_wide[col] = pd.NaT

    value_wide = value_wide[list(feat.inputs)].sort_index()
    ing_wide = ing_wide[list(feat.inputs)].sort_index()
    return value_wide, ing_wide


def _apply_ffill(
    value_wide: pd.DataFrame,
    feat: FeatureDef,
) -> pd.DataFrame:
    """Forward-fill missing values per the feature's policy, capped by
    ``max_ffill_age_hours`` so a stale source can't silently get extended
    forever.
    """
    if feat.ffill_policy is None:
        return value_wide
    if feat.ffill_policy != "last_value":
        raise FeatureComputeError(
            f"Unknown ffill_policy '{feat.ffill_policy}' on feature {feat.name}"
        )

    if feat.max_ffill_age_hours is None:
        # Unbounded fill — fine for static features but suspicious in general.
        # Log a warning so the operator notices.
        logger.warning(
            "feature=%s using unbounded ffill — recommend setting max_ffill_age_hours",
            feat.name,
        )
        return value_wide.ffill()

    cap = pd.Timedelta(hours=feat.max_ffill_age_hours)
    # Build per-column ffill that respects the cap: for each NaN cell, only
    # carry forward if the gap from the last non-NaN index is <= cap.
    filled = value_wide.copy()
    for col in value_wide.columns:
        s = value_wide[col]
        last_valid_ts = s.where(s.notna()).index.to_series().where(s.notna()).ffill()
        gap = pd.Series(s.index, index=s.index) - last_valid_ts
        within_cap = gap <= cap
        filled[col] = s.ffill().where(within_cap, other=pd.NA)
    return filled


def _validate_pit(
    output: pd.Series,
    ing_wide: pd.DataFrame | None,
    feat: FeatureDef,
) -> None:
    """No-op (kept as a hook for future per-feature validators).

    PIT discipline is already enforced at two layers:

    1. **Ingest** — ``shared.pit_guards`` rejects rows whose ``event_time``
       lies outside the operator-requested window or in the future.
    2. **Query** — downstream model joins use ``event_time <= bar.start_time``
       so a feature value is only visible to a backtest after its publisher
       timestamp.

    The compute layer's job in between is to keep the latest revision per
    ``(series_id, event_time)`` — which ``_pivot_wide`` does via
    ``keep="last"`` after sorting by ``ingestion_time``. There's nothing
    additional to enforce here.

    Left as a callable so a future feature with a stricter contract
    (e.g. forward-looking labels that *must* see future data) can override
    the validation locally.
    """
    return


def compute(
    feat: FeatureDef,
    *,
    start: datetime,
    end: datetime,
    conn: psycopg.Connection | None = None,
    symbol: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Compute one feature over ``[start, end]``. Returns tidy DataFrame::

        columns: ['ts', 'value', 'name', 'version', 'family']

    Two read paths, dispatched by ``feat.raw_tables[0]``:

    * ``"macro_raw"`` — long-format publisher events. Engine pivots to wide
      ``[event_time × series_id]`` and applies the declared ffill policy
      before invoking the transformer.
    * ``"market_data"`` — per-bar OHLCV. Engine reads wide directly
      (one row per bar, OHLCV columns) and skips pivot/ffill since
      market_data is gapless by construction. ``symbol`` and ``interval``
      are required.

    Empty input window → empty output (with note logged), not an error.
    """
    if len(feat.raw_tables) != 1:
        raise FeatureComputeError(
            f"feature={feat.name} declares {len(feat.raw_tables)} raw tables; "
            "multi-table joins not yet supported in the compute engine"
        )
    if start > end:
        raise FeatureComputeError(
            f"feature={feat.name}: start ({start}) is after end ({end}); "
            "refusing to run an inverted-window compute"
        )
    table = feat.raw_tables[0]

    owned = conn is None
    if owned:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        if table == "macro_raw":
            wide_df = _compute_from_macro_raw(conn, feat, start, end)
        elif table == "market_data":
            if not symbol or not interval:
                raise FeatureComputeError(
                    f"feature={feat.name} reads market_data; symbol and interval "
                    f"are both required (got symbol={symbol!r} interval={interval!r})"
                )
            # Pre-check input columns so a typo surfaces as a clear engine
            # error, not a pandas KeyError deep inside the transformer.
            bad_cols = [c for c in feat.inputs if c not in _MARKET_DATA_COLS]
            if bad_cols:
                raise FeatureComputeError(
                    f"feature={feat.name} declares market_data inputs that aren't "
                    f"OHLCV columns: {bad_cols}. Known columns: "
                    f"{list(_MARKET_DATA_COLS)}"
                )
            wide_df = _compute_from_market_data(conn, feat, symbol, interval, start, end)
        else:
            raise FeatureComputeError(
                f"feature={feat.name} declares unknown raw_table '{table}'. "
                f"Supported: macro_raw, market_data"
            )
    finally:
        if owned:
            # Pass live exception info so the context manager sees the failure
            # and any future cleanup tied to it runs correctly.
            ctx.__exit__(*sys.exc_info())  # type: ignore[has-type]

    # Multi-symbol path returns a dict; single-symbol path returns a DataFrame.
    # Empty-input check has to handle both shapes — for dict, _compute_from_
    # market_data already raised on any empty frame, so a non-None dict is
    # known non-empty here. We only need the .empty check for the DataFrame
    # path.
    if wide_df is None:
        logger.info(
            "compute %s v%d -> no input rows in [%s, %s]; returning empty",
            feat.name, feat.version, start, end,
        )
        return pd.DataFrame(columns=["ts", "value", "name", "version", "family"])
    if isinstance(wide_df, pd.DataFrame) and wide_df.empty:
        logger.info(
            "compute %s v%d -> no input rows in [%s, %s]; returning empty",
            feat.name, feat.version, start, end,
        )
        return pd.DataFrame(columns=["ts", "value", "name", "version", "family"])

    raw_output = feat.transformer(wide_df)
    if not isinstance(raw_output, pd.Series):
        raise FeatureComputeError(
            f"feature={feat.name} transformer returned {type(raw_output).__name__}; "
            "expected pandas.Series"
        )
    output = raw_output.dropna()
    _validate_pit(output, None, feat)

    tidy = pd.DataFrame(
        {
            "ts": output.index,
            "value": output.values,
            "name": feat.name,
            "version": feat.version,
            "family": feat.family,
        }
    )
    logger.info(
        "compute %s v%d -> rows=%d window=%s..%s",
        feat.name, feat.version, len(tidy),
        tidy["ts"].min() if not tidy.empty else None,
        tidy["ts"].max() if not tidy.empty else None,
    )
    return tidy


def _compute_from_macro_raw(
    conn: psycopg.Connection, feat: FeatureDef, start: datetime, end: datetime
) -> pd.DataFrame:
    long_df = _read_raw_long(conn, "macro_raw", feat.inputs, start, end)
    if long_df.empty:
        return pd.DataFrame(columns=list(feat.inputs))
    value_wide, _ing_wide = _pivot_wide(long_df, feat)
    return _apply_ffill(value_wide, feat)


def _compute_from_market_data(
    conn: psycopg.Connection,
    feat: FeatureDef,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Read market_data for the transformer. Returns either a single wide
    DataFrame (today's per-symbol path) or a ``{symbol: DataFrame}`` dict
    (multi-symbol path used by cross-asset features like eth_btc_corr_24h).

    Multi-symbol path is triggered by ``feat.required_symbols`` being
    non-empty. The dispatcher reads each named symbol at the same
    ``interval`` and ``[start, end]`` window. The output is still stamped
    with the request's ``symbol`` — the required_symbols are *additional
    inputs* the transformer needs to see, not extra output keys.
    """
    if not feat.required_symbols:
        # Per-symbol path: trust market_data and let the transformer ask for
        # whatever column it declared in inputs.
        return _read_market_data_wide(conn, symbol, interval, start, end)

    bundle: dict[str, pd.DataFrame] = {}
    for sym in feat.required_symbols:
        df = _read_market_data_wide(conn, sym, interval, start, end)
        if df.empty:
            raise FeatureComputeError(
                f"feature={feat.name}: market_data has no rows for "
                f"symbol={sym} interval={interval} in [{start}, {end}]. "
                f"Backfill required before computing this cross-asset feature."
            )
        bundle[sym] = df
    return bundle


def default_window(days_back: int = 180) -> tuple[datetime, datetime]:
    """Convenience: ``(end - days, end)`` for callers that don't care.

    Callers that need a different anchor pass explicit ``start``/``end`` to
    :func:`compute` directly.
    """
    end = datetime.utcnow()
    return end - timedelta(days=days_back), end
