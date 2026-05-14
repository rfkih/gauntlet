"""Declarative feature definitions.

Each :class:`FeatureDef` carries everything the compute engine needs:

* ``name`` / ``version`` — primary key, matches ``feature_registry`` rows.
* ``inputs`` — the ``series_id`` values to read from ``macro_raw`` (or its
  cousins). The engine pivots them into a wide DataFrame before calling the
  ``transformer``.
* ``transformer`` — pure function ``pd.DataFrame -> pd.Series`` over the
  wide input DataFrame. Index is the union of input event_times.
* ``ffill_policy`` — how to handle a missing input at a tick. Today only
  ``"last_value"`` (carry forward) or ``None`` (no fill) are supported; the
  engine refuses to carry a value more than ``max_ffill_age_hours``.
* ``pit_safe`` — declarative claim; the engine *also* validates by checking
  that no input row's ``event_time`` exceeds the output ``ts``.

The starter set (v0) covers three macro features so the pipeline can be
proven end-to-end before we commit to schema (V70+). Once persistence
lands, the full blueprint list of 22 features + 4 labels gets ported here
one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

FfillPolicy = Literal["last_value"]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    version: int
    family: str
    # ``inputs`` carries different meaning per raw table:
    # * macro_raw: list of ``series_id`` values to read & pivot (drives the
    #   SQL filter).
    # * market_data: list of column names the transformer requires from the
    #   wide OHLCV frame. Decorative for the *read* (we always SELECT all
    #   OHLCV columns), but validated against ``_MARKET_DATA_COLS`` in the
    #   engine so a typo surfaces clearly instead of as a pandas KeyError.
    inputs: tuple[str, ...]
    transformer: Callable[[pd.DataFrame], pd.Series]
    pit_safe: bool = True
    ffill_policy: FfillPolicy | None = None
    max_ffill_age_hours: int | None = None
    description: str = ""

    # Where the inputs live. ``("macro_raw",)`` (default) is the long-format
    # publisher-event store with ``series_id``/``event_time``/``value`` rows
    # — the engine pivots before calling the transformer.
    # ``("market_data",)`` is the per-bar OHLCV store; the engine reads it
    # already wide (one row per bar, OHLCV columns) and skips the pivot.
    raw_tables: tuple[str, ...] = field(default=("macro_raw",))

    # Per-instance scope. Empty tuple = "applies globally" (macro features
    # with no symbol/interval anchor). When set, the CLI loops the
    # cross-product and ``feature_values`` rows get the symbol/interval
    # stamped on them so downstream joins can scope correctly.
    #
    # Invariant (validated in ``__post_init__``): symbols and intervals are
    # both-empty or both-set. Half-specified shapes (one declared, the other
    # left empty) produce nonsense feature_values stamps and are rejected.
    symbols: tuple[str, ...] = field(default=())
    intervals: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if bool(self.symbols) != bool(self.intervals):
            raise ValueError(
                f"FeatureDef '{self.name}' v{self.version}: symbols and "
                f"intervals must be both empty (global feature) or both "
                f"set (per-bar feature). Got symbols={self.symbols!r} "
                f"intervals={self.intervals!r}."
            )


# ── Transformers ──────────────────────────────────────────────────────────────


def _passthrough(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def _impl(df: pd.DataFrame) -> pd.Series:
        return df[col].astype("float64")

    return _impl


def _rolling_zscore(col: str, window: int, min_periods: int) -> Callable[[pd.DataFrame], pd.Series]:
    def _impl(df: pd.DataFrame) -> pd.Series:
        s = df[col].astype("float64")
        mean = s.rolling(window, min_periods=min_periods).mean()
        std = s.rolling(window, min_periods=min_periods).std()
        # Avoid divide-by-zero when std is 0 — return NaN there (caller drops).
        return (s - mean) / std.where(std > 0)

    return _impl


def _change_pct(col: str, periods: int) -> Callable[[pd.DataFrame], pd.Series]:
    """Percent change of ``col`` over ``periods`` rows.

    Note: ``periods`` is row-count, not a fixed time window. The cadence
    of the input series dictates the time meaning — e.g. a 24-period
    change on a 1h series is "24 hours"; on a daily series it's "24
    business days".
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        return df[col].astype("float64").pct_change(periods=periods)

    return _impl


def _change_diff(col: str, periods: int) -> Callable[[pd.DataFrame], pd.Series]:
    """Absolute change of ``col`` over ``periods`` rows. See note on
    :func:`_change_pct` for the cadence semantics.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        return df[col].astype("float64").diff(periods=periods)

    return _impl


def _rolling_percentile_rank(
    col: str, window: int, min_periods: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """Percentile rank of the current value within the rolling window of
    size ``window`` rows. Returns values in [0, 1]. Uses pandas
    ``Rolling.rank(pct=True)`` which since 1.4 returns each row's percentile
    relative to the trailing window ending at that row.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        s = df[col].astype("float64")
        return s.rolling(window, min_periods=min_periods).rank(pct=True)

    return _impl


def _sum_then_change_pct(
    cols: tuple[str, ...], periods: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """Sum across ``cols`` row-wise, then ``periods``-row % change. NaN in
    any single column at a given row is treated as missing for that column
    only (``min_count=1`` returns NaN only when every column is NaN).
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        summed = df[list(cols)].sum(axis=1, min_count=1).astype("float64")
        return summed.pct_change(periods=periods)

    return _impl


def _ratio_momentum(
    num_col: str, den_col: str, periods: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """``periods``-row % change of the ratio ``num_col / den_col``. Guards
    against zero-denominator by masking to NaN.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        num = df[num_col].astype("float64")
        den = df[den_col].astype("float64")
        ratio = num / den.where(den != 0)
        return ratio.pct_change(periods=periods)

    return _impl


def _rolling_realized_vol(
    close_col: str,
    window_bars: int,
    min_periods: int,
    annualize_factor: float | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Rolling realized volatility from a close-price series.

    Algorithm: log returns -> rolling std over ``window_bars`` rows. If
    ``annualize_factor`` is set, scale by ``sqrt(annualize_factor)`` so the
    output is annualized vol — e.g. ``8760`` for hourly bars (24 × 365),
    ``2190`` for 4h, ``365`` for daily.

    Note: ``window_bars`` is in row-count, not time. Cadence assumption is
    encoded in the FeatureDef's declared interval.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        log_ret = np.log(c / c.shift(1))
        sigma = log_ret.rolling(window_bars, min_periods=min_periods).std()
        if annualize_factor is not None:
            sigma = sigma * (annualize_factor ** 0.5)
        return sigma

    return _impl


# ── Forward-looking transformers (labels) ────────────────────────────────────
#
# Labels read FUTURE bars. They invert the PIT direction of feature
# transformers: at row t, the output is "what will happen from t onward".
# Conventions:
#   * The output is anchored at row t (the decision time). The forward
#     horizon (h bars) determines how far we peek.
#   * Trailing rows (last h rows of the input) emit NaN because future
#     bars don't exist; the engine's .dropna() at the end of compute()
#     filters those out so they don't pollute feature_values.
#   * Labels declare ``pit_safe=False`` so a future "labels must not be
#     used as input features" guard can pivot off the field. Today the
#     engine's _validate_pit is a no-op, but the flag is the documented
#     hook for that policy.


def _forward_return(
    close_col: str, horizon_bars: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """``(close[t+horizon] - close[t]) / close[t]`` — simple forward return.

    On 1h bars: ``horizon_bars=168`` for 7-day return.
    On 4h bars: ``horizon_bars=42``.
    On 1d bars: ``horizon_bars=7``.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        return (c.shift(-horizon_bars) - c) / c

    return _impl


def _forward_sharpe_binary_sign(
    close_col: str, horizon_bars: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """Binary label: 1 if forward Sharpe over horizon > 0, else 0.

    Forward Sharpe = forward_return / forward_vol where forward_vol is the
    std of 1-bar log returns over (t+1, t+horizon). NaN rows (insufficient
    future data) stay NaN so the engine's dropna() filters them out.

    Reproduces blueprint § 5.6 ``label_regime_risk_on_48h`` semantics
    (horizon_bars=48 on 1h bars).
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        log_ret = np.log(c / c.shift(1))
        fwd_ret = (c.shift(-horizon_bars) - c) / c
        # rolling(N).std() at row t covers (t-N+1, t). Shift -N so it
        # appears at row t-N, representing "std of returns from (t-N+1, t)
        # observed at decision time t-N".
        fwd_vol = log_ret.rolling(horizon_bars).std().shift(-horizon_bars)
        sharpe = fwd_ret / fwd_vol.where(fwd_vol > 0)
        binary = (sharpe > 0).astype("float64")
        # (NaN > 0) -> False -> 0.0 in pandas; restore NaN so dropna()
        # can drop rows where forward data isn't fully available.
        return binary.where(sharpe.notna(), other=pd.NA)

    return _impl


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range over ``n`` bars (SMA, not Wilder's EMA — close
    enough for labeling; cleaner closed-form). Requires high/low/close
    columns in the wide market_data frame.
    """
    high = df["high_price"].astype("float64")
    low = df["low_price"].astype("float64")
    close_prev = df["close_price"].astype("float64").shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _forward_meanrev_atr(
    close_col: str,
    horizon_bars: int,
    atr_window: int = 14,
    clip_value: float = 3.0,
) -> Callable[[pd.DataFrame], pd.Series]:
    """``(close[t+horizon] - close[t]) / ATR_14[t]``, clipped to ±``clip``.

    Forward price move expressed in ATR units — the canonical mean-reversion
    target for positioning sub-models. Clipping bounds the loss function and
    suppresses fat-tailed outliers (López de Prado style).

    Reproduces blueprint § 5.6 ``label_meanrev_24h`` semantics on 1h bars.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        atr = _atr(df, atr_window)
        forward_close = c.shift(-horizon_bars)
        delta = forward_close - c
        out = delta / atr.where(atr > 0)
        return out.clip(-clip_value, clip_value)

    return _impl


def _forward_triple_barrier(
    close_col: str = "close_price",
    high_col: str = "high_price",
    low_col: str = "low_price",
    horizon_bars: int = 24,
    k_tp: float = 1.5,
    k_sl: float = 1.0,
    atr_window: int = 14,
) -> Callable[[pd.DataFrame], pd.Series]:
    """López de Prado triple-barrier labeler.

    For each entry row t::

        entry = close[t]
        atr   = ATR_n[t]            (Wilder's TR, SMA-smoothed)
        tp    = entry + k_tp * atr
        sl    = entry - k_sl * atr

    Walk forward over ``bars (t+1, t+horizon)``:

    * ``+1``   if high[bar] >= tp first
    * ``-1``   if low[bar]  <= sl first, OR both barriers touched in the
               same bar (conservative intra-bar fill rule — SL wins ties)
    * ``0``    if the full horizon elapses without either barrier hit
    * ``NaN``  if the future window extends past the end of the data
               (label undefined; engine's dropna() filters these out)
    * ``NaN``  if ATR is undefined at t (warmup) or non-positive

    Reproduces blueprint § 5.6 ``label_triple_barrier`` semantics:
        k_tp=1.5, k_sl=1.0, horizon=24 (medium), atr_window=14, conservative
        intra-bar fill. Funding cost is excluded from label simulation in v1
        (per blueprint).

    Implementation note: uses a numpy loop because the "first hit" /
    "conservative tie" semantics don't vectorize cleanly without
    rebuilding the same logic twice. At ~38k bars × horizon=24, the loop
    runs in low single-digit seconds — well below the time budget.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64").to_numpy()
        h = df[high_col].astype("float64").to_numpy()
        lo = df[low_col].astype("float64").to_numpy()
        atr = _atr(df, atr_window).to_numpy()
        n = len(c)
        out = np.full(n, np.nan, dtype="float64")

        for t in range(n):
            atr_t = atr[t]
            if not np.isfinite(atr_t) or atr_t <= 0:
                continue
            # Need a FULL horizon worth of future bars to assign a label.
            # If the window would walk off the end, label is undefined.
            last_bar = t + horizon_bars
            if last_bar >= n:
                continue

            entry = c[t]
            tp_level = entry + k_tp * atr_t
            sl_level = entry - k_sl * atr_t

            label = 0  # default: horizon timeout, no barrier hit
            for bar in range(t + 1, last_bar + 1):
                sl_hit = lo[bar] <= sl_level
                tp_hit = h[bar] >= tp_level
                if sl_hit:
                    # Pure SL OR conservative tie (sl_hit ∧ tp_hit) → -1
                    label = -1
                    break
                if tp_hit:
                    label = 1
                    break
            out[t] = float(label)

        return pd.Series(out, index=df.index)

    return _impl


# ── Starter feature set ───────────────────────────────────────────────────────

FEATURES: tuple[FeatureDef, ...] = (
    # ── Starter / debug passthroughs (kept for backwards compat with the
    # M3 validation runs; not in the blueprint's canonical 22) ──────────
    FeatureDef(
        name="vix_close",
        version=1,
        family="macro",
        inputs=("VIXCLS",),
        transformer=_passthrough("VIXCLS"),
        pit_safe=True,
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="VIX daily close. Passthrough from FRED VIXCLS — no transformation.",
    ),
    FeatureDef(
        name="dxy_close",
        version=1,
        family="macro",
        inputs=("DTWEXBGS",),
        transformer=_passthrough("DTWEXBGS"),
        pit_safe=True,
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="Trade-weighted USD index daily close. Passthrough from FRED DTWEXBGS.",
    ),
    FeatureDef(
        name="dxy_zscore_30d",
        version=1,
        family="macro",
        inputs=("DTWEXBGS",),
        transformer=_rolling_zscore("DTWEXBGS", window=30, min_periods=10),
        pit_safe=True,
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="30-day rolling z-score of DXY. Starter feature — superseded by "
        "dxy_zscore_252d for the blueprint canonical set.",
    ),
    # ── Blueprint § 5.1: Macro regime (6 features) ────────────────────
    FeatureDef(
        name="real_yield_10y_level",
        version=1,
        family="macro",
        inputs=("DFII10",),
        transformer=_passthrough("DFII10"),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="10-year TIPS real yield, level. Passthrough FRED DFII10.",
    ),
    FeatureDef(
        name="real_yield_10y_change_20d",
        version=1,
        family="macro",
        inputs=("DFII10",),
        transformer=_change_diff("DFII10", periods=20),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="20-business-day absolute change in 10y TIPS yield (DFII10, daily cadence).",
    ),
    FeatureDef(
        name="dxy_zscore_252d",
        version=1,
        family="macro",
        inputs=("DTWEXBGS",),
        transformer=_rolling_zscore("DTWEXBGS", window=252, min_periods=60),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="252-business-day rolling z-score of the trade-weighted USD (DTWEXBGS).",
    ),
    FeatureDef(
        name="dxy_momentum_20d",
        version=1,
        family="macro",
        inputs=("DTWEXBGS",),
        transformer=_change_pct("DTWEXBGS", periods=20),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="20-business-day % change in DXY.",
    ),
    FeatureDef(
        name="vix_percentile_252d",
        version=1,
        family="macro",
        inputs=("VIXCLS",),
        transformer=_rolling_percentile_rank("VIXCLS", window=252, min_periods=60),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="252-business-day rolling percentile rank of VIX close in [0,1].",
    ),
    FeatureDef(
        name="term_spread_2s10s",
        version=1,
        family="macro",
        inputs=("T10Y2Y",),
        transformer=_passthrough("T10Y2Y"),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="10y - 2y Treasury yield spread. Passthrough FRED T10Y2Y.",
    ),
    # ── Blueprint § 5.2: Positioning (5 features) ─────────────────────
    FeatureDef(
        name="btc_funding_8h",
        version=1,
        family="positioning",
        inputs=("binance_funding_rate_btcusdt",),
        transformer=_passthrough("binance_funding_rate_btcusdt"),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="Current 8h funding rate for BTCUSDT perp. Passthrough Binance fundingRate.",
    ),
    FeatureDef(
        name="btc_funding_zscore_30d",
        version=1,
        family="positioning",
        # Funding publishes every 8h -> 30d = 90 rows.
        inputs=("binance_funding_rate_btcusdt",),
        transformer=_rolling_zscore("binance_funding_rate_btcusdt", window=90, min_periods=20),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="30-day rolling z-score of BTC funding rate. 8h cadence (window=90 rows).",
    ),
    FeatureDef(
        name="btc_oi_change_24h_pct",
        version=1,
        family="positioning",
        # OI published hourly -> 24h = 24 rows.
        inputs=("binance_open_interest_btcusdt_1h",),
        transformer=_change_pct("binance_open_interest_btcusdt_1h", periods=24),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="24-hour % change in BTC perp open interest (USD, 1h cadence).",
    ),
    FeatureDef(
        name="taker_buy_ratio_4h",
        version=1,
        family="positioning",
        inputs=("binance_taker_buy_sell_ratio_btcusdt_4h",),
        transformer=_passthrough("binance_taker_buy_sell_ratio_btcusdt_4h"),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="4h taker buy / total volume ratio for BTCUSDT perp. Passthrough.",
    ),
    FeatureDef(
        name="topls_ratio_change_24h",
        version=1,
        family="positioning",
        # Top L/S published hourly -> 24h = 24 rows.
        inputs=("binance_long_short_ratio_btcusdt_1h",),
        transformer=_change_diff("binance_long_short_ratio_btcusdt_1h", periods=24),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="24-hour absolute change in top-trader long/short account ratio (1h cadence).",
    ),
    # ── Blueprint § 5.3: Flows (2 of 4 — exchange netflows need paid CM) ──
    FeatureDef(
        name="stablecoin_supply_change_7d",
        version=1,
        family="flow",
        inputs=("stablecoin_usdt_circulating_usd", "stablecoin_usdc_circulating_usd"),
        transformer=_sum_then_change_pct(
            ("stablecoin_usdt_circulating_usd", "stablecoin_usdc_circulating_usd"),
            periods=7,
        ),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="7-day % change in USDT+USDC combined circulating USD (DefiLlama, daily).",
    ),
    FeatureDef(
        name="stablecoin_supply_change_30d",
        version=1,
        family="flow",
        inputs=("stablecoin_usdt_circulating_usd", "stablecoin_usdc_circulating_usd"),
        transformer=_sum_then_change_pct(
            ("stablecoin_usdt_circulating_usd", "stablecoin_usdc_circulating_usd"),
            periods=30,
        ),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="30-day % change in USDT+USDC combined circulating USD (DefiLlama, daily).",
    ),
    # ── Blueprint § 5.4: Market structure (2 of 4 — perp_basis + realized_vol deferred) ──
    FeatureDef(
        name="btc_dominance_change_7d",
        version=1,
        family="market_structure",
        inputs=("btc_dominance_pct",),
        transformer=_change_diff("btc_dominance_pct", periods=7),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="7-period absolute change in BTC market-cap dominance (% points). "
        "CoinGecko cadence varies (hourly for short windows, daily beyond ~90d).",
    ),
    FeatureDef(
        name="eth_btc_ratio_momentum_20d",
        version=1,
        family="market_structure",
        inputs=("ethereum_price_usd", "bitcoin_price_usd"),
        transformer=_ratio_momentum("ethereum_price_usd", "bitcoin_price_usd", periods=20),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="20-period % change in the ETH/BTC price ratio. CoinGecko cadence-dependent.",
    ),
    # ── Blueprint § 5.5: Events (1 of 3 — FOMC calendar features deferred) ──
    FeatureDef(
        name="fear_greed_value",
        version=1,
        family="sentiment",
        inputs=("fear_and_greed",),
        transformer=_passthrough("fear_and_greed"),
        ffill_policy="last_value",
        max_ffill_age_hours=48,
        description="Crypto Fear & Greed Index 0-100 from alternative.me (daily).",
    ),
    # ── Blueprint § 5.4 (continued): from market_data ─────────────────
    FeatureDef(
        name="btc_realized_vol_30d",
        version=1,
        family="market_structure",
        # The "input" here is a COLUMN of market_data (close_price) rather
        # than a series_id from the long-format publisher store. The
        # compute engine branches on raw_tables=("market_data",) and reads
        # wide directly.
        inputs=("close_price",),
        transformer=_rolling_realized_vol(
            close_col="close_price",
            window_bars=720,       # 30 days × 24 1h bars
            min_periods=240,       # 10-day warmup
            annualize_factor=8760, # 24 × 365 -> annualized
        ),
        pit_safe=True,
        ffill_policy=None,         # market_data is gapless per-bar
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        description="30-day annualized realized volatility of BTC close-to-close log returns. "
        "Computed on 1h bars (window=720); annualization scales the rolling std "
        "by sqrt(8760) = sqrt(24 * 365).",
    ),
    # ── Blueprint § 5.6: Forward-looking labels (3 of 4 shipped here;
    # label_triple_barrier deferred — needs bar-stepping algorithm) ───
    FeatureDef(
        name="label_return_7d",
        version=1,
        family="label",
        inputs=("close_price",),
        transformer=_forward_return(close_col="close_price", horizon_bars=168),
        pit_safe=False,  # reads future bars by design
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        description="Forward 7-day simple return on BTC 1h bars (horizon=168). "
        "Consumed by the flow_btc_v1 sub-model.",
    ),
    FeatureDef(
        name="label_regime_risk_on_48h",
        version=1,
        family="label",
        inputs=("close_price",),
        transformer=_forward_sharpe_binary_sign(close_col="close_price", horizon_bars=48),
        pit_safe=False,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        description="Binary label: 1 if forward-48h Sharpe (return / vol) is positive, else 0. "
        "Drives regime_btc_v1 (risk-on / risk-off classification).",
    ),
    FeatureDef(
        name="label_meanrev_24h",
        version=1,
        family="label",
        # Needs OHLC for ATR_14. Declares high_price/low_price/close_price
        # so the engine's inputs-vs-columns check on market_data passes.
        inputs=("close_price", "high_price", "low_price"),
        transformer=_forward_meanrev_atr(
            close_col="close_price", horizon_bars=24, atr_window=14, clip_value=3.0
        ),
        pit_safe=False,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        description="Forward 24h price move in ATR_14 units, clipped to +/-3. "
        "Continuous label for positioning_btc_v1 (mean-reversion target).",
    ),
    FeatureDef(
        name="label_triple_barrier",
        version=1,
        family="label",
        inputs=("close_price", "high_price", "low_price"),
        transformer=_forward_triple_barrier(
            close_col="close_price",
            high_col="high_price",
            low_col="low_price",
            horizon_bars=24,   # medium horizon (blueprint § 5.6 default)
            k_tp=1.5,
            k_sl=1.0,
            atr_window=14,
        ),
        pit_safe=False,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        description="López de Prado triple-barrier label on BTC 1h bars. "
        "Class +1 if +1.5 ATR TP hit first, -1 if -1.0 ATR SL hit first "
        "(or both same bar — conservative fill), 0 on 24-bar timeout. "
        "Drives directional_btc_v1.",
    ),
)


_FEATURE_INDEX: dict[tuple[str, int], FeatureDef] = {
    (f.name, f.version): f for f in FEATURES
}


def get_feature(name: str, version: int = 1) -> FeatureDef:
    try:
        return _FEATURE_INDEX[(name, version)]
    except KeyError as exc:  # noqa: BLE001
        raise KeyError(
            f"No feature named '{name}' v{version}. "
            f"Known: {sorted((n, v) for (n, v) in _FEATURE_INDEX)}"
        ) from exc
