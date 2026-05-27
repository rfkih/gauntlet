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
from typing import Callable, Literal, Union

import numpy as np
import pandas as pd

FfillPolicy = Literal["last_value"]

# Single-symbol transformer (today's path): receives one wide DataFrame.
# Multi-symbol transformer (cross-asset features like eth_btc_corr_24h):
# receives {symbol: wide DataFrame}. Dispatcher picks based on
# ``FeatureDef.required_symbols``.
FeatureTransformer = Callable[
    [Union[pd.DataFrame, dict[str, pd.DataFrame]]], pd.Series
]


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
    transformer: FeatureTransformer
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

    # Extra symbols the transformer needs to READ besides the request's
    # output symbol. Empty (default) = single-symbol path: dispatcher reads
    # the request's symbol and passes a single DataFrame to the transformer.
    # Non-empty = multi-symbol path: dispatcher reads each named symbol from
    # market_data and passes a {symbol: DataFrame} dict to the transformer.
    # The output is still stamped with the request's symbol/interval — these
    # are *inputs*, not output keys.
    #
    # Example: eth_btc_corr_24h is BTCUSDT-stamped but reads ETH bars to
    # compute the correlation. required_symbols=("BTCUSDT", "ETHUSDT").
    required_symbols: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if bool(self.symbols) != bool(self.intervals):
            raise ValueError(
                f"FeatureDef '{self.name}' v{self.version}: symbols and "
                f"intervals must be both empty (global feature) or both "
                f"set (per-bar feature). Got symbols={self.symbols!r} "
                f"intervals={self.intervals!r}."
            )
        if self.required_symbols and self.raw_tables != ("market_data",):
            raise ValueError(
                f"FeatureDef '{self.name}' v{self.version}: required_symbols "
                f"is only supported for market_data features. Got "
                f"raw_tables={self.raw_tables!r}."
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


def _sign_streak(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Count of consecutive bars with the same sign as the current bar.

    Positive streak: +N when the last N bars (inclusive) are all positive.
    Negative streak: -N when the last N bars (inclusive) are all negative.
    Zero when the current bar is zero or NaN.

    Encoding: sign(current) * streak_length. This lets the model see both
    the direction (sign) and the persistence (magnitude) of the crowding.

    Funding cadence example: a streak of +5 means funding has been positive
    for 5 consecutive 8h periods (40h of crowded-long positioning).
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        s = df[col].astype("float64")
        signs = np.sign(s.to_numpy())
        n = len(signs)
        out = np.zeros(n, dtype="float64")
        for i in range(n):
            if signs[i] == 0 or np.isnan(signs[i]):
                out[i] = 0.0
                continue
            streak = 1
            j = i - 1
            while j >= 0 and signs[j] == signs[i]:
                streak += 1
                j -= 1
            out[i] = signs[i] * streak
        return pd.Series(out, index=s.index)

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


def _log_return(close_col: str, periods: int) -> Callable[[pd.DataFrame], pd.Series]:
    """``log(close[t] / close[t - periods])`` — periods-bar log return.

    Mirrors blackheart-train's ``_t_btc_log_return_24h`` exactly so the
    registry-sourced feature matches the train-time derived feature
    bit-for-bit on the same market_data input.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        return np.log(c / c.shift(periods))

    return _impl


def _cross_asset_correlation(
    close_col_a: str,
    close_col_b: str,
    window_bars: int,
) -> Callable[[dict[str, pd.DataFrame]], pd.Series]:
    """Rolling correlation between two symbols' 1-bar log returns.

    Multi-symbol transformer: takes ``{symbol_a: df, symbol_b: df}`` keyed
    by symbol (NOT column name — see note below). The dispatcher routes
    this shape based on ``FeatureDef.required_symbols``.

    Why dict-keyed-by-symbol vs the single-DataFrame contract: cross-asset
    features need two independent OHLCV histories. Stuffing both into one
    wide frame (e.g. ``close_price_BTC``, ``close_price_ETH``) would force
    every transformer to learn a symbol-suffixed column convention. A dict
    is cleaner — and matches blackheart-train's ``fetch_market_data_bundle``
    contract so train-time derived features and registry features share the
    same transformer shape.

    Args:
        close_col_a / close_col_b: the close column inside each symbol's
            wide frame. Same name for symmetric features (both
            ``close_price``); different names only if the two symbols come
            from different OHLCV schemas (not the case today).
        window_bars: rolling window for the correlation.

    Mirrors blackheart-train's ``_t_eth_btc_corr_24h`` semantics: explicit
    inner-join on the timestamp index before the rolling correlation so
    the window always has paired observations, then reindex back onto the
    primary symbol's grid.
    """
    # The first key in the dict is conventionally the OUTPUT-symbol's frame;
    # for eth_btc_corr_24h on BTCUSDT-stamped output, the reindex target is
    # the BTC index. Caller-side we'll pass required_symbols=("BTCUSDT",
    # "ETHUSDT") and the transformer below picks them up positionally.

    def _impl(md: dict[str, pd.DataFrame]) -> pd.Series:
        # Order-stable iteration: required_symbols order is preserved by the
        # dispatcher when building the dict, so the first symbol is the
        # output-symbol (BTC) and the second is the cross-asset input (ETH).
        symbols = list(md.keys())
        if len(symbols) != 2:
            raise ValueError(
                f"_cross_asset_correlation expected exactly 2 symbols; "
                f"got {symbols!r}"
            )
        sym_a, sym_b = symbols[0], symbols[1]
        close_a = md[sym_a][close_col_a].astype("float64")
        close_b = md[sym_b][close_col_b].astype("float64")
        ret_a = np.log(close_a / close_a.shift(1))
        ret_b = np.log(close_b / close_b.shift(1))
        # Inner-align: rows where either return is NaN drop out so the
        # rolling window sees only paired observations.
        paired = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()
        rho = paired["a"].rolling(window_bars).corr(paired["b"])
        # Reindex onto the OUTPUT symbol's index so the loader's
        # reindex(bar_index) step has the full index to project from. Bars
        # without paired observations remain NaN (compute()'s dropna()
        # filters them before persist).
        return rho.reindex(close_a.index)

    return _impl


def _threshold_flag(
    col: str, low: float, high: float
) -> Callable[[pd.DataFrame], pd.Series]:
    """Ternary extreme flag: +1 when value > high, -1 when < low, 0 otherwise.

    For taker buy ratio: >0.65 = buyer dominance, <0.35 = seller dominance.
    Encodes crowding extremes that a raw passthrough ratio can't expose as a
    discrete categorical signal.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        s = df[col].astype("float64")
        result = pd.Series(0.0, index=s.index)
        result = result.where(~(s > high), other=1.0)
        result = result.where(~(s < low), other=-1.0)
        return result.where(s.notna())

    return _impl


def _acceleration(col: str, periods: int) -> Callable[[pd.DataFrame], pd.Series]:
    """Second derivative of ``col``: diff of pct_change over ``periods`` rows.

    Captures the rate-of-change of a trend rather than the trend itself.
    On OI: positive = leverage build-up accelerating; negative = growth
    slowing or unwinding. Useful for detecting inflection points.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        s = df[col].astype("float64")
        velocity = s.pct_change(periods=periods)
        return velocity.diff(periods=periods)

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


def _forward_sharpe_binary_sign_train_compat(
    close_col: str, horizon_bars: int
) -> Callable[[pd.DataFrame], pd.Series]:
    """Bit-equivalent twin of ``_forward_sharpe_binary_sign`` matching
    blackheart-train's ``_t_label_regime_risk_on_24h`` pandas idiom.

    The difference: this version computes ``log_ret.shift(-N).rolling(N).
    std()`` (shift THEN rolling) — train-side's order — while the other
    version does ``log_ret.rolling(N).std().shift(-N)`` (rolling THEN
    shift). Both compute std of log_ret over the forward window for
    ``t >= N-1`` but diverge at the leading boundary: rolling-on-shifted
    is NaN for ``t < N-1`` (the rolling window has out-of-bounds rows on
    the left), while shifted-rolling produces valid values there.

    Used for ``label_regime_risk_on_24h`` v1 so the registry-resolved
    label is bit-equivalent to blackheart-train's derived label — a
    requirement for the regime_btc_v2 -> regime_btc_v3 spec swap to be
    safe by construction. Don't replace ``_forward_sharpe_binary_sign``
    with this — the existing v1 label features depend on the current
    idiom and must not change behavior.

    Train-side also multiplies fwd_std by ``np.sqrt(horizon_bars)``;
    since the output binarizes on ``sign(sharpe)`` and sqrt is positive,
    the multiplication is sign-preserving. We mirror it here for full
    semantic parity, but the resulting binary is identical with or
    without the factor.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        log_ret = np.log(c / c.shift(1))
        fwd_ret = (c.shift(-horizon_bars) - c) / c
        # shift FIRST, then rolling — this is the train-side order.
        fwd_std = (
            log_ret.shift(-horizon_bars).rolling(horizon_bars).std()
            * np.sqrt(horizon_bars)
        )
        sharpe = fwd_ret / fwd_std.where(fwd_std > 0)
        binary = (sharpe > 0).astype("float64")
        # Restore NaN where sharpe is NaN (insufficient future data) so
        # the engine's dropna() filters those rows out before persist.
        binary[sharpe.isna()] = np.nan
        return binary

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


def _rsi(close_col: str = "close_price", window: int = 14) -> Callable[[pd.DataFrame], pd.Series]:
    """Relative Strength Index over ``window`` bars.

    Standard definition: average gain / average loss over the trailing
    window, smoothed via SMA (Wilder's smoothing replaced with SMA for
    deterministic, closed-form computation matching the ``_atr`` precedent
    in this module). Bounded in [0, 100].

    PIT-safe: all inputs at row t come from rows <= t. NaN for the first
    ``window`` rows (insufficient history); engine's dropna() filters them.

    Path B (2026-05-21): adds bar-level entry-timing feature for
    directional_btc_1h_v4 — prior macro/24h-aggregation feature stack
    failed to provide bar-level signal (RUN_SUMMARY a957d7cf).
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        c = df[close_col].astype("float64")
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        # avoid divide-by-zero: when avg_loss is zero, RS is +infinity ->
        # RSI = 100; standard convention. NaN preserved by rolling().mean().
        rs = avg_gain / avg_loss.where(avg_loss > 0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # When avg_loss == 0 AND avg_gain > 0: rs is NaN (masked), rsi NaN.
        # Replace with 100.0 by the canonical convention.
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), other=100.0)
        # When avg_loss == 0 AND avg_gain == 0: flat market, RSI 50 by convention.
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), other=50.0)
        return rsi

    return _impl


def _atr_ratio(
    atr_window: int = 14,
    smoothing_window: int = 24,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Ratio of current ATR_n to its rolling mean over ``smoothing_window``
    bars. A regime-normalized volatility measure: > 1 means the current
    ATR is elevated vs the recent average, < 1 means compressed.

    PIT-safe: ATR_n at row t reads rows (t-n+1, t); the smoothing window
    further averages those ATRs over (t-smoothing+1, t). All inputs
    causal.

    Path B (2026-05-21): adds bar-level volatility-regime feature for
    directional_btc_1h_v4.
    """

    def _impl(df: pd.DataFrame) -> pd.Series:
        atr = _atr(df, atr_window)
        atr_avg = atr.rolling(smoothing_window, min_periods=smoothing_window).mean()
        return atr / atr_avg.where(atr_avg > 0)

    return _impl


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
    # ── funding_regime_v1 features (V121) ─────────────────────────────────
    FeatureDef(
        name="btc_funding_sign_streak",
        version=1,
        family="positioning",
        inputs=("binance_funding_rate_btcusdt",),
        transformer=_sign_streak("binance_funding_rate_btcusdt"),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="Signed streak of consecutive 8h funding bars with the same sign. "
        "Positive N means funding positive for N consecutive 8h periods (crowded longs); "
        "negative N means funding negative for N consecutive 8h periods (crowded shorts). "
        "V121 — funding_regime_v1 input.",
    ),
    FeatureDef(
        name="btc_funding_percentile_30d",
        version=1,
        family="positioning",
        # 30 days at 8h cadence = 90 rows.
        inputs=("binance_funding_rate_btcusdt",),
        transformer=_rolling_percentile_rank(
            "binance_funding_rate_btcusdt", window=90, min_periods=20
        ),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="30-day percentile rank of BTC 8h funding rate in [0,1]. "
        "window=90 rows (30d × 3 per day). High percentile = historically elevated "
        "longs-crowded; low = shorts-crowded. V121 — funding_regime_v1 input.",
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
        # V110 (2026-05-20): expanded to (BTCUSDT, ETHUSDT).
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="30-day annualized realized volatility of close-to-close log returns. "
        "Computed on 1h bars (window=720); annualization scales the rolling std "
        "by sqrt(8760) = sqrt(24 * 365). V110 added ETHUSDT scope. V123 added SOLUSDT+4h.",
    ),
    # ── Phase 4 / regime_btc_v2 derived features (4) — ported from
    # blackheart-train.derived_features so the live inference path can
    # source them from feature_values instead of computing at train time.
    # These match the four entries in DERIVED_FEATURES on the train side.
    FeatureDef(
        name="btc_log_return_24h",
        version=1,
        family="technical",
        inputs=("close_price",),
        transformer=_log_return(close_col="close_price", periods=24),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        # V110 (2026-05-20): added ETHUSDT so the same transformer runs
        # over ETH close_price and writes (feature_name, symbol=ETHUSDT)
        # rows to feature_values. The "btc_" name prefix is now a
        # historical scope artifact — the load-bearing symbol identifier
        # is the feature_values.symbol column. Renaming would require
        # spec re-registration on every downstream consumer.
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="24-bar log return of close. Originally ported from "
        "blackheart_train.derived_features._t_btc_log_return_24h for "
        "regime_btc_v2 deployment-readiness. V110 added ETHUSDT scope "
        "(same transformer, symbol-agnostic — operates on whatever "
        "close_price column comes in via market_data). V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_realized_vol_7d",
        version=1,
        family="technical",
        inputs=("close_price",),
        # Match blackheart-train: log_ret.rolling(168).std() with no
        # annualization. window_bars=168 = 7 days * 24h; min_periods=168
        # matches pandas' default rolling().std() behavior.
        transformer=_rolling_realized_vol(
            close_col="close_price",
            window_bars=168,
            min_periods=168,
            annualize_factor=None,
        ),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        # V110 (2026-05-20): expanded to (BTCUSDT, ETHUSDT) — see
        # btc_log_return_24h note re symbol-prefix-as-historical-artifact.
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="7-day realized volatility of log returns (no "
        "annualization). Originally ported from blackheart_train."
        "derived_features._t_btc_realized_vol_7d for regime_btc_v2. "
        "V110 added ETHUSDT scope. V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_volume_zscore_24h",
        version=1,
        family="technical",
        inputs=("volume",),
        # Match blackheart-train: 24-bar rolling z-score of volume. The
        # train-side uses std > 1e-12; ingest uses std > 0. For real BTC
        # volume data, std is always > 1e-12, so the difference produces
        # zero numerical disagreement. See _rolling_zscore.
        transformer=_rolling_zscore("volume", window=24, min_periods=24),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        # V110 (2026-05-20): expanded to (BTCUSDT, ETHUSDT).
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="24-bar rolling z-score of volume. Originally ported "
        "from blackheart_train.derived_features._t_btc_volume_zscore_24h "
        "for regime_btc_v2 deployment-readiness. V110 added ETHUSDT scope. V123 added SOLUSDT+4h.",
    ),
    # ── Path B (2026-05-21) — bar-level entry-timing features ────────
    # Added so directional_btc_1h_v4 can train on short-lookback,
    # bar-level signal instead of the 24h-aggregation macro stack that
    # prior v2 / v3 found provided no bar-level entry-timing edge
    # (RUN_SUMMARY a957d7cf). Stamped on BTCUSDT + ETHUSDT 1h bars.
    FeatureDef(
        name="btc_rsi_14_1h",
        version=1,
        family="technical",
        inputs=("close_price",),
        transformer=_rsi(close_col="close_price", window=14),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="14-bar Relative Strength Index on close, SMA-smoothed. "
        "Path B (2026-05-21) bar-level entry-timing feature for "
        "directional_btc_1h_v4. 'btc_' prefix is historical scope "
        "artifact; the symbol column is load-bearing. V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_atr_ratio_14_24",
        version=1,
        family="technical",
        # _atr_ratio reads high/low/close internally; declare all three
        # so the engine's inputs-vs-columns check passes for market_data.
        inputs=("close_price", "high_price", "low_price"),
        transformer=_atr_ratio(atr_window=14, smoothing_window=24),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="ATR_14 / mean(ATR_14, 24-bar window) — normalized "
        "intraday vol regime. > 1 = vol elevated vs recent baseline. "
        "Path B (2026-05-21). V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_log_return_1h",
        version=1,
        family="technical",
        inputs=("close_price",),
        transformer=_log_return(close_col="close_price", periods=1),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="1-bar log return on close (1h cadence). Bar-level "
        "momentum signal. Path B (2026-05-21). V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_log_return_4h",
        version=1,
        family="technical",
        inputs=("close_price",),
        transformer=_log_return(close_col="close_price", periods=4),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="4-bar log return on close (1h cadence, 4h horizon). "
        "Short-horizon momentum. Path B (2026-05-21). V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="btc_volume_zscore_4h",
        version=1,
        family="technical",
        inputs=("volume",),
        transformer=_rolling_zscore("volume", window=4, min_periods=4),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="4-bar rolling z-score of volume (1h cadence, 4h window). "
        "Short-horizon volume spike detector. Path B (2026-05-21). V123 added SOLUSDT+4h.",
    ),
    FeatureDef(
        name="eth_btc_corr_24h",
        version=1,
        family="cross_asset",
        inputs=("close_price",),
        transformer=_cross_asset_correlation(
            close_col_a="close_price",
            close_col_b="close_price",
            window_bars=24,
        ),
        pit_safe=True,
        ffill_policy=None,
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
        # Multi-symbol: needs BTC AND ETH bars. required_symbols order is
        # load-bearing — first symbol is the output-stamped one (BTC), the
        # rolling correlation is reindexed onto its grid.
        required_symbols=("BTCUSDT", "ETHUSDT"),
        description="24-bar rolling correlation between BTC and ETH 1h "
        "log returns. Cross-asset signal — when corr breaks down, the "
        "crypto regime is often shifting. Ported from blackheart_train."
        "derived_features._t_eth_btc_corr_24h for regime_btc_v2.",
    ),
    # ── Phase 4 / regime_btc_v2 derived LABEL ─────────────────────────
    FeatureDef(
        name="label_regime_risk_on_24h",
        version=1,
        family="label",
        inputs=("close_price",),
        # Uses the train-compat twin (shift-then-rolling) rather than the
        # default _forward_sharpe_binary_sign (rolling-then-shift). The two
        # are mathematically equivalent for t >= horizon-1 but diverge at
        # the leading boundary; train-compat matches blackheart-train's
        # _t_label_regime_risk_on_24h bit-for-bit so the regime_btc_v2 ->
        # v3 spec swap (reading from registry) preserves identical
        # training data.
        transformer=_forward_sharpe_binary_sign_train_compat(
            close_col="close_price", horizon_bars=24
        ),
        pit_safe=False,
        ffill_policy=None,
        raw_tables=("market_data",),
        # V110 (2026-05-20): expanded to (BTCUSDT, ETHUSDT) so an ETH ML
        # spec (e.g. regime_eth_v1) can train against the same forward-
        # Sharpe-sign label computed from ETH close_price.
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"),
        intervals=("1h", "4h"),
        description="Binary label: 1 if forward-24h Sharpe (return / vol) "
        "is positive, else 0. Bit-equivalent port of blackheart_train."
        "derived_features._t_label_regime_risk_on_24h for regime_btc_v2 "
        "deployment-readiness. Uses the shift-then-rolling pandas idiom. "
        "At 4h cadence horizon_bars=24 covers 96h forward regime. V123 added SOLUSDT+4h.",
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
    # ── V125: New BTC alpha surfaces (Rank 4 / 3 / 2 from data-scout) ────
    # Rank 4 — Taker imbalance momentum (zero new raw data, highest ROI)
    FeatureDef(
        name="btc_taker_imbalance_momentum_8h",
        version=1,
        family="positioning",
        inputs=("binance_taker_buy_sell_ratio_btcusdt_4h",),
        transformer=_change_pct("binance_taker_buy_sell_ratio_btcusdt_4h", periods=8),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="8-period % change of BTC 4h taker buy ratio (32h momentum window). "
        "Detects accelerating buy-side or sell-side crowding. V125.",
    ),
    FeatureDef(
        name="btc_taker_extreme_flag",
        version=1,
        family="positioning",
        inputs=("binance_taker_buy_sell_ratio_btcusdt_4h",),
        transformer=_threshold_flag(
            "binance_taker_buy_sell_ratio_btcusdt_4h", low=0.35, high=0.65
        ),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="+1 when 4h taker buy ratio >0.65 (buyer dominance), "
        "-1 when <0.35 (seller dominance), 0 otherwise. "
        "Flags positioning extremes as a discrete signal. V125.",
    ),
    # Rank 3 — OI term-structure acceleration (zero new raw data)
    FeatureDef(
        name="btc_oi_accel_4h",
        version=1,
        family="positioning",
        inputs=("binance_open_interest_btcusdt_1h",),
        transformer=_acceleration("binance_open_interest_btcusdt_1h", periods=4),
        ffill_policy="last_value",
        max_ffill_age_hours=24,
        description="Second derivative of BTC open interest: 4-period diff of "
        "4-period pct_change on 1h OI series. Positive = leverage build-up "
        "accelerating; negative = growth slowing or unwinding. V125.",
    ),
    # Rank 2 — On-chain free metrics (data already in DB via coinmetrics source)
    FeatureDef(
        name="btc_hashrate_momentum_30d",
        version=1,
        family="onchain",
        inputs=("coinmetrics_btc_hashrate",),
        transformer=_change_pct("coinmetrics_btc_hashrate", periods=30),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="30-day % change in BTC hash rate (CoinMetrics HashRate, daily). "
        "Rising hashrate signals miner confidence and network security growth. V125.",
    ),
    FeatureDef(
        name="btc_active_addr_momentum_14d",
        version=1,
        family="onchain",
        inputs=("coinmetrics_btc_adractcnt",),
        transformer=_change_pct("coinmetrics_btc_adractcnt", periods=14),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="14-day % change in BTC active address count (CoinMetrics AdrActCnt, "
        "daily). Proxy for on-chain demand and user activity momentum. V125.",
    ),
    FeatureDef(
        name="btc_txcnt_zscore_30d",
        version=1,
        family="onchain",
        inputs=("coinmetrics_btc_txcnt",),
        transformer=_rolling_zscore("coinmetrics_btc_txcnt", window=30, min_periods=10),
        ffill_policy="last_value",
        max_ffill_age_hours=72,
        description="30-day rolling z-score of BTC transaction count (CoinMetrics TxCnt, "
        "daily). Normalized on-chain activity level. V125.",
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
