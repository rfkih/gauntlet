"""Unit tests for the in-memory feature compute pieces.

Scoped to the deterministic pure-pandas functions:

* :func:`_apply_ffill`  — cap-aware forward-fill (subtle indexing logic)
* :func:`_pivot_wide`   — long-to-wide pivot with revision dedup

No DB connection required. These are the only pieces of the M3 engine
whose correctness can't be inspected by eye on the smoke output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackheart_ingest.features.compute import _apply_ffill, _pivot_wide
from blackheart_ingest.features.definitions import FeatureDef, _passthrough


def _feat(
    inputs: tuple[str, ...] = ("X",),
    ffill_policy: str | None = "last_value",
    max_ffill_age_hours: int | None = 24,
) -> FeatureDef:
    return FeatureDef(
        name="t",
        version=1,
        family="macro",
        inputs=inputs,
        transformer=_passthrough(inputs[0]),
        pit_safe=True,
        ffill_policy=ffill_policy,
        max_ffill_age_hours=max_ffill_age_hours,
    )


# ── _apply_ffill ────────────────────────────────────────────────────────────


def test_ffill_no_policy_returns_input_untouched():
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    df = pd.DataFrame({"X": [1.0, None, 3.0]}, index=idx)
    feat = _feat(ffill_policy=None)
    out = _apply_ffill(df, feat)
    assert out["X"].isna().sum() == 1
    assert out["X"].iloc[0] == 1.0
    assert out["X"].iloc[2] == 3.0


def test_ffill_fills_within_cap():
    # Gap of 12h, cap is 24h -> fill should propagate.
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 12:00", "2025-01-02 00:00"])
    df = pd.DataFrame({"X": [10.0, None, 30.0]}, index=idx)
    feat = _feat(max_ffill_age_hours=24)
    out = _apply_ffill(df, feat)
    assert out["X"].iloc[0] == 10.0
    assert out["X"].iloc[1] == 10.0  # filled from row 0
    assert out["X"].iloc[2] == 30.0


def test_ffill_skips_beyond_cap():
    # Gap of 48h, cap is 24h -> the NaN row stays NaN, not filled.
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-03 00:00", "2025-01-04 00:00"])
    df = pd.DataFrame({"X": [10.0, None, 30.0]}, index=idx)
    feat = _feat(max_ffill_age_hours=24)
    out = _apply_ffill(df, feat)
    assert out["X"].iloc[0] == 10.0
    assert pd.isna(out["X"].iloc[1])
    assert out["X"].iloc[2] == 30.0


def test_ffill_leading_nan_never_fills():
    # No prior value to carry forward — leading NaN stays NaN.
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    df = pd.DataFrame({"X": [None, None, 30.0]}, index=idx)
    feat = _feat(max_ffill_age_hours=72)
    out = _apply_ffill(df, feat)
    assert pd.isna(out["X"].iloc[0])
    assert pd.isna(out["X"].iloc[1])
    assert out["X"].iloc[2] == 30.0


def test_ffill_all_nan_column():
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    df = pd.DataFrame({"X": [None, None, None]}, index=idx)
    feat = _feat(max_ffill_age_hours=24)
    out = _apply_ffill(df, feat)
    assert out["X"].isna().all()


def test_ffill_cap_zero_means_no_fill():
    # Zero-hour cap means even a 1-second gap is too long. Only originally
    # non-NaN positions retain values.
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01", "2025-01-01 00:02"])
    df = pd.DataFrame({"X": [10.0, None, 30.0]}, index=idx)
    feat = _feat(max_ffill_age_hours=0)
    out = _apply_ffill(df, feat)
    assert out["X"].iloc[0] == 10.0
    assert pd.isna(out["X"].iloc[1])
    assert out["X"].iloc[2] == 30.0


def test_ffill_gap_exactly_at_cap_boundary():
    # 24h gap with cap = 24h -> "<=" semantics: fill is allowed.
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-02 00:00"])
    df = pd.DataFrame({"X": [10.0, None]}, index=idx)
    feat = _feat(max_ffill_age_hours=24)
    out = _apply_ffill(df, feat)
    assert out["X"].iloc[0] == 10.0
    assert out["X"].iloc[1] == 10.0


def test_ffill_unknown_policy_raises():
    from blackheart_ingest.features.compute import FeatureComputeError

    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame({"X": [1.0]}, index=idx)
    feat = FeatureDef(
        name="t",
        version=1,
        family="macro",
        inputs=("X",),
        transformer=_passthrough("X"),
        ffill_policy="weird_policy_name",  # type: ignore[arg-type]
    )
    with pytest.raises(FeatureComputeError, match="Unknown ffill_policy"):
        _apply_ffill(df, feat)


# ── _pivot_wide ─────────────────────────────────────────────────────────────


def test_pivot_revision_keeps_latest_ingestion_time():
    """When a series has two rows at the same event_time (a revision),
    the later ``ingestion_time`` wins — that's the operator-now view."""
    long_df = pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "event_time": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "ingestion_time": pd.to_datetime(["2025-01-02", "2025-01-05"]),
            "value": [10.0, 11.0],  # revised value
        }
    )
    feat = _feat(inputs=("A",))
    value_wide, ing_wide = _pivot_wide(long_df, feat)
    assert list(value_wide["A"].values) == [11.0]
    assert ing_wide["A"].iloc[0] == pd.Timestamp("2025-01-05")


def test_pivot_empty_long_df_returns_empty_wide_with_columns():
    long_df = pd.DataFrame(
        columns=["series_id", "event_time", "ingestion_time", "value"]
    )
    feat = _feat(inputs=("A", "B"))
    value_wide, ing_wide = _pivot_wide(long_df, feat)
    assert list(value_wide.columns) == ["A", "B"]
    assert value_wide.empty
    assert ing_wide.empty


def test_pivot_missing_input_column_filled_with_na():
    """Engine declares inputs ('A', 'B'); long_df only has A. Wide must
    still carry both columns (B all-NaN) so the transformer sees the
    promised schema."""
    long_df = pd.DataFrame(
        {
            "series_id": ["A"],
            "event_time": pd.to_datetime(["2025-01-01"]),
            "ingestion_time": pd.to_datetime(["2025-01-02"]),
            "value": [1.0],
        }
    )
    feat = _feat(inputs=("A", "B"))
    value_wide, _ = _pivot_wide(long_df, feat)
    assert list(value_wide.columns) == ["A", "B"]
    assert value_wide["A"].iloc[0] == 1.0
    assert pd.isna(value_wide["B"].iloc[0])


# ── _rolling_realized_vol ───────────────────────────────────────────────────


def test_rolling_realized_vol_constant_price_is_zero():
    """log-return of a flat price series is 0; realized vol is 0 too."""
    from blackheart_ingest.features.definitions import _rolling_realized_vol

    idx = pd.date_range("2025-01-01", periods=100, freq="1h")
    df = pd.DataFrame({"close_price": [100.0] * 100}, index=idx)
    out = _rolling_realized_vol("close_price", window_bars=24, min_periods=5)(df)
    # First row is NaN (no prior price), then either NaN until min_periods or 0.
    # After warmup, all values must be 0.
    assert (out.dropna() == 0.0).all()


def test_rolling_realized_vol_annualization_scales_by_sqrt():
    """Annualized vol = sigma * sqrt(annualize_factor). Compare a run with
    factor vs without."""
    from blackheart_ingest.features.definitions import _rolling_realized_vol

    rng = np.random.default_rng(seed=42)
    # Synthetic geometric brownian motion-ish series.
    log_rets = rng.normal(0, 0.01, size=500)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2025-01-01", periods=500, freq="1h")
    df = pd.DataFrame({"close_price": prices}, index=idx)

    raw = _rolling_realized_vol("close_price", window_bars=100, min_periods=50)(df)
    annual = _rolling_realized_vol(
        "close_price", window_bars=100, min_periods=50, annualize_factor=8760
    )(df)

    # Ratio of annualized to raw should be sqrt(8760) (modulo floating-point).
    ratio = (annual / raw).dropna()
    assert ratio.notna().sum() > 0
    np.testing.assert_allclose(ratio, np.sqrt(8760.0), rtol=1e-9)


def test_rolling_realized_vol_window_warmup_returns_nan():
    """Before min_periods rows of returns exist, result is NaN."""
    from blackheart_ingest.features.definitions import _rolling_realized_vol

    idx = pd.date_range("2025-01-01", periods=20, freq="1h")
    df = pd.DataFrame({"close_price": np.linspace(100, 110, 20)}, index=idx)
    out = _rolling_realized_vol("close_price", window_bars=10, min_periods=8)(df)
    # Returns start at row 1 (row 0 is NaN from .shift(1)); first non-NaN
    # rolling-std needs >= min_periods (8) RETURN observations, which means
    # at least row index 8 (returns 1..8 = 8 observations).
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[7])
    assert out.iloc[8:].notna().all()


# ── _instances_for ──────────────────────────────────────────────────────────


def test_instances_for_global_feature_one_instance():
    """A macro feature with empty symbols+intervals -> single (None, None)."""
    from blackheart_ingest.workers.compute_features import _instances_for

    feat = _feat()  # global by default
    inst = _instances_for(feat, cli_symbol=None, cli_interval=None)
    assert inst == [(None, None)]


def test_instances_for_per_bar_cross_product():
    """Declared symbols x intervals -> cross-product."""
    from blackheart_ingest.workers.compute_features import _instances_for

    feat = FeatureDef(
        name="t",
        version=1,
        family="market_structure",
        inputs=("close_price",),
        transformer=_passthrough("close_price"),
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1h", "4h"),
    )
    inst = _instances_for(feat, cli_symbol=None, cli_interval=None)
    assert set(inst) == {
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "4h"),
    }


def test_instances_for_cli_filter_narrows_set():
    """--symbol/--interval flags filter the declared cross-product."""
    from blackheart_ingest.workers.compute_features import _instances_for

    feat = FeatureDef(
        name="t",
        version=1,
        family="market_structure",
        inputs=("close_price",),
        transformer=_passthrough("close_price"),
        raw_tables=("market_data",),
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1h", "4h"),
    )
    inst = _instances_for(feat, cli_symbol="BTCUSDT", cli_interval="1h")
    assert inst == [("BTCUSDT", "1h")]


def test_instances_for_filter_empties_result():
    """No symbol match -> empty list; caller's responsibility to handle."""
    from blackheart_ingest.workers.compute_features import _instances_for

    feat = FeatureDef(
        name="t",
        version=1,
        family="market_structure",
        inputs=("close_price",),
        transformer=_passthrough("close_price"),
        raw_tables=("market_data",),
        symbols=("BTCUSDT",),
        intervals=("1h",),
    )
    inst = _instances_for(feat, cli_symbol="ETHUSDT", cli_interval="1h")
    assert inst == []


# ── FeatureDef validation ────────────────────────────────────────────────────


def test_featuredef_rejects_half_specified_scope():
    """symbols without intervals (or vice-versa) is a config bug."""
    with pytest.raises(ValueError, match="both empty .* or both set"):
        FeatureDef(
            name="bad",
            version=1,
            family="macro",
            inputs=("X",),
            transformer=_passthrough("X"),
            symbols=("BTCUSDT",),
            intervals=(),
        )
    with pytest.raises(ValueError, match="both empty .* or both set"):
        FeatureDef(
            name="bad",
            version=1,
            family="macro",
            inputs=("X",),
            transformer=_passthrough("X"),
            symbols=(),
            intervals=("1h",),
        )


# ── Forward-looking transformers (labels) ───────────────────────────────────


def test_forward_return_linear_price_matches_arithmetic():
    """On a price series with constant +1 per bar, forward return over
    horizon h equals h / current_price."""
    from blackheart_ingest.features.definitions import _forward_return

    idx = pd.date_range("2025-01-01", periods=20, freq="1h")
    # price 100, 101, ..., 119
    df = pd.DataFrame({"close_price": np.arange(100.0, 120.0)}, index=idx)
    out = _forward_return("close_price", horizon_bars=5)(df)
    # At row 0 (price 100), forward = (105 - 100) / 100 = 0.05
    assert abs(out.iloc[0] - 0.05) < 1e-9
    # At row 10 (price 110), forward = (115 - 110) / 110
    assert abs(out.iloc[10] - (5.0 / 110.0)) < 1e-9
    # Last 5 rows have no future data -> NaN
    assert out.iloc[-5:].isna().all()


def test_forward_return_anchored_at_decision_time_not_future():
    """The output index at row t is t (decision time), NOT t+horizon.
    Critical PIT invariant for label storage in feature_values."""
    from blackheart_ingest.features.definitions import _forward_return

    idx = pd.date_range("2025-01-01", periods=10, freq="1h")
    df = pd.DataFrame({"close_price": np.linspace(100, 110, 10)}, index=idx)
    out = _forward_return("close_price", horizon_bars=3)(df)
    # Output for row at idx[0] should be (close[idx[3]] - close[idx[0]]) / close[idx[0]]
    expected = (df["close_price"].iloc[3] - df["close_price"].iloc[0]) / df["close_price"].iloc[0]
    assert abs(out.loc[idx[0]] - expected) < 1e-9


def test_atr_synthetic_constant_range():
    """A series with HIGH-LOW=2 every bar and close at the midpoint -> ATR=2."""
    from blackheart_ingest.features.definitions import _atr

    idx = pd.date_range("2025-01-01", periods=20, freq="1h")
    base = 100.0
    df = pd.DataFrame(
        {
            "high_price": [base + 1.0] * 20,
            "low_price": [base - 1.0] * 20,
            "close_price": [base] * 20,
        },
        index=idx,
    )
    atr = _atr(df, n=5)
    # After warmup, ATR settles at 2.0 (the constant true range).
    assert abs(atr.iloc[-1] - 2.0) < 1e-9


def test_forward_meanrev_atr_clip_bounds():
    """The label is clipped to +/-clip_value regardless of underlying magnitude."""
    from blackheart_ingest.features.definitions import _forward_meanrev_atr

    idx = pd.date_range("2025-01-01", periods=30, freq="1h")
    # Massive future jump in close, small ATR -> raw value would exceed clip.
    closes = np.concatenate([np.full(20, 100.0), np.full(10, 1000.0)])
    df = pd.DataFrame(
        {
            "close_price": closes,
            "high_price": closes + 0.1,
            "low_price": closes - 0.1,
        },
        index=idx,
    )
    out = _forward_meanrev_atr(
        close_col="close_price", horizon_bars=5, atr_window=5, clip_value=3.0
    )(df)
    # Every defined value must respect the clip range.
    defined = out.dropna()
    assert (defined.between(-3.0, 3.0)).all()


def test_forward_sharpe_binary_nan_preserved_at_tail():
    """At rows where forward data is incomplete (last `horizon` rows), the
    binary label must be NaN, NOT 0. dropna() in compute() depends on this."""
    from blackheart_ingest.features.definitions import _forward_sharpe_binary_sign

    rng = np.random.default_rng(seed=7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=100)))
    idx = pd.date_range("2025-01-01", periods=100, freq="1h")
    df = pd.DataFrame({"close_price": closes}, index=idx)
    out = _forward_sharpe_binary_sign("close_price", horizon_bars=20)(df)
    # Last 20 rows -> NaN (no future close).
    assert out.iloc[-20:].isna().all()
    # Middle rows must be 0.0 or 1.0 only.
    middle = out.iloc[25:75].dropna()
    assert set(middle.unique().tolist()) <= {0.0, 1.0}


def test_forward_sharpe_binary_positive_drift_is_1():
    """Strongly trending up price -> every forward Sharpe positive -> label 1."""
    from blackheart_ingest.features.definitions import _forward_sharpe_binary_sign

    idx = pd.date_range("2025-01-01", periods=50, freq="1h")
    # Steady 1% per bar, no noise -> forward return > 0, forward vol > 0,
    # ratio > 0, label = 1.
    closes = 100.0 * (1.01 ** np.arange(50))
    df = pd.DataFrame({"close_price": closes}, index=idx)
    out = _forward_sharpe_binary_sign("close_price", horizon_bars=10)(df)
    # First 30-ish rows have enough data; all should be 1.
    defined = out.dropna()
    assert (defined == 1.0).all()


# ── _forward_triple_barrier ─────────────────────────────────────────────────


def _make_ohlc(closes, *, hi_offset=0.5, lo_offset=0.5):
    """Helper: synthetic OHLC frame from a close series."""
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "close_price": closes,
            "high_price": closes + hi_offset,
            "low_price": closes - lo_offset,
        },
        index=idx,
    )


def test_triple_barrier_pure_tp_hit():
    """Strong upward move within horizon -> +1 (TP hit before SL)."""
    from blackheart_ingest.features.definitions import _forward_triple_barrier

    # Stable warmup so ATR_5 ~= 1.0 (range ±0.5), then a big jump up at t=10.
    closes = np.concatenate([np.full(10, 100.0), np.full(15, 120.0)])
    df = _make_ohlc(closes)
    out = _forward_triple_barrier(
        horizon_bars=5, k_tp=1.5, k_sl=1.0, atr_window=5
    )(df)
    # At t=9 (last warmup bar, close=100), atr≈1.0. tp=101.5, sl=99.0.
    # First forward bar (t=10) jumps to 120 -> high=120.5 >= tp, low=119.5 > sl.
    # Pure TP hit -> +1.
    assert out.iloc[9] == 1.0


def test_triple_barrier_pure_sl_hit():
    """Strong downward move within horizon -> -1 (SL hit)."""
    from blackheart_ingest.features.definitions import _forward_triple_barrier

    closes = np.concatenate([np.full(10, 100.0), np.full(15, 80.0)])
    df = _make_ohlc(closes)
    out = _forward_triple_barrier(
        horizon_bars=5, k_tp=1.5, k_sl=1.0, atr_window=5
    )(df)
    # At t=9, atr≈1.0. sl=99.0. Next bar low=79.5 < sl -> -1.
    assert out.iloc[9] == -1.0


def test_triple_barrier_conservative_tie_sl_wins():
    """Bar where BOTH high>=tp AND low<=sl -> conservative SL fill (-1)."""
    from blackheart_ingest.features.definitions import _forward_triple_barrier

    # Stable warmup so atr stays small (range ±0.5 -> atr~1.0).
    # Then a "wide" bar at t=10 that spans well above tp and below sl.
    closes = np.full(25, 100.0)
    closes_list = closes.copy()
    df = _make_ohlc(closes_list)
    # Manually widen t=10's range to straddle both barriers (close stays 100,
    # but high jumps to 110 and low drops to 90 -> both tp=101.5 and sl=99.0
    # are hit in this single bar).
    df.loc[df.index[10], "high_price"] = 110.0
    df.loc[df.index[10], "low_price"] = 90.0
    out = _forward_triple_barrier(
        horizon_bars=5, k_tp=1.5, k_sl=1.0, atr_window=5
    )(df)
    # Entry at t=9, atr~1.0; tp=101.5, sl=99.0. t=10 bar straddles both.
    # Conservative rule: SL wins -> -1.
    assert out.iloc[9] == -1.0


def test_triple_barrier_horizon_timeout_returns_zero():
    """Flat price all the way through horizon -> 0 (neither barrier hit)."""
    from blackheart_ingest.features.definitions import _forward_triple_barrier

    closes = np.full(20, 100.0)  # flat
    df = _make_ohlc(closes, hi_offset=0.4, lo_offset=0.4)  # range too tight for ATR jumps
    out = _forward_triple_barrier(
        horizon_bars=5, k_tp=1.5, k_sl=1.0, atr_window=5
    )(df)
    # After warmup, atr ~= 0.8. tp = 100 + 1.5*0.8 = 101.2.  high = 100.4 < tp.
    # sl = 100 - 0.8 = 99.2. low = 99.6 > sl. Neither hit over 5 bars -> 0.
    defined = out.dropna()
    assert (defined == 0.0).all()


def test_triple_barrier_tail_rows_are_nan():
    """Last horizon_bars rows have no full future window -> NaN."""
    from blackheart_ingest.features.definitions import _forward_triple_barrier

    closes = np.full(20, 100.0)
    df = _make_ohlc(closes)
    horizon = 5
    out = _forward_triple_barrier(
        horizon_bars=horizon, k_tp=1.5, k_sl=1.0, atr_window=3
    )(df)
    # Last `horizon` rows: future window walks past end of data -> NaN.
    assert out.iloc[-horizon:].isna().all()
