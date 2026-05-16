"""Numerical equivalence: blackheart-train derived transformers vs
blackheart-ingest registry transformers.

The four derived features + one derived label that regime_btc_v2 needs are
computed two ways:

* train-time, by ``blackheart_train.derived_features._t_*``
* registry-resolved, by ``blackheart_ingest.features.definitions`` factories
  (the ones V77 wires up)

This test asserts the two implementations agree at every non-NaN
timestamp. Without it, swapping the source mid-spec (the Session 2 train-
side change — set ``spec.derived_features=[]`` and read from
feature_values) could silently change model outputs after promotion. The
``feature_registry_completion`` memo flags this as the load-bearing check.

Pure-pandas: synthesize a deterministic ~3-month BTC+ETH OHLCV bundle,
run both implementations on it, compare. No DB required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# blackheart-train lives next to blackheart-ingest but isn't pip-installed
# in this venv. Path-inject so we can import its transformers for the
# equivalence comparison without touching the install.
_TRAIN_SRC = Path(__file__).resolve().parents[2] / "blackheart-train" / "src"
if str(_TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SRC))

from blackheart_train.derived_features import (  # noqa: E402
    _t_btc_log_return_24h,
    _t_btc_realized_vol_7d,
    _t_btc_volume_zscore_24h,
    _t_eth_btc_corr_24h,
    _t_label_regime_risk_on_24h,
)

from blackheart_ingest.features.definitions import (  # noqa: E402
    _cross_asset_correlation,
    _forward_sharpe_binary_sign_train_compat,
    _log_return,
    _rolling_realized_vol,
    _rolling_zscore,
)


# ── Fixture: deterministic BTC + ETH OHLCV ───────────────────────────────


@pytest.fixture(scope="module")
def market_data_bundle() -> dict[str, pd.DataFrame]:
    """Two synthetic OHLCV series at 1h cadence — long enough to exercise
    the 168-bar realized-vol window plus 24-bar correlation and forward
    label windows."""
    rng = np.random.default_rng(seed=20260516)
    n = 2_000  # ~83 days at 1h cadence — well past the longest window
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")

    # Geometric Brownian close prices (BTC ~70k, ETH ~3.5k) with mild
    # cross-asset correlation injected so the corr feature has signal to
    # find but doesn't degenerate to corr=1.
    eps_btc = rng.normal(0, 0.005, n)
    eps_eth_indep = rng.normal(0, 0.007, n)
    eps_eth = 0.6 * eps_btc + 0.8 * eps_eth_indep  # rho ~0.6 in expectation
    btc_close = 70_000 * np.exp(np.cumsum(eps_btc))
    eth_close = 3_500 * np.exp(np.cumsum(eps_eth))

    def _build(close: np.ndarray) -> pd.DataFrame:
        # Synthesize OHLV consistent with close: high/low straddle close by
        # a small noise amount; volume is lognormal-distributed.
        wig = rng.uniform(0.0005, 0.003, n)
        high = close * (1 + wig)
        low = close * (1 - wig)
        # open[t] = close[t-1] (gap-free); first bar's open = first close.
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.lognormal(mean=7.0, sigma=0.4, size=n)  # ~thousands
        return pd.DataFrame(
            {
                "open_price": open_,
                "high_price": high,
                "low_price": low,
                "close_price": close,
                "volume": volume,
                # The remaining market_data columns aren't read by any of
                # the 5 transformers under test; fill with zeros to keep the
                # frame shape consistent with what _read_market_data_wide
                # would return.
                "quote_asset_volume": np.zeros(n),
                "taker_buy_base_volume": np.zeros(n),
                "taker_buy_quote_volume": np.zeros(n),
                "trade_count": np.zeros(n, dtype="float64"),
            },
            index=idx,
        )

    return {"BTCUSDT": _build(btc_close), "ETHUSDT": _build(eth_close)}


# ── Equivalence helpers ──────────────────────────────────────────────────


def _assert_series_equiv(
    train_out: pd.Series,
    ingest_out: pd.Series,
    *,
    name: str,
    atol: float = 1e-12,
) -> None:
    """Compare two pandas Series at every index where BOTH are non-NaN.

    Different windowing or threshold conventions can leave train-side NaN
    where ingest-side has a value (or vice versa) at the boundaries; we
    only require agreement where both implementations emit a value. NaN
    masks must agree too — otherwise one path is silently dropping data
    the other keeps.
    """
    train_out, ingest_out = train_out.align(ingest_out, join="outer")
    train_nan = train_out.isna()
    ingest_nan = ingest_out.isna()
    nan_mismatches = (train_nan != ingest_nan)
    assert not nan_mismatches.any(), (
        f"{name}: NaN masks disagree at "
        f"{int(nan_mismatches.sum())} timestamp(s) — implementations are "
        f"dropping different rows."
    )

    both_valid = (~train_nan) & (~ingest_nan)
    assert both_valid.any(), f"{name}: no timestamps where both produced values"

    diff = (train_out[both_valid] - ingest_out[both_valid]).abs()
    max_diff = float(diff.max())
    assert max_diff <= atol, (
        f"{name}: max abs diff = {max_diff:.3e} > atol={atol:.3e}. "
        f"Top 5 disagreements:\n{diff.sort_values(ascending=False).head().to_string()}"
    )


# ── Per-feature equivalence tests ─────────────────────────────────────────


def test_btc_log_return_24h_equivalence(market_data_bundle):
    train_out = _t_btc_log_return_24h(market_data_bundle)
    ingest_fn = _log_return(close_col="close_price", periods=24)
    ingest_out = ingest_fn(market_data_bundle["BTCUSDT"])
    _assert_series_equiv(train_out, ingest_out, name="btc_log_return_24h")


def test_btc_realized_vol_7d_equivalence(market_data_bundle):
    train_out = _t_btc_realized_vol_7d(market_data_bundle)
    ingest_fn = _rolling_realized_vol(
        close_col="close_price",
        window_bars=168,
        min_periods=168,
        annualize_factor=None,
    )
    ingest_out = ingest_fn(market_data_bundle["BTCUSDT"])
    _assert_series_equiv(train_out, ingest_out, name="btc_realized_vol_7d")


def test_btc_volume_zscore_24h_equivalence(market_data_bundle):
    train_out = _t_btc_volume_zscore_24h(market_data_bundle)
    ingest_fn = _rolling_zscore("volume", window=24, min_periods=24)
    ingest_out = ingest_fn(market_data_bundle["BTCUSDT"])
    # Train-side uses std > 1e-12, ingest uses std > 0. For real volume
    # data std is always well above 1e-12 — but the synthetic lognormal
    # volume here also lives in that regime, so the threshold difference
    # produces zero disagreement. Tight tolerance.
    _assert_series_equiv(
        train_out, ingest_out, name="btc_volume_zscore_24h", atol=1e-12
    )


def test_eth_btc_corr_24h_equivalence(market_data_bundle):
    train_out = _t_eth_btc_corr_24h(market_data_bundle)
    ingest_fn = _cross_asset_correlation(
        close_col_a="close_price",
        close_col_b="close_price",
        window_bars=24,
    )
    # Mirror the dispatcher's required_symbols ordering: BTC first, ETH
    # second. The transformer reindexes onto symbols[0]'s grid, so
    # ordering matters.
    ingest_out = ingest_fn(
        {"BTCUSDT": market_data_bundle["BTCUSDT"], "ETHUSDT": market_data_bundle["ETHUSDT"]}
    )
    _assert_series_equiv(train_out, ingest_out, name="eth_btc_corr_24h")


def test_label_regime_risk_on_24h_equivalence(market_data_bundle):
    """Bit-equivalent twin: ingest's _forward_sharpe_binary_sign_train_
    compat mirrors blackheart-train's pandas idiom (shift THEN rolling,
    plus the sqrt(N) scaling) so the output matches at every timestamp
    including the leading 23 rows that the rolling-then-shift idiom
    drops as NaN."""
    train_out = _t_label_regime_risk_on_24h(market_data_bundle)
    ingest_fn = _forward_sharpe_binary_sign_train_compat(
        close_col="close_price", horizon_bars=24
    )
    ingest_out = ingest_fn(market_data_bundle["BTCUSDT"])
    _assert_series_equiv(
        train_out.astype("float64"),
        ingest_out.astype("float64"),
        name="label_regime_risk_on_24h",
    )


# ── Sanity: 100-timestamp spot check ─────────────────────────────────────


def test_100_random_timestamps_spot_check(market_data_bundle):
    """The feature_registry_completion memo specifically asks for a 100-
    random-timestamp spot check. The per-feature tests above already
    exercise the full series; this test is the explicit memo-requested
    sample.
    """
    rng = np.random.default_rng(seed=42)
    btc = market_data_bundle["BTCUSDT"]
    # Pick 100 rows where every feature should have a value: skip the
    # first 168 (vol warmup) and last 24 (forward-label tail).
    valid_range = np.arange(168, len(btc) - 24)
    sample_idx = rng.choice(valid_range, size=100, replace=False)
    sample_ts = btc.index[sample_idx]

    pairs = [
        (
            "btc_log_return_24h",
            _t_btc_log_return_24h(market_data_bundle),
            _log_return("close_price", 24)(btc),
        ),
        (
            "btc_realized_vol_7d",
            _t_btc_realized_vol_7d(market_data_bundle),
            _rolling_realized_vol("close_price", 168, 168, None)(btc),
        ),
        (
            "btc_volume_zscore_24h",
            _t_btc_volume_zscore_24h(market_data_bundle),
            _rolling_zscore("volume", 24, 24)(btc),
        ),
        (
            "eth_btc_corr_24h",
            _t_eth_btc_corr_24h(market_data_bundle),
            _cross_asset_correlation("close_price", "close_price", 24)(
                {"BTCUSDT": btc, "ETHUSDT": market_data_bundle["ETHUSDT"]}
            ),
        ),
        (
            "label_regime_risk_on_24h",
            _t_label_regime_risk_on_24h(market_data_bundle).astype("float64"),
            _forward_sharpe_binary_sign_train_compat("close_price", 24)(btc).astype(
                "float64"
            ),
        ),
    ]

    for name, train_out, ingest_out in pairs:
        t_vals = train_out.reindex(sample_ts)
        i_vals = ingest_out.reindex(sample_ts)
        diffs = (t_vals - i_vals).abs()
        # NaN at the same indices in both is fine; we use fillna(0) on the
        # equality check below to skip those.
        nan_match = t_vals.isna() == i_vals.isna()
        assert nan_match.all(), (
            f"{name}: NaN mismatch in 100-sample spot check"
        )
        max_diff = float(diffs.fillna(0.0).max())
        assert max_diff <= 1e-12, (
            f"{name}: 100-sample spot-check max diff = {max_diff:.3e}"
        )
