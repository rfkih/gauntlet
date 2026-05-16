"""Pure-function tests for the inference module.

Covers the deterministic pieces that don't need a DB:

* ``persist.persist_predictions``    — NaN filtering, source whitelist
* ``api._reconstruct_spec``           — payload['spec'] dict -> ModelSpec
* ``api._booster_predict``            — binary / regression / multiclass mapping
* End-to-end smoke against the v3 on-disk artifact (no DB) — verifies
  the path-injection works and the booster predicts on a synthetic frame.

DB-touching tests are covered by the actual backfill run + signal_history
spot checks. Adding them here would require introducing pytest-postgresql
to this repo, which it doesn't currently use.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from blackheart_ingest.inference.api import _booster_predict, _reconstruct_spec
from blackheart_ingest.inference.artifacts import ModelSpec, load_artifact
from blackheart_ingest.inference.persist import (
    _ALLOWED_SOURCES,
    persist_predictions,
)


# ── persist.persist_predictions ─────────────────────────────────────────


def test_persist_rejects_invalid_source():
    with pytest.raises(ValueError, match="source="):
        persist_predictions([], conn=None, source="bogus_source")  # type: ignore[arg-type]


def test_persist_allowed_sources_match_schema():
    # V66's chk_signal_history_source CHECK constraint.
    assert _ALLOWED_SOURCES == frozenset({"stream", "catchup_scan", "historical_replay"})


def test_persist_returns_zero_on_empty_rows():
    # Empty input never opens a cursor, so conn=None is safe.
    assert persist_predictions([], conn=None, source="historical_replay") == 0  # type: ignore[arg-type]


class _StubCursor:
    """Minimal cursor stub that records executemany() params."""

    def __init__(self) -> None:
        self.executed_sql: str | None = None
        self.executed_params: list[dict] | None = None

    def __enter__(self) -> "_StubCursor":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def executemany(self, sql: str, params) -> None:
        self.executed_sql = sql
        self.executed_params = list(params)


class _StubConn:
    """Minimal connection stub that surfaces a single cursor."""

    def __init__(self) -> None:
        self.cur = _StubCursor()
        self.commits = 0

    def cursor(self) -> _StubCursor:
        return self.cur

    def commit(self) -> None:
        self.commits += 1


def test_persist_filters_nan_value_rows():
    sig_id = uuid4()
    rows = [
        {"signal_id": sig_id, "symbol": "BTCUSDT", "ts": datetime(2025, 1, 1), "value": 0.7},
        {"signal_id": sig_id, "symbol": "BTCUSDT", "ts": datetime(2025, 1, 2), "value": None},
        {"signal_id": sig_id, "symbol": "BTCUSDT", "ts": datetime(2025, 1, 3), "value": 0.3},
    ]
    conn = _StubConn()
    written = persist_predictions(rows, conn=conn, source="historical_replay")  # type: ignore[arg-type]
    assert written == 2
    params = conn.cur.executed_params
    assert params is not None
    assert len(params) == 2
    assert all(p["value"] is not None for p in params)
    assert conn.commits == 1


def test_persist_meta_is_json_encoded():
    sig_id = uuid4()
    rows = [
        {
            "signal_id": sig_id,
            "symbol": "BTCUSDT",
            "ts": datetime(2025, 1, 1),
            "value": 0.5,
            "confidence": 0.0,
            "meta": {"content_sha256": "abc", "n": 17, "nested": {"k": "v"}},
        }
    ]
    conn = _StubConn()
    persist_predictions(rows, conn=conn, source="historical_replay")  # type: ignore[arg-type]
    params = conn.cur.executed_params
    assert params is not None
    # meta must serialise as a JSON string so the SQL's ::jsonb cast works.
    import json
    assert isinstance(params[0]["meta"], str)
    decoded = json.loads(params[0]["meta"])
    assert decoded == {"content_sha256": "abc", "n": 17, "nested": {"k": "v"}}


# ── api._reconstruct_spec ────────────────────────────────────────────────


def test_reconstruct_spec_roundtrip_basic():
    """Serialised spec dict (from dataclasses.asdict + datetime stringify)
    rebuilds into a working ModelSpec."""
    serialised = {
        "name": "regime_btc_v3",
        "purpose": "regime",
        "label_feature": "label_regime_risk_on_24h",
        "label_version": 1,
        "objective": "binary",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "train_start": "2024-12-01 00:00:00",   # iso-like
        "train_end": "2026-05-14 00:00:00",
        "val_fraction": 0.2,
        "hyperparams": {"num_leaves": 31, "random_state": 42},
        "derived_features": [],
        "base_models": ["lightgbm"],
        "meta_label_enabled": False,
        "training_intervals": [],
        "feature_selection_enabled": False,
    }
    spec = _reconstruct_spec(serialised)
    assert isinstance(spec, ModelSpec)
    assert spec.name == "regime_btc_v3"
    assert spec.train_start == datetime(2024, 12, 1)
    assert spec.train_end == datetime(2026, 5, 14)
    assert spec.derived_features == ()
    assert spec.base_models == ("lightgbm",)
    assert spec.training_intervals == ()


def test_reconstruct_spec_idempotent_when_already_typed():
    """If the dict already has datetime/tuple values (e.g. caller already
    typed them), reconstruction must not double-convert."""
    serialised = {
        "name": "regime_btc_v3",
        "purpose": "regime",
        "label_feature": "label_regime_risk_on_24h",
        "label_version": 1,
        "objective": "binary",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "train_start": datetime(2024, 12, 1),
        "train_end": datetime(2026, 5, 14),
        "val_fraction": 0.2,
        "hyperparams": {"num_leaves": 31},
        "derived_features": (),
        "base_models": ("lightgbm",),
        "meta_label_enabled": False,
        "training_intervals": (),
        "feature_selection_enabled": False,
    }
    spec = _reconstruct_spec(serialised)
    assert spec.train_start == datetime(2024, 12, 1)
    assert spec.derived_features == ()


# ── api._booster_predict ──────────────────────────────────────────────────


class _StubBooster:
    """Pretend LightGBM booster — returns a canned array."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def predict(self, X) -> np.ndarray:
        return self._output


def test_booster_predict_binary_returns_proba_and_confidence():
    booster = _StubBooster(np.array([0.5, 0.9, 0.1, 1.0, 0.0]))
    X = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
    vals, conf = _booster_predict(booster, X, "binary")
    np.testing.assert_array_equal(vals, [0.5, 0.9, 0.1, 1.0, 0.0])
    # confidence = |p - 0.5| * 2
    np.testing.assert_array_almost_equal(conf, [0.0, 0.8, 0.8, 1.0, 1.0])


def test_booster_predict_regression_returns_values_and_no_confidence():
    booster = _StubBooster(np.array([1.5, -2.3, 0.0]))
    X = pd.DataFrame({"f": [1, 2, 3]})
    vals, conf = _booster_predict(booster, X, "regression")
    np.testing.assert_array_equal(vals, [1.5, -2.3, 0.0])
    assert conf is None


def test_booster_predict_multiclass_returns_argmax_and_max_proba():
    softmax = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ]
    )
    booster = _StubBooster(softmax)
    X = pd.DataFrame({"f": [1, 2, 3]})
    vals, conf = _booster_predict(booster, X, "multiclass")
    np.testing.assert_array_equal(vals, [0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(conf, [0.7, 0.8, 0.4])


def test_booster_predict_unknown_objective_raises():
    booster = _StubBooster(np.array([0.5]))
    with pytest.raises(ValueError, match="Unsupported objective"):
        _booster_predict(booster, pd.DataFrame({"f": [1]}), "ranking")


# ── End-to-end smoke against v3 on-disk artifact (no DB) ─────────────────


_V3_SHA = "06e40477cab164a6c75d05bba5983bcdf2bc9600d3d7487f6611a6e161df7b09"


def test_v3_artifact_loads_and_has_expected_shape():
    """Cross-validates the path-injection: blackheart-train must be
    importable from blackheart-ingest's venv via the sys.path tweak in
    inference.artifacts."""
    try:
        payload = load_artifact(_V3_SHA)
    except FileNotFoundError:
        pytest.skip(
            f"v3 artifact not present at default path. "
            f"This test asserts the artifact-loading round-trip; run "
            f"`blackheart_train.cli --model regime_btc_v3 --walk-forward "
            f"--gauntlet --register` to produce it."
        )
    assert payload["content_sha256"] == _V3_SHA
    assert payload["objective"] == "binary"
    assert payload["label_feature"] == "label_regime_risk_on_24h"
    # Trained v3 has exactly 20 input features (post-label-leak-fix).
    assert len(payload["feature_names"]) == 20
    assert "label_regime_risk_on_24h" not in payload["feature_names"], (
        "v3's feature_names must NOT include its own label — the loader fix "
        "in _list_input_features should have excluded it."
    )
    assert payload["deployment_readiness"]["deployment_ready"] is True
