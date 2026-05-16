"""High-level inference API: model + window -> prediction rows.

Reuses ``blackheart_train.loader.load_dataset`` for feature assembly to
guarantee bit-equivalence with training-time predictions. The training
loader is parameterised by a ``ModelSpec``; we reconstruct that spec
from ``payload['spec']`` and substitute the inference window's
start/end so the loader fetches only the bars we need.

Numerical equivalence guarantee:

Training does:
    load_dataset(spec).X -> booster.fit(X, y)
Inference does:
    load_dataset(spec_with_inference_window).X -> booster.predict(X)

The X matrices share assembly code, ordering (alphabetical from registry +
derived appended), and ffill rules. Predictions made at training-window
timestamps are bit-equivalent to the training-time out-of-fold predictions
the model would emit on those same rows — verified by the equivalence
test (test_inference_equivalence.py).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from .artifacts import ModelSpec, load_artifact

logger = logging.getLogger(__name__)


def _reconstruct_spec(payload_spec: dict[str, Any]) -> ModelSpec:
    """Rebuild a ModelSpec dataclass from its serialised dict form.

    ``payload['spec']`` is produced by ``dataclasses.asdict(spec)`` at
    training time. Re-instantiating via ``ModelSpec(**spec_dict)``
    works as long as the field set is unchanged. Datetime fields get
    ISO-formatted in the serialised form (because compute_content_sha
    uses ``default=str``) — we have to parse them back.

    Tuple fields (derived_features, base_models, training_intervals) are
    serialised as lists by dataclasses.asdict; we re-tuple them since
    ModelSpec is frozen and expects tuples for hashability.
    """
    fields = dict(payload_spec)  # copy — don't mutate caller's dict

    # Datetimes: training serialises via str(datetime). pd.to_datetime
    # is the safest round-trip — it handles both isoformat strings and
    # already-datetime objects.
    for date_field in ("train_start", "train_end"):
        if isinstance(fields.get(date_field), str):
            fields[date_field] = pd.Timestamp(fields[date_field]).to_pydatetime()

    # Tuple fields: dataclasses.asdict turns tuples into lists.
    for tup_field in ("derived_features", "base_models", "training_intervals"):
        if tup_field in fields and isinstance(fields[tup_field], list):
            fields[tup_field] = tuple(fields[tup_field])

    return ModelSpec(**fields)


def _booster_predict(
    booster: Any,
    X: pd.DataFrame,
    objective: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run prediction. Returns ``(values, confidences)``.

    Mapping by objective:

    * ``binary``     -> values = P(class=1); confidence = |P - 0.5|*2
                       in [0, 1] (1.0 = highly confident in either class,
                       0 = coin flip)
    * ``regression`` -> values = raw forecast; confidence = None
    * ``multiclass`` -> values = argmax class index;
                       confidence = max softmax probability

    The mapping mirrors the architecture-doc semantics for
    ``Booster.predict`` and keeps signal_history.value as a scalar
    per row regardless of objective.
    """
    preds = booster.predict(X)
    if objective == "binary":
        # LightGBM binary returns class-1 proba in [0,1].
        values = np.asarray(preds, dtype="float64")
        confidence = np.abs(values - 0.5) * 2.0
        return values, confidence
    if objective == "regression":
        return np.asarray(preds, dtype="float64"), None
    if objective == "multiclass":
        # Softmax per row -> argmax + max-proba.
        arr = np.asarray(preds, dtype="float64")
        if arr.ndim != 2:
            raise ValueError(
                f"multiclass booster.predict returned ndim={arr.ndim}; "
                f"expected 2-D softmax. shape={arr.shape}"
            )
        values = arr.argmax(axis=1).astype("float64")
        confidence = arr.max(axis=1)
        return values, confidence
    raise ValueError(f"Unsupported objective for inference: {objective!r}")


def compute_predictions(
    *,
    content_sha: str,
    signal_id: UUID,
    inference_start: datetime,
    inference_end: datetime,
    artifact_dir=None,
) -> pd.DataFrame:
    """Compute predictions for one model over an inference window.

    Returns a DataFrame with columns ``[signal_id, symbol, ts, value,
    confidence, meta]`` — ready to pass to
    :func:`persist_predictions`. Rows with NaN values (insufficient
    feature coverage at that ts) are kept in the returned frame so the
    caller can inspect them; ``persist_predictions`` filters them out
    before INSERT.

    Inference window:
        ``inference_start`` and ``inference_end`` follow the same
        half-open convention as the training loader: ``[start, end)``.
        Set them to the training-window range to backfill predictions
        over the full history; set them to a small forward window
        (e.g. the latest bar) for forward-streaming.

    DB connection scope:
        This function does NOT take a caller-provided connection. The
        feature-matrix assembly delegates to
        ``blackheart_train.loader.load_dataset``, which opens its own
        connection via blackheart-train's settings (TRAIN_DB_*) so the
        load runs as ``blackheart_research`` for read access to
        feature_values. The CLI worker's conn (running as
        blackheart_trading) is used separately for the
        ``signal_history`` write. Two short transactions, two roles,
        clean separation.
    """
    payload = load_artifact(content_sha, artifact_dir=artifact_dir)
    booster = payload.get("booster")
    if booster is None:
        # Ensemble path: payload['ensemble'] holds the multi-base wrapper.
        # The blueprint § 6.2 directional model uses this path; modulators
        # (regime/positioning/flow v2 + v3) use the single-booster path.
        # Refuse cleanly rather than guess.
        raise NotImplementedError(
            "Ensemble inference (payload['ensemble']) is not yet wired "
            "into the inference module. Today's targets are the single-"
            "booster modulator models (regime_btc_v3). Phase 7 directional "
            "ensemble support is a follow-up."
        )

    spec = _reconstruct_spec(payload["spec"])
    # Substitute inference window so the loader fetches only the bars we
    # need. All other spec fields (symbol, interval, derived_features,
    # label_feature, hyperparams) stay identical to training so the
    # feature matrix has the same shape + ordering.
    inference_spec = replace(
        spec, train_start=inference_start, train_end=inference_end
    )

    # Path-injected via inference.artifacts; the loader is part of the
    # same blackheart-train package.
    from blackheart_train.loader import load_dataset  # noqa: E402

    ds = load_dataset(inference_spec)

    expected = tuple(payload["feature_names"])
    if tuple(ds.feature_names) != expected:
        # If this ever fires, the registry's eligible-feature set has
        # drifted from what the model was trained on. Fail loudly — a
        # silent column-order swap is the worst inference bug.
        raise RuntimeError(
            f"feature_names drift detected for model content_sha={content_sha}: "
            f"training expected {expected!r}, loader returned "
            f"{tuple(ds.feature_names)!r}. Investigate feature_registry / "
            f"EXCLUDED_FROM_INPUTS / label_direction changes since training."
        )

    objective = payload["objective"]
    values, confidence = _booster_predict(booster, ds.X, objective)

    # Per-row metadata for audit. content_sha lets a reviewer reproduce
    # the exact model used; the rest are nice-to-haves.
    meta_template = {
        "content_sha256": content_sha,
        "objective": objective,
        "n_features": len(expected),
        "model_name": spec.name,
    }

    return pd.DataFrame(
        {
            "signal_id": [signal_id] * len(ds.X),
            "symbol": [spec.symbol] * len(ds.X),
            "ts": ds.X.index,
            "value": values,
            "confidence": confidence if confidence is not None else [None] * len(ds.X),
            "meta": [meta_template] * len(ds.X),
        }
    )


__all__ = ["compute_predictions"]
