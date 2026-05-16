"""ML model inference for the Blackheart system.

Implements Phase 4 / Phase C of the integration roadmap: a Python
"inference sidecar" that loads pickled artifacts from blackheart-train,
computes predictions over a (model, symbol, interval, ts-range) tuple,
and writes signal_history rows.

The architecture choice (Python sidecar over LightGBM4j-in-JVM) was
recorded in blackheart/research/PHASE_4_INFERENCE_ARCHITECTURE_2026-05-16.md
and confirmed by the operator. The blueprint § 13.3 fail-open policy
relies on inference being out-of-process so a JNI crash can't take live
trading down.

Public surface:

* :func:`load_artifact` — sha-verified pickle load (delegates to
  blackheart_train.artifacts.read_artifact via path-injected import)
* :func:`compute_predictions` — high-level: given a signal_id, an
  inference window, and a connection, returns a DataFrame of
  ``[ts, value, confidence, meta]`` rows ready to upsert into
  signal_history
* :func:`persist_predictions` — INSERT the rows into signal_history
  (idempotent via PK ``(signal_id, symbol, ts)`` UPSERT)

Numerical equivalence guarantee:

The feature-vector assembly path reuses ``blackheart_train.loader``
directly. That means the inference X-matrix is bit-equivalent to the
training X-matrix at the same timestamps, modulo the inference window
being a sub-range of the training window. Verified by
``test_inference_equivalence.py``.
"""
from __future__ import annotations

from .api import compute_predictions, load_artifact
from .persist import persist_predictions

__all__ = ["compute_predictions", "load_artifact", "persist_predictions"]
