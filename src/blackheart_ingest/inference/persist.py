"""INSERT/UPSERT predictions into ``signal_history``.

Schema (V66):

    signal_history (signal_id, symbol, ts) PRIMARY KEY
      + value DOUBLE PRECISION NOT NULL
      + confidence DOUBLE PRECISION
      + produced_at TIMESTAMPTZ NOT NULL
      + source VARCHAR(20) — stream | catchup_scan | historical_replay
      + meta JSONB
      + audit columns

Partitioned monthly on ts (2024-12 .. 2027-12, plus DEFAULT). Inserts to
the partition root land in the matching partition automatically.

Why UPSERT vs plain INSERT: re-running a backfill over the same window
should be idempotent — same (signal_id, symbol, ts) tuple already has
the same value if features and model haven't changed. The UPDATE branch
refreshes ``produced_at`` and ``meta`` so audit reads see the latest
inference run that touched the row.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

import psycopg

logger = logging.getLogger(__name__)

_DEFAULT_BY = "blackheart-ingest:inference"

# Allowed values per the V66 chk_signal_history_source CHECK constraint.
_ALLOWED_SOURCES = frozenset({"stream", "catchup_scan", "historical_replay"})


def persist_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    conn: psycopg.Connection,
    source: str = "historical_replay",
) -> int:
    """Upsert prediction rows into signal_history. Returns rows written.

    Each row must carry: signal_id (UUID), symbol (str), ts (datetime),
    value (float). Optional: confidence (float), meta (dict, will be
    JSON-encoded).

    ``produced_at`` is stamped to NOW() at insert time by the caller-
    provided value or the DB default. We pass NOW(UTC) explicitly so
    audit timestamps are consistent across clock-skewed hosts.

    ``source`` defaults to ``historical_replay`` because the research
    backfill use case dominates today. Switch to ``stream`` when the
    forward inference loop comes online (Phase D/E).
    """
    if source not in _ALLOWED_SOURCES:
        raise ValueError(
            f"source={source!r} not in allowed set {sorted(_ALLOWED_SOURCES)}"
        )
    if not rows:
        return 0

    produced_at = datetime.now(timezone.utc)
    params = []
    for r in rows:
        if r.get("value") is None:
            continue  # don't persist NaN predictions; let dropna at caller
        meta_json = json.dumps(r.get("meta") or {}, default=str)
        params.append({
            "signal_id": r["signal_id"],
            "symbol": r["symbol"],
            "ts": r["ts"],
            "value": float(r["value"]),
            "confidence": float(r["confidence"]) if r.get("confidence") is not None else None,
            "produced_at": produced_at,
            "source": source,
            "meta": meta_json,
            "by": _DEFAULT_BY,
        })

    if not params:
        return 0

    sql = """
        INSERT INTO signal_history (
            signal_id, symbol, ts, value, confidence,
            produced_at, source, meta, created_by, updated_by
        ) VALUES (
            %(signal_id)s, %(symbol)s, %(ts)s, %(value)s, %(confidence)s,
            %(produced_at)s, %(source)s, %(meta)s::jsonb,
            %(by)s, %(by)s
        )
        ON CONFLICT (signal_id, symbol, ts) DO UPDATE SET
            value = EXCLUDED.value,
            confidence = EXCLUDED.confidence,
            produced_at = EXCLUDED.produced_at,
            source = EXCLUDED.source,
            meta = EXCLUDED.meta,
            updated_time = NOW(),
            updated_by = EXCLUDED.updated_by
    """

    with conn.cursor() as cur:
        cur.executemany(sql, params)
    conn.commit()

    logger.info(
        "signal_history upserted | signal_id=%s symbol=%s rows=%d source=%s",
        params[0]["signal_id"], params[0]["symbol"], len(params), source,
    )
    return len(params)


__all__ = ["persist_predictions"]
