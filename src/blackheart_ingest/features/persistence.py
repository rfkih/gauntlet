"""Persistence layer for the feature compute pipeline.

Three operations:

* :func:`start_run`    — open a ``feature_compute_run`` row (status=running)
* :func:`write_values` — INSERT the tidy DataFrame into ``feature_values``
* :func:`finish_run`   — mark the run done with row count
* :func:`fail_run`     — mark the run failed with error message

The compute engine itself stays storage-agnostic. The CLI wraps each
feature in start_run → compute → write_values → finish_run, and a
``try/except`` around it converts unhandled exceptions into ``fail_run``.

Wire-format choices:

* ``feature_values.symbol`` and ``feature_values.interval`` are NOT NULL
  with default ``''``. We pass ``""`` for macro features (the schema's
  documented sentinel) so the primary key works without partial indexes.
* ``compute_run_id`` is the FK to ``feature_compute_run`` — every value
  row points to the run that produced it, so re-runs can be audited.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg

from ..shared.db import get_connection
from .definitions import FeatureDef

logger = logging.getLogger(__name__)

_DEFAULT_BY = "blackheart-ingest:compute"


def start_run(
    feat: FeatureDef,
    *,
    range_start: datetime,
    range_end: datetime,
    symbol: str | None = None,
    interval: str | None = None,
    conn: psycopg.Connection | None = None,
) -> uuid.UUID:
    """Open a feature_compute_run row (status='running'). Returns the run id."""
    run_id = uuid.uuid4()
    sql = """
        INSERT INTO feature_compute_run (
            run_id, feature_name, version, symbol, interval,
            range_start, range_end, rows_written, status, started_at,
            created_by, updated_by
        ) VALUES (
            %(run_id)s, %(feature_name)s, %(version)s, %(symbol)s, %(interval)s,
            %(range_start)s, %(range_end)s, 0, 'running', NOW(),
            %(by)s, %(by)s
        )
    """
    params = {
        "run_id": run_id,
        "feature_name": feat.name,
        "version": feat.version,
        "symbol": symbol,
        "interval": interval,
        "range_start": range_start,
        "range_end": range_end,
        "by": _DEFAULT_BY,
    }

    owned = conn is None
    if owned:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    except Exception:
        if owned:
            conn.rollback()
        raise
    finally:
        if owned:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]

    logger.info(
        "feature_compute_run started | run_id=%s feature=%s v%d", run_id, feat.name, feat.version
    )
    return run_id


def write_values(
    feat: FeatureDef,
    tidy: pd.DataFrame,
    *,
    run_id: uuid.UUID,
    symbol: str | None = None,
    interval: str | None = None,
    conn: psycopg.Connection | None = None,
) -> int:
    """Upsert the tidy DataFrame into feature_values. Returns rows inserted.

    Tidy columns expected: ``ts, value, name, version, family``. We discard
    ``name/version/family`` (always equal to ``feat.name/feat.version/feat.family``)
    and write the remaining ``ts, value`` plus the symbol/interval sentinels.

    ON CONFLICT DO UPDATE so re-running the same compute over the same
    window updates the value rather than failing on the PK. ``compute_run_id``
    is overwritten so the latest run is always traceable.
    """
    if tidy is None or tidy.empty:
        return 0

    symbol_col = symbol or ""
    interval_col = interval or ""

    # Drop rows with NaT timestamps — feature_values.ts is NOT NULL and the
    # raw PK violation message wouldn't tell the operator which transformer
    # produced a bad timestamp. Surfacing it as a warning here is kinder.
    valid_mask = tidy["ts"].notna()
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning(
            "feature=%s v%d dropping %d row(s) with NaT timestamps before write",
            feat.name, feat.version, n_dropped,
        )
    clean = tidy.loc[valid_mask]

    rows = [
        {
            "feature_name": feat.name,
            "version": feat.version,
            "symbol": symbol_col,
            "interval": interval_col,
            "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "value": float(v) if pd.notna(v) else None,
            "value_text": None,
            "compute_run_id": run_id,
            "by": _DEFAULT_BY,
        }
        for ts, v in zip(clean["ts"], clean["value"])
    ]
    if not rows:
        return 0

    sql = """
        INSERT INTO feature_values (
            feature_name, version, symbol, interval, ts,
            value, value_text, compute_run_id,
            created_by, updated_by
        ) VALUES (
            %(feature_name)s, %(version)s, %(symbol)s, %(interval)s, %(ts)s,
            %(value)s, %(value_text)s, %(compute_run_id)s,
            %(by)s, %(by)s
        )
        ON CONFLICT (feature_name, version, symbol, interval, ts)
        DO UPDATE SET
            value = EXCLUDED.value,
            value_text = EXCLUDED.value_text,
            compute_run_id = EXCLUDED.compute_run_id,
            updated_time = NOW(),
            updated_by = EXCLUDED.updated_by
    """

    owned = conn is None
    if owned:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
    except Exception:
        if owned:
            conn.rollback()
        raise
    finally:
        if owned:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]

    logger.info(
        "feature_values upserted | feature=%s v%d rows=%d", feat.name, feat.version, len(rows)
    )
    return len(rows)


def _update_run_status(
    run_id: uuid.UUID,
    *,
    status: str,
    rows_written: int = 0,
    error_message: str | None = None,
    conn: psycopg.Connection | None = None,
) -> None:
    sql = """
        UPDATE feature_compute_run
        SET status = %(status)s,
            rows_written = %(rows_written)s,
            finished_at = NOW(),
            error_message = %(error_message)s,
            updated_time = NOW(),
            updated_by = %(by)s
        WHERE run_id = %(run_id)s
    """
    params: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "rows_written": rows_written,
        "error_message": error_message,
        "by": _DEFAULT_BY,
    }

    owned = conn is None
    if owned:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    except Exception:
        if owned:
            conn.rollback()
        raise
    finally:
        if owned:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]


def _count_persisted_for_run(
    run_id: uuid.UUID,
    *,
    conn: psycopg.Connection | None = None,
) -> int:
    """Ground-truth count of feature_values rows currently attributed to a run.

    Note that ``write_values`` upserts with ``ON CONFLICT DO UPDATE`` and
    overwrites ``compute_run_id`` on conflict — so this counts only rows
    whose latest writer was this run, not historical attempts.
    """
    sql = "SELECT COUNT(*) FROM feature_values WHERE compute_run_id = %(run_id)s"
    owned = conn is None
    if owned:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"run_id": run_id})
            row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        if owned:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]


def finish_run(
    run_id: uuid.UUID,
    *,
    rows_written: int,
    conn: psycopg.Connection | None = None,
) -> None:
    """Mark a run done. Audit-row rows_written is reconciled against
    feature_values: write_values and the run-row UPDATE commit in separate
    transactions, so a crash between them can leave the audit row claiming
    more rows than landed (or vice-versa). The DB count is ground truth;
    a divergence emits WARN so the operator can investigate.
    """
    persisted = _count_persisted_for_run(run_id, conn=conn)
    if persisted != rows_written:
        logger.warning(
            "feature_compute_run reconciliation mismatch | run_id=%s "
            "caller_reported=%d db_count=%d -- recording db_count",
            run_id, rows_written, persisted,
        )
    _update_run_status(run_id, status="done", rows_written=persisted, conn=conn)
    logger.info("feature_compute_run done | run_id=%s rows=%d", run_id, persisted)


def fail_run(
    run_id: uuid.UUID,
    *,
    error_message: str,
    rows_written: int = 0,
    conn: psycopg.Connection | None = None,
) -> None:
    """Mark a feature_compute_run failed.

    ``rows_written`` defaults to 0 for the common case (compute or persist
    raised before any data landed). Pass the actual count when ``write_values``
    succeeded but a later step failed — the audit row must not lie about
    what's persisted in ``feature_values``.
    """
    _update_run_status(
        run_id,
        status="failed",
        rows_written=rows_written,
        error_message=error_message[:1000],
        conn=conn,
    )
    logger.warning(
        "feature_compute_run failed | run_id=%s rows_persisted=%d msg=%s",
        run_id, rows_written, error_message[:200],
    )
