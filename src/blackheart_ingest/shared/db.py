"""Postgres helpers — connection management + idempotent raw-row inserts.

Uses ``psycopg`` (v3) in sync mode. The FastAPI layer above is async, but
each request runs on a thread-pool worker, so blocking DB calls inside a
``def`` (not ``async def``) handler is the FastAPI-recommended pattern for
sync drivers.

The shape of every ``*_raw`` row matches the V66 schema:
    ingestion_id, source, source_uri, symbol, series_id, event_time,
    ingestion_time, value, value_text (or body_uri etc.), content_hash,
    schema_version.

We only expose ``write_raw_rows`` here — sources don't need direct cursor
access. If they ever do, plumb a context-manager via this module so
connection lifecycle stays centralised.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .settings import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Open a short-lived connection. Long-running scripts should reuse one
    connection across many calls — this helper is for one-shot uses.

    Always sets ``TIME ZONE 'UTC'`` on the new session so naive datetimes
    we send (Python convention) interpret as UTC when written to
    ``TIMESTAMPTZ`` columns. Without this, the session inherits the
    Postgres server's tz and timestamps drift silently on any non-UTC host.
    """
    settings = get_settings()
    conn = psycopg.connect(**settings.db_kwargs(), row_factory=dict_row, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")
        conn.commit()
        yield conn
    finally:
        conn.close()


def content_hash(*parts: Any) -> str:
    """Deterministic dedupe key. Concatenate the stable fields a source
    promises identify a row (e.g. series_id + event_time + value) and SHA-256.
    Stored in ``content_hash`` for cross-source deduplication.
    """
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def write_macro_raw_rows(
    rows: Sequence[dict[str, Any]],
    *,
    conn: psycopg.Connection | None = None,
) -> tuple[int, int]:
    """Insert into ``macro_raw`` with ``ON CONFLICT DO NOTHING``.

    Each row dict must contain:
        source, source_uri, symbol (or None), series_id, event_time (datetime),
        ingestion_time (datetime), value (numeric or None), value_text (str or None),
        content_hash, schema_version (int, default 1).

    Returns
    -------
    (inserted, skipped_duplicate)
    """
    if not rows:
        return 0, 0

    sql = """
        INSERT INTO macro_raw (
            source, source_uri, symbol, series_id, event_time, ingestion_time,
            value, value_text, content_hash, schema_version
        ) VALUES (
            %(source)s, %(source_uri)s, %(symbol)s, %(series_id)s,
            %(event_time)s, %(ingestion_time)s,
            %(value)s, %(value_text)s, %(content_hash)s, %(schema_version)s
        )
        ON CONFLICT DO NOTHING
        RETURNING ingestion_id
    """

    owned_conn = conn is None
    if owned_conn:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        inserted = 0
        with conn.cursor() as cur:
            for row in rows:
                row.setdefault("schema_version", 1)
                cur.execute(sql, row)
                if cur.rowcount == 1:
                    inserted += 1
        conn.commit()
        skipped = len(rows) - inserted
        return inserted, skipped
    except Exception:
        conn.rollback()
        raise
    finally:
        if owned_conn:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]


def update_source_health(
    source: str,
    *,
    success: bool,
    rows_inserted: int = 0,
    rows_rejected_pit: int = 0,
    error_message: str | None = None,
    conn: psycopg.Connection | None = None,
) -> None:
    """Apply the outcome of one pull to ``ml_source_health``.

    Health-status rules:
      - success + 0 PIT rejections → 'healthy'
      - success + PIT rejections    → 'degraded' (rows landed but quality flag raised)
      - failure                     → 'failed' on 3+ consecutive failures, else 'degraded'
    """
    now = datetime.utcnow()

    if success:
        sql = """
            UPDATE ml_source_health
            SET last_pull_at = %(now)s,
                last_success_at = %(now)s,
                consecutive_failures = 0,
                rows_inserted_total = rows_inserted_total + %(rows_inserted)s,
                rejected_pit_violations_total = rejected_pit_violations_total + %(rows_rejected_pit)s,
                health_status = CASE
                    WHEN %(rows_rejected_pit)s > 0 THEN 'degraded'
                    ELSE 'healthy'
                END,
                health_message = NULL,
                updated_at = %(now)s,
                updated_time = %(now)s,
                updated_by = 'blackheart-ingest'
            WHERE source = %(source)s
        """
        params = {
            "now": now,
            "source": source,
            "rows_inserted": rows_inserted,
            "rows_rejected_pit": rows_rejected_pit,
        }
    else:
        sql = """
            UPDATE ml_source_health
            SET last_pull_at = %(now)s,
                last_failure_at = %(now)s,
                consecutive_failures = consecutive_failures + 1,
                errors_total = errors_total + 1,
                health_status = CASE
                    WHEN consecutive_failures + 1 >= 3 THEN 'failed'
                    ELSE 'degraded'
                END,
                health_message = %(error_message)s,
                updated_at = %(now)s,
                updated_time = %(now)s,
                updated_by = 'blackheart-ingest'
            WHERE source = %(source)s
        """
        params = {"now": now, "source": source, "error_message": error_message}

    owned_conn = conn is None
    if owned_conn:
        ctx = get_connection()
        conn = ctx.__enter__()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.rowcount == 0:
                logger.warning(
                    "ml_source_health update affected 0 rows for source=%s (seed missing?)",
                    source,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if owned_conn:
            ctx.__exit__(None, None, None)  # type: ignore[has-type]
