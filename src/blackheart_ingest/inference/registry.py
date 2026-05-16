"""Postgres queries: signal_definition + model_registry lookups.

Read-only — the inference worker uses blackheart_trading role per
V66 grants. Writing to either table is the orchestrator's job
(model_registry via POST /models/register; signal_definition via Flyway
migration like V78).
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg

logger = logging.getLogger(__name__)


def get_signal_definition_by_name(
    conn: psycopg.Connection, name: str
) -> dict[str, Any] | None:
    """Look up a signal_definition row by its unique ``name``.

    Returns ``None`` if no row exists. The caller decides whether that
    means "operator hasn't bootstrapped yet" (404-equivalent) or
    "create one on the fly" (we never auto-create — signal_definition
    is operator-curated).
    """
    sql = """
        SELECT signal_id, name, model_id, horizon, value_range,
               description, status
        FROM signal_definition
        WHERE name = %(name)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"name": name})
        row = cur.fetchone()
    return row


def get_signal_definition_by_id(
    conn: psycopg.Connection, signal_id: UUID
) -> dict[str, Any] | None:
    sql = """
        SELECT signal_id, name, model_id, horizon, value_range,
               description, status
        FROM signal_definition
        WHERE signal_id = %(signal_id)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"signal_id": signal_id})
        row = cur.fetchone()
    return row


def get_model_registry(
    conn: psycopg.Connection, model_id: UUID
) -> dict[str, Any] | None:
    """Look up a model_registry row by ``id``.

    Returns the columns the inference worker actually needs:
    artifact identity (sha256, uri), serving scope (symbol, interval,
    horizon_bars), lifecycle (status, artifact_synced_to_vps).

    Schema note (verified against information_schema 2026-05-16):
    model_registry does NOT carry ``name``, ``objective``, or
    ``label_feature`` columns — those live inside the artifact's
    pickled spec block. The orchestrator's POST /models/register
    stores them in the artifact but doesn't denormalise into the
    registry row. Callers needing those fields must load the artifact.
    """
    sql = """
        SELECT id, version, purpose, family, symbol, interval, horizon_bars,
               status, artifact_sha256, artifact_uri,
               artifact_synced_to_vps, artifact_synced_at,
               random_seed, training_window, code_commit
        FROM model_registry
        WHERE id = %(model_id)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"model_id": model_id})
        row = cur.fetchone()
    return row


__all__ = [
    "get_signal_definition_by_id",
    "get_signal_definition_by_name",
    "get_model_registry",
]
