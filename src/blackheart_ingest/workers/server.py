"""FastAPI dispatcher for the ML ingest service.

The Trading JVM ``BackfillMl*`` handlers (and later the live_ingest worker)
POST to ``/pull/{source}``. We look up the matching ``sources.<source>``
module, validate the payload, and run its synchronous ``fetch`` on a thread
pool worker (FastAPI handles this automatically for ``def`` handlers).

Loopback only by default. In the home↔VPS deployment the server runs on
home and the Java handlers reach it via Tailscale.

Endpoints
---------
GET  /healthz                                  liveness — {"ok": true, "version": ...}
GET  /sources                                  list of registered source modules
POST /pull/{source}                            run a one-shot pull, blocks until complete
POST /compute/{feature_name}/v/{version}       run one feature compute, blocks until complete
POST /compute/incremental                      recompute all macro features over a lookback window
GET  /features                                 list features registered in the Python definitions.py

Automatic compute loop
----------------------
When ``INGEST_COMPUTE_AUTO=true``, the server starts a background asyncio
loop that calls ``_compute_all_macro`` every ``INGEST_COMPUTE_INTERVAL_HOURS``
hours (default 4).  ``INGEST_COMPUTE_LOOKBACK_HOURS`` (default 72) sets how
far back each incremental run reaches — wide enough to catch any raw-data row
that arrived late or was backfilled between ticks.

The Java ``MlIngestScheduleRefresher`` can also trigger a one-shot incremental
compute immediately after a successful pull by calling
``POST /compute/incremental``.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any
from types import ModuleType

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..features.compute import compute as compute_feature
from ..features.definitions import FEATURES, get_feature
from ..features.persistence import fail_run, finish_run, start_run, write_values
from ..shared.base import IngestionRequest
from ..shared.db import get_connection
from ..shared.logging_setup import configure as configure_logging
from ..shared.settings import get_settings

logger = logging.getLogger(__name__)


# Source registry.
# Names match ml_ingest_schedule.source AND the Python module names under
# blackheart_ingest.sources. Add a row here when implementing a new source;
# the rest of the system picks it up automatically.

_KNOWN_SOURCES: dict[str, str | None] = {
    "alternative_me": "blackheart_ingest.sources.alternative_me",
    "fred": "blackheart_ingest.sources.fred",
    "coingecko": "blackheart_ingest.sources.coingecko",
    "defillama": "blackheart_ingest.sources.defillama",
    "coinmetrics": "blackheart_ingest.sources.coinmetrics",
    "binance_macro": "blackheart_ingest.sources.binance_macro",
    "forexfactory": "blackheart_ingest.sources.forexfactory",
}


def _import_source(source: str) -> ModuleType:
    if source not in _KNOWN_SOURCES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown source '{source}'. Known: {sorted(_KNOWN_SOURCES)}",
        )
    mod_path = _KNOWN_SOURCES[source]
    if not mod_path:
        raise HTTPException(
            status_code=501,
            detail=f"Source '{source}' is not yet implemented (Phase 1 M2 staged rollout).",
        )
    return importlib.import_module(mod_path)


class PullRequest(BaseModel):
    """Request body for POST /pull/{source}. Mirrors IngestionRequest but
    expressed in pydantic so FastAPI validates + auto-documents.
    """

    start: datetime = Field(..., description="ISO LocalDateTime, e.g. 2024-12-01T00:00:00")
    end: datetime = Field(..., description="ISO LocalDateTime, e.g. 2026-05-14T23:59:59")
    symbol: str | None = Field(default=None, description="Optional symbol scope")
    config: dict[str, Any] = Field(default_factory=dict, description="Source-specific params")


def _compute_all_macro(lookback_hours: int) -> dict[str, Any]:
    """Run an incremental compute for every macro (non-market_data) feature.

    Returns a summary dict suitable for returning from an endpoint or logging.
    Failures on individual features are caught, logged, and counted — one
    bad transformer should not block the rest.
    """
    end = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    start = end - timedelta(hours=lookback_hours)
    macro_features = [f for f in FEATURES if f.raw_tables == ("macro_raw",)]

    results: list[dict[str, Any]] = []
    total_rows = 0
    failures = 0

    for feat in macro_features:
        t0 = time.monotonic()
        try:
            with get_connection() as conn:
                run_id = start_run(feat, range_start=start, range_end=end, conn=conn)
                try:
                    df = compute_feature(feat, start=start, end=end, conn=conn)
                except Exception as e:  # noqa: BLE001
                    conn.rollback()
                    fail_run(run_id, error_message=str(e), conn=conn)
                    raise

                if df is None or df.empty:
                    finish_run(run_id, rows_written=0, conn=conn)
                    results.append({"feature": feat.name, "rows": 0, "status": "empty"})
                    continue

                written = write_values(feat, df, run_id=run_id, conn=conn)
                finish_run(run_id, rows_written=written, conn=conn)
            total_rows += written
            results.append({
                "feature": feat.name,
                "rows": written,
                "status": "ok",
                "duration_s": round(time.monotonic() - t0, 2),
            })
        except Exception as e:  # noqa: BLE001
            failures += 1
            logger.exception("auto_compute failed | feature=%s", feat.name)
            results.append({"feature": feat.name, "status": "error", "error": str(e)[:200]})

    logger.info(
        "auto_compute done | features=%d rows=%d failures=%d lookback_h=%d",
        len(macro_features), total_rows, failures, lookback_hours,
    )
    return {
        "features_computed": len(macro_features),
        "total_rows": total_rows,
        "failures": failures,
        "lookback_hours": lookback_hours,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "detail": results,
    }


async def _compute_loop(interval_hours: int, lookback_hours: int) -> None:
    """Background async task: compute all macro features every N hours."""
    interval_seconds = interval_hours * 3600
    logger.info(
        "compute_loop started | interval_h=%d lookback_h=%d",
        interval_hours, lookback_hours,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(_compute_all_macro, lookback_hours),
            )
        except asyncio.CancelledError:
            logger.info("compute_loop cancelled")
            return
        except Exception:  # noqa: BLE001
            logger.exception("compute_loop iteration failed — continuing")


@asynccontextmanager
async def _lifespan(settings_ref: Any, app: FastAPI):  # noqa: ANN001
    settings = settings_ref
    compute_task: asyncio.Task | None = None
    if settings.compute_auto:
        compute_task = asyncio.create_task(
            _compute_loop(settings.compute_interval_hours, settings.compute_lookback_hours),
            name="ingest-compute-loop",
        )
        logger.info(
            "ingest auto-compute enabled | interval_h=%d lookback_h=%d",
            settings.compute_interval_hours, settings.compute_lookback_hours,
        )
    try:
        yield
    finally:
        if compute_task is not None:
            compute_task.cancel()
            try:
                await compute_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


app = FastAPI(
    title="blackheart-ingest",
    version=__version__,
    description="Pulls macro/sentiment/on-chain data from free sources into Postgres *_raw tables.",
    lifespan=partial(_lifespan, get_settings()),
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "implemented_sources": sorted(s for s, m in _KNOWN_SOURCES.items() if m),
    }


@app.get("/sources")
def list_sources() -> dict[str, Any]:
    return {
        "sources": [
            {"name": name, "implemented": bool(mod_path)}
            for name, mod_path in sorted(_KNOWN_SOURCES.items())
        ]
    }


@app.post("/pull/{source}")
def pull(source: str, body: PullRequest) -> dict[str, Any]:
    started = time.monotonic()
    module = _import_source(source)

    if not hasattr(module, "fetch"):
        raise HTTPException(
            status_code=500,
            detail=f"Source module '{source}' missing required `fetch` callable.",
        )

    request = IngestionRequest(
        start=body.start,
        end=body.end,
        symbol=body.symbol,
        config=body.config or {},
    )

    logger.info(
        "pull dispatch | source=%s start=%s end=%s symbol=%s",
        source,
        body.start.isoformat(),
        body.end.isoformat(),
        body.symbol,
    )

    try:
        result = module.fetch(request)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # Health row was already updated inside the source module on failure.
        logger.exception("pull failed | source=%s", source)
        raise HTTPException(
            status_code=502,
            detail=f"{source}: {type(e).__name__}: {str(e)[:500]}",
        ) from e

    payload = result.to_json()
    payload["dispatch_duration_seconds"] = round(time.monotonic() - started, 3)
    return payload


class IncrementalComputeRequest(BaseModel):
    lookback_hours: int = Field(
        default=72,
        ge=1,
        le=8760,
        description="How many hours back to (re)compute. Defaults to settings value.",
    )


@app.post("/compute/incremental")
def compute_incremental(body: IncrementalComputeRequest | None = None) -> dict[str, Any]:
    """Recompute all macro (non-market_data) features over a lookback window.

    Intended for two callers:
    * The Java ``MlIngestScheduleRefresher`` — call immediately after a
      successful ``/pull`` to update feature_values with the fresh raw rows.
    * Operators doing a manual catch-up without a full CLI backfill.

    Runs synchronously (blocks until all features are done). For large
    windows prefer the CLI ``python -m blackheart_ingest.workers.compute_features``.
    """
    settings = get_settings()
    lookback = body.lookback_hours if body else settings.compute_lookback_hours
    return _compute_all_macro(lookback)


class ComputeRequest(BaseModel):
    """Body for POST /compute/{feature_name}/v/{version}.

    Mirrors the args ``compute()`` accepts. ``symbol`` and ``interval`` are
    required for per-bar features (market_data raw_table); the engine
    raises a clear FeatureComputeError if they're missing.
    """

    start: datetime = Field(..., description="ISO datetime (UTC, naive). Compute window lower bound.")
    end: datetime = Field(..., description="ISO datetime (UTC, naive). Compute window upper bound.")
    symbol: str | None = Field(default=None, description="Required for per-bar features (market_data).")
    interval: str | None = Field(default=None, description="Required for per-bar features.")


@app.get("/features")
def list_features_endpoint() -> dict[str, Any]:
    """List features whose transformers are defined in this codebase.

    Distinct from the orchestrator's ``GET /features`` which reads
    ``feature_registry`` — this endpoint reflects what the Python compute
    engine can actually run right now, regardless of registry state.
    """
    return {
        "features": [
            {
                "name": f.name,
                "version": f.version,
                "family": f.family,
                "raw_tables": list(f.raw_tables),
                "symbols": list(f.symbols),
                "intervals": list(f.intervals),
                "pit_safe": f.pit_safe,
            }
            for f in FEATURES
        ]
    }


@app.post("/compute/{feature_name}/v/{version}")
def compute_endpoint(feature_name: str, version: int, body: ComputeRequest) -> dict[str, Any]:
    """Run one feature compute. Synchronous — caller blocks for the run.

    Pattern mirrors ``workers/compute_features.py`` main loop: open a
    ``feature_compute_run`` row, invoke ``compute()``, write values, mark
    done. Any failure rolls back the transaction and marks the run failed
    so the audit trail is consistent.

    For long backfills (multi-year windows on per-bar features), prefer
    the CLI — this endpoint targets agent-driven scoped backfills.
    """
    started = time.monotonic()
    try:
        feat = get_feature(feature_name, version)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Feature {feature_name!r} v{version} not in Python FEATURES tuple. "
                "Either the name is misspelled or the registry has a row whose "
                "transformer hasn't been shipped to blackheart-ingest yet."
            ),
        ) from e

    # If the feature declares per-bar scope, body must include symbol+interval
    # matching the declaration. The engine's own check will also fire, but
    # this is a clearer 400 vs the deeper FeatureComputeError.
    if feat.symbols and not body.symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature {feature_name} declares symbols={list(feat.symbols)}; "
                "request body must include 'symbol'."
            ),
        )
    if feat.intervals and not body.interval:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature {feature_name} declares intervals={list(feat.intervals)}; "
                "request body must include 'interval'."
            ),
        )

    logger.info(
        "compute dispatch | feature=%s v=%d symbol=%s interval=%s start=%s end=%s",
        feature_name, version, body.symbol, body.interval,
        body.start.isoformat(), body.end.isoformat(),
    )

    with get_connection() as conn:
        run_id = start_run(
            feat,
            range_start=body.start,
            range_end=body.end,
            symbol=body.symbol,
            interval=body.interval,
            conn=conn,
        )
        try:
            df = compute_feature(
                feat,
                start=body.start,
                end=body.end,
                conn=conn,
                symbol=body.symbol,
                interval=body.interval,
            )
        except Exception as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            fail_run(run_id, error_message=str(e), conn=conn)
            logger.exception("compute failed | feature=%s v=%d", feature_name, version)
            raise HTTPException(
                status_code=502,
                detail=f"compute failed for {feature_name} v{version}: {type(e).__name__}: {str(e)[:400]}",
            ) from e

        if df is None or df.empty:
            finish_run(run_id, rows_written=0, conn=conn)
            return {
                "run_id": str(run_id),
                "feature_name": feature_name,
                "version": version,
                "symbol": body.symbol,
                "interval": body.interval,
                "rows_written": 0,
                "rows_computed": 0,
                "status": "done",
                "note": "Compute produced no rows (empty input window or all-NaN output).",
                "duration_seconds": round(time.monotonic() - started, 3),
            }

        try:
            written = write_values(
                feat,
                df,
                run_id=run_id,
                symbol=body.symbol,
                interval=body.interval,
                conn=conn,
            )
        except Exception as e:  # noqa: BLE001
            # write_values raised before commit — no data landed.
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            fail_run(run_id, error_message=str(e), conn=conn)
            logger.exception("persist failed | feature=%s v=%d", feature_name, version)
            raise HTTPException(
                status_code=502,
                detail=f"persist failed for {feature_name} v{version}: {type(e).__name__}: {str(e)[:400]}",
            ) from e

        # write_values committed `written` rows into feature_values. From
        # here on, the data IS persistent — any failure must leave the
        # audit row reflecting that, not zeroed.
        try:
            finish_run(run_id, rows_written=written, conn=conn)
        except Exception as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            logger.exception(
                "finish_run failed after persist | feature=%s v=%d run_id=%s rows=%d "
                "(data persisted but audit row may be inconsistent)",
                feature_name, version, run_id, written,
            )
            # Try to record the truth: data IS in feature_values, but
            # finish_run failed. Pass rows_written so the audit row
            # doesn't lie. Best-effort — if even this update fails, the
            # data is at least durable and a future query can reconcile.
            try:
                fail_run(
                    run_id,
                    error_message=f"finish_run failed after persist: {e}",
                    rows_written=written,
                    conn=conn,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "audit reconciliation also failed | run_id=%s — row left as 'running'",
                    run_id,
                )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"audit update failed after compute persisted "
                    f"run_id={run_id} rows={written}: "
                    f"{type(e).__name__}: {str(e)[:400]}"
                ),
            ) from e

    return {
        "run_id": str(run_id),
        "feature_name": feature_name,
        "version": version,
        "symbol": body.symbol,
        "interval": body.interval,
        "rows_written": written,
        "rows_computed": len(df),
        "status": "done",
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    configure_logging()
    settings = get_settings()
    logger.info(
        "starting blackheart-ingest server | host=%s port=%d implemented=%s",
        settings.server_host,
        settings.server_port,
        sorted(s for s, m in _KNOWN_SOURCES.items() if m),
    )
    uvicorn.run(
        "blackheart_ingest.workers.server:app",
        host=settings.server_host,
        port=settings.server_port,
        log_config=None,  # let structlog handle it
    )


if __name__ == "__main__":
    main()
