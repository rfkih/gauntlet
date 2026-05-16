"""CLI: backfill model predictions over a (signal, ts-range) window.

Resolves a signal_definition by name (or by signal_id), looks up the
linked model_registry row, loads the artifact, computes predictions
over the requested window, and upserts ``signal_history`` rows.

Usage::

    # Backfill v3 over its full training window (default if --start/--end omitted)
    python -m blackheart_ingest.workers.inference_backfill --signal regime_btc_v3

    # Backfill a custom sub-window
    python -m blackheart_ingest.workers.inference_backfill \\
        --signal regime_btc_v3 \\
        --start 2025-06-01 --end 2025-07-01

    # Dry-run: compute but don't persist
    python -m blackheart_ingest.workers.inference_backfill \\
        --signal regime_btc_v3 --no-persist

Audit: every run logs ``[signal=NAME rows=N source=SRC dur=Ss]`` so
ops can grep for "wrote how many predictions when."

Source: defaults to ``historical_replay``. This is the correct label
for backfilled-over-existing-history predictions (the V66 source enum
distinguishes them from live ``stream`` writes and ``catchup_scan``
recovery passes).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from uuid import UUID

from ..inference.api import compute_predictions
from ..inference.artifacts import load_artifact
from ..inference.persist import persist_predictions
from ..inference.registry import (
    get_model_registry,
    get_signal_definition_by_id,
    get_signal_definition_by_name,
)
from ..shared.db import get_connection
from ..shared.logging_setup import configure as configure_logging

logger = logging.getLogger(__name__)


def _parse_iso(s: str) -> datetime:
    """Accept either a date (YYYY-MM-DD) or a full ISO datetime."""
    return datetime.fromisoformat(s)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blackheart-ingest-inference-backfill",
        description="Backfill model predictions into signal_history.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--signal", type=str, default=None,
        help="signal_definition.name (e.g. regime_btc_v3)",
    )
    grp.add_argument(
        "--signal-id", type=str, default=None,
        help="signal_definition.signal_id (UUID)",
    )
    p.add_argument(
        "--start", type=_parse_iso, default=None,
        help="ISO datetime. Default: spec.train_start from the model's payload.",
    )
    p.add_argument(
        "--end", type=_parse_iso, default=None,
        help="ISO datetime (exclusive). Default: spec.train_end.",
    )
    p.add_argument(
        "--source", type=str, default="historical_replay",
        choices=["stream", "catchup_scan", "historical_replay"],
        help="V66 signal_history.source value. Default: historical_replay.",
    )
    p.add_argument(
        "--no-persist", action="store_true",
        help="Compute predictions but skip the signal_history upsert. "
        "Useful for inspecting the prediction distribution before writing.",
    )
    p.add_argument(
        "--head", type=int, default=5,
        help="Number of head/tail prediction rows to print after compute.",
    )
    return p


def main() -> int:
    configure_logging()
    args = _build_arg_parser().parse_args()
    started = time.monotonic()

    with get_connection() as conn:
        # ── Resolve signal -----------------------------------------------
        if args.signal:
            sig = get_signal_definition_by_name(conn, args.signal)
            sig_label = args.signal
        else:
            sig = get_signal_definition_by_id(conn, UUID(args.signal_id))
            sig_label = args.signal_id

        if sig is None:
            print(
                f"ERR signal not found: {sig_label!r}. "
                f"Bootstrap a signal_definition row first (Flyway V78+ "
                f"pattern) or check the name / signal_id."
            )
            return 2

        signal_id = sig["signal_id"]
        model_id = sig["model_id"]
        print(
            f"signal={sig['name']} (status={sig['status']}) "
            f"signal_id={signal_id} model_id={model_id}"
        )

        # ── Resolve model artifact metadata ------------------------------
        model = get_model_registry(conn, model_id)
        if model is None:
            print(f"ERR model_registry row {model_id} not found")
            return 3

        content_sha = model["artifact_sha256"]
        if not content_sha:
            print(
                f"ERR model {model_id} has no artifact_sha256. "
                f"Check the model_registry row — register step did not write."
            )
            return 4

        # The artifact carries name/objective/label_feature (not denormalised
        # into model_registry — verified 2026-05-16). Load once here both to
        # surface those fields in the audit line AND to make
        # ``compute_predictions`` skip a redundant disk read by reusing
        # ``content_sha`` lookup.
        payload = load_artifact(content_sha)
        spec_dict = payload["spec"]
        spec_name = spec_dict.get("name", "<unknown>")
        spec_objective = payload.get("objective", "<unknown>")

        print(
            f"model {spec_name} v{model['version']} "
            f"objective={spec_objective} symbol={model['symbol']} "
            f"interval={model['interval']} purpose={model['purpose']} "
            f"status={model['status']} content_sha={content_sha[:12]}..."
        )

        # ── Resolve inference window. Default to spec.train_start..end. --
        if args.start is None or args.end is None:
            from datetime import datetime as _dt
            spec_start = spec_dict["train_start"]
            spec_end = spec_dict["train_end"]
            if isinstance(spec_start, str):
                spec_start = _dt.fromisoformat(spec_start)
            if isinstance(spec_end, str):
                spec_end = _dt.fromisoformat(spec_end)
            start = args.start or spec_start
            end = args.end or spec_end
        else:
            start = args.start
            end = args.end

        print(f"window=[{start.isoformat()}, {end.isoformat()})  source={args.source}")

        # Phase 4 research-mode caveat: when the inference window overlaps
        # the model's training window, predictions for bars the booster was
        # FITTED ON are training-set biased (artificially confident /
        # accurate). Only bars in the last walk-forward fold's validation
        # set + any bars OUTSIDE the training window are unbiased. The
        # Phase E shadow-log analytics needs to filter on this — typically
        # by restricting to ts > fold-N's training-window upper bound.
        # Documenting here rather than aborting because the operator
        # explicitly opted into the full-window backfill (option Y) to
        # validate the inference plumbing, not to claim unbiased coverage.
        print(
            "NOTE  predictions on bars inside the training window are "
            "training-set biased — only the last walk-forward fold's val "
            "rows are out-of-fold. Shadow-log analytics must filter."
        )

        # ── Compute -----------------------------------------------------
        pred_df = compute_predictions(
            content_sha=content_sha,
            signal_id=signal_id,
            inference_start=start,
            inference_end=end,
        )
        n = len(pred_df)
        print(f"computed {n} predictions")
        if n:
            print(f"  ts_range: {pred_df['ts'].min()} .. {pred_df['ts'].max()}")
            print("  value distribution:")
            print(pred_df["value"].describe().to_string())
            print()
            print(f"head({args.head}):")
            print(
                pred_df[["ts", "value", "confidence"]]
                .head(args.head)
                .to_string(index=False)
            )
            print(f"tail({args.head}):")
            print(
                pred_df[["ts", "value", "confidence"]]
                .tail(args.head)
                .to_string(index=False)
            )

        # ── Persist -----------------------------------------------------
        if args.no_persist:
            print("--no-persist: skipping signal_history upsert")
        else:
            persisted = persist_predictions(
                pred_df.to_dict("records"),
                conn=conn,
                source=args.source,
            )
            print(f"persisted {persisted} rows to signal_history")

    dur = time.monotonic() - started
    print(f"DONE  signal={sig['name']} rows={n} dur={dur:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
