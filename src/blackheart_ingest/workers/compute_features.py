"""CLI to compute the registered features over a date range.

Pipe one or more feature names via ``--features`` to scope the run; default
is "everything registered". Output goes to stdout (describe + tail) and
optionally to a Parquet file (one per feature) for offline inspection.

Usage::

    python -m blackheart_ingest.workers.compute_features --days 180
    python -m blackheart_ingest.workers.compute_features --features vix_close,dxy_zscore_30d
    python -m blackheart_ingest.workers.compute_features --start 2025-01-01 --end 2025-06-30 --parquet-dir ./out
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from ..features.compute import compute, default_window
from ..features.definitions import FEATURES, FeatureDef, get_feature
from ..features.persistence import fail_run, finish_run, start_run, write_values
from ..shared.db import get_connection
from ..shared.logging_setup import configure as configure_logging

logger = logging.getLogger(__name__)


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blackheart-ingest-compute",
        description="Run the registered macro/sentiment features over a date range.",
    )
    p.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature names to run. Empty = all registered.",
    )
    p.add_argument(
        "--start",
        type=_parse_iso,
        default=None,
        help="ISO datetime (UTC, naive). Default: end - <days>.",
    )
    p.add_argument(
        "--end",
        type=_parse_iso,
        default=None,
        help="ISO datetime (UTC, naive). Default: now.",
    )
    p.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days-back if --start/--end are not given. Default 180.",
    )
    p.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Optional output directory. One Parquet file per feature.",
    )
    p.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of head/tail rows to print per feature. Default 5.",
    )
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing to feature_values / feature_compute_run. "
        "Default is to persist — use this for the stdout-only debug mode.",
    )
    p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Restrict per-bar features to a single symbol (e.g. BTCUSDT). "
        "Macro features ignore this. When omitted, the feature's declared "
        "symbols tuple is used.",
    )
    p.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Restrict per-bar features to a single interval (e.g. 1h). "
        "Macro features ignore this. When omitted, the feature's declared "
        "intervals tuple is used.",
    )
    return p


def _resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.start and args.end:
        return args.start, args.end
    if args.start and not args.end:
        return args.start, datetime.utcnow()
    if args.end and not args.start:
        # explicit end + days-back from end
        return args.end - (datetime.utcnow() - default_window(args.days)[0]), args.end
    return default_window(args.days)


def _instances_for(
    feat: FeatureDef, cli_symbol: str | None, cli_interval: str | None
) -> list[tuple[str | None, str | None]]:
    """List of (symbol, interval) tuples to compute for one feature.

    * Empty symbols AND empty intervals → single global instance ``(None, None)``.
    * Declared symbols/intervals → cross-product, optionally filtered by
      CLI flags. An empty result triggers a clear error in the caller.
    """
    if not feat.symbols and not feat.intervals:
        return [(None, None)]

    symbols = list(feat.symbols) if feat.symbols else [None]
    intervals = list(feat.intervals) if feat.intervals else [None]

    if cli_symbol is not None:
        symbols = [s for s in symbols if s == cli_symbol]
    if cli_interval is not None:
        intervals = [i for i in intervals if i == cli_interval]

    return [(s, i) for s in symbols for i in intervals]


def main() -> int:
    configure_logging()
    args = _build_arg_parser().parse_args()

    if args.features.strip():
        names = [n.strip() for n in args.features.split(",") if n.strip()]
        feats = [get_feature(n) for n in names]
    else:
        feats = list(FEATURES)

    start, end = _resolve_window(args)
    persist = not args.no_persist
    logger.info(
        "computing %d feature(s) over window %s -> %s persist=%s",
        len(feats), start.isoformat(), end.isoformat(), persist,
    )

    if args.parquet_dir:
        args.parquet_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = 0
    overall_persisted = 0
    failures = 0

    def _fail_safely(conn, run_id, msg: str) -> None:
        """Mark a run failed even if the connection is in an aborted-
        transaction state. Rollback first, then UPDATE. Swallows secondary
        errors so the caller's primary exception keeps propagating cleanly.
        """
        if run_id is None:
            return
        try:
            conn.rollback()
        except Exception as rb_err:  # noqa: BLE001
            logger.warning("rollback before fail_run swallowed: %s", rb_err)
        try:
            fail_run(run_id, error_message=msg, conn=conn)
        except Exception as fr_err:  # noqa: BLE001
            logger.warning("fail_run itself failed (run %s): %s", run_id, fr_err)

    with get_connection() as conn:
        for feat in feats:
            instances = _instances_for(feat, args.symbol, args.interval)
            if not instances:
                print()
                print(f"=== {feat.name} v{feat.version} ({feat.family}) ===")
                print(
                    f"  SKIP: no (symbol, interval) match for "
                    f"--symbol={args.symbol} --interval={args.interval} "
                    f"vs declared symbols={feat.symbols} intervals={feat.intervals}"
                )
                continue

            for sym, ivl in instances:
                print()
                scope_tag = (
                    f" [{sym or '-'}/{ivl or '-'}]" if (sym or ivl) else ""
                )
                print(f"=== {feat.name} v{feat.version} ({feat.family}){scope_tag} ===")
                print(f"  {feat.description}")

                run_id = None
                if persist:
                    try:
                        run_id = start_run(
                            feat,
                            range_start=start,
                            range_end=end,
                            symbol=sym,
                            interval=ivl,
                            conn=conn,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:  # noqa: BLE001
                        print(f"  ABORT start_run failed: {e}")
                        try:
                            conn.rollback()
                        except Exception:  # noqa: BLE001
                            pass
                        failures += 1
                        continue

                try:
                    df = compute(
                        feat,
                        start=start,
                        end=end,
                        conn=conn,
                        symbol=sym,
                        interval=ivl,
                    )
                except KeyboardInterrupt:
                    _fail_safely(conn, run_id, "interrupted by SIGINT during compute")
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL compute: {e}")
                    failures += 1
                    _fail_safely(conn, run_id, str(e))
                    continue

                if df is None or df.empty:
                    print("  (no rows)")
                    if run_id is not None:
                        try:
                            finish_run(run_id, rows_written=0, conn=conn)
                        except Exception as e:  # noqa: BLE001
                            _fail_safely(conn, run_id, f"finish_run on empty: {e}")
                            failures += 1
                    continue

                overall_rows += len(df)
                print(f"  rows={len(df)}  ts_range={df['ts'].min()} .. {df['ts'].max()}")
                print()
                print(df["value"].describe().to_string())
                print()
                print("head:")
                print(df.head(args.head).to_string(index=False))
                print("tail:")
                print(df.tail(args.head).to_string(index=False))

                if args.parquet_dir:
                    suffix = f"_{sym}_{ivl}" if (sym or ivl) else ""
                    outfile = args.parquet_dir / f"{feat.name}_v{feat.version}{suffix}.parquet"
                    df.to_parquet(outfile, index=False)
                    print(f"  wrote -> {outfile}")

                if run_id is not None:
                    try:
                        written = write_values(
                            feat, df, run_id=run_id, symbol=sym, interval=ivl, conn=conn
                        )
                        finish_run(run_id, rows_written=written, conn=conn)
                        overall_persisted += written
                    except KeyboardInterrupt:
                        _fail_safely(
                            conn, run_id, "interrupted by SIGINT during persist"
                        )
                        raise
                    except Exception as e:  # noqa: BLE001
                        print(f"  FAIL persist: {e}")
                        failures += 1
                        _fail_safely(conn, run_id, str(e))
                        continue
                    # Print AFTER the run row is marked done so a print encoding
                    # error doesn't taint the audit trail. ASCII-only.
                    print(
                        f"  persisted: {written} rows to feature_values "
                        f"(run_id={run_id})"
                    )

    print()
    print(
        f"total: computed_rows={overall_rows} persisted_rows={overall_persisted} "
        f"failures={failures} persist={persist}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
