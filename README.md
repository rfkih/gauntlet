# blackheart-ingest

Python service that pulls macro / sentiment / on-chain data from free external
sources into the Blackheart Postgres `*_raw` tables. Called by the Trading
JVM's `BackfillMl*` handlers via HTTP; also runs as a scheduled live-ingest
worker.

Part of Phase 1 / M2 of the ML/sentiment integration. See
`../C--Project/memory/project_ml_blueprint.md` for the full architecture.

## Sources

| Source | Status | Auth | Notes |
|---|---|---|---|
| `alternative_me` | ✅ live | None | Fear & Greed Index. Simplest source. |
| `fred` | ✅ live | Free API key | FRED + ALFRED vintage for revision-prone series |
| `coingecko` | ✅ live | None | Free tier: BTC/ETH dominance + total mcap + per-coin price/mc/vol history |
| `defillama` | ✅ live | None | Per-stablecoin (USDT/USDC/…) circulating USD + per-chain TVL |
| `coinmetrics` | ✅ live | None | Community tier daily on-chain metrics (FlowOutNative, AdrActCnt, etc.) |
| `binance_macro` | ✅ live | None | Public futures macro: funding rate + open interest + L/S ratio + taker buy/sell |
| `forexfactory` | ✅ live (MVP) | None | faireconomy.media current-week JSON mirror; historical lookups deferred |

Stub sources have Java handlers that simulate progress but don't call this
service yet. The migration path is: implement the Python source module,
update the matching Java handler to delegate via HTTP, drop the stub
simulation. Same `historical_backfill_job` row, same UI.

## Setup

```powershell
# From repo root:
cd C:\Project\blackheart-ingest

# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install (editable)
pip install -e ".[dev]"

# Copy env template + fill in FRED_API_KEY
copy .env.example .env
notepad .env

# Run the HTTP server
.\.venv\Scripts\blackheart-ingest-server.exe
# OR
python -m blackheart_ingest.workers.server
```

The HTTP server listens on `127.0.0.1:8089` by default (loopback only). The
Trading JVM `BackfillMl*` handlers POST to it.

## Endpoints

```
GET  /healthz                        # liveness probe
GET  /sources                        # which source modules are registered
POST /pull/{source}                  # one-shot pull, blocks until complete
  body: {"start": "YYYY-MM-DDTHH:MM:SS", "end": "...", "symbol": "BTCUSDT|null", "config": {...}}
  returns: {"source": "...", "rows_inserted": N, "rows_rejected_pit": M, ...}
```

## Development

```powershell
# Lint
ruff check src

# Tests
pytest

# Type-check
mypy src
```
