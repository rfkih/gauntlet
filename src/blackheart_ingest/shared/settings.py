"""Centralised settings via pydantic-settings.

All env vars prefixed `INGEST_` to avoid collisions with the trading JVM's
own env (`SPRING_*`, `JWT_*`, etc.) when both run on the same host.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres — shared with trading JVM.
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_db"
    db_user: str = "blackheart_trading"
    db_password: str = ""

    server_host: str = "127.0.0.1"
    server_port: int = 8089

    fred_api_key: str = ""

    default_max_backfill_lag_hours: int = Field(default=72, ge=1, le=8760)
    log_level: str = "INFO"

    def db_kwargs(self) -> dict[str, object]:
        """Connection keyword args for ``psycopg.connect(**kwargs)``.

        Returns a dict rather than a DSN string so passwords with spaces,
        quotes, or backslashes can't break the connection-string parser.
        psycopg handles per-field quoting/escaping automatically when given
        keyword args.
        """
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
