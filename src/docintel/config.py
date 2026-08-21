"""Typed application config.

All configuration comes from environment variables (or a local .env file).
Validation fails fast and loudly: a missing or malformed value stops the process
at startup with a message naming every offending variable, instead of surfacing
as a confusing failure deep inside a pipeline run.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# EDGAR's fair-access policy requires a User-Agent identifying a person and a
# reachable email address, e.g. "Jane Doe jane@example.com".
_USER_AGENT_RE = re.compile(r".+\s+[^@\s]+@[^@\s]+\.[^@\s]+$")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- EDGAR ---------------------------------------------------------------
    edgar_user_agent: str
    edgar_max_requests_per_sec: float = 8.0  # SEC allows ~10/s; stay under it

    # --- Database ------------------------------------------------------------
    database_url: str = "postgresql://docintel:docintel@localhost:5433/docintel"

    # --- Pluggable backends --------------------------------------------------
    vector_backend: Literal["pgvector", "lancedb"] = "pgvector"
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_api_key: str | None = None

    # --- Storage ---------------------------------------------------------------
    data_dir: Path = Path("./data")

    @field_validator("edgar_user_agent")
    @classmethod
    def _user_agent_identifies_sender(cls, v: str) -> str:
        v = v.strip().strip('"')
        if not _USER_AGENT_RE.match(v):
            raise ValueError(
                "must be of the form 'Your Name email@domain.com' — "
                "SEC EDGAR rejects anonymous clients (https://www.sec.gov/os/accessing-edgar-data)"
            )
        return v

    @model_validator(mode="after")
    def _provider_keys_present(self) -> Settings:
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once; raise ConfigError with a readable summary."""
    try:
        return Settings()
    except ValidationError as exc:
        lines = ["Configuration is invalid — fix these environment variables (see .env.example):"]
        for err in exc.errors():
            var = ".".join(str(p) for p in err["loc"]).upper() or "(model)"
            lines.append(f"  {var}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc
