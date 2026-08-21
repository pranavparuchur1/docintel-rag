"""Database connection and a minimal, auditable migration runner.

Migrations are plain SQL files in sql/, applied in lexical order. Each applied
file's SHA-256 is recorded in schema_migrations; if a previously applied file
changes on disk, the runner refuses to continue rather than leaving the schema
in an ambiguous state. This is deliberately simpler than Alembic: the schema is
small, and every step must be explainable in an interview.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg

from docintel.config import Settings

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


class MigrationError(RuntimeError):
    pass


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def apply_migrations(conn: psycopg.Connection, sql_dir: Path = SQL_DIR) -> list[str]:
    """Apply pending sql/*.sql files in order. Returns the filenames applied."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   text PRIMARY KEY,
                sha256     text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute("SELECT filename, sha256 FROM schema_migrations")
        applied = dict(cur.fetchall())

    newly_applied: list[str] = []
    for path in sorted(sql_dir.glob("*.sql")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name in applied:
            if applied[path.name] != digest:
                raise MigrationError(
                    f"{path.name} changed after being applied "
                    f"(recorded {applied[path.name][:12]}, on disk {digest[:12]}). "
                    "Write a new migration file instead of editing an applied one."
                )
            continue
        with conn.cursor() as cur:
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute(
                "INSERT INTO schema_migrations (filename, sha256) VALUES (%s, %s)",
                (path.name, digest),
            )
        conn.commit()
        newly_applied.append(path.name)
    return newly_applied


def health(conn: psycopg.Connection) -> dict[str, str]:
    """Return server version and pgvector availability — used by `docintel db check`."""
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    return {
        "postgres": version.split(" on ")[0],
        "pgvector": row[0] if row else "NOT INSTALLED",
    }
