import os

import psycopg
import pytest

from docintel.config import Settings

VALID_UA = "Jane Doe jane@example.com"
TEST_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://docintel:docintel@localhost:5433/docintel"
)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, edgar_user_agent=VALID_UA, data_dir=tmp_path)


@pytest.fixture
def conn():
    """Postgres connection scoped to a throwaway schema; skips if no DB is up.

    Everything (including schema_migrations) is created inside test_docintel via
    search_path, so tests never touch real ingested data in public.
    """
    try:
        connection = psycopg.connect(TEST_DB_URL, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("postgres not reachable — run `docker compose up -d db`")

    from docintel.db import apply_migrations

    with connection.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS test_docintel CASCADE")
        cur.execute("CREATE SCHEMA test_docintel")
        cur.execute("SET search_path TO test_docintel, public")
    connection.commit()
    apply_migrations(connection)
    yield connection
    connection.rollback()
    with connection.cursor() as cur:
        cur.execute("DROP SCHEMA test_docintel CASCADE")
    connection.commit()
    connection.close()
