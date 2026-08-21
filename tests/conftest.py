import os
import shutil
from pathlib import Path

import psycopg
import pytest

from docintel.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"

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


@pytest.fixture
def seeded_corpus(conn, settings):
    """One MiniCorp 10-K row in documents, raw file under settings.data_dir."""
    raw_rel = "raw/0000000001-25-000001/mini10k.html"
    dest = settings.data_dir / raw_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "mini10k.html", dest)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (accession_no, cik, company, form_type, filing_date,
                                   period, source_url, raw_path, content_hash, size_bytes)
            VALUES ('0000000001-25-000001', '1', 'MiniCorp Inc.', '10-K', '2025-01-31',
                    '2024-12-31', 'https://example.test/mini10k.html', %s, 'deadbeef', 1)
            """,
            (raw_rel,),
        )
    conn.commit()
    return conn, settings
