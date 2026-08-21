import shutil
from pathlib import Path

from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies

FIXTURE = Path(__file__).parent / "fixtures" / "mini10k.html"


def seed_document(conn, settings) -> None:
    raw_rel = "raw/0000000001-25-000001/mini10k.html"
    dest = settings.data_dir / raw_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, dest)
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


def test_chunk_documents_idempotent_and_forceable(conn, settings):
    seed_document(conn, settings)
    strategies = get_strategies(["recursive"])

    first = chunk_documents(conn, settings, strategies)
    assert first.chunked == 1
    assert first.chunks_written > 0

    second = chunk_documents(conn, settings, strategies)
    assert second.skipped == 1
    assert second.chunks_written == 0

    forced = chunk_documents(conn, settings, strategies, force=True)
    assert forced.chunks_written == first.chunks_written

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == first.chunks_written  # replaced, not duplicated
