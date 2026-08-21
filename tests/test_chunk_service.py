from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies


def test_chunk_documents_idempotent_and_forceable(seeded_corpus):
    conn, settings = seeded_corpus
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
