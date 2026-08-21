"""The incrementality contract, proven against Postgres with a fake provider:
second run embeds zero; identical-param re-chunk embeds zero; a param change
creates a NEW index version while the old one's vectors are untouched."""

import hashlib

import pytest
from tests.test_chunk_service import seed_document

from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies
from docintel.embed.providers import EmbeddingProvider, LocalBgeProvider, get_provider
from docintel.embed.service import embed_pending, get_or_create_index_version


class FakeProvider(EmbeddingProvider):
    """Deterministic 8-dim vectors derived from the text hash. No network."""

    model_name = "fake-embedder-8d"

    def __init__(self):
        self.calls = 0

    @property
    def dim(self) -> int:
        return 8

    def embed_passages(self, texts):
        self.calls += len(texts)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [b / 255.0 for b in digest[:8]]
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / norm for x in vec])
        return out

    def embed_query(self, text):
        return self.embed_passages([text])[0]


@pytest.fixture
def corpus(conn, settings):
    seed_document(conn, settings)
    return conn, settings


def test_embed_is_incremental_and_versions_are_isolated(corpus):
    conn, settings = corpus
    provider = FakeProvider()
    [strategy] = get_strategies(["recursive"])
    chunk_documents(conn, settings, [strategy])

    # first run embeds everything
    v1 = get_or_create_index_version(conn, provider, strategy)
    first = embed_pending(conn, provider, v1)
    assert first.embedded > 0
    assert first.already_present == 0

    # second run: zero new embeddings, zero provider calls
    calls_before = provider.calls
    second = embed_pending(conn, provider, v1)
    assert second.embedded == 0
    assert second.already_present == first.embedded
    assert provider.calls == calls_before

    # force re-chunk with identical params: new chunk_ids, same hashes -> still zero
    chunk_documents(conn, settings, [strategy], force=True)
    third = embed_pending(conn, provider, v1)
    assert third.embedded == 0

    # param change -> new params_hash -> NEW index version, old vectors untouched
    [narrow] = get_strategies(["recursive"], max_tokens=80)
    assert narrow.params_hash != strategy.params_hash
    chunk_documents(conn, settings, [narrow])
    v2 = get_or_create_index_version(conn, provider, narrow)
    assert v2.index_version_id != v1.index_version_id
    fourth = embed_pending(conn, provider, v2)
    assert fourth.embedded > 0

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {v1.table_name}")
        assert cur.fetchone()[0] == first.embedded  # old version intact
        cur.execute(
            "SELECT status FROM index_versions WHERE index_version_id IN (%s, %s)",
            (v1.index_version_id, v2.index_version_id),
        )
        assert {r[0] for r in cur.fetchall()} == {"ready"}


def test_version_requires_existing_chunks(corpus):
    conn, _settings = corpus
    [strategy] = get_strategies(["fixed"])
    with pytest.raises(ValueError, match="run `docintel chunk`"):
        get_or_create_index_version(conn, FakeProvider(), strategy)


def test_provider_factory_and_bge_query_prefix(settings):
    provider = get_provider(settings)
    assert isinstance(provider, LocalBgeProvider)  # local by default, no key needed
    assert provider.prepare_query("net revenue").startswith("Represent this sentence")
    # constructing the provider must NOT load the model (lazy import)
    assert provider._model is None
