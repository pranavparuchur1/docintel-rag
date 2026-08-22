"""API tests on the fixture corpus: validation, health, metrics, rate limit,
and a full /query round-trip (fake provider, extractive answers)."""

import pytest
from fastapi.testclient import TestClient

from docintel.api import app as app_module
from docintel.api.app import create_app, get_conn
from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies
from docintel.embed.service import embed_pending, get_or_create_index_version
from fake_provider import FakeProvider


@pytest.fixture
def client(seeded_corpus, monkeypatch):
    conn, settings = seeded_corpus
    [strategy] = get_strategies(["section_aware"], max_tokens=120)
    strategy.min_tokens = 5
    chunk_documents(conn, settings, [strategy])
    provider = FakeProvider()
    version = get_or_create_index_version(conn, provider, strategy)
    embed_pending(conn, provider, version)

    # fake embeddings carry no meaning; zero the bars so /query can answer
    monkeypatch.setattr(settings, "refusal_threshold", 0.0)
    monkeypatch.setattr(settings, "refusal_threshold_no_entity", 0.0)

    app = create_app(settings)

    def override_conn():
        yield conn  # the schema-isolated test connection

    app.dependency_overrides[get_conn] = override_conn
    with TestClient(app) as test_client:
        app.state.provider = provider  # never load the real model in tests
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pgvector"] != "NOT INSTALLED"
    assert body["serving_index_version"] >= 1


def test_query_validation():
    from docintel.api.app import QueryRequest

    with pytest.raises(ValueError):
        QueryRequest(question="hi")  # too short
    with pytest.raises(ValueError):
        QueryRequest(question="valid question", mode="agentic")  # bad mode
    with pytest.raises(ValueError):
        QueryRequest(question="valid question", k=99)  # k out of range


def test_query_answers_with_citations(client):
    response = client.post(
        "/query", json={"question": "What risks come from the contract manufacturer?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"]
    assert body["run_id"] >= 1


def test_query_refuses_unknown_entity(client):
    response = client.post(
        "/query", json={"question": "What supply chain risks does Walmart report?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["route"] == "out_of_scope"


def test_metrics_after_queries(client):
    client.post("/query", json={"question": "What were the operating results?"})
    body = client.get("/metrics").json()
    assert body["query_count"] >= 1
    assert 0.0 <= body["refusal_rate"] <= 1.0
    assert body["latency_p95_ms"] >= body["latency_p50_ms"] >= 0


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(app_module, "RATE_LIMIT_MAX", 2)
    payload = {"question": "What were the operating results?"}
    assert client.post("/query", json=payload).status_code == 200
    assert client.post("/query", json=payload).status_code == 200
    response = client.post("/query", json=payload)
    assert response.status_code == 429
    assert "Retry-After" in response.headers
