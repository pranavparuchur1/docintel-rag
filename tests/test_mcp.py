"""MCP tool functions on the fixture corpus (connection + provider injected;
no real model, no real corpus)."""

from contextlib import nullcontext

import pytest

from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies
from docintel.embed.service import embed_pending, get_or_create_index_version
from docintel.mcp import server
from fake_provider import FakeProvider


@pytest.fixture
def mcp_env(seeded_corpus, monkeypatch):
    conn, settings = seeded_corpus
    [strategy] = get_strategies(["section_aware"], max_tokens=120)
    strategy.min_tokens = 5
    chunk_documents(conn, settings, [strategy])
    provider = FakeProvider()
    version = get_or_create_index_version(conn, provider, strategy)
    embed_pending(conn, provider, version)

    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(server.db, "connect", lambda _s: nullcontext(conn))
    monkeypatch.setattr(server, "_provider", provider)
    return conn


def test_list_companies(mcp_env):
    companies = server.list_companies()
    assert companies[0]["company"] == "MiniCorp Inc."
    assert any("10-K" in f for f in companies[0]["filings"])


def test_search_filings_and_get_chunk_round_trip(mcp_env):
    hits = server.search_filings("contract manufacturer disruption", k=5)
    assert hits, "hybrid search must return results (FTS leg)"
    assert {"chunk_id", "company", "section", "score", "excerpt"} <= set(hits[0])

    full = server.get_chunk(hits[0]["chunk_id"])
    assert full["company"] == "MiniCorp Inc."
    assert len(full["text"]) >= len(hits[0]["excerpt"].rstrip("…")) - 1
    assert full["source_url"].startswith("https://")


def test_get_chunk_missing(mcp_env):
    assert "error" in server.get_chunk(999_999_999)


def test_company_alias_resolution(mcp_env):
    assert server._canonical_company("nvidia") == "NVIDIA CORP"
    assert server._canonical_company("Coca Cola") == "COCA COLA CO"
    assert server._canonical_company(None) is None


def test_get_eval_report_returns_markdown():
    report = server.get_eval_report()
    assert isinstance(report, str) and report
