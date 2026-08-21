"""Routing rules, fan-out mechanics, and the full agent graph on the fixture
corpus (fake embeddings + extractive answerer — no keys, no network)."""

import pytest

from docintel.agent.graph import AgentRuntime, run_agent
from docintel.agent.llm import ExtractiveAnswerer
from docintel.agent.routing import route_question
from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies
from docintel.embed.service import embed_pending, get_or_create_index_version
from docintel.retrieve.fanout import _round_robin
from docintel.retrieve.service import RetrievedChunk, Retriever
from fake_provider import FakeProvider

# ------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    ("question", "expected_route"),
    [
        ("What risks does Nvidia face from export controls?", "lookup"),
        ("Compare how Apple and Microsoft describe AI risks.", "comparison"),
        ("How did Nvidia's data center revenue change between filings?", "temporal"),
        ("What supply chain risks does Walmart report?", "out_of_scope"),
        ("What is the capital of France?", "out_of_scope"),
        ("What is the current federal funds rate?", "lookup"),  # no entity -> high bar
    ],
)
def test_route_question(question, expected_route):
    assert route_question(question).route == expected_route


def test_corpus_entities_do_not_trip_the_guard():
    # capitalized non-company tokens that appear in legitimate questions
    for q in (
        "How does the EU Digital Markets Act affect Apple's App Store?",
        "How does Tesla describe its dependence on Elon Musk?",
        "Why is Apple's supply chain exposed to risks in Taiwan?",
        "How did Microsoft's Intelligent Cloud segment perform?",
    ):
        assert route_question(q).route != "out_of_scope", q


def test_comparison_detects_both_companies():
    decision = route_question("How do Nvidia and Microsoft describe export controls?")
    assert set(decision.companies) == {"NVIDIA CORP", "MICROSOFT CORP"}


# ------------------------------------------------------------------- fan-out


def _chunk(cid: int, company: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="a", company=company, form_type="10-K",
        section=None, text="t", score=1.0, rank=1,
    )


def test_round_robin_interleaves_and_dedupes():
    a = [_chunk(1, "A"), _chunk(2, "A"), _chunk(3, "A")]
    b = [_chunk(10, "B"), _chunk(1, "B"), _chunk(11, "B")]  # 1 is a duplicate
    merged = _round_robin([a, b], k=4)
    assert [c.chunk_id for c in merged] == [1, 10, 2, 3]  # deduped, interleaved
    assert [c.rank for c in merged] == [1, 2, 3, 4]


# ------------------------------------------------------------- agent end-to-end


@pytest.fixture
def runtime(seeded_corpus):
    conn, settings = seeded_corpus
    [strategy] = get_strategies(["section_aware"], max_tokens=120)
    strategy.min_tokens = 5
    chunk_documents(conn, settings, [strategy])
    provider = FakeProvider()
    version = get_or_create_index_version(conn, provider, strategy)
    embed_pending(conn, provider, version)
    return AgentRuntime(
        conn=conn, settings=settings,
        retriever=Retriever(conn, provider, version),
        answerer=ExtractiveAnswerer(),
    ), conn, settings


def test_agent_refuses_unknown_entity_and_logs(runtime):
    rt, conn, _settings = runtime
    result = run_agent(rt, "What supply chain risks does Walmart report?")
    assert result.refused
    assert result.route == "out_of_scope"
    assert "Walmart" in result.refusal_reason
    with conn.cursor() as cur:
        cur.execute("SELECT route, refused FROM query_runs WHERE run_id = %s", (result.run_id,))
        assert cur.fetchone() == ("out_of_scope", True)


def test_agent_answers_with_citations_when_bar_cleared(runtime, monkeypatch):
    rt, conn, settings = runtime
    # fake embeddings carry no meaning: force the similarity bar to 0 so the
    # answer path is exercised deterministically (FTS supplies real relevance)
    monkeypatch.setattr(settings, "refusal_threshold_no_entity", 0.0)
    monkeypatch.setattr(settings, "refusal_threshold", 0.0)
    result = run_agent(rt, "What risks come from the contract manufacturer?")
    assert not result.refused
    assert result.citations
    assert "chunk" in result.citations[0]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT retrieved_chunk_ids FROM query_runs WHERE run_id = %s", (result.run_id,)
        )
        chunk_ids = cur.fetchone()[0]
        assert chunk_ids
        cur.execute("SELECT count(*) FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        assert cur.fetchone()[0] == len(chunk_ids)  # citations resolve to real rows


def test_agent_rewrites_once_then_refuses_on_low_scores(runtime, monkeypatch):
    rt, conn, settings = runtime
    monkeypatch.setattr(settings, "refusal_threshold", 2.0)  # unreachable bar
    monkeypatch.setattr(settings, "refusal_threshold_no_entity", 2.0)
    result = run_agent(rt, "What were the operating results?")
    assert result.refused
    assert result.rewritten  # exactly one bounded retry happened
    assert "below bar" in result.refusal_reason


def test_form_hint():
    from docintel.agent.plan import form_hint

    assert form_hint("changes between its last two annual reports") == "10-K"
    assert form_hint("across its two most recent 10-Ks") == "10-K"
    assert form_hint("in the latest quarterly filing") == "10-Q"
    assert form_hint("how did gross margin change over time") is None
