"""End-to-end eval smoke on the fixture corpus with the deterministic fake
provider: proves the pipeline mechanics (retrieval -> judging -> metrics ->
report) without a model download. Semantic quality is NOT asserted here — the
fake embeddings carry no meaning; FTS carries the retrieval in this test."""

from pathlib import Path

from docintel.chunk.service import chunk_documents
from docintel.chunk.strategies import get_strategies
from docintel.embed.service import embed_pending, get_or_create_index_version
from docintel.eval.golden import load_golden, relevant_universe_size
from docintel.eval.runner import run_eval, write_report
from docintel.retrieve.service import Retriever
from fake_provider import FakeProvider

GOLDEN = Path(__file__).parent / "fixtures" / "golden_mini.yaml"


def build_index(conn, settings):
    [strategy] = get_strategies(["section_aware"], max_tokens=120)
    # min budget in tests: fixture doc is small
    strategy.min_tokens = 5
    chunk_documents(conn, settings, [strategy])
    provider = FakeProvider()
    version = get_or_create_index_version(conn, provider, strategy)
    embed_pending(conn, provider, version)
    return provider, version


def test_fts_retrieval_and_golden_predicate(seeded_corpus):
    conn, settings = seeded_corpus
    provider, version = build_index(conn, settings)
    retriever = Retriever(conn, provider, version)

    hits = retriever.fts_search("contract manufacturer disruption", k=5)
    assert hits, "FTS must find the risk-factor chunk"
    assert hits[0].section and hits[0].section.startswith("Item 1A")

    golden = load_golden(GOLDEN)
    for q in golden:
        if not q.refuse:
            assert relevant_universe_size(conn, version, q.expected) >= 1, q.id


def test_eval_end_to_end_writes_report(seeded_corpus, tmp_path):
    conn, settings = seeded_corpus
    provider, version = build_index(conn, settings)
    golden = load_golden(GOLDEN)

    run = run_eval(conn, provider, version, golden, mode="hybrid", threshold=0.60)
    assert len(run.judgments) == 4
    # require:any question judged as a single alternate-spec unit
    g4 = next(j for j in run.judgments if j.question_id == "g4")
    assert g4.specs_total == 1
    metrics = run.metrics()
    for key in ("recall@5", "mrr@10", "ndcg@10", "refusal_precision", "p95_ms"):
        assert key in metrics
        assert metrics[key] >= 0.0 or key.endswith("_ms")

    report = write_report([run], golden, tmp_path, name="smoke.md")
    content = report.read_text(encoding="utf-8")
    assert "recall@5" in content
    assert "human-verified" in content
    assert "threshold sweep" in content.lower()
