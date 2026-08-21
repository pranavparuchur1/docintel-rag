from pathlib import Path

import pytest

from docintel.chunk.base import content_hash
from docintel.chunk.strategies import (
    FixedSizeChunker,
    RecursiveChunker,
    SectionAwareChunker,
    get_strategies,
)
from docintel.parse import parse_filing

FIXTURE = Path(__file__).parent / "fixtures" / "mini10k.html"


@pytest.fixture(scope="module")
def doc():
    return parse_filing(FIXTURE.read_text(encoding="utf-8"), "10-K")


SMALL = dict(min_tokens=5)


def test_content_hash_ignores_whitespace_only_changes():
    assert content_hash("net  revenue\n grew") == content_hash("net revenue grew")
    assert content_hash("net revenue grew") != content_hash("net revenue fell")


def test_all_strategies_respect_token_budget(doc):
    for strategy in (
        FixedSizeChunker(size_tokens=120, overlap_tokens=20, **SMALL),
        RecursiveChunker(max_tokens=120, **SMALL),
        SectionAwareChunker(max_tokens=120, **SMALL),
    ):
        chunks = strategy.split(doc)
        assert chunks, strategy.name
        # small tolerance: merge accounting is additive across boundaries
        assert max(c.token_count for c in chunks) <= 132, strategy.name


def test_fixed_chunks_overlap(doc):
    chunks = FixedSizeChunker(size_tokens=120, overlap_tokens=40, **SMALL).split(doc)
    tail = chunks[0].text[-60:]
    assert tail.strip()[:25] in chunks[1].text  # boundary text appears in both


def test_section_aware_never_crosses_boundaries(doc):
    chunks = SectionAwareChunker(max_tokens=120, **SMALL).split(doc)
    risk = [c for c in chunks if c.section and c.section.startswith("Item 1A")]
    assert risk, "risk-factor chunks must carry their section label"
    for c in risk:
        assert "Net revenue increased" not in c.text  # MD&A content stays in Item 7


def test_recursive_prefers_paragraph_boundaries(doc):
    chunks = RecursiveChunker(max_tokens=200, **SMALL).split(doc)
    assert all(c.text == c.text.strip() for c in chunks)
    assert len(chunks) >= 3


def test_params_hash_changes_with_params():
    a = FixedSizeChunker(size_tokens=350, overlap_tokens=60)
    b = FixedSizeChunker(size_tokens=350, overlap_tokens=61)
    assert a.params_hash != b.params_hash
    assert a.params_hash == FixedSizeChunker(size_tokens=350, overlap_tokens=60).params_hash


def test_get_strategies_all_and_unknown():
    assert {s.name for s in get_strategies(["all"])} == {"fixed", "recursive", "section_aware"}
    with pytest.raises(ValueError, match="unknown"):
        get_strategies(["typo"])
