"""Golden question set: loading, and the relevance predicate.

A retrieved chunk is RELEVANT to a question iff it matches at least one of the
question's expected specs. A spec matches when ALL of its declared fields hold:

- company:        exact match on documents.company
- form_type:      exact match on documents.form_type (optional)
- accession:      exact match on documents.accession_no (optional, overrides)
- section_prefix: chunks.section starts with it (e.g. "Item 1A")
- must_contain:   case-insensitive substring of chunk text (optional)

The same predicate exists twice — in Python for judging retrieved chunks, and
in SQL (relevant_universe_size) for counting ALL relevant chunks in the corpus,
which nDCG's ideal ranking needs. Keep the two in sync.

Per-query recall@k is spec-coverage: |specs matched in top-k| / |specs|. For a
single-spec question that is plain hit/miss; for a cross-company question it
rewards covering BOTH companies, not just one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import yaml

from docintel.embed.service import IndexVersion
from docintel.retrieve.service import RetrievedChunk

QUESTION_TYPES = {"single_fact", "cross_company", "temporal", "out_of_corpus"}


@dataclass(frozen=True)
class ExpectedSpec:
    company: str | None = None
    form_type: str | None = None
    accession: str | None = None
    section_prefix: str | None = None
    must_contain: str | None = None

    def matches(self, chunk: RetrievedChunk) -> bool:
        if self.accession and chunk.document_id != self.accession:
            return False
        if self.company and chunk.company != self.company:
            return False
        if self.form_type and chunk.form_type != self.form_type:
            return False
        if self.section_prefix and not (chunk.section or "").startswith(self.section_prefix):
            return False
        if self.must_contain:
            return self.must_contain.lower() in chunk.text.lower()
        return True


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    type: str
    question: str
    refuse: bool = False
    verified: bool = False  # flipped to true only after human verification
    # "all": every spec must be covered for full recall credit (cross-company,
    #        temporal — the answer NEEDS every source).
    # "any": the specs are alternates; retrieving any one of them answers the
    #        question (e.g. the same fact lives in the 10-K's Item 7 and the
    #        10-Q's Item 2 MD&A).
    require: str = "all"
    expected: list[ExpectedSpec] = field(default_factory=list)


def load_golden(path: Path) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    for item in raw:
        qid = str(item["id"])
        if qid in seen_ids:
            raise ValueError(f"duplicate golden question id {qid}")
        seen_ids.add(qid)
        qtype = item["type"]
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"{qid}: unknown type {qtype!r}")
        refuse = bool(item.get("refuse", False))
        require = item.get("require", "all")
        if require not in ("all", "any"):
            raise ValueError(f"{qid}: require must be 'all' or 'any', got {require!r}")
        specs = [ExpectedSpec(**spec) for spec in item.get("expected", [])]
        if refuse and specs:
            raise ValueError(f"{qid}: refuse questions must not declare expected specs")
        if not refuse and not specs:
            raise ValueError(f"{qid}: non-refuse questions need at least one expected spec")
        questions.append(
            GoldenQuestion(
                id=qid, type=qtype, question=item["question"], refuse=refuse,
                verified=bool(item.get("verified", False)), require=require, expected=specs,
            )
        )
    return questions


def relevant_universe_size(
    conn: psycopg.Connection, version: IndexVersion, specs: list[ExpectedSpec]
) -> int:
    """Count all chunks in this version's chunk set matching ANY spec — the
    relevance universe nDCG's ideal ranking is computed against."""
    clauses, params = [], {
        "strategy": version.strategy, "params_hash": version.params_hash,
    }
    for i, spec in enumerate(specs):
        parts = []
        if spec.accession:
            parts.append(f"c.document_id = %(acc{i})s")
            params[f"acc{i}"] = spec.accession
        if spec.company:
            parts.append(f"d.company = %(co{i})s")
            params[f"co{i}"] = spec.company
        if spec.form_type:
            parts.append(f"d.form_type = %(ft{i})s")
            params[f"ft{i}"] = spec.form_type
        if spec.section_prefix:
            parts.append(f"c.section LIKE %(sec{i})s")
            params[f"sec{i}"] = spec.section_prefix + "%"
        if spec.must_contain:
            parts.append(f"c.text ILIKE %(sub{i})s")
            params[f"sub{i}"] = "%" + spec.must_contain + "%"
        clauses.append("(" + " AND ".join(parts) + ")")
    sql = (
        "SELECT count(*) FROM chunks c JOIN documents d ON d.accession_no = c.document_id "
        "WHERE c.strategy = %(strategy)s AND c.params_hash = %(params_hash)s "
        "AND (" + " OR ".join(clauses) + ")"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]
