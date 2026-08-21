"""Fan-out retrieval for multi-source questions.

The eval measured why this exists: single-shot retrieval scored recall@5 of
0.25-0.30 on cross-company questions and 0.12-0.31 on temporal ones, because
two sources rarely fit into the same top-k. Fan-out runs one filtered
retrieval per source and round-robin merges, so every expected source is
represented by construction.
"""

from __future__ import annotations

import time

import psycopg

from docintel.retrieve.service import RetrievalResult, RetrievedChunk, Retriever


def _round_robin(
    legs: list[list[RetrievedChunk]], k: int
) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[int] = set()
    for i in range(max((len(leg) for leg in legs), default=0)):
        for leg in legs:
            if i < len(leg) and leg[i].chunk_id not in seen and len(merged) < k:
                seen.add(leg[i].chunk_id)
                merged.append(leg[i])
    return [
        RetrievedChunk(**{**chunk.__dict__, "rank": rank + 1})
        for rank, chunk in enumerate(merged)
    ]


def comparison_search(
    retriever: Retriever, question: str, companies: list[str], k: int = 10,
    mode: str = "hybrid",
) -> RetrievalResult:
    """One leg per company; merged round-robin so each company holds ~k/n slots."""
    started = time.perf_counter()
    per_leg = max(2, k // max(len(companies), 1))
    legs, top_vec = [], 0.0
    for company in companies:
        result = retriever.search(question, k=per_leg, mode=mode, companies=[company])
        legs.append(result.chunks)
        top_vec = max(top_vec, result.top_vector_score)
    return RetrievalResult(
        chunks=_round_robin(legs, k),
        top_vector_score=top_vec,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def temporal_search(
    conn: psycopg.Connection, retriever: Retriever, question: str, company: str,
    k: int = 10, mode: str = "hybrid", max_filings: int = 4,
    form_type: str | None = None,
) -> RetrievalResult:
    """One leg per filing of the company (newest first), so 'how did X change'
    always has evidence from more than one point in time. form_type narrows the
    legs when the question names a cadence ('annual reports' -> 10-K only)."""
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT accession_no FROM documents WHERE company = %s "
            "AND (%s::text IS NULL OR form_type = %s) "
            "ORDER BY filing_date DESC LIMIT %s",
            (company, form_type, form_type, max_filings),
        )
        accessions = [r[0] for r in cur.fetchall()]
    per_leg = max(2, k // max(len(accessions), 1))
    legs, top_vec = [], 0.0
    for accession in accessions:
        result = retriever.search(question, k=per_leg, mode=mode, accessions=[accession])
        legs.append(result.chunks)
        top_vec = max(top_vec, result.top_vector_score)
    return RetrievalResult(
        chunks=_round_robin(legs, k),
        top_vector_score=top_vec,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
