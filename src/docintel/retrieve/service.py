"""Retrieval against one index version: vector, full-text, or hybrid (RRF).

Hybrid fuses the two ranked lists with Reciprocal Rank Fusion:
score(c) = Σ_lists 1/(rrf_k + rank). RRF is used instead of score blending
because cosine similarities and ts_rank_cd live on incomparable scales; rank
fusion needs no per-corpus weight tuning and is the standard baseline.

Refusal uses the top-1 VECTOR cosine score even in hybrid mode: RRF scores are
rank artifacts with no absolute meaning, while cosine against the query is a
calibratable confidence signal (threshold measured in the eval report).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg

from docintel.embed.providers import EmbeddingProvider
from docintel.embed.service import IndexVersion

RRF_K = 60  # standard constant; rank-1 contribution 1/61
CANDIDATE_POOL = 50  # each leg contributes its top-50 to the fusion


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: str
    company: str
    form_type: str
    section: str | None
    text: str
    score: float  # cosine (vector), ts_rank_cd (fts), or RRF (hybrid)
    rank: int  # 1-based


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    top_vector_score: float  # top-1 cosine — the refusal signal
    latency_ms: float


_SELECT = """
SELECT c.chunk_id, c.document_id, d.company, d.form_type, c.section, c.text, {score} AS score
FROM chunks c
JOIN documents d ON d.accession_no = c.document_id
{extra_join}
WHERE c.strategy = %(strategy)s AND c.params_hash = %(params_hash)s
{where}
ORDER BY {order}
LIMIT %(k)s
"""


class Retriever:
    def __init__(
        self, conn: psycopg.Connection, provider: EmbeddingProvider, version: IndexVersion
    ) -> None:
        self.conn = conn
        self.provider = provider
        self.version = version

    def _params(self, k: int) -> dict:
        return {"strategy": self.version.strategy, "params_hash": self.version.params_hash, "k": k}

    def _rows(self, sql: str, params: dict) -> list[RetrievedChunk]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                RetrievedChunk(
                    chunk_id=r[0], document_id=r[1], company=r[2], form_type=r[3],
                    section=r[4], text=r[5], score=float(r[6]), rank=i + 1,
                )
                for i, r in enumerate(cur.fetchall())
            ]

    def vector_search(self, question: str, k: int = 10) -> list[RetrievedChunk]:
        vec = "[" + ",".join(f"{x:.8f}" for x in self.provider.embed_query(question)) + "]"
        sql = _SELECT.format(
            score="1 - (e.embedding <=> %(vec)s::vector)",
            extra_join=f"JOIN {self.version.table_name} e ON e.content_hash = c.content_hash",
            where="",
            order="e.embedding <=> %(vec)s::vector",
        )
        return self._rows(sql, {**self._params(k), "vec": vec})

    def fts_search(self, question: str, k: int = 10) -> list[RetrievedChunk]:
        sql = _SELECT.format(
            score="ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(q)s))",
            extra_join="",
            where="AND c.tsv @@ websearch_to_tsquery('english', %(q)s)",
            order="score DESC",
        )
        return self._rows(sql, {**self._params(k), "q": question})

    def search(self, question: str, k: int = 10, mode: str = "hybrid") -> RetrievalResult:
        started = time.perf_counter()
        if mode == "vector":
            chunks = self.vector_search(question, k)
            top_vec = chunks[0].score if chunks else 0.0
        elif mode == "hybrid":
            vec_list = self.vector_search(question, CANDIDATE_POOL)
            fts_list = self.fts_search(question, CANDIDATE_POOL)
            top_vec = vec_list[0].score if vec_list else 0.0
            fused: dict[int, float] = {}
            by_id: dict[int, RetrievedChunk] = {}
            for ranked in (vec_list, fts_list):
                for chunk in ranked:
                    fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (
                        RRF_K + chunk.rank
                    )
                    by_id.setdefault(chunk.chunk_id, chunk)
            top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
            chunks = [
                RetrievedChunk(
                    **{**by_id[cid].__dict__, "score": score, "rank": i + 1},
                )
                for i, (cid, score) in enumerate(top)
            ]
        else:
            raise ValueError(f"unknown retrieval mode {mode!r} (vector | hybrid)")
        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResult(chunks=chunks, top_vector_score=top_vec, latency_ms=latency_ms)
