"""Retrieval metrics as pure functions over binary relevance judgments.

Definitions (documented because the numbers get quoted):

- recall@k       per query: |specs with ≥1 matching chunk in top-k| / |specs|.
                 Averaged over in-corpus queries. For single-spec questions
                 this is hit-rate@k; for multi-spec (cross-company) questions
                 it rewards covering every expected source.
- MRR@10         1 / rank of the first relevant chunk (0 if none in top 10),
                 averaged over in-corpus queries.
- nDCG@10        DCG with binary gains, normalized by the ideal DCG given the
                 TRUE number of relevant chunks in the corpus (computable here
                 because relevance is a SQL predicate, not human top-k labels).
- refusal P/R    positive class = refusal. Precision: of the queries refused,
                 how many should have been refused. Recall: of the queries
                 that should be refused, how many were.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class QueryJudgment:
    """Binary relevance of one query's ranked retrieval, plus refusal facts."""

    question_id: str
    question_type: str
    should_refuse: bool
    refused: bool
    specs_total: int  # 0 for refuse questions
    specs_hit_at: dict[int, int]  # k -> specs with a match in top-k
    first_relevant_rank: int | None  # 1-based, None if no relevant in top-k
    relevance: list[bool]  # per-rank relevance of the top-k list
    relevant_universe: int  # total relevant chunks in corpus
    latency_ms: float
    top_vector_score: float  # top-1 cosine — refusal signal, reused in sweeps


def recall_at(judgments: list[QueryJudgment], k: int) -> float:
    scored = [j for j in judgments if not j.should_refuse]
    if not scored:
        return 0.0
    return sum(j.specs_hit_at.get(k, 0) / j.specs_total for j in scored) / len(scored)


def mrr_at_10(judgments: list[QueryJudgment]) -> float:
    scored = [j for j in judgments if not j.should_refuse]
    if not scored:
        return 0.0
    total = 0.0
    for j in scored:
        if j.first_relevant_rank is not None and j.first_relevant_rank <= 10:
            total += 1.0 / j.first_relevant_rank
    return total / len(scored)


def ndcg_at_10(judgments: list[QueryJudgment]) -> float:
    scored = [j for j in judgments if not j.should_refuse]
    if not scored:
        return 0.0
    total = 0.0
    for j in scored:
        dcg = sum(1.0 / log2(i + 2) for i, rel in enumerate(j.relevance[:10]) if rel)
        ideal_hits = min(j.relevant_universe, 10)
        idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(scored)


def refusal_precision_recall(judgments: list[QueryJudgment]) -> tuple[float, float]:
    refused = [j for j in judgments if j.refused]
    should = [j for j in judgments if j.should_refuse]
    precision = (
        sum(1 for j in refused if j.should_refuse) / len(refused) if refused else 1.0
    )
    recall = sum(1 for j in should if j.refused) / len(should) if should else 1.0
    return precision, recall


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; enough precision for a latency report."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[idx]
