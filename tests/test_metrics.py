"""Hand-computed cases — if these numbers can't be defended, nothing above
them can. Judgments are built with a helper so intent stays readable."""

from math import log2

from docintel.eval.metrics import (
    QueryJudgment,
    mrr_at_10,
    ndcg_at_10,
    percentile,
    recall_at,
    refusal_precision_recall,
)


def judgment(
    relevance: list[bool],
    specs_total: int = 1,
    specs_hit_at: dict | None = None,
    should_refuse: bool = False,
    refused: bool = False,
    universe: int = 1,
) -> QueryJudgment:
    first = next((i + 1 for i, r in enumerate(relevance) if r), None)
    if specs_hit_at is None:
        specs_hit_at = {k: (1 if any(relevance[:k]) else 0) for k in (1, 5, 10)}
    return QueryJudgment(
        question_id="q", question_type="single_fact", should_refuse=should_refuse,
        refused=refused, specs_total=specs_total, specs_hit_at=specs_hit_at,
        first_relevant_rank=first, relevance=relevance, relevant_universe=universe,
        latency_ms=10.0, top_vector_score=0.8,
    )


HIT_AT_1 = judgment([True] + [False] * 9)
HIT_AT_3 = judgment([False, False, True] + [False] * 7)
MISS = judgment([False] * 10)


def test_recall_at_k_single_spec():
    js = [HIT_AT_1, HIT_AT_3, MISS]
    assert recall_at(js, 1) == 1 / 3  # only the rank-1 hit
    assert recall_at(js, 5) == 2 / 3
    assert recall_at(js, 10) == 2 / 3


def test_recall_multi_spec_partial_credit():
    # cross-company question expecting 2 sources; only 1 covered in top-5
    j = judgment([True] + [False] * 9, specs_total=2, specs_hit_at={1: 1, 5: 1, 10: 2})
    assert recall_at([j], 5) == 0.5
    assert recall_at([j], 10) == 1.0


def test_mrr():
    assert mrr_at_10([HIT_AT_1, HIT_AT_3, MISS]) == (1.0 + 1 / 3 + 0.0) / 3


def test_ndcg_perfect_and_positional():
    # one relevant chunk in corpus, retrieved at rank 1 -> nDCG 1.0
    assert ndcg_at_10([judgment([True] + [False] * 9, universe=1)]) == 1.0
    # same chunk at rank 3 -> DCG = 1/log2(4), IDCG = 1/log2(2) = 1
    expected = (1 / log2(4)) / 1.0
    got = ndcg_at_10([judgment([False, False, True] + [False] * 7, universe=1)])
    assert abs(got - expected) < 1e-9


def test_ndcg_uses_true_universe():
    # 3 relevant chunks exist; retrieving 1 at rank 1 is NOT a perfect ranking
    ideal = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    expected = 1.0 / ideal
    assert abs(ndcg_at_10([judgment([True] + [False] * 9, universe=3)]) - expected) < 1e-9


def test_refusal_questions_excluded_from_ranking_metrics():
    refuse_j = judgment([False] * 10, should_refuse=True, refused=True)
    assert recall_at([HIT_AT_1, refuse_j], 1) == 1.0
    assert mrr_at_10([HIT_AT_1, refuse_j]) == 1.0


def test_refusal_precision_recall():
    js = [
        judgment([False] * 10, should_refuse=True, refused=True),   # TP
        judgment([False] * 10, should_refuse=True, refused=False),  # FN
        judgment([True] + [False] * 9, refused=True),               # FP
        HIT_AT_1,                                                   # TN
    ]
    precision, recall = refusal_precision_recall(js)
    assert precision == 0.5  # 1 of 2 refusals was right
    assert recall == 0.5  # 1 of 2 should-refuse was caught


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 100.0
    assert percentile([], 95) == 0.0
