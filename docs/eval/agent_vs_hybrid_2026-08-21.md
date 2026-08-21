# Retrieval evaluation — 2026-08-21 23:55 UTC

- golden set: **50 questions** (42 in-corpus, 8 out-of-corpus), **0/50 human-verified**
- embedding model: `BAAI/bge-small-en-v1.5` · refusal threshold (top-1 cosine): 0.68
- caveat: with 42 in-corpus questions one question moves recall by ~0.024; two decimals are reported, differences smaller than ~0.05 are noise.

## Results

| configuration | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | refusal_precision | refusal_recall | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| v3 section_aware / hybrid | 0.40 | 0.60 | 0.68 | 0.55 | 0.35 | 1.00 | 0.50 | 48 | 71 |
| v3 section_aware / agent | 0.35 | 0.65 | 0.76 | 0.54 | 0.36 | 1.00 | 1.00 | 85 | 778 |

recall@k is spec coverage (a cross-company question counts fully only when every expected source is retrieved); definitions in `src/docintel/eval/metrics.py`.

## Per-type recall@5

| configuration | cross_company | single_fact | temporal |
|---|---|---|---|
| v3 section_aware / hybrid | 0.30 | 0.83 | 0.25 |
| v3 section_aware / agent | 0.35 | 0.92 | 0.25 |

## Refusal threshold sweep (top-1 cosine)

| threshold | v3 section_aware / hybrid | v3 section_aware / agent |
|---|---|---|
| 0.40 | P=1.00 R=0.00 | P=1.00 R=0.88 |
| 0.42 | P=1.00 R=0.00 | P=1.00 R=0.88 |
| 0.44 | P=1.00 R=0.00 | P=1.00 R=0.88 |
| 0.46 | P=1.00 R=0.00 | P=1.00 R=0.88 |
| 0.48 | P=1.00 R=0.00 | P=1.00 R=0.88 |
| 0.50 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.52 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.54 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.56 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.58 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.60 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.62 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.64 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.66 | P=1.00 R=0.12 | P=1.00 R=0.88 |
| 0.68 | P=1.00 R=0.50 | P=1.00 R=0.88 |
| 0.70 | P=0.56 R=0.62 | P=0.64 R=0.88 |

## Cost per query

Local embeddings + local Postgres: **$0.00 marginal cost per query** (measured configuration). For reference, the same query embedded with OpenAI text-embedding-3-small would cost ≈ $0.0000004 at ~20 query tokens ($0.02/1M tokens); retrieval itself stays free either way. LLM generation cost lands in Phase 5.

## Misses (no relevant chunk in top 10)

- v3 section_aware / hybrid: sf05, sf06, sf14, cc04, cc05, cc10, tp04, tp05, tp06, tp07
- v3 section_aware / agent: cc05, cc08, cc10, tp04, tp05, tp07
