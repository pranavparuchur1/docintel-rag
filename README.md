# docintel-rag

An incremental document-intelligence pipeline that ingests SEC filings, chunks and
embeds them into a **versioned vector index**, serves grounded answers with citations
through a LangGraph agent, and — critically — **measures its own retrieval quality**
against a golden question set instead of claiming it works.

Runs end to end with **zero paid API keys** on `docker compose`. Nothing in
this README is claimed without a measured number behind it.

🏗 [Architecture diagram](docs/architecture.md) ·
🕸 [Agent graph](docs/agent_graph.md) ·
📊 [Latest eval report](docs/eval/comparison_2026-08-21.md) ·
📜 [Design decisions (ADRs)](docs/decisions/) ·
🎥 3-minute walkthrough: *Loom link coming — script in
[docs/walkthrough_script.md](docs/walkthrough_script.md)*

## Principles

- **Content-hash everything** — re-runs re-embed only changed chunks.
- **The index is versioned** — `(embedding_model, chunk_strategy, chunk_params, schema_version)`
  defines an index version; serve one while building another.
- **Chunking is compared, not assumed** — three strategies evaluated on the same golden set.
- **Nothing is claimed without a number** — recall@k, MRR@10, nDCG@10, refusal
  precision/recall, p50/p95 latency, cost per query.
- **The system can say "I don't know"** — below-threshold retrieval refuses instead of
  hallucinating, and refusal quality is itself measured.
- **Zero paid keys required** — local embeddings (bge-small on CPU), local Postgres +
  pgvector, all via Docker Compose. Cloud providers are opt-in env vars.

## Quickstart (clean clone, zero paid keys)

```bash
cp .env.example .env          # edit EDGAR_USER_AGENT to identify yourself (SEC requires it)
docker compose up -d db       # Postgres 16 + pgvector
docker compose build app
docker compose run --rm app docintel db upgrade
docker compose run --rm app docintel db check

# ingest one company (Apple) to try it out — or `make corpus` for all six
docker compose run --rm app docintel ingest --cik 320193 --forms 10-K --limit 1
docker compose run --rm app docintel chunk --strategy all
docker compose run --rm app docintel parse-report   # measured section-detection hit rate
docker compose run --rm app docintel embed          # bge-small on CPU; incremental by content hash
docker compose run --rm app docintel index-versions

# ask (retrieval-only answers with citations; set LLM_PROVIDER for generation)
docker compose run --rm app docintel ask "What risks does Apple report about its supply chain?" --index-version 1
docker compose run --rm app docintel eval --mode hybrid,agent   # writes docs/eval/*.md
make serve                                          # FastAPI on :8000
```

Unit tests and lint (any Python ≥ 3.11 env): `pip install -r requirements-dev.txt
&& pip install --no-deps -e .`, then `make test` and `make lint` (needs the
compose db running for the Postgres-backed tests; they skip otherwise).

Ingestion is idempotent: re-running the same command re-downloads nothing and
duplicates nothing (the EDGAR accession number is the natural key). The EDGAR
client throttles below SEC's ~10 req/s fair-access cap, sends the required
identifying User-Agent, honors `Retry-After`, and backs off exponentially with
jitter on 429/5xx.

## Parsing & chunking (measured, not assumed)

`docintel parse-report` measures section detection against the key answerable
sections (10-K: Items 1, 1A, 7, 7A, 8; 10-Q: I.1, I.2, II.1A). On the current
18-document corpus the hit rate is **76/76 = 100%**, verified not just by
heading presence but by section content size (a mis-detected heading produces a
near-empty section). Two real-world traps this heuristic had to survive:
Microsoft prints bare `Item 1A` page headers on every page (titled headings now
beat bare ones), and JPMorgan's Item 7 is genuinely ~400 chars because it
incorporates MD&A by reference — a corpus limitation, not a parser bug.

Three chunk strategies live behind one `ChunkStrategy` interface — `fixed`
(350 tokens, 60 overlap), `recursive` (natural boundaries, merged up to 350),
and `section_aware` (recursive, but never across a detected section boundary).
Which one is *better* is a Phase 4 evaluation question; no claim is made here.

## Embedding & the versioned index

Embeddings are **content-addressed**: keyed by the SHA-256 of normalized chunk
text, stored in per-version tables `embeddings_v<id>` (dimension-typed pgvector
column + HNSW, cosine). An index version is the tuple
`(embedding_model, chunk_strategy, chunk_params, schema_version)` — change any
element and a new version builds in its own table while the old one keeps
serving. Measured on this corpus (bge-small-en-v1.5, CPU): first full embed of
15,365 vectors ≈ 25 min at ~10 chunks/s; the identical re-run embeds **0
chunks in 7.7 s**. Rationale in
[ADR 0002](docs/decisions/0002-content-addressed-versioned-embeddings.md).

## Retrieval quality (measured)

Evaluated on a 50-question golden set (42 in-corpus, 8 out-of-corpus hard
negatives) with `docintel eval`; full report incl. per-type breakdown,
refusal-threshold sweep, and misses in
[docs/eval/comparison_2026-08-21.md](docs/eval/comparison_2026-08-21.md).
**Verification status: every question's ground truth is mechanically confirmed
to exist in the corpus (`docintel check-golden`), 0/50 human-verified so far —
that count is printed in every report and is a stated limitation until closed.**

| configuration | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 | refusal P / R | p95 |
|---|---|---|---|---|---|---|---|
| fixed(350/60) / vector | 0.39 | 0.60 | 0.67 | 0.54 | 0.33 | 1.00 / 0.38 | 47ms |
| fixed(350/60) / hybrid | 0.40 | 0.60 | 0.65 | 0.53 | 0.34 | 1.00 / 0.38 | 63ms |
| recursive(350) / vector | 0.35 | 0.55 | 0.72 | 0.51 | 0.34 | 1.00 / 0.50 | 36ms |
| recursive(350) / hybrid | 0.38 | 0.57 | 0.72 | 0.53 | 0.34 | 1.00 / 0.50 | 44ms |
| **section_aware(350) / hybrid** | **0.40** | **0.60** | 0.67 | **0.55** | **0.35** | **1.00 / 0.50** | 44ms |
| section_aware(350) / vector | 0.36 | 0.57 | 0.67 | 0.52 | 0.34 | 1.00 / 0.50 | 40ms |
| section_aware(250) / vector | 0.40 | 0.51 | 0.70 | 0.52 | 0.32 | 1.00 / 0.38 | 36ms |
| section_aware(250) / hybrid | 0.42 | 0.54 | 0.70 | 0.54 | 0.32 | 1.00 / 0.38 | 38ms |

What the numbers actually say (caveat: 42 in-corpus questions, so one question
moves recall by ~0.024 — differences under ~0.05 are noise):

- **section_aware/hybrid leads on MRR, nDCG and refusal recall; recall@5 is a
  statistical tie with fixed.** Hypothesis: section boundaries stop chunks from
  bleeding between Risk Factors and MD&A (better ranking, exact citations), and
  the full-text leg catches exact terms cosine blurs ("Intelligent Cloud",
  "Digital Markets Act"). Fixed's overlap buys competitive recall through
  redundancy, but its duplicated boundary text wastes result slots.
- **Smaller chunks (250) hurt recall@5** (0.54 vs 0.60): facts split across
  more chunks compete for the same k slots.
- **Single-fact recall@5 is 0.83; cross-company is 0.25–0.30 and temporal
  0.12–0.31.** Single-shot retrieval rarely covers two sources in five slots —
  this measured gap is what the agent's per-company fan-out exists to close
  (its measured effect is in the next section).
- **Refusal at the swept threshold (0.68): precision 1.00, recall 0.50.**
  Cosine similarity alone cannot detect entity absence — "What supply chain
  risks does Walmart report?" retrieves other companies' supply-chain text at
  high similarity. The agent's entity-aware grader is the second line of
  defense; its measured effect is in the next section.

## The agent (LangGraph)

`docintel ask` runs route → retrieve → grade → answer | refuse
([diagram](docs/agent_graph.md)), bounded to one deterministic rewrite, with
every run logged to `query_runs` (route, chunk ids, scores, tokens, latency,
refusal). It runs **with zero API keys**: rule-based routing, entity-aware
grading, and extractive answers that quote evidence verbatim with citations
(`LLM_PROVIDER=anthropic|openai` upgrades answering to grounded generation).

Measured effect of agent planning vs. plain hybrid retrieval (v3, same golden set):

| configuration | recall@5 | recall@10 | MRR@10 | refusal P / R | p95 |
|---|---|---|---|---|---|
| plain hybrid | 0.60 | 0.68 | 0.55 | 1.00 / 0.50 | 71ms |
| agent-planned | 0.65 | 0.76 | 0.54 | **1.00 / 1.00** | 778ms |

- **Refusals fixed completely** (recall 0.50 → 1.00 at precision 1.00): the
  entity guard refuses questions naming non-corpus companies regardless of
  cosine score, and no-entity queries face a higher bar (0.75) — the closest
  call in the OOC set passed at 0.748 cosine, a 0.002 margin, which is why
  the Phase-5-optional LLM grader remains on the roadmap.
- **Comparison fan-out works**: both companies appear in the evidence for
  every answered comparison; single-fact recall@5 rose 0.83 → 0.92 from
  company-filtered retrieval.
- **Honest residual**: temporal recall@5 is unchanged at 0.25 — evidence from
  both filings now lands in the top 10 (recall@10 rose 0.68 → 0.76) but two
  accession-pinned specs rarely both fit in five slots. Latency also rises
  (one embedding + search per fan-out leg).

## Serving: HTTP API and MCP

`docintel serve` (or `make serve` for the containerized version) exposes:

- `POST /query` — the full agent (validated input, per-IP rate limit)
- `GET /health` — db, pgvector, and the index version being served
- `GET /metrics` — query count, refusal rate, p50/p95 latency from `query_runs`

`docintel mcp-serve` exposes the corpus as **MCP tools** over stdio —
`search_filings(query, company?, form_type?, k)`, `get_chunk(chunk_id)`,
`list_companies()`, `get_eval_report()` — so any MCP client can use the corpus
as a tool. Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "docintel": {
      "command": "docintel",
      "args": ["mcp-serve"],
      "cwd": "/path/to/docintel-rag"
    }
  }
}
```

Session transcript — the official `mcp` Python client (2.0.0) driving the
server over stdio, answering a corpus question end to end:

```text
$ connected to 'docintel' over stdio (official mcp 2.0.0 Python client)
$ tools/list -> search_filings, get_chunk, list_companies, get_eval_report

$ call search_filings(query='risks from export controls on China', company='nvidia', k=3)
    chunk=18358 score=0.032 NVIDIA CORP 10-Q Item 1A. Risk Factors
      "As a result, export controls have in the past and may in the future negatively impact demand..."
    chunk=17615 score=0.032 NVIDIA CORP 10-K Item 1A. Risk Factors
      "Excessive or shifting export controls have already and may in the future encourage customers..."

$ call get_chunk(18358)
    NVIDIA CORP 10-Q filed 2026-05-20, Item 1A. Risk Factors, 674 chars
    source: https://www.sec.gov/Archives/edgar/data/1045810/000104581026000052/nvda-20260426.htm
```

## Layout

```
src/docintel/   ingest | parse | chunk | embed | index | retrieve | agent | eval | api | mcp
sql/            plain-SQL migrations, applied in order by `docintel db upgrade`
tests/          pytest (unit + eval smoke test on a fixture corpus)
dags/           one thin Airflow DAG wrapping the CLI
docs/decisions/ architecture decision records
configs/        declarative chunking/eval configuration
```

## Design decisions

Full rationale as ADRs in [docs/decisions/](docs/decisions/); the short version:

- **Content-addressed, versioned embeddings** — vectors keyed by chunk text
  hash in one physical table per index version; rebuilds are cheap, cutover is
  zero-downtime, and a re-run costs nothing
  ([ADR 0002](docs/decisions/0002-content-addressed-versioned-embeddings.md)).
- **HNSW over IVFFlat** — no training step, so recall stays correct under
  incremental inserts (same ADR).
- **Adapters at every expensive boundary** — `VECTOR_BACKEND`,
  `EMBEDDING_PROVIDER`, `LLM_PROVIDER` are env switches; the free local path is
  the default and the tested one.
- **Chunking compared, never assumed** — three strategies behind one interface,
  ranked by the same golden set (table above).
- **Refusal thresholds are measured, not guessed** — 0.68 / 0.75 (no-entity)
  come from the report's threshold sweep, and refusal quality is itself part of
  the eval.
- **No MinIO** — ~20 immutable filings, one writer; `documents.raw_path` is the
  seam if the corpus outgrows a disk
  ([ADR 0001](docs/decisions/0001-raw-store-on-local-disk-not-minio.md)).
- **Orchestration stays thin** — the [Airflow DAG](dags/docintel_pipeline.py)
  shells out to the same CLI a human runs; retries are safe because idempotency
  lives in the pipeline (natural keys + content hashes), not the scheduler.

## Known limitations

Named deliberately — each one is a measurement or a conscious scope call:

- **Golden-set verification is incomplete**: all 42 in-corpus questions are
  mechanically validated (`docintel check-golden`), but **0/50 are
  human-verified** yet; every eval report prints that count.
- **Small corpus, small question set**: 18 filings, 42 scored questions — one
  question moves recall@5 by ~0.024, so a 95% CI on recall@5 ≈ ±0.15. The
  numbers rank configurations; they are not population estimates.
- **Embedding-model ceiling**: bge-small-en-v1.5 (384-dim, CPU) caps semantic
  quality; the versioned index exists precisely so a better model is a new
  version + cutover, not a migration.
- **Temporal questions**: recall@5 stuck at 0.25 even with fan-out (both
  filings reach the top 10, rarely the top 5 together).
- **Refusal edge**: the hardest out-of-corpus question cleared by 0.002 cosine.
  Deterministic guards got refusal to 1.00/1.00 on this set, but the set is 8
  questions.
- **JPMorgan's Item 7** is ~400 chars in the primary document (MD&A
  incorporated by reference) — a corpus property, not a parser bug.
- **Ops**: in-process rate limiter (per-worker), per-leg query embedding is
  uncached (agent p95 ≈ 0.8s), single-machine Compose deployment.

## With more time

LLM-graded relevance behind the deterministic guard (measured against the same
golden set before adoption) · query-embedding cache · cross-encoder reranking as
a fourth compared configuration · human verification of all 50 golden questions
· a second embedding model as index version v5 to demonstrate live A/B ·
Prometheus-format `/metrics`.
