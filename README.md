# docintel-rag

An incremental document-intelligence pipeline that ingests SEC filings, chunks and
embeds them into a **versioned vector index**, serves grounded answers with citations
through a LangGraph agent, and — critically — **measures its own retrieval quality**
against a golden question set instead of claiming it works.

> **Status: under construction.** Built phase by phase; nothing is claimed here
> without a measured number behind it. The evaluation table will appear in this
> README when the eval harness (Phase 4) produces it.

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

## Quickstart (Phase 0 scope)

```bash
cp .env.example .env          # edit EDGAR_USER_AGENT to identify yourself
docker compose up -d db       # Postgres 16 + pgvector
docker compose build app
docker compose run --rm app docintel db upgrade
docker compose run --rm app docintel db check
make test                     # unit tests (any Python >= 3.11 env with requirements-dev.txt)

# ingest one company (Apple), or the whole 6-company corpus:
docker compose run --rm app docintel ingest --cik 320193 --forms 10-K,10-Q --limit 2
make corpus
```

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

Recorded as ADRs in [docs/decisions/](docs/decisions/), starting with
[why there is no MinIO here](docs/decisions/0001-raw-store-on-local-disk-not-minio.md).
