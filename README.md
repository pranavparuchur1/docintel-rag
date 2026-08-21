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

Recorded as ADRs in [docs/decisions/](docs/decisions/), starting with
[why there is no MinIO here](docs/decisions/0001-raw-store-on-local-disk-not-minio.md).
