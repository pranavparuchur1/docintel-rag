# ADR 0002 — Content-addressed embeddings, per-version tables, HNSW

**Status:** accepted (Phase 3)

## Context

Embedding is the expensive pipeline stage. The system must (a) never re-embed
unchanged text, (b) never mix embeddings produced under different models or
chunk configs, and (c) serve one index while building another.

## Decisions

### 1. Embeddings are keyed by chunk `content_hash`, not `chunk_id`

The vector for a piece of text depends only on the text and the model — not on
which document/ordinal the text sits at. Keying by the SHA-256 of normalized
chunk text means:

- re-running `embed` is a no-op (hash already present);
- re-chunking with identical params creates new chunk_ids but identical hashes
  → zero re-embedding cost;
- boilerplate repeated across filings is embedded once.

Retrieval joins `chunks.content_hash = embeddings_vN.content_hash`, so
citations still resolve to concrete chunk rows.

### 2. One physical table per index version (`embeddings_v<id>`)

pgvector columns are dimension-typed, and different models have different
dimensions, so a single shared table cannot hold all versions cleanly. A
per-version table also means: the ANN index is isolated per version, building
version N+1 cannot degrade version N's query latency through index bloat, and
retiring a version is `DROP TABLE`. The `index_versions` row is the catalog
entry; the table is the artifact.

### 3. HNSW, not IVFFlat

- IVFFlat requires a training step (`lists` computed over existing rows) —
  built on an empty or partially-loaded table, its recall is silently poor,
  which is exactly the failure mode an incremental pipeline would hit.
- HNSW handles incremental inserts without retraining and gives a better
  recall/latency frontier; its costs (slower builds, more memory) are
  irrelevant at ~5k vectors per version.
- Parameters: `m=16, ef_construction=64` (pgvector defaults, adequate here);
  `ef_search` is left at default 40 and can be tuned per query session.

## Consequences

- The embed step needs a `chunk_param_sets` registry to record params JSON per
  hash (chunks rows carry only the hash).
- Cross-version storage cost grows linearly with versions kept; acceptable at
  this corpus size, and old versions are cheap to drop.
- A vector for a hash embedded under model A must never be read by a version
  using model B — guaranteed structurally, because versions never share tables.
