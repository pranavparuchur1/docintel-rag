# ADR 0001 — Raw filings land on local disk, not MinIO

**Status:** accepted (Phase 0)

## Context

The nyc311-pipeline repo uses MinIO as an object-storage landing zone, which is the
right shape for a batch ELT system with independent producers and lifecycle policies.
This repo ingests ~15–20 SEC filings, written by exactly one single-threaded ingest
process, read by exactly one parser, with no lifecycle management and no concurrent
writers.

## Decision

Raw documents are stored unmodified under `DATA_DIR/raw/<accession_no>/`, with the
path recorded in `documents.raw_path`. No MinIO container.

## Consequences

- One less service in docker-compose; faster cold start for anyone cloning the repo.
- The `documents.raw_path` column is the storage abstraction: swapping the value for
  an `s3://` URI (and the file I/O for an object-store client) is a contained change
  if the corpus ever outgrows a laptop.
- Trade-off accepted: no bucket-level versioning of raw documents. EDGAR filings are
  immutable once published (restatements arrive as *new* accession numbers), so
  versioning the raw store adds cost without a use case.
