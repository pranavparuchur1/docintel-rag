-- One row per SEC filing. accession_no is EDGAR's globally unique, immutable
-- identifier for a filing, so it is the natural primary key: re-ingesting can
-- never duplicate a document, and restated filings (e.g. 10-K/A) arrive as new
-- accession numbers rather than mutations of this row.
CREATE TABLE documents (
    accession_no text PRIMARY KEY,
    cik          text NOT NULL,
    company      text NOT NULL,
    form_type    text NOT NULL,
    filing_date  date NOT NULL,
    period       date,                       -- fiscal period the filing reports on
    source_url   text NOT NULL,
    raw_path     text NOT NULL,              -- relative to DATA_DIR (portable host <-> container)
    content_hash text NOT NULL,              -- sha256 of the raw bytes, exactly as served by EDGAR
    size_bytes   bigint NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_cik_form_idx ON documents (cik, form_type);
