-- Chunk identity is (document, strategy, params_hash, ordinal): the same
-- document chunked under different strategies/params coexists side by side,
-- which is what makes the Phase 4 strategy comparison possible.
-- content_hash (sha256 of whitespace-normalized text) is the incrementality
-- key for embedding: unchanged text is never re-embedded.
CREATE TABLE chunks (
    chunk_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  text NOT NULL REFERENCES documents (accession_no) ON DELETE CASCADE,
    strategy     text NOT NULL,
    params_hash  text NOT NULL,
    section      text,
    ordinal      int NOT NULL,
    text         text NOT NULL,
    token_count  int NOT NULL,
    content_hash text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, strategy, params_hash, ordinal)
);

CREATE INDEX chunks_content_hash_idx ON chunks (content_hash);
CREATE INDEX chunks_strategy_idx ON chunks (strategy, params_hash);
