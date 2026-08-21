-- Full-text search leg of hybrid retrieval. A stored generated column keeps
-- the tsvector consistent with chunk text by construction; GIN indexes it.
-- left(text, 100k) guards the 1MB tsvector ceiling (chunks are ~2KB anyway).
ALTER TABLE chunks
    ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', left(text, 100000))) STORED;

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
