-- Registry of chunk-parameter sets: chunks carry only params_hash, this table
-- keeps the actual params JSON so an index version can record exactly how its
-- chunks were produced.
CREATE TABLE chunk_param_sets (
    strategy    text  NOT NULL,
    params_hash text  NOT NULL,
    params      jsonb NOT NULL,
    PRIMARY KEY (strategy, params_hash)
);

-- An index version is the tuple (embedding_model, chunk strategy, chunk params,
-- schema_version). Changing ANY element creates a new version; nothing is ever
-- silently mixed. Embedding vectors live in per-version tables embeddings_v<id>
-- (created at embed time, because the vector column's dimension depends on the
-- model) — so one version can serve while another builds, and dropping a
-- version is dropping a table.
CREATE TABLE index_versions (
    index_version_id int   GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    embedding_model  text  NOT NULL,
    embedding_dim    int   NOT NULL,
    strategy         text  NOT NULL,
    params_hash      text  NOT NULL,
    chunk_params     jsonb NOT NULL,
    schema_version   int   NOT NULL,
    status           text  NOT NULL DEFAULT 'building' CHECK (status IN ('building', 'ready')),
    chunk_count      int   NOT NULL DEFAULT 0,
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (embedding_model, strategy, params_hash, schema_version)
);
