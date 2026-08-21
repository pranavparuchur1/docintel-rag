-- Full agent-run telemetry: every /ask (and later /query API call) is a row.
-- This is the substrate for debugging bad answers, A/B-ing prompts, and
-- auditing refusals — the answer to "how would you improve it in production".
CREATE TABLE query_runs (
    run_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts               timestamptz NOT NULL DEFAULT now(),
    query            text NOT NULL,
    route            text NOT NULL,           -- lookup | comparison | temporal | out_of_scope
    index_version_id int REFERENCES index_versions (index_version_id),
    mode             text NOT NULL,           -- vector | hybrid
    retrieved_chunk_ids bigint[] NOT NULL DEFAULT '{}',
    scores           double precision[] NOT NULL DEFAULT '{}',
    top_vector_score double precision,
    rewritten        boolean NOT NULL DEFAULT false,
    refused          boolean NOT NULL,
    refusal_reason   text,
    answer           text,
    llm_provider     text NOT NULL,
    tokens_in        int NOT NULL DEFAULT 0,
    tokens_out       int NOT NULL DEFAULT 0,
    latency_ms       double precision NOT NULL
);

CREATE INDEX query_runs_ts_idx ON query_runs (ts);
