# Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion"]
        EDGAR[SEC EDGAR<br/>full-text filings] -->|rate-limited httpx client<br/>UA + backoff + request log| RAW[raw store<br/>data/raw/&lt;accession&gt;]
        RAW --> DOCS[(documents<br/>accession_no PK)]
    end

    subgraph process["Parse + chunk"]
        DOCS --> PARSE[HTML -> text<br/>tables kept, sections detected<br/>hit rate measured]
        PARSE --> CHUNKS[(chunks<br/>strategy + params_hash<br/>+ content_hash + tsvector)]
    end

    subgraph index["Versioned index"]
        CHUNKS -->|only missing content_hash| EMB[bge-small on CPU<br/>OpenAI opt-in]
        EMB --> V[(embeddings_v&lt;id&gt;<br/>one table per index version<br/>HNSW cosine)]
        IV[(index_versions<br/>model x strategy x params x schema)] -.catalogs.- V
    end

    subgraph serve["Retrieval + agent + serving"]
        Q[question] --> ROUTE{route}
        ROUTE -->|lookup| R1[hybrid search<br/>vector + FTS, RRF]
        ROUTE -->|comparison / temporal| R2[fan-out per company / filing<br/>round-robin merge]
        ROUTE -->|out_of_scope| REF[refuse]
        R1 & R2 --> GRADE{grade:<br/>similarity bar +<br/>entity coverage}
        GRADE -->|ok| ANS[answer + citations<br/>doc / section / chunk_id]
        GRADE -->|retry once| ROUTE
        GRADE -->|fail| REF
        V --> R1 & R2
        ANS & REF --> RUNS[(query_runs)]
        API[FastAPI /query /health /metrics] --> ROUTE
        MCP[MCP server<br/>search_filings, get_chunk, ...] --> R1
    end

    subgraph eval["Evaluation (the point)"]
        GOLD[eval/golden.yaml<br/>50 questions, typed,<br/>mechanically validated] --> HARNESS[docintel eval]
        V --> HARNESS
        HARNESS --> REPORT[docs/eval/*.md<br/>recall@k, MRR, nDCG,<br/>refusal P/R, latency, sweep]
    end
```

The agent graph itself (LangGraph) is exported by `docintel export-graph` to
[agent_graph.md](agent_graph.md). Design rationale lives in
[decisions/](decisions/).
