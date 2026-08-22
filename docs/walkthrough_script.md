# 3-minute Loom walkthrough — talk track

Target: recruiter/interviewer deciding in 3 minutes whether to read the code.
One take, screen share, terminal + README side by side.

**0:00–0:25 — The pitch.** "This is docintel-rag: it ingests SEC filings,
chunks and embeds them into a versioned vector index, and serves grounded,
cited answers through a LangGraph agent. The differentiator: it measures its
own retrieval quality — every claim in the README has a number from the eval
harness."

**0:25–0:55 — The eval table (README).** Scroll to the retrieval-quality
table. "Three chunking strategies times two retrieval modes on a 50-question
golden set. Section-aware plus hybrid won on MRR and nDCG; recall@5 was a
statistical tie with fixed — and the README says so, because with 42 scored
questions differences under 0.05 are noise."

**0:55–1:30 — Incrementality (terminal).** Run `docintel embed` live. "Zero
chunks embedded, seven seconds — embeddings are keyed by content hash in
per-version tables, so re-runs are free and a model change builds a new
version while the old one keeps serving. First build was 25 minutes; that
asymmetry is the whole design."

**1:30–2:05 — The agent refusing (terminal).** Run
`docintel ask "What supply chain risks does Walmart report?" --index-version 3`.
"Refused — Walmart isn't in the corpus. Cosine similarity alone can't catch
this; the router's entity guard can, and refusal precision/recall are measured
in the eval: 1.00 and 1.00 through the agent." Then one real question —
citations point to document, section, chunk id.

**2:05–2:35 — MCP (README transcript or live client).** "The corpus is also an
MCP server — search_filings, get_chunk, list_companies, get_eval_report — so
any agent client can use it as a tool. Here's Claude Desktop querying it."

**2:35–3:00 — Close.** "Everything runs from a clean clone with zero paid
keys: docker compose, local embeddings, local Postgres. Limitations are in the
README on purpose — small corpus, verification status of the golden set, the
temporal-recall gap — because knowing where a system fails is the job."
