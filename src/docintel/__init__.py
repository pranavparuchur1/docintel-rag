"""docintel — incremental document-intelligence pipeline over SEC filings."""

__version__ = "0.1.0"

# Bumped whenever the relational schema changes in a way that affects chunk/embedding
# semantics. Part of the index-version tuple (model, chunk_strategy, chunk_params, schema).
SCHEMA_VERSION = 1
