"""Embedding service: content-addressed, incremental, versioned.

The unit of embedding is the chunk's content_hash, not its chunk_id. Vectors
live in embeddings_v<N>(content_hash PK, embedding vector(dim)); chunks join on
content_hash at retrieval time. Consequences:

- re-running embed re-embeds nothing (hash already present)
- re-chunking with identical params produces new chunk_ids but identical
  hashes, so it costs zero embeddings
- boilerplate repeated across filings is embedded exactly once
- a chunk-param or model change is a NEW index version building in its own
  table while the old version keeps serving
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import psycopg

from docintel import SCHEMA_VERSION
from docintel.chunk.base import ChunkStrategy
from docintel.embed.providers import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexVersion:
    index_version_id: int
    embedding_model: str
    embedding_dim: int
    strategy: str
    params_hash: str
    schema_version: int
    status: str

    @property
    def table_name(self) -> str:
        return f"embeddings_v{self.index_version_id}"


@dataclass
class EmbedStats:
    index_version_id: int = 0
    total_hashes: int = 0
    embedded: int = 0
    already_present: int = 0


def _row_to_version(row) -> IndexVersion:
    return IndexVersion(
        index_version_id=row[0], embedding_model=row[1], embedding_dim=row[2],
        strategy=row[3], params_hash=row[4], schema_version=row[5], status=row[6],
    )


_VERSION_COLS = (
    "index_version_id, embedding_model, embedding_dim, strategy, params_hash, "
    "schema_version, status"
)


def get_index_version(conn: psycopg.Connection, index_version_id: int) -> IndexVersion:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_VERSION_COLS} FROM index_versions WHERE index_version_id = %s",
            (index_version_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"index version {index_version_id} does not exist")
    return _row_to_version(row)


def get_or_create_index_version(
    conn: psycopg.Connection, provider: EmbeddingProvider, strategy: ChunkStrategy,
    params_hash: str | None = None,
) -> IndexVersion:
    """Resolve the index version for (model, strategy, params, schema); create
    it — and its dimension-typed embeddings table — if it doesn't exist."""
    params_hash = params_hash or strategy.params_hash
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_VERSION_COLS} FROM index_versions WHERE embedding_model=%s "
            "AND strategy=%s AND params_hash=%s AND schema_version=%s",
            (provider.model_name, strategy.name, params_hash, SCHEMA_VERSION),
        )
        row = cur.fetchone()
        if row is None:
            dim = provider.dim  # loads the model on first use
            cur.execute(
                """
                INSERT INTO index_versions
                    (embedding_model, embedding_dim, strategy, params_hash,
                     chunk_params, schema_version)
                SELECT %s, %s, strategy, params_hash, params, %s
                FROM chunk_param_sets WHERE strategy=%s AND params_hash=%s
                RETURNING """ + _VERSION_COLS,
                (provider.model_name, dim, SCHEMA_VERSION, strategy.name, params_hash),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"no chunks exist for {strategy.name}/{params_hash} — run `docintel chunk`"
                )
    version = _row_to_version(row)
    _ensure_version_table(conn, version)
    conn.commit()
    return version


def _ensure_version_table(conn: psycopg.Connection, version: IndexVersion) -> None:
    # Table name is derived from an integer id — safe to interpolate.
    table = version.table_name
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                content_hash text PRIMARY KEY,
                embedding    vector({version.embedding_dim}) NOT NULL,
                created_at   timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # HNSW over IVFFlat: no training step, so it stays correct under
        # incremental inserts, and recall/latency is better at this corpus size.
        # See docs/decisions/0002.
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def embed_pending(
    conn: psycopg.Connection,
    provider: EmbeddingProvider,
    version: IndexVersion,
    batch_size: int = 64,
) -> EmbedStats:
    """Embed every chunk content_hash of this version's chunk set that is not
    yet in the version's table. Second run embeds zero by construction."""
    stats = EmbedStats(index_version_id=version.index_version_id)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT content_hash) FROM chunks WHERE strategy=%s AND params_hash=%s",
            (version.strategy, version.params_hash),
        )
        stats.total_hashes = cur.fetchone()[0]
        cur.execute(
            f"""
            SELECT DISTINCT ON (c.content_hash) c.content_hash, c.text
            FROM chunks c
            WHERE c.strategy = %s AND c.params_hash = %s
              AND NOT EXISTS (SELECT 1 FROM {version.table_name} e
                              WHERE e.content_hash = c.content_hash)
            ORDER BY c.content_hash
            """,
            (version.strategy, version.params_hash),
        )
        pending = cur.fetchall()

    stats.already_present = stats.total_hashes - len(pending)
    started = time.monotonic()
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors = provider.embed_passages([text for _h, text in batch])
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {version.table_name} (content_hash, embedding) "
                "VALUES (%s, %s::vector) ON CONFLICT (content_hash) DO NOTHING",
                [(h, _vector_literal(v)) for (h, _t), v in zip(batch, vectors, strict=True)],
            )
        conn.commit()
        stats.embedded += len(batch)
        rate = stats.embedded / max(time.monotonic() - started, 1e-9)
        logger.info(
            "index_version=%d embedded %d/%d (%.1f chunks/s)",
            version.index_version_id, stats.embedded, len(pending), rate,
        )

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE index_versions SET status='ready', chunk_count=%s WHERE index_version_id=%s",
            (stats.total_hashes, version.index_version_id),
        )
    conn.commit()
    return stats


def list_index_versions(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT index_version_id, embedding_model, strategy, params_hash, chunk_params, "
            "schema_version, status, chunk_count, created_at FROM index_versions "
            "ORDER BY index_version_id"
        )
        rows = cur.fetchall()
        result = []
        for row in rows:
            cur.execute(f"SELECT count(*) FROM embeddings_v{row[0]}")
            result.append(
                {
                    "index_version_id": row[0], "embedding_model": row[1], "strategy": row[2],
                    "params_hash": row[3], "chunk_params": row[4], "schema_version": row[5],
                    "status": row[6], "chunk_count": row[7], "created_at": row[8],
                    "vectors": cur.fetchone()[0],
                }
            )
    return result
