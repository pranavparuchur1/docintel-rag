"""Chunking pipeline: documents table -> parsed text -> chunks table.

Idempotent per (document, strategy, params_hash): already-chunked combinations
are skipped unless --force, which replaces them atomically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psycopg

from docintel.chunk.base import ChunkStrategy
from docintel.config import Settings
from docintel.parse import load_raw_text, parse_filing

logger = logging.getLogger(__name__)


@dataclass
class ChunkStats:
    documents: int = 0
    chunked: int = 0
    skipped: int = 0
    chunks_written: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)


def chunk_documents(
    conn: psycopg.Connection,
    settings: Settings,
    strategies: list[ChunkStrategy],
    force: bool = False,
) -> ChunkStats:
    stats = ChunkStats()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT accession_no, company, form_type, raw_path FROM documents ORDER BY accession_no"
        )
        documents = cur.fetchall()

    for accession_no, company, form_type, raw_path in documents:
        stats.documents += 1
        parsed = None  # parse lazily: skip the expensive parse if nothing to do
        for strategy in strategies:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM chunks WHERE document_id=%s AND strategy=%s "
                    "AND params_hash=%s",
                    (accession_no, strategy.name, strategy.params_hash),
                )
                existing = cur.fetchone()[0]
            if existing and not force:
                stats.skipped += 1
                logger.info(
                    "skip %s %s/%s (%d chunks exist)",
                    accession_no, strategy.name, strategy.params_hash, existing,
                )
                continue

            if parsed is None:
                parsed = parse_filing(load_raw_text(settings.data_dir / raw_path), form_type)
            chunks = strategy.split(parsed)

            with conn.cursor() as cur:
                if existing:
                    cur.execute(
                        "DELETE FROM chunks WHERE document_id=%s AND strategy=%s "
                        "AND params_hash=%s",
                        (accession_no, strategy.name, strategy.params_hash),
                    )
                cur.executemany(
                    """
                    INSERT INTO chunks (document_id, strategy, params_hash, section,
                                        ordinal, text, token_count, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            accession_no, strategy.name, strategy.params_hash, c.section,
                            c.ordinal, c.text, c.token_count, c.content_hash,
                        )
                        for c in chunks
                    ],
                )
            conn.commit()
            stats.chunked += 1
            stats.chunks_written += len(chunks)
            stats.by_strategy[strategy.name] = stats.by_strategy.get(strategy.name, 0) + len(chunks)
            logger.info(
                "chunked %s %s %s -> %d chunks (%s)",
                company, form_type, accession_no, len(chunks), strategy.name,
            )
    return stats
