"""Ingestion: EDGAR submissions index -> raw files on disk -> documents table.

Idempotency contract: re-running the same command must not duplicate rows or
re-download unchanged filings. The accession number is the unit of identity;
a filing is skipped when its row exists AND its raw file is present on disk.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import PurePosixPath
from typing import Protocol

import psycopg

from docintel.config import Settings
from docintel.ingest.models import Filing

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"


class EdgarLike(Protocol):
    """What ingestion needs from a client — lets tests substitute a fake."""

    def get_json(self, url: str) -> dict: ...
    def download(self, url: str, dest) -> tuple[str, int]: ...


@dataclass
class IngestStats:
    selected: int = 0
    downloaded: int = 0
    skipped: int = 0
    refreshed: int = 0  # row existed but raw file was missing -> re-downloaded
    by_form: dict[str, int] = field(default_factory=dict)


def normalize_cik(cik: str) -> str:
    try:
        return str(int(cik.strip()))
    except ValueError as exc:
        raise ValueError(f"CIK must be numeric, got {cik!r}") from exc


def fetch_submissions(client: EdgarLike, cik: str) -> dict:
    return client.get_json(SUBMISSIONS_URL.format(cik=normalize_cik(cik)))


def select_filings(submissions: dict, forms: list[str], limit: int) -> list[Filing]:
    """Pick the `limit` most recent filings per requested form type.

    Pure function over the submissions JSON so it is unit-testable without
    network access. Only the 'recent' window is considered, which covers years
    of filings — far more than this project's corpus needs.
    """
    company = submissions["name"]
    cik = str(int(submissions["cik"]))
    recent = submissions["filings"]["recent"]
    wanted = {f.strip().upper() for f in forms}

    candidates: list[Filing] = []
    rows = zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["reportDate"],
        recent["primaryDocument"],
        strict=True,
    )
    for accession_no, form, filed, report, primary in rows:
        if form.upper() not in wanted or not primary:
            continue
        candidates.append(
            Filing(
                accession_no=accession_no,
                cik=cik,
                company=company,
                form_type=form.upper(),
                filing_date=date.fromisoformat(filed),
                period=date.fromisoformat(report) if report else None,
                primary_document=primary,
            )
        )

    candidates.sort(key=lambda f: f.filing_date, reverse=True)
    picked: dict[str, list[Filing]] = defaultdict(list)
    for filing in candidates:
        if len(picked[filing.form_type]) < limit:
            picked[filing.form_type].append(filing)
    return [f for filings in picked.values() for f in filings]


def ingest_documents(
    conn: psycopg.Connection,
    client: EdgarLike,
    settings: Settings,
    cik: str,
    forms: list[str],
    limit: int,
) -> IngestStats:
    submissions = fetch_submissions(client, cik)
    filings = select_filings(submissions, forms, limit)
    stats = IngestStats(selected=len(filings))

    for filing in filings:
        stats.by_form[filing.form_type] = stats.by_form.get(filing.form_type, 0) + 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT raw_path FROM documents WHERE accession_no = %s",
                (filing.accession_no,),
            )
            row = cur.fetchone()

        # raw_path is stored relative to DATA_DIR so the same row is valid on
        # the host (./data) and in the container (/app/data).
        rel_path = PurePosixPath("raw") / filing.accession_no / filing.primary_document
        if row is not None:
            existing = settings.data_dir / row[0]
            if existing.exists():
                stats.skipped += 1
                logger.info(
                    "skip %s %s %s (already ingested, raw file present)",
                    filing.company, filing.form_type, filing.accession_no,
                )
                continue
            logger.warning(
                "row exists but raw file missing for %s — re-downloading", filing.accession_no
            )
            stats.refreshed += 1

        sha256, size = client.download(filing.source_url, settings.data_dir / rel_path)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (accession_no, cik, company, form_type, filing_date, period,
                     source_url, raw_path, content_hash, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_no) DO UPDATE SET
                    raw_path = EXCLUDED.raw_path,
                    content_hash = EXCLUDED.content_hash,
                    size_bytes = EXCLUDED.size_bytes,
                    ingested_at = now()
                """,
                (
                    filing.accession_no, filing.cik, filing.company, filing.form_type,
                    filing.filing_date, filing.period, filing.source_url,
                    str(rel_path), sha256, size,
                ),
            )
        conn.commit()
        stats.downloaded += 1
        logger.info(
            "ingested %s %s %s (%d bytes, sha256 %s...)",
            filing.company, filing.form_type, filing.accession_no, size, sha256[:12],
        )

    logger.info(
        "ingest complete cik=%s: selected=%d downloaded=%d skipped=%d refreshed=%d",
        normalize_cik(cik), stats.selected, stats.downloaded, stats.skipped, stats.refreshed,
    )
    return stats
