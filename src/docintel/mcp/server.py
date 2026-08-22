"""MCP server: the corpus as tools for any MCP client (Claude Desktop, Claude
Code, or anything speaking the protocol). Run with `docintel mcp-serve`
(stdio transport).

Tools return compact JSON-friendly structures; full chunk text only via
get_chunk, so a client can search cheaply and fetch precisely.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from docintel import db
from docintel.agent.routing import COMPANY_ALIASES
from docintel.config import get_settings
from docintel.embed.providers import get_provider
from docintel.embed.service import get_latest_ready_version
from docintel.retrieve.service import Retriever

mcp = MCPServer(
    "docintel",
    instructions=(
        "Search and cite SEC 10-K/10-Q filings for Apple, Microsoft, Nvidia, "
        "Tesla, Coca-Cola, and JPMorgan. Retrieval quality is measured; see "
        "get_eval_report for the current numbers."
    ),
)

_provider = None


def _get_provider():
    global _provider
    if _provider is None:
        _provider = get_provider(get_settings())
    return _provider


def _canonical_company(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.lower().strip()
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in lowered or lowered == canonical.lower():
            return canonical
    return name  # pass through; filter simply matches nothing if unknown


@mcp.tool()
def search_filings(
    query: str, company: str | None = None, form_type: str | None = None, k: int = 5
) -> list[dict]:
    """Hybrid (vector + full-text) search over SEC 10-K/10-Q filings.
    Optional filters: company (e.g. 'Nvidia'), form_type ('10-K' | '10-Q').
    Returns scored chunks with citations; use get_chunk for full text."""
    k = max(1, min(k, 20))
    settings = get_settings()
    with db.connect(settings) as conn:
        version = get_latest_ready_version(conn)
        retriever = Retriever(conn, _get_provider(), version)
        canonical = _canonical_company(company)
        result = retriever.search(
            query, k=k, mode="hybrid", companies=[canonical] if canonical else None
        )
        chunks = result.chunks
        if form_type:
            chunks = [c for c in chunks if c.form_type == form_type.upper()]
    return [
        {
            "chunk_id": c.chunk_id, "company": c.company, "form_type": c.form_type,
            "accession": c.document_id, "section": c.section,
            "score": round(c.score, 4),
            "excerpt": " ".join(c.text.split())[:300],
        }
        for c in chunks
    ]


@mcp.tool()
def get_chunk(chunk_id: int) -> dict:
    """Fetch one chunk's full text and provenance by chunk_id."""
    settings = get_settings()
    with db.connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.chunk_id, d.company, d.form_type, c.document_id, d.filing_date,
                   c.section, c.text, d.source_url
            FROM chunks c JOIN documents d ON d.accession_no = c.document_id
            WHERE c.chunk_id = %s
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
    if row is None:
        return {"error": f"chunk {chunk_id} not found"}
    return {
        "chunk_id": row[0], "company": row[1], "form_type": row[2], "accession": row[3],
        "filing_date": str(row[4]), "section": row[5], "text": row[6], "source_url": row[7],
    }


@mcp.tool()
def list_companies() -> list[dict]:
    """List companies in the corpus with their filings."""
    settings = get_settings()
    with db.connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT company, cik,
                   array_agg(form_type || ' ' || filing_date ORDER BY filing_date DESC)
            FROM documents GROUP BY company, cik ORDER BY company
            """
        )
        rows = cur.fetchall()
    return [{"company": r[0], "cik": r[1], "filings": r[2]} for r in rows]


@mcp.tool()
def get_eval_report() -> str:
    """Return the most recent retrieval-quality evaluation report (markdown):
    recall@k, MRR, nDCG, refusal precision/recall per configuration."""
    reports = sorted(
        Path("docs/eval").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not reports:
        return "no evaluation reports found — run `docintel eval`"
    return reports[0].read_text(encoding="utf-8")[:50_000]


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
