"""docintel CLI — the real interface to the pipeline.

Every pipeline stage is a CLI command; the Airflow DAG (later phase) is a thin
wrapper around these same commands, so orchestration never hides behavior.
"""

from __future__ import annotations

import logging
import sys

import typer

from docintel import __version__
from docintel.config import ConfigError, get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
db_app = typer.Typer(no_args_is_help=True)
app.add_typer(db_app, name="db", help="Database utilities (migrations, health).")


@app.callback()
def _configure(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _settings():
    try:
        return get_settings()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def version() -> None:
    """Print the docintel version."""
    typer.echo(f"docintel {__version__} (python {sys.version.split()[0]})")


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply pending SQL migrations from sql/."""
    from docintel import db

    settings = _settings()
    with db.connect(settings) as conn:
        applied = db.apply_migrations(conn)
    if applied:
        for name in applied:
            typer.echo(f"applied  {name}")
    else:
        typer.echo("schema up to date — nothing to apply")


@db_app.command("check")
def db_check() -> None:
    """Verify database connectivity and pgvector availability."""
    from docintel import db

    settings = _settings()
    with db.connect(settings) as conn:
        info = db.health(conn)
    for key, value in info.items():
        typer.echo(f"{key:<10} {value}")
    if info["pgvector"] == "NOT INSTALLED":
        typer.secho("pgvector extension missing — run `docintel db upgrade`", fg="red", err=True)
        raise typer.Exit(code=1)


def _not_yet(phase: str) -> None:
    typer.secho(f"Not implemented yet — arrives in {phase}.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


@app.command()
def ingest(
    cik: str = typer.Option(..., help="Company CIK (zero-padding optional)."),
    forms: str = typer.Option("10-K,10-Q", help="Comma-separated form types."),
    limit: int = typer.Option(3, help="Max filings per form type."),
) -> None:
    """Download filings from SEC EDGAR into the raw store + documents table."""
    from docintel import db
    from docintel.ingest.edgar_client import EdgarClient
    from docintel.ingest.service import ingest_documents

    settings = _settings()
    with (
        db.connect(settings) as conn,
        EdgarClient(
            settings.edgar_user_agent,
            max_requests_per_sec=settings.edgar_max_requests_per_sec,
        ) as client,
    ):
        stats = ingest_documents(
            conn, client, settings, cik=cik, forms=forms.split(","), limit=limit
        )
    typer.echo(
        f"selected={stats.selected} downloaded={stats.downloaded} "
        f"skipped={stats.skipped} refreshed={stats.refreshed} by_form={stats.by_form}"
    )


@app.command()
def chunk(
    strategy: str = typer.Option("all", help="fixed | recursive | section_aware | all"),
    force: bool = typer.Option(False, help="Re-chunk even if chunks already exist."),
) -> None:
    """Parse raw filings and write chunks for the selected strategies."""
    from docintel import db
    from docintel.chunk.service import chunk_documents
    from docintel.chunk.strategies import get_strategies

    settings = _settings()
    strategies = get_strategies(strategy.split(","))
    with db.connect(settings) as conn:
        stats = chunk_documents(conn, settings, strategies, force=force)
    typer.echo(
        f"documents={stats.documents} chunked={stats.chunked} skipped={stats.skipped} "
        f"chunks_written={stats.chunks_written} by_strategy={stats.by_strategy}"
    )


@app.command("parse-report")
def parse_report() -> None:
    """Measure section-detection hit rate across the ingested corpus."""
    from docintel import db
    from docintel.parse import load_raw_text, parse_filing
    from docintel.parse.sections import EXPECTED_ITEMS, section_hit_rate

    settings = _settings()
    with db.connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT accession_no, company, form_type, raw_path FROM documents "
            "WHERE form_type = ANY(%s) ORDER BY company, filing_date",
            (list(EXPECTED_ITEMS),),
        )
        rows = cur.fetchall()

    total_expected = total_found = 0
    for accession_no, company, form_type, raw_path in rows:
        parsed = parse_filing(load_raw_text(settings.data_dir / raw_path), form_type)
        expected, found = section_hit_rate(parsed.sections, form_type)
        total_expected += len(expected)
        total_found += len(found)
        missing = "".join(f" MISSING:{k}" for k in sorted(expected - found))
        typer.echo(
            f"{company[:24]:<24} {form_type:<5} {accession_no}  "
            f"{len(found)}/{len(expected)} key sections "
            f"({len(parsed.sections)} total detected){missing}"
        )
    if total_expected:
        rate = 100.0 * total_found / total_expected
        typer.echo(f"\nsection-detection hit rate: {total_found}/{total_expected} = {rate:.1f}%")


@app.command()
def embed() -> None:
    """Embed new/changed chunks into the target index version."""
    _not_yet("Phase 3 (embedding & versioned index)")


@app.command()
def query(question: str = typer.Argument(...)) -> None:
    """Ask a question against the corpus."""
    _not_yet("Phase 4 (retrieval)")


@app.command("eval")
def eval_cmd(index_version: int = typer.Option(None, "--index-version")) -> None:
    """Run the golden-set evaluation and write a markdown report."""
    _not_yet("Phase 4 (evaluation harness)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
