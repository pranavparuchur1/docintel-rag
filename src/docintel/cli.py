"""docintel CLI — the real interface to the pipeline.

Every pipeline stage is a CLI command; the Airflow DAG (later phase) is a thin
wrapper around these same commands, so orchestration never hides behavior.
"""

from __future__ import annotations

import sys

import typer

from docintel import __version__
from docintel.config import ConfigError, get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
db_app = typer.Typer(no_args_is_help=True)
app.add_typer(db_app, name="db", help="Database utilities (migrations, health).")


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
    _not_yet("Phase 1 (ingestion)")


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
