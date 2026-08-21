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
    max_tokens: int = typer.Option(None, help="Override the token budget (new params_hash)."),
) -> None:
    """Parse raw filings and write chunks for the selected strategies."""
    from docintel import db
    from docintel.chunk.service import chunk_documents
    from docintel.chunk.strategies import get_strategies

    settings = _settings()
    strategies = get_strategies(strategy.split(","), max_tokens=max_tokens)
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
def embed(
    strategy: str = typer.Option("all", help="fixed | recursive | section_aware | all"),
    batch_size: int = typer.Option(64, help="Texts per embedding batch."),
    max_tokens: int = typer.Option(None, help="Target the chunk set built with this budget."),
) -> None:
    """Embed chunks whose content_hash is missing from the target index version."""
    from docintel import db
    from docintel.chunk.strategies import get_strategies
    from docintel.embed.providers import get_provider
    from docintel.embed.service import embed_pending, get_or_create_index_version

    settings = _settings()
    provider = get_provider(settings)
    with db.connect(settings) as conn:
        for strat in get_strategies(strategy.split(","), max_tokens=max_tokens):
            version = get_or_create_index_version(conn, provider, strat)
            stats = embed_pending(conn, provider, version, batch_size=batch_size)
            typer.echo(
                f"index_version={version.index_version_id} strategy={strat.name} "
                f"params={strat.params_hash} embedded={stats.embedded} "
                f"already_present={stats.already_present} total={stats.total_hashes}"
            )


@app.command("index-versions")
def index_versions() -> None:
    """List index versions with vector counts and status."""
    from docintel import db
    from docintel.embed.service import list_index_versions

    settings = _settings()
    with db.connect(settings) as conn:
        versions = list_index_versions(conn)
    if not versions:
        typer.echo("no index versions yet — run `docintel embed`")
        return
    for v in versions:
        typer.echo(
            f"v{v['index_version_id']:<3} {v['status']:<8} {v['strategy']:<14} "
            f"model={v['embedding_model']} params={v['chunk_params']} "
            f"schema={v['schema_version']} vectors={v['vectors']}/{v['chunk_count']} "
            f"created={v['created_at']:%Y-%m-%d %H:%M}"
        )


@app.command()
def query(
    question: str = typer.Argument(...),
    index_version: int = typer.Option(..., "--index-version", help="Index version to query."),
    mode: str = typer.Option("hybrid", help="vector | hybrid"),
    k: int = typer.Option(5),
) -> None:
    """Ask a question against the corpus (retrieval-only until Phase 5)."""
    from docintel import db
    from docintel.embed.providers import get_provider
    from docintel.embed.service import get_index_version
    from docintel.retrieve.service import Retriever

    settings = _settings()
    with db.connect(settings) as conn:
        version = get_index_version(conn, index_version)
        result = Retriever(conn, get_provider(settings), version).search(question, k=k, mode=mode)
    if result.top_vector_score < settings.refusal_threshold:
        typer.secho(
            f"REFUSAL: top similarity {result.top_vector_score:.3f} < "
            f"threshold {settings.refusal_threshold} — the corpus does not appear to cover this.",
            fg=typer.colors.YELLOW,
        )
    for c in result.chunks:
        header = f"[{c.rank}] {c.company} {c.form_type} {c.document_id} chunk={c.chunk_id}"
        typer.secho(f"{header}  score={c.score:.4f}", fg=typer.colors.CYAN)
        typer.echo(f"    {c.section or '(no section)'}")
        typer.echo("    " + " ".join(c.text.split())[:220])
    typer.echo(f"latency={result.latency_ms:.0f}ms top_vector={result.top_vector_score:.3f}")


@app.command("eval")
def eval_cmd(
    index_version: list[int] = typer.Option(
        None, "--index-version", help="Version(s) to evaluate; repeatable. Default: all ready."
    ),
    mode: str = typer.Option("both", help="vector | hybrid | both"),
    golden_path: str = typer.Option("eval/golden.yaml", help="Golden question set."),
    name: str = typer.Option(None, help="Report filename (under docs/eval/)."),
) -> None:
    """Run the golden-set evaluation and write a markdown report to docs/eval/."""
    from pathlib import Path

    from docintel import db
    from docintel.embed.providers import get_provider
    from docintel.embed.service import get_index_version, list_index_versions
    from docintel.eval.golden import load_golden
    from docintel.eval.runner import run_eval, write_report

    settings = _settings()
    golden = load_golden(Path(golden_path))
    modes = ["vector", "hybrid"] if mode == "both" else [mode]
    provider = get_provider(settings)
    runs = []
    with db.connect(settings) as conn:
        if index_version:
            versions = [get_index_version(conn, v) for v in index_version]
        else:
            versions = [
                get_index_version(conn, row["index_version_id"])
                for row in list_index_versions(conn)
                if row["status"] == "ready"
            ]
        for version in versions:
            for m in modes:
                typer.echo(f"evaluating v{version.index_version_id} {version.strategy} / {m} ...")
                runs.append(
                    run_eval(conn, provider, version, golden, m, settings.refusal_threshold)
                )
        report = write_report(runs, golden, Path("docs/eval"), name=name)
    for run in runs:
        m = run.metrics()
        typer.echo(
            f"{run.label:<32} recall@5={m['recall@5']:.2f} mrr@10={m['mrr@10']:.2f} "
            f"ndcg@10={m['ndcg@10']:.2f} refusalP={m['refusal_precision']:.2f} "
            f"refusalR={m['refusal_recall']:.2f} p95={m['p95_ms']:.0f}ms"
        )
    typer.echo(f"report: {report}")


@app.command("check-golden")
def check_golden(
    index_version: int = typer.Option(..., "--index-version"),
    golden_path: str = typer.Option("eval/golden.yaml"),
) -> None:
    """Mechanically verify every in-corpus golden question has ≥1 ground-truth
    chunk in the given version's chunk set (content correctness still needs a
    human — this catches impossible questions and typos in specs)."""
    from pathlib import Path

    from docintel import db
    from docintel.embed.service import get_index_version
    from docintel.eval.golden import load_golden, relevant_universe_size

    settings = _settings()
    golden = load_golden(Path(golden_path))
    bad = 0
    with db.connect(settings) as conn:
        version = get_index_version(conn, index_version)
        for q in golden:
            if q.refuse:
                continue
            # every spec must be individually satisfiable, so a multi-spec
            # question can't hide one impossible spec behind a matching one
            dead = [
                spec
                for spec in q.expected
                if relevant_universe_size(conn, version, [spec]) == 0
            ]
            if dead:
                bad += 1
                typer.secho(
                    f"{q.id}: {len(dead)} spec(s) match 0 chunks — {q.question}",
                    fg=typer.colors.RED,
                )
                for spec in dead:
                    typer.echo(f"    {spec}")
    verified = sum(1 for q in golden if q.verified)
    in_corpus = sum(1 for q in golden if not q.refuse)
    typer.echo(
        f"{in_corpus - bad}/{in_corpus} in-corpus questions have ground-truth chunks; "
        f"{verified}/{len(golden)} human-verified"
    )
    if bad:
        raise typer.Exit(code=1)


def main() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app()


if __name__ == "__main__":
    main()
