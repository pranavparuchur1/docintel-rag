from typer.testing import CliRunner

from docintel import __version__
from docintel.cli import app

runner = CliRunner()


def test_help_lists_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "embed", "query", "eval", "db"):
        assert command in result.output


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_query_requires_index_version():
    result = runner.invoke(app, ["query", "anything"])
    assert result.exit_code == 2  # typer: missing required --index-version
