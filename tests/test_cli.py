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


def test_unimplemented_commands_exit_nonzero():
    result = runner.invoke(app, ["embed"])
    assert result.exit_code == 2
