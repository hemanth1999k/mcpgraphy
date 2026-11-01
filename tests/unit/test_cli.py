import pytest

from mcpgraphy import __version__
from mcpgraphy.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "mcpgraphy" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "COMMAND" in capsys.readouterr().out


def test_version_prints_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
