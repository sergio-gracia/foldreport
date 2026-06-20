"""End-to-end CLI test: one command produces the report."""

from __future__ import annotations

from click.testing import CliRunner

from foldreport.cli import main


def test_cli_generates_report(tmp_path, colabfold_dir):
    out = tmp_path / "cli.html"
    runner = CliRunner()
    result = runner.invoke(main, [str(colabfold_dir), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "prediction(s) via 'colabfold'" in result.output


def test_cli_writes_csv(tmp_path, colabfold_dir):
    out = tmp_path / "cli.html"
    csv = tmp_path / "metrics.csv"
    runner = CliRunner()
    result = runner.invoke(
        main, [str(colabfold_dir), "-o", str(out), "--csv", str(csv)]
    )
    assert result.exit_code == 0, result.output
    assert csv.exists()
    assert "overall_rank" in csv.read_text(encoding="utf-8")


def test_cli_errors_on_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    runner = CliRunner()
    result = runner.invoke(main, [str(empty)])
    assert result.exit_code == 1
