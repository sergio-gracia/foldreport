"""Phase 3 acceptance: a single self-contained HTML report is produced."""

from __future__ import annotations

from foldreport.parsers import parse_folder
from foldreport.report import build_report


def test_build_report_single_self_contained_file(tmp_path, colabfold_dir):
    preds = parse_folder(colabfold_dir)
    out = tmp_path / "report.html"
    result = build_report(preds, out, title="Test Report")

    assert result == out
    assert out.exists()
    # Exactly one file is produced (no adjacent assets).
    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]

    text = out.read_text(encoding="utf-8")
    # Self-contained: 3Dmol.js and figures are inlined, nothing is fetched remotely.
    assert "$3Dmol" in text
    assert "data:image/png;base64," in text
    # No external resources are referenced (URLs inside the inlined library's own
    # source comments are fine; what matters is that nothing is loaded over the network).
    assert 'src="http' not in text
    assert 'href="http' not in text
    # Ranking + a row per prediction are present.
    assert "Ranking by confidence" in text
    for p in preds:
        assert p.name in text


def test_report_handles_multiple_tools(tmp_path, colabfold_dir, af3_dir, boltz_dir):
    pooled = []
    for d in (colabfold_dir, af3_dir, boltz_dir):
        pooled.extend(parse_folder(d))
    out = tmp_path / "multi.html"
    build_report(pooled, out)
    text = out.read_text(encoding="utf-8")
    for tool in ("colabfold", "af3_server", "boltz"):
        assert tool in text
