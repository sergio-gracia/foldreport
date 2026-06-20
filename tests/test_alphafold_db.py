"""AlphaFold DB parser: real-world single-model entries through the same abstraction.

These run fully offline against synthetic fixtures shaped like real AlphaFold DB
downloads; ``examples/fetch_afdb.py`` exercises the same parser on live data.
"""

from __future__ import annotations

import numpy as np

from foldreport.metrics import confidence_score
from foldreport.parsers import detect_parser, parse_folder


def test_alphafold_db_autodetect(alphafold_db_dir):
    parser = detect_parser(alphafold_db_dir)
    assert parser is not None and parser.name == "alphafold_db"


def test_alphafold_db_parse(alphafold_db_dir):
    preds = parse_folder(alphafold_db_dir)
    assert len(preds) == 2
    for pred in preds:
        assert pred.source_tool == "alphafold_db"
        # Single published model per entry: one chain, no interface metrics.
        assert pred.metrics.n_chains == 1
        assert pred.metrics.ptm is None
        assert pred.metrics.iptm is None
        # pLDDT comes from the structure B-factors; PAE from the DB JSON.
        assert len(pred.plddt) == pred.metrics.n_residues > 0
        assert pred.metrics.mean_plddt is not None
        assert isinstance(pred.pae, np.ndarray)
        assert pred.pae.shape == (pred.metrics.n_residues, pred.metrics.n_residues)
        # The gene from metadata enriches the display name.
        assert "(" in pred.name and ")" in pred.name
        # Confidence falls back to mean pLDDT when pTM/ipTM are absent.
        assert confidence_score(pred) == pred.metrics.mean_plddt / 100.0


def test_alphafold_db_report_renders(tmp_path, alphafold_db_dir):
    from foldreport.report import build_report

    preds = parse_folder(alphafold_db_dir)
    out = tmp_path / "afdb.html"
    build_report(preds, out)
    text = out.read_text(encoding="utf-8")
    assert "alphafold_db" in text
    for pred in preds:
        assert pred.name in text
