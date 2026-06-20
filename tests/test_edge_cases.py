"""Edge cases: malformed inputs, single-chain monomers, missing data, empty folders.

These guard the project's core promise: missing metrics surface as N/A (never invented)
and a single bad file must not crash report generation.
"""

from __future__ import annotations

import pytest

from foldreport import figures
from foldreport.metrics import confidence_score
from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import detect_parser, parse_folder
from foldreport.parsers.base import read_structure
from foldreport.report import build_report


# --- Empty / unrecognized folders -----------------------------------------------------


def test_detect_parser_returns_none_for_empty_folder(empty_dir):
    assert detect_parser(empty_dir) is None


def test_parse_folder_raises_for_unrecognized_folder(empty_dir):
    with pytest.raises(ValueError):
        parse_folder(empty_dir)


# --- Malformed files ------------------------------------------------------------------


def test_corrupt_sidecar_json_falls_back_cleanly(malformed_dir):
    """An invalid scores JSON must not crash: pLDDT falls back to B-factors,
    and the JSON-only metrics stay None."""
    parser = detect_parser(malformed_dir)
    assert parser is not None and parser.name == "colabfold"

    preds = parse_folder(malformed_dir)
    assert len(preds) == 1
    pred = preds[0]
    # B-factor pLDDT recovered from the (valid) structure.
    assert len(pred.plddt) == pred.metrics.n_residues > 0
    assert pred.metrics.mean_plddt is not None
    # Metrics that only exist in the corrupt JSON are absent, not fabricated.
    assert pred.metrics.ptm is None
    assert pred.metrics.iptm is None
    assert pred.pae is None


def test_corrupt_structure_file_degrades_cleanly(tmp_path):
    """A garbage PDB yields an empty structure (no usable chains/pLDDT), never a crash."""
    from foldreport.parsers.base import chains_from_structure, plddt_from_bfactors

    bad = tmp_path / "garbage.pdb"
    bad.write_text("this is not a structure file\n" * 5, encoding="utf-8")
    structure = read_structure(bad)  # gemmi is lenient: no exception
    assert chains_from_structure(structure) == []
    assert plddt_from_bfactors(bad) == []


# --- Single-chain monomer -------------------------------------------------------------


def test_single_chain_prediction(single_chain_dir):
    preds = parse_folder(single_chain_dir)
    assert len(preds) == 1
    pred = preds[0]
    assert pred.metrics.n_chains == 1
    # A monomer has no interface, so ipTM is absent (never invented).
    assert pred.metrics.iptm is None
    assert pred.metrics.ptm == 0.88
    # confidence_score still works, falling back from ipTM to pTM.
    assert confidence_score(pred) == pred.metrics.ptm


def test_single_chain_report_renders(tmp_path, single_chain_dir):
    preds = parse_folder(single_chain_dir)
    out = tmp_path / "monomer.html"
    build_report(preds, out)
    text = out.read_text(encoding="utf-8")
    assert preds[0].name in text
    # The absent ipTM is rendered as N/A somewhere in the document.
    assert "N/A" in text


# --- Absent pLDDT / PAE ---------------------------------------------------------------


def test_figures_none_when_plddt_and_pae_absent():
    pred = Prediction(
        name="bare",
        source_tool="test",
        structure_path=__file__,
        plddt=[],
        pae=None,
        metrics=PredictionMetrics(n_chains=1, n_residues=3),
    )
    figs = figures.make_figures(pred)
    assert figs["plddt"] is None
    assert figs["pae"] is None
    assert figs["pae_interactive"] is None


def test_report_renders_not_available_for_missing_figures(tmp_path, single_chain_dir):
    """A prediction with a valid structure but no pLDDT/PAE renders textual fallbacks."""
    pred = parse_folder(single_chain_dir)[0]
    pred.plddt = []
    pred.pae = None
    pred.metrics.mean_plddt = None
    out = tmp_path / "missing.html"
    build_report([pred], out)
    text = out.read_text(encoding="utf-8")
    assert "pLDDT not available" in text
    assert "PAE not available" in text
