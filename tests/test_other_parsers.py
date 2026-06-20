"""Phase 4 acceptance: AF3 Server and Boltz parse through the same abstraction."""

from __future__ import annotations

import numpy as np

from foldreport.parsers import detect_parser, parse_folder
from foldreport.parsers.openfold3 import _CONF_RE, _SUMMARY_RE


def test_openfold3_confidence_patterns_do_not_overlap():
    """The summary file must not also match the per-token confidences pattern.

    Both files end in ``_confidences.json``; if the patterns overlap, directory
    iteration order decides which file is read as the confidences (PAE) source,
    which made the parser drop PAE on filesystems with arbitrary readdir order.
    """
    summary = "mycomplex_summary_confidences.json"
    conf = "mycomplex_confidences.json"
    assert _SUMMARY_RE.match(summary) and not _SUMMARY_RE.match(conf)
    assert _CONF_RE.match(conf) and not _CONF_RE.match(summary)


def test_af3_autodetect_and_parse(af3_dir):
    parser = detect_parser(af3_dir)
    assert parser is not None and parser.name == "af3_server"

    preds = parse_folder(af3_dir)
    assert len(preds) == 2
    top = min(preds, key=lambda p: p.rank)
    assert top.metrics.n_residues == 56
    assert top.metrics.ptm == 0.85
    assert top.metrics.iptm == 0.78
    assert top.metrics.ranking_score == 0.81
    assert top.pae.shape == (56, 56)
    # AF3 stores pLDDT in the B-factor column (0-100 scale).
    assert 0 <= top.metrics.mean_plddt <= 100


def test_boltz_autodetect_and_parse(boltz_dir):
    parser = detect_parser(boltz_dir)
    assert parser is not None and parser.name == "boltz"

    preds = parse_folder(boltz_dir)
    assert len(preds) == 2
    top = min(preds, key=lambda p: p.rank)
    assert top.metrics.n_residues == 56
    assert top.metrics.ptm == 0.83
    assert top.metrics.iptm == 0.76
    assert top.pae.shape == (56, 56)
    # Boltz reports 0-1 pLDDT; the parser rescales to 0-100.
    assert 50 <= top.metrics.mean_plddt <= 100
    assert len(top.plddt) == 56
    assert max(top.plddt) > 1.5


def test_openfold3_autodetect_and_parse(openfold3_dir):
    parser = detect_parser(openfold3_dir)
    assert parser is not None and parser.name == "openfold3"

    preds = parse_folder(openfold3_dir)
    assert len(preds) == 2
    top = min(preds, key=lambda p: p.rank)
    assert top.source_tool == "openfold3"
    assert top.metrics.n_residues == 56
    assert top.metrics.n_chains == 2
    assert top.metrics.ptm == 0.86
    assert top.metrics.iptm == 0.79
    assert top.metrics.ranking_score == 0.83
    assert top.pae.shape == (56, 56)
    # pLDDT is aggregated from per-atom values to one value per residue.
    assert len(top.plddt) == 56
    assert 0 <= top.metrics.mean_plddt <= 100
    # The summary's mean pLDDT is recovered from the per-atom aggregation.
    assert abs(top.metrics.mean_plddt - float(np.mean(top.plddt))) < 1e-6


def test_all_tools_share_the_representation(colabfold_dir, af3_dir, boltz_dir, openfold3_dir):
    """The same downstream code consumes every tool without knowing the format."""
    pooled = []
    for d in (colabfold_dir, af3_dir, boltz_dir, openfold3_dir):
        pooled.extend(parse_folder(d))
    assert len({p.source_tool for p in pooled}) == 4
    for p in pooled:
        assert isinstance(p.plddt, list)
        assert p.pae is None or isinstance(p.pae, np.ndarray)
        assert p.metrics.n_residues == 56
