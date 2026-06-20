"""Phase 1 acceptance: the ColabFold parser returns correct Predictions."""

from __future__ import annotations

import numpy as np

from foldreport.parsers import detect_parser, parse_folder
from foldreport.parsers.colabfold import ColabFoldParser


def test_autodetect_picks_colabfold(colabfold_dir):
    parser = detect_parser(colabfold_dir)
    assert parser is not None
    assert parser.name == "colabfold"


def test_can_handle_rejects_unrelated(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert ColabFoldParser().can_handle(tmp_path) is False


def test_parse_returns_predictions(colabfold_dir):
    preds = parse_folder(colabfold_dir)
    assert len(preds) == 2
    for p in preds:
        assert p.source_tool == "colabfold"
        assert p.structure_path.exists()


def test_plddt_pae_and_metrics(colabfold_dir):
    preds = {p.rank: p for p in parse_folder(colabfold_dir)}
    top = preds[1]

    # 56 residues total (chain A: 32, chain B: 24).
    assert len(top.plddt) == 56
    assert top.metrics.n_residues == 56
    assert top.metrics.n_chains == 2

    # PAE is a square matrix matching residue count.
    assert isinstance(top.pae, np.ndarray)
    assert top.pae.shape == (56, 56)

    # Scalar metrics parsed from the scores JSON.
    assert top.metrics.ptm == 0.82
    assert top.metrics.iptm == 0.74
    assert top.metrics.mpdockq is None  # ColabFold provides none -> stays None.

    # Mean pLDDT matches the per-residue array.
    assert top.metrics.mean_plddt == np.mean(top.plddt)


def test_rank_extracted_from_filename(colabfold_dir):
    ranks = sorted(p.rank for p in parse_folder(colabfold_dir))
    assert ranks == [1, 2]
