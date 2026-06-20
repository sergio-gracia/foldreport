"""Provenance / reproducibility: each parser captures what its tool records on disk.

The contract mirrors metrics: fields are filled only from files actually present and
stay ``None`` (rendered "N/A") otherwise, never invented.
"""

from __future__ import annotations

from foldreport.parsers import parse_folder
from foldreport.report import build_report


def test_colabfold_provenance(colabfold_dir):
    top = {p.rank: p for p in parse_folder(colabfold_dir)}[1]
    prov = top.provenance
    assert prov.model_name == "alphafold2_multimer_v3"
    assert prov.tool_version == "1.5.5"
    assert prov.seeds == [0]
    assert prov.msa_mode == "mmseqs2_uniref_env"
    assert prov.num_recycles == 3
    assert prov.use_templates is False
    assert prov.msa_depth == 3  # three sequences in the fixture a3m
    assert prov.extra.get("model_number") == "3"


def test_af3_provenance(af3_dir):
    top = min(parse_folder(af3_dir), key=lambda p: p.rank)
    prov = top.provenance
    assert prov.seeds == [42]
    assert prov.extra.get("request_dialect") == "alphafoldserver v1"
    # AF3 Server does not stamp a model version, so it must stay None (not invented).
    assert prov.model_name is None


def test_openfold3_provenance(openfold3_dir):
    preds = parse_folder(openfold3_dir)
    prov = preds[0].provenance
    assert prov.seeds == [1]
    assert prov.model_name == "openfold3-1.0.0"
    assert prov.extra.get("weights") == "openfold3_initial.pt"
    assert prov.extra.get("sample") in {"0", "1"}


def test_alphafold_db_provenance(alphafold_db_dir):
    preds = parse_folder(alphafold_db_dir)
    prov = preds[0].provenance
    assert prov.model_name == "v6"
    assert prov.created_date == "2024-06-01"
    assert prov.database_snapshot == "2021-09-29"
    assert prov.extra.get("organism") == "Homo sapiens"


def test_boltz_provenance_is_empty_not_invented(boltz_dir):
    """Boltz ships no provenance config; the parser must not fabricate one."""
    preds = parse_folder(boltz_dir)
    assert preds[0].provenance.is_empty()


def test_report_embeds_provenance(tmp_path, colabfold_dir):
    preds = parse_folder(colabfold_dir)
    out = tmp_path / "report.html"
    build_report(preds, out)
    text = out.read_text(encoding="utf-8")

    # Human-readable panel.
    assert "Provenance &amp; reproducibility" in text
    assert "alphafold2_multimer_v3" in text
    assert "mmseqs2_uniref_env" in text
    # Machine-readable block for downstream extraction.
    assert "FOLDREPORT_PROVENANCE" in text
    assert '"msa_mode": "mmseqs2_uniref_env"' in text
    # Still a single self-contained file.
    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]
