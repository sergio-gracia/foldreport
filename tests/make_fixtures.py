"""Generate synthetic but format-faithful fixtures for each supported tool.

These stand in for real tool outputs in the test suite. They mirror the exact file
names, JSON keys and array shapes each tool emits, so the parsers exercise the same
code paths they would on real data. Replace with captured real outputs when available.

The complex is intentionally non-trivial (a 56-residue two-chain heterodimer) and each
tool folder ships the auxiliary files the real tools scatter alongside their results
(logs, configs, timings) so the parsers are tested against the noise they must ignore.

Run:  python tests/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"

# A 56-residue two-chain complex: chain A (32 residues) + chain B (24 residues).
_SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"
_SEQ_B = "GVALDEFTPAELRKLLDTAYKQGY"
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# Default two-chain layout reused across tools.
_CHAINS = (("A", _SEQ_A), ("B", _SEQ_B))


def _build_structure(plddt_per_residue: list[float], chains=_CHAINS, bfactor_scale: float = 1.0):
    """Build a minimal CA-only gemmi structure with given per-residue B-factors."""
    import gemmi

    structure = gemmi.Structure()
    structure.name = "fixture"
    model = gemmi.Model("1")
    residue_index = 0
    x = 0.0
    for chain_name, seq in chains:
        chain = gemmi.Chain(chain_name)
        for i, letter in enumerate(seq, start=1):
            residue = gemmi.Residue()
            residue.name = _ONE_TO_THREE[letter]
            residue.seqid = gemmi.SeqId(str(i))
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(x, 2.0 * ((i % 2) - 0.5), 0.0)
            atom.b_iso = plddt_per_residue[residue_index] * bfactor_scale
            residue.add_atom(atom)
            chain.add_residue(residue)
            residue_index += 1
            x += 3.8
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def _plddt(n: int, rng: np.random.Generator) -> list[float]:
    return list(np.round(rng.uniform(60, 95, size=n), 2))


def _pae(n: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.uniform(0.5, 20.0, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return np.round(m, 2)


def _n_residues(chains=_CHAINS) -> int:
    return sum(len(seq) for _name, seq in chains)


# --- Per-tool fixtures ----------------------------------------------------------------


def make_colabfold() -> None:
    out = DATA / "colabfold"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = _n_residues()
    job = "complex"
    for rank, model_n, ptm, iptm in [(1, 3, 0.82, 0.74), (2, 1, 0.79, 0.68)]:
        plddt = _plddt(n, rng)
        pae = _pae(n, rng)
        tag = f"alphafold2_multimer_v3_model_{model_n}_seed_000"
        structure = _build_structure(plddt)
        structure.write_pdb(str(out / f"{job}_unrelaxed_rank_{rank:03d}_{tag}.pdb"))
        scores = {
            "plddt": plddt,
            "max_pae": float(pae.max()),
            "pae": pae.tolist(),
            "ptm": ptm,
            "iptm": iptm,
        }
        with open(out / f"{job}_scores_rank_{rank:03d}_{tag}.json", "w", encoding="utf-8") as fh:
            json.dump(scores, fh)
    # Auxiliary files ColabFold drops alongside results; the parser must ignore them.
    (out / "config.json").write_text(json.dumps({"num_models": 2}), encoding="utf-8")
    (out / "cite.bibtex").write_text("@article{colabfold}\n", encoding="utf-8")
    (out / "log.txt").write_text("Running ColabFold...\nDone.\n", encoding="utf-8")
    (out / f"{job}.a3m").write_text(">query\nMKTAYI\n", encoding="utf-8")


def make_af3_server() -> None:
    out = DATA / "af3_server"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = _n_residues()
    job = "fold_mycomplex"
    token_chain_ids = ["A"] * len(_SEQ_A) + ["B"] * len(_SEQ_B)
    token_res_ids = list(range(1, len(_SEQ_A) + 1)) + list(range(1, len(_SEQ_B) + 1))
    for idx, (ptm, iptm, rscore) in enumerate([(0.85, 0.78, 0.81), (0.80, 0.70, 0.75)]):
        plddt = _plddt(n, rng)
        pae = _pae(n, rng)
        structure = _build_structure(plddt)  # AF3 stores pLDDT in B-factors (0-100)
        structure.make_mmcif_document().write_file(str(out / f"{job}_model_{idx}.cif"))
        full_data = {
            "atom_plddts": plddt,  # one CA atom per residue in this fixture
            "pae": pae.tolist(),
            "token_chain_ids": token_chain_ids,
            "token_res_ids": token_res_ids,
        }
        with open(out / f"{job}_full_data_{idx}.json", "w", encoding="utf-8") as fh:
            json.dump(full_data, fh)
        summary = {
            "ptm": ptm,
            "iptm": iptm,
            "ranking_score": rscore,
            "fraction_disordered": 0.1,
            "has_clash": 0.0,
        }
        with open(out / f"{job}_summary_confidences_{idx}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
    (out / "terms_of_use.md").write_text("Terms of use.\n", encoding="utf-8")
    (out / f"{job}_job_request.json").write_text(
        json.dumps({"name": "mycomplex"}), encoding="utf-8"
    )


def make_boltz() -> None:
    name = "mycomplex"
    out = DATA / "boltz" / "predictions" / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    n = _n_residues()
    for idx, (ptm, iptm, cscore) in enumerate([(0.83, 0.76, 0.80), (0.78, 0.69, 0.73)]):
        plddt01 = np.round(rng.uniform(0.6, 0.95, size=n), 3)  # Boltz uses 0-1 scale
        pae = _pae(n, rng)
        structure = _build_structure(list(plddt01), bfactor_scale=1.0)
        structure.make_mmcif_document().write_file(str(out / f"{name}_model_{idx}.cif"))
        np.savez_compressed(out / f"plddt_{name}_model_{idx}.npz", plddt01)
        np.savez_compressed(out / f"pae_{name}_model_{idx}.npz", pae)
        conf = {
            "confidence_score": cscore,
            "ptm": ptm,
            "iptm": iptm,
            "complex_plddt": float(plddt01.mean()),
        }
        with open(out / f"confidence_{name}_model_{idx}.json", "w", encoding="utf-8") as fh:
            json.dump(conf, fh)
    # Boltz run-level auxiliary files outside the per-prediction directory.
    (DATA / "boltz" / "log.txt").write_text("boltz predict ...\n", encoding="utf-8")


def make_openfold3() -> None:
    """OpenFold3 layout: one seed-*_sample-* dir per draw, with per-atom pLDDT."""
    job = "mycomplex"
    root = DATA / "openfold3" / job
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    n = _n_residues()
    # Per-atom pLDDT arrays (more than one atom per residue) exercise the
    # atom -> residue aggregation that OpenFold3 uniquely requires.
    atoms_per_res = 3
    atom_chain_ids: list[str] = []
    atom_res_ids: list[int] = []
    for chain_name, seq in _CHAINS:
        for res_i in range(1, len(seq) + 1):
            atom_chain_ids.extend([chain_name] * atoms_per_res)
            atom_res_ids.extend([res_i] * atoms_per_res)

    specs = [
        ("seed-1_sample-0", 0.86, 0.79, 0.83),
        ("seed-1_sample-1", 0.81, 0.71, 0.76),
    ]
    for sample_name, ptm, iptm, rscore in specs:
        sample_dir = root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        residue_plddt = _plddt(n, rng)
        # Atom pLDDTs jitter around their residue's value; the mean recovers it.
        atom_plddts: list[float] = []
        for value in residue_plddt:
            atom_plddts.extend(
                float(np.round(value + j, 2)) for j in (-1.0, 0.0, 1.0)[:atoms_per_res]
            )
        pae = _pae(n, rng)
        structure = _build_structure(residue_plddt)
        structure.make_mmcif_document().write_file(str(sample_dir / f"{job}_model.cif"))
        confidences = {
            "atom_plddts": atom_plddts,
            "pae": pae.tolist(),
            "pde": pae.tolist(),
            "token_chain_ids": atom_chain_ids,
            "token_res_ids": atom_res_ids,
        }
        with open(sample_dir / f"{job}_confidences.json", "w", encoding="utf-8") as fh:
            json.dump(confidences, fh)
        summary = {
            "plddt": float(np.mean(residue_plddt)),
            "ptm": ptm,
            "iptm": iptm,
            "ranking_score": rscore,
            "has_clash": 0.0,
            "fraction_disordered": 0.05,
        }
        with open(
            sample_dir / f"{job}_summary_confidences.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh)
    (root / "experiment_config.json").write_text(
        json.dumps({"seeds": [1], "samples": 2}), encoding="utf-8"
    )


# --- Edge-case fixtures ---------------------------------------------------------------


def make_single_chain() -> None:
    """A single-chain ColabFold prediction (no interface, so no ipTM)."""
    out = DATA / "single_chain"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(10)
    chains = (("A", _SEQ_A),)
    n = _n_residues(chains)
    job = "monomer"
    rank, model_n, ptm = 1, 3, 0.88
    plddt = _plddt(n, rng)
    pae = _pae(n, rng)
    tag = f"alphafold2_ptm_model_{model_n}_seed_000"
    structure = _build_structure(plddt, chains=chains)
    structure.write_pdb(str(out / f"{job}_unrelaxed_rank_{rank:03d}_{tag}.pdb"))
    scores = {
        "plddt": plddt,
        "max_pae": float(pae.max()),
        "pae": pae.tolist(),
        "ptm": ptm,
        # Monomers carry no ipTM; omitting the key keeps it None downstream.
    }
    with open(out / f"{job}_scores_rank_{rank:03d}_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh)


def make_malformed() -> None:
    """A ColabFold-looking folder whose sidecar JSON is invalid/incomplete.

    The structure file is valid, but the scores JSON is corrupt, so the parser must
    fall back to B-factor pLDDT and leave the JSON-only metrics as None.
    """
    out = DATA / "malformed"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    chains = (("A", _SEQ_A),)
    n = _n_residues(chains)
    job = "broken"
    tag = "alphafold2_ptm_model_1_seed_000"
    plddt = _plddt(n, rng)
    structure = _build_structure(plddt, chains=chains)
    structure.write_pdb(str(out / f"{job}_unrelaxed_rank_001_{tag}.pdb"))
    # Truncated / invalid JSON.
    (out / f"{job}_scores_rank_001_{tag}.json").write_text(
        '{"plddt": [1, 2, 3', encoding="utf-8"
    )


def make_empty() -> None:
    """A directory with no recognizable prediction files."""
    out = DATA / "empty"
    out.mkdir(parents=True, exist_ok=True)
    (out / "readme.txt").write_text("nothing to see here\n", encoding="utf-8")


def make_alphafold_db() -> None:
    """Two single-chain entries mimicking AlphaFold DB downloads (offline fixture).

    Mirrors the real DB layout: ``AF-<ACC>-F1-model_v<ver>.cif`` with pLDDT in the
    B-factors, the DB-style PAE JSON, and the saved API ``metadata.json``. The fixture
    is synthetic; ``examples/fetch_afdb.py`` downloads the real equivalents.
    """
    out = DATA / "alphafold_db"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    ver = 6
    entries = [
        ("P00001", "CYC", "Cytochrome c", _SEQ_A),
        ("P00002", "LYS", "Lysozyme", _SEQ_B),
    ]
    for acc, gene, description, seq in entries:
        chains = (("A", seq),)
        n = len(seq)
        plddt = _plddt(n, rng)
        pae = _pae(n, rng)
        structure = _build_structure(plddt, chains=chains)  # pLDDT in B-factors (0-100)
        structure.make_mmcif_document().write_file(str(out / f"AF-{acc}-F1-model_v{ver}.cif"))
        pae_doc = [
            {
                "predicted_aligned_error": pae.tolist(),
                "max_predicted_aligned_error": float(pae.max()),
            }
        ]
        with open(
            out / f"AF-{acc}-F1-predicted_aligned_error_v{ver}.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(pae_doc, fh)
        metadata = {
            "uniprotAccession": acc,
            "uniprotId": f"{gene}_TEST",
            "gene": gene,
            "uniprotDescription": description,
            "organismScientificName": "Homo sapiens",
            "globalMetricValue": float(np.mean(plddt)),
            "latestVersion": ver,
        }
        with open(out / f"AF-{acc}-F1-metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh)


def main() -> None:
    make_colabfold()
    make_af3_server()
    make_boltz()
    make_openfold3()
    make_single_chain()
    make_malformed()
    make_empty()
    make_alphafold_db()
    print(f"Fixtures written under {DATA}")


if __name__ == "__main__":
    main()
